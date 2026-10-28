"""Serving and release guards: decoding settings and encoder input match the evaluation, model cards fill,
unchecked models never ship."""
import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import asr  # noqa: E402

spec = importlib.util.spec_from_file_location("publish_model", ROOT / "scripts" / "publish_model.py")
publish = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publish)

CHECK = {"ct2": "ct2/E3-int8", "serving_version": asr.SERVING_VERSION, "wer_ct2": 0.2401,
         "wer_official_same_clips": 0.2390, "difference_points": 0.11, "identical_outputs": 431, "clips": 500,
         "within_rule": True, "cpu_model_this_machine": "AMD EPYC 7B12"}
CONTROL = {**CHECK, "ct2": "ct2/E3-float32", "wer_ct2": 0.2235, "difference_points": -0.05, "identical_outputs": 490}


def test_serving_decode_matches_evaluation():
    cfg = yaml.safe_load(open(ROOT / "configs" / "decode.yaml"))
    s = asr.decode_settings()
    assert (s["language"], s["task"], s["beam_size"], s["max_new_tokens"]) == (
        cfg["language"], cfg["task"], cfg["num_beams"], cfg["max_new_tokens"])
    # greedy with no temperature fallback, no timestamps, no VAD, no previous-text prompt
    assert s["temperature"] == 0.0 and s["best_of"] == 1 and s["compression_ratio_threshold"] is None
    assert s["log_prob_threshold"] is None and s["no_speech_threshold"] is None
    assert s["without_timestamps"] and not s["vad_filter"] and not s["condition_on_previous_text"]


def encoder_input_faster_whisper(audio):
    """What faster-whisper 1.2.1 feeds the encoder for a clip of at most 30 s (feature_extractor.py, transcribe.py:
    features of the audio + 160 samples, the last frame dropped, then zero-padded or cut to 3000 frames)."""
    from faster_whisper.audio import pad_or_trim
    from faster_whisper.feature_extractor import FeatureExtractor

    f = FeatureExtractor()(audio)
    return pad_or_trim(f[:, : f.shape[-1] - 1])


# Clips that end within 12.5 ms of 30 s differ in the last of 3000 frames only (edge handling of the final
# window); clips over 30 s are rejected by the service and were dropped from every evaluation (rule R2).
@pytest.mark.parametrize("seconds", [0.4, 4.0, 15.0, 29.5])
def test_serving_encoder_input_matches_transformers(seconds):
    pytest.importorskip("faster_whisper")
    from transformers import WhisperFeatureExtractor

    rng = np.random.default_rng(0)
    t = np.arange(int(seconds * asr.SR)) / asr.SR
    audio = (0.3 * np.sin(2 * np.pi * 220 * t) + 0.05 * rng.standard_normal(len(t))).astype(np.float32)
    want = WhisperFeatureExtractor()(audio, sampling_rate=asr.SR, return_tensors="np").input_features[0]
    got = encoder_input_faster_whisper(asr.pad_to_window(audio))
    assert got.shape == want.shape == (80, 3000)
    assert np.abs(got - want).max() < 1e-3
    if seconds < 29:  # without the fix, the padded frames differ (0.0 instead of the log-mel of silence)
        assert np.abs(encoder_input_faster_whisper(audio) - want).max() > 0.1


def test_model_cards_fill_completely():
    values = publish.card_values("someone", "int8", CHECK, CONTROL)
    assert values["CT2_WER"] == "24.01%" and values["CT2_DIFF"] == "+0.11"
    for name in ("model_card.md", "model_card_ct2.md"):
        text = publish.fill((ROOT / "docs" / name).read_text(), values)
        assert "{" + "USER}" not in text and "someone/whisper-small-edacc" in text
        assert text.startswith("---\nlicense: cc-by-sa-4.0")
    ct2_card = publish.fill((ROOT / "docs" / "model_card_ct2.md").read_text(), values)
    assert "this check ran on an AMD EPYC 7B12 CPU" in ct2_card
    assert "431 of 500 transcripts identical" in ct2_card and "22.35% on the same clips (-0.05 points, 490 of 500" in ct2_card


def test_fill_refuses_leftover_placeholder():
    with pytest.raises(ValueError):
        publish.fill("a {USER} b {NEW_THING}", {"USER": "x"})


def write_check(tmp_path, compute_type, **changes):
    d = tmp_path / f"ct2_check_{compute_type}_{asr.SERVING_VERSION}"
    d.mkdir()
    (d / "summary.json").write_text(json.dumps({**CHECK, **changes}))


def test_load_check_accepts_checked_passing_folder(tmp_path):
    write_check(tmp_path, "int8")
    assert publish.load_check("ct2/E3-int8", "int8", runs_dir=tmp_path)["identical_outputs"] == 431


@pytest.mark.parametrize("changes, ct2", [({"within_rule": False}, "ct2/E3-int8"),  # failed the rule
                                          ({}, "ct2/E3-float32"),                   # a different folder
                                          ({"serving_version": "v1"}, "ct2/E3-int8"),  # old serving code
                                          (None, "ct2/E3-int8")])                   # never checked
def test_load_check_refuses(tmp_path, changes, ct2):
    if changes is not None:
        write_check(tmp_path, "int8", **changes)
    with pytest.raises(SystemExit):
        publish.load_check(ct2, "int8", runs_dir=tmp_path)


def test_control_gate(tmp_path):
    write_check(tmp_path, "float32", **CONTROL)
    assert publish.load_control(runs_dir=tmp_path)["identical_outputs"] == 490
    for changes in ({"difference_points": 0.81}, {"serving_version": "v1"}):
        d = tmp_path / "bad"
        d.mkdir(exist_ok=True)
        write_check(d, "float32", **{**CONTROL, **changes})
        with pytest.raises(SystemExit):
            publish.load_control(runs_dir=d)
        import shutil
        shutil.rmtree(d)
    with pytest.raises(SystemExit):  # never run
        publish.load_control(runs_dir=tmp_path / "empty")
