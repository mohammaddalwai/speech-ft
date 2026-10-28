"""Serving and release guards: decoding settings match the evaluation, model cards fill, unchecked models never ship."""
import importlib.util
import json
import pathlib
import sys

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import asr  # noqa: E402

spec = importlib.util.spec_from_file_location("publish_model", ROOT / "scripts" / "publish_model.py")
publish = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publish)

CHECK = {"ct2": "ct2/E3-int8", "wer_ct2": 0.2401, "wer_official_same_clips": 0.2390, "difference_points": 0.11,
         "identical_outputs": 431, "clips": 500,
         "within_rule": True}


def test_serving_decode_matches_evaluation():
    cfg = yaml.safe_load(open(ROOT / "configs" / "decode.yaml"))
    s = asr.decode_settings()
    assert (s["language"], s["task"], s["beam_size"], s["max_new_tokens"]) == (
        cfg["language"], cfg["task"], cfg["num_beams"], cfg["max_new_tokens"])
    # greedy with no temperature fallback, no timestamps, no VAD, no previous-text prompt
    assert s["temperature"] == 0.0 and s["best_of"] == 1 and s["compression_ratio_threshold"] is None
    assert s["log_prob_threshold"] is None and s["no_speech_threshold"] is None
    assert s["without_timestamps"] and not s["vad_filter"] and not s["condition_on_previous_text"]


def test_model_cards_fill_completely():
    values = publish.card_values("someone", "int8", CHECK)
    assert values["CT2_WER"] == "24.01%" and values["CT2_DIFF"] == "+0.11"
    for name in ("model_card.md", "model_card_ct2.md"):
        text = publish.fill((ROOT / "docs" / name).read_text(), values)
        assert "{" + "USER}" not in text and "someone/whisper-small-edacc" in text
        assert text.startswith("---\nlicense: cc-by-sa-4.0")
    assert "431 of 500 transcripts identical" in publish.fill((ROOT / "docs" / "model_card_ct2.md").read_text(), values)


def test_fill_refuses_leftover_placeholder():
    with pytest.raises(ValueError):
        publish.fill("a {USER} b {NEW_THING}", {"USER": "x"})


def write_check(tmp_path, compute_type, **changes):
    d = tmp_path / f"ct2_check_{compute_type}"
    d.mkdir()
    (d / "summary.json").write_text(json.dumps({**CHECK, **changes}))


def test_load_check_accepts_checked_passing_folder(tmp_path):
    write_check(tmp_path, "int8")
    assert publish.load_check("ct2/E3-int8", "int8", runs_dir=tmp_path)["identical_outputs"] == 431


@pytest.mark.parametrize("changes, ct2", [({"within_rule": False}, "ct2/E3-int8"),  # failed the rule
                                          ({}, "ct2/E3-float32"),                   # a different folder
                                          (None, "ct2/E3-int8")])                   # never checked
def test_load_check_refuses(tmp_path, changes, ct2):
    if changes is not None:
        write_check(tmp_path, "int8", **changes)
    with pytest.raises(SystemExit):
        publish.load_check(ct2, "int8", runs_dir=tmp_path)
