"""Serving-side transcription with faster-whisper (CTranslate2).

Decoding matches configs/decode.yaml, i.e. exactly what every reported WER used: English, transcribe, greedy
(1 beam, temperature 0, no fallback), no timestamps, no voice-activity filter, no conditioning on previous text,
at most 224 new tokens. faster-whisper's own defaults differ (beam 5, temperature fallback, timestamps), so they
are all set explicitly here.
"""
import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]


def decode_settings(path=ROOT / "configs" / "decode.yaml"):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return {
        "language": cfg["language"], "task": cfg["task"], "beam_size": cfg["num_beams"], "best_of": 1,
        "temperature": 0.0, "compression_ratio_threshold": None, "log_prob_threshold": None,
        "no_speech_threshold": None, "condition_on_previous_text": False, "without_timestamps": True,
        "vad_filter": False, "max_new_tokens": cfg["max_new_tokens"],
    }


def load_model(path_or_repo, compute_type="int8", cpu_threads=0):
    """path_or_repo: a local CTranslate2 folder or a Hugging Face repo id. cpu_threads=0 lets CTranslate2 decide."""
    from faster_whisper import WhisperModel

    return WhisperModel(path_or_repo, device="cpu", compute_type=compute_type, cpu_threads=cpu_threads)


def transcribe(model, audio, settings=None):
    """audio: float32 numpy array, 16 kHz mono, at most 30 s. Returns the text as one string."""
    segments, _ = model.transcribe(audio, **(settings or decode_settings()))
    return " ".join(s.text.strip() for s in segments).strip()
