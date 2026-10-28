"""Serving-side transcription with faster-whisper (CTranslate2).

Decoding matches configs/decode.yaml, i.e. exactly what every reported WER used: English, transcribe, greedy
(1 beam, temperature 0, no fallback), no timestamps, no voice-activity filter, no conditioning on previous text,
at most 224 new tokens. faster-whisper's own defaults differ (beam 5, temperature fallback, timestamps), so they
are all set explicitly here.

Input features match too (serving v2). faster-whisper 1.2.1 pads the log-mel FEATURES with 0.0 up to 30 s, while
transformers (training and every reported WER) pads the AUDIO with silence, so its padded frames hold the log-mel
of silence (a negative value that depends on the clip's loudness). Padding the audio to 30 s here, before
faster-whisper sees it, gives the encoder exactly the features the model was trained and evaluated on
(tests/test_release.py checks this).
"""
import pathlib

import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
SERVING_VERSION = "v2"  # v1: no audio padding (2026-10-27/28 checks); v2: audio padded to 30 s
SR = 16000
N_SAMPLES = 30 * SR  # Whisper's input window


def decode_settings(path=ROOT / "configs" / "decode.yaml"):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return {
        "language": cfg["language"], "task": cfg["task"], "beam_size": cfg["num_beams"], "best_of": 1,
        "temperature": 0.0, "compression_ratio_threshold": None, "log_prob_threshold": None,
        "no_speech_threshold": None, "condition_on_previous_text": False, "without_timestamps": True,
        "vad_filter": False, "max_new_tokens": cfg["max_new_tokens"],
    }


def pad_to_window(audio):
    """16 kHz mono audio -> exactly 30 s of float32: cut after 30 s and padded with silence, as transformers'
    WhisperFeatureExtractor does. Callers that must not cut long audio check the length first."""
    audio = np.asarray(audio, dtype=np.float32)[:N_SAMPLES]
    return np.pad(audio, (0, N_SAMPLES - len(audio)))


def load_model(path_or_repo, compute_type="int8", cpu_threads=0):
    """path_or_repo: a local CTranslate2 folder or a Hugging Face repo id. cpu_threads=0 lets CTranslate2 decide."""
    from faster_whisper import WhisperModel

    return WhisperModel(path_or_repo, device="cpu", compute_type=compute_type, cpu_threads=cpu_threads)


def transcribe(model, audio, settings=None):
    """audio: float32 numpy array, 16 kHz mono, at most 30 s. Returns the text as one string."""
    segments, _ = model.transcribe(pad_to_window(audio), **(settings or decode_settings()))
    return " ".join(s.text.strip() for s in segments).strip()
