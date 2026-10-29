"""The HTTP service with a stand-in model: limits, error codes, and audio decoding identical to the evaluation's."""
import io
import pathlib
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
pytest.importorskip("fastapi")
pytest.importorskip("av")
from fastapi.testclient import TestClient  # noqa: E402

from src import asr, data  # noqa: E402
from src.serve import audio  # noqa: E402
from src.serve.app import create_app  # noqa: E402


class FakeModel:
    """Records what it is given; optionally slow."""

    def __init__(self, delay=0.0):
        self.delay, self.seen = delay, []

    def transcribe(self, wav, **settings):
        self.seen.append((wav, settings))
        time.sleep(self.delay)
        return [SimpleNamespace(text=" hello there")], None


def wav_bytes(seconds, sr=48000, channels=2, subtype="PCM_16"):
    t = np.arange(int(seconds * sr)) / sr
    x = 0.3 * np.sin(2 * np.pi * 220 * t)
    x = np.stack([x, 0.5 * x], axis=1) if channels == 2 else x
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="WAV", subtype=subtype)
    return buf.getvalue()


def client(model, **kw):
    return TestClient(create_app(load_model=lambda *a: model, **kw))


def test_health_reports_model_and_cold_start():
    with client(FakeModel()) as c:
        r = c.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["serving_version"] == asr.SERVING_VERSION
    assert body["load_seconds"] >= 0 and body["warmup_seconds"] >= 0
    assert body["cpu_threads"] == 0 and body["cpus_visible"] >= 1
    assert abs(body["started_at"] - time.time()) < 60


def test_transcribe_returns_text_and_timings():
    m = FakeModel()
    with client(m) as c:
        r = c.post("/transcribe", content=wav_bytes(2.0))
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["text"] == "hello there" and body["audio_seconds"] == 2.0
    wav, settings = m.seen[-1]  # [0] is the warm-up call
    assert wav.dtype == np.float32 and len(wav) == asr.N_SAMPLES  # padded to 30 s (serving v2)
    assert settings == asr.decode_settings()


@pytest.mark.parametrize("sr, channels, subtype", [(48000, 2, "PCM_16"), (16000, 1, "PCM_16"), (32000, 1, "FLOAT")])
def test_serving_decode_equals_evaluation_decode(sr, channels, subtype):
    raw = wav_bytes(1.5, sr, channels, subtype)
    got = audio.decode(raw, max_seconds=30)
    want = data.decode_audio({"bytes": raw})  # the decoder behind every reported WER
    assert got.shape == want.shape
    assert np.abs(got - want).max() < 1e-6


def test_rejects_large_upload():
    with client(FakeModel(), max_upload_mb=0.05) as c:
        r = c.post("/transcribe", content=wav_bytes(2.0))
    assert r.status_code == 413


def test_rejects_audio_over_30_s():
    with client(FakeModel()) as c:
        r = c.post("/transcribe", content=wav_bytes(31.0, sr=8000, channels=1))
    assert r.status_code == 422 and "30" in r.json()["detail"]


@pytest.mark.parametrize("payload", [b"", b"this is not audio at all" * 100])
def test_rejects_non_audio(payload):
    with client(FakeModel()) as c:
        r = c.post("/transcribe", content=payload)
    assert r.status_code == 415


def test_timeout():
    m = FakeModel()
    with client(m, timeout_s=0.2) as c:
        m.delay = 1.0  # after the warm-up
        r = c.post("/transcribe", content=wav_bytes(1.0))
    assert r.status_code == 504


def test_page_is_served():
    with client(FakeModel()) as c:
        r = c.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert 'fetch("transcribe"' in r.text and "never stored or logged" in r.text


def test_serving_code_only_uses_files_the_image_copies():
    """The Dockerfile copies src/__init__.py, src/asr.py and src/serve/ only (see .dockerignore). Importing
    anything else from the project (e.g. src/data.py, which needs the training libraries) would break the image."""
    import subprocess

    root = pathlib.Path(__file__).resolve().parents[1]
    code = "import sys, src.serve.app; print(sorted(m for m in sys.modules if m == 'src' or m.startswith('src.')))"
    out = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True, check=True).stdout
    assert eval(out) == ["src", "src.asr", "src.serve", "src.serve.app", "src.serve.audio"]
    allowed = (root / ".dockerignore").read_text()
    for path in ("!src/__init__.py", "!src/asr.py", "!src/serve/", "!configs/decode.yaml", "!requirements-serve.lock"):
        assert path in allowed


def test_space_gets_exactly_the_image_files():
    import importlib.util

    root = pathlib.Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("publish_space", root / "scripts" / "publish_space.py")
    ps = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ps)
    serve_files = {str(p.relative_to(root)) for p in (root / "src" / "serve").rglob("*")
                   if p.is_file() and "__pycache__" not in p.parts and p.name != ".gitkeep"}
    assert serve_files <= set(ps.FILES)  # a new serving file must also go to the Space
    allowed = (root / ".dockerignore").read_text()
    for f in ps.FILES[1:]:  # everything but the Dockerfile itself is inside the image whitelist
        assert f"!{f}" in allowed or f"!{f.split('/')[0]}/{f.split('/')[1]}/" in allowed, f
    assert ps.space_host("mddalwai", "whisper-small-edacc-demo") == "mddalwai-whisper-small-edacc-demo.hf.space"
