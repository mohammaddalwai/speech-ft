"""Transcription service (Phase 4).

    uvicorn src.serve.app:app --port 8000
    curl --data-binary @clip.wav localhost:8000/transcribe

POST /transcribe  body = the audio file itself (any common format, at most 30 s and MAX_UPLOAD_MB).
                  The upload limit is enforced while the body streams in, and the audio stays in memory.
GET  /health      200 once the model is loaded and warmed up, with load and warm-up times (cold start).
GET  /            the demo page (src/serve/static/index.html): record or upload a clip, see the transcript.

One model per process, one transcription at a time (whisper-small is CPU-bound). A request waits at most
QUEUE_WAIT_S for its turn (else 503) and gets its answer within TIMEOUT_S (else 504). Logs hold timings only,
never transcripts or audio.
"""
import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager

import pathlib

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse

from src import asr
from src.serve import audio

PAGE = (pathlib.Path(__file__).parent / "static" / "index.html").read_text()

# Libraries log at WARNING only: faster-whisper would log "Processing audio with duration 00:30.000" for every
# request (always 30 s, because of the padding), huggingface_hub every download request. Our own lines: timings.
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("speech-ft")
log.setLevel(logging.INFO)
logging.getLogger("faster_whisper").setLevel(logging.WARNING)


def create_app(load_model=asr.load_model, model_id="mddalwai/whisper-small-edacc-ct2", model_label=None, compute_type="int8",
               cpu_threads=0, max_upload_mb=10.0, max_seconds=30.0, timeout_s=60.0, queue_wait_s=30.0):
    max_bytes = int(max_upload_mb * 1e6)
    label = model_label or model_id  # e.g. "repo@commit" when model_id is a local folder baked into the image
    settings = asr.decode_settings()

    @asynccontextmanager
    async def lifespan(app):
        app.state.started_at = round(time.time(), 1)  # tells a restarted container from the old one
        t0 = time.perf_counter()
        app.state.model = load_model(model_id, compute_type, cpu_threads)
        t1 = time.perf_counter()
        asr.transcribe(app.state.model, np.zeros(audio.SR, dtype=np.float32), settings)  # first call is slow
        app.state.load_s, app.state.warmup_s = round(t1 - t0, 2), round(time.perf_counter() - t1, 2)
        app.state.turn = asyncio.Semaphore(1)
        log.info("model %s (%s) loaded in %.2f s, warm-up %.2f s", label, compute_type,
                 app.state.load_s, app.state.warmup_s)
        yield

    app = FastAPI(title="speech-ft", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    def page():
        return PAGE

    @app.get("/health")
    def health(request: Request):
        s = request.app.state
        return {"status": "ok", "model": label, "compute_type": compute_type, "cpu_threads": cpu_threads,
                "cpus_visible": os.cpu_count(), "serving_version": asr.SERVING_VERSION,
                "load_seconds": s.load_s, "warmup_seconds": s.warmup_s, "started_at": s.started_at}

    async def read_body(request):
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > max_bytes:
            raise HTTPException(413, f"file is larger than {max_upload_mb:g} MB")
        body = bytearray()
        async for chunk in request.stream():
            body += chunk
            if len(body) > max_bytes:
                raise HTTPException(413, f"file is larger than {max_upload_mb:g} MB")
        return bytes(body)

    @app.post("/transcribe")
    async def transcribe(request: Request):
        data = await read_body(request)
        try:
            wav = audio.decode(data, max_seconds)
        except audio.TooLong as e:
            raise HTTPException(422, str(e))
        except audio.AudioError as e:
            raise HTTPException(415, str(e))

        s = request.app.state
        try:
            await asyncio.wait_for(s.turn.acquire(), queue_wait_s)
        except TimeoutError:
            raise HTTPException(503, "busy, try again shortly")
        t0 = time.perf_counter()
        work = asyncio.ensure_future(asyncio.to_thread(asr.transcribe, s.model, wav, settings))
        work.add_done_callback(lambda _: s.turn.release())  # free the model only when the work really ends
        try:
            text = await asyncio.wait_for(asyncio.shield(work), timeout_s)
        except TimeoutError:
            log.warning("timeout after %.0f s on %.1f s of audio", timeout_s, len(wav) / audio.SR)
            raise HTTPException(504, f"transcription took longer than {timeout_s:g} s")
        secs = time.perf_counter() - t0
        audio_s = len(wav) / audio.SR
        log.info("transcribed %.2f s of audio in %.2f s (RTF %.3f)", audio_s, secs, secs / audio_s)
        return {"text": text, "audio_seconds": round(audio_s, 2), "processing_seconds": round(secs, 3),
                "real_time_factor": round(secs / audio_s, 3), "model": label, "compute_type": compute_type,
                "serving_version": asr.SERVING_VERSION}

    return app


def env_app():
    e = os.environ.get
    return create_app(model_id=e("MODEL_ID", "mddalwai/whisper-small-edacc-ct2"), model_label=e("MODEL_LABEL"),
                      compute_type=e("COMPUTE_TYPE", "int8"), cpu_threads=int(e("CPU_THREADS", "0")),
                      max_upload_mb=float(e("MAX_UPLOAD_MB", "10")), timeout_s=float(e("TIMEOUT_S", "60")),
                      queue_wait_s=float(e("QUEUE_WAIT_S", "30")))


app = env_app()
