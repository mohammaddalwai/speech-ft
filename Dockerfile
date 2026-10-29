# Serving image for the demo (Phase 4): CPU only, about 30 s of audio per request, one request at a time.
#   docker build -t speech-ft .
#   docker run --rm -p 7860:7860 speech-ft        ->  http://localhost:7860
# The released int8 model is baked in at build time, pinned to the Hub commit that passed the conversion
# checks (docs/decisions.md), so the container never downloads anything when it starts.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# Hugging Face Spaces run containers as user 1000; the same user is used everywhere.
RUN useradd --create-home --uid 1000 app
WORKDIR /app

COPY requirements-serve.lock .
RUN pip install -r requirements-serve.lock

ARG MODEL_REPO=mddalwai/whisper-small-edacc-ct2
ARG MODEL_REVISION=fc9badc32977c997c2468386ec7a3390c60422c1
RUN python -c "from huggingface_hub import snapshot_download; \
snapshot_download('${MODEL_REPO}', revision='${MODEL_REVISION}', local_dir='/app/model')" \
 && rm -rf /app/model/.cache

COPY configs/decode.yaml configs/
COPY src/__init__.py src/asr.py src/
COPY src/serve/ src/serve/

ENV MODEL_ID=/app/model MODEL_LABEL=${MODEL_REPO}@${MODEL_REVISION} HF_HUB_OFFLINE=1
USER app
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/health', timeout=4)"
CMD ["uvicorn", "src.serve.app:app", "--host", "0.0.0.0", "--port", "7860"]
