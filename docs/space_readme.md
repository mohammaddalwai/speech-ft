---
title: Accented English ASR
emoji: 🎙️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
license: cc-by-sa-4.0
models:
  - {USER}/whisper-small-edacc-ct2
short_description: Whisper-small fine-tuned on accented conversational English
---

Demo of [{USER}/whisper-small-edacc](https://huggingface.co/{USER}/whisper-small-edacc), served as the int8
CTranslate2 version on a free 2-vCPU CPU. Record or upload up to 30 s of English speech; the model writes it word
for word in lowercase, fillers included. Results and limitations are in the model card.

The page uses a small HTTP API, which you can call directly:

```
curl --data-binary @clip.wav https://{HOST}/transcribe
curl https://{HOST}/health
```

Audio is decoded in memory and never stored or logged. Built from
[github.com/mohammaddalwai/speech-ft](https://github.com/mohammaddalwai/speech-ft) at commit `{COMMIT}`.
