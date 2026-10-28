---
license: cc-by-sa-4.0
language:
- en
base_model: {USER}/whisper-small-edacc
pipeline_tag: automatic-speech-recognition
tags:
- whisper
- ctranslate2
- faster-whisper
---

# Whisper-small fine-tuned on accented conversational English: CTranslate2 version

The model [{USER}/whisper-small-edacc](https://huggingface.co/{USER}/whisper-small-edacc), converted to
CTranslate2 ({CT2_TYPE} weights) for fast CPU inference with [faster-whisper](https://github.com/SYSTRAN/faster-whisper).
Results, limitations, training data and license details are in that model card.

**Conversion check:** on {N} random validation clips, this version's WER was {CT2_WER} against {OFF_WER} for the
original model (float16 on a GPU), a difference of {CT2_DIFF} points ({SAME} of {N} transcripts identical).

```python
from faster_whisper import WhisperModel

model = WhisperModel("{USER}/whisper-small-edacc-ct2", device="cpu", compute_type="{CT2_TYPE}")
segments, _ = model.transcribe("clip.wav", language="en", beam_size=1, temperature=0.0,
                               without_timestamps=True, condition_on_previous_text=False)
print(" ".join(s.text.strip() for s in segments))
```

License: CC BY-SA 4.0 (EdAcc is CC BY-SA). Code: [github.com/mohammaddalwai/speech-ft](https://github.com/mohammaddalwai/speech-ft).
