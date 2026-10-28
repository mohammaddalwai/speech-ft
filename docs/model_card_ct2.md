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
A float32 conversion (not published) scored {F32_WER} on the same clips ({F32_DIFF} points, {F32_SAME} of {N}
identical), so the serving code reproduces the original model and the remaining {CT2_TYPE} difference is rounding.

```python
import numpy as np
from faster_whisper import WhisperModel, decode_audio

model = WhisperModel("{USER}/whisper-small-edacc-ct2", device="cpu", compute_type="{CT2_TYPE}")
audio = decode_audio("clip.wav")[:480000]           # 16 kHz mono, at most 30 s
audio = np.pad(audio, (0, 480000 - len(audio)))     # pad the AUDIO to 30 s, as in training (see below)
segments, _ = model.transcribe(audio, language="en", beam_size=1, temperature=0.0, without_timestamps=True,
                               condition_on_previous_text=False)
print(" ".join(s.text.strip() for s in segments))
```

Why the padding line: faster-whisper pads its log-mel features with zeros, while the model was trained on features
of audio padded with silence. Without the line, clips shorter than 30 s get input the model never saw in training.

License: CC BY-SA 4.0 (EdAcc is CC BY-SA). Code: [github.com/mohammaddalwai/speech-ft](https://github.com/mohammaddalwai/speech-ft).
