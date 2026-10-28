---
license: cc-by-sa-4.0
language:
- en
base_model: openai/whisper-small
pipeline_tag: automatic-speech-recognition
tags:
- whisper
- accented-english
- conversational-speech
datasets:
- edinburghcstr/edacc
- openslr/librispeech_asr
metrics:
- wer
---

# Whisper-small fine-tuned on accented conversational English (EdAcc)

`openai/whisper-small` (244M parameters), fully fine-tuned on 15.9 hours of conversational English from speakers
of many accents ([EdAcc](https://huggingface.co/datasets/edinburghcstr/edacc)), with 15% of training clips taken
from LibriSpeech read speech to limit forgetting. All code, data splits, decisions and per-utterance results:
[github.com/mohammaddalwai/speech-ft](https://github.com/mohammaddalwai/speech-ft).

A CTranslate2 version for fast CPU inference with faster-whisper is at
[{USER}/whisper-small-edacc-ct2](https://huggingface.co/{USER}/whisper-small-edacc-ct2).

## Results

EdAcc test split: 28 speakers and 14 conversations that never appear in training or validation (3,351 clips of at
most 30 s, 3,341 with words to score). WER is pooled over words after Whisper's English text normalizer plus a few
extra filler and spelling rules (see the repo). Greedy decoding for every model.

| Model | Params | EdAcc test WER | LibriSpeech test-clean / test-other |
|---|---|---|---|
| Whisper-small, zero-shot | 244M | 22.5% | 3.3% / 7.5% |
| Whisper-medium, zero-shot | 769M | 19.8% | - |
| Whisper-large-v3-turbo, zero-shot | 809M | 18.2% | - |
| **This model** | 244M | **16.3%** (3 training seeds: mean 16.8%, std 0.4) | 3.9% / 8.6% |

- vs zero-shot Whisper-small: -6.2 points, 95% interval [-8.7, -4.2] (paired bootstrap over speakers).
- vs zero-shot large-v3-turbo: -1.9 points, 95% interval [-3.2, -0.7]. On the validation split the two were tied
  (23.9% vs 23.8%), so the advantage depends on which speakers are tested: read it as "on par with or better than".
- Zero-shot Whisper-small looped on 6 short test clips (under 8 s), repeating a word or phrase until the output
  limit ("no no no ..."), which adds about 1.8 points to its WER. Without the 14 clips where any model looped:
  zero-shot small 20.7%, turbo 17.9%, this model 16.2%
  (a check done after seeing the results, reported next to, not instead of, the official numbers).

## Where the improvement comes from

Word-by-word comparison with zero-shot Whisper-small on the test split: +5.7 points come from words the base model
skipped (mostly short common words such as "i", "like", "it", "yeah"; 22% are repeated words), +2.4 from words it got
wrong and this model gets right, and -2.6 from words it got right and this model gets wrong. **Most of the gain is
learning EdAcc's word-for-word transcription style and removing looping outputs, not better recognition of accented
words.** A hand review of the 50 worst test clips found the main remaining causes to be everyday words misheard in
fast or accented speech (60% of their errors) and names, titles and rare words (21%).

## Limitations

- Output is lowercase without punctuation, and fillers and repeats ("uh", "i i") are written out, because the
  training transcripts are written that way.
- Short-form only: evaluated on clips of at most 30 s. Long recordings were not evaluated.
- Worse than the base model on read speech: LibriSpeech WER is about 16% higher (relative) than zero-shot
  Whisper-small. Without replay it was about 40% higher.
- Accent groups have only 2-6 test speakers each, so per-accent numbers are anecdotal.
- Not directly comparable with WERs in the EdAcc paper: segments over 30 s were dropped and text normalization
  differs.
- Trained and tested on casual video-call conversations in English. Other domains, noise conditions and
  languages were not tested.
- Rare looping outputs still happen (1 of 3,351 test clips).
- Not for high-stakes use (medical, legal, safety, or identifying people) and not as unreviewed captions.

## Training

Full fine-tune, learning rate 1e-5 (linear, 200 warmup steps), effective batch 32, 2,000 steps, bf16, best of
8 checkpoints by validation WER (step 1750). Training data: the EdAcc train split (speaker- and
conversation-disjoint from validation and test; segments over 30 s, untranscribed segments and foreign-language
segments removed) plus 2,077 LibriSpeech train-clean-100 clips (7.4 h). Text targets are lowercased, with tags
such as `<laugh>` removed.

## Usage

```python
from transformers import pipeline

asr = pipeline("automatic-speech-recognition", model="{USER}/whisper-small-edacc")
print(asr("clip.wav", generate_kwargs={"language": "en", "task": "transcribe"})["text"])
```

## License and attribution

This model is released under **CC BY-SA 4.0**, because EdAcc is licensed CC BY-SA.
- EdAcc: Sanabria et al., "The Edinburgh International Accents of English Corpus: Towards the Democratization of
  English ASR", ICASSP 2023. CC BY-SA.
- LibriSpeech: Panayotov et al., ICASSP 2015. CC BY 4.0.
- Base model: OpenAI Whisper (`openai/whisper-small`), Apache 2.0.

No audio from either dataset is redistributed here.
