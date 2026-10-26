"""Text handling.

normalize():      applied to references AND hypotheses before every WER number. Whisper's English normalizer:
                  lowercases, removes punctuation and <TAGS>, drops fillers (uh, um, hmm, mm), expands
                  contractions, unifies spellings and numbers.
training_text():  R5, the target text the model is fine-tuned on (lowercase, tags removed, nothing else).
"""
import functools
import re

TAG_RE = re.compile(r"<[^>]*>")


@functools.lru_cache(maxsize=1)
def _whisper_normalizer():
    from transformers import WhisperTokenizer

    return WhisperTokenizer.from_pretrained("openai/whisper-small").normalize


def normalize(text: str) -> str:
    return _whisper_normalizer()(text)


def training_text(text: str) -> str:
    return " ".join(TAG_RE.sub(" ", text).lower().split())
