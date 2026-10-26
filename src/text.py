"""Text handling.

normalize():      applied to references AND hypotheses before every WER number.
                  Step 1, Whisper's English normalizer: lowercases, removes punctuation and <TAGS>, drops the
                  fillers uh/um/hmm/mm/mhm/mmm, expands contractions, unifies spellings and numbers.
                  Step 2 (v2, our additions, found in the Gate 1 review): more filler spellings, and spelling
                  conventions that differ between EdAcc transcripts and Whisper output but mean the same words.
training_text():  R5, the target text the model is fine-tuned on (lowercase, tags removed, nothing else).
"""
import functools
import re

NORMALIZER_VERSION = "v2"
TAG_RE = re.compile(r"<[^>]*>")
# hm, hmm, hmmm, mm, mmmm..., mhm, uh, uhh, um, umm, uhm, er, erm, ah, ahh
FILLER_RE = re.compile(r"^(?:h+m+|mm+|m+h+m+|u+h+|u+m+|u+h+m+|e+r+m*|a+h+)$")
SPELLINGS = {
    "alright": "all right", "ok": "okay", "cannot": "can not",
    "kinda": "kind of", "sorta": "sort of", "dunno": "do not know", "lemme": "let me", "gimme": "give me",
    "cause": "because", "cuz": "because", "coz": "because", "cos": "because",
}


@functools.lru_cache(maxsize=1)
def _whisper_normalizer():
    from transformers import WhisperTokenizer

    return WhisperTokenizer.from_pretrained("openai/whisper-small").normalize


def normalize_whisper_only(text: str) -> str:
    return _whisper_normalizer()(text)


def normalize(text: str) -> str:
    words = []
    for w in normalize_whisper_only(text).split():
        if FILLER_RE.match(w):
            continue
        words.extend(SPELLINGS.get(w, w).split())
    return " ".join(words)


def normalize_basic(text: str) -> str:
    """Only lowercase, drop <tags> and punctuation. Used to show how much normalization changes WER."""
    return " ".join(re.sub(r"[^\w\s']", " ", TAG_RE.sub(" ", text).lower()).split())


def training_text(text: str) -> str:
    return " ".join(TAG_RE.sub(" ", text).lower().split())
