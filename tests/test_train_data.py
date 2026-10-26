"""Label construction for training. Downloads whisper-small's tokenizer and feature-extractor configs once."""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
torch = pytest.importorskip("torch")
from src import train_data  # noqa: E402


@pytest.fixture(scope="module")
def processor():
    from transformers import WhisperProcessor

    p = WhisperProcessor.from_pretrained("openai/whisper-small")
    p.tokenizer.set_prefix_tokens(language="en", task="transcribe", predict_timestamps=False)
    return p


def test_collator_labels(processor):
    tok = processor.tokenizer
    start = tok.convert_tokens_to_ids("<|startoftranscript|>")
    items = [{"input_features": np.zeros((80, 3000), np.float32), "labels": tok(t).input_ids}
             for t in ("yeah", "i think so")]
    batch = train_data.Collator(processor, start)(items)
    labels = batch["labels"]
    assert tuple(batch["input_features"].shape) == (2, 80, 3000)
    # <|startoftranscript|> is dropped (the model adds it back); labels start with the language token
    assert tok.convert_ids_to_tokens(int(labels[0, 0])) == "<|en|>"
    assert tok.convert_ids_to_tokens(int(labels[0, 1])) == "<|transcribe|>"
    assert tok.convert_ids_to_tokens(int(labels[0, 2])) == "<|notimestamps|>"
    # the shorter label is padded with -100 after its end token, so padding is ignored by the loss
    assert (labels[0] == -100).any() and not (labels[1] == -100).any()
    assert tok.decode([t for t in labels[1].tolist() if t != -100], skip_special_tokens=True).strip() == "i think so"


def test_wer_from_ids_ignores_padding_and_filler_only_refs(processor):
    tok = processor.tokenizer
    pred = [tok("yeah").input_ids + [-100] * 3, tok("i think").input_ids, tok("uh").input_ids]
    lab = [tok("yeah").input_ids[1:] + [-100] * 3, tok("i think so").input_ids[1:], tok("uhh").input_ids[1:]]
    assert abs(train_data.wer_from_ids(pred, lab, tok) - 0.25) < 1e-9  # 1 deletion / 4 words
