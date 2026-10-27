"""Training items for Whisper: log-mel features + label token ids of the R5 training text."""
import torch

from src import data, metrics, text

MAX_LABEL_TOKENS = 448  # Whisper's decoder length limit


class SplitDataset(torch.utils.data.Dataset):
    """Decodes audio on the fly (no big feature cache on disk)."""

    def __init__(self, rows, dsd, processor):
        self.rows, self.dsd = rows, dsd
        self.fe, self.tok = processor.feature_extractor, processor.tokenizer

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        audio, raw = data.fetch(self.dsd, self.rows[i])
        feats = self.fe(audio, sampling_rate=data.SR, return_tensors="np").input_features[0]
        labels = self.tok(text.training_text(raw)).input_ids  # prefix: <|startoftranscript|><|en|><|transcribe|>...
        if len(labels) > MAX_LABEL_TOKENS:
            raise ValueError(f"{self.rows[i]['utt_id']}: {len(labels)} label tokens > {MAX_LABEL_TOKENS}")
        return {"input_features": feats, "labels": labels}


class ReplayDataset(torch.utils.data.Dataset):
    """E3: LibriSpeech clips mixed into training to reduce forgetting. Same item format as SplitDataset."""

    def __init__(self, ds, indices, processor):
        self.ds, self.idx = ds, indices
        self.fe, self.tok = processor.feature_extractor, processor.tokenizer

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        item = self.ds[self.idx[i]]
        audio = data.decode_audio(item["audio"])
        feats = self.fe(audio, sampling_rate=data.SR, return_tensors="np").input_features[0]
        return {"input_features": feats, "labels": self.tok(text.training_text(item["text"])).input_ids}


class Collator:
    """Pads features and labels. Padded label positions become -100 so the loss ignores them. The leading
    <|startoftranscript|> is dropped from labels because the model adds it itself when shifting labels right."""

    def __init__(self, processor, decoder_start_token_id):
        self.processor, self.start = processor, decoder_start_token_id

    def __call__(self, items):
        batch = self.processor.feature_extractor.pad(
            [{"input_features": it["input_features"]} for it in items], return_tensors="pt")
        lab = self.processor.tokenizer.pad([{"input_ids": it["labels"]} for it in items], return_tensors="pt")
        labels = lab["input_ids"].masked_fill(lab["attention_mask"].ne(1), -100)
        if (labels[:, 0] == self.start).all():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def wer_from_ids(pred_ids, label_ids, tokenizer):
    """Pooled val WER for checkpoint selection, with the same normalizer as evaluate.py."""
    # the Trainer pads both predictions and labels with -100 when it joins batches of different lengths
    label_ids = [[int(t) for t in seq if t != -100] for seq in label_ids]
    pred_ids = [[int(t) for t in seq if t != -100] for seq in pred_ids]
    hyps = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    refs = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    utts = [metrics.utterance_errors(text.normalize(r), text.normalize(h)) for r, h in zip(refs, hyps)]
    return metrics.pooled([u for u in utts if u["n_ref"] > 0])["wer"]
