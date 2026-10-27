"""Loading EdAcc utterances listed in the frozen split files. Audio is decoded here (no torchcodec)."""
import csv
import io
import pathlib

import numpy as np
import soundfile as sf
import soxr

REPO = "edinburghcstr/edacc"
REVISION = "d9ae7bd344f0562b766ec93ee5ce8f2f9568ce66"  # must match splits/manifest.json
ROOT = pathlib.Path(__file__).resolve().parents[1]
SR = 16000


def split_rows(split, subset_file=None, limit=None):
    """Rows of splits/<split>.csv, optionally restricted to the ids in subset_file, then the first `limit`."""
    with open(ROOT / "splits" / f"{split}.csv") as f:
        rows = list(csv.DictReader(f))
    if subset_file:
        keep = set(pathlib.Path(subset_file).read_text().split())
        rows = [r for r in rows if r["utt_id"] in keep]
        missing = keep - {r["utt_id"] for r in rows}
        if missing:
            raise ValueError(f"{len(missing)} ids in {subset_file} are not in the {split} split")
    return rows[:limit] if limit else rows


def load_edacc():
    from datasets import Audio, load_dataset

    dsd = load_dataset(REPO, revision=REVISION)
    return {name: ds.cast_column("audio", Audio(decode=False)) for name, ds in dsd.items()}


def decode_audio(audio_field):
    """-> mono float32 numpy array at 16 kHz."""
    src = io.BytesIO(audio_field["bytes"]) if audio_field.get("bytes") else audio_field["path"]
    wav, sr = sf.read(src, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        wav = soxr.resample(wav, sr, SR)
    return np.ascontiguousarray(wav, dtype=np.float32)


def fetch(dsd, row):
    """Audio and reference text for one split row. Fails loudly if the dataset rows have shifted."""
    item = dsd[row["official_split"]][int(row["row"])]
    if item["speaker"] != row["speaker"]:
        raise RuntimeError(f"{row['utt_id']}: expected {row['speaker']}, found {item['speaker']} (dataset changed?)")
    return decode_audio(item["audio"]), item["text"]


# ---------- LibriSpeech (forgetting check only) ----------
LIBRI_REPO = "openslr/librispeech_asr"
LIBRI_FILES = {"test-clean": "clean/test/*.parquet", "test-other": "other/test/*.parquet"}
MAX_SEC = 30.0


def librispeech_items(split, limit=None):
    """Rows, audio and reference texts for one LibriSpeech test set. Rule L1: clips longer than 30 s are dropped
    (Whisper's input limit, same as R2 for EdAcc) and counted. Only that split's files are downloaded."""
    from datasets import Audio, load_dataset

    ds = load_dataset(LIBRI_REPO, data_files={"test": LIBRI_FILES[split]}, split="test")
    ds = ds.cast_column("audio", Audio(decode=False))
    rows, audios, refs, dropped = [], [], [], 0
    for i in range(min(limit, len(ds)) if limit else len(ds)):
        item = ds[i]
        wav = decode_audio(item["audio"])
        if len(wav) / SR > MAX_SEC:
            dropped += 1
            continue
        rows.append({"utt_id": str(item["id"]), "speaker": str(item["speaker_id"]), "group": split,
                     "l1_english": "n/a", "duration": round(len(wav) / SR, 3)})
        audios.append(wav)
        refs.append(item["text"])
    return rows, audios, refs, dropped


def librispeech_replay(n, seed, shards=4):
    """E3 replay data: n random clips (<= 30 s) from LibriSpeech train-clean-100. LibriSpeech's train and test
    speakers do not overlap, so this cannot leak into the forgetting check. Downloads only a few shards."""
    import random

    from datasets import Audio, load_dataset
    from huggingface_hub import HfApi

    files = sorted(f for f in HfApi().list_repo_files(LIBRI_REPO, repo_type="dataset")
                   if f.startswith("clean/train.100/") and f.endswith(".parquet"))[:shards]
    ds = load_dataset(LIBRI_REPO, data_files={"train": files}, split="train")
    ds = ds.cast_column("audio", Audio(decode=False))
    order = list(range(len(ds)))
    random.Random(seed).shuffle(order)
    picked, seconds = [], 0.0
    for i in order:
        a = ds[i]["audio"]
        info = sf.info(io.BytesIO(a["bytes"]) if a.get("bytes") else a["path"])
        dur = info.frames / info.samplerate
        if dur <= MAX_SEC:
            picked.append(i)
            seconds += dur
            if len(picked) == n:
                break
    if len(picked) < n:
        raise ValueError(f"only {len(picked)} replay clips in {len(files)} shards; raise `shards`")
    return ds, picked, files, round(seconds / 3600, 2)
