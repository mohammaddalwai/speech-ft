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
