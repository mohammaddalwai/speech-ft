"""Leakage guard: fails if any utterance, speaker or conversation appears in more than one split.

    pytest tests/
"""
import csv
import pathlib

import pytest

SPLITS_DIR = pathlib.Path(__file__).resolve().parents[1] / "splits"
NAMES = ("train", "val", "test")
KEYS = ("utt_id", "speaker", "conversation")


def find_overlaps(data):
    """data: {split_name: [row dicts]} -> list of (key, split_a, split_b, examples)."""
    problems = []
    for key in KEYS:
        sets = {s: {r[key] for r in rows} for s, rows in data.items()}
        names = list(data)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                both = sets[a] & sets[b]
                if both:
                    problems.append((key, a, b, sorted(both)[:5]))
    return problems


def load_splits():
    if not all((SPLITS_DIR / f"{s}.csv").exists() for s in NAMES):
        pytest.skip("split files not created yet")
    data = {}
    for s in NAMES:
        with open(SPLITS_DIR / f"{s}.csv") as f:
            data[s] = list(csv.DictReader(f))
    return data


def test_checker_catches_planted_leak():
    """Proves the checker can fail: a speaker planted in two splits must be reported."""
    row = {"utt_id": "u1", "speaker": "S1", "conversation": "C1"}
    data = {"train": [row], "val": [{**row, "utt_id": "u2"}], "test": []}
    keys = {p[0] for p in find_overlaps(data)}
    assert keys == {"speaker", "conversation"}


def test_no_overlap_between_splits():
    problems = find_overlaps(load_splits())
    assert not problems, f"leakage between splits: {problems}"


def test_every_split_is_nonempty_and_clips_fit_whisper():
    for s, rows in load_splits().items():
        assert rows, f"{s} split is empty"
        assert max(float(r["duration"]) for r in rows) <= 30.0, f"{s} has a clip longer than 30 s"


def test_gate1_sample_comes_from_val():
    data = load_splits()
    sample = (SPLITS_DIR / "gate1_val_sample.txt").read_text().split()
    assert sample and set(sample) <= {r["utt_id"] for r in data["val"]}
