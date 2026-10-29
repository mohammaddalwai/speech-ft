"""Phase 1, step 3: build and FREEZE the train/val/test splits. Run once; never rerun after seeing any WER.

Run on a Colab CPU runtime:
    python scripts/make_splits.py | tee results/splits_summary.txt

Writes (all small, no audio, no transcripts):
    splits/train.csv, splits/val.csv, splits/test.csv   one row per usable utterance
    splits/gate1_val_sample.txt                          1,000 val utterance ids for Gate 1
    splits/manifest.json                                 dataset revision, rules, seed, counts
"""
import collections
import csv
import io
import json
import random
import re
import statistics as st

import soundfile as sf
import yaml

REPO = "edinburghcstr/edacc"
REVISION = "d9ae7bd344f0562b766ec93ee5ce8f2f9568ce66"  # pinned: utterance ids refer to this dataset version
SEED = 0
N_CANDIDATES = 20000
MIN_SPK_MIN = 2.0     # a speaker counts toward a group in a split only with >= 2 minutes of usable audio
TARGETS = {"train": 0.70, "val": 0.10, "test": 0.20}  # share of usable hours, split by conversation
MAX_DEV = 0.15        # val and test hours must be within 15% (relative) of their targets
MAX_SEC = 30.0
GATE1_N = 1000
DROP_MARKERS = ("IGNORE_TIME_SEGMENT_IN_SCORING", "<FOREIGN>")
RULES = {
    "R1": "drop segments containing IGNORE_TIME_SEGMENT_IN_SCORING (untranscribed speech)",
    "R1b": "drop segments containing <FOREIGN> (untranscribed foreign-language speech)",
    "R2": f"drop segments longer than {MAX_SEC:.0f} s (Whisper limit; no word timings to cut them)",
    "R3": "drop segments that are empty after Whisper's English normalizer (tags only)",
    "R6": (f"split by conversation {TARGETS}, best of {N_CANDIDATES} random candidates (seed {SEED}): "
           f"val/test hours within {MAX_DEV:.0%} of target, then up to 2 speakers per group in test, then every "
           f"group in train, then in val, then closest hours; a speaker counts only with >= {MIN_SPK_MIN:.0f} min"),
    "R7": "accent groups from configs/accent_groups.yaml; groups below min_speakers merged into 'other'",
}


# ---------- data ----------
def build_rows(dsd, norm):
    from datasets import Audio

    rows = []
    for split, ds in dsd.items():
        ds = ds.cast_column("audio", Audio(decode=False))
        for i, r in enumerate(ds):
            a = r["audio"]
            info = sf.info(io.BytesIO(a["bytes"]) if a.get("bytes") else a["path"])
            text, dur = r["text"], info.frames / info.samplerate
            reason = None
            if any(m in text for m in DROP_MARKERS):
                reason = "marker"
            elif dur > MAX_SEC:
                reason = "too_long"
            elif not norm(text).strip():
                reason = "empty"
            rows.append(dict(utt_id=f"{split}-{i:05d}", official_split=split, row=i, speaker=r["speaker"],
                             conversation=r["speaker"].rsplit("-", 1)[0], l1=r["l1"], duration=round(dur, 3),
                             drop=reason))
    return rows


def assign_groups(rows, cfg):
    l1_to_group = {l1: g for g, l1s in cfg["groups"].items() for l1 in l1s}
    spk_l1 = {r["speaker"]: r["l1"] for r in rows}
    group = {s: cfg["speaker_overrides"].get(s, l1_to_group.get(l1, "UNMAPPED")) for s, l1 in spk_l1.items()}
    unmapped = sorted(s for s, g in group.items() if g == "UNMAPPED")
    if unmapped:
        raise SystemExit(f"Speakers without a group: {unmapped}. Fix configs/accent_groups.yaml first.")
    counts = collections.Counter(group.values())
    small = {g for g, n in counts.items() if n < cfg["min_speakers"]}
    final = {s: ("other" if g in small else g) for s, g in group.items()}
    return final, small, {s: "English" in l1 for s, l1 in spk_l1.items()}


# ---------- split search ----------
def conversation_info(usable, group):
    """Per conversation: usable hours and the speakers who count toward a group (>= MIN_SPK_MIN minutes)."""
    secs = collections.defaultdict(lambda: collections.defaultdict(float))
    for r in sorted(usable, key=lambda r: r["utt_id"]):  # fixed order so float sums are reproducible
        secs[r["conversation"]][r["speaker"]] += float(r["duration"])
    info = {}
    for c, spk in secs.items():
        info[c] = {"hours": round(sum(spk.values()) / 3600, 6),
                   "groups": [group[s] for s, t in sorted(spk.items()) if t / 60 >= MIN_SPK_MIN]}
    return info


def candidate(convs, info, total, rng):
    order = sorted(convs)
    rng.shuffle(order)
    h, assign = {s: 0.0 for s in TARGETS}, {}
    for c in order:
        ch = info[c]["hours"]
        for s in ("test", "val"):  # give a conversation to test/val only if it moves them closer to target
            target = TARGETS[s] * total
            if abs(h[s] + ch - target) < abs(h[s] - target):
                break
        else:
            s = "train"
        assign[c] = s
        h[s] += ch
    return assign, h


def score(assign, h, info, total, groups):
    dev = max(abs(h[s] / total - TARGETS[s]) / TARGETS[s] for s in ("val", "test"))
    n = {s: collections.Counter() for s in TARGETS}
    for c, s in assign.items():
        n[s].update(info[c]["groups"])
    test_cov = sum(min(n["test"][g], 2) for g in groups)   # up to 2 counted speakers per group in test
    train_cov = sum(min(n["train"][g], 1) for g in groups)
    val_cov = sum(min(n["val"][g], 1) for g in groups)
    return dev, test_cov, train_cov, val_cov


def choose_split(info, seed=SEED, n=N_CANDIDATES):
    rng = random.Random(seed)
    total = sum(c["hours"] for c in info.values())
    groups = sorted({g for c in info.values() for g in c["groups"]})
    best, best_key = None, None
    for _ in range(n):
        assign, h = candidate(list(info), info, total, rng)
        dev, test_cov, train_cov, val_cov = score(assign, h, info, total, groups)
        key = (dev <= MAX_DEV, test_cov, train_cov, val_cov, -dev)
        if best_key is None or key > best_key:
            best, best_key = assign, key
    if not best_key[0]:
        print(f"WARNING: no candidate within {MAX_DEV:.0%} of target hours; using the closest one.")
    stats = {"within_tolerance": best_key[0], "test_coverage": f"{best_key[1]}/{2 * len(groups)}",
             "train_coverage": f"{best_key[2]}/{len(groups)}", "val_coverage": f"{best_key[3]}/{len(groups)}",
             "max_rel_dev": round(-best_key[4], 4)}
    return best, stats


# ---------- outputs ----------
def write_outputs(usable, assign, group, l1_eng, small, stats, revision, n_dropped, out="splits"):
    by_split = collections.defaultdict(list)
    for r in usable:
        by_split[assign[r["conversation"]]].append(r)
    cols = ["utt_id", "official_split", "row", "speaker", "conversation", "group", "l1_english", "duration"]
    for s in TARGETS:
        with open(f"{out}/{s}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in sorted(by_split[s], key=lambda r: r["utt_id"]):
                w.writerow({**{k: r[k] for k in cols if k in r}, "group": group[r["speaker"]],
                            "l1_english": l1_eng[r["speaker"]]})

    gate1 = sorted(random.Random(SEED).sample(sorted(r["utt_id"] for r in by_split["val"]),
                                              min(GATE1_N, len(by_split["val"]))))
    with open(f"{out}/gate1_val_sample.txt", "w") as f:
        f.write("\n".join(gate1) + "\n")

    summary = {}
    for s in TARGETS:
        rs = by_split[s]
        summary[s] = {
            "utterances": len(rs), "hours": round(sum(r["duration"] for r in rs) / 3600, 2),
            "speakers": len({r["speaker"] for r in rs}), "conversations": len({r["conversation"] for r in rs}),
            "hours_by_group": {g: round(sum(r["duration"] for r in rs if group[r["speaker"]] == g) / 3600, 2)
                               for g in sorted(set(group.values()))},
            "speakers_by_group": dict(sorted(collections.Counter(
                group[sp] for sp in {r["speaker"] for r in rs}).items())),
        }
    manifest = {"dataset": REPO, "revision": revision, "rules": RULES,
                "merged_into_other": sorted(small), "dropped_segments": n_dropped, "search": stats,
                "gate1_sample": len(gate1), "splits": summary}
    with open(f"{out}/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def print_summary(m):
    print(f"\ndataset {m['dataset']} @ {m['revision']}")
    print("dropped segments by reason:", m["dropped_segments"], "| merged into 'other':", m["merged_into_other"])
    print("search:", m["search"])
    for s, d in m["splits"].items():
        print(f"  {s:5}: {d['utterances']:6} utts | {d['hours']:5.2f} h | {d['speakers']:3} speakers | "
              f"{d['conversations']:2} conversations")
    groups = list(m["splits"]["train"]["hours_by_group"])
    print(f"\n  {'group':22} " + " ".join(f"{s:>16}" for s in m["splits"]))
    for g in groups:
        cells = [f"{m['splits'][s]['speakers_by_group'].get(g, 0):3} spk {m['splits'][s]['hours_by_group'][g]:5.2f} h"
                 for s in m["splits"]]
        print(f"  {g:22} " + " ".join(f"{c:>16}" for c in cells))
    print(f"\nGate 1 sample: {m['gate1_sample']} val utterances")


def main():
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import WhisperTokenizer

    with open("configs/accent_groups.yaml") as f:
        cfg = yaml.safe_load(f)
    latest = HfApi().dataset_info(REPO).sha
    if latest != REVISION:
        print(f"NOTE: {REPO} has a newer revision ({latest}); using the pinned one ({REVISION}).")
    revision = REVISION
    norm = WhisperTokenizer.from_pretrained("openai/whisper-small").normalize
    rows = build_rows(load_dataset(REPO, revision=revision), norm)
    group, small, l1_eng = assign_groups(rows, cfg)
    usable = [r for r in rows if r["drop"] is None]
    n_dropped = dict(collections.Counter(r["drop"] for r in rows if r["drop"]))
    assign, stats = choose_split(conversation_info(usable, group))
    print_summary(write_outputs(usable, assign, group, l1_eng, small, stats, revision, n_dropped))


if __name__ == "__main__":
    main()
