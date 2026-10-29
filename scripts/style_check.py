"""Does the gain survive once transcription style is taken out? A check added after the test results
(docs/decisions.md), with the rule fixed before computing it. No GPU: re-scores saved outputs.

    python scripts/style_check.py

The reference transcripts are word for word (repeats, "yeah", "like" ...), and the fine-tuned models learned to
write that way while zero-shot Whisper did not. Fillers (uh, um, hmm ...) are already removed from both sides by
normalizer v2. On top of that, the same transform is applied to reference and output:
  A  repeats removed: an immediately repeated 1-, 2- or 3-word sequence is cut to one copy ("i i think" -> "i think")
  B  A plus these discourse words removed everywhere: yeah, like, so, okay, oh, well, right, "you know", "i mean".
     Deliberately aggressive (it also removes some real content words), so it bounds the style effect.
Clips whose reference becomes empty are left out for every system alike (the reference decides).
Paired speaker bootstrap as scripts/bootstrap.py (2,000 resamples, seed 0).
Writes results/error_analysis/style_check_test.json.
"""
import json
import pathlib
import sys
import csv
import collections

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import metrics  # noqa: E402

RUNS = ["E0_small_test", "E0b_medium_test", "E0c_turbo_test", "E1_test", "E3_test", "E3_seed43_test",
        "E3_seed44_test"]
PAIRS = [("E0_small_test", "E3_test"), ("E0c_turbo_test", "E3_test")]
DISCOURSE_WORDS = {"yeah", "like", "so", "okay", "oh", "well", "right"}
DISCOURSE_PAIRS = [("you", "know"), ("i", "mean")]


def drop_repeats(words):
    words = list(words)
    changed = True
    while changed:
        changed = False
        for n in (3, 2, 1):
            i = 0
            while i + 2 * n <= len(words):
                if words[i:i + n] == words[i + n:i + 2 * n]:
                    del words[i + n:i + 2 * n]
                    changed = True
                else:
                    i += 1
    return words


def drop_discourse(words):
    out, i = [], 0
    while i < len(words):
        if tuple(words[i:i + 2]) in DISCOURSE_PAIRS:
            i += 2
            continue
        if words[i] not in DISCOURSE_WORDS:
            out.append(words[i])
        i += 1
    return out


def transform(text, variant):
    w = text.split()
    if variant in ("A", "B"):
        w = drop_repeats(w)
    if variant == "B":
        w = drop_repeats(drop_discourse(w))
    return " ".join(w)


def load(run):
    with open(ROOT / "results" / "runs" / run / "utterances.csv", newline="") as f:
        return {u["utt_id"]: u for u in csv.DictReader(f)}


def score(rows, variant):
    out = {}
    for k, u in rows.items():
        e = metrics.utterance_errors(transform(u["ref_norm"], variant), transform(u["hyp_norm"], variant))
        out[k] = {**e, "speaker": u["speaker"]}
    return out


def paired_bootstrap(base, new, n_boot=2000, seed=0):
    spk = collections.defaultdict(lambda: [0, 0, 0])
    for k, b in base.items():
        if b["n_ref"] == 0:
            continue
        s = spk[b["speaker"]]
        s[0] += b["n_ref"]
        s[1] += b["sub"] + b["del"] + b["ins"]
        s[2] += new[k]["sub"] + new[k]["del"] + new[k]["ins"]
    a = np.array(list(spk.values()), float)
    idx = np.random.default_rng(seed).integers(0, len(a), size=(n_boot, len(a)))
    w = a[idx, 0].sum(1)
    diff = a[idx, 2].sum(1) / w - a[idx, 1].sum(1) / w
    point = a[:, 2].sum() / a[:, 0].sum() - a[:, 1].sum() / a[:, 0].sum()
    return {"diff": point, "ci95": [float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))],
            "speakers": len(a)}


def main():
    rows = {r: load(r) for r in RUNS}
    result = {"rule": __doc__.split("\n\n")[1], "variants": {}}
    for variant in ("official", "A", "B"):
        scored = {r: score(rows[r], variant) for r in RUNS}
        v = {"systems": {}, "pairs": {}}
        for r in RUNS:
            p = metrics.pooled([u for u in scored[r].values() if u["n_ref"] > 0])
            v["systems"][r] = p
        for base, new in PAIRS:
            v["pairs"][f"{new} vs {base}"] = paired_bootstrap(scored[base], scored[new])
        result["variants"][variant] = v
        some = v["systems"]["E3_test"]
        print(f"\n== {variant}: {some['utterances']} clips, {some['ref_words']} reference words")
        for r in RUNS:
            p = v["systems"][r]
            print(f"  {r:16} WER {p['wer']:.2%}  (sub {p['sub_rate']:.2%}, del {p['del_rate']:.2%}, ins {p['ins_rate']:.2%})")
        for name, b in v["pairs"].items():
            print(f"  {name:28} {100 * b['diff']:+.2f} points [95% CI {100 * b['ci95'][0]:+.2f}, {100 * b['ci95'][1]:+.2f}]")
    out = ROOT / "results" / "error_analysis" / "style_check_test.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\nsaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
