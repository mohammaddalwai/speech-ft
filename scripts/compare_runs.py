"""Where did a WER change come from? Follows every reference word from one run to another on the same utterances.

    python scripts/compare_runs.py --base E0_small_val --new E1_val

For each reference word: correct / substituted / deleted in the base run, and the same in the new run.
  deleted -> correct:   words the base model skipped and the new model gets right (often transcription style)
  substituted -> correct: words the base model misheard and the new model gets right (recognition gains)
  correct -> wrong:     words the base model got right and the new model breaks (regressions)
Writes results/runs/<new>/compare_<base>.json. No GPU.
"""
import argparse
import collections
import csv
import json
import pathlib

import jiwer

ROOT = pathlib.Path(__file__).resolve().parents[1]
LABEL = {"equal": "C", "substitute": "S", "delete": "D"}
NAMES = {"C": "correct", "S": "substituted", "D": "deleted"}


def load(run):
    with open(ROOT / "results" / "runs" / run / "utterances.csv") as f:
        return {u["utt_id"]: u for u in csv.DictReader(f)}


def word_labels(u):
    """One label per reference word: C, S or D. None for references with no words."""
    ref, hyp = u["ref_norm"].split(), u["hyp_norm"]
    if not ref:
        return None
    if not hyp.strip():
        return ["D"] * len(ref)
    out = [None] * len(ref)
    for a in jiwer.process_words(u["ref_norm"], hyp).alignments[0]:
        for k in range(a.ref_start_idx, a.ref_end_idx):
            out[k] = LABEL[a.type]
    return out


def compare(base, new):
    if set(base) != set(new):
        raise SystemExit("The two runs cover different utterances; compare runs on the same split only.")
    moves, words = collections.Counter(), collections.defaultdict(collections.Counter)
    repeats = 0
    for k, ub in base.items():
        lb, ln = word_labels(ub), word_labels(new[k])
        if lb is None:
            continue
        ref = ub["ref_norm"].split()
        for i, (x, y) in enumerate(zip(lb, ln)):
            moves[(x, y)] += 1
            if x != y:
                words[f"{NAMES[x]} -> {NAMES[y]}"][ref[i]] += 1
            if x == "D" and y == "C" and ((i > 0 and ref[i - 1] == ref[i]) or (i + 1 < len(ref) and ref[i + 1] == ref[i])):
                repeats += 1
    return moves, words, repeats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--new", required=True)
    args = ap.parse_args()
    moves, words, repeats = compare(load(args.base), load(args.new))
    n = sum(moves.values())
    pts = lambda *keys: 100 * sum(moves[k] for k in keys) / n  # noqa: E731

    print(f"{args.base} -> {args.new}: {n} reference words (cells are % of all reference words)")
    header = "before / after"
    print(f"{header:16}" + "".join(f"{NAMES[y]:>14}" for y in "CSD"))
    for x in "CSD":
        print(f"{NAMES[x]:16}" + "".join(f"{100 * moves[(x, y)] / n:13.2f}%" for y in "CSD"))
    summary = {
        "deleted_to_correct": pts(("D", "C")), "substituted_to_correct": pts(("S", "C")),
        "correct_to_wrong": pts(("C", "S"), ("C", "D")), "deleted_to_substituted": pts(("D", "S")),
        "deleted_to_correct_that_are_repeats_pct": 100 * repeats / max(moves[("D", "C")], 1),
    }
    print(f"\ndeleted -> correct     {summary['deleted_to_correct']:5.2f} pts "
          f"({summary['deleted_to_correct_that_are_repeats_pct']:.0f}% of them are immediate repeats)")
    print(f"substituted -> correct {summary['substituted_to_correct']:5.2f} pts")
    print(f"correct -> wrong       {summary['correct_to_wrong']:5.2f} pts")
    print(f"deleted -> substituted {summary['deleted_to_substituted']:5.2f} pts (attempted now, but wrong)")
    for move in ("deleted -> correct", "substituted -> correct", "correct -> substituted"):
        print(f"\ntop words, {move}:", words[move].most_common(15))
    out = {"base": args.base, "new": args.new, "reference_words": n, "summary": summary,
           "moves_pct": {f"{NAMES[x]} -> {NAMES[y]}": 100 * moves[(x, y)] / n for x in "CSD" for y in "CSD"},
           "top_words": {m: c.most_common(30) for m, c in words.items()}}
    path = ROOT / "results" / "runs" / args.new / f"compare_{args.base}.json"
    path.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
