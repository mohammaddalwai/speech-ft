"""Paired bootstrap over speakers: is the WER difference between two runs real, or noise? No GPU.

    python scripts/bootstrap.py --base E0_small_test --new E3_test

Speakers are resampled with replacement (all utterances of a speaker move together, because utterances from
one speaker are not independent). For each resample, both runs are scored on the SAME speakers, so the
interval is for the paired difference. "Significant" only if the 95% interval of the difference excludes 0.
Per-group intervals resample speakers inside each group and are shown only for groups with at least 5 speakers:
with fewer, there are too few distinct resamples and the interval looks falsely narrow.
Writes results/runs/<new>/bootstrap_vs_<base>.json.
"""
import argparse
import collections
import csv
import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
MIN_GROUP_SPEAKERS = 5


def per_speaker(base_run, new_run):
    def load(run):
        with open(ROOT / "results" / "runs" / run / "utterances.csv") as f:
            return {u["utt_id"]: u for u in csv.DictReader(f)}

    base, new = load(base_run), load(new_run)
    if set(base) != set(new):
        raise SystemExit("The two runs cover different utterances; bootstrap needs the same split.")
    spk = collections.defaultdict(lambda: {"words": 0, "base": 0, "new": 0, "group": None})
    for k, b in base.items():
        n = int(b["n_ref"])
        if n == 0:
            continue
        s = spk[b["speaker"]]
        s["words"] += n
        s["base"] += int(b["sub"]) + int(b["del"]) + int(b["ins"])
        nw = new[k]
        s["new"] += int(nw["sub"]) + int(nw["del"]) + int(nw["ins"])
        s["group"] = b["group"]
    return spk


def bootstrap(speakers, n_boot, rng, min_speakers=2):
    words = np.array([s["words"] for s in speakers], float)
    eb = np.array([s["base"] for s in speakers], float)
    en = np.array([s["new"] for s in speakers], float)
    point = {"base": eb.sum() / words.sum(), "new": en.sum() / words.sum()}
    point["diff"] = point["new"] - point["base"]
    if len(speakers) < min_speakers:
        return point, None
    idx = rng.integers(0, len(speakers), size=(n_boot, len(speakers)))
    w = words[idx].sum(1)
    samples = {"base": eb[idx].sum(1) / w, "new": en[idx].sum(1) / w}
    samples["diff"] = samples["new"] - samples["base"]
    ci = {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in samples.items()}
    return point, ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--n", type=int, default=2000, help="bootstrap resamples")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    spk = per_speaker(args.base, args.new)

    groups = collections.defaultdict(list)
    for s in spk.values():
        groups[s["group"]].append(s)
    out = {"base": args.base, "new": args.new, "resamples": args.n, "speakers": len(spk), "groups": {}}

    point, ci = bootstrap(list(spk.values()), args.n, rng)
    sig = ci["diff"][0] > 0 or ci["diff"][1] < 0
    out["pooled"] = {"point": point, "ci95": ci, "significant": sig}
    print(f"{args.base} -> {args.new} | {len(spk)} speakers, {args.n} resamples")
    print(f"  pooled WER {point['base']:.2%} -> {point['new']:.2%} | difference {point['diff']:+.2%} "
          f"[95% CI {ci['diff'][0]:+.2%}, {ci['diff'][1]:+.2%}] | relative {point['diff'] / point['base']:+.1%} | "
          f"{'SIGNIFICANT' if sig else 'not significant'}")
    print(f"  {args.new} WER 95% CI [{ci['new'][0]:.2%}, {ci['new'][1]:.2%}]")
    print(f"\n  {'group':22} {'spk':>3}  {'base':>7} {'new':>7}  difference [95% CI]")
    for g, ss in sorted(groups.items()):
        p, c = bootstrap(ss, args.n, rng, MIN_GROUP_SPEAKERS)
        cell = f"[{c['diff'][0]:+.2%}, {c['diff'][1]:+.2%}]" if c else "(too few speakers for an interval)"
        print(f"  {g:22} {len(ss):3}  {p['base']:7.2%} {p['new']:7.2%}  {p['diff']:+.2%} {cell}")
        out["groups"][g] = {"speakers": len(ss), "point": p, "ci95": c}
    path = ROOT / "results" / "runs" / args.new / f"bootstrap_vs_{args.base}.json"
    path.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
