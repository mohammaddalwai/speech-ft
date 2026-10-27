"""Training-seed variation: mean and standard deviation of WER over runs that differ only in seed. No GPU.

    python scripts/seed_summary.py E3_test E3_seed43_test E3_seed44_test

The bootstrap interval (scripts/bootstrap.py) covers which speakers happen to be in the test set; this covers
how much the result moves when the same recipe is trained again. Report both.
"""
import csv
import pathlib
import statistics as st
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main(runs):
    with open(ROOT / "results" / "results.csv") as f:
        rows = {r["run"]: r for r in csv.DictReader(f)}
    for col in ("wer", "sub_rate", "del_rate", "ins_rate", "macro_group_wer"):
        vals = [float(rows[r][col]) for r in runs]
        sd = st.stdev(vals) if len(vals) > 1 else float("nan")
        print(f"{col:16} " + "  ".join(f"{v:.2%}" for v in vals) + f"  | mean {st.mean(vals):.2%} +/- {sd:.2%} (std)")


if __name__ == "__main__":
    main(sys.argv[1:])
