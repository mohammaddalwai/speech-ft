"""Forgetting check: did fine-tuning make LibriSpeech worse? Applies the pre-set E3 trigger. No GPU.

    python scripts/forgetting_check.py --base E0_small --new E1

Reads results/results.csv rows <name>_libri_clean and <name>_libri_other for both models.
E3 (replay) is triggered if the new model's WER is more than 10% higher (relative) on either test set.
"""
import argparse
import csv
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
THRESHOLD = 0.10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--new", required=True)
    args = ap.parse_args()
    with open(ROOT / "results" / "results.csv") as f:
        wer = {r["run"]: float(r["wer"]) for r in csv.DictReader(f)}
    triggered = False
    for part in ("clean", "other"):
        b, n = wer[f"{args.base}_libri_{part}"], wer[f"{args.new}_libri_{part}"]
        rel = (n - b) / b
        triggered |= rel > THRESHOLD
        print(f"test-{part:5}: {args.base} {b:.2%} -> {args.new} {n:.2%} | {n - b:+.2%} absolute, {rel:+.1%} relative")
    verdict = "FAILED" if triggered else "passed"
    print(f"\nforgetting rule {verdict} (fails if > {THRESHOLD:.0%} relative worse on either set)")
    if args.new == "E1":
        print(f"E3 (replay) {'TRIGGERED' if triggered else 'not triggered'}")


if __name__ == "__main__":
    main()
