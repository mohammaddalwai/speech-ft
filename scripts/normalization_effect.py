"""How much does text normalization change WER? Re-scores a finished run from its saved raw texts. No GPU.

    python scripts/normalization_effect.py --run E0_small_gate1

Writes results/runs/<run>/normalization_effect.json. Does not touch results/results.csv.
"""
import argparse
import csv
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import metrics, text  # noqa: E402

NORMALIZERS = [
    ("basic: lowercase, no punctuation or tags", text.normalize_basic),
    ("Whisper English normalizer", text.normalize_whisper_only),
    (f"{text.NORMALIZER_VERSION}: Whisper + our rules (used for all results)", text.normalize),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    run_dir = ROOT / "results" / "runs" / args.run
    with open(run_dir / "utterances.csv") as f:
        utts = list(csv.DictReader(f))

    out = {}
    for name, norm in NORMALIZERS:
        scored = []
        for u in utts:
            r, h = norm(u["ref"]), norm(u["hyp"])
            if r.strip():  # an utterance whose reference is empty under this normalizer has no words to score
                scored.append(metrics.utterance_errors(r, h))
        p = metrics.pooled(scored)
        out[name] = {**p, "skipped_empty_refs": len(utts) - len(scored)}
        print(f"{name:62} WER {p['wer']:7.2%} | {p['ref_words']:6} reference words | "
              f"{len(utts) - len(scored)} utterances with empty reference skipped")
    (run_dir / "normalization_effect.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
