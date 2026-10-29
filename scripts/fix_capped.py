"""One-off fix: recount "capped" outputs for runs made before the token-cap bug was fixed.

The old check flagged the longest output in every batch, because generate() leaves the end token off it.
This script re-tokenizes each saved output with the run's own tokenizer and flags outputs with at least
max_new_tokens - 6 text tokens (re-tokenizing can differ from the original ids by a few tokens), OR with a
"word" of 40+ characters (a character loop like "mmmm..." re-tokenizes into far fewer tokens than the model
generated). This is a heuristic recount for old runs only; new runs count generated tokens exactly.
WER is not affected. Run on the Mac:

    python scripts/fix_capped.py E0_small_gate1 E0_small_val E0b_medium_val E0c_turbo_val
"""
import csv
import json
import pathlib
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]


def recount(run, count_tokens, cap_at):
    run_dir = ROOT / "results" / "runs" / run
    with open(run_dir / "utterances.csv") as f:
        utts = list(csv.DictReader(f))
    old = sum(u.get("capped") == "True" for u in utts)
    for u in utts:
        u["capped"] = count_tokens(u["hyp"]) >= cap_at or any(len(w) >= 40 for w in u["hyp"].split())
    new = sum(u["capped"] for u in utts)
    fields = list(utts[0]) if "capped" in utts[0] else list(utts[0]) + ["capped"]
    with open(run_dir / "utterances.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(utts)
    summary = json.loads((run_dir / "summary.json").read_text())
    summary["capped_outputs"] = new
    summary["capped_outputs_note"] = "heuristic recount by scripts/fix_capped.py (see its docstring)"
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return old, new


def update_results_csv(new_counts):
    path = ROOT / "results" / "results.csv"
    with open(path) as f:
        rows = list(csv.DictReader(f))
        fields = list(rows[0])
    for r in rows:
        if r["run"] in new_counts:
            r["capped_outputs"] = new_counts[r["run"]]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main(runs):
    from transformers import WhisperTokenizer

    with open(ROOT / "configs" / "decode.yaml") as f:
        cap_at = yaml.safe_load(f)["max_new_tokens"] - 6
    new_counts = {}
    for run in runs:
        model = json.loads((ROOT / "results" / "runs" / run / "summary.json").read_text())["model"]
        tok = WhisperTokenizer.from_pretrained(model)
        old, new = recount(run, lambda s: len(tok(s, add_special_tokens=False).input_ids), cap_at)
        new_counts[run] = new
        print(f"{run:18} capped outputs: old (buggy) {old} -> recounted {new}")
    update_results_csv(new_counts)


if __name__ == "__main__":
    main(sys.argv[1:])
