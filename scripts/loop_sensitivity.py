"""Post-hoc sensitivity check: how much do looping outputs move each system's WER? No GPU.

    python scripts/loop_sensitivity.py E0_small_test E0b_medium_test E0c_turbo_test E1_test E3_test E3_seed43_test E3_seed44_test

The official numbers count loops (never removed). This re-scores every run WITHOUT the clips on which ANY of the
listed runs hit the token cap, so all runs are compared on the same clips. Report it as a sensitivity check,
labelled post-hoc, next to the official numbers; never instead of them.
"""
import csv
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main(runs):
    data = {}
    for r in runs:
        with open(ROOT / "results" / "runs" / r / "utterances.csv") as f:
            data[r] = {u["utt_id"]: u for u in csv.DictReader(f)}
    loop_clips = sorted({k for d in data.values() for k, u in d.items() if u["capped"] == "True"})
    err = lambda u: int(u["sub"]) + int(u["del"]) + int(u["ins"])  # noqa: E731
    out = {"loop_clips": loop_clips, "runs": {}}
    print(f"{len(loop_clips)} clips where at least one run hit the token cap\n")
    print(f"{'run':18} {'official WER':>13} {'own loops':>10} {'their errors':>13} {'WER without loop clips':>24}")
    for r, d in data.items():
        scored = [u for u in d.values() if int(u["n_ref"]) > 0]
        keep = [u for u in scored if u["utt_id"] not in loop_clips]
        own = [u for u in scored if u["capped"] == "True"]
        w_all = sum(map(err, scored)) / sum(int(u["n_ref"]) for u in scored)
        w_keep = sum(map(err, keep)) / sum(int(u["n_ref"]) for u in keep)
        out["runs"][r] = {"official_wer": w_all, "own_loops": len(own), "loop_errors": sum(map(err, own)),
                          "wer_without_loop_clips": w_keep}
        print(f"{r:18} {w_all:13.2%} {len(own):10} {sum(map(err, own)):13} {w_keep:24.2%}")
    path = ROOT / "results" / "error_analysis" / "loop_sensitivity.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
