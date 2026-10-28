"""Does the CTranslate2 model transcribe like the official model? Applies the pre-set rule in docs/decisions.md.

    python scripts/ct2_check.py --ct2 ct2/E3-int8 --compute-type int8
    python scripts/ct2_check.py --ct2 ct2/E3-float32 --compute-type float32

Uses 500 random VAL clips (seed 0); val, not test, because this result decides what ships. The comparison is the
official E3_val run (transformers, float16, L4) on exactly the same clips. Also prints the real-time factor on
THIS machine, which is not the deployment CPU, so the speed number here is for orientation only.
Writes results/runs/ct2_check_<compute type>/ (summary.json, utterances.csv).
"""
import argparse
import csv
import json
import os
import pathlib
import random
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import asr, data, metrics, text  # noqa: E402

MAX_POINTS_WORSE = 1.0  # pre-set rule: int8 ships if at most 1.0 WER point above the official model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct2", required=True, help="local CTranslate2 folder")
    ap.add_argument("--compute-type", required=True, choices=["int8", "float32"])
    ap.add_argument("--n", type=int, default=500)
    args = ap.parse_args()

    rows = random.Random(0).sample(data.split_rows("val"), args.n)
    with open(ROOT / "results" / "runs" / "E3_val" / "utterances.csv") as f:
        official = {u["utt_id"]: u for u in csv.DictReader(f)}
    dsd = data.load_edacc()
    model = asr.load_model(args.ct2, args.compute_type, cpu_threads=os.cpu_count() or 0)
    settings = asr.decode_settings()

    first_audio, _ = data.fetch(dsd, rows[0])
    asr.transcribe(model, first_audio, settings)  # warm-up, not timed

    utts, secs_audio, secs_compute = [], 0.0, 0.0
    for i, r in enumerate(rows, 1):
        audio, ref = data.fetch(dsd, r)
        t0 = time.perf_counter()
        hyp = asr.transcribe(model, audio, settings)
        secs_compute += time.perf_counter() - t0
        secs_audio += len(audio) / data.SR
        rn, hn = text.normalize(ref), text.normalize(hyp)
        o = official[r["utt_id"]]
        utts.append({"utt_id": r["utt_id"], "group": r["group"], "ref_norm": rn, "hyp_norm": hn,
                     "official_hyp_norm": o["hyp_norm"], "same_as_official": hn == o["hyp_norm"],
                     **metrics.utterance_errors(rn, hn)})
        if i % 100 == 0:
            print(f"  {i}/{len(rows)} clips", flush=True)

    scored_ids = {u["utt_id"] for u in utts if u["n_ref"] > 0}
    ct2 = metrics.pooled([u for u in utts if u["utt_id"] in scored_ids])
    off = metrics.pooled([{k: int(official[i][k]) for k in ("n_ref", "sub", "del", "ins")} for i in scored_ids])
    diff = 100 * (ct2["wer"] - off["wer"])
    same = sum(u["same_as_official"] for u in utts)
    passed = diff <= MAX_POINTS_WORSE
    summary = {"ct2": args.ct2, "compute_type": args.compute_type, "clips": len(utts), "scored": len(scored_ids),
               "wer_ct2": ct2["wer"], "wer_official_same_clips": off["wer"], "difference_points": diff,
               "identical_outputs": same, "rtf_this_machine": secs_compute / secs_audio,
               "cpu_count_this_machine": os.cpu_count(), "rule_max_points_worse": MAX_POINTS_WORSE,
               "within_rule": passed}
    out = ROOT / "results" / "runs" / f"ct2_check_{args.compute_type}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    with open(out / "utterances.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(utts[0]))
        w.writeheader()
        w.writerows(utts)

    print(f"\n{args.compute_type}: WER {ct2['wer']:.2%} vs official {off['wer']:.2%} on the same {len(scored_ids)} clips "
          f"-> {diff:+.2f} points | identical outputs {same}/{len(utts)}")
    print(f"real-time factor on this machine ({os.cpu_count()} CPUs): {summary['rtf_this_machine']:.3f} "
          f"(orientation only; the reported speed comes from the Space)")
    print(f"rule (at most {MAX_POINTS_WORSE} point worse): {'WITHIN' if passed else 'OUTSIDE'}")


if __name__ == "__main__":
    main()
