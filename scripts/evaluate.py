"""Evaluate a Whisper model on a frozen EdAcc split. Used for Gate 1, every baseline and every fine-tuned model.

Examples (run from the repo root on the L4):
    python scripts/evaluate.py --model openai/whisper-small --split val --limit 64 --run smoke
    python scripts/evaluate.py --model openai/whisper-small --split val \
        --subset splits/gate1_val_sample.txt --run E0_small_gate1

Writes results/runs/<run>/utterances.csv and summary.json, and appends one row to results/results.csv.
"""
import argparse
import csv
import datetime as dt
import json
import pathlib
import subprocess
import sys
import time

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import data, metrics, text  # noqa: E402

RESULT_COLS = ["date", "run", "model", "split", "subset", "utterances", "hours", "wer", "sub_rate", "del_rate",
               "ins_rate", "macro_group_wer", "long_outputs", "commit", "gpu", "eval_minutes"]


def git_commit():
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True).stdout
        return (sha.strip() + ("-dirty" if dirty.strip() else "")) if sha.strip() else "unknown"
    except OSError:
        return "unknown"


def transcribe(model_name, audios, cfg, batch_size):
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, cfg["dtype"]) if device == "cuda" else torch.float32
    proc = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name, dtype=dtype).to(device).eval()
    hyps = []
    for start in range(0, len(audios), batch_size):
        batch = audios[start:start + batch_size]
        feats = proc.feature_extractor(batch, sampling_rate=data.SR, return_tensors="pt").input_features
        with torch.inference_mode():
            ids = model.generate(feats.to(device, dtype), language=cfg["language"], task=cfg["task"],
                                 num_beams=cfg["num_beams"], max_new_tokens=cfg["max_new_tokens"])
        hyps += [h.strip() for h in proc.batch_decode(ids, skip_special_tokens=True)]
        print(f"  {min(start + batch_size, len(audios))}/{len(audios)} utterances", flush=True)
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    return hyps, gpu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id or local checkpoint folder")
    ap.add_argument("--split", choices=["train", "val", "test"], required=True)
    ap.add_argument("--subset", help="file with utterance ids, e.g. splits/gate1_val_sample.txt")
    ap.add_argument("--limit", type=int, help="only the first N utterances (smoke tests)")
    ap.add_argument("--run", required=True, help="run name, e.g. E0_small_gate1")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    out_dir = ROOT / "results" / "runs" / args.run
    if out_dir.exists() and args.run != "smoke":
        raise SystemExit(f"{out_dir} exists. Pick a new run name; never overwrite a result.")
    with open(ROOT / "configs" / "decode.yaml") as f:
        cfg = yaml.safe_load(f)

    rows = data.split_rows(args.split, args.subset, args.limit)
    print(f"{args.run}: {args.model} on {args.split} ({len(rows)} utterances), decode={cfg}")
    dsd = data.load_edacc()
    audios, refs = zip(*(data.fetch(dsd, r) for r in rows))

    t0 = time.time()
    hyps, gpu = transcribe(args.model, list(audios), cfg, args.batch_size)
    minutes = (time.time() - t0) / 60

    utts = []
    for r, ref, hyp in zip(rows, refs, hyps):
        rn, hn = text.normalize(ref), text.normalize(hyp)
        e = metrics.utterance_errors(rn, hn)
        utts.append({"utt_id": r["utt_id"], "speaker": r["speaker"], "group": r["group"],
                     "l1_english": r["l1_english"], "duration": float(r["duration"]),
                     "ref": ref, "hyp": hyp, "ref_norm": rn, "hyp_norm": hn, **e})
    long_outputs = sum(len(u["hyp_norm"].split()) > 2 * u["n_ref"] + 4 for u in utts)  # likely hallucination

    by_group, by_l1 = metrics.report(utts, "group"), metrics.report(utts, "l1_english")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "utterances.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(utts[0]))
        w.writeheader()
        w.writerows(utts)
    summary = {"run": args.run, "model": args.model, "split": args.split, "subset": args.subset,
               "limit": args.limit, "decode": cfg, "commit": git_commit(), "gpu": gpu,
               "eval_minutes": round(minutes, 2), "long_outputs": long_outputs,
               "by_group": by_group, "by_l1_english": by_l1}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    p = by_group["pooled"]
    if args.run != "smoke":
        results = ROOT / "results" / "results.csv"
        new = not results.exists()
        with open(results, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=RESULT_COLS)
            if new:
                w.writeheader()
            w.writerow({"date": dt.date.today().isoformat(), "run": args.run, "model": args.model,
                        "split": args.split, "subset": args.subset or "", "utterances": p["utterances"],
                        "hours": round(sum(u["duration"] for u in utts) / 3600, 2), "wer": round(p["wer"], 4),
                        "sub_rate": round(p["sub_rate"], 4), "del_rate": round(p["del_rate"], 4),
                        "ins_rate": round(p["ins_rate"], 4), "macro_group_wer": round(by_group["macro_over_groups"], 4),
                        "long_outputs": long_outputs, "commit": summary["commit"], "gpu": gpu,
                        "eval_minutes": round(minutes, 2)})

    print(f"\nPOOLED WER {p['wer']:.2%}  (sub {p['sub_rate']:.2%}, del {p['del_rate']:.2%}, ins {p['ins_rate']:.2%}) "
          f"| {p['ref_words']} reference words | macro over groups {by_group['macro_over_groups']:.2%}")
    print(f"likely hallucinations (output > 2x reference + 4 words): {long_outputs} | GPU {gpu} | {minutes:.1f} min")
    for key, rep in (("group", by_group), ("l1_english", by_l1)):
        print(f"\nby {key}:")
        for g, v in rep["by_" + key].items():
            print(f"  {str(g):22} WER {v['wer']:7.2%} | {v['utterances']:5} utts | {v['ref_words']:6} words")
    print("\n5 examples (reference -> hypothesis, normalized):")
    for u in utts[:5]:
        print(f"  REF: {u['ref_norm']}\n  HYP: {u['hyp_norm']}\n")
    if args.subset and "gate1" in args.subset:
        verdict = "EdAcc PASSES Gate 1 (>= 10%)" if p["wer"] >= 0.10 else "EdAcc FAILS Gate 1: measure L2-ARCTIC"
        print(f"GATE 1: pooled WER {p['wer']:.2%} -> {verdict}")


if __name__ == "__main__":
    main()
