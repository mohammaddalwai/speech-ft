"""Fine-tune Whisper on the frozen EdAcc train split; keep the checkpoint with the best val WER.

Run from the repo root on Colab:
  Debug, L4 (~15 min):  python scripts/train.py --config configs/E1.yaml --run E1_debug --max-steps 200 \
                            --warmup-steps 20 --eval-steps 100 --train-limit 1600 --eval-limit 200 --no-push
  Overfit 32 clips:     python scripts/train.py --config configs/E1.yaml --run E1_overfit --overfit 32 \
                            --max-steps 120 --warmup-steps 10 --lr 1e-4 --eval-steps 40 --no-push
  Full run (A100):      python scripts/train.py --config configs/E1.yaml --run E1
  Resume after a drop:  python scripts/train.py --config configs/E1.yaml --run E1 --resume

The val WER printed during training is only for picking the checkpoint. The official val number comes from
scripts/evaluate.py on the saved model, with exactly the same settings as the baselines.
"""
import argparse
import json
import pathlib
import random
import subprocess
import sys
import time

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import data, train_data  # noqa: E402


def git_commit():
    out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout
    return out.strip() or "unknown"


def pick_rows(args):
    train_rows, eval_rows = data.split_rows("train"), data.split_rows("val")
    rng = random.Random(0)
    if args.overfit:  # sanity check: can the model memorize a few clips? (evaluated on the same clips)
        train_rows = rng.sample(train_rows, args.overfit)
        return train_rows, list(train_rows)
    if args.train_limit:
        train_rows = rng.sample(train_rows, args.train_limit)
    if args.eval_limit:
        eval_rows = rng.sample(eval_rows, args.eval_limit)
    return train_rows, eval_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--warmup-steps", type=int)
    ap.add_argument("--eval-steps", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--train-limit", type=int, help="random subset of train utterances (debug)")
    ap.add_argument("--eval-limit", type=int, help="random subset of val utterances (debug)")
    ap.add_argument("--overfit", type=int, help="train AND evaluate on this many train clips")
    ap.add_argument("--no-push", action="store_true", help="do not push checkpoints to the Hugging Face Hub")
    ap.add_argument("--resume", action="store_true", help="continue from the last checkpoint")
    args = ap.parse_args()

    import torch
    from huggingface_hub import HfApi, snapshot_download
    from transformers import (EarlyStoppingCallback, Seq2SeqTrainer, Seq2SeqTrainingArguments,
                              WhisperForConditionalGeneration, WhisperProcessor, set_seed)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    with open(ROOT / "configs" / "decode.yaml") as f:
        dec = yaml.safe_load(f)
    for key, val in (("max_steps", args.max_steps), ("warmup_steps", args.warmup_steps),
                     ("eval_steps", args.eval_steps), ("learning_rate", args.lr)):
        if val is not None:
            cfg[key] = val
    set_seed(cfg["seed"])
    if not torch.cuda.is_available():
        raise SystemExit("No GPU. Switch the Colab runtime to L4 or A100.")
    bf16 = torch.cuda.is_bf16_supported()

    train_rows, eval_rows = pick_rows(args)
    print(f"{args.run}: {len(train_rows)} train / {len(eval_rows)} eval utterances | config {cfg}")
    dsd = data.load_edacc()

    processor = WhisperProcessor.from_pretrained(cfg["base_model"])
    processor.tokenizer.set_prefix_tokens(language="en", task="transcribe", predict_timestamps=False)
    model = WhisperForConditionalGeneration.from_pretrained(cfg["base_model"])
    model.generation_config.language = dec["language"]
    model.generation_config.task = dec["task"]
    model.generation_config.forced_decoder_ids = None
    model.generation_config.num_beams = dec["num_beams"]
    model.generation_config.max_new_tokens = dec["max_new_tokens"]
    model.config.use_cache = not cfg["gradient_checkpointing"]

    out_dir = ROOT / "checkpoints" / args.run
    push = not args.no_push
    hub_id = f"{HfApi().whoami()['name']}/speech-ft-{args.run}" if push else None

    def compute_metrics(pred):
        preds = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
        return {"wer": train_data.wer_from_ids(preds, pred.label_ids, processor.tokenizer)}

    targs = Seq2SeqTrainingArguments(
        output_dir=str(out_dir), seed=cfg["seed"],
        learning_rate=cfg["learning_rate"], lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_steps=cfg["warmup_steps"], weight_decay=cfg["weight_decay"], max_steps=cfg["max_steps"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        gradient_checkpointing=cfg["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=bf16, fp16=not bf16,
        eval_strategy="steps", eval_steps=cfg["eval_steps"], save_strategy="steps", save_steps=cfg["eval_steps"],
        save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="wer", greater_is_better=False,
        predict_with_generate=True, per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        logging_steps=25, report_to="none", dataloader_num_workers=cfg["dataloader_num_workers"],
        remove_unused_columns=False, label_names=["labels"],
        push_to_hub=push, hub_model_id=hub_id, hub_strategy="checkpoint", hub_private_repo=True,
    )
    trainer = Seq2SeqTrainer(
        model=model, args=targs,
        train_dataset=train_data.SplitDataset(train_rows, dsd, processor),
        eval_dataset=train_data.SplitDataset(eval_rows, dsd, processor),
        data_collator=train_data.Collator(processor, model.config.decoder_start_token_id),
        processing_class=processor, compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg["early_stopping_patience"])],
    )

    resume = None
    if args.resume:
        if push:  # a new Colab session has no local files: fetch the last checkpoint from the Hub
            snapshot_download(hub_id, local_dir=str(out_dir), allow_patterns=["last-checkpoint/*"])
            resume = str(out_dir / "last-checkpoint")
        else:
            resume = True
    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume)
    minutes = (time.time() - t0) / 60

    best_dir = out_dir / "best"  # load_best_model_at_end: the trainer now holds the best checkpoint
    trainer.save_model(str(best_dir))
    processor.save_pretrained(str(best_dir))
    if push:
        trainer.push_to_hub(commit_message=f"{args.run}: best checkpoint by val WER")
        HfApi().super_squash_history(repo_id=hub_id)  # drop old checkpoint versions to save Hub storage

    st = trainer.state
    log = {"run": args.run, "config": cfg, "commit": git_commit(), "gpu": torch.cuda.get_device_name(0),
           "bf16": bf16, "train_utterances": len(train_rows), "eval_utterances": len(eval_rows),
           "overfit": args.overfit, "minutes": round(minutes, 1), "global_step": st.global_step,
           "best_val_wer_during_training": st.best_metric, "best_checkpoint": st.best_model_checkpoint,
           "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1), "hub_repo": hub_id,
           "log_history": st.log_history}
    run_dir = ROOT / "results" / "runs" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "train_log.json").write_text(json.dumps(log, indent=2))

    print("\n=== Training summary")
    for h in st.log_history:
        if "loss" in h:
            print(f"  step {h['step']:5} | train loss {h['loss']:.3f}")
        if "eval_wer" in h:
            print(f"  step {h['step']:5} | eval loss {h['eval_loss']:.3f} | eval WER {h['eval_wer']:.2%}")
    best = f"{st.best_metric:.2%}" if st.best_metric is not None else "n/a (no evaluation ran)"
    print(f"best eval WER {best} at {st.best_model_checkpoint} | {st.global_step} steps in "
          f"{minutes:.1f} min ({minutes * 60 / max(st.global_step, 1):.2f} s/step) | peak GPU {log['peak_gpu_gb']} GB")
    print(f"best model saved to {best_dir}" + (f" and pushed to {hub_id} (private)" if push else ""))


if __name__ == "__main__":
    main()
