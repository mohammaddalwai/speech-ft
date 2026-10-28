"""Publish the final model (Phase 4). Run on Colab after scripts/ct2_check.py.

    python scripts/publish_model.py --ct2 ct2/E3-int8 --compute-type int8            # build + check, no upload
    python scripts/publish_model.py --ct2 ct2/E3-int8 --compute-type int8 --upload   # then publish

Builds two folders under release/ and, with --upload, publishes them as PUBLIC repos under your account:
  <user>/whisper-small-edacc       transformers weights + processor files + model card (docs/model_card.md)
  <user>/whisper-small-edacc-ct2   CTranslate2 weights for faster-whisper + short card (docs/model_card_ct2.md)
Only inference files are copied from the private training repo: no optimizer state, no checkpoints, no audio.
Refuses to publish a CTranslate2 folder that was not checked with the current serving code (src/asr.py), or that
failed the conversion rule.
"""
import argparse
import json
import pathlib
import re
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import asr  # noqa: E402

SOURCE = "mddalwai/speech-ft-E3"
INFERENCE_FILES = ["config.json", "generation_config.json", "model.safetensors", "tokenizer.json",
                   "tokenizer_config.json", "processor_config.json", "preprocessor_config.json",
                   "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt", "normalizer.json"]
PLACEHOLDER = re.compile(r"\{[A-Z0-9_]+\}")


def fill(template, values):
    out = template
    for k, v in values.items():
        out = out.replace("{" + k + "}", v)
    left = PLACEHOLDER.findall(out)
    if left:
        raise ValueError(f"unfilled placeholders in model card: {sorted(set(left))}")
    return out


def card_values(user, compute_type, check):
    return {"USER": user, "CT2_TYPE": compute_type, "CT2_WER": f"{check['wer_ct2']:.2%}",
            "OFF_WER": f"{check['wer_official_same_clips']:.2%}", "CT2_DIFF": f"{check['difference_points']:+.2f}",
            "SAME": str(check["identical_outputs"]), "N": str(check["clips"])}


def load_check(ct2, compute_type, runs_dir=ROOT / "results" / "runs"):
    path = pathlib.Path(runs_dir) / f"ct2_check_{compute_type}_{asr.SERVING_VERSION}" / "summary.json"
    if not path.exists():
        raise SystemExit(f"no conversion check found at {path}; run scripts/ct2_check.py first.")
    check = json.loads(path.read_text())
    if check.get("serving_version") != asr.SERVING_VERSION:
        raise SystemExit(f"{path} was made with serving {check.get('serving_version')}, not {asr.SERVING_VERSION}.")
    if pathlib.Path(check["ct2"]).resolve() != pathlib.Path(ct2).resolve():
        raise SystemExit(f"the check was run on {check['ct2']}, not on {ct2}; not publishing an unchecked folder.")
    if not check["within_rule"]:
        raise SystemExit(f"{compute_type} failed the conversion rule; not publishing it.")
    return check


def build(ct2, values, source, out_dir):
    from huggingface_hub import snapshot_download
    from transformers import WhisperFeatureExtractor

    hf_dir, ct2_dir = out_dir / "hf", out_dir / "ct2"
    shutil.rmtree(out_dir, ignore_errors=True)
    snapshot_download(source, allow_patterns=INFERENCE_FILES, local_dir=str(hf_dir))
    shutil.rmtree(hf_dir / ".cache", ignore_errors=True)
    if not (hf_dir / "preprocessor_config.json").exists():  # the audio front end is unchanged by fine-tuning
        WhisperFeatureExtractor.from_pretrained("openai/whisper-small").save_pretrained(str(hf_dir))
    (hf_dir / "README.md").write_text(fill((ROOT / "docs" / "model_card.md").read_text(), values))

    shutil.copytree(ct2, ct2_dir)
    (ct2_dir / "README.md").write_text(fill((ROOT / "docs" / "model_card_ct2.md").read_text(), values))
    return hf_dir, ct2_dir


def describe(folder):
    files = sorted(p.name for p in folder.iterdir())
    size_mb = sum(p.stat().st_size for p in folder.iterdir()) / 1e6
    return f"{size_mb:.0f} MB | {files}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct2", required=True)
    ap.add_argument("--compute-type", required=True, choices=["int8", "float32"])
    ap.add_argument("--source", default=SOURCE)
    ap.add_argument("--upload", action="store_true", help="publish; without it, only build and check release/")
    args = ap.parse_args()

    from huggingface_hub import HfApi
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    check = load_check(args.ct2, args.compute_type)
    api = HfApi()
    user = api.whoami()["name"]
    values = card_values(user, args.compute_type, check)
    hf_dir, ct2_dir = build(args.ct2, values, args.source, ROOT / "release")

    WhisperProcessor.from_pretrained(str(hf_dir))  # fails here, not after publishing, if a file is missing
    WhisperForConditionalGeneration.from_pretrained(str(hf_dir))
    print(f"release/hf  loads with transformers | {describe(hf_dir)}")
    print(f"release/ct2 | {describe(ct2_dir)}")

    repos = ((f"{user}/whisper-small-edacc", hf_dir), (f"{user}/whisper-small-edacc-ct2", ct2_dir))
    if not args.upload:
        print("\nNot uploaded. Read release/hf/README.md and release/ct2/README.md, then run again with --upload "
              f"to publish {repos[0][0]} and {repos[1][0]} as PUBLIC repos.")
        return
    for repo, folder in repos:
        api.create_repo(repo, private=False, exist_ok=True)
        api.update_repo_settings(repo, private=False)
        api.upload_folder(repo_id=repo, folder_path=str(folder), commit_message="Release: E3 seed 42")
        print(f"published https://huggingface.co/{repo}")


if __name__ == "__main__":
    main()
