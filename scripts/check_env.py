"""Phase 0 environment check. Run once on the GPU machine and paste the FULL output back.

    python scripts/check_env.py          # versions, GPU, disk, HF login, whisper-tiny smoke test
    python scripts/check_env.py --ct2    # also test the Phase-4 path (CTranslate2 + faster-whisper on CPU)
"""
import argparse
import importlib.metadata as md
import io
import shutil
import subprocess
import sys
import time
import traceback

EXPECTED = {
    "transformers": "5.20.2", "datasets": "5.1.0", "accelerate": "1.15.0", "peft": "0.21.2",
    "huggingface_hub": "2.2.0", "jiwer": "4.0.0", "soundfile": "0.14.0", "soxr": "1.1.0",
    "ctranslate2": "4.8.3", "faster-whisper": "1.2.1",
}
problems = []


def section(title):
    print(f"\n=== {title}", flush=True)


def check_versions():
    section("Library versions")
    print(f"python           {sys.version.split()[0]}")
    for pkg, want in EXPECTED.items():
        try:
            have = md.version(pkg)
        except md.PackageNotFoundError:
            have = "MISSING"
        if have != want:
            problems.append(f"{pkg} is {have}, expected {want}")
        print(f"{pkg:16} {have}" + ("" if have == want else f"   <-- expected {want}"))


def check_torch():
    section("PyTorch and GPU")
    import torch
    from packaging.version import Version

    print("torch", torch.__version__, "| CUDA build:", torch.version.cuda)
    if Version(torch.__version__.split("+")[0]) < Version("2.6"):
        problems.append("torch < 2.6, but transformers 5.x needs torch >= 2.6")
    if not torch.cuda.is_available():
        problems.append("No CUDA GPU visible. Fine-tuning whisper-small on CPU is not practical.")
        return "cpu"
    p = torch.cuda.get_device_properties(0)
    mem = p.total_memory / 1024**3
    print(f"GPU: {p.name} | {mem:.1f} GB | compute capability {p.major}.{p.minor} | GPUs: {torch.cuda.device_count()}")
    print("bf16 supported:", torch.cuda.is_bf16_supported())
    if (p.major, p.minor) < (7, 0):
        problems.append(f"{p.name} has no tensor cores, so fp16 training will be slow. On Kaggle choose T4, not P100.")
    if mem < 15:
        problems.append(f"Only {mem:.1f} GB GPU memory. Full fine-tune of whisper-small may not fit; plan on LoRA.")
    return "cuda"


def check_system():
    section("Disk and Hugging Face login")
    free = shutil.disk_usage(".").free / 1024**3
    print(f"free disk in current folder: {free:.0f} GB")
    if free < 30:
        problems.append(f"Only {free:.0f} GB free disk. EdAcc parquet is several GB before any cache.")
    try:
        from huggingface_hub import whoami
        print("HF user:", whoami()["name"])
    except Exception as e:
        problems.append(f"Not logged in to Hugging Face ({type(e).__name__}). Needed to push checkpoints.")


def load_hf(repo, *args, **kw):
    """datasets 5.x refuses repos that still use a loading script; retry on the auto-converted parquet branch."""
    from datasets import load_dataset

    try:
        return load_dataset(repo, *args, **kw)
    except RuntimeError as e:
        if "scripts are no longer supported" not in str(e):
            raise
        print(f"  {repo} uses a loading script -> retrying with revision='refs/convert/parquet'")
        return load_dataset(repo, *args, revision="refs/convert/parquet", **kw)


def read_audio(audio_field):
    """Decode ourselves (no torchcodec): mono float32 at 16 kHz."""
    import soundfile as sf
    import soxr

    src = audio_field["bytes"] if audio_field.get("bytes") else audio_field["path"]
    wav, sr = sf.read(io.BytesIO(src) if isinstance(src, bytes) else src, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        wav = soxr.resample(wav, sr, 16000)
    return wav, sr


def smoke_test(device):
    section("Smoke test: whisper-tiny on one LibriSpeech clip")
    import jiwer
    import torch
    from datasets import Audio
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    ds = load_hf("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    print("columns:", ds.column_names)
    ds = ds.cast_column("audio", Audio(decode=False))
    ex = ds[0]
    wav, sr = read_audio(ex["audio"])
    print(f"audio: {len(wav) / 16000:.1f} s, original sample rate {sr}")

    proc = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny").to(device).eval()
    feats = proc.feature_extractor(wav, sampling_rate=16000, return_tensors="pt").input_features.to(device)
    t = time.time()
    with torch.no_grad():
        ids = model.generate(feats, language="en", task="transcribe", num_beams=1, max_new_tokens=128)
    hyp = proc.batch_decode(ids, skip_special_tokens=True)[0]
    norm = proc.tokenizer.normalize  # Whisper's English text normalizer
    out = jiwer.process_words(norm(ex["text"]), norm(hyp))
    print(f"REF : {ex['text']}\nHYP : {hyp}\ngenerate time: {time.time() - t:.2f} s")
    print(f"WER {out.wer:.3f} | S={out.substitutions} D={out.deletions} I={out.insertions}")
    if out.wer > 0.3:
        problems.append("Smoke-test WER > 30% on a clean clip: something in the pipeline is wrong.")
    return wav, norm(ex["text"]), norm


def ct2_test(wav, ref_norm, norm):
    section("Phase-4 path: convert whisper-tiny to CTranslate2, run faster-whisper on CPU")
    import jiwer
    from faster_whisper import WhisperModel

    out_dir = "ct2-whisper-tiny"
    cmd = ["ct2-transformers-converter", "--model", "openai/whisper-tiny", "--output_dir", out_dir,
           "--quantization", "int8", "--copy_files", "tokenizer.json", "preprocessor_config.json", "--force"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-3000:])
        problems.append("CTranslate2 conversion failed with this transformers version (error printed above).")
        return
    model = WhisperModel(out_dir, device="cpu", compute_type="int8")
    t = time.time()
    segments, _ = model.transcribe(wav, language="en", task="transcribe", beam_size=1)
    hyp = " ".join(s.text for s in segments)  # segments is lazy; decoding happens here
    secs = time.time() - t
    print(f"HYP : {hyp}\nWER {jiwer.wer(ref_norm, norm(hyp)):.3f} | RTF on this CPU: {secs / (len(wav) / 16000):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct2", action="store_true", help="also test the CTranslate2 / faster-whisper path")
    args = ap.parse_args()

    check_versions()
    try:
        device = check_torch()
    except Exception as e:
        problems.append(f"PyTorch check crashed: {type(e).__name__}: {e}")
        device = "cpu"
    check_system()

    result = None
    try:
        result = smoke_test(device)
    except Exception as e:
        traceback.print_exc()
        problems.append(f"Smoke test crashed: {type(e).__name__}: {e}")
    if args.ct2 and result is not None:
        try:
            ct2_test(*result)
        except Exception as e:
            traceback.print_exc()
            problems.append(f"CTranslate2 test crashed: {type(e).__name__}: {e}")

    section("Summary")
    print("ALL CHECKS PASSED" if not problems else "\n".join(f"- {p}" for p in problems))


if __name__ == "__main__":
    main()
