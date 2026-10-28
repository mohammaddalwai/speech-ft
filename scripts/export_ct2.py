"""Convert the fine-tuned model to CTranslate2 for fast CPU serving (Phase 4). Run on Colab; a CPU runtime is fine.

    python scripts/export_ct2.py --model mddalwai/speech-ft-E3 --out ct2/E3
    -> ct2/E3-int8 and ct2/E3-float32 (model.bin, config.json, vocabulary.json, tokenizer.json,
       preprocessor_config.json)

Fine-tuning does not change the audio front end, so preprocessor_config.json comes from openai/whisper-small.
"""
import argparse
import pathlib


def convert(model, out, quantization, feature_extractor_source="openai/whisper-small"):
    import ctranslate2
    from transformers import WhisperFeatureExtractor

    out = pathlib.Path(f"{out}-{quantization}")
    ctranslate2.converters.TransformersConverter(model, copy_files=["tokenizer.json"]).convert(
        str(out), quantization=quantization, force=True)
    WhisperFeatureExtractor.from_pretrained(feature_extractor_source).save_pretrained(str(out))
    size_mb = sum(p.stat().st_size for p in out.iterdir()) / 1e6
    print(f"{out}: {size_mb:.0f} MB | files: {sorted(p.name for p in out.iterdir())}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mddalwai/speech-ft-E3", help="HF repo id or local folder (transformers format)")
    ap.add_argument("--out", default="ct2/E3")
    ap.add_argument("--quantization", nargs="+", default=["int8", "float32"])
    args = ap.parse_args()
    for q in args.quantization:
        convert(args.model, args.out, q)


if __name__ == "__main__":
    main()
