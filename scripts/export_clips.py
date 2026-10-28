"""Export the audio of the worst utterances as WAV files, so you can listen while labelling. Run on Colab (CPU).

    python scripts/export_clips.py
    -> /content/worst50_audio/<rank>_<utt_id>.wav  (16 kHz mono)

Listen locally only: EdAcc audio must not be committed or published (it is in .gitignore).
"""
import csv
import pathlib
import sys

import soundfile as sf

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import data  # noqa: E402


def main(csv_path=ROOT / "results" / "error_analysis" / "worst50_E3_test.csv", out=pathlib.Path("/content/worst50_audio")):
    with open(csv_path, encoding="utf-8") as f:
        wanted = list(csv.DictReader(f))
    rows = {r["utt_id"]: r for s in ("val", "test") for r in data.split_rows(s)}
    dsd = data.load_edacc()
    out.mkdir(parents=True, exist_ok=True)
    for w in wanted:
        audio, _ = data.fetch(dsd, rows[w["utt_id"]])
        sf.write(out / f"{int(w['rank']):02d}_{w['utt_id']}.wav", audio, data.SR)
    print(f"wrote {len(wanted)} clips to {out}")


if __name__ == "__main__":
    main()
