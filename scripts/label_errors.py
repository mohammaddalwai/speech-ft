"""Label the worst utterances one at a time in the Terminal. Saves after every answer; you can stop and resume.

    python scripts/label_errors.py                                  # text only
    python scripts/label_errors.py --audio ~/Desktop/worst50_audio  # also lets you press p to play the clip

Only rows with an empty `label` are shown, so running it again continues where you stopped.
"""
import argparse
import csv
import os
import pathlib
import shutil
import subprocess
import sys

import jiwer

ROOT = pathlib.Path(__file__).resolve().parents[1]
CSV = ROOT / "results" / "error_analysis" / "worst50_E3_test.csv"
CATS = [
    ("names_rare_words", "names, titles, rare words are wrong"),
    ("accent_or_fast_speech", "common words misheard"),
    ("disfluency_mismatch", "repeats / false starts / fillers written differently"),
    ("outside_reference", "model wrote speech the reference does not contain"),
    ("masked_word", "a bleeped word (<DTMF>), model guessed something"),
    ("hallucination_loop", "looping or invented text"),
    ("reference_error", "the reference itself looks wrong"),
]
BOLD, RED, GREEN, DIM, END = ("\033[1m", "\033[31m", "\033[32m", "\033[2m", "\033[0m") if sys.stdout.isatty() else ("",) * 5


def diff(ref, hyp):
    """Reference words with the model's mistakes marked: [ref->hyp] substituted, -word- missing, +word+ extra."""
    r, h = ref.split(), hyp.split()
    if not r:
        return hyp
    if not h:
        return " ".join(f"{RED}-{w}-{END}" for w in r)
    out = []
    for a in jiwer.process_words(ref, hyp).alignments[0]:
        rw, hw = r[a.ref_start_idx:a.ref_end_idx], h[a.hyp_start_idx:a.hyp_end_idx]
        if a.type == "equal":
            out += rw
        elif a.type == "substitute":
            out.append(f"{RED}[{' '.join(rw)} -> {' '.join(hw)}]{END}")
        elif a.type == "delete":
            out.append(f"{RED}-{' '.join(rw)}-{END}")
        else:
            out.append(f"{GREEN}+{' '.join(hw)}+{END}")
    return " ".join(out)


def save(rows):
    tmp = CSV.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, CSV)  # never leaves a half-written file


def play(audio_dir, row):
    if not audio_dir:
        print("  (no --audio folder given)")
        return
    path = pathlib.Path(audio_dir).expanduser() / f"{int(row['rank']):02d}_{row['utt_id']}.wav"
    if not path.exists():
        print(f"  (no audio file {path})")
    elif shutil.which("afplay"):
        subprocess.run(["afplay", str(path)])
    else:
        print(f"  audio file: {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", help="folder with the exported clips (scripts/export_clips.py)")
    args = ap.parse_args()
    with open(CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    todo = [r for r in rows if not r["label"].strip()]
    print(f"{len(rows) - len(todo)} of {len(rows)} already labelled; {len(todo)} to go.\n")
    for row in todo:
        print("=" * 100)
        print(f"{BOLD}#{row['rank']}{END} {row['utt_id']} | {row['group']} | {float(row['duration']):.1f} s | "
              f"{row['ref_words']} words | E3 errors {row['errors']} (zero-shot small {row['base_errors']})")
        print(f"{DIM}raw reference (shows tags like <DTMF>):{END} {row['ref_raw']}")
        print(f"\n{BOLD}E3 vs reference:{END}         {diff(row['ref_norm'], row['new_norm'])}")
        print(f"{BOLD}zero-shot vs reference:{END}  {diff(row['ref_norm'], row['base_norm'])}")
        print(f"\nautomatic guess: {row['suggested']}")
        for i, (name, desc) in enumerate(CATS, 1):
            print(f"  {i} {name:22} {desc}")
        while True:
            ans = input("label 1-7 | a = accept guess | p = play | s = skip | q = quit > ").strip().lower()
            if ans == "q":
                print("saved; run the same command to continue.")
                return
            if ans == "s":
                break
            if ans == "p":
                play(args.audio, row)
                continue
            if ans == "a" or (ans.isdigit() and 1 <= int(ans) <= len(CATS)):
                row["label"] = row["suggested"] if ans == "a" else CATS[int(ans) - 1][0]
                row["note"] = input("note (optional, Enter to skip) > ").strip()
                save(rows)
                print(f"  -> {row['label']} (saved)")
                break
            print("  type a number 1-7, a, p, s or q")
    left = sum(not r["label"].strip() for r in rows)
    print(f"\ndone with this pass; {left} rows still unlabelled.")


if __name__ == "__main__":
    main()
