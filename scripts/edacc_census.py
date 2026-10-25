"""Phase 1, step 2: segment census, per-speaker table and accent-group counts. Still no models, no WER.

Run on a Colab CPU runtime:
    python scripts/edacc_census.py | tee results/edacc_census.txt
Writes splits/speakers.csv (speaker metadata and hours only; no audio, no transcripts).
"""
import collections
import csv
import io
import re

import soundfile as sf
import yaml

IGNORE = "IGNORE_TIME_SEGMENT_IN_SCORING"
TAG_RE = re.compile(r"<[^>]+>|\[[^\]]+\]|\b[A-Z]+(?:_[A-Z]+)+\b")  # <TAG>, [TAG], UNDER_SCORE markers
MAX_SEC = 30.0


def collect_rows(dsd, norm):
    from datasets import Audio

    rows = []
    for split, ds in dsd.items():
        ds = ds.cast_column("audio", Audio(decode=False))
        for i, r in enumerate(ds):
            a = r["audio"]
            info = sf.info(io.BytesIO(a["bytes"]) if a.get("bytes") else a["path"])
            text = r["text"]
            rows.append(dict(
                split=split, speaker=r["speaker"], conv=r["speaker"].rsplit("-", 1)[0],
                accent=r["accent"], l1=r["l1"], gender=r["gender"], dur=info.frames / info.samplerate,
                text=text, ignore=IGNORE in text, ignore_only=text.strip() == IGNORE,
                tags=TAG_RE.findall(text), empty=not norm(text.replace(IGNORE, " ")).strip(),
            ))
            if (i + 1) % 4000 == 0:
                print(f"  ...{split}: {i + 1} rows", flush=True)
    for r in rows:
        r["usable"] = not r["ignore"] and r["dur"] <= MAX_SEC and not r["empty"]
    return rows


def hours(rs):
    return sum(r["dur"] for r in rs) / 3600


def line(label, rs, total):
    print(f"  {label:44} rows {len(rs):6} | {hours(rs):6.2f} h | {100 * hours(rs) / total:5.1f}% of hours")


def report(rows, l1_to_group, csv_path="splits/speakers.csv"):
    total = hours(rows)
    long_ = [r for r in rows if r["dur"] > MAX_SEC]
    usable = [r for r in rows if r["usable"]]
    print("\n=== A. Segment census (both official splits together)")
    line("all segments", rows, total)
    line(f"text contains {IGNORE}", [r for r in rows if r["ignore"]], total)
    line("  ...of which the marker is the whole text", [r for r in rows if r["ignore_only"]], total)
    line(f"longer than {MAX_SEC:.0f} s", long_, total)
    line("  ...of which contain the IGNORE marker", [r for r in long_ if r["ignore"]], total)
    line("empty after normalization (tags only)", [r for r in rows if r["empty"] and not r["ignore"]], total)
    line("USABLE (no IGNORE, <= 30 s, not empty)", usable, total)
    line("  ...usable but shorter than 1 s", [r for r in usable if r["dur"] < 1], total)

    print("\n=== B. Tags and markers in transcripts (rows containing each)")
    tags = collections.Counter(t for r in rows for t in set(r["tags"]))
    print(" ", tags.most_common(30))

    print("\n=== C. Ten longest segments")
    for r in sorted(rows, key=lambda r: -r["dur"])[:10]:
        print(f"  {r['dur']:6.1f} s | {r['speaker']} | ignore={r['ignore']} | {r['text'][:70]}")

    # per-speaker table
    spk = collections.defaultdict(list)
    for r in rows:
        spk[r["speaker"]].append(r)
    table = []
    for s, rs in spk.items():
        meta = {(r["accent"], r["l1"], r["gender"]) for r in rs}
        if len(meta) > 1:
            print(f"WARNING: {s} has inconsistent metadata: {meta}")
        accent, l1, gender = rs[0]["accent"], rs[0]["l1"], rs[0]["gender"]
        table.append(dict(
            speaker=s, conversation=rs[0]["conv"], official_split=rs[0]["split"], accent=accent, l1=l1,
            group=l1_to_group.get(l1, "UNMAPPED"), l1_english="English" in l1, gender=gender,
            usable_hours=round(hours([r for r in rs if r["usable"]]), 3),
        ))
    table.sort(key=lambda t: (t["group"], t["speaker"]))

    print("\n=== D. Speakers (sorted by group)")
    print(f"  {'speaker':13} {'group':21} {'l1':24} {'accent':26} usable_h")
    for t in table:
        print(f"  {t['speaker']:13} {t['group']:21} {t['l1'][:24]:24} {t['accent'][:26]:26} {t['usable_hours']:.2f}")
    unmapped = sorted({t["l1"] for t in table if t["group"] == "UNMAPPED"})
    print("UNMAPPED l1 values:", unmapped or "none")

    print("\n=== E. Accent groups (usable data)")
    for key in ("group", "l1_english"):
        agg = collections.defaultdict(lambda: [0, set(), 0.0])
        for t in table:
            a = agg[t[key]]
            a[0] += 1
            a[1].add(t["conversation"])
            a[2] += t["usable_hours"]
        print(f"  by {key}:")
        for g, (n, convs, h) in sorted(agg.items(), key=lambda kv: -kv[1][2]):
            print(f"    {str(g):22} speakers {n:3} | conversations {len(convs):3} | usable {h:5.2f} h")

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    print(f"\nwrote {csv_path} ({len(table)} speakers, {len({t['conversation'] for t in table})} conversations)")


def main():
    from datasets import load_dataset
    from transformers import WhisperTokenizer

    with open("configs/accent_groups.yaml") as f:
        groups = yaml.safe_load(f)["groups"]
    l1_to_group = {l1: g for g, l1s in groups.items() for l1 in l1s}
    norm = WhisperTokenizer.from_pretrained("openai/whisper-small").normalize  # Whisper's English normalizer
    rows = collect_rows(load_dataset("edinburghcstr/edacc"), norm)
    report(rows, l1_to_group)


if __name__ == "__main__":
    main()
