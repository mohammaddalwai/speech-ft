"""Phase 1, step 1: look at EdAcc before writing any loader. No audio decoding, no models.

Run on a Colab CPU runtime:
    python scripts/inspect_edacc.py | tee results/edacc_inspect.txt
"""
import collections
import io
import re
import statistics as st

import soundfile as sf
from datasets import Audio, DatasetDict, load_dataset

REPO = "edinburghcstr/edacc"
TAG_RE = re.compile(r"<[^>]*>|\[[^\]]*\]|\([^)]*\)|\b[A-Z_]{4,}\b")  # <tags>, [tags], (tags), ALL_CAPS markers


def load_edacc():
    """datasets 5.x refuses repos that still use a loading script; fall back to the auto-converted parquet branch."""
    try:
        return load_dataset(REPO), "main"
    except RuntimeError as e:
        if "scripts are no longer supported" not in str(e):
            raise
        return load_dataset(REPO, revision="refs/convert/parquet"), "refs/convert/parquet"


def duration_and_sr(audio):
    src = io.BytesIO(audio["bytes"]) if audio.get("bytes") else audio["path"]
    info = sf.info(src)
    return info.frames / info.samplerate, info.samplerate


def describe_split(name, ds):
    print(f"\n{'=' * 70}\nSPLIT: {name} | rows: {len(ds)}\nfeatures: {ds.features}")
    audio_col = next((c for c, f in ds.features.items() if isinstance(f, Audio)), None)
    if audio_col is None:
        print("No Audio column found. Paste the features line above so the loader can be adapted.")
        return
    ds = ds.cast_column(audio_col, Audio(decode=False))
    text_cols = [c for c, f in ds.features.items() if getattr(f, "dtype", None) == "string"]

    print("\n--- first 3 rows (audio omitted)")
    for i in range(min(3, len(ds))):
        print({c: ds[i][c] for c in text_cols})

    # one pass over the data: durations from audio headers, plus all string columns
    durs, srs, cols = [], collections.Counter(), {c: [] for c in text_cols}
    for i, row in enumerate(ds):
        d, sr = duration_and_sr(row[audio_col])
        durs.append(d)
        srs[sr] += 1
        for c in text_cols:
            cols[c].append(row[c])
        if (i + 1) % 2000 == 0:
            print(f"  ...read {i + 1} rows", flush=True)

    print(f"\n--- audio: total {sum(durs) / 3600:.2f} h | mean {st.mean(durs):.1f} s | median {st.median(durs):.1f} s "
          f"| min {min(durs):.2f} s | max {max(durs):.1f} s")
    print(f"clips > 30 s: {sum(d > 30 for d in durs)} | clips < 1 s: {sum(d < 1 for d in durs)} | sample rates: {dict(srs)}")

    for c in text_cols:
        vals = cols[c]
        uniq = set(vals)
        print(f"\n--- column '{c}': {len(uniq)} unique values")
        if len(uniq) <= 80:
            hours = collections.defaultdict(float)
            for v, d in zip(vals, durs):
                hours[v] += d / 3600
            for v, n in collections.Counter(vals).most_common():
                print(f"  {str(v)[:45]:45} rows {n:6} | {hours[v]:6.2f} h")
        else:
            print("  examples:", [str(v)[:60] for v in list(uniq)[:8]])
            if "speaker" in c.lower():
                per_spk = collections.defaultdict(float)
                for v, d in zip(vals, durs):
                    per_spk[v] += d / 60
                m = sorted(per_spk.values())
                print(f"  minutes per speaker: min {m[0]:.1f} | median {st.median(m):.1f} | max {m[-1]:.1f}")

    # transcript annotations that the normalizer will need rules for
    longest = max(text_cols, key=lambda c: st.mean(len(str(v)) for v in cols[c]))
    texts = [str(t) for t in cols[longest]]
    tags = collections.Counter(m for t in texts for m in TAG_RE.findall(t))
    print(f"\n--- transcript column guess: '{longest}' | empty: {sum(not t.strip() for t in texts)} "
          f"| rows with tags/markers: {sum(bool(TAG_RE.search(t)) for t in texts)}")
    print("most common tags/markers:", tags.most_common(25))


def main():
    dsd, revision = load_edacc()
    print(f"loaded {REPO} from revision: {revision}")
    if not isinstance(dsd, DatasetDict):
        dsd = DatasetDict({"all": dsd})
    for name, ds in dsd.items():
        describe_split(name, ds)


if __name__ == "__main__":
    main()
