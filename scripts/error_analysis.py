"""Phase 3 error analysis. No GPU.

  python scripts/error_analysis.py make --run E3_test --base E0_small_test --extra E0c_turbo_test
      -> results/error_analysis/worst50_E3_test.csv   the 50 utterances with the most word errors
         results/error_analysis/overview_E3_test.json  whole-test breakdowns (clip length, tags, examples)
  Open the CSV in a spreadsheet. For each row read ref_norm / new_norm / base_norm and write ONE category in
  `label`: the main cause of that utterance's errors. `suggested` is a rough automatic guess; override it.
  python scripts/error_analysis.py summarize --run E3_test
      -> counts and share of errors per label (empty labels fall back to `suggested` and are counted as unreviewed)

Categories:
  names_rare_words       proper nouns, titles, rare words ("sherman" -> "shaman")
  accent_or_fast_speech  common words misheard: pronunciation, speed, mumbling
  disfluency_mismatch    repeats, false starts or fillers written differently from the reference
  outside_reference      the model wrote speech the reference does not contain (segment edges, other speaker)
  masked_word            the reference has <DTMF>: a word bleeped for privacy, and the model guesses something
  hallucination_loop     looping or invented text
  reference_error        the reference itself looks wrong
"""
import argparse
import collections
import csv
import json
import pathlib
import re

import jiwer

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "error_analysis"
CATEGORIES = ["names_rare_words", "accent_or_fast_speech", "disfluency_mismatch", "outside_reference",
              "masked_word", "hallucination_loop", "reference_error"]
BUCKETS = [(0, 1), (1, 3), (3, 10), (10, 30.01)]


def load(run):
    with open(ROOT / "results" / "runs" / run / "utterances.csv") as f:
        return {u["utt_id"]: u for u in csv.DictReader(f)}


def errors(u):
    return int(u["sub"]) + int(u["del"]) + int(u["ins"])


def vocabulary():
    """Word counts over every EdAcc reference we have scored (val + test), to spot rare words."""
    counts = collections.Counter()
    for run in ("E0_small_val", "E0_small_test"):
        for u in load(run).values():
            counts.update(u["ref_norm"].split())
    return counts


def looping(words):
    for n in (1, 2, 3):
        for i in range(len(words) - 4 * n + 1):
            if all(words[i + k * n:i + (k + 1) * n] == words[i:i + n] for k in range(1, 4)):
                return True
    return False


def suggest(u, vocab):
    ref, hyp = u["ref_norm"].split(), u["hyp_norm"].split()
    if u.get("capped") == "True" or (looping(hyp) and not looping(ref)):
        return "hallucination_loop"
    if int(u["ins"]) >= 0.3 * len(ref) or (re.search(r"<(NO-SPEECH|OVERLAP)>", u["ref"]) and int(u["ins"]) >= 3):
        return "outside_reference"
    rare, dup, total = 0, 0, max(errors(u), 1)
    if hyp:
        for a in jiwer.process_words(u["ref_norm"], u["hyp_norm"]).alignments[0]:
            if a.type == "equal":
                continue
            rw, hw = ref[a.ref_start_idx:a.ref_end_idx], hyp[a.hyp_start_idx:a.hyp_end_idx]
            rare += sum(vocab[w] <= 2 for w in rw if a.type == "substitute")
            for seq, words, start in ((rw, ref, a.ref_start_idx), (hw, hyp, a.hyp_start_idx)):
                for j, w in enumerate(seq):
                    k = start + j
                    dup += (k > 0 and words[k - 1] == w) or (k + 1 < len(words) and words[k + 1] == w)
    if rare >= 2:
        return "names_rare_words"
    if "<DTMF>" in u["ref"]:
        return "masked_word"
    if dup >= 0.5 * total:
        return "disfluency_mismatch"
    return "accent_or_fast_speech"


def pooled(utts):
    n = sum(int(u["n_ref"]) for u in utts)
    return sum(errors(u) for u in utts) / n if n else float("nan"), n


def make(args):
    target = OUT / f"worst{args.n}_{args.run}.csv"
    if target.exists() and not args.force:
        with open(target) as f:
            if any(r["label"].strip() for r in csv.DictReader(f)):
                raise SystemExit(f"{target} already has your labels. Refusing to overwrite (use --force to discard them).")
    new, base = load(args.run), load(args.base)
    extra = {r: load(r) for r in args.extra}
    scored = [u for u in new.values() if int(u["n_ref"]) > 0]
    vocab = vocabulary()
    worst = sorted(scored, key=lambda u: (errors(u), errors(u) / int(u["n_ref"])), reverse=True)[:args.n]
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [{"rank": i, "utt_id": u["utt_id"], "group": u["group"], "speaker": u["speaker"],
             "duration": u["duration"], "ref_words": u["n_ref"], "errors": errors(u),
             "base_errors": errors(base[u["utt_id"]]), "suggested": suggest(u, vocab), "label": "", "note": "",
             "ref_norm": u["ref_norm"], "new_norm": u["hyp_norm"], "base_norm": base[u["utt_id"]]["hyp_norm"],
             "ref_raw": u["ref"]} for i, u in enumerate(worst, 1)]
    with open(OUT / f"worst{args.n}_{args.run}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    total = sum(errors(u) for u in scored)
    share = sum(r["errors"] for r in rows) / total
    print(f"{args.run}: the {args.n} worst utterances hold {share:.1%} of all {total} word errors")
    print("suggested labels:", dict(collections.Counter(r["suggested"] for r in rows).most_common()))

    runs = {args.base: base, **extra, args.run: new}
    by_len = {}
    print(f"\nWER by clip length (utterances with a scorable reference):")
    print(f"  {'clip length':12}" + "".join(f"{r:>18}" for r in runs) + "   utterances")
    for lo, hi in BUCKETS:
        ids = [u["utt_id"] for u in scored if lo <= float(u["duration"]) < hi]
        vals = {r: pooled([d[k] for k in ids])[0] for r, d in runs.items()}
        by_len[f"{lo}-{hi:g}s"] = {"utterances": len(ids), **vals}
        print(f"  {f'{lo}-{hi:g} s':12}" + "".join(f"{v:18.2%}" for v in vals.values()) + f"   {len(ids):6}")

    tags = {}
    print(f"\nutterances whose reference contains a tag ({args.run}):")
    for tag in ("<DTMF>", "<OVERLAP>", "<NO-SPEECH>", "<LAUGH>"):
        with_tag = [u for u in scored if tag in u["ref"]]
        wer_t, _ = pooled(with_tag)
        err_share = sum(errors(u) for u in with_tag) / total
        tags[tag] = {"utterances": len(with_tag), "wer": wer_t, "share_of_errors": err_share}
        print(f"  {tag:12} {len(with_tag):5} utterances | WER {wer_t:6.2%} | {err_share:5.1%} of all errors")

    delta = sorted(scored, key=lambda u: errors(base[u["utt_id"]]) - errors(u), reverse=True)
    examples = [("improved", u) for u in delta[:3]] + [("got worse", u) for u in delta[-2:][::-1]]
    print("\nbefore/after examples (normalized text):")
    ex_out = []
    for kind, u in examples:
        b = base[u["utt_id"]]
        print(f"\n  [{kind}] {u['utt_id']} ({u['group']}) errors {errors(b)} -> {errors(u)}")
        print(f"    REF : {u['ref_norm']}\n    BASE: {b['hyp_norm']}\n    NEW : {u['hyp_norm']}")
        ex_out.append({"kind": kind, "utt_id": u["utt_id"], "group": u["group"], "ref": u["ref_norm"],
                       "base": b["hyp_norm"], "new": u["hyp_norm"], "errors": [errors(b), errors(u)]})
    overview = {"run": args.run, "base": args.base, "worst_n": args.n, "worst_share_of_errors": share,
                "wer_by_clip_length": by_len, "tags": tags, "examples": ex_out}
    (OUT / f"overview_{args.run}.json").write_text(json.dumps(overview, indent=2))


def summarize(args):
    with open(OUT / f"worst{args.n}_{args.run}.csv") as f:
        rows = list(csv.DictReader(f))
    bad = sorted({r["label"] for r in rows if r["label"].strip() and r["label"].strip() not in CATEGORIES})
    if bad:
        raise SystemExit(f"unknown labels {bad}; use one of {CATEGORIES}")
    unreviewed = sum(not r["label"].strip() for r in rows)
    counts, errs = collections.Counter(), collections.Counter()
    for r in rows:
        lab = r["label"].strip() or r["suggested"]
        counts[lab] += 1
        errs[lab] += int(r["errors"])
    total = sum(errs.values())
    print(f"{args.run}: {len(rows)} worst utterances, {total} word errors, {unreviewed} still unreviewed")
    for lab, n in counts.most_common():
        print(f"  {lab:22} {n:3} utterances | {errs[lab]:4} errors ({errs[lab] / total:5.1%})")
    out = {"run": args.run, "reviewed": len(rows) - unreviewed, "unreviewed": unreviewed,
           "labels": {lab: {"utterances": counts[lab], "errors": errs[lab]} for lab in counts}}
    (OUT / f"worst{args.n}_summary_{args.run}.json").write_text(json.dumps(out, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["make", "summarize"])
    ap.add_argument("--run", required=True)
    ap.add_argument("--base", default="E0_small_test")
    ap.add_argument("--extra", nargs="*", default=[], help="more runs for the clip-length table")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--force", action="store_true", help="make: overwrite a CSV that already has labels")
    args = ap.parse_args()
    make(args) if args.mode == "make" else summarize(args)


if __name__ == "__main__":
    main()
