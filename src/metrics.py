"""WER pieces. Every number is pooled over words: (S + D + I) / reference words."""
import collections

import jiwer


def utterance_errors(ref_norm: str, hyp_norm: str) -> dict:
    n = len(ref_norm.split())
    if not hyp_norm.strip():
        return {"n_ref": n, "sub": 0, "del": n, "ins": 0}
    o = jiwer.process_words(ref_norm, hyp_norm)
    return {"n_ref": n, "sub": o.substitutions, "del": o.deletions, "ins": o.insertions}


def pooled(utts) -> dict:
    n = sum(u["n_ref"] for u in utts)
    s, d, i = (sum(u[k] for u in utts) for k in ("sub", "del", "ins"))
    return {"utterances": len(utts), "ref_words": n, "wer": (s + d + i) / n if n else float("nan"),
            "sub_rate": s / n if n else float("nan"), "del_rate": d / n if n else float("nan"),
            "ins_rate": i / n if n else float("nan")}


def report(utts, key="group") -> dict:
    groups = collections.defaultdict(list)
    for u in utts:
        groups[u[key]].append(u)
    by_group = {g: pooled(us) for g, us in sorted(groups.items())}
    macro = sum(v["wer"] for v in by_group.values()) / len(by_group)
    return {"pooled": pooled(utts), "macro_over_groups": macro, "by_" + key: by_group}
