"""Chart: test WER by accent group, before (zero-shot small) and after (E3), with zero-shot turbo for reference.

    python scripts/plot_groups.py
    -> results/figures/test_wer_by_group.png

95% intervals (speaker bootstrap) are drawn only where there are at least 5 test speakers. No GPU.
"""
import collections
import csv
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS = {  # run -> (legend label, color, marker). Colors: validated categorical slots 1-3 (CVD-safe, all pairs)
    "E3_test": ("Fine-tuned small (E3)", "#2a78d6", "s"),
    "E0_small_test": ("Zero-shot small", "#eb6834", "o"),
    "E0c_turbo_test": ("Zero-shot large-v3-turbo", "#1baf7a", "D"),
}
NAMES = {"african": "African", "british_irish": "British & Irish", "caribbean": "Caribbean",
         "east_southeast_asian": "East & SE Asian", "european_other": "Other European",
         "european_romance": "Romance European", "north_american": "North American", "other": "Other",
         "south_asian": "South Asian"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def main():
    summ = {r: json.loads((ROOT / "results" / "runs" / r / "summary.json").read_text()) for r in RUNS}
    boot = json.loads((ROOT / "results" / "runs" / "E3_test" / "bootstrap_vs_E0_small_test.json").read_text())
    speakers = collections.defaultdict(set)
    with open(ROOT / "results" / "runs" / "E3_test" / "utterances.csv") as f:
        for u in csv.DictReader(f):
            if int(u["n_ref"]) > 0:
                speakers[u["group"]].add(u["speaker"])

    groups = sorted(speakers, key=lambda g: summ["E0_small_test"]["by_group"]["by_group"][g]["wer"])
    rows = [(f"{NAMES[g]} ({len(speakers[g])})", g) for g in groups] + \
           [(f"All test speakers ({sum(len(s) for s in speakers.values())})", "pooled")]

    def wer(run, g):
        rep = summ[run]["by_group"]
        return 100 * (rep["pooled"]["wer"] if g == "pooled" else rep["by_group"][g]["wer"])

    def interval(g):
        ci = boot["pooled"]["ci95"] if g == "pooled" else (boot["groups"][g]["ci95"] or {})
        return ci.get("new") if ci else None

    fig, ax = plt.subplots(figsize=(8.6, 5.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for y, (_, g) in enumerate(rows):
        before, after = wer("E0_small_test", g), wer("E3_test", g)
        ax.plot([before, after], [y, y], color="#b9b8b3", lw=2, zorder=1, solid_capstyle="round")
        ci = interval(g)
        if ci:  # drawn just under the row, with end caps, so it is not confused with the before/after line
            lo, hi, yc = 100 * ci[0], 100 * ci[1], y - 0.24
            ax.plot([lo, hi], [yc, yc], color=RUNS["E3_test"][1], lw=1.6, zorder=2)
            ax.plot([lo, lo], [yc - 0.07, yc + 0.07], color=RUNS["E3_test"][1], lw=1.6, zorder=2)
            ax.plot([hi, hi], [yc - 0.07, yc + 0.07], color=RUNS["E3_test"][1], lw=1.6, zorder=2)
        for run, (label, color, marker) in RUNS.items():
            ax.scatter(wer(run, g), y, s=58, color=color, marker=marker, edgecolor=SURFACE, linewidth=1.2,
                       zorder=3, label=label if y == 0 else None)
    pooled_y = len(rows) - 1
    for run in RUNS:  # direct labels on the headline row only
        ax.annotate(f"{wer(run, 'pooled'):.1f}%", (wer(run, "pooled"), pooled_y), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=8.5, color=INK)

    ax.axhline(pooled_y - 0.5, color=GRID, lw=1)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=9, color=INK)
    ax.set_ylim(-0.6, len(rows) - 0.3)
    ax.set_xlim(0, max(wer(r, g) for r in RUNS for _, g in rows) * 1.08)
    ax.set_xlabel("Word error rate on the test split (%)  -  lower is better", fontsize=9, color=MUTED)
    ax.tick_params(axis="x", labelsize=8.5, colors=MUTED)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_title("EdAcc test WER by accent group (number of test speakers)", fontsize=11, color=INK, loc="left", pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, frameon=False, fontsize=8.5,
              labelcolor=INK, handletextpad=0.3, columnspacing=1.2, borderaxespad=0.2)
    fig.text(0.01, 0.01, "Blue bracket under a row: 95% speaker-bootstrap interval for E3, shown only for rows with "
             "at least 5 speakers. Groups with 2-4 speakers are anecdotal.", fontsize=7.5, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = ROOT / "results" / "figures" / "test_wer_by_group.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
