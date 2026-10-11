"""Build the two paper figures from results/, so they cannot drift from the data.

    .venv/bin/python paper/make_figures.py

Writes paper/figures/{natural_rates,confidence_vs_verdict}.{png,pdf}.

FIGURE 1 shows the gap between two lines, not the lines themselves. Acceptance of
correct answers and acceptance of wrong answers are plotted against how much
evidence the judge was given. If a judge is discriminating, the two separate; if it
is merely becoming more (or less) willing to accept, they move together and the gap
stays shut. That gap is balanced accuracy in visual form, and it is the paper's
whole first result, so the figure encodes it directly rather than asking the reader
to difference two bars by eye.

FIGURE 2 scores each condition twice on one axis - by its ACCEPT/REJECT token
(balanced accuracy) and by the confidence number beside it (AUROC) - so the gap
between them, which is what binarising the decision discards, reads as a length.

Palette: categorical slots 1 and 2 of the validated default (blue #2a78d6, orange
#eb6834); worst-pair CVD dE 24.7, normal-vision 33.6, both clear. Print figures, so
light surface only.
"""
from __future__ import annotations

import collections
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import common as C  # noqa: E402

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------- tokens ----
SURFACE = "#fcfcfb"
INK, SECONDARY, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
CORRECT, WRONG = "#2a78d6", "#eb6834"      # categorical slots 1, 2
GREY_BAR = "#d9d8d2"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})

JUDGES = [("A", "M1  Claude Sonnet 4.6", "thinking off"),
          ("B", "M2  GPT-5 mini", "minimal reasoning"),
          ("C", "M3  GPT-5.6 Terra", "medium reasoning")]
LEVELS = [("floor", "answer\nonly"), ("strip", "+ executed\nnotebook"), ("full", "+ notebook\n+ prose")]


def load():
    lab = {r["question_id"]: r for r in C.read_csv(C.RESULTS / "labelled.csv")}
    rows = []
    for r in C.read_csv(C.RESULTS / "judgments.csv"):
        r["label"] = lab[r["question_id"]]["label"]
        if r["label"] in ("correct", "incorrect"):
            rows.append(r)
    return rows


def rates(rows, cond):
    sub = [r for r in rows if r["condition"] == cond]
    out = {}
    for lab in ("correct", "incorrect"):
        s = [r for r in sub if r["label"] == lab]
        out[lab] = sum(1 for r in s if r["verdict"] == "ACCEPT") / len(s)
    return out["correct"], out["incorrect"]


def style(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(length=0, colors=MUTED, labelsize=8)


# ============================================================== figure 1 ====
def figure_natural(rows):
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.6), dpi=220, sharey=True)
    xs = [0, 1, 2]

    for ax, (pfx, name, effort) in zip(axes, JUDGES):
        cor = [rates(rows, f"{pfx}-{lv}")[0] for lv, _ in LEVELS]
        wro = [rates(rows, f"{pfx}-{lv}")[1] for lv, _ in LEVELS]

        ax.fill_between(xs, cor, wro, color=MUTED, alpha=0.13, linewidth=0, zorder=1)
        ax.plot(xs, cor, color=CORRECT, linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)
        ax.plot(xs, wro, color=WRONG, linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=3)

        # the gap IS the discrimination; name it where it is widest
        widest = max(range(3), key=lambda i: abs(cor[i] - wro[i]))
        gap = abs(cor[widest] - wro[widest])
        if gap > 0.15:
            ax.annotate("", xy=(widest + 0.13, cor[widest]), xytext=(widest + 0.13, wro[widest]),
                        arrowprops=dict(arrowstyle="<->", color=SECONDARY, linewidth=1))
            ax.text(widest + 0.2, (cor[widest] + wro[widest]) / 2,
                    f"gap {gap:.2f}", fontsize=7.5, color=SECONDARY, va="center")

        if pfx == "A":                       # the deployed configuration
            sc, sw = rates(rows, "A-self")
            ax.plot([2.34, 2.34], [sc, sw], color=BASELINE, linewidth=1, zorder=2)
            ax.plot([2.34], [sc], marker="D", markersize=6, color=SURFACE,
                    markeredgecolor=CORRECT, markeredgewidth=1.8, zorder=4)
            ax.plot([2.34], [sw], marker="D", markersize=6, color=SURFACE,
                    markeredgecolor=WRONG, markeredgewidth=1.8, zorder=4)
            ax.text(2.34, sw - 0.10, "A-self\nsame evidence,\nown prior turn", fontsize=6.8,
                    color=SECONDARY, ha="center", va="top", linespacing=1.35)

        ax.set_xticks(xs)
        ax.set_xticklabels([lbl for _, lbl in LEVELS], fontsize=7.5, color=SECONDARY,
                           linespacing=1.3)
        ax.set_xlim(-0.35, 2.95)
        ax.set_ylim(-0.04, 1.12)
        ax.set_title(name, fontsize=9.5, loc="left", color=INK, pad=13)
        ax.text(0, 1.015, effort, transform=ax.transAxes, fontsize=7, color=MUTED)
        style(ax)

    axes[0].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axes[0].set_ylabel("share accepted", fontsize=8.5, color=SECONDARY)

    fig.legend(handles=[
        Line2D([], [], color=CORRECT, linewidth=2, marker="o", markersize=7,
               markeredgecolor=SURFACE, markeredgewidth=2, label="correct answers accepted"),
        Line2D([], [], color=WRONG, linewidth=2, marker="o", markersize=7,
               markeredgecolor=SURFACE, markeredgewidth=2, label="wrong answers accepted"),
        Line2D([], [], color=BASELINE, linewidth=1, marker="D", markersize=6,
               markerfacecolor=SURFACE, markeredgecolor=SECONDARY,
               label="A-self (M1 reviewing its own turn)")],
        loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.09))

    fig.suptitle("Does more evidence make the reviewer better, or just more agreeable?",
                 fontsize=11, x=0.008, ha="left", color=INK)
    fig.text(0.008, 0.905,
             "The gap between the two lines is discrimination. It stays shut for M1 and M2 as "
             "evidence is added — both rates move together — and opens only for M3.",
             fontsize=7.4, color=MUTED)
    fig.tight_layout(rect=(0, 0.045, 1, 0.885))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"natural_rates.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT/'natural_rates.png'}")


# ============================================================== figure 2 ====
# The paper's central contrast. Each condition gets two scores on one axis:
# what its ACCEPT/REJECT token achieves (balanced accuracy) and what its
# confidence number achieves (AUROC against ground truth). A dumbbell rather
# than paired bars because the quantity of interest is the GAP between the two,
# and a gap is read far more accurately as a length than as a bar difference.
TOKEN_C, CONF_C = "#eb6834", "#2a78d6"


def _auroc(scores, pos):
    a = [s for s, p in zip(scores, pos) if p]
    b = [s for s, p in zip(scores, pos) if not p]
    if not a or not b:
        return float("nan")
    w = sum(1.0 if x > y else 0.5 if x == y else 0.0 for x in a for y in b)
    return w / (len(a) * len(b))


def figure_confidence():
    import numpy as np
    import collections

    lab = {r["question_id"]: r for r in C.read_csv(C.RESULTS / "labelled.csv")}
    rows = []
    for r in C.read_csv(C.RESULTS / "judgments.csv"):
        r["label"] = lab[r["question_id"]]["label"]
        if (r["label"] in ("correct", "incorrect") and r["verdict"] in ("ACCEPT", "REJECT")
                and str(r["confidence"]).strip()):
            c = float(r["confidence"])
            r["score"] = c if r["verdict"] == "ACCEPT" else 100.0 - c
            rows.append(r)

    conds = ["A-self", "A-full", "A-strip", "A-floor",
             "B-full", "B-strip", "B-floor", "C-full", "C-strip", "C-floor"]

    def balacc(sub):
        cor = [r for r in sub if r["label"] == "correct"]
        inc = [r for r in sub if r["label"] == "incorrect"]
        a = np.mean([r["verdict"] == "ACCEPT" for r in cor])
        b = np.mean([r["verdict"] == "ACCEPT" for r in inc])
        return (a + (1 - b)) / 2

    by_cap = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        by_cap[r["capsule_uuid"]][r["condition"]].append(r)
    caps = sorted(by_cap)
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(2000):
        pick = rng.choice(caps, size=len(caps), replace=True)
        pooled = collections.defaultdict(list)
        for cap in pick:
            for c in conds:
                pooled[c].extend(by_cap[cap].get(c, []))
        draws.append({c: _auroc([r["score"] for r in pooled[c]],
                                [r["label"] == "correct" for r in pooled[c]])
                      for c in conds})

    fig, ax = plt.subplots(figsize=(8.2, 4.4), dpi=220)
    ys = list(range(len(conds)))[::-1]
    for y, c in zip(ys, conds):
        sub = [r for r in rows if r["condition"] == c]
        t = balacc(sub)
        a = _auroc([r["score"] for r in sub], [r["label"] == "correct" for r in sub])
        vals = [d[c] for d in draws if not np.isnan(d[c])]
        lo, hi = np.percentile(vals, 2.5), np.percentile(vals, 97.5)
        ax.plot([t, a], [y, y], color=BASELINE, linewidth=1.4, zorder=1)
        ax.plot([lo, hi], [y, y], color=CONF_C, linewidth=1, alpha=0.35, zorder=2)
        ax.plot([t], [y], marker="o", markersize=8, color=TOKEN_C,
                markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=3)
        ax.plot([a], [y], marker="o", markersize=8, color=CONF_C,
                markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=3)
        if lo > 0.5:
            ax.text(hi + 0.012, y, "*", fontsize=11, color=CONF_C, va="center")

    ax.axvline(0.5, color=MUTED, linewidth=0.9, linestyle=(0, (3, 3)), zorder=0)
    ax.text(0.5, len(conds) - 0.35, " chance", fontsize=7.5, color=MUTED, va="bottom")

    ax.set_yticks(ys)
    ax.set_yticklabels(conds, fontsize=9, color=SECONDARY)
    ax.set_xlim(0.42, 0.90)
    ax.set_xlabel("discrimination between correct and incorrect answers",
                  fontsize=8.5, color=SECONDARY)
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(length=0, colors=MUTED, labelsize=8)

    fig.legend(handles=[
        Line2D([], [], marker="o", linestyle="", markersize=8, color=TOKEN_C,
               label="ACCEPT/REJECT token (balanced accuracy)"),
        Line2D([], [], marker="o", linestyle="", markersize=8, color=CONF_C,
               label="confidence number (AUROC, 95% CI)")],
        loc="lower center", ncol=2, frameon=False, fontsize=8,
        bbox_to_anchor=(0.5, -0.13))
    fig.suptitle("The verdict is at chance; the confidence beside it is not",
                 fontsize=11, x=0.008, ha="left", color=INK)
    fig.text(0.008, 0.885,
             "Same judgements, same models. * marks a confidence AUROC whose 95% interval "
             "excludes chance.\nThe three answer-only conditions are the exception: without "
             "the notebook, neither channel carries signal.",
             fontsize=7.4, color=MUTED, va="top", linespacing=1.45)
    fig.tight_layout(rect=(0, 0.05, 1, 0.86))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"confidence_vs_verdict.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT/'confidence_vs_verdict.png'}")


if __name__ == "__main__":
    rows = load()
    figure_natural(rows)
    figure_confidence()
