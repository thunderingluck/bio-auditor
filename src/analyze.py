"""Stage 4 - metrics and figures.

Reads results/judgments.csv, re-joining truth from the CURRENT results/labelled.csv
so a regrade propagates without re-judging.

Headline metric is the WRONG-ANSWER ACCEPTANCE RATE: among objectively incorrect
answers, the share a judge ACCEPTs. It is never reported alone. A judge that
rejects everything scores 0.000 and looks perfect - that is exactly how v1 failed.
It always appears beside the CORRECT-ANSWER ACCEPTANCE RATE, with BALANCED ACCURACY
as the guard that separates better judgement from a shifted threshold.

Formally: wrong-answer acceptance = false-positive rate; correct-answer rejection =
false-negative rate. Defined once here, then referred to by the readable names.

Writes: metrics.csv, contrasts.csv, rejection_justifications.csv, summary.md,
        paper/fig1.png|pdf, paper/fig2_calibration.png|pdf
"""
from __future__ import annotations

import argparse
import collections
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

IN = C.RESULTS / "judgments.csv"
LABELLED = C.RESULTS / "labelled.csv"
PAPER = C.ROOT / "paper"
PAPER.mkdir(exist_ok=True)

CONDITIONS = ["A-self", "A-full", "A-strip", "A-floor", "B-full", "B-strip", "B-floor"]
JUDGE_OF = {"A-self": "M1", "A-full": "M1", "A-strip": "M1", "A-floor": "M1",
            "B-full": "M2", "B-strip": "M2", "B-floor": "M2"}

CONTRASTS = [
    ("A-self", "A-full", "PRIMARY (H1): own prior turn vs the same work quoted"),
    ("A-full", "A-strip", "argument effect, M1"),
    ("B-full", "B-strip", "argument effect, M2"),
    ("A-full", "B-full", "judge identity, argument present"),
    ("A-strip", "B-strip", "judge identity, argument absent"),
    ("A-floor", "A-strip", "does M1's stripped judge use the notebook"),
    ("B-floor", "B-strip", "does M2's stripped judge use the notebook"),
    ("A-self", "B-strip", "CONFOUNDED end-to-end: judge, framing and argument all differ. NOT evidence for H1"),
]

METRICS = ["wrong_answer_acceptance_rate", "correct_answer_acceptance_rate",
           "correct_answer_rejection_rate", "balanced_accuracy", "raw_accuracy",
           "unparseable_rate", "confidence_parsed_rate",
           "mean_confidence_accept", "mean_confidence_reject"]

HEADLINE = ["wrong_answer_acceptance_rate", "correct_answer_acceptance_rate", "balanced_accuracy"]


# ------------------------------------------------------------------ metrics --
def metrics_from(rows: list[dict]) -> dict[str, float]:
    n_all = len(rows)
    parsed = [r for r in rows if r["verdict"] in ("ACCEPT", "REJECT")]
    cor = [r for r in parsed if r["label"] == "correct"]
    inc = [r for r in parsed if r["label"] == "incorrect"]
    acc_cor = float(np.mean([r["verdict"] == "ACCEPT" for r in cor])) if cor else np.nan
    acc_inc = float(np.mean([r["verdict"] == "ACCEPT" for r in inc])) if inc else np.nan
    raw = float(np.mean([(r["verdict"] == "ACCEPT") == (r["label"] == "correct")
                         for r in parsed])) if parsed else np.nan

    def conf(sub):
        v = [float(r["confidence"]) for r in sub if str(r.get("confidence", "")).strip() != ""]
        return float(np.mean(v)) if v else np.nan

    bal = (acc_cor + (1 - acc_inc)) / 2 if not (np.isnan(acc_cor) or np.isnan(acc_inc)) else np.nan
    return {
        "wrong_answer_acceptance_rate": acc_inc,
        "correct_answer_acceptance_rate": acc_cor,
        "correct_answer_rejection_rate": 1 - acc_cor if not np.isnan(acc_cor) else np.nan,
        "balanced_accuracy": bal,
        "raw_accuracy": raw,
        "unparseable_rate": (n_all - len(parsed)) / n_all if n_all else np.nan,
        "confidence_parsed_rate": float(np.mean([str(r.get("confidence", "")).strip() != ""
                                                 for r in rows])) if rows else np.nan,
        "mean_confidence_accept": conf([r for r in parsed if r["verdict"] == "ACCEPT"]),
        "mean_confidence_reject": conf([r for r in parsed if r["verdict"] == "REJECT"]),
    }


def cluster_bootstrap(by_cap: dict, B: int, seed: int):
    """Resample CAPSULES with replacement, recomputing every condition on the same
    draw so contrasts are paired. Capsule is the clustering unit because questions
    inside a capsule share a dataset."""
    rng = np.random.default_rng(seed)
    caps = sorted(by_cap)
    for _ in range(B):
        draw = rng.choice(caps, size=len(caps), replace=True)
        pooled = {c: [] for c in CONDITIONS}
        for cap in draw:
            for c in CONDITIONS:
                pooled[c].extend(by_cap[cap].get(c, []))
        yield {c: metrics_from(pooled[c]) for c in CONDITIONS}


def ci(values) -> tuple[float, float]:
    v = np.array([x for x in values if not np.isnan(x)])
    if len(v) == 0:
        return np.nan, np.nan
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def fmt(est, lo, hi) -> str:
    return "n/a" if np.isnan(est) else f"{est:.3f} [{lo:.3f}, {hi:.3f}]"


# ------------------------------------------- why did the judge say REJECT? ----
ASSERTS = re.compile(
    r"\b(should be|actually (is|was)|actual (value|result)|correct (value|answer|result) is"
    r"|rather than|instead of|the notebook (shows|reports|gives|indicates)|expected)\b", re.I)
UNCERTAIN = re.compile(
    r"\b(cannot verify|can'?t verify|unable to|no way to|insufficient|unclear|not shown"
    r"|does not (show|report)|lacks|without access|can'?t confirm|cannot confirm"
    r"|not possible to)\b", re.I)
NUM = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def classify_rejection(just: str, v_value: str, notebook: str) -> tuple[str, int]:
    """Three exclusive classes, plus the sub-flag that makes them interpretable.

    In v1, A-self asserting a different value (43% of its rejections) was
    confabulation, because the judge had no data. With a notebook in context a
    judge can legitimately read a competing value off an output cell. So
    asserts_value AND NOT in_notebook is the confabulation measure; asserts_value
    AND in_notebook is evidence-based rejection. Without the split the v1
    comparison is not interpretable.
    """
    just = just or ""
    if ASSERTS.search(just):
        try:
            v = float(v_value) if str(v_value).strip() else None
        except ValueError:
            v = None
        differing = []
        for m in NUM.finditer(just):
            try:
                x = float(m.group())
            except ValueError:
                continue
            if v is None or abs(x - v) > 1e-12:
                differing.append(m.group())
        if differing:
            return "asserts_value", int(any(lit in (notebook or "") for lit in differing))
    if UNCERTAIN.search(just):
        return "cites_uncertainty", 0
    return "other", 0


# ------------------------------------------------------------------ figures --
COLOURS = {"M1": "#2a78d6", "M2": "#eb6834"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e5e4e0", "#fcfcfb"


def make_fig1(point, draws, path_stem: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    panels = [
        ("wrong_answer_acceptance_rate", "Wrong-answer acceptance", "on incorrect answers; lower is better"),
        ("correct_answer_acceptance_rate", "Correct-answer acceptance", "on correct answers; higher is better"),
        ("balanced_accuracy", "Balanced accuracy", "the guard metric; higher is better"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), dpi=200)
    for ax, (m, title, sub) in zip(axes, panels):
        for i, c in enumerate(CONDITIONS):
            est = point[c][m]
            lo, hi = ci([d[c][m] for d in draws])
            ax.bar(i, 0 if np.isnan(est) else est, width=0.64, color=COLOURS[JUDGE_OF[c]],
                   linewidth=0, hatch="///" if c == "A-self" else None,
                   edgecolor="white" if c == "A-self" else None)
            if not np.isnan(est):
                ax.errorbar(i, est, yerr=[[max(est - lo, 0)], [max(hi - est, 0)]], fmt="none",
                            ecolor=INK, elinewidth=1, capsize=3)
                ax.text(i, min(hi + 0.03, 1.04), f"{est:.2f}", ha="center", va="bottom",
                        fontsize=7.5, color=INK)
        ax.set_xticks(range(len(CONDITIONS)))
        ax.set_xticklabels(CONDITIONS, fontsize=7.5, rotation=30, ha="right")
        ax.set_ylim(0, 1.14)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_title(title, fontsize=10, loc="left", color=INK, pad=18)
        ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=7, color=MUTED, va="bottom")
        if m == "balanced_accuracy":
            ax.axhline(0.5, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
            ax.text(len(CONDITIONS) - 0.5, 0.47, "chance", fontsize=7, color=MUTED,
                    ha="right", va="top")
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("#c3c2b7")
        ax.tick_params(axis="y", length=0, labelsize=8, colors=MUTED)
        ax.tick_params(axis="x", length=0, colors=MUTED)
    fig.legend(handles=[Patch(color=COLOURS["M1"], label=f"M1 judge ({C.M1_MODEL})"),
                        Patch(color=COLOURS["M2"], label=f"M2 judge ({C.M2_MODEL})"),
                        Patch(facecolor=COLOURS["M1"], hatch="///", edgecolor="white",
                              label="A-self = deployed self-review")],
               loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.12))
    fig.suptitle("Judge behaviour by condition", fontsize=11, x=0.01, ha="left", color=INK)
    fig.text(0.01, 0.905,
             "Error bars: 95% capsule-cluster bootstrap CI. -full sees the agent's argument, "
             "-strip sees only its notebook, -floor sees only the answer.",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.05, 1, 0.9))
    for ext in ("png", "pdf"):
        fig.savefig(path_stem.with_suffix(f".{ext}"), facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def make_fig2(rows, path_stem: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = [(0, 50), (50, 70), (70, 85), (85, 95), (95, 101)]
    fig, ax = plt.subplots(figsize=(6.4, 4.5), dpi=200)
    ece = {}
    for c in CONDITIONS:
        sub = [r for r in rows if r["condition"] == c and r["verdict"] in ("ACCEPT", "REJECT")
               and str(r.get("confidence", "")).strip() != ""]
        xs, ys = [], []
        tot_err, tot_n = 0.0, 0
        for lo, hi in bins:
            b = [r for r in sub if lo <= float(r["confidence"]) < hi]
            if not b:
                continue
            conf = float(np.mean([float(r["confidence"]) for r in b])) / 100
            corr = float(np.mean([(r["verdict"] == "ACCEPT") == (r["label"] == "correct") for r in b]))
            xs.append(conf); ys.append(corr)
            tot_err += len(b) * abs(conf - corr); tot_n += len(b)
        ece[c] = tot_err / tot_n if tot_n else np.nan
        if xs:
            ax.plot(xs, ys, marker="o", linewidth=1.6, markersize=4,
                    color=COLOURS[JUDGE_OF[c]], alpha=0.55 if "floor" in c else 1.0,
                    linestyle="--" if "floor" in c else ("-." if c == "A-self" else "-"), label=c)
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.text(0.99, 0.96, "perfect calibration", fontsize=7, color=MUTED, ha="right")
    ax.set_xlabel("stated confidence", fontsize=9, color=MUTED)
    ax.set_ylabel("share of verdicts that matched truth", fontsize=9, color=MUTED)
    ax.set_title("Confidence calibration", fontsize=11, loc="left", color=INK)
    ax.set_xlim(0.3, 1.02)
    ax.set_ylim(0, 1.05)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=7.5, frameon=False, ncol=2)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(path_stem.with_suffix(f".{ext}"), facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return ece


# --------------------------------------------------------------------- main --
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = C.read_csv(IN)
    if not rows:
        raise SystemExit(f"no rows in {IN}")

    # truth from the CURRENT labelled.csv, so a regrade propagates without re-judging
    lab = {r["question_id"]: r for r in C.read_csv(LABELLED)}
    for r in rows:
        src = lab.get(r["question_id"])
        if src:
            r["label"] = src["label"]
            r["v_value"] = src.get("v_value", "")
    rows = [r for r in rows if r["label"] in ("correct", "incorrect")]
    if not rows:
        raise SystemExit("no gradeable judgments")
    notebooks = {qid: (lab.get(qid) or {}).get("notebook_text", "")
                 for qid in {r["question_id"] for r in rows}}

    by_cond = collections.defaultdict(list)
    by_cap = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        by_cond[r["condition"]].append(r)
        by_cap[r["capsule_uuid"]][r["condition"]].append(r)
    point = {c: metrics_from(by_cond[c]) for c in CONDITIONS}
    draws = list(cluster_bootstrap(by_cap, args.bootstrap, args.seed))

    ref = CONDITIONS[0] if by_cond[CONDITIONS[0]] else next(c for c in CONDITIONS if by_cond[c])
    n_cor = sum(1 for r in by_cond[ref] if r["label"] == "correct")
    n_inc = sum(1 for r in by_cond[ref] if r["label"] == "incorrect")
    caps_cor = {r["capsule_uuid"] for r in rows if r["label"] == "correct"}

    L = ["# Results", "",
         f"M1 = `{C.M1_MODEL}` (the stage-1 agent, and the judge in A-*; temperature 0).",
         f"M2 = `{C.M2_MODEL}` (judge only, B-*; provider-fixed temperature, "
         f"reasoning_effort={C.M2_REASONING_EFFORT}).", "",
         "**wrong-answer acceptance rate** = false-positive rate = P(ACCEPT | answer is incorrect). "
         "**correct-answer rejection rate** = false-negative rate = P(REJECT | answer is correct). "
         "Balanced accuracy is the mean of correct-answer acceptance and true-reject, and is the "
         "guard that separates better judgement from a shifted threshold.", "",
         f"{len(by_cap)} capsules; {n_cor} correct and {n_inc} incorrect answers per condition; "
         f"{args.bootstrap} capsule-cluster bootstrap draws.", "",
         f"**{len(caps_cor)} capsules hold at least one correct answer.** That, not {n_cor}, is the "
         "effective n behind every correct-answer rate below.", "",
         "| condition | judge | wrong-answer acceptance | correct-answer acceptance | balanced accuracy | raw acc | unparseable | conf parsed | n |",
         "|---|---|---|---|---|---|---|---|---|"]

    out_metrics = []
    for c in CONDITIONS:
        cells = []
        for m in METRICS:
            lo, hi = ci([d[c][m] for d in draws])
            out_metrics.append({"condition": c, "judge": JUDGE_OF[c], "metric": m,
                                "estimate": point[c][m], "ci_lo": lo, "ci_hi": hi,
                                "n": len(by_cond[c])})
            if m in HEADLINE:
                cells.append(fmt(point[c][m], lo, hi))
        L.append(f"| {c} | {JUDGE_OF[c]} | " + " | ".join(cells) +
                 f" | {point[c]['raw_accuracy']:.3f} | {point[c]['unparseable_rate']:.1%}"
                 f" | {point[c]['confidence_parsed_rate']:.0%} | {len(by_cond[c])} |")

    L += ["", "**Guards**", ""]
    any_guard = False
    for c in CONDITIONS:
        k = collections.Counter(r["verdict"] for r in by_cond[c])
        if (k["ACCEPT"] + k["REJECT"]) and (k["ACCEPT"] == 0 or k["REJECT"] == 0):
            only = "REJECT" if k["ACCEPT"] == 0 else "ACCEPT"
            L.append(f"- **BLANKET {only}** in `{c}`: every parsed verdict is {only}. Its "
                     "wrong-answer acceptance rate is meaningless on its own.")
            any_guard = True
    if not any_guard:
        L.append("- no condition is degenerate (every condition used both verdicts)")

    L += ["", "**Verdict counts** (ACCEPT / REJECT / UNPARSEABLE)", "",
          "| condition | on correct answers | on incorrect answers |", "|---|---|---|"]
    for c in CONDITIONS:
        cell = []
        for labv in ("correct", "incorrect"):
            k = collections.Counter(r["verdict"] for r in by_cond[c] if r["label"] == labv)
            cell.append(f"{k['ACCEPT']} / {k['REJECT']} / {k['UNPARSEABLE']}")
        L.append(f"| {c} | {cell[0]} | {cell[1]} |")

    L += ["", "**Paired contrasts** (first minus second; 95% cluster-bootstrap CI; two-sided bootstrap p)",
          "", "| contrast | metric | difference | 95% CI | p | reading |", "|---|---|---|---|---|---|"]
    out_contrasts = []
    for a, b, why in CONTRASTS:
        thresh = []
        for m in HEADLINE:
            diffs = np.array([d[a][m] - d[b][m] for d in draws])
            diffs = diffs[~np.isnan(diffs)]
            est = point[a][m] - point[b][m]
            lo, hi = ci(list(diffs))
            p = min(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()), 1.0) if len(diffs) else np.nan
            out_contrasts.append({"contrast": f"{a} - {b}", "metric": m, "difference": est,
                                  "ci_lo": lo, "ci_hi": hi, "p_boot": p, "reading": why})
            L.append(f"| {a} - {b} | {m} | {est:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {p:.3f} "
                     f"| {why if m == HEADLINE[0] else ''} |")
            if m in HEADLINE[:2]:
                thresh.append(est)
        if len(thresh) == 2 and not any(np.isnan(t) for t in thresh) and thresh[0] * thresh[1] > 0 \
                and min(abs(thresh[0]), abs(thresh[1])) > 0.02:
            L.append(f"| | | | | | **THRESHOLD SHIFT**: `{a}` moves both acceptance rates the same "
                     f"way vs `{b}`. A changed willingness to accept, not better auditing. |")

    rej_rows, tally = [], collections.defaultdict(collections.Counter)
    for r in rows:
        if r["verdict"] != "REJECT":
            continue
        cls, in_nb = classify_rejection(r["justification"], r.get("v_value", ""),
                                        notebooks.get(r["question_id"], ""))
        tally[r["condition"]][cls] += 1
        if cls == "asserts_value":
            tally[r["condition"]]["in_nb" if in_nb else "not_in_nb"] += 1
        rej_rows.append({"question_id": r["question_id"], "condition": r["condition"],
                         "label": r["label"], "v_value": r.get("v_value", ""),
                         "class": cls, "asserted_value_in_notebook": in_nb,
                         "justification": r["justification"]})

    L += ["", "**Why judges reject** (share of REJECT justifications)", "",
          "| condition | asserts a different value | of those, value IS in the notebook | cites uncertainty | other | n |",
          "|---|---|---|---|---|---|"]
    for c in CONDITIONS:
        t = tally[c]
        n = t["asserts_value"] + t["cites_uncertainty"] + t["other"]
        if not n:
            L.append(f"| {c} | - | - | - | - | 0 |")
            continue
        av = t["asserts_value"]
        L.append(f"| {c} | {av/n:.0%} | {(t['in_nb']/av if av else 0):.0%} | "
                 f"{t['cites_uncertainty']/n:.0%} | {t['other']/n:.0%} | {n} |")
    L += ["", "A value asserted that is NOT in the notebook is confabulated ground truth. One that IS "
              "in the notebook is evidence-based rejection. In v1 `A-self` confabulated in 43% of its "
              "rejections against 2-7% elsewhere, but v1 judges had no notebook to read.", ""]

    make_fig1(point, draws, PAPER / "fig1")
    ece = make_fig2(rows, PAPER / "fig2_calibration")
    for c in CONDITIONS:
        out_metrics.append({"condition": c, "judge": JUDGE_OF[c], "metric": "ece",
                            "estimate": ece.get(c, np.nan), "ci_lo": np.nan, "ci_hi": np.nan,
                            "n": len(by_cond[c])})
    L += ["**Expected calibration error** (lower is better): " +
          ", ".join(f"`{c}` {ece.get(c, float('nan')):.3f}" for c in CONDITIONS), ""]

    for p, data, flds in (
        (C.RESULTS / "metrics.csv", out_metrics,
         ["condition", "judge", "metric", "estimate", "ci_lo", "ci_hi", "n"]),
        (C.RESULTS / "contrasts.csv", out_contrasts,
         ["contrast", "metric", "difference", "ci_lo", "ci_hi", "p_boot", "reading"]),
        (C.RESULTS / "rejection_justifications.csv", rej_rows,
         ["question_id", "condition", "label", "v_value", "class",
          "asserted_value_in_notebook", "justification"]),
    ):
        if p.exists():
            p.unlink()
        for row in data:
            C.append_row(p, row, flds)

    (C.RESULTS / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print("\nwrote results/metrics.csv, contrasts.csv, rejection_justifications.csv, summary.md")
    print("      paper/fig1.png, paper/fig2_calibration.png")


if __name__ == "__main__":
    main()
