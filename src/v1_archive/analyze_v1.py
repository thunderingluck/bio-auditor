"""Stage 4 - metrics + one figure.

Reads results/judgments.csv. Writes:
  results/metrics.csv      long format: set, condition, metric, estimate, ci_lo, ci_hi, n
  results/contrasts.csv    paired contrasts (B-C, A-A', C-A, A-B) with cluster-bootstrap CIs
  results/summary.md       human-readable tables
  paper/fig1.png, .pdf     balanced accuracy / false-accept / false-reject per condition

Headline metric: balanced accuracy = mean(true-accept rate on correct, true-reject rate on
incorrect). Capsule is the clustering unit: all CIs are percentile intervals from a
cluster bootstrap that resamples capsules with replacement (B draws), recomputing every
condition on the same resampled capsules so contrasts are paired.

Unparseable verdicts are excluded from the rate denominators and reported as a rate.
Two analysis sets: `all` (natural + synthetic_correct answers) and `natural` (natural only).
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

IN = C.RESULTS / "judgments.csv"
PAPER = C.ROOT / "paper"
PAPER.mkdir(exist_ok=True)

CONDITIONS = ["A", "A'", "B", "C"]
JUDGE_OF = {"A": "M1", "A'": "M1", "B": "M2", "C": "M2"}
CONTRASTS = [("B", "C", "H1: trace vs no trace, same judge (M2)"),
             ("A", "A'", "H1/H3: author with vs without its trace"),
             ("C", "A", "H2: quarantined auditor vs deployed self-review (headline)"),
             ("A", "B", "descriptive only: judge, role and authorship all differ")]
METRICS = ["balanced_accuracy", "false_accept_rate", "false_reject_rate", "true_accept_rate",
           "true_reject_rate", "raw_accuracy", "unparseable_rate"]


def metrics_from(rows: list[dict]) -> dict[str, float]:
    """rows: judgments for one condition. NaN where undefined."""
    n_all = len(rows)
    parsed = [r for r in rows if r["verdict"] in ("CORRECT", "INCORRECT")]
    cor = [r for r in parsed if r["label"] == "correct"]
    inc = [r for r in parsed if r["label"] == "incorrect"]
    tar = np.mean([r["verdict"] == "CORRECT" for r in cor]) if cor else np.nan
    trr = np.mean([r["verdict"] == "INCORRECT" for r in inc]) if inc else np.nan
    acc = np.mean([(r["verdict"] == "CORRECT") == (r["label"] == "correct") for r in parsed]) if parsed else np.nan
    return {
        "balanced_accuracy": (tar + trr) / 2,
        "false_accept_rate": 1 - trr,
        "false_reject_rate": 1 - tar,
        "true_accept_rate": tar,
        "true_reject_rate": trr,
        "raw_accuracy": acc,
        "unparseable_rate": (n_all - len(parsed)) / n_all if n_all else np.nan,
    }


def cluster_bootstrap(by_cap: dict[str, dict[str, list[dict]]], B: int, seed: int):
    """by_cap[capsule][condition] -> rows. Yields per-draw {condition: metrics}."""
    rng = np.random.default_rng(seed)
    caps = sorted(by_cap)
    for _ in range(B):
        draw = rng.choice(caps, size=len(caps), replace=True)
        pooled = {c: [] for c in CONDITIONS}
        for cap in draw:
            for c in CONDITIONS:
                pooled[c].extend(by_cap[cap].get(c, []))
        yield {c: metrics_from(pooled[c]) for c in CONDITIONS}


def ci(values: list[float]) -> tuple[float, float]:
    v = np.array([x for x in values if not np.isnan(x)])
    if len(v) == 0:
        return np.nan, np.nan
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def analyse_set(name: str, rows: list[dict], B: int, seed: int, out_metrics: list, out_contrasts: list, lines: list):
    by_cond = collections.defaultdict(list)
    by_cap = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        by_cond[r["condition"]].append(r)
        by_cap[r["capsule_uuid"]][r["condition"]].append(r)
    point = {c: metrics_from(by_cond[c]) for c in CONDITIONS}
    draws = list(cluster_bootstrap(by_cap, B, seed))

    n_cor = sum(r["label"] == "correct" for r in by_cond["A"])
    n_inc = sum(r["label"] == "incorrect" for r in by_cond["A"])
    lines.append(f"\n## Set `{name}`: {len(by_cap)} capsules, {n_cor} correct + {n_inc} incorrect answers per condition, "
                 f"{B} cluster-bootstrap draws\n")
    lines.append("| condition | judge | balanced acc | false-accept (wrong answers) | false-reject (right answers) | raw acc | unparseable | n |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for c in CONDITIONS:
        cells = []
        for m in METRICS:
            lo, hi = ci([d[c][m] for d in draws])
            out_metrics.append({"set": name, "condition": c, "judge": JUDGE_OF[c], "metric": m,
                                "estimate": point[c][m], "ci_lo": lo, "ci_hi": hi, "n": len(by_cond[c])})
            if m in ("balanced_accuracy", "false_accept_rate", "false_reject_rate", "raw_accuracy", "unparseable_rate"):
                cells.append(_fmt(point[c][m], lo, hi) if m != "unparseable_rate" else f"{point[c][m]:.1%}")
        lines.append(f"| {c} | {JUDGE_OF[c]} | " + " | ".join(cells) + f" | {len(by_cond[c])} |")

    lines.append("\n**Verdict counts** (rows: truth label; cells: CORRECT / INCORRECT / UNPARSEABLE verdicts)\n")
    lines.append("| condition | correct answers | incorrect answers |")
    lines.append("|---|---|---|")
    for c in CONDITIONS:
        cell = []
        for lab in ("correct", "incorrect"):
            cnt = collections.Counter(r["verdict"] for r in by_cond[c] if r["label"] == lab)
            cell.append(f"{cnt['CORRECT']} / {cnt['INCORRECT']} / {cnt['UNPARSEABLE']}")
        lines.append(f"| {c} | {cell[0]} | {cell[1]} |")

    lines.append("\n**Paired contrasts** (first minus second; 95% cluster-bootstrap CI; two-sided bootstrap p)\n")
    lines.append("| contrast | metric | difference | 95% CI | p | reading |")
    lines.append("|---|---|---|---|---|---|")
    for a, b, why in CONTRASTS:
        for m in ("balanced_accuracy", "false_accept_rate", "false_reject_rate"):
            diffs = np.array([d[a][m] - d[b][m] for d in draws])
            diffs = diffs[~np.isnan(diffs)]
            est = point[a][m] - point[b][m]
            lo, hi = ci(list(diffs))
            p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()) if len(diffs) else np.nan
            p = min(p, 1.0)
            out_contrasts.append({"set": name, "contrast": f"{a} - {b}", "metric": m, "difference": est,
                                  "ci_lo": lo, "ci_hi": hi, "p_boot": p, "reading": why})
            lines.append(f"| {a} − {b} | {m} | {est:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {p:.3f} | {why if m == 'balanced_accuracy' else ''} |")

    # why do judges reject? keyword rates in the justification of INCORRECT verdicts
    import re
    kw = {"cannot verify / no data": re.compile(r"(without|no|cannot|can't|lack).{0,40}(access|data|verify|verif|confirm|reproduc)", re.I),
          "asserts the actual value differs": re.compile(r"(actual|true|correct|expected|known) (value|answer|result|p-value|number)", re.I)}
    lines.append("\n**Stated reason for INCORRECT verdicts** (share of rejections whose justification matches each pattern)\n")
    lines.append("| condition | " + " | ".join(kw) + " | n rejections |")
    lines.append("|---|" + "---|" * (len(kw) + 1))
    for c in CONDITIONS:
        rej = [r for r in by_cond[c] if r["verdict"] == "INCORRECT"]
        cells = [f"{sum(bool(p.search(r['verdict_text'])) for r in rej) / len(rej):.0%}" if rej else "n/a" for p in kw.values()]
        lines.append(f"| {c} | " + " | ".join(cells) + f" | {len(rej)} |")

    # acceptance of correct answers by source (style-confound check)
    srcs = sorted({r["source"] for r in rows if r["label"] == "correct"})
    if len(srcs) > 1:
        lines.append("\n**True-accept rate on correct answers, by source** (synthetic vs genuinely derived)\n")
        lines.append("| condition | " + " | ".join(srcs) + " |")
        lines.append("|---|" + "---|" * len(srcs))
        for c in CONDITIONS:
            cells = []
            for s in srcs:
                sub = [r for r in by_cond[c] if r["label"] == "correct" and r["source"] == s and r["verdict"] != "UNPARSEABLE"]
                cells.append(f"{np.mean([r['verdict'] == 'CORRECT' for r in sub]):.2f} (n={len(sub)})" if sub else "n/a")
            lines.append(f"| {c} | " + " | ".join(cells) + " |")
    return point, draws


def _fmt(est, lo, hi):
    return f"{est:.3f} [{lo:.3f}, {hi:.3f}]" if not np.isnan(est) else "n/a"


def make_figure(point: dict, draws: list, path_stem: Path, subtitle: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"M1": "#2a78d6", "M2": "#eb6834"}
    panels = [("balanced_accuracy", "Balanced accuracy", "higher is better"),
              ("false_accept_rate", "False-accept rate", "on incorrect answers; lower is better"),
              ("false_reject_rate", "False-reject rate", "on correct answers; lower is better")]
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), dpi=200)
    for ax, (m, title, sub) in zip(axes, panels):
        for i, c in enumerate(CONDITIONS):
            est = point[c][m]
            lo, hi = ci([d[c][m] for d in draws])
            ax.bar(i, est, width=0.62, color=colours[JUDGE_OF[c]], linewidth=0)
            ax.errorbar(i, est, yerr=[[est - lo], [hi - est]], fmt="none", ecolor="#0b0b0b", elinewidth=1, capsize=3)
            ax.text(i, min(hi + 0.03, 1.02), f"{est:.2f}", ha="center", va="bottom", fontsize=8, color="#0b0b0b")
        ax.set_xticks(range(len(CONDITIONS)))
        ax.set_xticklabels([f"{c}\n({JUDGE_OF[c]})" for c in CONDITIONS], fontsize=8)
        ax.set_ylim(0, 1.12)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_title(title, fontsize=10, loc="left", color="#0b0b0b", pad=18)
        ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=7, color="#52514e", va="bottom")
        if m == "balanced_accuracy":
            ax.axhline(0.5, color="#52514e", linewidth=0.8, linestyle=(0, (3, 3)))
            ax.text(len(CONDITIONS) - 0.55, 0.47, "chance", fontsize=7, color="#52514e", ha="right", va="top")
        ax.grid(axis="y", color="#e5e4e0", linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("#c3c2b7")
        ax.tick_params(axis="y", length=0, labelsize=8, colors="#52514e")
        ax.tick_params(axis="x", length=0)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=colours["M1"], label=f"M1 judge ({C.M1_MODEL})"),
                        Patch(color=colours["M2"], label=f"M2 judge ({C.M2_MODEL})")],
               loc="lower center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"Judge performance by condition — {subtitle}", fontsize=10, x=0.01, ha="left", color="#0b0b0b")
    fig.text(0.01, 0.905, "Error bars: 95% capsule-cluster bootstrap CI. A: self-review with own trace; A': author, answer only; "
             "B: other model with trace; C: other model, answer only.", fontsize=7, color="#52514e")
    fig.tight_layout(rect=(0, 0.02, 1, 0.9))
    fig.savefig(path_stem.with_suffix(".png"), facecolor="#fcfcfb", bbox_inches="tight")
    fig.savefig(path_stem.with_suffix(".pdf"), facecolor="#fcfcfb", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = C.read_csv(IN)
    if not rows:
        raise SystemExit(f"no rows in {IN}")
    # Truth labels come from the current labelled.csv, not the copy stored at judge time,
    # so a regrade (e.g. the numeric-tolerance fix) propagates without re-judging.
    labels = {(r["question_id"], r["sample_idx"], r["source"]): r["label"]
              for r in C.read_csv(C.RESULTS / "labelled.csv")}
    for r in rows:
        r["label"] = labels.get((r["question_id"], r["sample_idx"], r["source"]), r["label"])
    rows = [r for r in rows if r["label"] in ("correct", "incorrect")]
    lines = ["# Results summary", "",
             f"M1 = `{C.M1_MODEL}` (answerer, judge in A/A'; temperature {C.JUDGE_TEMPERATURE} when judging). "
             f"M2 = `{C.M2_MODEL}` (judge in B/C; temperature fixed at 1 by provider, reasoning_effort={C.M2_REASONING_EFFORT}).",
             "Unparseable verdicts are excluded from rate denominators and reported separately. "
             "CIs are percentile intervals from a capsule-cluster bootstrap; contrasts are paired on the same draws."]
    out_metrics, out_contrasts = [], []
    sets = {"all": rows, "natural": [r for r in rows if r["source"] == "natural"]}
    figs = {}
    for name, sub in sets.items():
        if not sub:
            continue
        figs[name] = analyse_set(name, sub, args.bootstrap, args.seed, out_metrics, out_contrasts, lines)

    fields_m = ["set", "condition", "judge", "metric", "estimate", "ci_lo", "ci_hi", "n"]
    fields_c = ["set", "contrast", "metric", "difference", "ci_lo", "ci_hi", "p_boot", "reading"]
    for p in (C.RESULTS / "metrics.csv", C.RESULTS / "contrasts.csv"):
        if p.exists():
            p.unlink()
    for r in out_metrics:
        C.append_row(C.RESULTS / "metrics.csv", r, fields_m)
    for r in out_contrasts:
        C.append_row(C.RESULTS / "contrasts.csv", r, fields_c)
    (C.RESULTS / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if "all" in figs:
        make_figure(*figs["all"], PAPER / "fig1", "all answers (natural + synthetic-correct)")
    if "natural" in figs:
        make_figure(*figs["natural"], PAPER / "fig1_natural_only", "genuinely derived answers only")
    print("\n".join(lines))
    print(f"\nwrote results/metrics.csv, results/contrasts.csv, results/summary.md, paper/fig1.png")


if __name__ == "__main__":
    main()
