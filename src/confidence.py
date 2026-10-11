"""Does the confidence number carry the signal the ACCEPT/REJECT token throws away?

Confidence is stated in the decision, not in correctness, so verdict and confidence
are folded into one accept-propensity score:

    score = confidence        if the verdict was ACCEPT
            100 - confidence  if the verdict was REJECT

A judge that accepts everything at 95 and rejects at 60 has a flat verdict
distribution but a graded score. AUROC of the score against ground truth asks
whether that grading tracks correctness; balanced accuracy asks the same of the token
alone. The pair isolates what binarising the decision discards.

Also reports, on the same score:
  - the confidence gap between accepted-correct and accepted-incorrect answers, i.e.
    the grading that survives inside a single verdict class;
  - the paired A-self minus A-full difference, the self-review test run on a channel
    without the ceiling that limits the verdict comparison.

Intervals are capsule-cluster bootstraps (seed 0). Draw order is fixed, so the
numbers reproduce exactly.

Usage:
  .venv/bin/python src/confidence.py        # prints the tables, writes results/confidence.csv
"""
from __future__ import annotations

import collections
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

OUT = C.RESULTS / "confidence.csv"
ORDER = ["A-self", "A-full", "A-strip", "A-floor",
         "B-full", "B-strip", "B-floor", "C-full", "C-strip", "C-floor"]
BOOT = 2000


def load() -> list[dict]:
    lab = {r["question_id"]: r for r in C.read_csv(C.RESULTS / "labelled.csv")}
    rows = []
    for r in C.read_csv(C.RESULTS / "judgments.csv"):
        r["label"] = lab[r["question_id"]]["label"]
        if r["label"] in ("correct", "incorrect") and r["verdict"] in ("ACCEPT", "REJECT") \
                and str(r["confidence"]).strip():
            r["conf"] = float(r["confidence"])
            r["score"] = r["conf"] if r["verdict"] == "ACCEPT" else 100.0 - r["conf"]
            rows.append(r)
    return rows


def auroc(scores, labels) -> float:
    """Mann-Whitney U / (n_pos * n_neg), ties counted as one half."""
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float("nan")
    wins = sum((1.0 if p > n else 0.5 if p == n else 0.0) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def balacc(rows_) -> float:
    cor = [r for r in rows_ if r["label"] == "correct"]
    inc = [r for r in rows_ if r["label"] == "incorrect"]
    if not cor or not inc:
        return float("nan")
    a = np.mean([r["verdict"] == "ACCEPT" for r in cor])
    b = np.mean([r["verdict"] == "ACCEPT" for r in inc])
    return (a + (1 - b)) / 2


def ci(vals) -> tuple[float, float]:
    v = np.array([x for x in vals if not np.isnan(x)])
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) else (np.nan, np.nan)


def main() -> None:
    rows = load()
    records = []

    by_cap = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        by_cap[r["capsule_uuid"]][r["condition"]].append(r)
    caps = sorted(by_cap)
    rng = np.random.default_rng(0)
    draws = []
    for _ in range(BOOT):
        pick = rng.choice(caps, size=len(caps), replace=True)
        pooled = collections.defaultdict(list)
        for cap in pick:
            for c in ORDER:
                pooled[c].extend(by_cap[cap].get(c, []))
        draws.append({c: auroc([r["score"] for r in pooled[c]],
                               [r["label"] == "correct" for r in pooled[c]]) for c in ORDER})

    print("=" * 96)
    print("VERDICT vs CONFIDENCE, same judgements")
    print("=" * 96)
    print(f"{'condition':9} {'n':>4} {'bal.acc (token)':>16} {'AUROC (conf)':>13} {'95% CI':>17} "
          f"{'dconf in ACCEPT':>16}")
    for c in ORDER:
        sub = [r for r in rows if r["condition"] == c]
        a = auroc([r["score"] for r in sub], [r["label"] == "correct" for r in sub])
        lo, hi = ci([d[c] for d in draws])
        acc = [r for r in sub if r["verdict"] == "ACCEPT"]
        cc = [r["conf"] for r in acc if r["label"] == "correct"]
        ic = [r["conf"] for r in acc if r["label"] == "incorrect"]
        gap = (np.mean(cc) - np.mean(ic)) if cc and ic else np.nan
        star = "*" if lo > 0.5 else " "
        print(f"{c:9} {len(sub):>4} {balacc(sub):>16.3f} {a:>13.3f}{star} "
              f"[{lo:.3f}, {hi:.3f}] {gap:>+15.1f}")
        records.append({"analysis": "verdict_vs_confidence", "condition": c, "n": len(sub),
                        "balanced_accuracy": round(balacc(sub), 4), "auroc": round(a, 4),
                        "auroc_ci_lo": round(lo, 4), "auroc_ci_hi": round(hi, 4),
                        "conf_gap_within_accept": round(gap, 2) if not np.isnan(gap) else ""})
    print("  * 95% interval excludes 0.5")

    print()
    print("=" * 96)
    print("SELF-REVIEW on the accept-propensity score: paired A-self minus A-full")
    print("=" * 96)
    bys = {r["question_id"]: r for r in rows if r["condition"] == "A-self"}
    byf = {r["question_id"]: r for r in rows if r["condition"] == "A-full"}
    common = sorted(set(bys) & set(byf), key=C._qid_sort_key)
    for subset, name in ((common, "all gradeable"),
                         ([q for q in common if bys[q]["label"] == "incorrect"], "wrong answers"),
                         ([q for q in common if bys[q]["label"] == "correct"], "correct answers")):
        capmap = collections.defaultdict(list)
        for q in subset:
            capmap[bys[q]["capsule_uuid"]].append(q)
        ks = sorted(capmap)
        bs = []
        for _ in range(BOOT):
            pick = rng.choice(ks, size=len(ks), replace=True)
            bs.append(np.mean([bys[q]["score"] - byf[q]["score"] for cap in pick for q in capmap[cap]]))
        est = float(np.mean([bys[q]["score"] - byf[q]["score"] for q in subset]))
        lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
        print(f"  {name:16} n={len(subset):>3}  mean difference {est:+6.2f}  [{lo:+.2f}, {hi:+.2f}]")
        records.append({"analysis": "self_review_paired", "condition": f"A-self - A-full ({name})",
                        "n": len(subset), "paired_difference": round(est, 4),
                        "paired_ci_lo": round(lo, 4), "paired_ci_hi": round(hi, 4)})

    flds = ["analysis", "condition", "n", "balanced_accuracy", "auroc", "auroc_ci_lo",
            "auroc_ci_hi", "conf_gap_within_accept", "paired_difference", "paired_ci_lo",
            "paired_ci_hi"]
    if OUT.exists():
        OUT.unlink()
    for rec in records:
        C.append_row(OUT, rec, flds)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
