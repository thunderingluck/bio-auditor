"""Stage 2 - deterministic truth labelling. No API calls, no model in the loop.

Reads  results/answers.csv
Writes results/labelled.csv, results/normalisation_log.csv, results/adjudications.csv

THE PROBLEM THIS FILE SOLVES
BixBench open mode returns free prose, not a value. Measured over 36 trajectories,
50% of answers contain two or more distinct numbers (range questions 73%), and no
positional extraction rule exceeds 56% fidelity to the value the agent meant
(first 6/16, last 6/16, rounding-cue 9/16, cue-with-fallback 11/16). A heuristic
would mislabel about a third of ambiguous answers, and a wrong value flips a
correctness label. So we do not guess: single-value answers are auto-labelled and
multi-value answers go to a human for BLIND adjudication.

BLIND MATTERS. The review file deliberately omits the ideal, the interval and the
distractors. If the adjudicator can see the target they have become the grader,
and the deterministic-ground-truth strength of the design is gone. The human picks
which number the agent was putting forward; the code decides whether it is right.

Labels: correct | incorrect | missing | ambiguous | excluded

Usage:
  .venv/bin/python src/grade_truth.py --emit-review   # pass 1, writes the review file
  .venv/bin/python src/grade_truth.py                 # pass 2, reads decisions back
  .venv/bin/python src/grade_truth.py --self-test
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import math
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

IN = C.RESULTS / "answers.csv"
OUT = C.RESULTS / "labelled.csv"
LOG = C.RESULTS / "normalisation_log.csv"
REVIEW = C.RESULTS / "adjudications.csv"

NUM_RE = re.compile(r"[-+]?(?:\d+(?:,\d{3})*(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
DASHES = {"–": "-", "—": "-", "−": "-", "‒": "-", "‐": "-", "‑": "-"}
LONG_STRING = 120          # a "string answer" longer than this is prose, not a token

REVIEW_FIELDS = [
    "question_id", "answer_sha256", "eval_mode", "reason", "n_candidates",
    "question", "answer_body", "candidates_display", "candidates_json",
    "decision", "chosen_index", "chosen_value", "override_label", "override_reason",
    "adjudicator", "notes",
]


# ============================================================ v1 core, verbatim ==
# These five functions are carried over unchanged from the v1 grader. They are
# correct and tested. In particular grade_str uses math.isclose with rel_tol only:
# an absolute floor previously made 2.2e-16 equal 9.92E-35. Do not reintroduce one.

def basic_norm(s: str) -> str:
    s = str(s).strip()
    for k, v in DASHES.items():
        s = s.replace(k, v)
    s = re.sub(r"\s+", " ", s).casefold()
    s = s.strip(" .;:\"'`*")
    return s


def coerce_number(s: str) -> tuple[float | None, str]:
    """Return (value, note). note='' if the string is a bare number."""
    t = basic_norm(s)
    if not t:
        return None, ""
    bare = t.replace(",", "")
    try:
        return float(bare), ""
    except ValueError:
        pass
    if bare.endswith("%"):
        try:
            return float(bare[:-1].strip()), "stripped %"
        except ValueError:
            pass
    m = re.fullmatch(r"[~≈≅=]?\s*(" + NUM_RE.pattern + r")\s*[a-z%°/()\s\-]*", t)
    if m:
        try:
            return float(m.group(1).replace(",", "")), f"extracted number from {t!r}"
        except ValueError:
            pass
    return None, ""


def parse_range(ideal: str) -> tuple[float, float]:
    m = re.match(r"^\s*[\(\[]\s*([^,]+?)\s*,\s*([^\)\]]+?)\s*[\)\]]\s*$", str(ideal))
    if not m:
        raise ValueError(f"cannot parse range ideal {ideal!r}")
    lo, hi = float(m.group(1)), float(m.group(2))
    return (lo, hi) if lo <= hi else (hi, lo)


def grade_str(answer: str, ideal: str) -> tuple[str, str]:
    a, i = basic_norm(answer), basic_norm(ideal)
    if a == i:
        note = "" if str(answer).strip() == str(ideal).strip() else "basic normalisation"
        return "correct", note
    va, na = coerce_number(a)
    vi, ni = coerce_number(i)
    if va is not None and vi is not None:
        if math.isclose(va, vi, rel_tol=1e-9, abs_tol=0.0):
            return "correct", "numeric coercion" + (f"; answer: {na}" if na else "") + (f"; ideal: {ni}" if ni else "")
        return "incorrect", ("numeric coercion" + (f"; answer: {na}" if na else "")) if (na or ni) else ""
    return "incorrect", ""


def grade_range(answer: str, ideal: str) -> tuple[str, str]:
    lo, hi = parse_range(ideal)
    v, note = coerce_number(answer)
    if v is None:
        return "unparseable", f"could not coerce {answer!r} to a number"
    return ("correct" if lo <= v <= hi else "incorrect"), note


# ==================================================== V extraction (new in v2) ==
@dataclass(frozen=True)
class Candidate:
    value: float
    is_percent: bool
    literal: str
    start: int
    context: str


# Spans that generate nuisance numbers. Masked (blanked, preserving offsets) before
# candidates are extracted. Imperfect masking is fine: every miss pushes a row to
# the human, never to a wrong label.
MASKS = [
    re.compile(r"\bGO:\s*\d+", re.I),                       # GO:0050863 -> 50863
    re.compile(r"\b[A-Za-z]+[-‐-―]?\d+[A-Za-z0-9]*\b"),  # CCND1, JBX97, COVID-19
    re.compile(r"\(\s*9\d\s*%\s*CI[^)]*\)", re.I),          # (95% CI: 0.76-2.40)
    re.compile(r"\bp\s*[-\s]?(?:value)?\s*[=<>≈]\s*[\d.eE+-]+", re.I),
    re.compile(r"\b(?:n|df|N)\s*=\s*[\d,]+"),
    re.compile(r"\b(?:padj|FDR|alpha)\s*[=<>]\s*[\d.eE+-]+", re.I),
]


def mask_traps(text: str) -> str:
    """Blank nuisance spans, preserving string offsets so contexts stay aligned."""
    out = list(text)
    for rx in MASKS:
        for m in rx.finditer(text):
            for i in range(m.start(), m.end()):
                out[i] = " "
    return "".join(out)


def numeric_candidates(text: str) -> list[Candidate]:
    masked = mask_traps(text)
    cands: list[Candidate] = []
    for m in NUM_RE.finditer(masked):
        lit = m.group()
        try:
            val = float(lit.replace(",", ""))
        except ValueError:
            continue
        after = text[m.end():m.end() + 1]
        lo, hi = max(0, m.start() - 45), min(len(text), m.end() + 45)
        ctx = " ".join(text[lo:hi].split())
        cands.append(Candidate(val, after == "%", lit, m.start(), ctx))
    return cands


def distinct(cands: list[Candidate]) -> list[Candidate]:
    """Unique by (is_percent, value)."""
    seen, out = set(), []
    for c in cands:
        key = (c.is_percent, round(c.value, 12))
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def ideal_is_numeric(ideal: str, eval_mode: str) -> bool:
    """Can this ideal be compared numerically at all?

    Mode-aware on purpose. A range ideal like "(1.50,1.54)" does NOT coerce to a
    float, so a coerce-only check would route every range question down the string
    path and mark it incorrect. The string path exists only for ideals that are
    genuinely tokens, e.g. CDKN1A or 1-50.
    """
    if eval_mode == "range_verifier":
        try:
            parse_range(ideal)
            return True
        except ValueError:
            return False
    return coerce_number(ideal)[0] is not None


# ---------------------------------------------------------------- unit policy --
def apply_units(value: float, is_percent: bool, ideal: str, eval_mode: str) -> tuple[float, str]:
    """Eva's rule: convert ONLY when unambiguous. Never try all scalings.

    Divide by 100 when the answer is an explicit percentage and the ideal range
    lies entirely within [0,1]; multiply when the reverse holds.
    """
    if eval_mode != "range_verifier":
        return value, ""
    try:
        lo, hi = parse_range(ideal)
    except ValueError:
        return value, ""
    if is_percent and 0.0 <= lo and hi <= 1.0:
        return value / 100.0, f"percent->proportion ({value} -> {value / 100.0})"
    if not is_percent and 0.0 <= value <= 1.0 and lo >= 1.0:
        return value * 100.0, f"proportion->percent ({value} -> {value * 100.0})"
    return value, ""


def grade_value(value: float, ideal: str, eval_mode: str) -> tuple[str, str]:
    if eval_mode == "range_verifier":
        return grade_range(repr(value), ideal)
    return grade_str(repr(value), ideal)


# ========================================================================= I/O ==
def sha(s: str) -> str:
    return hashlib.sha256(str(s).encode("utf-8")).hexdigest()[:16]


def candidates_display(cands: list[Candidate]) -> str:
    return "  ".join(
        f"[{i}] {c.literal}{'%' if c.is_percent else ''} <{c.context}>"
        for i, c in enumerate(cands)
    )


def emit_review(rows: list[dict]) -> int:
    """Pass 1. Merge rather than overwrite: filled decisions survive."""
    import json
    prior = {r["question_id"]: r for r in C.read_csv(REVIEW)}
    if REVIEW.exists():
        shutil.copy2(REVIEW, REVIEW.with_suffix(".prev.csv"))
        REVIEW.unlink()

    n_new = n_kept = n_stale = 0
    for r in rows:
        if r["_label"] != "ambiguous":
            continue
        cands = r["_candidates"]
        digest = sha(r["answer_raw"])
        old = prior.get(r["question_id"])
        decision = chosen_index = chosen_value = adjudicator = notes = ""
        if old:
            if old.get("answer_sha256") == digest and old.get("decision", "").strip():
                decision = old["decision"]; chosen_index = old.get("chosen_index", "")
                chosen_value = old.get("chosen_value", ""); adjudicator = old.get("adjudicator", "")
                notes = old.get("notes", ""); n_kept += 1
            elif old.get("decision", "").strip():
                notes = "STALE - answer changed since this was adjudicated"; n_stale += 1
            else:
                n_new += 1
        else:
            n_new += 1
        C.append_row(REVIEW, {
            "question_id": r["question_id"],
            "answer_sha256": digest,
            "eval_mode": r["eval_mode"],
            "reason": r["_reason"],
            "n_candidates": len(cands),
            # NOTE: no ideal, no interval, no distractors. Blind by construction.
            "question": " ".join(r["question"].split())[:400],
            "answer_body": r["answer_raw"],
            "candidates_display": candidates_display(cands),
            "candidates_json": json.dumps([
                {"index": i, "value": c.value, "is_percent": c.is_percent, "literal": c.literal}
                for i, c in enumerate(cands)]),
            "decision": decision, "chosen_index": chosen_index, "chosen_value": chosen_value,
            "adjudicator": adjudicator, "notes": notes,
        }, REVIEW_FIELDS)
    print(f"\nreview file -> {REVIEW}")
    print(f"  rows needing a decision: {n_new}   already decided (kept): {n_kept}   stale: {n_stale}")
    return n_new


def classify(r: dict) -> dict:
    """Route one row. Sets _label, _reason, _candidates, _v, _note."""
    ideal, mode, ans = r["ideal"], r["eval_mode"], r["answer_raw"]
    r["_candidates"], r["_v"], r["_note"], r["_v_source"], r["_unit_flag"] = [], "", "", "", ""

    if str(r.get("answered", "1")) != "1" or not str(ans).strip():
        r["_label"], r["_reason"] = "missing", "agent never submitted"
        return r
    if str(r.get("trace_complete", "1")) != "1":
        r["_label"], r["_reason"] = "excluded", "contaminated_trace"
        return r

    # string path: triggered by the IDEAL failing to parse, which is the only
    # reliable signal. CCND1 floats to 1.0 on the answer side, so answer-side
    # detection does not work.
    if not ideal_is_numeric(ideal, mode):
        if len(str(ans)) > LONG_STRING:
            lab, note = grade_str(ans, ideal)
            if lab != "correct":
                r["_label"], r["_reason"] = "ambiguous", "long_string"
                r["_candidates"] = distinct(numeric_candidates(ans))
                return r
        lab, note = grade_str(ans, ideal)
        r["_label"], r["_reason"], r["_note"], r["_v_source"] = lab, "string_path", note, "auto_string"
        r["_v"] = basic_norm(ans)
        return r

    cands = distinct(numeric_candidates(ans))
    r["_candidates"] = cands
    if len(cands) == 0:
        r["_label"], r["_reason"] = "ambiguous", "no_value"
        return r
    if len(cands) > 1:
        r["_label"], r["_reason"] = "ambiguous", "multi_value"
        return r

    c = cands[0]
    v, unit_note = apply_units(c.value, c.is_percent, ideal, mode)
    lab, note = grade_value(v, ideal, mode)
    r["_label"] = "ambiguous" if lab == "unparseable" else lab
    r["_reason"] = "auto_single" if lab != "unparseable" else "no_value"
    r["_v"], r["_note"], r["_v_source"], r["_unit_flag"] = v, "; ".join(x for x in (note, unit_note) if x), "auto_single", unit_note
    return r


def resolve_adjudications(rows: list[dict]) -> int:
    """Pass 2. A human supplies V; the same deterministic comparator grades it."""
    decisions = {r["question_id"]: r for r in C.read_csv(REVIEW)}
    n = 0
    for r in rows:
        if r["_label"] != "ambiguous":
            continue
        d = decisions.get(r["question_id"])
        if not d or not d.get("decision", "").strip():
            continue
        if d.get("answer_sha256") != sha(r["answer_raw"]):
            continue
        # HUMAN OVERRIDE. Used where the deterministic comparator is too strict on a
        # semantically identical answer (e.g. "CD14 Mono" vs "CD14 Monocytes"). A human,
        # not a model, sets this, it is logged, and the count is reported in the summary.
        # Preferred over loosening the global string comparator, which would silently
        # change other rows we have not inspected.
        ov = str(d.get("override_label", "")).strip().lower()
        if ov in ("correct", "incorrect"):
            r["_label"], r["_reason"] = ov, "human_override"
            r["_v"] = str(d.get("chosen_value", "")).strip()
            r["_v_source"] = "human_override"
            r["_note"] = f"OVERRIDE: {d.get('override_reason', '') or 'no reason given'}"
            n += 1
            continue
        dec = d["decision"].strip().lower()
        if dec == "missing":
            r["_label"], r["_reason"] = "missing", "adjudicated_missing"
            n += 1
            continue
        if dec != "value":
            continue
        cv = str(d.get("chosen_value", "")).strip()
        # A string-path answer (ideal is a token like CD14 or 5:1) cannot be
        # represented by a numeric candidate index, so the adjudicator supplies the
        # claimed text instead and it is compared with the same string comparator.
        if cv and not ideal_is_numeric(r["ideal"], r["eval_mode"]):
            lab, note = grade_str(cv, r["ideal"])
            r["_label"], r["_reason"] = lab, "adjudicated_string"
            r["_v"], r["_v_source"], r["_note"] = cv, "adjudicated", note
            n += 1
            continue
        val, is_pct = None, False
        if cv:
            val, _ = coerce_number(cv)
            is_pct = "%" in cv
        elif str(d.get("chosen_index", "")).strip():
            try:
                c = r["_candidates"][int(d["chosen_index"])]
                val, is_pct = c.value, c.is_percent
            except (ValueError, IndexError):
                val = None
        if val is None:
            continue
        v, unit_note = apply_units(val, is_pct, r["ideal"], r["eval_mode"])
        lab, note = grade_value(v, r["ideal"], r["eval_mode"])
        r["_label"] = "ambiguous" if lab == "unparseable" else lab
        r["_reason"] = "adjudicated"
        r["_v"], r["_v_source"] = v, "adjudicated"
        r["_note"] = "; ".join(x for x in (note, unit_note) if x)
        r["_unit_flag"] = unit_note
        n += 1
    return n


def strict_variant(rows: list[dict]) -> collections.Counter:
    """Pre-registered sensitivity: the same labels with NO unit conversion."""
    c = collections.Counter()
    for r in rows:
        if r["_label"] not in ("correct", "incorrect") or not r["_unit_flag"]:
            c[r["_label"]] += 1
            continue
        raw = r["_v"] * 100.0 if "percent->proportion" in r["_unit_flag"] else r["_v"] / 100.0
        lab, _ = grade_value(raw, r["ideal"], r["eval_mode"])
        c[lab if lab != "unparseable" else "ambiguous"] += 1
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit-review", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    rows = C.read_csv(IN)
    if not rows:
        raise SystemExit(f"no rows in {IN} - run collect.py first")
    rows = [classify(dict(r)) for r in rows]

    if args.emit_review:
        emit_review(rows)
        amb = [r for r in rows if r["_label"] == "ambiguous"]
        print(f"\n  {len(amb)} of {len(rows)} rows need adjudication")
        print("  first 5, exactly as the adjudicator will see them (no ideal shown):\n")
        for r in amb[:5]:
            print(f"    {r['question_id']}  [{r['eval_mode']}]  reason={r['_reason']}")
            print(f"      Q: {' '.join(r['question'].split())[:150]}")
            print(f"      A: {' '.join(r['answer_raw'].split())[:220]}")
            print(f"      candidates: {candidates_display(r['_candidates'])[:300]}\n")
        return

    n_adj = resolve_adjudications(rows)

    fields = [f for f in rows[0] if not f.startswith("_") and f not in ("notebook_text", "prose_text")]
    fields += ["label", "reason", "v_value", "v_source", "unit_flag", "norm_note"]
    for p in (OUT, LOG):
        if p.exists():
            p.unlink()
    log_fields = ["question_id", "eval_mode", "answer_raw", "ideal", "label", "norm_note", "unit_flag"]

    counts, by_mode = collections.Counter(), collections.defaultdict(collections.Counter)
    for r in rows:
        out = {k: v for k, v in r.items() if not k.startswith("_")}
        out.update(label=r["_label"], reason=r["_reason"], v_value=r["_v"],
                   v_source=r["_v_source"], unit_flag=r["_unit_flag"], norm_note=r["_note"])
        C.append_row(OUT, out, fields + ["notebook_text", "prose_text"])
        if r["_note"] or r["_unit_flag"]:
            C.append_row(LOG, out, log_fields)
        counts[r["_label"]] += 1
        by_mode[r["eval_mode"]][r["_label"]] += 1

    n = len(rows)
    print(f"\nlabelled {n} answers -> {OUT}   ({n_adj} resolved from adjudications)")
    print("  overall:", {k: f"{v} ({v/n:.0%})" for k, v in sorted(counts.items())})
    for mode, c in sorted(by_mode.items()):
        m = sum(c.values())
        print(f"    {mode:16} n={m:4d}", {k: f"{v}" for k, v in sorted(c.items())})

    gradeable = [r for r in rows if r["_label"] in ("correct", "incorrect")]
    caps_correct = {r["capsule_uuid"] for r in rows if r["_label"] == "correct"}
    print(f"\n  gradeable (correct+incorrect): {len(gradeable)}")
    print(f"  capsules holding >=1 correct answer: {len(caps_correct)}  <- the real n for the")
    print("     correct-answer acceptance rate; a small number here means a wide CI")
    sv = strict_variant(rows)
    print(f"  strict no-unit-conversion sensitivity: {dict(sorted(sv.items()))}")
    print(f"  normalisation log rows: {len(C.read_csv(LOG))} -> {LOG}")


def self_test() -> None:
    cases = [
        ("2e-4", "str_verifier", "0.0002", "correct"),
        ("2.2e-16", "str_verifier", "9.92E-35", "incorrect"),
        ("9.92e-35", "str_verifier", "9.92E-35", "correct"),
        ("166 genes", "str_verifier", "166", "correct"),
        ("1–50", "str_verifier", "1-50", "correct"),
        ("CCND1", "str_verifier", "CDKN1A", "incorrect"),
        ("CD14", "str_verifier", "CD14", "correct"),
        ("1.52", "range_verifier", "(1.50,1.54)", "correct"),
        ("1.55", "range_verifier", "(1.50,1.54)", "incorrect"),
        ("19,159", "str_verifier", "19159", "correct"),
    ]
    fails = 0
    print("=== grading core ===")
    for ans, mode, ideal, want in cases:
        r = classify({"question_id": "t", "ideal": ideal, "eval_mode": mode, "answer_raw": ans,
                      "answered": "1", "trace_complete": "1", "question": ""})
        got = r["_label"]
        ok = got == want
        fails += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {ans!r:14} vs {ideal!r:14} -> {got:10}"
              f"{'' if ok else '  want ' + want}")

    print("\n=== unit policy (convert only when unambiguous) ===")
    unit = [
        ("68.42%", "range_verifier", "(0.80,1.00)", "incorrect", True),   # 0.6842, outside
        ("95%", "range_verifier", "(0.90,1.00)", "correct", True),        # 0.95, inside
        ("0.95", "range_verifier", "(90,100)", "correct", True),          # 95, inside
        ("1.52", "range_verifier", "(1.50,1.54)", "correct", False),      # no conversion
    ]
    for ans, mode, ideal, want, want_flag in unit:
        r = classify({"question_id": "t", "ideal": ideal, "eval_mode": mode, "answer_raw": ans,
                      "answered": "1", "trace_complete": "1", "question": ""})
        ok = r["_label"] == want and bool(r["_unit_flag"]) == want_flag
        fails += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {ans!r:9} vs {ideal!r:14} -> {r['_label']:10}"
              f" flag={r['_unit_flag'] or '-'}")

    print("\n=== routing ===")
    route = [
        ({"answered": "0", "trace_complete": "1", "answer_raw": ""}, "missing"),
        ({"answered": "1", "trace_complete": "0", "answer_raw": "5"}, "excluded"),
        ({"answered": "1", "trace_complete": "1",
          "answer_raw": "the p-value is 0.001 (raw 0.0009561536, rounded 0.0010)"}, "ambiguous"),
        ({"answered": "1", "trace_complete": "1", "answer_raw": "no usable value here"}, "ambiguous"),
    ]
    for extra, want in route:
        r = classify({"question_id": "t", "ideal": "0.0002", "eval_mode": "str_verifier",
                      "question": "", **extra})
        ok = r["_label"] == want
        fails += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {str(extra['answer_raw'])[:46]!r:50} -> {r['_label']:10}"
              f" ({r['_reason']})")

    print("\n=== nuisance masking ===")
    for text, want_n in [("GO:0050863 gives 0.001", 1), ("COVID-19 severity 1.35", 1),
                         ("OR 1.35 (95% CI: 0.76-2.40; p=0.307)", 1)]:
        got = len(distinct(numeric_candidates(text)))
        ok = got == want_n
        fails += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {text!r:42} -> {got} candidate(s), want {want_n}")

    print("\n" + ("ALL CHECKS PASSED" if not fails else f"{fails} FAILURE(S)"))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
