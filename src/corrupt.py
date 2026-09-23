"""Build the corrupted answer set - a manipulation check for stage 3.

THE QUESTION THIS WAS BUILT TO ANSWER. The main grid shows judges accepting 90-100%
of wrong answers once they can see a notebook. Two readings survive that: either the
judges read the evidence and are bad at it, or they do not read it at all and are
cued by the presence of competent-looking code.

THE ANSWER IT ACTUALLY GAVE, which is not the one this file was written expecting.
The judges read it. 35 of the 37 ACCEPT verdicts on this set quote the original
value - a number deleted from the answer, surviving only in the notebook - so they
located the contradiction and then declined to act on it, calling it a typo or a
formatting artefact. An ACCEPT here is leniency, not blindness. The earlier claim
in this docstring, that sustained acceptance "is not looking", is contradicted by
the data this set produced; analyze.py now measures the quote rate rather than
assuming either reading.

A LIMITATION THIS EXPOSED IN THE RULE BELOW. y = -10x flips a sign and shifts a
decimal, which is what a transcription slip looks like, so the judges' typo reading
is reasonable and the absolute acceptance rate should not be read as error. The
comparison across conditions is unaffected - identical bytes, identical available
excuse - so a gap between conditions is a gap in willingness to forgive. Measuring
an absolute rate would need a corruption no clerical story can explain: a wrong gene
name, an inverted conclusion. This run does not contain one.

REPORTED SEPARATELY, ALWAYS. These rows are constructed, not observed. Pooling them
with the main grid would inflate the wrong-answer pool with items whose difficulty
was chosen by us. `corrupted=1` travels with every row and analyze.py keeps them in
their own section.

THE CORRUPTION RULE, in one line:

    y = -10x

applied to the numeric literal as it appears in the answer text, with the
surrounding prose and the number's notation left alone.

Why that rule and not something gentler. Two obvious alternatives both leave the
judge an honest escape route, which would make an ACCEPT uninformative:

  - multiplying a proportion by 100 (0.9564 -> 95.64) reads as a percent
    conversion, a unit difference rather than a contradiction;
  - flipping the sign alone (-4.96 -> 4.96) reads as a reference-direction
    convention on a log fold change, which is a real ambiguity in this data.

Negating AND shifting an order of magnitude is simultaneously impossible under
both readings, so no unit or convention story explains it. The cost is that a
REJECT cannot be attributed to magnitude or to sign separately; this set measures
whether the evidence is consulted at all, not which cue does the work.

WHAT THIS SET CANNOT CONTROL. Some answers show their own derivation - "difference
= 0.4900 - 0.0613 = 0.4287". Corrupting the result leaves arithmetic that no longer
works, so those rows are detectable from the answer text alone, without ever
reading the notebook. Rewriting the derivation would change far more than the
value, so we do not. This is measured rather than hidden: the `-floor` conditions
see the answer and nothing else, so their acceptance rate on this set IS the
answer-only detection baseline, and the notebook's contribution is
`-strip` minus `-floor`. `answer_alone_may_reveal` flags the affected rows.

ELIGIBILITY. Every criterion is a requirement, not a preference:

  label == correct                  the row must start out true, or "contradicts
                                    its own notebook" is not what we created
  trace_complete == 1               the agent wrote the notebook it submitted
  notebook_truncated == 0           the judge sees the whole notebook
  n_exec_timeouts == 0              no cell the agent believed it ran is missing
  v_value numeric and non-zero      -10x is undefined as a corruption of zero
  literal present in answer_raw     there is something to substitute
  literal present in notebook_text  THE CONTRADICTION IS VISIBLE. Without this the
                                    judge has nothing to check the answer against
                                    and a REJECT would only mean "looks odd".

RANKING. Eligible rows are sorted by notebook_chars ascending and the first N are
taken. A shorter notebook is one a judge can read end to end inside its context
without effort, so failing to notice a contradiction in it is the least excusable.
Ties break on question_id. The ordering is fully determined by the data, so this
script takes no seed and repeats exactly.

Usage:
  .venv/bin/python src/corrupt.py            # build results/corrupted.csv
  .venv/bin/python src/corrupt.py --n 20
  .venv/bin/python src/corrupt.py --dry-run  # print the audit table, write nothing
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import grade_truth as G  # noqa: E402

IN = C.RESULTS / "labelled.csv"
OUT = C.RESULTS / "corrupted.csv"

CORRUPTION_RULE = "y = -10x"
DEFAULT_N = 20

FIELDS = [
    "question_id", "capsule_uuid", "eval_mode", "ideal",
    "label", "original_label",
    "answer_raw", "original_answer_raw",
    "v_value", "original_v_value",
    "corrupted_literal", "original_literal",
    "corruption_rule", "n_replacements", "n_other_numbers_in_answer",
    "answer_alone_may_reveal",
    "original_value_occurrences_in_notebook", "corrupted_value_appears_in_notebook",
    "notebook_chars", "n_cells", "rank", "work_json",
    "answer_sha256_original", "answer_sha256_corrupted",
]


# ------------------------------------------------------------------ numbers --
def as_float(s) -> float | None:
    try:
        return float(str(s).replace(",", ""))
    except (TypeError, ValueError):
        return None


def same_value(a: float | None, b: float | None) -> bool:
    """Relative comparison, matching the grader's rel_tol discipline. An absolute
    floor here would make 2.2e-16 and 9.9e-35 the same token."""
    if a is None or b is None:
        return False
    return abs(a - b) <= abs(b) * 1e-9


def corrupt_literal(lit: str) -> str:
    """Apply y = -10x to a numeric literal, keeping its notation.

    Scientific notation is rewritten in place - negate the mantissa, add one to the
    exponent, keep the exponent's original field width - because round-tripping
    7.29e+07 through a float would emit 729000000.0 and change how the answer reads
    as well as what it claims. Plain decimals go through Decimal so no binary
    floating-point noise (0.31500000000000006) can enter the prompt.
    """
    raw = lit.replace(",", "")
    if "e" in raw or "E" in raw:
        marker = "e" if "e" in raw else "E"
        mant, exp_field = raw.split(marker)
        negative = mant.startswith("-")
        mant_abs = mant.lstrip("+-")
        new_mant = mant_abs if negative else "-" + mant_abs
        exp_digits = len(exp_field.lstrip("+-"))
        new_exp = int(exp_field) + 1
        sign = "-" if new_exp < 0 else "+"
        return f"{new_mant}{marker}{sign}{abs(new_exp):0{exp_digits}d}"

    y = -(Decimal(raw) * 10)
    out = format(y, "f")            # never exponential; normalize() would go to 8.2E+5
    if "." in out:
        out = out.rstrip("0").rstrip(".")
    if "," in lit:                  # 72,900,000 -> -729,000,000, not -729000000
        neg = out.startswith("-")
        whole, _, frac = out.lstrip("-").partition(".")
        out = ("-" if neg else "") + f"{int(whole):,}" + (f".{frac}" if frac else "")
    return out


def substitute(answer: str, target: float) -> tuple[str, int, int]:
    """Apply the rule to every token that restates the claimed value.

    Token-level, not str.replace: replacing the substring "1" would hit the 1 inside
    21.81.

    Two details that matter more than they look.

    MATCH ON MAGNITUDE. An answer often restates its own value unsigned - "the
    coefficient is -0.0750 ... a decrease of 0.0750". Corrupting only the signed
    copy would leave the original magnitude sitting in the prose, so the answer
    would contradict itself about a number rather than contradict the notebook.
    Any token with |t| == |x| is therefore rewritten.

    CORRUPT EACH TOKEN IN ITS OWN NOTATION. The same value can appear twice in
    different forms - "7.29e+07 (72,900,000)". Substituting one rendering for both
    would change how the answer is written as well as what it claims. Each token
    gets y = -10t computed and formatted from itself, so notation survives and only
    the number moves.

    Returns the new answer, how many tokens were rewritten, and how many unrelated
    numbers remain in the prose.
    """
    out, pos, n_rep, n_other = [], 0, 0, 0
    for m in G.NUM_RE.finditer(answer):
        t = as_float(m.group())
        if t is not None and same_value(abs(t), abs(target)):
            out.append(answer[pos:m.start()])
            out.append(corrupt_literal(m.group()))
            pos = m.end()
            n_rep += 1
        else:
            n_other += 1
    out.append(answer[pos:])
    return "".join(out), n_rep, n_other


def count_value(text: str, target: float) -> int:
    return sum(1 for m in G.NUM_RE.finditer(text or "")
               if same_value(as_float(m.group()), target))


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------- selection --
def eligible(r: dict) -> tuple[bool, str]:
    if r["label"] != "correct":
        return False, "not correct"
    if str(r.get("trace_complete")) != "1":
        return False, "trace incomplete"
    if str(r.get("notebook_truncated")) not in ("0", "False", ""):
        return False, "notebook truncated"
    if str(r.get("n_exec_timeouts")) not in ("0", ""):
        return False, "exec timeout"
    v = as_float(r.get("v_value"))
    if v is None:
        return False, "v_value not numeric"
    if v == 0:
        return False, "v_value is zero (-10x is not a corruption of 0)"
    if not count_value(r.get("answer_raw", ""), v):
        return False, "value literal absent from the answer text"
    if not count_value(r.get("notebook_text", ""), v):
        return False, "value absent from the notebook (contradiction would be invisible)"
    return True, ""


def build(rows: list[dict], n: int) -> tuple[list[dict], list[tuple[str, str]]]:
    keep, dropped = [], []
    for r in rows:
        ok, why = eligible(r)
        if not ok:
            if r["label"] == "correct":
                dropped.append((r["question_id"], why))
            continue
        keep.append(r)

    keep.sort(key=lambda r: (int(r.get("notebook_chars") or 0),
                             C._qid_sort_key(r["question_id"])))
    chosen = keep[:n]

    out = []
    for rank, r in enumerate(chosen, 1):
        v = as_float(r["v_value"])
        ans = r["answer_raw"]
        lit = next(m.group() for m in G.NUM_RE.finditer(ans) if same_value(as_float(m.group()), v))
        new_lit = corrupt_literal(lit)
        new_ans, n_rep, n_other = substitute(ans, v)
        new_v = as_float(new_lit)

        # --- the guarantees, asserted rather than assumed ---------------------
        assert new_v is not None, f"{r['question_id']}: corrupted literal does not parse"
        assert n_rep >= 1, f"{r['question_id']}: nothing was substituted"
        assert new_ans != ans, f"{r['question_id']}: answer unchanged"
        assert same_value(new_v, -10 * v), \
            f"{r['question_id']}: {new_lit} is not -10x of {lit}"
        verdict, why = G.grade_value(new_v, r["ideal"], r["eval_mode"])
        assert verdict == "incorrect", \
            f"{r['question_id']}: corrupted value {new_v} still grades {verdict} ({why})"

        out.append({
            "question_id": r["question_id"], "capsule_uuid": r["capsule_uuid"],
            "eval_mode": r["eval_mode"], "ideal": r["ideal"],
            "label": "incorrect", "original_label": "correct",
            "answer_raw": new_ans, "original_answer_raw": ans,
            "v_value": new_v, "original_v_value": v,
            "corrupted_literal": new_lit, "original_literal": lit,
            "corruption_rule": CORRUPTION_RULE,
            "n_replacements": n_rep, "n_other_numbers_in_answer": n_other,
            "answer_alone_may_reveal": int(n_other > 0),
            "original_value_occurrences_in_notebook": count_value(r.get("notebook_text", ""), v),
            "corrupted_value_appears_in_notebook": count_value(r.get("notebook_text", ""), new_v),
            "notebook_chars": int(r.get("notebook_chars") or 0),
            "n_cells": int(r.get("n_cells") or 0),
            "rank": rank, "work_json": r["work_json"],
            "answer_sha256_original": sha(ans), "answer_sha256_corrupted": sha(new_ans),
        })
    return out, dropped


# --------------------------------------------------------------------- main --
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=DEFAULT_N)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = C.read_csv(IN)
    if not rows:
        raise SystemExit(f"no rows in {IN} - run grade_truth.py first")

    out, dropped = build(rows, args.n)
    n_correct = sum(1 for r in rows if r["label"] == "correct")
    print(f"{n_correct} correct answers -> {n_correct - len(dropped)} eligible -> {len(out)} selected")
    print(f"rule: {CORRUPTION_RULE}   (ranked by notebook_chars ascending)")
    if len(out) < args.n:
        print(f"  WARNING: asked for {args.n}, only {len(out)} rows are eligible")

    print("\ncorrect answers that were NOT eligible:")
    for qid, why in sorted(dropped, key=lambda z: C._qid_sort_key(z[0])):
        print(f"  {qid:12} {why}")

    print(f"\n{'#':>2} {'question_id':12} {'eval_mode':15} {'ideal':20} "
          f"{'was':>12} -> {'now claims':>13} {'subs':>4} {'inNB':>4} {'nbChars':>8}")
    print("-" * 104)
    for r in out:
        print(f"{r['rank']:>2} {r['question_id']:12} {r['eval_mode']:15} {str(r['ideal'])[:20]:20} "
              f"{r['original_literal']:>12} -> {r['corrupted_literal']:>13} "
              f"{r['n_replacements']:>4} {r['original_value_occurrences_in_notebook']:>4} "
              f"{r['notebook_chars']:>8,}")

    leaks = [r for r in out if r["corrupted_value_appears_in_notebook"]]
    print(f"\ncorrupted value that happens to appear in its own notebook: {len(leaks)} "
          f"(these are weaker contradictions)")
    for r in leaks:
        print(f"  {r['question_id']} claims {r['corrupted_literal']}, which the notebook does contain")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    if OUT.exists():
        OUT.unlink()
    for r in out:
        C.append_row(OUT, r, FIELDS)
    print(f"\nwrote {len(out)} rows -> {OUT}")
    print("next: .venv/bin/python src/judge.py --corrupted --limit 3")


if __name__ == "__main__":
    main()
