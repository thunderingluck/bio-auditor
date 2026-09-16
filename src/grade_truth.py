"""Stage 2 - deterministic truth labelling. No API calls.

Reads results/answers.csv, writes results/labelled.csv and results/normalisation_log.csv.

Labels: correct | incorrect | unparseable
- str_verifier: equality after normalisation (whitespace, case, dashes, numeric coercion).
- range_verifier: ideal like "(1.50,1.54)"; correct iff the answer coerces to a number
  inside the closed interval. Non-coercible -> unparseable (never incorrect).
- An answer with no ANSWER: line -> unparseable.
Every row where normalisation did more than strip/casefold is logged for audit.
"""
from __future__ import annotations

import collections
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

IN = C.RESULTS / "answers.csv"
OUT = C.RESULTS / "labelled.csv"
LOG = C.RESULTS / "normalisation_log.csv"

NUM_RE = re.compile(r"[-+]?(?:\d+(?:,\d{3})*(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
DASHES = {"–": "-", "—": "-", "−": "-", "‒": "-", "‐": "-", "‑": "-"}


def basic_norm(s: str) -> str:
    s = str(s).strip()
    for k, v in DASHES.items():
        s = s.replace(k, v)
    s = re.sub(r"\s+", " ", s).casefold()
    s = s.strip(" .;:\"'`*")
    return s


def coerce_number(s: str) -> tuple[float | None, str]:
    """Return (value, note). note='' if the string is a bare number, else describes what was stripped."""
    t = basic_norm(s)
    if not t:
        return None, ""
    bare = t.replace(",", "")
    try:
        return float(bare), ""
    except ValueError:
        pass
    pct = bare.endswith("%")
    if pct:
        try:
            return float(bare[:-1].strip()), "stripped %"
        except ValueError:
            pass
    # A number optionally preceded by ~ / ≈ / = and followed only by units or words
    # (e.g. "~0.45", "166 genes", "1.9e-05 (adjusted)"). Anything else is not coerced.
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
        note = "" if answer.strip() == ideal.strip() else "basic normalisation"
        return "correct", note
    va, na = coerce_number(a)
    vi, ni = coerce_number(i)
    if va is not None and vi is not None:
        # numeric ideal: compare as numbers (2e-4 == 0.0002). Relative tolerance only:
        # an absolute floor would make 2.2e-16 "equal" to 9.92e-35 (this happened; bix-9-q4).
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


def grade(answer_raw: str, parsed_flag: int, eval_mode: str, ideal: str) -> tuple[str, str]:
    if not parsed_flag or answer_raw is None or str(answer_raw).strip() == "":
        return "unparseable", "no ANSWER: line"
    if eval_mode == "str_verifier":
        return grade_str(answer_raw, ideal)
    if eval_mode == "range_verifier":
        return grade_range(answer_raw, ideal)
    raise ValueError(f"unsupported eval_mode {eval_mode} - llm_verifier rows are excluded by design")


def main() -> None:
    answers = C.read_csv(IN)
    if not answers:
        raise SystemExit(f"no rows in {IN}")
    ideal_by_qid = {r["question_id"]: r["ideal"] for r in C.load_target_rows()}

    fields = list(answers[0].keys()) + ["ideal", "label", "norm_note"]
    fields = [f for f in fields if f != "response_text"] + ["response_text"]
    if OUT.exists():
        OUT.unlink()
    if LOG.exists():
        LOG.unlink()
    log_fields = ["question_id", "sample_idx", "eval_mode", "answer_raw", "ideal", "label", "norm_note"]

    counts = collections.Counter()
    by_mode = collections.defaultdict(collections.Counter)
    per_q = collections.defaultdict(collections.Counter)
    for r in answers:
        ideal = ideal_by_qid[r["question_id"]]
        label, note = grade(r["answer_raw"], int(r["answer_parsed"]), r["eval_mode"], ideal)
        row = dict(r, ideal=ideal, label=label, norm_note=note)
        C.append_row(OUT, row, fields)
        if note:
            C.append_row(LOG, row, log_fields)
        counts[label] += 1
        by_mode[r["eval_mode"]][label] += 1
        per_q[r["question_id"]][label] += 1

    n = sum(counts.values())
    print(f"labelled {n} answers -> {OUT}")
    print("overall:", {k: f"{v} ({v/n:.0%})" for k, v in sorted(counts.items())})
    for mode, c in sorted(by_mode.items()):
        m = sum(c.values())
        print(f"  {mode:15s} n={m:4d}", {k: f"{v} ({v/m:.0%})" for k, v in sorted(c.items())})
    q_any_correct = sum(1 for c in per_q.values() if c["correct"] > 0)
    q_all_correct = sum(1 for c in per_q.values() if c["correct"] == sum(c.values()))
    q_mixed = sum(1 for c in per_q.values() if 0 < c["correct"] < sum(c.values()))
    print(f"questions: {len(per_q)} total | >=1 correct: {q_any_correct} | all correct: {q_all_correct} | mixed: {q_mixed}")
    print(f"normalisation log rows: {sum(1 for _ in C.read_csv(LOG))} -> {LOG}")


if __name__ == "__main__":
    main()
