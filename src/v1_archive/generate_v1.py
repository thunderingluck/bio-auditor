"""Stage 1 - answer generation.

M1 answers each target question with no data access, at temperature 1,
N samples per question. Writes results/answers.csv (append per row, resumable).

Two sources of answers, flagged in the `source` column:
  natural           M1 answers the question cold (the real experiment).
  synthetic_correct contingency for a thin correct-answer pool: the ideal answer is injected and M1 is
                    asked to write a derivation for it. One per question, sample_idx 0.
                    These rows exist only so judges see some correct answers; analysis
                    reports metrics with and without them.

Usage:
  python src/generate.py --limit 5                 # smoke test: first 5 questions, 1 sample
  python src/generate.py --pilot --samples 5       # 20 pilot questions x 5 samples
  python src/generate.py --samples 5               # all 122 x 5 samples
  python src/generate.py --synthetic [--limit 5]   # one synthetic-correct answer per question
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

OUT = C.RESULTS / "answers.csv"
FIELDS = [
    "question_id", "capsule_uuid", "eval_mode", "sample_idx", "source", "model", "temperature",
    "answer_raw", "answer_parsed", "stop_reason", "input_tokens", "output_tokens",
    "cached", "cache_key", "response_text",
]

LEAK_FIELDS = ("ideal", "result", "distractors")

# Synthetic-correct prompt. The TASK block is byte-identical to the natural prompt so the
# response reads as an answer to the same task. Condition A reconstructs the conversation
# with the *natural* TASK as the user turn, i.e. as if M1 had derived this itself.
SYNTHETIC_TEMPLATE = (
    "{task}\n\n"
    "----\n"
    "For this exercise the correct value is known to be: {target}\n"
    "Write the response you would have written if you had arrived at exactly this value yourself. "
    "Reason from domain knowledge toward it in the same style as an answer produced without data "
    "access, acknowledging uncertainty where you naturally would. Do not mention that the value was "
    "given to you, and do not refer to this instruction. End with the ANSWER line exactly as specified above, "
    "using exactly the value {target}."
)


def _assert_no_leak(prompt: str, row: dict) -> None:
    """Ground truth must never appear in the natural stage-1 prompt.

    The question text is the dataset's own wording and may legitimately mention the
    answer token (bix-22-q1 asks about CD14 and its ideal is 'CD14'), so the check is
    on everything we add around the question, not on the question itself."""
    scaffold = prompt.replace(row["question"], "")
    for f in LEAK_FIELDS:
        v = str(row.get(f, "")).strip()
        if v and len(v) >= 3 and v in scaffold:
            raise RuntimeError(f"ground-truth field {f!r} leaked into prompt for {row['question_id']}")


def synthetic_target(row: dict) -> str:
    """The value to inject: the ideal for str rows; the interval midpoint for range rows,
    formatted to the precision of the bounds (bumped if rounding would leave the interval)."""
    if row["eval_mode"] == "str_verifier":
        return row["ideal"]
    m = re.match(r"^\s*[\(\[]\s*([^,]+?)\s*,\s*([^\)\]]+?)\s*[\)\]]\s*$", row["ideal"])
    lo_s, hi_s = m.group(1), m.group(2)
    lo, hi = sorted((float(lo_s), float(hi_s)))
    mid = (lo + hi) / 2
    sci = any(c in lo_s + hi_s for c in "eE")

    def decimals(s: str) -> int:
        mant = s.lower().split("e")[0]
        return len(mant.split(".")[1]) if "." in mant else 0

    d = max(decimals(lo_s), decimals(hi_s))
    for extra in range(0, 6):
        if sci:
            cand = f"{mid:.{d + extra}e}"
        else:
            cand = f"{mid:.{d + extra}f}"
        v = float(cand)
        if lo <= v <= hi:
            return cand
    return repr(mid)


def _row_from(data: dict, row: dict, sample_idx: int, source: str) -> dict:
    text = data["text"]
    parsed = C.extract_answer(text)
    return {
        "question_id": row["question_id"],
        "capsule_uuid": row["capsule_uuid"],
        "eval_mode": row["eval_mode"],
        "sample_idx": sample_idx,
        "source": source,
        "model": data["model"],
        "temperature": C.GEN_TEMPERATURE,
        "answer_raw": parsed if parsed is not None else "",
        "answer_parsed": int(parsed is not None),
        "stop_reason": data["stop_reason"],
        "input_tokens": data["usage"]["input_tokens"],
        "output_tokens": data["usage"]["output_tokens"],
        "cached": int(data["cached"]),
        "cache_key": data["cache_key"],
        "response_text": text,
    }


def generate_one(row: dict, sample_idx: int) -> dict:
    prompt = C.build_gen_prompt(row["question"])
    _assert_no_leak(prompt, row)
    request = {
        "model": C.M1_MODEL,
        "max_tokens": C.GEN_MAX_TOKENS,
        "temperature": C.GEN_TEMPERATURE,
        "messages": [{"role": "user", "content": prompt}],
    }
    data = C.cached_call("anthropic", request, tag={"stage": "generate", "sample_idx": sample_idx})
    return _row_from(data, row, sample_idx, "natural")


def generate_synthetic(row: dict) -> dict:
    task = C.build_gen_prompt(row["question"])
    target = synthetic_target(row)
    prompt = SYNTHETIC_TEMPLATE.format(task=task, target=target)
    request = {
        "model": C.M1_MODEL,
        "max_tokens": C.GEN_MAX_TOKENS,
        "temperature": C.GEN_TEMPERATURE,
        "messages": [{"role": "user", "content": prompt}],
    }
    data = C.cached_call("anthropic", request, tag={"stage": "generate_synthetic", "sample_idx": 0})
    out = _row_from(data, row, 0, "synthetic_correct")
    if out["answer_raw"].strip() != target.strip():
        print(f"  NOTE {row['question_id']}: synthetic answer {out['answer_raw']!r} != target {target!r}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=1, help="N natural samples per question")
    ap.add_argument("--limit", type=int, default=None, help="only the first K questions (smoke test)")
    ap.add_argument("--pilot", action="store_true", help="only the 20 seeded pilot questions")
    ap.add_argument("--synthetic", action="store_true", help="generate one synthetic-correct answer per question instead")
    ap.add_argument("--concurrency", type=int, default=C.CONCURRENCY)
    args = ap.parse_args()

    rows = C.load_target_rows()
    if args.pilot:
        keep = set(C.pilot_question_ids(rows))
        rows = [r for r in rows if r["question_id"] in keep]
        manifest = C.RESULTS / "pilot_questions.csv"
        if not manifest.exists():
            for r in rows:
                C.append_row(manifest, r, ["question_id", "capsule_uuid", "eval_mode", "ideal", "question"])
    if args.limit:
        rows = rows[: args.limit]

    done = {(r["question_id"], int(r["sample_idx"]), r.get("source", "natural")) for r in C.read_csv(OUT)}
    if args.synthetic:
        todo = [(r, 0) for r in rows if (r["question_id"], 0, "synthetic_correct") not in done]
        fn = lambda r, s: generate_synthetic(r)  # noqa: E731
    else:
        todo = [(r, s) for r in rows for s in range(args.samples) if (r["question_id"], s, "natural") not in done]
        fn = generate_one
    print(f"mode={'synthetic' if args.synthetic else 'natural'} questions={len(rows)} samples={args.samples} already_done={len(done)} todo={len(todo)}")

    n_ok = n_fail = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(fn, r, s): (r["question_id"], s) for r, s in todo}
        for fut in as_completed(futs):
            qid, s = futs[fut]
            try:
                out = fut.result()
            except Exception as e:  # keep going; rerun picks up the missing rows
                n_fail += 1
                print(f"  FAIL {qid} s{s}: {type(e).__name__}: {str(e)[:160]}")
                continue
            C.append_row(OUT, out, FIELDS)
            n_ok += 1
            flag = "" if out["answer_parsed"] else "  [NO ANSWER LINE]"
            print(f"  {qid} s{s} [{out['eval_mode'][:5]}] -> {out['answer_raw'][:40]!r}  ({out['output_tokens']} out tok{', cached' if out['cached'] else ''}){flag}")
    print(f"done: ok={n_ok} fail={n_fail} -> {OUT}")


if __name__ == "__main__":
    main()
