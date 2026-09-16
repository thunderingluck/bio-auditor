"""Stage 3 - judgment. Four conditions per answer, one call each.

| id | judge | content shown            | placement of M1's response        |
| A  | M1    | full reasoning + answer  | assistant turn (own prior output) |
| A' | M1    | answer only              | user turn, quoted                 |
| B  | M2    | full reasoning + answer  | user turn, quoted                 |
| C  | M2    | answer only              | user turn, quoted                 |

Invariants: the review instruction I is byte-identical everywhere; V is the value
parsed in stage 1 (labelled.csv answer_raw); M1 judges at temperature 0; M2 runs at
its only supported temperature (1) with minimal reasoning. Never put R in M2's
assistant slot.

Reads results/labelled.csv (correct/incorrect rows only), writes results/judgments.csv.
Usage: python src/judge.py [--limit 5] [--conditions A,A',B,C]
"""
from __future__ import annotations

import argparse
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

IN = C.RESULTS / "labelled.csv"
OUT = C.RESULTS / "judgments.csv"
FIELDS = [
    "question_id", "capsule_uuid", "eval_mode", "sample_idx", "source", "label", "answer_raw",
    "condition", "judge_model", "verdict", "verdict_parsed", "stop_reason",
    "input_tokens", "output_tokens", "reasoning_tokens", "cached", "cache_key", "verdict_text",
]
CONDITIONS = ["A", "A'", "B", "C"]

I = C.REVIEW_INSTRUCTION


def build_messages(condition: str, task: str, R: str, V: str) -> list[dict]:
    if condition == "A":
        return [
            {"role": "user", "content": task},
            {"role": "assistant", "content": R},
            {"role": "user", "content": I},
        ]
    if condition == "B":
        return [{"role": "user", "content":
                 "Another agent was given the following task.\n\n"
                 f"TASK:\n{task}\n\n"
                 f"THE AGENT'S FULL RESPONSE:\n{R}\n\n"
                 f"{I}"}]
    if condition in ("A'", "C"):
        return [{"role": "user", "content":
                 "Another agent was given the following task.\n\n"
                 f"TASK:\n{task}\n\n"
                 f"THE AGENT'S FINAL ANSWER:\n{V}\n\n"
                 f"{I}"}]
    raise ValueError(condition)


_WORD = re.compile(r"\b(INCORRECT|CORRECT)\b", re.IGNORECASE)


def parse_verdict(text: str) -> tuple[str, int]:
    """First non-empty line decides. Widened parser: markdown/punctuation tolerated.
    Returns (verdict, parsed_flag). Unparseable -> ('UNPARSEABLE', 0)."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        return "UNPARSEABLE", 0
    first = re.sub(r"[*_`#>\[\]\"'.:!,()-]", " ", lines[0]).strip()
    tokens = first.split()
    # The instruction asks for the verdict as the first word; honour that first
    # ("INCORRECT. The correct answer is likely ..." is INCORRECT).
    if tokens and tokens[0].upper() in ("CORRECT", "INCORRECT"):
        return tokens[0].upper(), 1
    words = _WORD.findall(first)
    if len(set(w.upper() for w in words)) == 1:
        return words[0].upper(), 1
    # "Not correct", or reasoning with no verdict word -> unparseable rather than a guess
    return "UNPARSEABLE", 0


def judge_one(row: dict, condition: str, task: str) -> dict:
    R, V = row["response_text"], row["answer_raw"]
    messages = build_messages(condition, task, R, V)
    if condition in ("A", "A'"):
        provider, request = "anthropic", {
            "model": C.M1_MODEL, "max_tokens": C.JUDGE_MAX_TOKENS,
            "temperature": C.JUDGE_TEMPERATURE, "messages": messages,
        }
    else:
        provider, request = "openai", {
            "model": C.M2_MODEL, "max_completion_tokens": C.M2_MAX_COMPLETION_TOKENS,
            "reasoning_effort": C.M2_REASONING_EFFORT, "messages": messages,
        }
    tag = {"stage": "judge", "condition": condition, "question_id": row["question_id"],
           "sample_idx": int(row["sample_idx"]), "source": row["source"]}
    data = C.cached_call(provider, request, tag=tag)
    verdict, ok = parse_verdict(data["text"])
    return {
        "question_id": row["question_id"], "capsule_uuid": row["capsule_uuid"], "eval_mode": row["eval_mode"],
        "sample_idx": row["sample_idx"], "source": row["source"], "label": row["label"], "answer_raw": V,
        "condition": condition, "judge_model": data["model"], "verdict": verdict, "verdict_parsed": ok,
        "stop_reason": data["stop_reason"],
        "input_tokens": data["usage"]["input_tokens"], "output_tokens": data["usage"]["output_tokens"],
        "reasoning_tokens": data["usage"].get("reasoning_tokens"),
        "cached": int(data["cached"]), "cache_key": data["cache_key"], "verdict_text": data["text"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only the first K labelled answers (smoke test)")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--concurrency", type=int, default=C.CONCURRENCY)
    args = ap.parse_args()
    conds = [c.strip() for c in args.conditions.split(",")]
    assert all(c in CONDITIONS for c in conds), conds

    labelled = C.read_csv(IN)
    skipped = [r for r in labelled if r["label"] not in ("correct", "incorrect")]
    rows = [r for r in labelled if r["label"] in ("correct", "incorrect")]
    rows.sort(key=lambda r: (C._qid_sort_key(r["question_id"]), r["source"], int(r["sample_idx"])))
    if args.limit:
        rows = rows[: args.limit]
    task_by_qid = {r["question_id"]: C.build_gen_prompt(r["question"]) for r in C.load_target_rows()}

    done = {(r["question_id"], int(r["sample_idx"]), r["source"], r["condition"]) for r in C.read_csv(OUT)}
    todo = [(r, c) for r in rows for c in conds if (r["question_id"], int(r["sample_idx"]), r["source"], c) not in done]
    print(f"answers={len(rows)} (skipped unparseable={len(skipped)}) conditions={conds} already_done={len(done)} todo={len(todo)}")

    n_ok = n_fail = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(judge_one, r, c, task_by_qid[r["question_id"]]): (r["question_id"], r["sample_idx"], r["source"], c)
                for r, c in todo}
        for fut in as_completed(futs):
            qid, s, src, c = futs[fut]
            try:
                out = fut.result()
            except Exception as e:
                n_fail += 1
                print(f"  FAIL {qid} s{s} {src} {c}: {type(e).__name__}: {str(e)[:160]}")
                continue
            C.append_row(OUT, out, FIELDS)
            n_ok += 1
            if n_ok <= 40 or not out["verdict_parsed"]:
                print(f"  {qid} s{s} {src[:9]:9} truth={out['label']:9} {c:2} -> {out['verdict']:11} | {out['verdict_text'][:90]!r}")
    print(f"done: ok={n_ok} fail={n_fail} -> {OUT}")


if __name__ == "__main__":
    main()
