"""Stage 3 - judgment. Seven conditions, one call per (answer, condition).

| id       | judge | sees        | the agent's work is tagged as  |
|----------|-------|-------------|--------------------------------|
| A-self   | M1    | N + P + V   | assistant - its own prior turn |
| A-full   | M1    | N + P + V   | user - quoted                  |
| A-strip  | M1    | N + V       | user - quoted                  |
| A-floor  | M1    | V only      | user - quoted                  |
| B-full   | M2    | N + P + V   | user - quoted                  |
| B-strip  | M2    | N + V       | user - quoted                  |
| B-floor  | M2    | V only      | user - quoted                  |

A-self vs A-full is the primary contrast (H1). The two carry the SAME underlying
P + N + V content and differ in conversational framing: own prior assistant turn
versus quoted work from another agent. That is both a message-structure difference
and a textual-attribution difference, and both are part of the manipulation. Do
not describe it as "only the role differs".

V is the VERBATIM answer text the agent submitted, not an extracted scalar. The
agent never wrote a bare number, and putting one into A-self would tell the model
it said something it did not. The cost is that the answer sentence carries some
argument into -strip and -floor; concede that in the paper.

Never place the agent's work in M2's assistant slot. M2 did not write it. Genuine
self-review exists only for M1.

Usage:
  .venv/bin/python src/judge.py --assert-only     # spends nothing; run this first
  .venv/bin/python src/judge.py --limit 5
  .venv/bin/python src/judge.py
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import render as R  # noqa: E402

IN = C.RESULTS / "labelled.csv"
OUT = C.RESULTS / "judgments.csv"
EXAMPLE = C.RESULTS / "payload_example.txt"

CONDITIONS = ["A-self", "A-full", "A-strip", "A-floor", "B-full", "B-strip", "B-floor"]
JUDGE_OF = {"A-self": "M1", "A-full": "M1", "A-strip": "M1", "A-floor": "M1",
            "B-full": "M2", "B-strip": "M2", "B-floor": "M2"}

I = C.REVIEW_INSTRUCTION
assert hashlib.sha256(I.encode("utf-8")).hexdigest() == C.REVIEW_INSTRUCTION_SHA256, \
    "the review instruction changed - every condition must share it byte for byte"

WRAPPER_WORK = (
    "Another agent was given the following task.\n\n"
    "TASK:\n{q}\n\n"
    "THE AGENT'S WORK:\n{w}\n\n"
    "THE AGENT'S FINAL ANSWER:\n{v}\n\n"
    "{i}"
)
WRAPPER_FLOOR = (
    "Another agent was given the following task.\n\n"
    "TASK:\n{q}\n\n"
    "THE AGENT'S FINAL ANSWER:\n{v}\n\n"
    "{i}"
)

FIELDS = [
    "question_id", "capsule_uuid", "eval_mode", "label", "v_value", "answer_raw",
    "condition", "judge_model", "verdict", "verdict_parsed", "confidence",
    "confidence_parsed", "justification", "stop_reason",
    "input_tokens", "output_tokens", "reasoning_tokens", "cached", "cache_key", "verdict_text",
]


def load_work(row: dict) -> dict:
    w = json.loads(Path(row["work_json"]).read_text(encoding="utf-8"))
    w["blocks"] = [R.Block(b["kind"], b["index"], b["text"]) for b in w["blocks"]]
    return w


def m1_request(messages: list[dict]) -> tuple[str, dict]:
    return "anthropic", {"model": C.M1_MODEL, "max_tokens": C.JUDGE_MAX_TOKENS,
                         "temperature": C.JUDGE_TEMPERATURE, "messages": messages}


def m2_request(messages: list[dict]) -> tuple[str, dict]:
    return "openai", {"model": C.M2_MODEL, "max_completion_tokens": C.M2_MAX_COMPLETION_TOKENS,
                      "reasoning_effort": C.M2_REASONING_EFFORT, "messages": messages}


def build_all_payloads(work: dict) -> dict[str, tuple[str, dict]]:
    """All seven together, so the cross-condition assertions can run per row."""
    q, v = work["question"], work["answer_raw"]
    blocks = work["blocks"]
    work_full = R.render_work(blocks, include_prose=True)
    work_strip = R.render_work(blocks, include_prose=False)

    full_user = WRAPPER_WORK.format(q=q, w=work_full, v=v, i=I)
    strip_user = WRAPPER_WORK.format(q=q, w=work_strip, v=v, i=I)
    floor_user = WRAPPER_FLOOR.format(q=q, v=v, i=I)
    self_assistant = f"{work_full}\n\nANSWER: {v}"

    return {
        "A-self": m1_request([{"role": "user", "content": q},
                              {"role": "assistant", "content": self_assistant},
                              {"role": "user", "content": I}]),
        "A-full": m1_request([{"role": "user", "content": full_user}]),
        "A-strip": m1_request([{"role": "user", "content": strip_user}]),
        "A-floor": m1_request([{"role": "user", "content": floor_user}]),
        "B-full": m2_request([{"role": "user", "content": full_user}]),
        "B-strip": m2_request([{"role": "user", "content": strip_user}]),
        "B-floor": m2_request([{"role": "user", "content": floor_user}]),
    }


def assert_invariants(qid: str, work: dict, P: dict[str, tuple[str, dict]]) -> None:
    """Fail loudly. These are the guarantees the whole design rests on."""
    def user(c):
        return P[c][1]["messages"][0]["content"]

    blocks = work["blocks"]
    work_full = R.render_work(blocks, True)
    work_strip = R.render_work(blocks, False)
    v = work["answer_raw"]

    assert user("A-full") == user("B-full"), f"{qid}: A-full != B-full byte-identical"
    assert user("A-strip") == user("B-strip"), f"{qid}: A-strip != B-strip byte-identical"
    assert user("A-floor") == user("B-floor"), f"{qid}: A-floor != B-floor byte-identical"

    # -full differs from -strip ONLY by removal of the prose blocks
    assert R.strip_blocks(blocks) == [b for b in blocks if b.kind != "prose"], f"{qid}: strip_blocks"
    assert work_full == R.SEP.join(b.text for b in blocks), f"{qid}: work_full != join(blocks)"
    assert work_strip == R.SEP.join(b.text for b in R.strip_blocks(blocks)), f"{qid}: work_strip"
    for b in blocks:
        if b.kind == "prose" and len(b.text) > 40:
            assert b.text not in work_strip, f"{qid}: prose block {b.index} leaked into -strip"

    # A-self carries the same P + N + V content as A-full
    assert P["A-self"][1]["messages"][1]["content"] == f"{work_full}\n\nANSWER: {v}", f"{qid}: A-self body"
    assert work_full in user("A-full"), f"{qid}: work_full missing from A-full"
    assert v in user("A-full"), f"{qid}: answer missing from A-full"
    assert P["A-self"][1]["messages"][0]["content"] == work["question"], f"{qid}: A-self question turn"

    # I is terminal and identical everywhere
    assert P["A-self"][1]["messages"][2]["content"] == I, f"{qid}: A-self instruction turn"
    for c in CONDITIONS:
        if c == "A-self":
            continue
        assert P[c][1]["messages"][-1]["content"].endswith(I), f"{qid}: {c} does not end with I"

    # models and sampling pinned
    for c in ("A-self", "A-full", "A-strip", "A-floor"):
        assert P[c][1]["model"] == C.M1_MODEL and P[c][1]["temperature"] == 0, f"{qid}: {c} M1 pin"
    for c in ("B-full", "B-strip", "B-floor"):
        r = P[c][1]
        assert r["model"] == C.M2_MODEL and "temperature" not in r \
            and r["reasoning_effort"] == C.M2_REASONING_EFFORT, f"{qid}: {c} M2 pin"


# ------------------------------------------------------------------ parsing --
WORD_RE = re.compile(r"\b(ACCEPT|REJECT)\b", re.I)
CONF_RE = re.compile(r"(\d{1,3})\s*%?")


def parse_verdict(text: str) -> tuple[str, int | None, str, int, int]:
    """Verdict and confidence parse INDEPENDENTLY.

    A missing confidence must never turn a clean ACCEPT into an unparseable row,
    otherwise the unparseable rate mixes two different failures. Widen this parser
    if it misses; never touch I, which would break the byte-identical invariant.
    """
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    verdict, v_ok = "UNPARSEABLE", 0
    if lines:
        first = re.sub(r"[*_`#>\[\]\"'.:!,()-]", " ", lines[0]).strip()
        tokens = first.split()
        if tokens and tokens[0].upper() in ("ACCEPT", "REJECT"):
            verdict, v_ok = tokens[0].upper(), 1
        else:
            found = {w.upper() for w in WORD_RE.findall(" ".join(lines[:3]))}
            if len(found) == 1:
                verdict, v_ok = found.pop(), 1

    confidence, c_ok = None, 0
    for ln in lines[1:3]:
        m = CONF_RE.search(re.sub(r"confidence", " ", ln, flags=re.I))
        if m:
            val = int(m.group(1))
            if 0 <= val <= 100:
                confidence, c_ok = val, 1
                break

    justification = " ".join(lines[2:]) if len(lines) > 2 else (lines[-1] if lines else "")
    return verdict, confidence, justification, v_ok, c_ok


def judge_one(row: dict, work: dict, condition: str) -> dict:
    provider, request = build_all_payloads(work)[condition]
    tag = {"stage": "judge_v2", "condition": condition, "question_id": row["question_id"]}
    data = C.cached_call(provider, request, tag=tag)
    verdict, conf, just, v_ok, c_ok = parse_verdict(data["text"])
    return {
        "question_id": row["question_id"], "capsule_uuid": row["capsule_uuid"],
        "eval_mode": row["eval_mode"], "label": row["label"], "v_value": row.get("v_value", ""),
        "answer_raw": row["answer_raw"], "condition": condition,
        "judge_model": data["model"], "verdict": verdict, "verdict_parsed": v_ok,
        "confidence": "" if conf is None else conf, "confidence_parsed": c_ok,
        "justification": just, "stop_reason": data["stop_reason"],
        "input_tokens": data["usage"]["input_tokens"], "output_tokens": data["usage"]["output_tokens"],
        "reasoning_tokens": data["usage"].get("reasoning_tokens"),
        "cached": int(data["cached"]), "cache_key": data["cache_key"], "verdict_text": data["text"],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assert-only", action="store_true", help="verify invariants, spend nothing")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--concurrency", type=int, default=C.CONCURRENCY)
    args = ap.parse_args()
    conds = [c.strip() for c in args.conditions.split(",")]
    assert all(c in CONDITIONS for c in conds), conds

    labelled = C.read_csv(IN)
    if not labelled:
        raise SystemExit(f"no rows in {IN} - run grade_truth.py first")
    by_label = dict(collections.Counter(r["label"] for r in labelled))
    gradeable = sorted((r for r in labelled if r["label"] in ("correct", "incorrect")),
                       key=lambda r: C._qid_sort_key(r["question_id"]))

    if args.assert_only:
        n = 0
        for r in labelled:                      # assert over ALL rows, not just gradeable
            wp = Path(r["work_json"])
            if not wp.exists():
                continue
            work = load_work(r)
            assert_invariants(r["question_id"], work, build_all_payloads(work))
            n += 1
        print(f"  all invariants hold on {n} rows")
        if not gradeable:
            print("  (no gradeable rows yet, so no example payload written)")
            return
        r = gradeable[0]
        work = load_work(r)
        P = build_all_payloads(work)
        with EXAMPLE.open("w", encoding="utf-8") as f:
            f.write(f"question_id: {r['question_id']}   truth: {r['label']}\n")
            for c in CONDITIONS:
                f.write(f"\n{'=' * 78}\n=== {c}  ({JUDGE_OF[c]})\n{'=' * 78}\n")
                for m in P[c][1]["messages"]:
                    f.write(f"\n--- role={m['role']} ---\n{m['content']}\n")
        print(f"  wrote all seven payloads for {r['question_id']} -> {EXAMPLE}")
        for c in CONDITIONS:
            size = sum(len(m["content"]) for m in P[c][1]["messages"])
            print(f"     {c:8} {JUDGE_OF[c]}  {size:>7,} chars")
        return

    if args.limit:
        gradeable = gradeable[: args.limit]
    done = {(r["question_id"], r["condition"]) for r in C.read_csv(OUT)}
    todo = [(r, c) for r in gradeable for c in conds if (r["question_id"], c) not in done]
    print(f"gradeable={len(gradeable)} conditions={len(conds)} already_done={len(done)} todo={len(todo)}")
    print(f"labels in labelled.csv: {by_label}")

    n_ok = n_fail = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {}
        for r, c in todo:
            work = load_work(r)
            assert_invariants(r["question_id"], work, build_all_payloads(work))
            futs[ex.submit(judge_one, r, work, c)] = (r["question_id"], c, r["label"])
        for fut in as_completed(futs):
            qid, c, lab = futs[fut]
            try:
                out = fut.result()
            except Exception as e:
                n_fail += 1
                print(f"  FAIL {qid} {c}: {type(e).__name__}: {str(e)[:150]}")
                continue
            C.append_row(OUT, out, FIELDS)
            n_ok += 1
            if n_ok <= 60 or not out["verdict_parsed"]:
                conf = out["confidence"] if out["confidence"] != "" else "--"
                print(f"  {qid:12} truth={lab:9} {c:8} -> {out['verdict']:11} conf={str(conf):>4} | "
                      f"{' '.join(out['justification'].split())[:80]}")
    print(f"done: ok={n_ok} fail={n_fail} -> {OUT}")


if __name__ == "__main__":
    main()
