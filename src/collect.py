"""Stage 1.5 - turn BixBench trajectories into a flat table. No API calls.

Reads  ~/bixbench/data/trajectories/bioauditor-s1/<qid>.{json,jsonl}
Writes results/answers.csv           one row per question
       results/work/<qid>.json       the canonical Block list for judge.py

Everything comes from d["metadata"], which carries the full HuggingFace record
(question, ideal, eval_mode, capsule_uuid). No HuggingFace call is needed.

Two things this file is responsible for catching:

* CONTAMINATION. fhda's NBEnvironment does `if self.nb_path.exists(): nbformat.read(...)`
  and workspace dirs persist across runs, so a re-run question opens the previous
  run's finished notebook and may submit without writing a line. For those rows P
  does not explain N, which is exactly the contrast H1 rests on. Detector:
  n_cells_appended < n_cells - 1 (the -1 absorbs the harness's template cell).
* EXEC TIMEOUTS. fhda caps cell execution at 1200 s; past that the tool call fails,
  aviary hands the error back as an observation and the agent continues. The
  notebook is then missing output the agent believed it had, so N and P can
  legitimately disagree. Counted, not excluded.

Usage:
  .venv/bin/python src/collect.py                 # incremental
  .venv/bin/python src/collect.py --rebuild       # start clean
  .venv/bin/python src/collect.py --list-contaminated
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C          # noqa: E402
import render as R          # noqa: E402

TRAJ_DIR = Path.home() / "bixbench" / "data" / "trajectories" / "bioauditor-s1"
CONFIG = Path.home() / "bixbench" / "bioauditor_config.yaml"
WORKSPACE = Path.home() / "bixbench" / "data" / "workspace" / "bioauditor-s1"
OUT = C.RESULTS / "answers.csv"
WORK = C.RESULTS / "work"

SCHEMA_VERSION = 2
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S | re.I)
TIMEOUT_RE = re.compile(r"Notebook execution timed out", re.I)

FIELDS = [
    "schema_version", "question_id", "capsule_uuid", "short_id", "eval_mode", "ideal",
    "question", "answered", "answer_source", "answer_raw",
    "num_actions", "n_prose_turns", "n_cells", "n_cells_appended", "n_cells_edited",
    "hit_step_cap", "trace_complete", "n_exec_timeouts",
    "notebook_chars", "notebook_truncated", "n_cells_elided", "n_cell_outputs_elided",
    "prose_chars_raw", "prose_chars_stripped",
    "model", "traj_json", "traj_jsonl", "work_json",
    "notebook_text", "prose_text",
]


def max_steps_from_config() -> int:
    """Read the cap from the harness config rather than hardcoding it twice."""
    import yaml
    cfg = yaml.safe_load(CONFIG.read_text())
    return int(cfg["rollout"]["max_steps"])


def scan_jsonl(jsonl_path: Path) -> tuple[dict | None, list[str]]:
    """Stream the file once. Return (last_transition, observation_texts).

    Observations must come from the transitions, NOT from agent_state.messages.
    The final step's observation never reaches messages - there is no next step to
    fold it into - so counting appends, edits or timeouts from messages alone
    silently misses the last action. Each transition's `next_observation` yields
    exactly one observation per step, with no double counting (transition k+1's
    `observation` is the same object as transition k's `next_observation`).
    """
    last = None
    obs: list[str] = []
    with jsonl_path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line == "null":
                continue
            try:
                o = json.loads(line)
            except Exception:
                continue
            if not isinstance(o, dict):
                continue
            last = o
            v = o.get("next_observation")
            if isinstance(v, list):
                for item in v:
                    if isinstance(item, dict) and item.get("content"):
                        obs.append(str(item["content"]))
    return last, obs


def extract_answer(agent_answer) -> tuple[str, str]:
    """Return (body, source) with source in {wrapper, bare, missing}."""
    if agent_answer is None or not str(agent_answer).strip():
        return "", "missing"
    s = str(agent_answer)
    m = ANSWER_RE.search(s)
    if m:
        return m.group(1).strip(), "wrapper"
    return s.strip(), "bare"


def collect_one(qid: str, max_steps: int, traj_dir: Path) -> dict | None:
    jf, lf = traj_dir / f"{qid}.json", traj_dir / f"{qid}.jsonl"
    if not lf.exists():
        print(f"  SKIP {qid}: no .jsonl (orphan - delete the .json so it re-runs)")
        return None
    try:
        d = json.loads(jf.read_text(encoding="utf-8", errors="replace"))
    except Exception as e:
        print(f"  SKIP {qid}: .json unparsable ({type(e).__name__}) - probably mid-write")
        return None
    last, obs_texts = scan_jsonl(lf)
    if last is None:
        print(f"  SKIP {qid}: no transitions in .jsonl")
        return None

    meta = d.get("metadata") or {}
    msgs = (last.get("next_agent_state") or last.get("agent_state") or {}).get("messages", [])
    nb = d["nb"] if isinstance(d["nb"], dict) else json.loads(d["nb"])

    raw_thoughts = [m.get("content") or "" for m in msgs
                    if m.get("role") == "assistant"
                    and (m.get("content") or "").strip().startswith("Thought:")]
    prose = R.extract_prose(msgs)
    owner = R.cell_owner(msgs)
    blocks, stats = R.build_blocks(nb, prose, owner)

    n_appended = sum(len(R.APPENDED_RE.findall(t)) for t in obs_texts)
    n_edited = sum(len(R.EDITED_RE.findall(t)) for t in obs_texts)
    n_timeouts = sum(1 for t in obs_texts if TIMEOUT_RE.search(t))

    body, source = extract_answer(d.get("agent_answer"))
    n_cells = stats["n_cells"]

    notebook_text = R.render_work(blocks, include_prose=False)
    prose_text = R.SEP.join(b.text for b in blocks if b.kind == "prose")

    WORK.mkdir(parents=True, exist_ok=True)
    work_path = WORK / f"{qid}.json"
    work_path.write_text(json.dumps({
        "question_id": qid,
        "question": meta.get("question", ""),
        "answer_raw": body,
        "blocks": [{"kind": b.kind, "index": b.index, "text": b.text} for b in blocks],
        "notebook_text": notebook_text,
        "prose_text": prose_text,
    }, ensure_ascii=False), encoding="utf-8")

    return {
        "schema_version": SCHEMA_VERSION,
        "question_id": qid,
        "capsule_uuid": meta.get("capsule_uuid", ""),
        "short_id": meta.get("short_id", ""),
        "eval_mode": meta.get("eval_mode", ""),
        "ideal": meta.get("ideal", d.get("ideal_answer", "")),
        "question": meta.get("question", ""),
        "answered": int(source != "missing"),
        "answer_source": source,
        "answer_raw": body,
        "num_actions": d.get("num_actions", 0),
        "n_prose_turns": len(prose),
        "n_cells": n_cells,
        "n_cells_appended": n_appended,
        "n_cells_edited": n_edited,
        "hit_step_cap": int(d.get("num_actions", 0) >= max_steps),
        "trace_complete": int(n_appended >= n_cells - 1),
        "n_exec_timeouts": n_timeouts,
        "notebook_chars": len(notebook_text),
        "notebook_truncated": stats["notebook_truncated"],
        "n_cells_elided": stats["n_cells_elided"],
        "n_cell_outputs_elided": stats["n_cell_outputs_elided"],
        "prose_chars_raw": sum(len(t) for t in raw_thoughts),
        "prose_chars_stripped": sum(len(p) for p in prose),
        "model": d.get("model", ""),
        "traj_json": str(jf),
        "traj_jsonl": str(lf),
        "work_json": str(work_path),
        "notebook_text": notebook_text,
        "prose_text": prose_text,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--list-contaminated", action="store_true")
    ap.add_argument("--traj-dir", default=str(TRAJ_DIR))
    args = ap.parse_args()

    traj_dir = Path(args.traj_dir)
    max_steps = max_steps_from_config()

    if args.rebuild and OUT.exists():
        OUT.unlink()
    existing = C.read_csv(OUT)
    if existing and existing[0].get("schema_version") != str(SCHEMA_VERSION):
        raise SystemExit(f"{OUT} is not schema_version={SCHEMA_VERSION} (v1 leftovers?). "
                         f"Move it aside or pass --rebuild.")
    done = {r["question_id"] for r in existing}

    qids = sorted(p.stem for p in traj_dir.glob("*.json"))
    rows = list(existing)
    for qid in qids:
        if qid in done:
            continue
        row = collect_one(qid, max_steps, traj_dir)
        if row is None:
            continue
        C.append_row(OUT, row, FIELDS)
        rows.append(row)

    if args.list_contaminated:
        bad = [r for r in rows if str(r["trace_complete"]) == "0"]
        print(" ".join(r["question_id"] for r in bad) if bad else "(none)")
        return

    def n(key, pred=lambda v: str(v) == "1"):
        return sum(1 for r in rows if pred(r[key]))

    print(f"\ncollected {len(rows)} / 122   (max_steps={max_steps} from {CONFIG.name})")
    print(f"  answered            : {n('answered')}   missing: {len(rows) - n('answered')}")
    print(f"  hit the step cap    : {n('hit_step_cap')}")
    print(f"  exec timeouts       : {sum(int(r['n_exec_timeouts']) for r in rows)} "
          f"across {n('n_exec_timeouts', lambda v: int(v) > 0)} trajectories")
    print(f"  notebook truncated  : {n('notebook_truncated')}")
    bad = [r for r in rows if str(r["trace_complete"]) == "0"]
    print(f"  CONTAMINATED traces : {len(bad)}")
    for r in bad:
        print(f"      {r['question_id']:12} actions={r['num_actions']:>3} "
              f"cells={r['n_cells']:>3} appended={r['n_cells_appended']:>3}")

    def q(key):
        v = sorted(int(r[key]) for r in rows)
        if not v:
            return "-"
        return f"min {v[0]:,} / med {v[len(v)//2]:,} / p90 {v[int(len(v)*0.9)]:,} / max {v[-1]:,}"
    print(f"  notebook chars      : {q('notebook_chars')}")
    print(f"  prose chars (strip) : {q('prose_chars_stripped')}")
    raw = sum(int(r["prose_chars_raw"]) for r in rows)
    strp = sum(int(r["prose_chars_stripped"]) for r in rows)
    if raw:
        print(f"  harness boilerplate : {(raw - strp) / raw:.1%} of raw prose removed")
    print(f"\n  -> {OUT}\n  -> {WORK}/<qid>.json")


if __name__ == "__main__":
    main()
