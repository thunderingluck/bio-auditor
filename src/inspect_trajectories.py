"""Archived analysis: can BixBench's published raw_trajectory_data.csv replace stage 1?

Conclusion (2026-09-16): no. Findings it reproduces:
- The CSV is one row per *capsule run* (6,048 rows; 2,016 open-format), not per question.
  agent_answer is a JSON dict {q1: ..., qN: ...}; nb is a Jupyter notebook.
- The notebooks contain only code cells + execution outputs. Zero markdown cells across
  all open-format notebooks and no message log, so there is no natural-language reasoning
  trace - only actions.
- The trajectories answered an older question set. Under index mapping (CSV q3 -> HF
  bix-N-q3) the CSV's own ideal matches HF v1.5 only ~35% of the time; wording changed
  and some answer keys changed (e.g. bix-43-q3 679 -> 677). Only ~10 of the 122 str/range
  questions align reliably for the Claude no-image run.
- Author models are Claude 3.5 Sonnet and GPT-4o (per the BixBench paper), so condition A
  would not be genuine self-review for our M1.

Usage: python src/inspect_trajectories.py [path/to/raw_trajectory_data.csv]
Streams the 1.46 GB file; takes a few minutes.
"""
from __future__ import annotations

import ast
import collections
import csv
import difflib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

csv.field_size_limit(2**31 - 1)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip().casefold()


def main(path: Path) -> None:
    from datasets import load_dataset

    d = load_dataset("futurehouse/BixBench")["train"]
    hf_by_cap: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    for r in d:
        hf_by_cap[r["capsule_uuid"]][r["question_id"].split("-")[-1]] = r
    target = {r["question_id"] for r in d if r["eval_mode"] in ("str_verifier", "range_verifier")}

    n_rows = n_open = md_cells = code_cells = 0
    runs = collections.Counter()
    csv_caps: set[str] = set()
    idx_pairs = idx_same_ideal = idx_similar_text = 0
    reliable: dict[str, set[str]] = collections.defaultdict(set)
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            n_rows += 1
            runs[row["run_name"]] += 1
            cap = row["problem_id"].replace("CapsuleFolder-", "")
            csv_caps.add(cap)
            if row["question_format"] != "open":
                continue
            n_open += 1
            nb = row["nb"]
            md_cells += nb.count("'cell_type': 'markdown'")
            code_cells += nb.count("'cell_type': 'code'")
            if cap not in hf_by_cap:
                continue
            try:
                qs = ast.literal_eval(row["mcq_question"])
                ideals = ast.literal_eval(row["ideal_answer"])
                ans = json.loads(row["agent_answer"]) if row["agent_answer"].strip() else {}
            except Exception:
                continue
            for qk, qtext in qs.items():
                hf = hf_by_cap[cap].get(qk)
                if hf is None:
                    continue
                idx_pairs += 1
                same_ideal = norm(ideals.get(qk, "")) == norm(hf["ideal"])
                ratio = difflib.SequenceMatcher(None, norm(qtext), norm(hf["question"])).ratio()
                idx_same_ideal += same_ideal
                idx_similar_text += ratio >= 0.8
                if same_ideal and ratio >= 0.6 and hf["question_id"] in target and str(ans.get(qk, "")).strip():
                    reliable[row["run_name"]].add(hf["question_id"])

    print(f"rows={n_rows} open-format={n_open} unique capsules={len(csv_caps)} (in HF: {len(csv_caps & set(hf_by_cap))})")
    print("runs:", dict(runs))
    print(f"open notebooks: code cells={code_cells} markdown cells={md_cells}")
    print(f"index-mapped question pairs={idx_pairs} same ideal={idx_same_ideal} ({idx_same_ideal/max(idx_pairs,1):.0%}) "
          f"text similarity>=0.8={idx_similar_text} ({idx_similar_text/max(idx_pairs,1):.0%})")
    print("reliably aligned str/range questions (same ideal, text sim>=0.6, non-empty answer):")
    for run, s in sorted(reliable.items()):
        print(f"  {run:28s} {len(s):3d} / {len(target)}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else C.ROOT / "raw_trajectory_data.csv")
