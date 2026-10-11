"""Render the two judge payloads, N and P, through a single code path.

This module exists so that the `-strip` conditions are provably `-full` minus the
prose, rather than hopefully so. Everything downstream builds its prompt from the
same `Block` list; `-strip` is `render_work(blocks, include_prose=False)`.

Two invariants that must not be broken:

1. Truncation is computed from N ALONE, never from the assembled prompt. If you
   truncate the assembled `-full` and `-strip` strings to a common budget the cut
   points differ, because `-full` is longer, and `-strip` stops being `-full`
   minus the prose. The manipulation would then be contaminated by a rendering
   artefact rather than by the thing we are testing.
2. `render_work` is one function. `-full` and `-strip` differ only by which blocks
   are in the list, never by formatting, separator or budget.

On what P actually is (verified in ldp/graph/modules/react.py). Each agent step is
two LLM calls. The first runs with stop sequences ["Observation:", "Action:"], so
the model can emit nothing but prose. The harness then rewrites that output:

    reasoning = (react_message.content or "").removeprefix("Thought: ")
    Message(content=f"Thought: {reasoning}."
                    " Based on this reasoning, let's select the appropriate tool!"
                    "\nAction: ", role="assistant")
    Message(content="Continue...")

So the model authored only `reasoning`. The `Thought:` prefix, the bare `.` that
creates the doubled terminator, the exhortation, the `\nAction:` and the
`Continue...` turn are all harness-authored. `strip_prose` removes exactly those,
which is why we can say in the paper that everything stripped is provably not the
model's words.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# ----------------------------------------------------------------- constants --
SEP = "\n\n"
CELL_BUDGET = 12_000          # per rendered cell block
NB_BUDGET = 50_000            # whole notebook; passes 46/47 observed notebooks whole
SNAP = 200                    # snap an elision cut to a newline within this many chars

ELIDE_CELL = "\n[... {n:,} characters elided from the middle of this output ...]\n"
ELIDE_NB = "### [... {k} notebook cells ({n:,} characters) elided ...]"

BOILERPLATE = re.compile(
    r"\s*Based on this reasoning,\s*let'?s select the appropriate tool!\s*", re.I
)
APPENDED_RE = re.compile(r"Appended new cell \(#(\d+)\)")
EDITED_RE = re.compile(r"Edited cell #(\d+)")


@dataclass(frozen=True)
class Block:
    """One renderable unit. `kind` is 'prose', 'cell' or 'elision'."""

    kind: str
    index: int
    text: str


# ------------------------------------------------------- truncation primitive --
def elide(text: str, budget: int, marker: str) -> tuple[str, bool]:
    """Head + marker + tail. Returns (text, was_truncated).

    Cut points snap to the nearest newline within SNAP chars so we do not slice a
    line in half. A no-op when the text already fits.
    """
    if len(text) <= budget:
        return text, False
    # The marker counts against the budget, so the result never exceeds it and a
    # second pass is a no-op. Reserving the widest possible marker (n = len(text))
    # keeps that true regardless of how many digits the real count needs.
    reserve = len(marker.format(n=len(text)))
    half = max(budget - reserve, 200) // 2
    head, tail = text[:half], text[-half:]
    nl = head.rfind("\n")
    if nl != -1 and len(head) - nl <= SNAP:
        head = head[:nl]
    nl = tail.find("\n")
    if nl != -1 and nl <= SNAP:
        tail = tail[nl + 1:]
    dropped = len(text) - len(head) - len(tail)
    return head + marker.format(n=dropped) + tail, True


# --------------------------------------------------------------------- P ------
def strip_prose(thought: str) -> str:
    """Reduce a harness-wrapped Thought turn to the model's own words."""
    t = thought.strip()
    if t.startswith("Thought:"):
        t = t[len("Thought:"):]
    t = BOILERPLATE.sub(" ", t)
    t = re.sub(r"\bAction:\s*$", "", t.strip())
    # the harness appends a bare '.' to reasoning that usually already ends in one
    t = re.sub(r"\.\s*\.(\s*\.)*", ".", t)
    t = re.sub(r"[ \t]+\n", "\n", t)
    return t.strip()


def extract_prose(messages: list[dict]) -> list[str]:
    """The agent's prose turns, in order, stripped of harness scaffolding."""
    out = []
    for m in messages:
        if m.get("role") != "assistant":
            continue
        c = m.get("content") or ""
        if c.strip().startswith("Thought:"):
            out.append(strip_prose(c))
    return out


# --------------------------------------------------------------------- N ------
def _output_text(cell: dict) -> str:
    parts = []
    for o in cell.get("outputs", []) or []:
        t = o.get("text")
        if t is None:
            t = (o.get("data") or {}).get("text/plain")
        if t is None and o.get("ename"):
            t = f"{o['ename']}: {o.get('evalue', '')}"
        if t is None:
            continue
        parts.append("".join(t) if isinstance(t, list) else str(t))
    return "\n".join(p for p in parts if p)


def render_cell(cell: dict, index: int) -> str:
    src = cell.get("source", "")
    if isinstance(src, list):
        src = "".join(src)
    body = f"### Cell {index} [{cell.get('cell_type', 'code')}]\n```\n{src.rstrip()}\n```"
    out = _output_text(cell)
    if out:
        body += f"\nOutput:\n{out.rstrip()}"
    return body


def render_notebook_blocks(nb: dict) -> tuple[list[Block], dict]:
    """Cell blocks plus truncation stats. Depends on N only, never on P."""
    cells = nb.get("cells", []) if isinstance(nb, dict) else []
    blocks: list[Block] = []
    n_cell_outputs_elided = 0
    for i, c in enumerate(cells):
        text, truncated = elide(render_cell(c, i), CELL_BUDGET, ELIDE_CELL)
        n_cell_outputs_elided += int(truncated)
        blocks.append(Block("cell", i, text))

    stats = {
        "n_cells": len(cells),
        "n_cell_outputs_elided": n_cell_outputs_elided,
        "n_cells_elided": 0,
        "notebook_truncated": 0,
    }

    total = sum(len(b.text) for b in blocks) + SEP.__len__() * max(len(blocks) - 1, 0)
    if total <= NB_BUDGET or len(blocks) <= 2:
        stats["notebook_chars"] = total
        return blocks, stats

    # still too big: drop whole cells from the middle, head-and-tail-wise
    keep_head, keep_tail, running = [], [], 0
    lo, hi = 0, len(blocks) - 1
    while lo <= hi:
        if running + len(blocks[lo].text) > NB_BUDGET:
            break
        keep_head.append(blocks[lo]); running += len(blocks[lo].text); lo += 1
        if lo > hi:
            break
        if running + len(blocks[hi].text) > NB_BUDGET:
            break
        keep_tail.append(blocks[hi]); running += len(blocks[hi].text); hi -= 1
    dropped = blocks[lo:hi + 1]
    if dropped:
        marker = Block("elision", -1, ELIDE_NB.format(
            k=len(dropped), n=sum(len(b.text) for b in dropped)))
        blocks = keep_head + [marker] + list(reversed(keep_tail))
        stats["n_cells_elided"] = len(dropped)
        stats["notebook_truncated"] = 1
    stats["notebook_chars"] = sum(len(b.text) for b in blocks)
    return blocks, stats


# ------------------------------------------------------------------ P x N -----
def cell_owner(messages: list[dict]) -> dict[int, int]:
    """cell index -> index of the prose turn whose action first appended it.

    Parsed from the `Observation: Appended new cell (#N).` tool results, which are
    100% regular across every trajectory inspected. First-append attribution: an
    edited cell shows its final content under the prose that created it.
    """
    owner: dict[int, int] = {}
    prose_idx = -1
    for m in messages:
        role = m.get("role")
        content = m.get("content") or ""
        if role == "assistant" and content.strip().startswith("Thought:"):
            prose_idx += 1
        elif role == "tool":
            hit = APPENDED_RE.search(content)
            if hit:
                owner.setdefault(int(hit.group(1)), max(prose_idx, 0))
    return owner


def build_blocks(nb: dict, prose: list[str], owner: dict[int, int]) -> tuple[list[Block], dict]:
    """Interleave prose and cells into the single canonical block list."""
    cell_blocks, stats = render_notebook_blocks(nb)
    blocks: list[Block] = []
    emitted: set[int] = set()

    def flush_upto(p_idx: int) -> None:
        for k in range(len(prose)):
            if k not in emitted and k <= p_idx and prose[k].strip():
                blocks.append(Block("prose", k, f"### Agent note {k}\n{prose[k]}"))
                emitted.add(k)

    for cb in cell_blocks:
        if cb.kind == "cell":
            flush_upto(owner.get(cb.index, -1))
        blocks.append(cb)
    # prose turns that created no cell (list_workdir, the final summary, submit_answer)
    flush_upto(len(prose))
    stats["n_prose_turns"] = len(prose)
    stats["prose_chars_stripped"] = sum(len(p) for p in prose)
    return blocks, stats


def render_work(blocks: list[Block], include_prose: bool) -> str:
    """The ONE renderer. `-strip` is this with include_prose=False."""
    keep = blocks if include_prose else [b for b in blocks if b.kind != "prose"]
    return SEP.join(b.text for b in keep)


def strip_blocks(blocks: list[Block]) -> list[Block]:
    return [b for b in blocks if b.kind != "prose"]
