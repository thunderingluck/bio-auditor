"""Checks for render.py against real trajectories. Run: .venv/bin/python src/test_render.py"""
import json, sys, glob
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import render as R

TRAJ = Path.home() / "bixbench/data/trajectories/bioauditor-s1"
fails = []
def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'' if cond else '  <- ' + detail}")
    if not cond: fails.append(name)

def load(qid):
    d = json.load(open(TRAJ / f"{qid}.json"))
    last = None
    for line in open(TRAJ / f"{qid}.jsonl"):
        line = line.strip()
        if not line: continue
        try: o = json.loads(line)
        except Exception: continue
        if isinstance(o, dict): last = o
    msgs = (last.get("next_agent_state") or last.get("agent_state"))["messages"]
    nb = d["nb"] if isinstance(d["nb"], dict) else json.loads(d["nb"])
    return d, msgs, nb

print("=== strip_prose ===")
raw = ("Thought: Now I have a clear picture of the data. Let me write the R analysis code.."
       " Based on this reasoning, let's select the appropriate tool!"
       " Based on this reasoning, let's select the appropriate tool! Action:")
s = R.strip_prose(raw)
check("removes Thought: prefix", not s.startswith("Thought:"), s[:60])
check("removes boilerplate (x2)", "appropriate tool" not in s, s[:80])
check("removes trailing Action:", not s.rstrip().endswith("Action:"), s[-40:])
check("collapses doubled terminator", ".." not in s, repr(s[-40:]))
check("keeps the model's words", "clear picture of the data" in s, s[:60])

print("\n=== elide ===")
t = "line\n" * 5000
e1, tr1 = R.elide(t, 1000, R.ELIDE_CELL)
e2, tr2 = R.elide(e1, 1000, R.ELIDE_CELL)
check("truncates when over budget", tr1 and len(e1) < len(t))
check("idempotent (second pass is a no-op)", e1 == e2 and not tr2, f"{len(e1)} vs {len(e2)}")
short = "abc"
check("no-op when under budget", R.elide(short, 1000, R.ELIDE_CELL) == (short, False))
check("marker present", "elided" in e1)

print("\n=== the load-bearing invariant: -strip == -full minus prose ===")
qids = [Path(f).stem for f in sorted(glob.glob(str(TRAJ / "*.json")))]
tested = 0
for qid in qids:
    try: d, msgs, nb = load(qid)
    except Exception: continue
    prose = R.extract_prose(msgs)
    owner = R.cell_owner(msgs)
    blocks, stats = R.build_blocks(nb, prose, owner)
    full = R.render_work(blocks, True)
    strip = R.render_work(blocks, False)
    if not R.strip_blocks(blocks) == [b for b in blocks if b.kind != "prose"]:
        check(f"{qid} strip_blocks consistent", False); break
    if strip != R.SEP.join(b.text for b in R.strip_blocks(blocks)):
        check(f"{qid} strip == join(strip_blocks)", False); break
    leaked = [b for b in blocks if b.kind == "prose" and len(b.text) > 40 and b.text in strip]
    if leaked:
        check(f"{qid} no prose leaks into -strip", False, f"{len(leaked)} leaked"); break
    if not all(b.text in full for b in blocks):
        check(f"{qid} every block present in -full", False); break
    tested += 1
check(f"invariant holds on all {tested} trajectories", tested == len(qids), f"{tested}/{len(qids)}")

print("\n=== truncation on the largest notebook ===")
big = max(qids, key=lambda q: (TRAJ / f"{q}.json").stat().st_size)
d, msgs, nb = load(big)
blocks, stats = R.build_blocks(nb, R.extract_prose(msgs), R.cell_owner(msgs))
strip = R.render_work(blocks, False)
print(f"  largest notebook: {big}  cells={stats['n_cells']}  rendered={len(strip):,} chars")
print(f"  cell-level elisions={stats['n_cell_outputs_elided']}  whole cells dropped={stats['n_cells_elided']}"
      f"  truncated={stats['notebook_truncated']}")
check("respects the notebook budget exactly", len(strip) <= R.NB_BUDGET, f"{len(strip):,}")
for b in blocks:
    if b.kind == "cell" and len(b.text) > R.CELL_BUDGET:
        check(f"cell {b.index} within CELL_BUDGET", False, f"{len(b.text):,}"); break
else:
    check("every cell block within CELL_BUDGET", True)

print("\n=== prose/cell interleaving sanity ===")
d, msgs, nb = load("bix-1-q1")
prose = R.extract_prose(msgs); owner = R.cell_owner(msgs)
blocks, stats = R.build_blocks(nb, prose, owner)
check("all prose turns emitted exactly once",
      sum(1 for b in blocks if b.kind == "prose") == sum(1 for p in prose if p.strip()),
      f"{sum(1 for b in blocks if b.kind=='prose')} vs {len(prose)}")
check("all cells emitted", sum(1 for b in blocks if b.kind == "cell") == stats["n_cells"])
check("cell_owner found appends", len(owner) > 0, f"owner={len(owner)}")
print(f"  block order: {[b.kind[0] for b in blocks][:24]}")

print("\n" + ("ALL CHECKS PASSED" if not fails else f"{len(fails)} FAILURE(S): {fails}"))
sys.exit(1 if fails else 0)
