# Stage 1: running the BixBench agent

These three files are ours. They are not part of
[Future-House/BixBench](https://github.com/Future-House/BixBench) and are kept here rather than
inside a clone of someone else's repository.

| file | what it is |
|---|---|
| `run_bioauditor.py` | subclasses their `TrajectoryGenerator`; restricts the run to our 122 questions and fixes a cost bug in their loop |
| `bioauditor_config.yaml` | our run config, a patched copy of their `generate_trajectories.yaml` |
| `status.sh` | progress, container health, disk and ETA while a run is going |

## Setup

Pinned to BixBench commit `4931118` ("Merge pull request #39 from Future-House/revisions").
Later commits may move the file this patch depends on.

```bash
git clone https://github.com/Future-House/BixBench.git ~/bixbench
cd ~/bixbench && git checkout 4931118
cp /path/to/bio-auditor/harness/{run_bioauditor.py,bioauditor_config.yaml,status.sh} .
uv sync --python 3.13
huggingface-cli login
docker pull futurehouse/bixbench:aviary-notebook-env
cp /path/to/bio-auditor/.env .env          # ANTHROPIC_API_KEY, HF_TOKEN
uv run --python 3.13 python run_bioauditor.py
```

## Four things that will bite you

**Python 3.13, not 3.12.** BixBench declares `requires-python = ">=3.12"`, but `bixbench/utils.py`
does `async for` over `asyncio.as_completed`, which only became async-iterable in 3.13. On 3.12 it
dies immediately with `TypeError: 'async for' requires an object with __aiter__`.

**Their default model is retired.** The shipped config uses `claude-3-7-sonnet-20250219`, which
now returns a 404. Ours pins `claude-sonnet-4-6`, which also has to match the judge model for the
self-review condition to mean anything.

**The image is arm64-only.** `futurehouse/bixbench:aviary-notebook-env` publishes no x86 manifest.
On an Intel or AMD machine you will need emulation.

**Docker Desktop on WSL needs integration enabled for your distro**, or there is no
`/var/run/docker.sock` and `aiodocker` fails. Settings → Resources → WSL Integration.

## The cost bug we patch

Upstream `generate_trajectories.py:456`:

```python
for i in range(0, len(bixbench), self.config.rollout.batch_size):   # advances by 4
    batch = bixbench[i : i + (4 * bsz)]                             # but takes 16
```

The window is four times the stride, so the windows overlap and **every question is submitted
about 3.8 times**. `rollout_with_saving` never re-checks for an existing trajectory mid-run, so
each duplicate re-runs the agent from scratch and overwrites the result: correct data, roughly
four times the API spend. Simulated over 121 questions that is 460 rollouts instead of 121.

`run_bioauditor.py` overrides `run()` with stride equal to window. Note that window and stride are
separate knobs and collapsing both to `batch_size` is *also* wrong: it removes the duplicates but
reintroduces a barrier where all four start and the loop waits for the slowest before the next
four begin. The correct setting is **stride = window = `4 * batch_size`, concurrency =
`batch_size`**.

## Two failure modes worth knowing

**Workspace reuse silently contaminates re-runs.** `fhda/notebook_env.py` does
`if self.nb_path.exists(): nbformat.read(...)`, and workspace directories persist. A re-run
question opens the *previous* run's finished notebook and may submit without writing a line — we
saw agents submit after a single action against a nine-cell notebook they never wrote. For those
rows the prose does not explain the notebook, which destroys the contrast the experiment rests on.
`collect.py` detects it (`n_cells_appended < n_cells - 1`). **Always delete the workspace
directory, not just the trajectory, before re-running a question.**

**The window timeout abandons work silently.** `as_completed_with_concurrency(..., timeout=3600)`
applies to a whole window, not per question, and the `except TimeoutError: continue` branch drops
whatever is still pending and carries on to log "Completed trajectory generation". It cost us one
question outright. Check the final count against your target rather than trusting the exit code.

Also: containers leak after a run exits. Clean them up, and never put container cleanup inside a
polling loop that can time out — it will kill work that is still running.
