"""Stage 1 runner for bio-auditor.

Wraps BixBench's TrajectoryGenerator. Two deliberate changes:

1. WHICH questions run: only the 122 str_verifier / range_verifier rows, so ground
   truth stays deterministic.
2. A fix for a cost bug in their `run()`. Upstream advances the loop by `batch_size`
   but slices a window of `4 * batch_size`, so the windows overlap and every question
   is submitted ~3.8 times. `rollout_with_saving` does not check for an existing
   trajectory mid-run, so each duplicate re-runs the agent and overwrites the result:
   correct data, ~4x the API spend. We iterate non-overlapping windows instead, and
   re-check for a completed trajectory at submit time as belt-and-braces.

Prompts, agent, notebook environment and trajectory serialisation are BixBench's own
code, untouched.

  uv run python run_bioauditor.py --only bix-1-q1
  uv run python run_bioauditor.py --limit 5
  uv run python run_bioauditor.py
"""
from __future__ import annotations

import argparse
import asyncio
import logging

import datasets
from tqdm import tqdm

from bixbench.generate_trajectories import TrajectoryGenerator
from bixbench.utils import as_completed_with_concurrency

logger = logging.getLogger(__name__)
TARGET_EVAL_MODES = {"str_verifier", "range_verifier"}


class FilteredTrajectoryGenerator(TrajectoryGenerator):
    def __init__(self, config_path, replica_id=None, only=None, limit=None):
        super().__init__(config_path, replica_id)
        self.only = set(only) if only else None
        self.limit = limit

    async def load_bixbench(self):
        rows = datasets.load_dataset(
            self.config.paths.hf_repo_id, split=self.config.dataset_split
        ).to_list()
        rows = [q for q in rows if q["eval_mode"] in TARGET_EVAL_MODES]
        assert len(rows) == 122, f"expected 122 target questions, got {len(rows)}"
        if self.only:
            rows = [q for q in rows if q["question_id"] in self.only]
        if self.limit:
            rows = rows[: self.limit]
        print(f"[bio-auditor] {len(rows)} question(s) selected", flush=True)
        await asyncio.gather(*[
            self.process_capsule_data(z) for z in {q["data_folder"] for q in rows}
        ])
        await asyncio.gather(*[self.process_question(q) for q in rows])
        return rows

    def _already_done(self, question) -> bool:
        return self.get_trajectory_path(question["question_id"]).exists()

    async def run(self) -> None:
        rows = await self.load_bixbench()
        if self.config.rollout.skip_existing_trajectories:
            rows = self.filter_completed(rows)
        bsz = self.config.rollout.batch_size
        print(f"[bio-auditor] {len(rows)} to run, batch_size={bsz}, "
              f"window={4*bsz} stride={4*bsz} concurrency={bsz} (each question submitted exactly once)", flush=True)
        window = 4 * bsz          # upstream's window size, which was fine
        with tqdm(total=len(rows), desc="Processing benchmark tasks") as pbar:
            for i in range(0, len(rows), window):       # stride == window: no overlap
                batch = [q for q in rows[i : i + window] if not self._already_done(q)]
                if not batch:
                    pbar.update(len(rows[i : i + window]))
                    continue
                rollouts = (
                    self.rollout_with_saving(self.environment_factory(q)) for q in batch
                )
                try:
                    async for _ in as_completed_with_concurrency(rollouts, bsz, timeout=3600):
                        pbar.update(1)
                except TimeoutError:
                    logger.warning("Timeout while rolling out environments")
                    continue
        logger.info("Completed trajectory generation")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="BixBench stage 1 for bio-auditor")
    p.add_argument("--config_file", default="bioauditor_config.yaml")
    p.add_argument("--replica_id", type=int, default=None)
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args()
    gen = FilteredTrajectoryGenerator(a.config_file, a.replica_id, a.only, a.limit)
    asyncio.run(gen.run())
