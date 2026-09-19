# bio-auditor — run of 2026-09-16: findings and flags

Full tables: `summary.md`, `metrics.csv`, `contrasts.csv`. Figure: `../paper/fig1.png`
(all answers) and `../paper/fig1_natural_only.png`. Raw data: `answers.csv`, `labelled.csv`,
`normalisation_log.csv`, `judgments.csv`. Every API response is in `../cache/`.

## What was run

| stage | what | count |
|---|---|---|
| 1 | M1 = `claude-sonnet-4-6`, temperature 1, 5 natural samples × 122 str/range questions | 610 answers |
| 1 | synthetic-correct answers (ideal injected, M1 writes the derivation), 1 per question | 122 answers |
| 2 | deterministic labelling, no model involved | 151 correct, 575 incorrect, 6 unparseable |
| 3 | 4 conditions × 726 labelled answers, one call each; M1 at temperature 0, M2 = `gpt-5-mini-2025-08-07` at its fixed temperature 1, reasoning_effort minimal | 2,904 verdicts, 1 unparseable |
| 4 | capsule-cluster bootstrap (36 capsules, 2,000 draws) | — |

Total spend ≈ $12.

Pilot (20 seeded questions × 5 samples) gave 3 correct of 100, all from one categorical question.
N = 5 was kept for the full run to maximise genuinely derived correct answers; the §7
synthetic-correct contingency was applied because 29 natural correct answers is too few for a
false-reject signal on its own.

## Headline result: blanket rejection, balanced accuracy at chance in every condition

| condition | judge | sees | balanced acc [95% CI] | false-accept | false-reject |
|---|---|---|---|---|---|
| A  | M1 | own trace, assistant slot | 0.511 [0.496, 0.530] | 0.005 | 0.974 |
| A' | M1 | answer only | 0.512 [0.465, 0.571] | 0.063 | 0.913 |
| B  | M2 | trace, quoted | 0.503 [0.500, 0.510] | 0.000 | 0.993 |
| C  | M2 | answer only | 0.500 [0.500, 0.500] | 0.000 | 1.000 |

Condition C accepted nothing at all (0 of 726). B accepted one answer. A accepted 7 (4 right, 3 wrong).
A' accepted 49 (13 right, 36 wrong). No hypothesis is supported:

- H1 (trace → more false accepts): B − C false-accept difference is exactly 0. A − A' is −0.057
  [−0.093, −0.029], the opposite sign: the author with its trace accepts *fewer* wrong answers
  than without it. It also rejects more right ones (+0.060), so balanced accuracy is unchanged.
- H2 (C beats A): C − A balanced accuracy −0.011 [−0.030, +0.004]. C's perfect false-accept rate
  is bought entirely by rejecting everything; this is the §7 "scepticism, not discrimination" case.
- H3: A' − A balanced accuracy +0.001; nothing there.

## Why the judges reject everything

The justifications were checked by pattern (table in `summary.md`):

| condition | "cannot verify without the data" | "the actual value is different from X" |
|---|---|---|
| A  | 45% | 43% |
| A' | 89% | 25% |
| B  | 94% | 7% |
| C  | 93% | 2% |

The TASK text shown to every judge is the stage-1 prompt, which tells the model it has no data
access and should give a best estimate. Every judge therefore treats every answer as an
unverifiable guess and rejects it. The judgment is dominated by the framing of the task, not by
the answer or the trace. This is a design problem, not a prompt-tuning problem, and CLAUDE.md
says not to iterate prompts until a gap appears, so nothing was changed.

One descriptive observation survives and may be worth keeping: in condition A, M1 reviewing its
own turn frequently rejects by asserting that "the actual value ... is different from X" — a
claim about the dataset it cannot know — rather than by saying it cannot verify. The other three
conditions almost never do this. With the trace removed (A'), the same model becomes the only
judge that accepts anything, and it accepts wrong answers at a higher rate than right ones
(36 vs 13). Genuinely derived correct answers were accepted by A' 34% of the time, synthetic
ones 2%; judges are not fooled by the injected derivations.

## Deviations from CLAUDE.md as originally written (all recorded in CLAUDE.md §5)

1. M1 is Sonnet 4.6, not Haiku 4.5 (Eva's request), and not Sonnet 5, which rejects `temperature`.
2. M2 judges at temperature 1: OpenAI returns 400 for anything else on GPT-5 mini.
3. Both judges run without hidden reasoning (M1 thinking off, M2 reasoning_effort minimal) for symmetry.
4. Single judge call per (answer, condition), no replicates (Eva's spec).
5. Direct API with 8 threads instead of the Batches API; cache makes reruns free.
6. Synthetic-correct answers were added without a pause for confirmation, because Eva asked for an
   autonomous run. They are flagged in `source` and every metric is also reported on natural
   answers only (`summary.md`, set `natural`). Condition A on a synthetic row means the model
   reviews a derivation it was told to write, not one it produced.

## Things found and fixed along the way

- **Grader bug (fixed, regraded):** the numeric comparison used an absolute tolerance floor, which
  made 2.2e-16 equal to 9.92E-35 and labelled five bix-9-q4 answers correct. Now a pure relative
  tolerance; unit cases in `src/grade_truth.py` history. Judgments were rebuilt from cache so
  labels are current.
- **Leak guard false positive:** bix-22-q1 lists "CD14" among its options and its ideal is CD14.
  The guard now checks only the scaffold around the question.
- **Verdict parser widened** for "INCORRECT. The correct answer is likely ...": the first word wins.
  One verdict (A', bix-46-q1) contains reasoning and no verdict and stays unparseable.
- **bix-3-q2** asks for a 95% confidence interval; all five natural answers are intervals and
  the range grader cannot score them. Labelled unparseable and excluded (5 of the 6 unparseable
  answers). bix-31-q2 answered "NA" once.
- `raw_trajectory_data.csv` was rejected for stage 1; see `src/inspect_trajectories.py`.

## Decisions for Eva before a second run

Any of these changes what a condition means, so none was made:

1. **Change the review instruction I** (identically in all four conditions) to ask whether the
   value is the one the analysis would most plausibly produce, rather than whether it is
   "correct". Keeps the byte-identical invariant; removes the "cannot verify" exit.
2. **Hide the no-data framing from the judge.** Put the no-data instruction in a system prompt at
   generation time so the TASK the judges see is only the question. Condition A's reconstruction
   would then omit that system prompt, a small departure from "as it happened".
3. **Use answers that come with a real derivation** (an agent with data access, e.g. re-running
   BixBench agents on the current question set), so judges have something checkable. This is the
   setting the paper's motivation actually describes.
