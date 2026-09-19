# Results

M1 = `claude-sonnet-4-6` (the stage-1 agent, and the judge in A-*; temperature 0).
M2 = `gpt-5-mini-2025-08-07` (judge only, B-*; provider-fixed temperature, reasoning_effort=minimal).

**wrong-answer acceptance rate** = false-positive rate = P(ACCEPT | answer is incorrect). **correct-answer rejection rate** = false-negative rate = P(REJECT | answer is correct). Balanced accuracy is the mean of correct-answer acceptance and true-reject, and is the guard that separates better judgement from a shifted threshold.

36 capsules; 46 correct and 70 incorrect answers per condition; 5000 capsule-cluster bootstrap draws.

**20 capsules hold at least one correct answer.** That, not 46, is the effective n behind every correct-answer rate below.

| condition | judge | wrong-answer acceptance | correct-answer acceptance | balanced accuracy | raw acc | unparseable | conf parsed | n |
|---|---|---|---|---|---|---|---|---|
| A-self | M1 | 0.900 [0.833, 0.958] | 0.978 [0.924, 1.000] | 0.539 [0.501, 0.577] | 0.448 | 0.0% | 100% | 116 |
| A-full | M1 | 0.857 [0.753, 0.957] | 0.957 [0.881, 1.000] | 0.550 [0.489, 0.610] | 0.466 | 0.0% | 100% | 116 |
| A-strip | M1 | 0.900 [0.803, 0.978] | 0.978 [0.924, 1.000] | 0.539 [0.493, 0.589] | 0.448 | 0.0% | 100% | 116 |
| A-floor | M1 | 0.300 [0.200, 0.419] | 0.348 [0.245, 0.478] | 0.524 [0.459, 0.596] | 0.560 | 0.0% | 100% | 116 |
| B-full | M2 | 0.971 [0.919, 1.000] | 1.000 [1.000, 1.000] | 0.514 [0.500, 0.540] | 0.414 | 0.0% | 100% | 116 |
| B-strip | M2 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.500 [0.500, 0.500] | 0.397 | 0.0% | 100% | 116 |
| B-floor | M2 | 0.057 [0.014, 0.103] | 0.152 [0.026, 0.267] | 0.548 [0.483, 0.606] | 0.629 | 0.0% | 100% | 116 |

**Guards**

- **BLANKET ACCEPT** in `B-strip`: every parsed verdict is ACCEPT. Its wrong-answer acceptance rate is meaningless on its own.

**Verdict counts** (ACCEPT / REJECT / UNPARSEABLE)

| condition | on correct answers | on incorrect answers |
|---|---|---|
| A-self | 45 / 1 / 0 | 63 / 7 / 0 |
| A-full | 44 / 2 / 0 | 60 / 10 / 0 |
| A-strip | 45 / 1 / 0 | 63 / 7 / 0 |
| A-floor | 16 / 30 / 0 | 21 / 49 / 0 |
| B-full | 46 / 0 / 0 | 68 / 2 / 0 |
| B-strip | 46 / 0 / 0 | 70 / 0 / 0 |
| B-floor | 7 / 39 / 0 | 4 / 66 / 0 |

**Paired contrasts** (first minus second; 95% cluster-bootstrap CI; two-sided bootstrap p)

| contrast | metric | difference | 95% CI | p | reading |
|---|---|---|---|---|---|
| A-self - A-full | wrong_answer_acceptance_rate | +0.043 | [-0.059, +0.134] | 0.491 | PRIMARY (H1): own prior turn vs the same work quoted |
| A-self - A-full | correct_answer_acceptance_rate | +0.022 | [+0.000, +0.077] | 0.720 |  |
| A-self - A-full | balanced_accuracy | -0.011 | [-0.061, +0.045] | 0.769 |  |
| | | | | | **THRESHOLD SHIFT**: `A-self` moves both acceptance rates the same way vs `A-full`. A changed willingness to accept, not better auditing. |
| A-full - A-strip | wrong_answer_acceptance_rate | -0.043 | [-0.123, +0.038] | 0.397 | argument effect, M1 |
| A-full - A-strip | correct_answer_acceptance_rate | -0.022 | [-0.077, +0.000] | 0.720 |  |
| A-full - A-strip | balanced_accuracy | +0.011 | [-0.037, +0.056] | 0.704 |  |
| | | | | | **THRESHOLD SHIFT**: `A-full` moves both acceptance rates the same way vs `A-strip`. A changed willingness to accept, not better auditing. |
| B-full - B-strip | wrong_answer_acceptance_rate | -0.029 | [-0.081, +0.000] | 0.250 | argument effect, M2 |
| B-full - B-strip | correct_answer_acceptance_rate | +0.000 | [+0.000, +0.000] | 1.000 |  |
| B-full - B-strip | balanced_accuracy | +0.014 | [+0.000, +0.040] | 0.250 |  |
| A-full - B-full | wrong_answer_acceptance_rate | -0.114 | [-0.222, +0.000] | 0.060 | judge identity, argument present |
| A-full - B-full | correct_answer_acceptance_rate | -0.043 | [-0.119, +0.000] | 0.260 |  |
| A-full - B-full | balanced_accuracy | +0.035 | [-0.029, +0.096] | 0.306 |  |
| | | | | | **THRESHOLD SHIFT**: `A-full` moves both acceptance rates the same way vs `B-full`. A changed willingness to accept, not better auditing. |
| A-strip - B-strip | wrong_answer_acceptance_rate | -0.100 | [-0.197, -0.022] | 0.010 | judge identity, argument absent |
| A-strip - B-strip | correct_answer_acceptance_rate | -0.022 | [-0.076, +0.000] | 0.740 |  |
| A-strip - B-strip | balanced_accuracy | +0.039 | [-0.007, +0.089] | 0.110 |  |
| | | | | | **THRESHOLD SHIFT**: `A-strip` moves both acceptance rates the same way vs `B-strip`. A changed willingness to accept, not better auditing. |
| A-floor - A-strip | wrong_answer_acceptance_rate | -0.600 | [-0.712, -0.475] | 0.000 | does M1's stripped judge use the notebook |
| A-floor - A-strip | correct_answer_acceptance_rate | -0.630 | [-0.731, -0.500] | 0.000 |  |
| A-floor - A-strip | balanced_accuracy | -0.015 | [-0.084, +0.062] | 0.700 |  |
| | | | | | **THRESHOLD SHIFT**: `A-floor` moves both acceptance rates the same way vs `A-strip`. A changed willingness to accept, not better auditing. |
| B-floor - B-strip | wrong_answer_acceptance_rate | -0.943 | [-0.986, -0.897] | 0.000 | does M2's stripped judge use the notebook |
| B-floor - B-strip | correct_answer_acceptance_rate | -0.848 | [-0.974, -0.733] | 0.000 |  |
| B-floor - B-strip | balanced_accuracy | +0.048 | [-0.017, +0.106] | 0.173 |  |
| | | | | | **THRESHOLD SHIFT**: `B-floor` moves both acceptance rates the same way vs `B-strip`. A changed willingness to accept, not better auditing. |
| A-self - B-strip | wrong_answer_acceptance_rate | -0.100 | [-0.167, -0.042] | 0.000 | CONFOUNDED end-to-end: judge, framing and argument all differ. NOT evidence for H1 |
| A-self - B-strip | correct_answer_acceptance_rate | -0.022 | [-0.076, +0.000] | 0.740 |  |
| A-self - B-strip | balanced_accuracy | +0.039 | [+0.001, +0.077] | 0.046 |  |
| | | | | | **THRESHOLD SHIFT**: `A-self` moves both acceptance rates the same way vs `B-strip`. A changed willingness to accept, not better auditing. |

**Why judges reject** (share of REJECT justifications)

| condition | asserts a different value | of those, value IS in the notebook | cites uncertainty | other | n |
|---|---|---|---|---|---|
| A-self | 12% | 100% | 0% | 88% | 8 |
| A-full | 8% | 100% | 0% | 92% | 12 |
| A-strip | 25% | 100% | 0% | 75% | 8 |
| A-floor | 5% | 100% | 54% | 41% | 79 |
| B-full | 0% | 0% | 50% | 50% | 2 |
| B-strip | - | - | - | - | 0 |
| B-floor | 4% | 100% | 49% | 48% | 105 |

A value asserted that is NOT in the notebook is confabulated ground truth. One that IS in the notebook is evidence-based rejection. In v1 `A-self` confabulated in 43% of its rejections against 2-7% elsewhere, but v1 judges had no notebook to read.

**Expected calibration error** (lower is better): `A-self` 0.391, `A-full` 0.413, `A-strip` 0.425, `A-floor` 0.297, `B-full` 0.485, `B-strip` 0.499, `B-floor` 0.271

