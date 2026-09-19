# Results summary

M1 = `claude-sonnet-4-6` (answerer, judge in A/A'; temperature 0.0 when judging). M2 = `gpt-5-mini-2025-08-07` (judge in B/C; temperature fixed at 1 by provider, reasoning_effort=minimal).
Unparseable verdicts are excluded from rate denominators and reported separately. CIs are percentile intervals from a capsule-cluster bootstrap; contrasts are paired on the same draws.

## Set `all`: 36 capsules, 151 correct + 575 incorrect answers per condition, 2000 cluster-bootstrap draws

| condition | judge | balanced acc | false-accept (wrong answers) | false-reject (right answers) | raw acc | unparseable | n |
|---|---|---|---|---|---|---|---|
| A | M1 | 0.511 [0.496, 0.530] | 0.005 [0.000, 0.014] | 0.974 [0.936, 1.000] | 0.793 [0.770, 0.811] | 0.0% | 726 |
| A' | M1 | 0.512 [0.465, 0.571] | 0.063 [0.031, 0.101] | 0.913 [0.801, 1.000] | 0.761 [0.725, 0.791] | 0.1% | 726 |
| B | M2 | 0.503 [0.500, 0.510] | 0.000 [0.000, 0.000] | 0.993 [0.981, 1.000] | 0.793 [0.766, 0.815] | 0.0% | 726 |
| C | M2 | 0.500 [0.500, 0.500] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.792 [0.763, 0.814] | 0.0% | 726 |

**Verdict counts** (rows: truth label; cells: CORRECT / INCORRECT / UNPARSEABLE verdicts)

| condition | correct answers | incorrect answers |
|---|---|---|
| A | 4 / 147 / 0 | 3 / 572 / 0 |
| A' | 13 / 137 / 1 | 36 / 539 / 0 |
| B | 1 / 150 / 0 | 0 / 575 / 0 |
| C | 0 / 151 / 0 | 0 / 575 / 0 |

**Paired contrasts** (first minus second; 95% cluster-bootstrap CI; two-sided bootstrap p)

| contrast | metric | difference | 95% CI | p | reading |
|---|---|---|---|---|---|
| B − C | balanced_accuracy | +0.003 | [+0.000, +0.010] | 0.707 | H1: trace vs no trace, same judge (M2) |
| B − C | false_accept_rate | +0.000 | [+0.000, +0.000] | 1.000 |  |
| B − C | false_reject_rate | -0.007 | [-0.019, +0.000] | 0.707 |  |
| A − A' | balanced_accuracy | -0.001 | [-0.042, +0.033] | 0.977 | H1/H3: author with vs without its trace |
| A − A' | false_accept_rate | -0.057 | [-0.093, -0.029] | 0.000 |  |
| A − A' | false_reject_rate | +0.060 | [+0.000, +0.135] | 0.079 |  |
| C − A | balanced_accuracy | -0.011 | [-0.030, +0.004] | 0.327 | H2: quarantined auditor vs deployed self-review (headline) |
| C − A | false_accept_rate | -0.005 | [-0.014, +0.000] | 0.260 |  |
| C − A | false_reject_rate | +0.026 | [+0.000, +0.064] | 0.237 |  |
| A − B | balanced_accuracy | +0.007 | [-0.004, +0.021] | 0.336 | descriptive only: judge, role and authorship all differ |
| A − B | false_accept_rate | +0.005 | [+0.000, +0.014] | 0.260 |  |
| A − B | false_reject_rate | -0.020 | [-0.046, +0.000] | 0.237 |  |

**Stated reason for INCORRECT verdicts** (share of rejections whose justification matches each pattern)

| condition | cannot verify / no data | asserts the actual value differs | n rejections |
|---|---|---|---|
| A | 45% | 43% | 719 |
| A' | 89% | 25% | 676 |
| B | 94% | 7% | 725 |
| C | 93% | 2% | 726 |

**True-accept rate on correct answers, by source** (synthetic vs genuinely derived)

| condition | natural | synthetic_correct |
|---|---|---|
| A | 0.07 (n=29) | 0.02 (n=122) |
| A' | 0.34 (n=29) | 0.02 (n=121) |
| B | 0.03 (n=29) | 0.00 (n=122) |
| C | 0.00 (n=29) | 0.00 (n=122) |

## Set `natural`: 36 capsules, 29 correct + 575 incorrect answers per condition, 2000 cluster-bootstrap draws

| condition | judge | balanced acc | false-accept (wrong answers) | false-reject (right answers) | raw acc | unparseable | n |
|---|---|---|---|---|---|---|---|
| A | M1 | 0.532 [0.494, 0.576] | 0.005 [0.000, 0.014] | 0.931 [0.843, 1.000] | 0.950 [0.920, 0.973] | 0.0% | 604 |
| A' | M1 | 0.641 [0.462, 0.811] | 0.063 [0.031, 0.101] | 0.655 [0.316, 1.000] | 0.909 [0.865, 0.944] | 0.0% | 604 |
| B | M2 | 0.517 [0.500, 0.539] | 0.000 [0.000, 0.000] | 0.966 [0.922, 1.000] | 0.954 [0.921, 0.979] | 0.0% | 604 |
| C | M2 | 0.500 [0.500, 0.500] | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] | 0.952 [0.917, 0.979] | 0.0% | 604 |

**Verdict counts** (rows: truth label; cells: CORRECT / INCORRECT / UNPARSEABLE verdicts)

| condition | correct answers | incorrect answers |
|---|---|---|
| A | 2 / 27 / 0 | 3 / 572 / 0 |
| A' | 10 / 19 / 0 | 36 / 539 / 0 |
| B | 1 / 28 / 0 | 0 / 575 / 0 |
| C | 0 / 29 / 0 | 0 / 575 / 0 |

**Paired contrasts** (first minus second; 95% cluster-bootstrap CI; two-sided bootstrap p)

| contrast | metric | difference | 95% CI | p | reading |
|---|---|---|---|---|---|
| B − C | balanced_accuracy | +0.017 | [+0.000, +0.039] | 0.707 | H1: trace vs no trace, same judge (M2) |
| B − C | false_accept_rate | +0.000 | [+0.000, +0.000] | 1.000 |  |
| B − C | false_reject_rate | -0.034 | [-0.078, +0.000] | 0.707 |  |
| A − A' | balanced_accuracy | -0.109 | [-0.239, +0.034] | 0.267 | H1/H3: author with vs without its trace |
| A − A' | false_accept_rate | -0.057 | [-0.093, -0.029] | 0.000 |  |
| A − A' | false_reject_rate | +0.276 | [+0.000, +0.537] | 0.237 |  |
| C − A | balanced_accuracy | -0.032 | [-0.076, +0.006] | 0.707 | H2: quarantined auditor vs deployed self-review (headline) |
| C − A | false_accept_rate | -0.005 | [-0.014, +0.000] | 0.260 |  |
| C − A | false_reject_rate | +0.069 | [+0.000, +0.157] | 0.707 |  |
| A − B | balanced_accuracy | +0.015 | [-0.006, +0.037] | 0.707 | descriptive only: judge, role and authorship all differ |
| A − B | false_accept_rate | +0.005 | [+0.000, +0.014] | 0.260 |  |
| A − B | false_reject_rate | -0.034 | [-0.078, +0.000] | 0.707 |  |

**Stated reason for INCORRECT verdicts** (share of rejections whose justification matches each pattern)

| condition | cannot verify / no data | asserts the actual value differs | n rejections |
|---|---|---|---|
| A | 43% | 44% | 599 |
| A' | 90% | 25% | 558 |
| B | 95% | 7% | 603 |
| C | 94% | 2% | 604 |
