---
phase: 4
title: "Full Sweep + Noise Ablation"
status: pending
priority: P1
effort: "3h"
dependencies: [3]
---

# Phase 4: Full Sweep + Noise Ablation

## Overview

Main evidence table for the paper: all 5 datasets × 2 models × 20 seeds × 3 noise levels.
Tests the core hypothesis: TBDC beats class_proportional at higher asymmetric noise rates
(where Type B density diverges further from class frequency).

## Experiment Design

| Parameter | Value |
|-----------|-------|
| Datasets | pima, credit-g, yeast, ecoli, phoneme |
| Models | lr, hgb |
| Seeds | 20: [13,17,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101] |
| Noise levels | low (10%/5%), medium (30%/10%), high (40%/20%) |
| Methods | no_cleaning, global_top_loss, class_proportional, crcc_m, tbdc, oracle_deletion |
| Budget | 10% fixed |
| n_min floor | Skip combo if n_minority < 30 after imbalance induction |

Max rows: 5 × 2 × 20 × 3 × 6 = 3600. After n_min skips: ~2800–3200 rows.

## Pre-Registered Hypotheses

H1 (primary): TBDC BA > class_proportional at medium+high noise in ≥ 60% of valid combos
H2 (noise scaling): TBDC advantage increases with noise level (high > medium > low)
  — because Type B density diverges further from class frequency at higher noise rates
H3 (minority recall): TBDC minority recall ≥ class_proportional at medium+high noise
H4 (CMDR): TBDC CMDR ≤ class_proportional CMDR (by construction — more minority budget)
H5 (negative control): At low noise (10%/5%), TBDC ≈ class_proportional (small signal)

Failure criterion: H1 fails at medium noise → method does not work; report negative result.

## Related Code Files

- Create: `scripts/run_tbdc_sweep.py` (≤ 160 lines)
- Modify: `pipeline/core/config.py` — add noise-level variants as named configs OR pass
  noise rates as constructor args at runtime

## Implementation Steps

1. Create `scripts/run_tbdc_sweep.py`:

```python
"""Full TBDC sweep: 5 datasets × 2 models × 20 seeds × 3 noise levels.
Run: python3 scripts/run_tbdc_sweep.py
Output: outputs/tbdc-sweep-results.csv
"""
SEEDS = [13,17,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101]
NOISE_LEVELS = {
    "low":    {"minority_to_majority_noise": 0.10, "majority_to_minority_noise": 0.05},
    "medium": {"minority_to_majority_noise": 0.30, "majority_to_minority_noise": 0.10},
    "high":   {"minority_to_majority_noise": 0.40, "majority_to_minority_noise": 0.20},
}
DATASETS = ("pima", "credit-g", "yeast", "ecoli", "phoneme")
```

2. For each noise level: create a `BaseExperimentConfig` subclass with those noise rates.
   Pass `noise_level` as a column in each output row.

3. Progress reporting: print progress every 50 combos.

4. Output columns: all standard metric columns PLUS `noise_level` column.

5. Write to `outputs/tbdc-sweep-results.csv`. If file exists, append (to allow resume).

## Analysis to Run After Sweep

After CSV is written:
```python
# H1: TBDC vs class_proportional win rate at medium+high
medium_high = df[df.noise_level.isin(["medium", "high"])]
wins = (tbdc_ba > cp_ba) per (dataset, model, seed, noise_level) combo
win_rate = wins.mean()  # target ≥ 0.60

# H2: noise scaling — TBDC advantage by noise level
for level in ["low", "medium", "high"]:
    delta = (tbdc_ba - cp_ba) at that level  # expect delta_high > delta_medium > delta_low

# H3: recall
# H4: CMDR comparison
```

## Runtime Estimate

5 × 2 × 20 × 3 = 600 combos. Each combo runs 6 methods after one OOF pass.
LR OOF: ~0.3s, HGB OOF: ~0.5s. Total: ~3–4h. Print progress every 50 combos.
Run with `nohup` if needed.

## Success Criteria

- [ ] ≥ 2800 rows in `outputs/tbdc-sweep-results.csv`
- [ ] All 3 noise levels present
- [ ] H1 tested: win rate at medium+high reported
- [ ] H2 tested: noise-level scaling table printed
- [ ] H3 and H4 tested: recall and CMDR reported per noise level
- [ ] Ecoli edge cases (small dataset) handled via n_min floor (NaN rows acceptable)

## Risk Assessment

- If TBDC only beats class_proportional at HIGH noise but not medium: still publishable as
  "TBDC requires noise rate ≥ 20% majority→minority to gain meaningful advantage."
- Ecoli n_minority may fall below 30 at some seeds → NaN rows acceptable, excluded from H1.
- Resume capability: check for existing `outputs/tbdc-sweep-results.csv` and skip completed
  (dataset, model, seed, noise_level) combos.
