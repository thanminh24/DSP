---
phase: 3
title: "Pilot Go/No-Go"
status: pending
priority: P1
effort: "30min"
dependencies: [2]
---

# Phase 3: Pilot Go/No-Go

## Overview

Run all 5 datasets × 2 models × 20 seeds at medium noise (30%/10%), compare TBDC vs
class_proportional on BA and minority recall. Gate for Phases 4+5. If TBDC fails here,
report as negative result and stop.

## Go/No-Go Threshold

**GO**: TBDC BA > class_proportional BA with Cohen's d ≥ 0.3 AND Wilcoxon p < 0.05
(paired by seed) in ≥ 6/8 valid (dataset×model) combos (ecoli excluded if n_minority < 30).
AND TBDC minority recall ≥ class_proportional recall in ≥ 6/8 combos.

**PARTIAL GO**: d ≥ 0.2 and p < 0.10 in ≥ 5/8 combos. Proceed to Phase 4 scoped to
the passing metric only; report as "moderate evidence."

**NO-GO**: Fails both thresholds. Document as negative result. The mechanism: OOF
disagreement rate overestimates Type B at n_minority~75, causing over-deletion of clean
minority. This is still a publishable negative result (Type B-directed allocation fails
at standard imbalance ratios due to unreliable OOF calibration).

## Related Code Files

- Create: `scripts/run_pilot_tbdc.py` (≤ 100 lines)
- Read: existing `scripts/run_full_experiment.py` for loop structure

## Implementation Steps

1. Write `scripts/run_pilot_tbdc.py`:

```python
"""Pilot: TBDC vs class_proportional at medium noise, 20 seeds, all datasets.
Go/no-go gate for full sweep. Run: python3 scripts/run_pilot_tbdc.py
"""
SEEDS = [13,17,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101]
DATASETS = ["pima", "credit-g", "yeast", "phoneme"]  # ecoli separate
METHODS = ["class_proportional", "tbdc", "crcc_m", "no_cleaning"]
```

   - Reuse `run_single()` from `pipeline.core.experiment` — it already dispatches all methods
   - Override only: `datasets = DATASETS`, `seeds = SEEDS`
   - Write results to `outputs/pilot-tbdc-results.csv`

2. Print go/no-go table:
   - For each (dataset, model): mean BA and recall for class_proportional vs tbdc
   - Flag combos where tbdc > class_proportional (BA and recall)
   - Count wins; print GO/NO-GO decision

3. Update `plan.md` Phase 3 status after running.

## Pilot Script Structure

```python
from pipeline.core.config import BaseExperimentConfig
from pipeline.core.experiment import run_single
import pandas as pd

@dataclass(frozen=True)
class PilotConfig(BaseExperimentConfig):
    datasets: tuple = ("pima", "credit-g", "yeast", "phoneme")
    seeds: tuple = (13,17,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101)

cfg = PilotConfig()
rows = []
for ds in cfg.datasets:
    for model in ("lr", "hgb"):
        for seed in cfg.seeds:
            rows.extend(run_single(ds, model, seed, cfg))

df = pd.DataFrame(rows)
df.to_csv("outputs/pilot-tbdc-results.csv", index=False)

# Go/no-go summary
for ds in cfg.datasets:
    for model in ("lr", "hgb"):
        sub = df[(df.dataset==ds) & (df.model==model)]
        cp = sub[sub.method=="class_proportional"]["balanced_accuracy"].mean()
        tb = sub[sub.method=="tbdc"]["balanced_accuracy"].mean()
        cp_r = sub[sub.method=="class_proportional"]["minority_recall"].mean()
        tb_r = sub[sub.method=="tbdc"]["minority_recall"].mean()
        ba_win = "TBDC>" if tb > cp else "prop>=" if cp >= tb else "tie"
        rc_win = "TBDC>" if tb_r > cp_r else "prop>=" if cp_r >= tb_r else "tie"
        print(f"{ds}/{model}: BA={cp:.4f}→{tb:.4f} {ba_win}  recall={cp_r:.4f}→{tb_r:.4f} {rc_win}")
```

## Expected Runtime

4 datasets × 2 models × 20 seeds = 160 combos. Each combo: one OOF pass + all methods.
LR ~0.3s, HGB ~0.5s per combo. Total: ~90s. Target: < 5 min.

## Success Criteria

- [ ] Script runs without error, produces `outputs/pilot-tbdc-results.csv`
- [ ] All 160 combos (4 datasets × 2 models × 20 seeds) present in CSV
- [ ] Go/no-go table printed with per-combo BA and recall comparison
- [ ] Decision documented: GO / PARTIAL GO / NO-GO
- [ ] If GO: update plan.md to mark Phase 3 complete, proceed to Phase 4
- [ ] If NO-GO: document failure mode and stop — do NOT proceed to Phase 4

## Risk Assessment

- If NO-GO: the negative result is still publishable. The finding would be:
  "Disagreement-rate-based allocation is unreliable at n_minority~75; CL thresholding
  at severe imbalance over-estimates Type B due to OOF calibration failure." This joins
  the oracle paradox and lambda insensitivity as empirical findings about what does NOT work.
