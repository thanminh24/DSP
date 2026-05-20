---
phase: 3
title: "Method B Pilot — OOF-Filtered SMOTE"
status: pending
effort: "2h"
---

# Phase 3: Method B Pilot

## Overview

Test whether **OOF-filtered SMOTE** (drop top-5% suspect Type B from minority seed
pool before synthesis) beats `class_proportional` and `plain_smote` on 4 datasets
× 2 models × 20 seeds at medium noise (30%/10%).

**CRITICAL ABANDONMENT GATE:** If Phase 2 (Method A) FAIL AND this phase FAIL,
write the negative result summary and STOP. Do not proceed to Phase 4 or Phase 5.

## Context Links

- Method B implementation: phase-01 → `pipeline/augmentation/filtered_smote.py`
- Runner: phase-01 → `scripts/run_augment_experiment.py`
- Phase 2 verdict: `outputs/pilot-method-a-decision.md`

## Hypothesis

H1: Removing top-5% minority pool by OOF loss (high Type B precision in that band
per diagnostic study) before SMOTE produces a cleaner seed set; synthetic samples
land closer to the true minority manifold; BA exceeds `plain_smote` AND
`class_proportional`.

H0: OOF filtering removes both Type B noise AND hard-but-clean minority samples;
the trade-off cancels. `oof_filtered_smote ≈ plain_smote`.

## Methods Under Test

| Method | Role |
|--------|------|
| `no_cleaning` | Lower baseline |
| `class_proportional` | Strong baseline (the bar to beat) |
| `plain_smote` | SMOTE baseline (no filtering) |
| `oof_filtered_smote` | **Method B** |

## Requirements

- Use Phase 1 `run_single_augment`.
- Same 20 seeds, datasets, models, noise as Phase 2.
- `filter_frac=0.05` (5% of minority pool).
- `sampling_strategy="auto"` (balance to majority count).
- `k_neighbors=5` (default; guarded down if n_minority_kept < 6).
- Output: `outputs/pilot-method-b-results.csv`.

## Related Code Files

**Create:**
- `scripts/run_pilot_method_b.py` (≤ 100 lines)
- `outputs/pilot-method-b-results.csv`
- `outputs/pilot-method-b-decision.md`
- (Conditionally) `outputs/negative-result-summary.md`

**Modify:** none.

## Implementation Steps

### Step 1 — `scripts/run_pilot_method_b.py`

```python
"""Pilot for Method B — OOF-filtered SMOTE."""
from __future__ import annotations
import itertools
import pandas as pd
from pipeline.core.config import BaseExperimentConfig
from scripts.run_augment_experiment import run_single_augment

DATASETS = ["pima", "credit-g", "yeast", "phoneme"]
MODELS = ["lr", "hgb"]
SEEDS = [13,17,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101]
METHODS = ["no_cleaning", "class_proportional", "plain_smote", "oof_filtered_smote"]
OUT_CSV = "outputs/pilot-method-b-results.csv"


def main():
    cfg = BaseExperimentConfig()
    rows = []
    total = len(DATASETS) * len(MODELS) * len(SEEDS)
    done = 0
    for ds, mdl, sd in itertools.product(DATASETS, MODELS, SEEDS):
        try:
            rows.extend(run_single_augment(ds, mdl, sd, METHODS, cfg))
        except Exception as e:
            print(f"FAIL {ds}/{mdl}/{sd}: {e}")
        done += 1
        if done % 8 == 0:
            print(f"  progress: {done}/{total}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"wrote {len(df)} rows → {OUT_CSV}")
    _print_summary(df)


def _print_summary(df: pd.DataFrame):
    pivot = df.pivot_table(
        index=["dataset","model"], columns="method",
        values="balanced_accuracy", aggfunc="mean")
    pivot["d_cp"] = pivot["oof_filtered_smote"] - pivot["class_proportional"]
    pivot["d_plain"] = pivot["oof_filtered_smote"] - pivot["plain_smote"]
    print("\n── Method B BA summary (mean over 20 seeds) ──")
    print(pivot.round(4).to_string())
    wins_cp = (pivot["d_cp"] > 0).sum()
    print(f"\nMethod B wins vs class_proportional: {wins_cp} / {len(pivot)}")
    print(f"Mean Δ vs class_proportional: {pivot['d_cp'].mean():+.4f}")
    print(f"Mean Δ vs plain_smote:        {pivot['d_plain'].mean():+.4f}")


if __name__ == "__main__":
    main()
```

### Step 2 — run

```bash
cd /home/than-minh/project/DSP
/home/than-minh/miniconda3/bin/python3 scripts/run_pilot_method_b.py 2>&1 | tee outputs/pilot-method-b-log.txt
```

### Step 3 — apply hard gate

```python
from scipy.stats import wilcoxon
import numpy as np, pandas as pd
df = pd.read_csv("outputs/pilot-method-b-results.csv")
pivot = df.pivot_table(index=["dataset","model","seed"], columns="method",
                      values="balanced_accuracy").reset_index()
diff = pivot["oof_filtered_smote"] - pivot["class_proportional"]
W, p = wilcoxon(diff.dropna())
wins = pivot.groupby(["dataset","model"]).apply(
    lambda g: g["oof_filtered_smote"].mean() > g["class_proportional"].mean()).sum()

def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return (a.mean() - b.mean()) / np.sqrt(((a.std(ddof=1)**2 + b.std(ddof=1)**2)/2))
mean_d = float(np.mean([cohens_d(g["oof_filtered_smote"], g["class_proportional"])
                        for _, g in pivot.groupby(["dataset","model"])]))
print(f"wins {wins}/8  p={p:.4g}  mean_d={mean_d:.3f}")
diff2 = pivot["oof_filtered_smote"] - pivot["plain_smote"]
W2, p2 = wilcoxon(diff2.dropna())
print(f"vs plain_smote: p={p2:.4g}  delta={diff2.mean():+.4f}")
```

**Gate logic (same as Phase 2):**

```
GO       : wins ≥ 6/8 AND mean_d ≥ 0.3 AND p < 0.05
           AND  BA(oof_filtered_smote) > BA(plain_smote) (filtering adds real value)
PARTIAL  : wins ≥ 5/8 AND mean_d ≥ 0.2 AND delta vs plain > 0
FAIL     : otherwise
```

Note: For Method B, the secondary comparison is vs `plain_smote` (not vs a random
filter). The reason: filtering 5% by OOF is implicitly testing whether OOF
suspiciousness in the minority pool is informative for Type B detection. If
`oof_filtered_smote ≈ plain_smote`, the OOF filter contributed nothing.

### Step 4 — abandonment decision

After this phase completes, evaluate the joint state:

| Method A | Method B | Action |
|----------|----------|--------|
| GO or PARTIAL | any | Proceed to Phase 4 with surviving methods |
| any | GO or PARTIAL | Proceed to Phase 4 with surviving methods |
| FAIL | FAIL | **STOP.** Write `outputs/negative-result-summary.md`, do not run Phase 4–5 |

### Step 5 — `outputs/negative-result-summary.md` template (if abandoning)

```markdown
# Negative Result: Cleaning under Asymmetric Class-Dependent Noise

## Setup
- Datasets: pima, credit-g, yeast, phoneme
- Models: lr, hgb
- 20 seeds; medium noise 30% min→maj, 10% maj→min; minority ratio 15%
- Budget 10%

## Methods Tested (all FAIL vs class_proportional)
1. Deletion-based: CRCC-P (= class_proportional), CRCC-M, CRCC-Adaptive, TBDC, OOF relabel @ 0.65–0.95
2. Type A relabeling via balanced OOF (Phase 2) — see pilot-method-a-results.csv
3. OOF-filtered SMOTE (Phase 3) — see pilot-method-b-results.csv

## Numerical evidence
[insert pivot tables from both pilots]

## Structural cause
OOF unreliability at n_minority ~ 75 (15% of typical n_train ~ 500). At this scale:
- Standard OOF: majority-biased; useful only in top 5–8% of minority pool.
- Balanced OOF: relabel correctness too low (insert observed value) to beat random.
- SMOTE filtering: removes both Type B and clean-but-hard minority; net null.

## Implication
Per-class budget allocation improves only when the noise-density estimator has
sufficient calibration quality — i.e. larger minority pool. At n_minority ≤ 100
under 30%/10% asymmetric noise, `class_proportional` is the practical ceiling.

## Decision
Abandon this research direction. Do not proceed to Phase 4 or Phase 5.
```

## Todo List

- [ ] Write `scripts/run_pilot_method_b.py`
- [ ] Run pilot; produce `outputs/pilot-method-b-results.csv`
- [ ] Compute Wilcoxon + Cohen's d + wins (vs class_proportional AND vs plain_smote)
- [ ] Write `outputs/pilot-method-b-decision.md` with verdict
- [ ] Evaluate joint Method A + Method B state
- [ ] If both FAIL: write `outputs/negative-result-summary.md`, mark plan cancelled
- [ ] If at least one survives: proceed to Phase 4

## Success Criteria

- [ ] CSV has 4 × 2 × 20 × 4 = 640 rows (< 5% loss tolerated)
- [ ] Pivot summary printed
- [ ] Decision document written with verdict
- [ ] Joint A+B decision documented
- [ ] If abandoning: negative-result-summary.md written; Phase 4/5 marked cancelled

## Test Matrix

| Check | Pass condition |
|-------|----------------|
| Schema | 14 columns; `n_synthetic` populated for SMOTE rows |
| No NaN BA | < 5% rows NaN |
| SMOTE applied | `n_synthetic > 0` for ≥ 95% of plain_smote and oof_filtered_smote rows |
| Filter size | for oof_filtered_smote: ~ 0.05 × n_minority samples filtered (logged separately) |

## Risk Assessment

| Risk | L | I | Mitigation |
|------|---|---|----|
| SMOTE n_minority_kept < 6 → no synthesis | M | M | Phase 1 guard returns unsynthesized (X_keep, y_keep) |
| Method B beats class_proportional but not plain_smote | M | H | Gate requires delta vs plain_smote |
| Both pilots fail and direction is abandoned | M | H | Negative result is the deliverable; documented and submission-ready |
| imbalanced-learn not installed | L | H | Phase 1 install step is prerequisite |
| Train/test SMOTE leakage | L | H | Verified: SMOTE only on X_tr, y_noisy in Phase 1 runner |

## Rollback Plan

Read-only on existing code. Discard CSV; remove negative-result file if written
prematurely.

## Next Steps

- If at least one method GO/PARTIAL → Phase 4 (full sweep) on surviving methods
- If both FAIL → STOP. Phase 4/5 cancelled. Negative result is the final artifact.
