---
phase: 2
title: "Full Imbalance Sweep"
status: pending
priority: P1
effort: "4h"
dependencies: [1]
---

# Phase 2: Full Imbalance Sweep

## Overview

Run all methods across 8 imbalance ratios, 5 datasets (4 real + 1 synthetic), 20 seeds, 2 models.
This is the paper's main evidence table. Wider than any prior ablation in this project.

## Experiment Design

| Parameter | Value |
|-----------|-------|
| Ratios | [0.02, 0.03, 0.05, 0.07, 0.08, 0.10, 0.12, 0.15] |
| Datasets | pima, credit-g, yeast, phoneme, synthetic |
| Models | lr, hgb |
| Seeds | 20: [13,17,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101] |
| Methods | no_cleaning, global_top_loss, class_proportional, crcc_m, crcc_adaptive, majority_only |
| n_min floor | 30 — skip combo if below |

Max rows: 8 × 5 × 2 × 20 × 6 = 9600. After skips: ~6000–7000 rows.

## Synthetic Dataset

Controlled dataset where ground truth is known exactly. Generated in-script, no file.

```python
def make_synthetic(n_total: int, minority_ratio: float, seed: int):
    rng = np.random.default_rng(seed)
    n_min = int(n_total * minority_ratio)
    n_maj = n_total - n_min
    X_min = rng.normal([2.0] + [0.0]*9, 1.0, size=(n_min, 10))
    X_maj = rng.normal([0.0]*10, 1.0, size=(n_maj, 10))
    X = np.vstack([X_min, X_maj])
    y = np.array([1]*n_min + [0]*n_maj)
    return X, y  # no cat_cols
```

n_total=3000 fixed; only minority_ratio varies. AUC baseline ~0.80. Same noise as real data.
Purpose: if CRCC-Adaptive fails on synthetic, the problem is fundamental, not dataset-specific.

## ir_threshold Sensitivity

Secondary pass at ratio=0.05 only, ir_threshold ∈ {10, 15, 20, 25}.
Methods: crcc_adaptive vs class_proportional only. ~200 extra rows.
Output: `outputs/ir-threshold-sensitivity.csv`
Pass if: recall gain std < 0.005 across thresholds.

## Script Layout

Two files to stay under 200 lines each:
- `pipeline/data/synthetic.py` — make_synthetic() function only (~20 lines)
- `scripts/run_imbalance_sweep.py` — main loop (~160 lines)

Core loop in run_imbalance_sweep.py:

```python
SEEDS = [13,17,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101]
RATIOS = [0.02, 0.03, 0.05, 0.07, 0.08, 0.10, 0.12, 0.15]
MIN_MINORITY = 30

# For each (ratio, dataset, model, seed):
#   1. Load/generate data
#   2. Induce imbalance to ratio
#   3. Skip if n_minority < MIN_MINORITY
#   4. Inject noise
#   5. Compute suspiciousness (out_of_fold_loss)
#   6. Run all 6 selectors
#   7. Evaluate each, append row
```

## Pre-Registered Hypotheses

H1: crcc_adaptive recall > class_proportional at ratio ≤ 0.07 in ≥ 60% eligible combos  
H2: majority_only recall ≤ crcc_adaptive recall (adaptive beats trivial rule)  
H3: crcc_adaptive recall ≥ class_proportional at ratio ≥ 0.10 (no regression at moderate IR)  
H4: ir_threshold recall gain std < 0.005 across {10,15,20,25}  
H5: crcc_adaptive beats class_proportional on synthetic at ratio ≤ 0.07 (must pass)

Failure criterion: if phoneme 5% recall gain < 0.02 across 20 seeds → method fails; report negative.

## Related Code Files

- Create: `scripts/run_imbalance_sweep.py` (≤ 160 lines)
- Create: `pipeline/data/synthetic.py` (~20 lines)
- Read: `pipeline/core/experiment.py`
- Read: `pipeline/scoring/oof_loss.py`

## Success Criteria

- [ ] ≥ 5000 rows produced across all datasets
- [ ] Synthetic rows present in output
- [ ] ir-threshold sensitivity CSV produced
- [ ] crcc_adaptive and majority_only both in results
- [ ] Scripts ≤ 200 lines each

## Runtime Estimate

~6000 combos × 5-fold = ~30,000 LR fits. LR ~0.05s each = 25 min.
HGB ~0.1s = 50 min. Total ~1.5–2.5h depending on hardware.
Print progress every 100 combos.
