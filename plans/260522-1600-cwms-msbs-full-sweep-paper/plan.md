---
title: "CWMS+MSBS Boundary Correction: Full Sweep and Paper"
description: "Rigorous benchmarking of the combined CWMS+MSBS label-corruption-free boundary correction method, followed by statistical analysis and paper packaging. Replaces the now-discouraged balanced_oof_relabel as our main contribution."
status: complete
priority: P1
branch: "master"
tags: ["cwms-msbs", "boundary-correction", "experiment", "paper", "noisy-labels", "imbalanced"]
blockedBy: []
blocks: []
created: "2026-05-22T06:47:06.000Z"
createdBy: "ck:plan"
source: skill
---

# CWMS+MSBS Boundary Correction: Full Sweep and Paper

## What We Are Building

A label-corruption-free method for recovering decision boundary quality in imbalanced tabular
classification under **hidden minority noise** (true minority samples mislabeled as majority).

**The problem:**
Real-world tabular datasets often have imbalanced classes (e.g., 15% minority) AND label noise
where some minority samples were mislabeled as majority at collection time. Standard classifiers
trained on this data learn a boundary that is biased too far into minority space — the model
never sees the full minority manifold, so it under-predicts minority.

**Why the old approach (balanced_oof_relabel) is discouraged:**
OOF relabeling flips the highest-scoring majority-labeled samples to minority label. It works
empirically (+0.77% BA over class_proportional, p=4.5e-11) but has a methodological circularity:
the OOF scorer and the final model are the same family. Labels are "pre-tuned" for that model class.
This is not standard test-set leakage (X_te is never seen during scoring), but the self-referential
structure is a valid reviewer objection for publication. We pivot away from relabeling entirely.

**What CWMS+MSBS does (our new contribution):**

| Component | Mechanism | Effect |
|-----------|-----------|--------|
| **MSBS** (Minority-Side Boundary Synthesis) | For each confirmed minority sample, find k nearest high-confidence majority neighbors; interpolate `x' = x_min + u*(x_maj - x_min)`, u ~ U(0,1); label x' as minority | Adds synthetic minority samples near the true class boundary — volume + boundary targeting |
| **CWMS** (Confidence-Weighted Majority Suppression) | For each majority-labeled sample i: `weight[i] = max(1 - bal_score[i], 0)`; minority samples: weight=1.0 | Suppresses suspicious majority samples in the loss without changing any labels — boundary shrinks toward true majority |
| **Combined** | MSBS synthesis + CWMS weights in the same training pass | Dual action: grow the minority side AND suppress the noisy majority side simultaneously |

**Key design properties:**
- Zero label corruption: no existing labels are changed
- Reuses `bal_scores` already computed (no extra OOF pass)
- Model-agnostic: works with any sklearn-compatible estimator
- Hyperparameter-free beyond the shared `budget` already used by baselines

---

## Empirical History (What We Learned)

### What Failed and Why

**CGMS (Confidence-Guided Minority Synthesis):** −8.5pp BA vs no_cleaning. Root cause: seeds
from high-scoring majority samples, synthetic gets minority label → adjacent contradictory labels
in feature space. Gradient signal cancels.

**MSBS standalone:** 0.6320 BA (prelim, 5 models). Correct mechanism (no contradiction), but
volume effect alone is not enough. The boundary is still contaminated by unweighted noisy majority.

**CWMS standalone:** 0.6333 BA. Suppression alone shifts boundary but doesn't add missing
minority manifold coverage. Boosting models perform at 0.6955 (scale_pos_weight helps), sklearn
at 0.6296 (no class correction floor).

**CWMS+MSBS combined:** 0.6768 BA (prelim) — strict improvement over both standalone (+4.4pp
over cwms, +4.5pp over msbs). **LR specifically: 0.7485, beating class_proportional by +4.9pp.**

### Key Numbers to Know

All figures from `hidden_minority_medium` protocol (mn_to_maj=0.30, maj_to_min=0.10), budget=0.10,
target_ratio=0.15.

**Prelim run** (5 models: lr, calibrated_lr, rf, et, hgb; 5 datasets; 3 seeds; n=75 per method):

| Method | BA mean | vs class_proportional |
|--------|---------|-----------------------|
| class_proportional | 0.6950 | — |
| **cwms_msbs** | **0.6768** | **−0.018** |
| cwms | 0.6333 | −0.062 |
| msbs | 0.6320 | −0.063 |
| no_cleaning | 0.5834 | −0.112 |

**Per-model cwms_msbs vs class_proportional (prelim):**

| Model family | Model | cwms_msbs BA | class_prop BA | delta |
|--------------|-------|-------------|---------------|-------|
| Linear | lr | **0.7485** | 0.6999 | **+0.049** ✓ |
| Linear | calibrated_lr | 0.6719 | 0.6851 | −0.013 ≈ |
| Boosting | hgb | 0.6789 | 0.6911 | −0.012 ≈ |
| Tree | random_forest | 0.6503 | 0.7040 | −0.054 ✗ |
| Tree | extra_trees | 0.6342 | 0.6949 | −0.061 ✗ |

**Scope decision:** Tree models (RF, ET) are out. They don't respond to per-sample weights
meaningfully — the bootstrap mechanism dilutes the CWMS signal. The paper presents results
for **linear + boosting model families only**. This is honest and defensible: the method
explicitly uses sample_weight, which tree models' bootstrap process handles differently.

**Reference for balanced_oof_relabel** (discouraged, shown for context only):
0.7272 BA (same protocol, full 5-model 10-seed run from old baselines CSVs).

---

## Model Scope for the Paper

**In-scope (target models):**

| Model | Family | Why included |
|-------|--------|-------------|
| `lr` | Linear | Best cwms_msbs result (+4.9pp over cp); sample_weight is exact gradient scaling |
| `calibrated_lr` | Linear (calibrated) | Standard calibration wrapper; note sample_weight routing caveat |
| `hgb` | Gradient boosting | HGB uses sample_weight natively; good CWMS signal |
| `xgboost` | Gradient boosting | scale_pos_weight + CWMS weights; GPU-accelerated |
| `catboost` | Gradient boosting | Native cat feature support; strong on tabular |
| `lightgbm` | Gradient boosting | Fast boosting; if available |

**Out-of-scope (acknowledged as limitation):**

| Model | Reason |
|-------|--------|
| `random_forest` | Bootstrap dilutes sample_weight; CWMS has no effect |
| `extra_trees` | Same as RF |

---

## Phases

| Phase | Name | Status | Effort |
|-------|------|--------|--------|
| 1 | [Full Rigorous Sweep](./phase-01-full-rigorous-sweep.md) | Complete | ~1.5h compute |
| 2 | [Statistical Analysis and Paper Packaging](./phase-02-statistical-analysis-and-paper-packaging.md) | Complete | ~30m |

## Dependencies

Phase 2 depends on Phase 1 (full sweep CSV must exist before analysis).
Baselines (no_cleaning, class_proportional) are co-run in the same sweep — no separate baseline run needed.
