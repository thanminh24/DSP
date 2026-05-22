---
title: "CWMS+MSBS v2: Boosting Fix, Deep Sweep, Cleanup"
description: "Fix the double-correction bug in CWMS for XGBoost/LightGBM, run a full multi-metric deep sweep across all 6 model families, clean dead code and old scripts, archive superseded outputs, drop the venv, and update docs to reflect the new primary method."
status: pending
priority: P1
branch: "master"
tags: ["cwms-msbs", "boosting-fix", "deep-sweep", "cleanup", "conda"]
blockedBy: []
blocks: []
created: "2026-05-22T08:13:25.302Z"
createdBy: "ck:plan"
source: skill
---

# CWMS+MSBS v2: Boosting Fix, Deep Sweep, Cleanup

## Context

Full sweep results (Phase 1 of previous plan, 5 models × 5 datasets × 10 seeds) showed:

| Model | cwms_msbs vs class_prop | p | verdict |
|-------|------------------------|---|---------|
| lr | +4.22pp | 6.2e-07 | **Strong win** |
| hgb | +0.25pp | 0.53 | Neutral |
| lightgbm | −0.40pp | 0.71 | Neutral |
| calibrated_lr | −0.83pp | 6.4e-03 | **Significantly worse** |
| xgboost | −1.93pp | 4.0e-03 | **Significantly worse** |

**Root causes identified:**

1. **XGBoost double-correction**: `std_factory` uses `scale_pos_weight=spw` (class ratio) AND
   CWMS applies `sample_weight` suppression. The two corrections compound: minority samples are
   upscaled by `spw` while suspicious majority is suppressed by CWMS, leading to over-correction.
   Fix: use `scale_pos_weight=1.0` and fold the class balance signal into CWMS weights directly
   (`weights[minority] = spw`, `weights[majority] = max(1-score, 0)`).

2. **calibrated_lr sample_weight routing**: sklearn bug — `CalibratedClassifierCV` applies
   `sample_weight` only at the calibration stage, not to the base LR. CWMS weights distort
   calibration without correcting the boundary. Fix: exclude `calibrated_lr` from CWMS-based
   methods; report it as a known limitation.

3. **LightGBM**: `std_factory` has `class_weight=None` — no built-in correction, so neutral
   result. Should benefit from class-balanced CWMS like HGB.

4. **catboost**: previously failed with sklearn 1.6 incompatibility. This was in the old `.venv`.
   The `dsp` conda env (catboost 1.2.8, sklearn 1.6.1) is confirmed compatible — catboost
   pipeline passes smoke test. Must be included in the new sweep.

5. **Metrics gap**: current sweep tracks only `balanced_accuracy`, `macro_f1`, `minority_recall`.
   Paper needs: `accuracy`, `weighted_f1`, `minority_precision`, `majority_recall` for a
   complete picture of method behavior under imbalance.

## What We Are Building

- **`pipeline/baselines/soft_weighting.py`**: add `confidence_weighted_sample_weights_balanced()`
  that folds `scale_pos_weight` into CWMS minority weights. The updated `cwms` and `cwms_msbs`
  dispatchers in the sweep script will call this when using boosting models.
- **`pipeline/evaluation/augment_metrics.py`**: add `accuracy`, `weighted_f1`,
  `minority_precision`, `majority_recall` to the returned dict.
- **`scripts/run_cwms_msbs_deep_sweep.py`**: new sweep runner, replaces the old full-sweep
  script, runs with conda `dsp` env. 6 models × 5 datasets × 10 seeds × 3 protocols × 5
  methods = **4,500 rows** in `outputs/cwms-msbs-deep-sweep.csv`.
- **`scripts/analyze_cwms_msbs_deep_results.py`**: updated analysis that reports all metrics.

## Conda Environment

All scripts run under: `/home/than-minh/miniconda3/envs/dsp/bin/python`

Confirmed packages: pandas 2.3.3, sklearn 1.6.1, xgboost 3.1.2, lightgbm 4.6.0,
catboost 1.2.8 (**works** with sklearn 1.6.1 in this env).

The `.venv/` folder (1.6 GB) is deleted in Phase 4.

## Model Scope

| Model | Family | cwms_msbs treatment | In sweep |
|-------|--------|---------------------|----------|
| `lr` | Linear | Standard CWMS | ✓ |
| `calibrated_lr` | Linear (calibrated) | **Excluded from cwms/cwms_msbs** — report `no_cleaning` and `class_proportional` only for this model; sklearn routing bug | Partial (2 methods) |
| `hgb` | Boosting | Class-balanced CWMS (no spw) | ✓ |
| `xgboost` | Boosting | Class-balanced CWMS (spw=1.0 factory, spw folded into weights) | ✓ |
| `lightgbm` | Boosting | Class-balanced CWMS (spw=1.0 factory) | ✓ |
| `catboost` | Boosting | Class-balanced CWMS (spw=1.0 factory) | ✓ |

## Phases

| Phase | Name | Status | Effort |
|-------|------|--------|--------|
| 1 | [Fix CWMS for Boosting and Extend Metrics](./phase-01-fix-cwms-for-boosting-and-extend-metrics.md) | Pending | ~1h |
| 2 | [Deep Multi-Metric Benchmark Sweep](./phase-02-deep-multi-metric-benchmark-sweep.md) | Pending | ~3h compute |
| 3 | [Repo Cleanup](./phase-03-repo-cleanup.md) | Pending | ~30m |
| 4 | [Archive Outputs and Drop venv](./phase-04-archive-outputs-and-drop-venv.md) | Pending | ~15m |
| 5 | [Update Docs](./phase-05-update-docs.md) | Pending | ~30m |

## Dependencies

Phase 2 depends on Phase 1 (new metrics and weight fix must be in place before sweeping).
Phases 3–5 are independent of each other and can run after Phase 2, or even in parallel with it.
