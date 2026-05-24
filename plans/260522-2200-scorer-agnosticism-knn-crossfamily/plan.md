---
title: >-
  Full Benchmark Rerun — CUDA, New Models, Scorer Agnosticism, Competitor
  Baselines
description: >-
  Complete GPU-accelerated benchmark sweep with expanded model set, structured
  run categories, scorer agnosticism validation, and head-to-head competitor
  comparisons
status: completed
priority: P1
branch: master
tags: []
blockedBy: []
blocks: []
created: '2026-05-22T11:35:28.709Z'
updatedAt: '2026-05-22'
createdBy: 'ck:plan'
source: skill
---

# Full Benchmark Rerun — CUDA, New Models, Scorer Agnosticism, Competitor Baselines

## Overview

Expanded benchmark plan covering five goals:
1. **CUDA-forced rerun**: all boosting models (XGBoost, LightGBM, CatBoost) run on GPU; auto-detect at startup and abort if CUDA unavailable
2. **New models**: add SVM (RBF + sample_weight → CWMS-ready); reinstate RF/ET; total 9 models
3. **Structured runs**: separate CSVs per scope (Table 1 sweep, scorer comparison, Table 2 competitor head-to-head)
4. **New method dispatchers**: plain SMOTE, k-NN ratio scorer, cross-family OOF scorer, IW-SMOTE (real code), SW (approx), sw-framework
5. **Two final paper tables** with full column/row specs and caption language

## Phases

| Phase | Name | Status |
|-------|------|--------|
| 1 | [Literature Audit](./phase-01-literature-audit.md) | Completed |
| 2 | [Implement Alternative Scorers + New Models](./phase-02-implement-alternative-scorers.md) | Completed |
| 3 | [Structured 4-Category Sweep](./phase-03-comparative-sweep-and-analysis.md) | Completed |
| 4 | [Competitor Baseline Implementation + Head-to-Head](./phase-04-competitor-baselines.md) | Completed |

## GPU Status (verified 2026-05-22)

| Framework | CUDA | Note |
|-----------|------|------|
| XGBoost 3.1.2 | ✅ OK | `device="cuda"` |
| LightGBM 4.6.0 | ✅ OK | `device="gpu"` |
| CatBoost 1.2.8 | ✅ OK | `task_type="GPU"` |
| HGB (sklearn) | ❌ CPU only | No GPU support in sklearn |
| LR / SVM / RF / ET | ❌ CPU only | sklearn CPU-bound |

## Model Roster

| Model | CWMS+MSBS | CUDA | Include |
|-------|-----------|------|---------|
| lr | ✅ Full | ❌ | ✅ |
| svm | ✅ Full (new) | ❌ | ✅ |
| hgb | ✅ Full | ❌ | ✅ |
| lightgbm | ✅ Full | ✅ | ✅ |
| catboost | ✅ Full | ✅ | ✅ |
| random_forest | ✅ sample_weight OK | ❌ | ✅ (restore) |
| extra_trees | ✅ sample_weight OK | ❌ | ✅ (restore) |
| xgboost | ❌ scale_pos_weight conflict | ✅ | ✅ baselines only |
| calibrated_lr | ❌ sklearn routing bug | ❌ | ✅ baselines only |

## Structured Run Categories

### Run A — Internal Benchmark (all models × our methods) → Table 1
Methods: `no_cleaning`, `class_proportional`, `smote`, `msbs`, `cwms`, `cwms_msbs`, `cwms_msbs_shuffled`
Models: lr, svm, hgb, lightgbm, catboost, random_forest, extra_trees (full CWMS); xgboost, calibrated_lr (baselines only)
Protocols: low + medium + high (all 3)
Output: `outputs/full-benchmark-solution.csv`

### Run B — Scorer Comparison (CWMS+MSBS-compatible models)
Methods: `no_cleaning`, `class_proportional`, `cwms_msbs`, `cwms_msbs_knn`, `cwms_msbs_crossfamily`
Models: lr, svm, hgb, lightgbm, catboost
Protocol: medium only
Output: `outputs/scorer-agnosticism-sweep.csv`

### Run C — Competitor Head-to-Head → Table 2
Methods: `no_cleaning`, `class_proportional`, `smote`, `iw_smote`, `sw_framework`, `cwms_msbs`
Models: lr only (consensus classifier across all competitor papers)
Protocol: medium only (ε_mn=0.25 ≈ GK-SMOTE's 20–30%)
Output: `outputs/competitor-headtohead.csv`

## All Method Dispatchers (complete list)

| Method key | Status | Where used |
|------------|--------|-----------|
| `no_cleaning` | ✅ existing | A, B, C |
| `class_proportional` | ✅ existing | A, B, C |
| `smote` | 🔲 Phase 2 | A, C |
| `msbs` | ✅ existing | A |
| `cwms` | ✅ existing | A |
| `cwms_msbs` | ✅ existing | A, B, C |
| `cwms_msbs_shuffled` | ✅ existing | A |
| `cwms_msbs_knn` | 🔲 Phase 2 | B |
| `cwms_msbs_crossfamily` | 🔲 Phase 2 | B |
| `iw_smote` | 🔲 Phase 4 | C |
| `sw_framework` | 🔲 Phase 4 | C |

## Two Final Paper Tables

### Table 1 — Internal Benchmark
**Question answered:** Does CWMS+MSBS work across model families? Does it outperform standard baselines?

```
Model        | no_clean | class_prop | smote | cwms_msbs | ΔBA(vs cp) | p-value   | wins/150
LR           | 0.6XXX   | 0.7032     | 0.XXX | 0.7454    | +4.22pp    | 6.2e-07   | 143/150
SVM          | ...
HGB          | ...
LightGBM     | ...
CatBoost     | ...
RF           | ...
ET           | ...
```
- Aggregated: 3 protocols × 5 datasets × 10 seeds = 150 pairs per row
- Shuffled ablation sub-table: cwms_msbs vs cwms_msbs_shuffled per model (confirms OOF scores load-bearing)
- XGBoost + calibrated_lr footnoted (structural exclusions from cwms_msbs)

### Table 2 — External Comparison
**Question answered:** Does CWMS+MSBS outperform published noise-robust SMOTE methods?

```
Method              | Source              | BA     | G-mean | Min.Recall
no_cleaning         | —                   | 0.XXX  | 0.XXX  | 0.XXX
SMOTE               | [Chawla 2002]       | 0.XXX  | 0.XXX  | 0.XXX
class_proportional  | [He & Garcia 2009]  | 0.XXX  | 0.XXX  | 0.XXX
IW-SMOTE            | [Zhang et al. 2022] | 0.XXX  | 0.XXX  | 0.XXX
SW-approx†          | [X et al. 2022]     | 0.XXX  | 0.XXX  | 0.XXX
CWMS+MSBS (ours)    | —                   | 0.XXX  | 0.XXX  | 0.XXX
```
- LR classifier; hidden_minority_medium (ε_mn=0.25); 5 datasets × 10 seeds = 50 pairs
- G-mean = sqrt(minority_recall × majority_recall) — derived, no re-run needed
- †SW reproduced as k-NN label inconsistency approximation (no public code for original)
- GK-SMOTE (APWeb-WAIM 2025): cited in text, not reproduced — no public code; symmetric noise incompatible with our asymmetric protocol

## Key Context

- **Baseline result**: LR+CWMS+MSBS BA=0.7454 (+4.22pp vs class_proportional, p=6.2e-7)
- **Literature**: Both Q1 (k-NN ratio) and Q2 (cross-family OOF) CONFIRMED NOVEL
- **Closest competitors**: SW (2022) 52-60% overlap, IW-SMOTE (2022) 60% overlap
- **Gap audit**: [gap-audit-q1q2-astar-2020-2025.md](../reports/gap-audit-q1q2-astar-2020-2025.md)
- **Conda env**: `/home/than-minh/miniconda3/envs/dsp/bin/python`
- **Existing sweep**: `outputs/cwms-msbs-deep-sweep.csv` (4350 rows, CPU, no SVM/RF/ET)

## Dependencies

Phases 3 and 4 depend on Phase 2 (new models registered in factories).
