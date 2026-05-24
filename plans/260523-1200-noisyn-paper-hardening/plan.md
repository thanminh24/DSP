---
title: "NoiSyn Paper Hardening — Dataset Expansion, Ablations, Stat Corrections"
description: "Address red-team findings to make NoiSyn defensible at Pattern Recognition / KBS tier"
status: in-progress
priority: P1
branch: "master"
tags: ["paper", "experiments", "ablations"]
blockedBy: []
blocks: []
created: "2026-05-23T03:56:41.364Z"
updatedAt: "2026-05-23"
createdBy: "ck:plan"
source: skill
---

# NoiSyn Paper Hardening — Dataset Expansion, Ablations, Stat Corrections

## Overview

Red-team review identified 5 critical, 4 high, and 6 moderate issues with the current NoiSyn paper draft. This plan addresses all of them systematically. The goal: a paper defensible at Pattern Recognition / Knowledge-Based Systems / IEEE TNNLS.

The LR signal is real and strong (d=0.816, p=6.1e-15). The work here is about scope, framing, and honest statistics — not changing the core claim.

## Phases

| Phase | Name | Status | Effort |
|-------|------|--------|--------|
| 1 | [Quick Fixes (no experiments)](./phase-01-quick-fixes-no-experiments.md) | **Completed** | 1h |
| 2 | [Fast New Sweeps](./phase-02-fast-new-sweeps.md) | **Completed** | 2.5h |
| 3 | [Dataset Expansion](./phase-03-dataset-expansion.md) | **Completed** (24,750 rows) | 8h |
| 4 | [IR Sweep](./phase-04-ir-sweep.md) | **Completed** (8,250 rows, Table 6 updated) | 4h |
| 5 | [Expanded External Comparison](./phase-05-expanded-external-comparison.md) | **Completed** (8,100 rows) | 2h |
| 6 | [Updated Analysis and Paper](./phase-06-updated-analysis-and-paper.md) | **Completed** (all tables updated; final 24,750-row numbers confirmed) | 3h |

## Red-Team Issue Tracker

| ID | Issue | Severity | Phase | Fixed by |
|----|-------|----------|-------|----------|
| C1 | Table 2: only LR × medium × 1 budget | Critical | 5 | Expand to LR+SVM+HGB × 3 protocols |
| C2 | 5 datasets (venue needs 15–27) | Critical | 3 | Add 10 more UCI/KEEL datasets |
| C3 | Only IR=0.15 tested | Critical | 4 | Add IR=0.30 sweep |
| C4 | Symmetric/reverse protocols never tested | Critical | 2 | Run failure-mode protocols |
| C5 | 150 pairs treated as i.i.d. | Critical | 1, 6 | Per-dataset Wilcoxon + Stouffer's Z combination |
| H1 | SW-approx validation missing | High | 2 | Validate or remove from Table 2 |
| H2 | RF/ET regression unexplained | High | 2 | CWMS-only vs MSBS-only ablation |
| H3 | Precision collapse not framed | High | 1, 6 | Add PR-AUC, scope as recall-first |
| H4 | Same-family claim confounded | High | 6 | Soften claim, note capacity confound |
| M1 | IW-SMOTE synthesis budget asymmetry | Moderate | 6 | Report mean n_synthetic per protocol |
| M2 | Suspiciousness conflates noise + hardness | Moderate | 2 | Clean-data ablation |
| M3 | IW-SMOTE lamda=30 vs original 100 | Moderate | 2 | Run sensitivity check |
| M4 | No multi-class discussion | Moderate | 6 | Add limitations text |
| M5 | No complexity analysis | Moderate | 6 | Add complexity table |
| M6 | "1,350 pairs" wrong (actual 1,050) | Moderate | 1 | Fix number in paper |
| m1 | Wilcoxon key inconsistency | Minor | 1 | Align analysis scripts |
| m2 | HGB+SMOTE missing-value asymmetry | Minor | 6 | Add footnote |
| m3 | majority_label hardcoded assumption | Minor | 1 | Add defensive check |
| m4 | Oracle relabel absent from benchmark | Minor | 6 | Appendix only — reference as upper bound |

## Key Numbers (post-hardening, 5-dataset baseline with corrected statistics)

- LR + NoiSyn: BA=0.7394 (+3.47pp, Stouffer Z=7.30, p=1.5e-13, 3/5 sig datasets)
- SVM + NoiSyn: BA=0.6776 (+2.16pp, Stouffer Z=5.01, p=2.8e-07, 2/5 sig datasets)
- vs IW-SMOTE: +1.84pp BA (LR, medium, 5 datasets)
- RF/ET: -4.64pp/-3.80pp (ablation: CWMS primary harm, cwms=-7.95pp vs msbs=-4.35pp for RF)
- Failure mode: symmetric=-1.21pp, reverse_asymmetric=-10.21pp (operating condition confirmed)
- Clean data: +2.62pp LR, +2.01pp SVM (boundary synthesis works even without noise)
- IW-SMOTE λ gate: λ=30 ≡ λ=100 (ΔBA=-0.15pp → no change to baseline)
- Phase 5 CONFIRMED (15ds, 3 protos, LR+SVM+HGB): LR vs class_prop combined +3.16pp Z=9.31 p≈0 9/15 sig; vs IW-SMOTE +0.71pp Z=1.17 p=0.12 NOT sig; IW-SMOTE wins SVM/HGB
- Phase 3 partial (15ds, low protocol + CatBoost partial): LR 10/15 sig, Stouffer Z=7.22

## Conda / Run Environment

- Python: `/home/than-minh/miniconda3/envs/dsp/bin/python`
- GPU: CUDA confirmed (XGBoost, LightGBM, CatBoost)
- Existing outputs: `outputs/full-benchmark-solution.csv` (8,250 rows), `outputs/competitor-headtohead.csv` (300 rows)
