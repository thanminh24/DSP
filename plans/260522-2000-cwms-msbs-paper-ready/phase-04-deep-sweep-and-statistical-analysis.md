---
phase: 4
title: "Deep Sweep and Statistical Analysis"
status: in-progress
priority: P1
effort: "~3h compute + 30m analysis"
dependencies: [3]
---

# Phase 4: Deep Sweep and Statistical Analysis

## Overview

Run the definitive benchmark sweep with v2 CWMS weights, all 6 model families
(catboost now confirmed working in conda `dsp`), full metric suite, 3 noise protocols,
10 seeds. Then produce the complete statistical analysis that feeds directly into
the paper's results section.

## Sweep Configuration

| Dimension | Values |
|-----------|--------|
| **Models** | `lr`, `hgb`, `lightgbm`, `catboost` (main); `calibrated_lr` (no_cleaning + cp only); `xgboost` (skipped for cwms/cwms_msbs, included for no_cleaning + cp only) |
| **Datasets** | `pima`, `credit-g`, `yeast`, `phoneme`, `ecoli` |
| **Seeds** | `[13, 17, 23, 29, 31, 37, 41, 43, 47, 53]` (10 seeds) |
| **Noise protocols** | `hidden_minority_medium`, `hidden_minority_low`, `hidden_minority_high` |
| **Budget** | 0.10 |
| **Target ratio** | 0.15 |
| **Methods** | `no_cleaning`, `class_proportional`, `msbs`, `cwms`, `cwms_msbs`, `cwms_msbs_shuffled` (added in Phase 3) |

**Row count:** ~5,400 rows (4 full models × 5 datasets × 10 seeds × 3 protocols × 6 methods + partial rows for calibrated_lr/xgboost)

**Output:** `outputs/cwms-msbs-deep-sweep.csv`

**Command:**
```bash
/home/than-minh/miniconda3/envs/dsp/bin/python scripts/run_cwms_msbs_deep_sweep.py --medium-only
# Then extend to all protocols:
/home/than-minh/miniconda3/envs/dsp/bin/python scripts/run_cwms_msbs_deep_sweep.py
```

## Required Metrics

All returned by `evaluate_augmented()` after Phase 1/2 fix:
- `balanced_accuracy` — primary metric
- `accuracy` — secondary
- `macro_f1` — secondary
- `weighted_f1` — secondary
- `minority_recall` — key claim (synthesis should recover recall)
- `minority_precision` — key claim (synthesis shouldn't hurt precision)
- `majority_recall` — robustness check
- `n_synthetic` — mechanism check (must be > 0 in cwms_msbs rows)

## Analysis Script: analyze_cwms_msbs_deep_results.py

The script must produce all tables needed for the paper. Sections:

### Table 1 — Primary summary (hidden_minority_medium, 4 main models + cwms_msbs)

```
Method           | BA   | Recall | Precision | Macro F1 | Δ BA vs cp | p
cwms_msbs        | ...  | ...    | ...       | ...      | ...        | ...
class_proportional | ... | ...    | ...       | ...      | —          | —
cwms             | ...  | ...    | ...       | ...      | ...        | ...
msbs             | ...  | ...    | ...       | ...      | ...        | ...
no_cleaning      | ...  | ...    | ...       | ...      | ...        | ...
```

### Table 2 — Per-model breakdown (paper main table)

For hidden_minority_medium, cwms_msbs vs class_proportional:

```
Model       | cwms_msbs BA | cp BA | Δ    | Win% | p       | Verdict
lr          | ...          | ...   | +Xpp | X%   | X.Xe-XX | ✓
hgb         | ...          | ...   | ...  | ...  | ...     | ...
lightgbm    | ...          | ...   | ...  | ...  | ...     | ...
catboost    | ...          | ...   | ...  | ...  | ...     | ...
```

### Table 3 — Multi-metric comparison (cwms_msbs vs class_proportional, LR focus)

Show all metrics side-by-side for LR to make the paper claim multi-dimensional.

### Table 4 — Ablation: cwms_msbs vs cwms_msbs_shuffled

Proves OOF confidence scores are load-bearing (not just any sample weighting).

### Table 5 — Cross-protocol robustness

cwms_msbs vs class_proportional across all 3 noise protocols for each model.

### Statistical Tests

Paired Wilcoxon signed-rank for all comparisons. Bonferroni correction for multiple
comparisons (10 model × method pairs). Report: statistic, p-value, effect size (r).

## Success Criteria

- [ ] `outputs/cwms-msbs-deep-sweep.csv` has ≥4,000 clean rows
- [ ] Zero error rows for lr, hgb, lightgbm, catboost on medium protocol
- [ ] `n_synthetic > 0` in all cwms_msbs rows
- [ ] `cwms_msbs_shuffled` BA < `cwms_msbs` BA for LR (confirms load-bearing scores)
- [ ] LR cwms_msbs delta vs class_proportional: p < 0.001
- [ ] All 5 tables produced by analysis script without errors
- [ ] Numbers stable vs. mini-sweep (LR delta ~+4-5pp, not a dramatic swing)

## Risk Assessment

- **catboost compute time**: 300 iterations × 5 datasets × 10 seeds × 3 protocols = slow.
  Run `--medium-only` first (primary result), then extend.
- **cwms_msbs_shuffled may not be worse**: if shuffled weights happen to perform well on
  some datasets, the ablation is inconclusive. Report honestly — this would mean the score
  discriminativeness is lower than expected.
- **Resume safety**: the deep sweep is resume-safe. Any crash can be restarted cleanly.
