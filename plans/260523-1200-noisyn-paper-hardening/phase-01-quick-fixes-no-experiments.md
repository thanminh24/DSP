---
phase: 1
title: "Quick Fixes (no experiments)"
status: pending
priority: P1
effort: "1h"
dependencies: []
---

# Phase 1: Quick Fixes (no experiments)

## Overview

Zero-experiment fixes: wrong number in paper, defensible statistical correction, PR-AUC addition to both evaluation paths, and defensive code checks. Do these first so Phase 2+ builds on a clean base.

## Related Code Files

- Modify: `docs/paper-draft.md`
- Modify: `scripts/analyze_full_benchmark.py`
- Modify: `scripts/analyze_competitor_headtohead.py`
- Modify: `pipeline/evaluation/augment_metrics.py`
- Modify: `pipeline/evaluation/metrics.py`

## Implementation Steps

### 1. Fix M6 — wrong pair count in paper

In `docs/paper-draft.md`, replace "1,350 paired comparisons" → "1,050 paired comparisons".
Derivation: 7 CWMS-compatible models × 150 pairs = 1,050. XGBoost and calibrated_lr are baselines-only (confirmed by `_methods_for()` in `run_full_benchmark_solution.py:29`) and not included in the paired comparison.
Run `grep -r "1,350" docs/` before and after — must return zero after fix.

### 2. Fix C5 — Replace crude NB correction with per-dataset Wilcoxon

The original plan used fixed `n_train=200, n_test=75` — not defensible after dataset expansion where sizes range from ~200 to ~4000.

**Correct approach: per-dataset Wilcoxon + Stouffer combination.**

For each of the N datasets independently, compute a Wilcoxon over (seed × protocol) pairs — 30 pairs per dataset at current settings (10 seeds × 3 protocols). These 30 observations share the same marginal distribution, so within-dataset correlation is at most a calibration concern, not a bias. Between datasets they are independent.

Then combine the N dataset-level p-values using Stouffer's method:

```python
from scipy import stats

def per_dataset_wilcoxon_stouffer(df, method_a, method_b, metric="balanced_accuracy"):
    """Per-dataset Wilcoxon + Stouffer combination. Fully defensible."""
    datasets = sorted(df["dataset"].unique())
    z_scores, pvals, deltas = [], [], []
    for ds in datasets:
        sub = df[df["dataset"] == ds]
        a = sub[sub["method"] == method_a].set_index(["seed", "noise_protocol"])[metric]
        b = sub[sub["method"] == method_b].set_index(["seed", "noise_protocol"])[metric]
        shared = a.index.intersection(b.index)
        diff = (a - b)[shared]
        if len(diff) < 5:
            continue
        try:
            stat, p = stats.wilcoxon(diff, alternative="greater")
            # Convert to one-sided z-score for Stouffer
            z = stats.norm.ppf(1 - p)
        except ValueError:
            p, z = 1.0, 0.0
        pvals.append(p)
        z_scores.append(z)
        deltas.append(diff.mean())
    # Stouffer's combined z
    z_combined = sum(z_scores) / (len(z_scores) ** 0.5)
    p_combined = 1 - stats.norm.cdf(z_combined)
    return {
        "per_dataset_pvals": dict(zip(datasets, pvals)),
        "per_dataset_deltas": dict(zip(datasets, deltas)),
        "stouffer_z": z_combined,
        "stouffer_p": p_combined,
        "n_datasets_significant": sum(p < 0.05 for p in pvals),
        "n_datasets_total": len(pvals),
    }
```

Replace the existing Wilcoxon aggregate in `analyze_full_benchmark.py` with this. Report:
- "X/15 datasets individually significant (Wilcoxon, p<0.05)"
- "Combined Stouffer Z=X.X, p=X.Xe-X across 15 independent datasets"

This eliminates the independence assumption entirely. Do NOT use the old aggregate 150-pair Wilcoxon as the primary statistic.

### 3. Add PR-AUC to both evaluation paths

Both `evaluate()` (`pipeline/evaluation/metrics.py`) and `evaluate_augmented()` (`pipeline/evaluation/augment_metrics.py`) need `average_precision_score`. The models are sklearn pipelines with `predict_proba`.

In both files, after model fitting and before computing metrics, add:

```python
from sklearn.metrics import average_precision_score

# After model.fit(...)
if hasattr(model, "predict_proba"):
    proba = model.predict_proba(X_test)
    classes = list(model.classes_)
    min_col = classes.index(minority_label)
    pr_auc = average_precision_score(
        (y_test == minority_label).astype(int),
        proba[:, min_col],
    )
elif hasattr(model, "decision_function"):
    scores = model.decision_function(X_test)
    pr_auc = average_precision_score(
        (y_test == minority_label).astype(int),
        scores,
    )
else:
    pr_auc = float("nan")
```

Add `"pr_auc": pr_auc` to the returned dict in both evaluators. This column will appear in all future sweep CSVs (Phases 2–5). Existing CSVs do not have it — note this in the analysis scripts.

### 4. Fix m3 — defensive binary label assertion in augment_metrics.py

In `pipeline/evaluation/augment_metrics.py`, before `majority_label = 1 - minority_label`:

```python
assert minority_label in (0, 1), (
    f"Binary {{0,1}} labels assumed; got minority_label={minority_label}. "
    "Extend majority_label computation for non-binary label sets."
)
majority_label = 1 - minority_label
```

### 5. Fix m1 — Wilcoxon key alignment comment

In `scripts/analyze_competitor_headtohead.py`, add comment near Wilcoxon pairing:
```python
# Pairing on (dataset, seed) only — competitor CSV has single protocol.
# Full benchmark uses (dataset, seed, noise_protocol) — see analyze_full_benchmark.py.
```

## Unresolved Questions (resolved here)

- **Oracle rows**: Appendix only. Not added to main Phase 3 benchmark run. Phase 6 references existing oracle results from `outputs/cwms-msbs-deep-sweep.csv` as a note, not a full Table 1 row.
- **Live OpenML vs cached parquet**: Cached parquet only at experiment time. Phase 3 extends `download_datasets.py` first, runs once, then all sweeps read from `data/*.parquet`.

## Success Criteria

- [ ] `grep -r "1,350" docs/` returns zero results
- [ ] `analyze_full_benchmark.py` uses per-dataset Wilcoxon + Stouffer; old aggregate Wilcoxon removed as primary statistic
- [ ] Per-dataset p-value table printed for LR
- [ ] `pr_auc` key present in both `evaluate()` and `evaluate_augmented()` return dicts
- [ ] Binary label assertion added to `augment_metrics.py`
- [ ] Wilcoxon key comment added to `analyze_competitor_headtohead.py`

## Risk Assessment

Low — all additive or text corrections. Switching to per-dataset Wilcoxon + Stouffer may show fewer datasets are individually significant for HGB/LGB (correct outcome). The LR story (d=0.816) will survive this change.
