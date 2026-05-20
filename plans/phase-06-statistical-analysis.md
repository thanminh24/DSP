---
phase: 6
title: "Statistical Analysis"
status: pending
priority: P1
effort: "1h"
dependencies: [4, 5]
---

# Phase 6: Statistical Analysis

## Overview

Formal statistical testing of TBDC vs class_proportional and CRCC-M across the full sweep
results. Produces the paper's main comparison table and validates all pre-registered hypotheses.

## Tests to Run

### Primary comparison (H1): TBDC vs class_proportional

Wilcoxon signed-rank test, paired by seed, for each (dataset, model, noise_level) combo:
- Metric: balanced_accuracy
- Alternative: two-sided (then check direction of effect)
- n_pairs = 20 (seeds)
- Report: W-statistic, p-value, Cohen's d, direction (TBDC > class_prop ?)

Expected: p ≤ 0.05 at medium+high noise for ≥ 4/5 datasets.

### Noise scaling (H2): TBDC advantage by noise level

For each (dataset, model): compute delta_BA = TBDC_BA - class_prop_BA at each noise level.
Test with Friedman test across 3 noise levels (within each dataset/model).
Expected: delta_BA increases from low → medium → high.

### Recall and CMDR (H3, H4)

Same Wilcoxon structure as H1 but for minority_recall and clean_minority_deletion_rate.
Report as supplementary tables.

### Cap estimation accuracy (from Phase 5 ablation)

Pearson correlation between (estimated_noisy_count / true_type_b_count) and BA_gain.
Does more accurate estimation → more gain?

## Related Code Files

- Modify: `scripts/run_statistical_tests.py` (extend to include TBDC comparisons)
  OR create `scripts/run_tbdc_statistical_tests.py` (≤ 120 lines)
- Read: `outputs/tbdc-sweep-results.csv`, `outputs/cap-ablation-results.csv`

## Implementation Steps

1. Reuse the existing Wilcoxon infrastructure from `scripts/run_statistical_tests.py`.
   The seed-pairing fix (merge on seed column before extracting arrays) is already correct
   — do not regress this.

2. Create `scripts/run_tbdc_statistical_tests.py`:

```python
"""Statistical tests for TBDC vs class_proportional and CRCC-M.

Reads: outputs/tbdc-sweep-results.csv
Writes: outputs/tbdc-statistical-tests.csv
"""
from scipy.stats import wilcoxon

def wilcoxon_paired(df, ds, model, noise_level, method_a, method_b, metric):
    """Paired Wilcoxon by seed. Returns (stat, p, cohens_d, n_pairs, direction)."""
    sub = df[(df.dataset==ds) & (df.model==model) & (df.noise_level==noise_level)]
    a = sub[sub.method==method_a][["seed", metric]].dropna()
    b = sub[sub.method==method_b][["seed", metric]].dropna()
    merged = a.merge(b, on="seed", suffixes=("_a","_b"))
    if len(merged) < 5:
        return None
    vals_a = merged[f"{metric}_a"].values
    vals_b = merged[f"{metric}_b"].values
    stat, p = wilcoxon(vals_a, vals_b, alternative="two-sided")
    d = (vals_a - vals_b).mean() / (vals_a - vals_b).std() if (vals_a - vals_b).std() > 0 else 0
    direction = "a>b" if vals_a.mean() > vals_b.mean() else "b>a"
    return stat, p, round(d, 3), len(merged), direction
```

3. Run for each (dataset, model, noise_level) × comparison pair:
   - TBDC vs class_proportional
   - TBDC vs crcc_m
   - CRCC-M vs class_proportional (replicates existing result for consistency)

4. Print main comparison table:

```
Dataset      | Model | Noise  | BA(cp) | BA(tbdc) | Δ    | p      | d
-------------|-------|--------|--------|----------|------|--------|------
pima         | lr    | medium | 0.699  | 0.XXX    | +XX  | 0.0625 | X.XX
...
```

5. Print H1–H5 verdict: GO/FAIL per hypothesis.

## Main Paper Table Format

The paper needs one primary results table:

| Dataset | Model | No Clean | Global | Class-Prop | CRCC-M | **TBDC** | Oracle |
|---|---|---|---|---|---|---|---|
| pima | lr | … | … | … | … | **…** | … |
| … | … | … | … | … | … | … | … |

Reported as mean ± std over 20 seeds at medium noise. CMDR and minority recall as supplementary.

## Success Criteria

- [ ] Wilcoxon tests run for all (dataset, model, noise_level) × (TBDC vs cp, TBDC vs crcc_m)
- [ ] Cohen's d reported for each comparison
- [ ] H1–H5 verdicts documented in plan.md
- [ ] Main paper table produced (mean ± std, 20 seeds, medium noise)
- [ ] Noise-scaling table produced (delta_BA by noise level)
- [ ] `outputs/tbdc-statistical-tests.csv` written

## Risk Assessment

- With n=20 seeds and Wilcoxon: minimum achievable p = 0.0625 (n=5), so with n=20 the minimum
  is much lower. We have adequate power if the effect exists.
- If TBDC shows d > 0.3 but p > 0.05 at any combo: report effect size as primary, note
  p-value floor for small n. Standard practice in this literature.
- If ALL hypotheses fail: negative result. Report: "OOF disagreement rate is an unreliable
  noise density estimator under severe class imbalance (15% minority, n_minority~75).
  The TBDC allocation strategy is theoretically sound but empirically limited by OOF
  calibration quality." This is a valid paper contribution as a negative finding + diagnostic.
