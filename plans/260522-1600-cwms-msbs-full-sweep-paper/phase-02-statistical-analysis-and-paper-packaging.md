---
phase: 2
title: "Statistical Analysis and Paper Packaging"
status: complete
priority: P1
effort: "~2h"
dependencies: [1]
---

# Phase 2: Statistical Analysis and Paper Packaging

## Overview

Load `outputs/cwms-msbs-full-sweep.csv`, run paired statistical tests, produce the summary
table and per-model breakdown that form the paper's main results section. Write the README
and paper abstract with updated claims — no longer using `balanced_oof_relabel` as our method.

## Related Code Files

- **Create:** `scripts/analyze_cwms_msbs_results.py`
- **Modify:** `README.md` — replace relabeling viability verdict with CWMS+MSBS results
- **Read:** `outputs/cwms-msbs-full-sweep.csv` (from Phase 1)

## Analysis Steps

### Step 1 — Load and validate

```python
import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('outputs/cwms-msbs-full-sweep.csv')
if 'error' in df.columns:
    df = df[df['error'].isna()].copy()
df = df.reset_index(drop=True)

# Primary slice: hidden_minority_medium only
df_med = df[df['noise_protocol'] == 'hidden_minority_medium'].copy()

print(df_med.groupby(['method','model'])['balanced_accuracy'].count().unstack())
# Verify: 50 rows per method/model cell (5 datasets × 10 seeds)
```

### Step 2 — Primary summary table

```python
metrics = ['balanced_accuracy', 'minority_recall']
summary = df_med.groupby('method')[metrics].agg(['mean','std']).round(4)
summary.columns = ['BA_mean','BA_std','recall_mean','recall_std']
print(summary.sort_values('BA_mean', ascending=False))
```

Expected order (based on prelim): class_proportional > cwms_msbs > cwms > msbs > no_cleaning.
If cwms_msbs beats class_proportional overall: headline claim changes to "beats the deletion baseline."
If cwms_msbs is below: "competitive within X% of class_proportional, without any label modification."

### Step 3 — Paired Wilcoxon signed-rank tests

```python
from scipy.stats import wilcoxon

idx_cols = ['dataset', 'model', 'seed']
pivot = df_med.pivot_table(index=idx_cols, columns='method', values='balanced_accuracy')

target = 'cwms_msbs'
comparisons = ['no_cleaning', 'class_proportional', 'msbs', 'cwms']

print(f"{'comparison':30s} {'wins':>5} {'total':>5} {'win%':>6} {'mean_delta':>12} {'p-value':>12}")
for comp in comparisons:
    sub = pivot[[target, comp]].dropna()
    wins = (sub[target] > sub[comp]).sum()
    n = len(sub)
    delta = (sub[target] - sub[comp]).mean()
    try:
        stat, p = wilcoxon(sub[target], sub[comp])
    except Exception:
        p = float('nan')
    print(f"{target} vs {comp:20s} {wins:>5} {n:>5} {100*wins/n:>5.1f}% {delta:>+12.4f} {p:>12.2e}")
```

### Step 4 — Per-model breakdown (key table for paper)

```python
model_tab = df_med[df_med['method'].isin(['cwms_msbs','class_proportional','no_cleaning'])].pivot_table(
    index='model', columns='method', values='balanced_accuracy', aggfunc='mean'
).round(4)
model_tab['delta_vs_cp'] = (model_tab['cwms_msbs'] - model_tab['class_proportional']).round(4)
model_tab['win_rate_vs_cp'] = df_med.pivot_table(
    index=idx_cols, columns='method', values='balanced_accuracy'
).apply(lambda r: r['cwms_msbs'] > r['class_proportional'], axis=1).groupby(
    df_med.pivot_table(index=idx_cols, columns='method', values='balanced_accuracy').index.get_level_values('model')
).mean().round(3)
print(model_tab.sort_values('delta_vs_cp', ascending=False))
```

Interpret using the in-scope / out-of-scope split:
- **In-scope** (lr, calibrated_lr, hgb, xgboost, catboost, lightgbm): present as main results
- **Tree models** (rf, et): excluded from sweep — not in this CSV

### Step 5 — Robustness across noise protocols

```python
print('=== Cross-protocol robustness ===')
proto_tab = df[df['method'].isin(['cwms_msbs','class_proportional','no_cleaning'])].pivot_table(
    index='noise_protocol', columns='method', values='balanced_accuracy', aggfunc='mean'
).round(4)
proto_tab['delta_vs_cp'] = proto_tab['cwms_msbs'] - proto_tab['class_proportional']
print(proto_tab)
```

**Expected pattern:**
- `hidden_minority_medium` (primary): cwms_msbs should be best or competitive
- `hidden_minority_low` (less noise, less opportunity): gains shrink
- `hidden_minority_high` (more noise): gains may increase (more signal in bal_scores)

**Failure mode to watch:** if cwms_msbs underperforms no_cleaning on any protocol, check
whether bal_scores are degenerate (high noise rate makes OOF unreliable).

### Step 6 — MSBS volume check

```python
msbs_rows = df_med[df_med['method'].isin(['msbs','cwms_msbs'])]
print(msbs_rows.groupby('method')['n_synthetic'].describe().round(1))
print(f"Zero-synthetic cases: {(msbs_rows['n_synthetic'] == 0).sum()}")
```

If many zero-synthetic cases: budget_count < n_minority happened often. Note this as a boundary
condition in paper.

### Step 7 — Write verdict

Answer these questions for the paper:

1. **Primary claim**: Does cwms_msbs beat `class_proportional` on the target model families?
   - YES (overall or for linear): "CWMS+MSBS outperforms confidence-guided deletion..."
   - CLOSE (within 1-2pp): "CWMS+MSBS matches class_proportional with zero label modification..."
   - NO: "CWMS+MSBS improves over no_cleaning (+Xpp) and closes the gap to class_proportional, but does not yet surpass it. We identify [model family] as the key beneficiary."

2. **Secondary claim**: Is cwms_msbs strictly better than either standalone method?
   Target: yes (prelim showed 88% and 75% win rates).

3. **Recall claim**: Does cwms_msbs recover minority recall vs class_proportional?
   Deletion removes samples; synthesis adds them. Recall should favor cwms_msbs.

4. **Operating condition**: method targets `hidden_minority_*` protocols only. Document explicitly.

## Paper Sections to Update

### README.md rewrite (after analysis)

Replace the current "Verdict: VIABLE — balanced_oof_relabel" section with:

```markdown
## Method: CWMS+MSBS Boundary Correction

Label-corruption-free alternative to relabeling under hidden minority noise.

| Method | BA (hidden_minority_medium) | vs class_proportional |
|--------|-----------------------------|-----------------------|
| cwms_msbs | X.XXXX | +X.XXpp (p=Y) |
| cwms | X.XXXX | −X.XXpp |
| msbs | X.XXXX | −X.XXpp |
| class_proportional | X.XXXX | — |
| no_cleaning | X.XXXX | −X.XXpp |

### Per-model results (linear + boosting families)
[table from Step 4]

### Why not tree models (RF/ET)?
Tree-based models (RandomForest, ExtraTrees) use bootstrap aggregation internally.
Per-sample weights (used by CWMS) are diluted during bootstrap sampling, making
the confidence-targeted suppression ineffective. We exclude these from the main results
and recommend class_proportional as the baseline for tree models.
```

### Paper abstract (template, fill numbers after analysis)

> Hidden minority label noise — where true minority samples are mislabeled as majority —
> degrades decision boundary quality in imbalanced tabular classification. Existing methods
> either delete suspicious samples (losing scarce minority evidence) or relabel them (introducing
> label corruption). We propose CWMS+MSBS, a dual boundary correction approach that synthesizes
> new minority samples toward the contaminated boundary (MSBS) while simultaneously suppressing
> suspicious majority samples via confidence-derived weights (CWMS). Both components reuse
> OOF confidence scores already computed for detection, requiring no additional model training.
> On 5 tabular benchmarks with hidden minority noise, CWMS+MSBS achieves [X]% balanced accuracy
> across linear and gradient boosting classifiers, [beating / matching] confidence-proportional
> deletion while preserving all original labels. Minority recall improves by [X]pp versus deletion.

## Success Criteria

- [ ] Analysis script produces clean summary table without errors
- [ ] Paired Wilcoxon p-values computed for all comparisons
- [ ] Per-model table shows linear family (lr, calibrated_lr) benefiting most
- [ ] Robustness table covers all 3 noise protocols
- [ ] MSBS n_synthetic > 0 in >95% of rows
- [ ] Verdict written: proceed / partial / stop, with specific paper claim
- [ ] README.md updated with new method name, numbers, and scope

## Scope Decision (confirmed at plan creation)

**Tree models excluded** from main results. Paper explicitly states:
> "CWMS relies on sample_weight to suppress noisy majority samples. Tree-based ensemble methods
> (RF, ET) implement bootstrap aggregation which does not linearly scale gradient contributions
> by sample_weight in the same way as gradient-based methods. We therefore restrict evaluation
> to linear models and gradient boosting families where sample weighting is well-defined."

This is not a weakness — it is scope honesty. The method works for 4-6 model families
that together cover the most common high-performance classifiers for tabular data.
