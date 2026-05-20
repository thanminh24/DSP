---
phase: 6
title: "Results Analysis and Visualization"
status: completed
priority: P1
effort: "2h"
dependencies: [5]
---

# Phase 6: Results Analysis and Visualization

## Overview

Write `scripts/plot_results.py` to produce the summary table and 3 publication-quality plots from `outputs/full-experiment-results.csv`. Also produce `outputs/summary-table.csv`.

## Requirements

- Functional:
  - `outputs/summary-table.csv`: mean ± std for all metrics, grouped by `(dataset, model, method)`
  - Plot 1: Clean-minority deletion rate by method and dataset (bar chart, main harm metric)
  - Plot 2: Minority recall by method and dataset (bar chart)
  - Plot 3: Balanced accuracy by method and dataset (bar chart or table)
  - Optional Plot 4: Lambda ablation — clean-minority deletion rate vs λ for CRCC-P per dataset
- Non-functional: plots saved as PNG 300 DPI to `outputs/plots/`; seaborn styling; no interactive display.

## Architecture

```
plot_results.py
  └─ load_dataset: pd.read_csv("outputs/full-experiment-results.csv")
  └─ compute_summary(): groupby + agg(mean, std)
  └─ plot_bar(metric, title, filename): grouped bar chart per method × dataset
  └─ plot_lambda_ablation(): filter crcc_p_l* rows, line plot per dataset
  └─ main(): save summary CSV + generate all plots
```

**Method display order** (for plots, ordered from baseline to proposed):
1. no_cleaning
2. random_deletion
3. global_top_loss
4. class_proportional
5. crcc_p_l0
6. crcc_p_l025
7. crcc_p_l05
8. crcc_p_l10
9. crcc_m
10. oracle_deletion

For the main paper figure: use `crcc_p_l05` (λ=0.5) as the primary CRCC representative. Show lambda ablation in a separate subplot or table.

**Summary table format:**

```
dataset | model | method | bal_acc_mean | bal_acc_std | macro_f1_mean | ... | cmdr_mean | cmdr_std
```
(cmdr = clean_minority_deletion_rate)

**Plot 1 — Clean-Minority Deletion Rate (main harm metric):**
- X-axis: method (6 main methods: no_cleaning, random, global, class_prop, crcc_p_l05, oracle)
- Y-axis: mean clean_minority_deletion_rate
- Error bars: ± std across seeds
- Facet: one panel per dataset (3 panels side by side)
- Color: highlight crcc_p_l05 vs global_top_loss

**Plot 2 — Minority Recall:**
- Same layout as Plot 1
- Y-axis: mean minority_recall

**Plot 3 — Balanced Accuracy:**
- Same layout
- Y-axis: mean balanced_accuracy

**Plot 4 (optional) — Lambda Ablation:**
- X-axis: λ ∈ {0, 0.25, 0.5, 1.0}
- Y-axis: mean clean_minority_deletion_rate
- Line per dataset
- Shows sensitivity to λ

## Related Code Files

- Create: `scripts/plot_results.py`
- Read: `outputs/full-experiment-results.csv`
- Output:
  - `outputs/summary-table.csv`
  - `outputs/plots/fig1-clean-minority-deletion-rate.png`
  - `outputs/plots/fig2-minority-recall.png`
  - `outputs/plots/fig3-balanced-accuracy.png`
  - `outputs/plots/fig4-lambda-ablation.png` (optional)

## Implementation Steps

1. Write `scripts/plot_results.py`:

   ```python
   import pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt
   from pathlib import Path

   MAIN_METHODS = ["no_cleaning", "random_deletion", "global_top_loss",
                   "class_proportional", "crcc_p_l05", "oracle_deletion"]
   METHOD_LABELS = {
       "no_cleaning": "No Clean", "random_deletion": "Random",
       "global_top_loss": "Global Top-Loss", "class_proportional": "Class-Prop",
       "crcc_p_l05": "CRCC-P (λ=0.5)", "oracle_deletion": "Oracle",
   }
   ```

2. `compute_summary(df)`:
   - `df.groupby(["dataset", "model", "method"])[metric_cols].agg(["mean", "std"])`
   - Flatten MultiIndex columns: `bal_acc_mean`, `bal_acc_std`, etc.
   - Save to `outputs/summary-table.csv`

3. `plot_harm_metric(df, metric, ylabel, filename)`:
   - Filter to MAIN_METHODS
   - Compute per-group mean/std
   - `sns.catplot(data=..., kind="bar", x="method", y=mean, col="dataset", ...)`
   - Add error bars manually or use seaborn's ci
   - Rotate x labels 45°
   - Save 300 DPI

4. Call `plot_harm_metric` for:
   - `clean_minority_deletion_rate` → `fig1-clean-minority-deletion-rate.png`
   - `minority_recall` → `fig2-minority-recall.png`
   - `balanced_accuracy` → `fig3-balanced-accuracy.png`

5. `plot_lambda_ablation(df)`:
   - Filter: method in `["crcc_p_l0", "crcc_p_l025", "crcc_p_l05", "crcc_p_l10"]`
   - Map to numeric λ: `{"crcc_p_l0": 0.0, "crcc_p_l025": 0.25, ...}`
   - Line plot per dataset: x=λ, y=mean clean_minority_deletion_rate
   - Save → `fig4-lambda-ablation.png`

6. Print key findings to stdout:
   - For each dataset: `global_top_loss` CMDR vs `crcc_p_l05` CMDR (% reduction)
   - Main success condition pass/fail per dataset

## Success Criteria

- [ ] `outputs/summary-table.csv` written with all 300 rows aggregated
- [ ] All 3 required PNGs saved to `outputs/plots/`
- [ ] Plots legible (font size ≥ 10, legend present)
- [ ] Lambda ablation plot generated (optional but expected)
- [ ] Key findings printed to stdout showing CMDR reduction for each dataset
- [ ] Module ≤ 200 lines

## Risk Assessment

- Seaborn version: seaborn 0.13 has slightly different catplot API than 0.12. Test with actual data before finalizing.
- If oracle_deletion has `noise_precision_deleted < 1.0` (budget > true noise count): document in plot title/caption.
- If any dataset shows CMDR=0 for all methods (noise count too low): note this means no harm was observed under any strategy — still a valid finding.
