---
phase: 5
title: "Budget Ablation"
status: pending
priority: P1
effort: "2h"
dependencies: [3]
---

# Phase 5: Budget Ablation

## Overview

Run the main experiment configuration (85/15 imbalance, 30%/10% noise) across 5 budget levels to produce the CMDR-vs-budget curve — key figure showing how the cleaning budget size interacts with class protection. All 5 datasets × 2 models × 5 seeds × 10 methods × 5 budget levels = 2500 rows.

Budget levels: {5%, 10%, 15%, 20%, 30%}.

Expected behavior: `global_top_loss` CMDR grows roughly linearly with budget; `crcc_p` CMDR stays near zero (cap always binding); the gap between them is the value of CRCC.

## Architecture

`scripts/run_budget_ablation.py` imports from `pipeline/` and iterates over budget levels. Adds `budget` float column to each result row.

## Related Code Files

- Create: `scripts/run_budget_ablation.py` (≤ 150 lines)
- Write: `outputs/budget-ablation-results.csv`, `outputs/budget-ablation-summary.csv`

## Implementation Steps

1. **Create `scripts/run_budget_ablation.py`**:
   ```python
   import sys
   from pathlib import Path
   import pandas as pd
   
   PROJECT_ROOT = Path(__file__).resolve().parent.parent
   sys.path.insert(0, str(PROJECT_ROOT))
   
   from pipeline.core.config import BaseExperimentConfig
   from pipeline.core.experiment import run_single
   
   BUDGET_LEVELS = [0.05, 0.10, 0.15, 0.20, 0.30]
   
   def main():
       all_results = []
       for budget in BUDGET_LEVELS:
           cfg = BaseExperimentConfig(cleaning_budget=budget)
           total = len(cfg.datasets) * 2 * len(cfg.seeds)
           n_done = 0
           for dataset in cfg.datasets:
               for model in ("lr", "hgb"):
                   for seed in cfg.seeds:
                       n_done += 1
                       print(f"[budget={budget}] [{n_done}/{total}] {dataset}/{model}/seed={seed}")
                       rows = run_single(dataset, model, seed, cfg)
                       for r in rows:
                           r["budget"] = budget
                       all_results.extend(rows)
       
       df = pd.DataFrame(all_results)
       df.to_csv(PROJECT_ROOT / "outputs" / "budget-ablation-results.csv", index=False)
       metric_cols = ["balanced_accuracy","macro_f1","minority_recall",
                      "noise_precision_deleted","clean_minority_deletion_rate"]
       summary = (df.groupby(["budget","dataset","model","method"])[metric_cols]
                  .agg(["mean","std"]).round(4))
       summary.columns = [f"{m}_{s}" for m, s in summary.columns]
       summary.reset_index().to_csv(
           PROJECT_ROOT / "outputs" / "budget-ablation-summary.csv", index=False)
       print(f"Done. {len(df)} rows saved.")
   
   if __name__ == "__main__":
       main()
   ```

2. **Run the experiment**:
   ```bash
   python3 scripts/run_budget_ablation.py
   ```
   Expected: ~2500 rows.

3. **Sanity check** after run:
   ```python
   import pandas as pd
   df = pd.read_csv("outputs/budget-ablation-results.csv")
   check = (df[df["method"]=="global_top_loss"]
              .groupby("budget")["clean_minority_deletion_rate"].mean())
   print(check)  # should increase with budget
   ```

4. **Check line count**: `wc -l scripts/run_budget_ablation.py` ≤ 150.

## Success Criteria

- [ ] `scripts/run_budget_ablation.py` exists, ≤ 150 lines
- [ ] `outputs/budget-ablation-results.csv` exists with `budget` column and ~2500 rows
- [ ] `outputs/budget-ablation-summary.csv` exists
- [ ] `global_top_loss` mean CMDR increases monotonically from budget=0.05 → 0.30
- [ ] `crcc_p_l05` mean CMDR stays ≤ 0.05 across all budget levels

## Risk Assessment

- Budget=30% on Ecoli (336 rows): deletes ~101 samples from training. With only ~36 minority samples (10.7%), this wipes out all minority with global deletion. One-class guard handles it. Document NaN rows.
- Runtime: 5 levels × 5 datasets × 2 models × 5 seeds = 250 combos ≈ 20 min.
- Monotonicity check may fail for specific dataset/model combos due to seed variance. Report per-combo monotonicity, not aggregate only.
