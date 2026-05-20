---
phase: 4
title: "Noise Rate Ablation"
status: pending
priority: P1
effort: "2h"
dependencies: [3]
---

# Phase 4: Noise Rate Ablation

## Overview

Run the main experiment across three noise level configurations to show CRCC robustness. Produces the noise-rate robustness table for the paper.

Current experiment: 30%/10% noise (minority_to_majority / majority_to_minority).
Ablation levels: low (10%/5%), medium (20%/10%), high (40%/20%).

All 5 literature-verified datasets × 2 models × 5 seeds × 10 methods × 3 noise levels = 1500 rows.

## Architecture

`scripts/run_noise_ablation.py` imports from `pipeline/` and iterates over noise configs. Adds `noise_level` column to each result row.

```python
NOISE_LEVELS = {
    "low":    {"min_to_maj": 0.10, "maj_to_min": 0.05},
    "medium": {"min_to_maj": 0.20, "maj_to_min": 0.10},
    "high":   {"min_to_maj": 0.40, "maj_to_min": 0.20},
}
```

## Related Code Files

- Create: `scripts/run_noise_ablation.py` (≤ 150 lines)
- Read: `pipeline/core/experiment.py`, `pipeline/core/config.py`
- Write: `outputs/noise-ablation-results.csv`, `outputs/noise-ablation-summary.csv`

## Implementation Steps

1. **Create `scripts/run_noise_ablation.py`**:
   ```python
   import sys
   from dataclasses import dataclass
   from pathlib import Path
   import pandas as pd
   
   PROJECT_ROOT = Path(__file__).resolve().parent.parent
   sys.path.insert(0, str(PROJECT_ROOT))
   
   from pipeline.core.config import BaseExperimentConfig
   from pipeline.core.experiment import run_single
   
   NOISE_LEVELS = {
       "low":    {"min_to_maj": 0.10, "maj_to_min": 0.05},
       "medium": {"min_to_maj": 0.20, "maj_to_min": 0.10},
       "high":   {"min_to_maj": 0.40, "maj_to_min": 0.20},
   }
   
   def main():
       all_results = []
       for level_name, noise in NOISE_LEVELS.items():
           cfg = BaseExperimentConfig(
               minority_to_majority_noise=noise["min_to_maj"],
               majority_to_minority_noise=noise["maj_to_min"],
           )
           total = len(cfg.datasets) * 2 * len(cfg.seeds)
           n_done = 0
           for dataset in cfg.datasets:
               for model in ("lr", "hgb"):
                   for seed in cfg.seeds:
                       n_done += 1
                       print(f"[{level_name}] [{n_done}/{total}] {dataset}/{model}/seed={seed}")
                       rows = run_single(dataset, model, seed, cfg)
                       for r in rows:
                           r["noise_level"] = level_name
                       all_results.extend(rows)
       
       df = pd.DataFrame(all_results)
       df.to_csv(PROJECT_ROOT / "outputs" / "noise-ablation-results.csv", index=False)
       # summary
       metric_cols = ["balanced_accuracy","macro_f1","minority_recall",
                      "noise_precision_deleted","clean_minority_deletion_rate"]
       summary = (df.groupby(["noise_level","dataset","model","method"])[metric_cols]
                  .agg(["mean","std"]).round(4))
       summary.columns = [f"{m}_{s}" for m, s in summary.columns]
       summary.reset_index().to_csv(
           PROJECT_ROOT / "outputs" / "noise-ablation-summary.csv", index=False)
       print(f"Done. {len(df)} rows saved.")
   
   if __name__ == "__main__":
       main()
   ```

2. **Run the experiment**:
   ```bash
   python3 scripts/run_noise_ablation.py
   ```
   Expected: ~1500 rows (some NaN rows for Ecoli + high noise edge cases).

3. **Verify key expectation**: CMDR for `crcc_p_l05` stays ≤ 0.05 across all noise levels (cap still binding). If not, investigate.

4. **Check line count**: `wc -l scripts/run_noise_ablation.py` ≤ 150.

## Success Criteria

- [ ] `scripts/run_noise_ablation.py` exists, ≤ 150 lines
- [ ] `outputs/noise-ablation-results.csv` exists with `noise_level` column and ~1500 rows
- [ ] `outputs/noise-ablation-summary.csv` exists
- [ ] CMDR for `crcc_p_l05` ≤ 0.05 across low/medium/high noise (cap binding)
- [ ] Script runs without unhandled exceptions

## Risk Assessment

- Ecoli (10.7% minority) + high noise (40% flip): nearly all minority labels become corrupted. One-class guard handles this with NaN metrics. Count NaN rows and report.
- Runtime: 3 levels × 5 datasets × 2 models × 5 seeds ≈ 150 combos × ~5s = ~12 min. Acceptable.
- Phoneme minority_label=0: ensure `run_single()` reads from `cfg.minority_label_map` (Phase 3 fix).
