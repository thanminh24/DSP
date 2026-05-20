---
phase: 5
title: "Run Full Experiments"
status: completed
priority: P1
effort: "30m"
dependencies: [4]
---

# Phase 5: Run Full Experiments

## Overview

Execute the full experiment pipeline and collect results. This is primarily an execution phase — no new code is written. Verify the output CSV is complete and sane before proceeding to analysis.

## Requirements

- Functional:
  - Run `scripts/run-full-experiment.py` to completion.
  - Output CSV has 300 rows, no NaN in metric columns.
  - Verify key sanity checks on results.
- Non-functional: wall time < 15 minutes on local hardware.

## Expected Runtime

| Dataset | LR/seed | HGB/seed | Notes |
|---------|---------|----------|-------|
| pima (432 train) | ~5s | ~5s | Small, fast |
| credit-g (562 train) | ~8s | ~8s | Medium |
| sick (2079 train) | ~20s | ~15s | Largest |

Per seed: ~60s. Total: 5 seeds × ~60s = ~5 min. With 10 method variants: ×1 (selectors are O(n) post-scoring). Scoring (5-fold CV) dominates. Total estimate: **5–10 minutes**.

## Implementation Steps

1. Ensure all phase-2, 3, 4 scripts are complete and importable:
   ```bash
   /home/than-minh/miniconda3/bin/python3 -c "
   import sys; sys.path.insert(0, 'scripts')
   from data_loader import load_dataset
   from scoring import out_of_fold_loss
   from selectors import select_crcc_p
   from evaluator import evaluate
   print('all imports ok')
   "
   ```
   Note: Python module names use underscores; file names use hyphens. Import via `sys.path` or rename files to use underscores. **Use underscores in filenames** for importability: `data_loader.py`, `scoring.py`, `selectors.py`, `evaluator.py`.

2. Run the experiment:
   ```bash
   /home/than-minh/miniconda3/bin/python3 scripts/run_full_experiment.py
   ```

3. Validate output CSV:
   ```bash
   /home/than-minh/miniconda3/bin/python3 -c "
   import pandas as pd
   df = pd.read_csv('outputs/full-experiment-results.csv')
   print(df.shape)          # expect (300, 10)
   print(df.isnull().sum()) # expect all 0
   print(df['method'].value_counts())
   print(df.groupby(['dataset','model','method'])['balanced_accuracy'].mean().round(3))
   "
   ```

4. Sanity checks to assert:
   - `no_cleaning` has `deleted == 0` for all rows
   - `oracle_deletion` has `noise_precision_deleted ≈ 1.0` for all rows
   - `global_top_loss` has highest `noise_precision_deleted` among non-oracle methods
   - `global_top_loss` has highest `clean_minority_deletion_rate` (the harm it causes)
   - `crcc_p_l10` has lower `clean_minority_deletion_rate` than `global_top_loss` on ≥ 2 of 3 datasets (main hypothesis check — acceptable to fail here and reframe)

5. If any assertion fails:
   - Check for bugs in `select_crcc_p` cap logic
   - Check `noisy_mask` is correctly passed to `evaluate`
   - Re-run with a single seed for debugging

## Success Criteria

- [ ] `outputs/full-experiment-results.csv` exists with 300 rows
- [ ] No NaN in metric columns
- [ ] `no_cleaning` rows: `deleted == 0`, harm metrics == 0.0
- [ ] `oracle_deletion` rows: `noise_precision_deleted` ≥ 0.9 (may be < 1.0 if noise count < budget)
- [ ] Pipeline completes in < 15 minutes

## Risk Assessment

- Module import naming: use underscores in script filenames to allow `import data_loader` etc. Confirm this convention in phase 4.
- Memory: Sick dataset (3772 rows) with 5-fold CV: OOF probability matrix is 3772×2 = trivial.
- Stochastic variation: with 5 seeds, results are averaged — outlier seeds don't invalidate the method.
- If `crcc_p_l10` doesn't outperform `global_top_loss` on any dataset: this is a legitimate negative result per proposal scope. Document and reframe in report.
