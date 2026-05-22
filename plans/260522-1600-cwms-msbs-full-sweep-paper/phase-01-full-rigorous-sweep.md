---
phase: 1
title: "Full Rigorous Sweep"
status: complete
priority: P1
effort: "~4h compute"
dependencies: []
---

# Phase 1: Full Rigorous Sweep

## Overview

Run the full benchmarking grid for `cwms_msbs`, `cwms`, `msbs`, `class_proportional`, and
`no_cleaning` across all in-scope models, all 5 datasets, 10 seeds (QUICK_SEEDS), and
3 noise protocols. Output goes to `outputs/cwms-msbs-full-sweep.csv`. Resume-safe.

## Experiment Scope

| Dimension | Values |
|-----------|--------|
| Models | `lr`, `calibrated_lr`, `hgb`, `xgboost`, `catboost`, `lightgbm` |
| Datasets | `pima`, `credit-g`, `yeast`, `phoneme`, `ecoli` |
| Seeds | `[13, 17, 23, 29, 31, 37, 41, 43, 47, 53]` (QUICK_SEEDS) |
| Noise protocols | `hidden_minority_medium`, `hidden_minority_high`, `hidden_minority_low` |
| Budget | 0.10 |
| Target ratio | 0.15 |
| Methods | `no_cleaning`, `class_proportional`, `msbs`, `cwms`, `cwms_msbs` |

Total rows: 6 models × 5 datasets × 10 seeds × 3 protocols × 5 methods = **4,500 rows**

Note: only `hidden_minority_medium` is the primary result. Low/high are robustness checks.
If compute is tight, run medium first, then extend to low/high.

## Related Code Files

- **Modify:** `scripts/run_relabeling_viability_sweep.py` — `cwms_msbs` dispatch already implemented
- **Create:** `scripts/run_cwms_msbs_full_sweep.py` — full sweep runner
- **Read:** `outputs/cwms-msbs-combined-prelim.csv` — prelim results (3 seeds, 5 models) for sanity check

## Architecture

Data flow for `cwms_msbs` in `_run_method()` (already implemented):
```
bal_scores = balanced_oof_majority_scores(X_tr, y_noisy, bal_factory, ...)  # OOF
X_aug, y_aug, n_synth = minority_side_boundary_synthesis(                   # MSBS
    X_tr, y_noisy, bal_scores, budget_count, min_label, maj_label, seed=seed
)
sw_orig = confidence_weighted_sample_weights(y_noisy, bal_scores, maj_label) # CWMS weights
sw_combined = concat([sw_orig, ones(n_synth)])                               # synthetics = 1.0
evaluate_augmented(X_aug, y_aug, X_te, y_te, factory, min_label,
                   n_synthetic=n_synth, sample_weight=sw_combined)
```

## Implementation Steps

### Step 1 — Write `scripts/run_cwms_msbs_full_sweep.py`

Template from `scripts/run_cwms_msbs_combined_prelim.py` with these changes:
- `FULL_METHODS = ["no_cleaning", "class_proportional", "msbs", "cwms", "cwms_msbs"]`
- `FULL_SEEDS = QUICK_SEEDS`  (all 10)
- `FULL_PROTOCOLS = ["hidden_minority_medium", "hidden_minority_low", "hidden_minority_high"]`
- `OUTPUT_CSV = "outputs/cwms-msbs-full-sweep.csv"`
- Inner loop: iterate protocols, not just one fixed protocol
- Use `list_publication_models()` but skip RF/ET in the model list:
  ```python
  EXCLUDED_MODELS = {"random_forest", "extra_trees"}
  models = [m for m in list_publication_models() if m not in EXCLUDED_MODELS]
  ```

```python
"""Full rigorous sweep for CWMS+MSBS boundary correction method.

Scope: lr, calibrated_lr, hgb, xgboost, catboost, lightgbm
       × 5 datasets × 10 seeds × 3 hidden_minority protocols × 5 methods
Output: outputs/cwms-msbs-full-sweep.csv  (resume-safe)
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_relabeling_viability_sweep import (
    DATASETS, NOISE_PROTOCOLS, QUICK_SEEDS, run_single_viability,
)
from pipeline.models.factories import list_publication_models

FULL_METHODS = ["no_cleaning", "class_proportional", "msbs", "cwms", "cwms_msbs"]
FULL_SEEDS = QUICK_SEEDS
FULL_PROTOCOLS = ["hidden_minority_medium", "hidden_minority_low", "hidden_minority_high"]
EXCLUDED_MODELS = {"random_forest", "extra_trees"}
POC_BUDGET = 0.10
POC_RATIO = 0.15
OUTPUT_CSV = PROJECT_ROOT / "outputs" / "cwms-msbs-full-sweep.csv"


def _load_completed(path: Path) -> set:
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path)
    except Exception:
        return set()
    return {
        (r["dataset"], r["model"], int(r["seed"]), r["noise_protocol"],
         float(r["budget"]), float(r["target_ratio"]), r["method"])
        for _, r in df.iterrows()
    }


def main():
    args = sys.argv[1:]
    use_gpu = "--gpu" in args
    medium_only = "--medium-only" in args
    protocols = ["hidden_minority_medium"] if medium_only else FULL_PROTOCOLS

    models = [m for m in list_publication_models() if m not in EXCLUDED_MODELS]
    completed = _load_completed(OUTPUT_CSV)
    total_written = 0

    for proto_name in protocols:
        mn, mj = NOISE_PROTOCOLS[proto_name]
        for model_name in models:
            for dataset in DATASETS:
                for seed in FULL_SEEDS:
                    needed = [
                        m for m in FULL_METHODS
                        if (dataset, model_name, seed, proto_name,
                            float(POC_BUDGET), float(POC_RATIO), m) not in completed
                    ]
                    if not needed:
                        continue
                    try:
                        batch = run_single_viability(
                            dataset, model_name, seed, proto_name, mn, mj,
                            POC_BUDGET, POC_RATIO,
                            use_gpu=use_gpu, methods=needed,
                        )
                    except Exception as exc:
                        batch = [
                            {
                                "dataset": dataset, "model": model_name, "seed": seed,
                                "noise_protocol": proto_name, "method": m,
                                "error": str(exc), "mn_to_maj": mn, "maj_to_min": mj,
                                "budget": POC_BUDGET, "target_ratio": POC_RATIO,
                            }
                            for m in needed
                        ]
                        print(f"FAIL {dataset}/{model_name}/{seed}/{proto_name}: {exc}", flush=True)

                    df_batch = pd.DataFrame(batch)
                    write_header = not OUTPUT_CSV.exists() or OUTPUT_CSV.stat().st_size == 0
                    df_batch.to_csv(OUTPUT_CSV, mode="a", header=write_header, index=False)
                    total_written += len(batch)
                    print(
                        f"  {proto_name}/{dataset}/{model_name}/{seed}: {needed} -> {len(batch)} rows",
                        flush=True,
                    )

    print(f"\nDone. {total_written} rows -> {OUTPUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
```

**CLI usage:**
```bash
# Medium-only first (primary result), ~1.5h on CPU:
python scripts/run_cwms_msbs_full_sweep.py --medium-only

# All protocols (full robustness), ~4h on CPU:
python scripts/run_cwms_msbs_full_sweep.py

# With GPU (xgboost/catboost):
python scripts/run_cwms_msbs_full_sweep.py --gpu
```

### Step 2 — Verify first batch before leaving unattended

After the first 5 rows (one model/dataset/seed combo), check:
```python
import pandas as pd
df = pd.read_csv('outputs/cwms-msbs-full-sweep.csv')
print(df[['model','dataset','seed','noise_protocol','method','balanced_accuracy','n_synthetic']].head(10))
# Verify: cwms_msbs n_synthetic > 0, no NaN BA for lr/hgb/xgboost
```

### Step 3 — Monitor for errors

Resume is automatic (completed combos are skipped). If errors appear:
- `calibrated_lr` + CWMS sample_weight warning: expected (sklearn bug — sample_weight only used in calibration, not base LR). Results are still valid — calibrated LR just gets CWMS weights at calibration stage only.
- xgboost/lightgbm ImportError: models not installed → skip them, re-run without GPU flag.
- NaN balanced_accuracy: class collapse — check n_minority in that seed/dataset combo.

## Success Criteria

- [ ] `outputs/cwms-msbs-full-sweep.csv` exists with expected row count (medium-only: 1500, full: 4500)
- [ ] Zero ERROR rows for in-scope models on medium protocol
- [ ] `cwms_msbs` has `n_synthetic > 0` in every row
- [ ] `no_cleaning`, `class_proportional` have no NaN balanced_accuracy
- [ ] LR cwms_msbs mean BA > 0.70 on medium protocol (prelim showed 0.7485 across 3 seeds)
- [ ] HGB cwms_msbs mean BA > class_proportional on medium protocol

## Risk Assessment

- **calibrated_lr sample_weight routing**: sklearn bug where sample_weight doesn't propagate through CalibratedClassifierCV to the base LR. The wrapper only applies weights at calibration. This means CWMS weights are partially applied for calibrated_lr. Results are still reported but noted in paper.
- **xgboost/lightgbm availability**: if not installed, they won't appear. Paper simply excludes them and notes it.
- **Budget < n_minority**: when budget_count < n_minority_samples, MSBS distributes budget_count synthetics across seeds (per_seed=0 for most, only remainder seeds fire). This case is rare at budget=0.10 but could cause n_synthetic < expected. The CSV column `n_synthetic` reveals this.
- **Long compute time**: 6 models × 5 datasets × 10 seeds × 3 protocols = 900 combos × 5 methods each. Run `--medium-only` first for the primary result; extend to all protocols if time allows.
