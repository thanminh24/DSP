---
phase: 2
title: "Deep Multi-Metric Benchmark Sweep"
status: pending
priority: P1
effort: "~3h compute"
dependencies: [1]
---

# Phase 2: Deep Multi-Metric Benchmark Sweep

## Overview

Run the definitive CWMS+MSBS benchmark across all 6 in-scope model families, all 3 noise
protocols, 10 seeds, and 5 datasets. Uses the v2 CWMS weights (Phase 1 fix) and full metric
suite. Output: `outputs/cwms-msbs-deep-sweep.csv`. This is the paper's primary result table.

## Experiment Scope

| Dimension | Values |
|-----------|--------|
| Models | `lr`, `calibrated_lr` (2 methods only), `hgb`, `xgboost`, `lightgbm`, `catboost` |
| Datasets | `pima`, `credit-g`, `yeast`, `phoneme`, `ecoli` |
| Seeds | `[13, 17, 23, 29, 31, 37, 41, 43, 47, 53]` |
| Noise protocols | `hidden_minority_medium` (primary), `hidden_minority_high`, `hidden_minority_low` |
| Budget | 0.10 |
| Target ratio | 0.15 |
| Methods (all models except calibrated_lr) | `no_cleaning`, `class_proportional`, `msbs`, `cwms`, `cwms_msbs` |
| Methods (calibrated_lr only) | `no_cleaning`, `class_proportional` |

Expected rows (if catboost included): ~4,300 rows (~200 fewer for calibrated_lr's excluded methods).

## Conda Environment

**All commands must use the `dsp` conda env python:**
`/home/than-minh/miniconda3/envs/dsp/bin/python`

Catboost 1.2.8 is confirmed working in this env (smoke test passed in Phase 1).

## Related Code Files

- Create: `scripts/run_cwms_msbs_deep_sweep.py`
- Create: `scripts/analyze_cwms_msbs_deep_results.py`
- Read: `outputs/cwms-msbs-deep-sweep.csv` (output target)

## Script: run_cwms_msbs_deep_sweep.py

```python
"""Deep multi-metric sweep for CWMS+MSBS v2.

Scope: lr, calibrated_lr (no_cleaning+cp only), hgb, xgboost, lightgbm, catboost
       × 5 datasets × 10 seeds × 3 hidden_minority protocols × up to 5 methods
Output: outputs/cwms-msbs-deep-sweep.csv  (resume-safe)
Env: /home/than-minh/miniconda3/envs/dsp/bin/python
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
CALIBRATED_LR_METHODS = ["no_cleaning", "class_proportional"]   # sklearn routing bug
FULL_SEEDS = QUICK_SEEDS
FULL_PROTOCOLS = ["hidden_minority_medium", "hidden_minority_low", "hidden_minority_high"]
EXCLUDED_MODELS = {"random_forest", "extra_trees"}
POC_BUDGET = 0.10
POC_RATIO = 0.15
OUTPUT_CSV = PROJECT_ROOT / "outputs" / "cwms-msbs-deep-sweep.csv"


def _model_methods(model_name: str) -> list[str]:
    if model_name == "calibrated_lr":
        return CALIBRATED_LR_METHODS
    return FULL_METHODS


def _load_completed(path: Path) -> set:
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path)
        df = df[pd.to_numeric(df["balanced_accuracy"], errors="coerce").notna()]
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
            methods_for_model = _model_methods(model_name)
            for dataset in DATASETS:
                for seed in FULL_SEEDS:
                    needed = [
                        m for m in methods_for_model
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
/home/than-minh/miniconda3/envs/dsp/bin/python scripts/run_cwms_msbs_deep_sweep.py --medium-only

# All protocols (full robustness), ~3h on CPU:
/home/than-minh/miniconda3/envs/dsp/bin/python scripts/run_cwms_msbs_deep_sweep.py

# With GPU (xgboost/catboost):
/home/than-minh/miniconda3/envs/dsp/bin/python scripts/run_cwms_msbs_deep_sweep.py --gpu
```

## Script: analyze_cwms_msbs_deep_results.py

Key analysis steps (to be implemented in the script):

1. **Load and validate**: filter rows where `pd.to_numeric(balanced_accuracy, errors='coerce').isna()`
2. **Primary summary** (hidden_minority_medium): all metrics, grouped by method
3. **Per-model delta table**: cwms_msbs vs class_proportional for each metric
4. **Wilcoxon per-model**: BA, minority_recall, minority_precision, weighted_f1
5. **Boosting family aggregate**: hgb + xgboost + lightgbm + catboost combined
6. **LR family**: lr only (calibrated_lr shown separately with 2-method comparison)
7. **Cross-protocol robustness**: cwms_msbs vs class_proportional across all 3 protocols
8. **Metric correlation report**: does BA improvement co-occur with recall/precision improvement?

Expected improvements in v2 (target, not guaranteed):

| Model | v1 delta BA | v2 expected delta BA |
|-------|-------------|----------------------|
| lr | +4.22pp | ~same (fix doesn't affect LR) |
| hgb | +0.25pp | +1 to +2pp (balanced weights should help) |
| xgboost | −1.93pp | +1 to +3pp (fix removes double-correction) |
| lightgbm | −0.40pp | 0 to +2pp |
| catboost | N/A (failed) | similar to xgboost trend |

## Success Criteria

- [ ] `outputs/cwms-msbs-deep-sweep.csv` exists with expected row count
- [ ] No error rows for lr, hgb, xgboost, lightgbm, catboost on medium protocol
- [ ] calibrated_lr has rows for no_cleaning and class_proportional only (NaN for cwms/cwms_msbs)
- [ ] All rows contain: accuracy, weighted_f1, minority_precision, majority_recall columns
- [ ] xgboost cwms_msbs delta vs class_proportional is non-negative (verifies double-correction fix)
- [ ] Analysis script produces summary tables and per-model Wilcoxon p-values

## Risk Assessment

- **catboost compute time**: catboost 300 iterations on CPU is slow. If time is a concern,
  reduce to 200 iterations or run with `--gpu`. Alternatively, run medium-only first and
  extend to all protocols if results look good.
- **Resume correctness**: the CSV is written in append mode. If a row was partially written
  due to a crash, the `_load_completed()` filter on `pd.to_numeric(balanced_accuracy)` will
  reject corrupt rows and re-run them.
- **spw calibration**: for calibrated_lr, even `class_proportional` (deletion-based) doesn't
  benefit from spw. Results for this model should be interpreted as "deletion baseline only."
