---
phase: 3
title: "Pipeline Restructure + Codebase Refactor"
status: pending
priority: P1
effort: "3h"
dependencies: [2]
---

# Phase 3: Pipeline Restructure + Codebase Refactor

## Overview

Move all reusable ML logic into a proper `pipeline/` package (importable modules), leaving `scripts/` as entry-point runners only. Simultaneously fix all code quality violations: 200-line limit, snake_case, DRY (centralize LAMBDA_NAMES), docstrings, type hints, input validation, error handling.

## Target Directory Structure

```
DSP/
├── pipeline/                          # NEW — importable ML module package
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── loaders.py                 # from data_loader.py
│   ├── cleaning/
│   │   ├── __init__.py
│   │   └── selectors.py               # from cleaning_selectors.py
│   ├── scoring/
│   │   ├── __init__.py
│   │   └── oof_loss.py                # from scoring.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py                 # from evaluator.py
│   └── core/
│       ├── __init__.py
│       ├── experiment.py              # shared run_single(), model factories, LAMBDA_NAMES
│       └── config.py                  # ExperimentConfig, MildConfig base class
├── scripts/                           # entry-point runners only (no reusable logic)
│   ├── run_full_experiment.py         # imports from pipeline/, config + main() only
│   ├── run_mild_imbalance_experiment.py
│   ├── run_noise_ablation.py          # (created in Phase 4)
│   ├── run_budget_ablation.py         # (created in Phase 5)
│   ├── run_statistical_tests.py       # (created in Phase 6)
│   ├── run_oracle_paradox_analysis.py # (created in Phase 7)
│   ├── run_crcc_smoke_test.py         # RENAMED (was run-crcc-smoke-test.py)
│   ├── download_datasets.py           # (created in Phase 2)
│   └── validate_environment.py        # existing
├── data/                              # parquet files
├── outputs/                           # CSV + plots
└── docs/
```

## Module Responsibilities

| Module | Responsibility | Max Lines |
|--------|---------------|-----------|
| `pipeline/data/loaders.py` | `load_dataset()`, all dataset loaders | 150 |
| `pipeline/cleaning/selectors.py` | 7 selector functions + input validation + docstrings | 180 |
| `pipeline/scoring/oof_loss.py` | `out_of_fold_loss()` | 50 |
| `pipeline/evaluation/metrics.py` | `evaluate()` | 60 |
| `pipeline/core/experiment.py` | `run_single()`, `_encode_dataframe()`, model factories, `print_lambda_sensitivity()` (moved from mild), one-class guard | 180 |
| `pipeline/core/config.py` | `BaseExperimentConfig`, `LAMBDA_NAMES` | 80 |
| `scripts/run_full_experiment.py` | `ExperimentConfig(BaseExperimentConfig)` + `main()` | 100 |
| `scripts/run_mild_imbalance_experiment.py` | `MildConfig(BaseExperimentConfig)` + `main()` | 100 |

## Related Code Files

### Create (pipeline/ package)
- `pipeline/__init__.py`
- `pipeline/data/__init__.py`, `pipeline/data/loaders.py`
- `pipeline/cleaning/__init__.py`, `pipeline/cleaning/selectors.py`
- `pipeline/scoring/__init__.py`, `pipeline/scoring/oof_loss.py`
- `pipeline/evaluation/__init__.py`, `pipeline/evaluation/metrics.py`
- `pipeline/core/__init__.py`, `pipeline/core/experiment.py`, `pipeline/core/config.py`

### Modify (scripts/ — shrink to entry points)
- `scripts/run_full_experiment.py`
- `scripts/run_mild_imbalance_experiment.py`
- `scripts/run_crcc_smoke_test.py` (rename)

### Delete (content moved to pipeline/)
- `scripts/cleaning_selectors.py` — content moved to `pipeline/cleaning/selectors.py`
- `scripts/scoring.py` — content moved to `pipeline/scoring/oof_loss.py`
- `scripts/evaluator.py` — content moved to `pipeline/evaluation/metrics.py`
- `scripts/data_loader.py` — content moved to `pipeline/data/loaders.py`

## Implementation Steps

### Step 1 — Create pipeline/ package skeleton

```bash
mkdir -p pipeline/{data,cleaning,scoring,evaluation,core}
touch pipeline/__init__.py
touch pipeline/{data,cleaning,scoring,evaluation,core}/__init__.py
```

### Step 2 — Move `cleaning_selectors.py` → `pipeline/cleaning/selectors.py`

Copy content, then apply all code quality fixes:
- Add docstring to every selector (see template in Phase 02 original plan)
- Add input validation to `select_crcc_p` and `select_crcc_m`:
  ```python
  if not 0.0 <= lambda_risk <= 1.0:
      raise ValueError(f"lambda_risk must be in [0, 1], got {lambda_risk}")
  ```
- Remove any import of `LAMBDA_NAMES` (that lives in `pipeline/core/config.py`)

### Step 3 — Move `scoring.py` → `pipeline/scoring/oof_loss.py`

Add `Callable` type hint:
```python
from typing import Callable
def out_of_fold_loss(X: np.ndarray, y: np.ndarray, model_factory: Callable,
                     n_splits: int = 5, seed: int = 42) -> np.ndarray:
```

### Step 4 — Move `evaluator.py` → `pipeline/evaluation/metrics.py`

No logic changes — just move and update imports.

### Step 5 — Move `data_loader.py` → `pipeline/data/loaders.py`

Include new loaders from Phase 2 (yeast, ecoli, phoneme). Remove sick loader.
Update `DATA_DIR` path to be relative to package root:
```python
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
```

### Step 6 — Create `pipeline/core/config.py`

```python
from dataclasses import dataclass, field
from typing import Callable

LAMBDA_NAMES: dict[float, str] = {
    0.0: "crcc_p_l0",
    0.25: "crcc_p_l025",
    0.5: "crcc_p_l05",
    1.0: "crcc_p_l10",
}

@dataclass(frozen=True)
class BaseExperimentConfig:
    """Base config shared by all experiment variants."""
    datasets: tuple[str, ...] = ("pima", "credit-g", "yeast", "ecoli", "phoneme")
    seeds: tuple[int, ...] = (13, 29, 47, 61, 83)
    test_size: float = 0.25
    target_minority_ratio: float = 0.15
    minority_to_majority_noise: float = 0.30
    majority_to_minority_noise: float = 0.10
    cleaning_budget: float = 0.10
    lambda_grid: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)
    minority_cap_factor_m: float = 0.5
    n_cv_folds: int = 5
    minority_label: int = 1

    @property
    def method_names(self) -> list[str]:
        names = ["no_cleaning", "random_deletion", "global_top_loss",
                 "class_proportional", "oracle_deletion"]
        for lam in self.lambda_grid:
            names.append(LAMBDA_NAMES[lam])
        names.append("crcc_m")
        return names
```

### Step 7 — Create `pipeline/core/experiment.py`

Move `_encode_dataframe()`, `_make_lr_factory()`, `_make_hgb_factory()`, `run_single()` here. Import from `pipeline.cleaning.selectors`, `pipeline.scoring.oof_loss`, `pipeline.evaluation.metrics`, `pipeline.data.loaders`, `pipeline.core.config`.

Critical: the one-class safety guard MUST be in `run_single()` (currently only in `run_mild_imbalance_experiment.py`, missing from `run_full_experiment.py`). Port the full guard from `run_mild_imbalance_experiment.py:169-196`:
```python
keep_mask = np.ones(n, dtype=bool)
keep_mask[selected] = False
if len(np.unique(y_noisy[keep_mask])) < 2:
    metrics = {
        "deleted": int(len(selected)),
        "balanced_accuracy": float("nan"),
        "macro_f1": float("nan"),
        "minority_recall": float("nan"),
        "noise_precision_deleted": float("nan"),
        "clean_minority_deletion_rate": float("nan"),
    }
else:
    try:
        metrics = evaluate(...)
    except ValueError:
        metrics = { ... same NaN dict ... }
```
This is mandatory for Ecoli (10.7% minority) edge cases.

### Step 8 — Refactor `scripts/run_full_experiment.py`

After Phase 3, this file should be ≤ 100 lines:
```python
from pipeline.core.config import BaseExperimentConfig
from pipeline.core.experiment import run_single
# ExperimentConfig inherits BaseExperimentConfig, overrides nothing (uses all defaults)
# main(): loop, call run_single(), save CSV
```

### Step 9 — Refactor `scripts/run_mild_imbalance_experiment.py`

```python
from pipeline.core.config import BaseExperimentConfig
from pipeline.core.experiment import run_single

@dataclass(frozen=True)
class MildConfig(BaseExperimentConfig):
    target_minority_ratio: float = 0.30   # override
    cleaning_budget: float = 0.20          # override
```

### Step 10 — Rename smoke test

```bash
git mv scripts/run-crcc-smoke-test.py scripts/run_crcc_smoke_test.py
```
Update its imports to use `pipeline.*` paths.

### Step 11 — Regression test

```bash
# Backup existing CSVs
cp outputs/full-experiment-results.csv outputs/full-experiment-results-backup.csv

# Re-run full experiment with new pipeline imports
python3 scripts/run_full_experiment.py

# Check CSV has same shape (300 rows × same columns)
python3 -c "
import pandas as pd
orig = pd.read_csv('outputs/full-experiment-results-backup.csv')
new  = pd.read_csv('outputs/full-experiment-results.csv')
print('Shape match:', orig.shape == new.shape)
# Note: values differ because datasets changed (sick→yeast+ecoli+phoneme)
# Just verify the new CSV has same structure and no all-NaN columns
print('NaN check:', new.isna().all().any())
"
```

Note: CSV values will differ from backup because datasets changed. Shape: 5 datasets × 2 models × 5 seeds × 10 methods = 500 rows (was 300 with 3 datasets).

### Step 12 — Verify all line counts

```bash
find pipeline/ scripts/ -name "*.py" | xargs wc -l | sort -n
```
Every file must be ≤ 200 lines.

## Success Criteria

- [ ] `pipeline/` package exists with all 6 modules
- [ ] `pipeline/cleaning/selectors.py` has docstrings on all 7 functions + input validation
- [ ] `pipeline/core/config.py` contains the single `LAMBDA_NAMES` dict — no other file defines it
- [ ] `pipeline/scoring/oof_loss.py` has `Callable` type hint on `model_factory`
- [ ] `scripts/run_full_experiment.py` ≤ 100 lines (config + main only)
- [ ] `scripts/run_mild_imbalance_experiment.py` ≤ 100 lines
- [ ] `scripts/run_crcc_smoke_test.py` exists (hyphen removed)
- [ ] Old `scripts/cleaning_selectors.py`, `scripts/scoring.py`, `scripts/evaluator.py`, `scripts/data_loader.py` deleted
- [ ] All files in `pipeline/` and `scripts/` ≤ 200 lines
- [ ] `scripts/run_full_experiment.py` runs to completion, producing 500-row CSV

## Risk Assessment

- Import paths: all experiment scripts must add project root to `sys.path` or be run from project root. Use `Path(__file__).resolve().parent.parent` pattern consistently.
- `pipeline/data/loaders.py` `DATA_DIR` must point to `DSP/data/`, not relative to the pipeline subpackage. Use `Path(__file__).resolve().parent.parent.parent / "data"`.
- Phoneme `minority_label=0`: the default config has `minority_label=1`. **REQUIRED**: add `minority_label_map: dict = field(default_factory=lambda: {"phoneme": 0})` to `BaseExperimentConfig`. In `run_single()`, resolve `minority_label = cfg.minority_label_map.get(dataset_name, cfg.minority_label)` and pass to all selectors + evaluator. This is a hard requirement — without it, Phoneme metrics are inverted.
