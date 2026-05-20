---
phase: 2
title: "Data Loading and Preprocessing Module"
status: completed
priority: P1
effort: "2h"
dependencies: [1]
---

# Phase 2: Data Loading and Preprocessing Module

## Overview

Write `scripts/data-loader.py` — the single source of truth for dataset loading, imbalance induction (85/15 subsampling), and class-dependent label noise injection. All downstream scripts import from this module.

## Requirements

- Functional:
  - `load_dataset(name)` returns `(X: np.ndarray, y: np.ndarray, categorical_mask: np.ndarray | None, feature_names: list[str])`
  - `induce_imbalance(X, y, minority_label, target_ratio, rng)` returns `(X_imb, y_imb)`
  - `inject_noise(y_clean, minority_label, min_to_maj_rate, maj_to_min_rate, rng)` returns `(y_noisy, noisy_mask)`
  - Datasets: `"pima"` (OpenML 37), `"credit-g"` (OpenML 31), `"sick"` (OpenML 38)
- Non-functional: module ≤ 200 lines; no global state; all functions pure given rng.

## Architecture

```
load_dataset(name: str) -> (X, y, cat_mask, feat_names)
  └─ pd.read_parquet(f"data/{name}.parquet") → encode labels to {0,1} → return numpy
  └─ No network call — all 3 datasets pre-cached in data/

induce_imbalance(X, y, minority_label, target_ratio, rng) -> (X, y)
  └─ subsample majority OR minority to hit target_ratio
  └─ Pima: subsample minority to 15%
  └─ Credit-G: subsample minority to 15%
  └─ Sick: skip (natural ~6.1% already stronger than 85/15)

inject_noise(y_clean, minority_label, rates, rng) -> (y_noisy, noisy_mask)
  └─ per-sample Bernoulli flip using class-conditional rates
```

**Dataset specifics:**

| Dataset | Local file | Features | Preprocessing for LR |
|---------|-----------|----------|-----------------------|
| pima | `data/pima.parquet` | 8 numeric | StandardScaler |
| credit-g | `data/credit-g.parquet` | 20 mixed | OrdinalEncoder + StandardScaler |
| sick | `data/sick.parquet` | 29 mixed + missing | SimpleImputer(median/mode) + OrdinalEncoder + StandardScaler |

**Label encoding:** Convert string labels to int. For each dataset, the minority class becomes label `1`, majority becomes `0`.

| Dataset | minority (label=1) | majority (label=0) |
|---------|--------------------|--------------------|
| pima | tested_positive (268) | tested_negative (500) |
| credit-g | bad (300) | good (700) |
| sick | sick (231) | negative (3541) |

**Imbalance induction logic:**
- If natural minority ratio ≥ target_ratio already: keep natural imbalance (sick).
- Otherwise: subsample minority to `floor(n_majority × target_ratio / (1 - target_ratio))`.
- Never subsample majority; never drop test samples (imbalance induction applies only to train split).

**Noise protocol:**
- `min_to_maj_rate = 0.30` (30% of minority flipped to majority label)
- `maj_to_min_rate = 0.10` (10% of majority flipped to minority label)
- `noisy_mask[i] = True` iff sample i was flipped — used by oracle selector and harm metric

## Related Code Files

- Create: `scripts/data-loader.py`
- Read for context: `scripts/run-crcc-smoke-test.py` (reuse `induce_training_imbalance`, `inject_class_dependent_noise` logic)

## Implementation Steps

1. Read `scripts/run-crcc-smoke-test.py` — extract the imbalance and noise functions as reference.

2. Write `scripts/data-loader.py`:

   ```python
   DATASETS = {
       "pima":     {"openml_id": 37,  "target": "class",  "minority_str": "tested_positive"},
       "credit-g": {"openml_id": 31,  "target": "class",  "minority_str": "bad"},
       "sick":     {"openml_id": 38,  "target": "Class",  "minority_str": "sick"},
   }
   TARGET_MINORITY_RATIO = 0.15
   ```

   - `load_dataset(name)`:
     - `pd.read_parquet(f"data/{name}.parquet")` → split off `target` column
     - Encode y: minority_str → 1, else → 0
     - Return X as pd.DataFrame (preserves dtypes), y as int array, list of categorical column names, feature_names

   - `induce_imbalance(X, y, minority_label=1, target_ratio=0.15, rng)`:
     - Current minority ratio = `mean(y == minority_label)`
     - If current ratio ≤ target_ratio: subsample minority down to target count
     - If current ratio > target_ratio already: return unchanged (sick case)
     - Use `rng.choice` without replacement on minority indices

   - `inject_noise(y_clean, minority_label=1, min_to_maj=0.30, maj_to_min=0.10, rng)`:
     - Port directly from smoke test `inject_class_dependent_noise`
     - Return `(y_noisy, noisy_mask)` where noisy_mask is bool array

3. Add `if __name__ == "__main__":` quick-check that loads all 3 datasets, prints shapes and class distributions.

4. Run validation:
   ```
   /home/than-minh/miniconda3/bin/python3 scripts/data-loader.py
   ```

## Success Criteria

- [ ] `load_dataset("pima")` returns shape `(768, 8)`, y with ~268 positives
- [ ] `load_dataset("credit-g")` returns shape `(1000, 20)`, y with 300 minority
- [ ] `load_dataset("sick")` returns shape `(3772, 29)`, y with 231 minority
- [ ] `induce_imbalance` on pima/credit-g yields ≈15% minority after split
- [ ] `inject_noise` flips roughly 30% of minority and 10% of majority (within ±5% tolerance across seeds)
- [ ] `noisy_mask` is True only for actually flipped samples
- [ ] Module ≤ 200 lines

## Risk Assessment

- Sick dataset has missing values: `SimpleImputer` handles this in the LR preprocessing pipeline (not in loader — loader returns raw values, preprocessor built in orchestrator).
- Credit-G has 13 categorical columns: loader returns them as object dtype; downstream pipeline uses OrdinalEncoder.
- All datasets loaded from local Parquet — zero network dependency, instant load.
