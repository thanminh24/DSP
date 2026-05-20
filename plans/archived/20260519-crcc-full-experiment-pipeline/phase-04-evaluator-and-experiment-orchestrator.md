---
phase: 4
title: "Evaluator and Experiment Orchestrator"
status: completed
priority: P1
effort: "3h"
dependencies: [2, 3]
---

# Phase 4: Evaluator and Experiment Orchestrator

## Overview

Write `scripts/evaluator.py` (retrain + 5-metric evaluation) and `scripts/run-full-experiment.py` (the main loop that combines all modules and runs the full 180-run experiment). This is the most complex module — keep each file under 200 lines through careful separation.

## Requirements

- Functional:
  - `evaluator.py`: `evaluate(selected_idx, X_train, y_noisy, y_clean, noisy_mask, X_test, y_test, model_factory, minority_label) -> dict`
  - `run-full-experiment.py`: loops over all `(dataset, model, seed, method)` combinations, writes `outputs/full-experiment-results.csv`
  - Lambda ablation: runs CRCC-P with λ ∈ {0.0, 0.25, 0.5, 1.0}, stored as separate method names (`crcc_p_l0`, `crcc_p_l025`, `crcc_p_l05`, `crcc_p_l10`)
- Non-functional: progress printed to stdout; single CSV output; no global state.

## Architecture

### evaluator.py

```
evaluate(selected_idx, X_train, y_noisy, y_clean, noisy_mask,
         X_test, y_test, model_factory, minority_label=1) -> dict
  └─ keep_mask = ~isin(arange(n), selected_idx)
  └─ model = model_factory().fit(X_train[keep_mask], y_noisy[keep_mask])
  └─ y_pred = model.predict(X_test)
  └─ metrics:
       balanced_accuracy  = balanced_accuracy_score(y_test, y_pred)
       macro_f1           = f1_score(y_test, y_pred, average="macro")
       minority_recall    = recall_score(y_test, y_pred, pos_label=minority_label)
       noise_precision_deleted = noisy_mask[selected_idx].mean()  (0.0 if empty)
       clean_minority_deletion_rate = (y_clean[selected_idx]==minority_label
                                       & ~noisy_mask[selected_idx]).mean()  (0.0 if empty)
  └─ return dict with all metrics + metadata (dataset, model, method, seed, deleted_count)
```

### run-full-experiment.py

**Experiment config (dataclass):**
```python
@dataclass
class Config:
    datasets: tuple = ("pima", "credit-g", "sick")
    seeds: tuple = (13, 29, 47, 61, 83)
    test_size: float = 0.25
    target_minority_ratio: float = 0.15
    minority_to_majority_noise: float = 0.30
    majority_to_minority_noise: float = 0.10
    cleaning_budget: float = 0.10
    lambda_grid: tuple = (0.0, 0.25, 0.5, 1.0)
    minority_cap_factor_m: float = 0.5
    n_cv_folds: int = 5
```

**Model factories:**
```python
def make_lr_pipeline(X, categorical_mask):
    # if categorical cols exist: ColumnTransformer(OrdinalEncoder + StandardScaler)
    # else: StandardScaler pipeline
    # LogisticRegression(max_iter=1000, random_state=seed)

def make_hgb_pipeline(X, categorical_mask):
    # HistGradientBoostingClassifier(random_state=seed)
    # categorical_features=categorical_mask (native support)
    # No separate encoder needed
```

**Per-dataset loop:**
```
load_dataset(name) → X_raw, y_raw, cat_mask, feat_names
train_test_split(X_raw, y_raw, test_size=0.25, stratify=y_raw, random_state=seed)
induce_imbalance(X_train, y_train, target_ratio=0.15, rng)
inject_noise(y_imb, rates, rng) → y_noisy, noisy_mask
scoring: out_of_fold_loss(X_imb, y_noisy, model_factory, n_folds, seed)
budget_count = max(1, round(0.10 × len(y_noisy)))
run all 6+4 method variants → evaluate each → append to results list
```

**Methods run per combo:**
| Method name | Selector call |
|-------------|--------------|
| `no_cleaning` | `select_none(n)` |
| `random_deletion` | `select_random(n, budget, rng)` |
| `global_top_loss` | `select_global(susp, budget)` |
| `class_proportional` | `select_class_proportional(susp, y, budget)` |
| `oracle_deletion` | `select_oracle(noisy_mask, budget)` |
| `crcc_p_l0` | `select_crcc_p(susp, y, budget, lambda_risk=0.0)` |
| `crcc_p_l025` | `select_crcc_p(susp, y, budget, lambda_risk=0.25)` |
| `crcc_p_l05` | `select_crcc_p(susp, y, budget, lambda_risk=0.5)` |
| `crcc_p_l10` | `select_crcc_p(susp, y, budget, lambda_risk=1.0)` |
| `crcc_m` | `select_crcc_m(susp, y, budget, lambda_risk=0.5)` |

**Total runs:** 3 datasets × 2 models × 5 seeds × 10 method variants = **300 rows** in CSV.

**Preprocessing per model:**
- LR on pima: `StandardScaler`
- LR on credit-g/sick: `ColumnTransformer` with `OrdinalEncoder` (categoricals) + `StandardScaler` (numerics) + `SimpleImputer(strategy="median")` (missing, sick only)
- HGB on all: `HistGradientBoostingClassifier(categorical_features=cat_mask)` — no encoding needed

**Key implementation note — HGB and categorical encoding:**
`HistGradientBoostingClassifier` in scikit-learn 1.1+ accepts `categorical_features` as a boolean mask. For OpenML datasets with object-dtype columns, encode to integer codes first using pandas `Categorical.codes`. Return the integer-encoded array from `data-loader.py` for HGB; LR pipeline applies OrdinalEncoder internally via ColumnTransformer.

Actually, simpler approach: loader returns raw pandas DataFrame; orchestrator converts to numpy depending on model type. Or: loader always returns numpy with categorical columns as integer-encoded (OrdinalEncoder applied in loader). See implementation note below.

**Simpler approach (recommended):** Loader returns `(X_df: pd.DataFrame, y: np.ndarray, cat_cols: list[str])`. Orchestrator builds the sklearn pipeline using `ColumnTransformer` for LR and a simpler encoding for HGB.

## Related Code Files

- Create: `scripts/evaluator.py`
- Create: `scripts/run-full-experiment.py`
- Read: `scripts/data-loader.py`, `scripts/scoring.py`, `scripts/selectors.py`

## Implementation Steps

1. Write `scripts/evaluator.py`:
   - Import: numpy, sklearn.metrics
   - Single function `evaluate(...)` returning dict
   - Handle empty `selected_idx` (no_cleaning case) gracefully — `noise_precision` and `clean_minority_rate` default to 0.0
   - Handle edge case: if all kept samples are one class, metrics still compute (sklearn handles it with `zero_division=0`)

2. Write `scripts/run-full-experiment.py`:
   - Top-level: `ExperimentConfig` dataclass
   - `build_lr_pipeline(X_df, cat_cols, seed)` → sklearn Pipeline
   - `build_hgb_pipeline(X_df, cat_cols, seed)` → sklearn Pipeline with categorical support
   - `run_single(dataset_name, model_name, seed, cfg)` → list of dicts (one per method)
   - `main()`: iterate all combos, collect results, `pd.DataFrame(results).to_csv(...)`
   - Print progress: `[dataset/model/seed] N/total runs completed`

3. For LR pipeline on mixed datasets:
   ```python
   from sklearn.compose import ColumnTransformer
   from sklearn.impute import SimpleImputer
   from sklearn.preprocessing import OrdinalEncoder, StandardScaler
   from sklearn.pipeline import Pipeline

   numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
   cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))])
   preprocessor = ColumnTransformer([("num", numeric_pipe, numeric_cols), ("cat", cat_pipe, cat_cols)])
   full_pipe = Pipeline([("prep", preprocessor), ("clf", LogisticRegression(max_iter=1000, random_state=seed))])
   ```

4. For HGB pipeline:
   ```python
   # Encode categoricals to integer codes for HGB
   # HGB handles missing values natively (NaN supported)
   from sklearn.preprocessing import OrdinalEncoder
   cat_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
   X_encoded = X_df.copy()
   X_encoded[cat_cols] = cat_encoder.fit_transform(X_df[cat_cols].fillna("__missing__"))
   # cat_mask = boolean array True for cat_cols positions
   hgb = HistGradientBoostingClassifier(categorical_features=cat_mask, random_state=seed)
   ```

5. Output: `outputs/full-experiment-results.csv` with columns:
   `dataset, model, seed, method, deleted, balanced_accuracy, macro_f1, minority_recall, noise_precision_deleted, clean_minority_deletion_rate`

## Success Criteria

- [ ] `evaluate()` returns dict with all 5 metrics + metadata
- [ ] `no_cleaning` rows have `deleted=0`, harm metrics = 0.0
- [ ] `oracle_deletion` rows have `noise_precision_deleted` ≈ 1.0 (all deleted are truly noisy)
- [ ] `run-full-experiment.py` completes without error on all 3 datasets × 2 models
- [ ] Output CSV has 300 rows (3 datasets × 2 models × 5 seeds × 10 methods)
- [ ] Each module ≤ 200 lines

## Risk Assessment

- HGB categorical_features: In scikit-learn ≥ 1.2, `categorical_features` accepts boolean mask or list of indices. Use `list(np.where(cat_mask)[0])` for compatibility.
- Sick missing values: HGB handles NaN natively. LR pipeline uses SimpleImputer. Verify no remaining NaN after imputation before fitting LR.
- Train split size: After 85/15 imbalance and 25% test split, pima train set ≈ 432 rows. 5-fold CV means ≈86 samples per fold — adequate for LR; acceptable for HGB.
- Class cap rounding for small datasets: with budget=10% of 432 ≈ 43 samples and minority=15% ≈ 65 samples, minority cap ≈ `round(43 × 0.15) = 6`. Always clamp to `max(1, cap)`.
