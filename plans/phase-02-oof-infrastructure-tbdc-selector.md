---
phase: 2
title: "OOF Infrastructure + TBDC Selector"
status: pending
priority: P1
effort: "1h"
dependencies: [1]
---

# Phase 2: OOF Infrastructure + TBDC Selector

## Overview

Add `out_of_fold_scores()` to `oof_loss.py` (returns both loss AND probabilities, no
recomputation), add `select_tbdc()` to `selectors.py`, wire both into `experiment.py`.
Apply damping factor from Phase 1 diagnostic if overestimation_ratio > 2.0.

## Architecture

```
out_of_fold_scores(X, y, factory, n_splits, seed)
    → (losses: ndarray(n,), probs: ndarray(n, n_classes))

select_tbdc(suspiciousness, oof_probs, y_noisy, budget_count, minority_label, damping)
    → indices_to_delete: ndarray

experiment.py:
    suspiciousness, oof_probs = out_of_fold_scores(...)   # replaces out_of_fold_loss call
    selectors_map["tbdc"] = lambda: select_tbdc(suspiciousness, oof_probs, ...)
```

## Related Code Files

- Modify: `pipeline/scoring/oof_loss.py` (add `out_of_fold_scores`, keep `out_of_fold_loss` as wrapper)
- Modify: `pipeline/cleaning/selectors.py` (add `select_tbdc`, stays ≤ 200 lines)
- Modify: `pipeline/core/config.py` (add `tbdc_damping: float = 1.0` and `"tbdc"` to method_names)
- Modify: `pipeline/core/experiment.py` (swap out_of_fold_loss → out_of_fold_scores, add tbdc dispatch)

## Implementation Steps

### Step 1: `pipeline/scoring/oof_loss.py`

Read current file (39 lines). Add `out_of_fold_scores()` that returns both outputs.
Keep `out_of_fold_loss` as a one-line wrapper for backward compatibility.
File must stay ≤ 200 lines (it will be ~70 lines after this change).

```python
def out_of_fold_scores(
    X: np.ndarray,
    y_noisy: np.ndarray,
    model_factory: Callable,
    n_splits: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """OOF cross-entropy loss AND calibrated class probabilities.

    Returns:
        losses: shape (n,) — cross-entropy per sample (higher = more suspicious)
        probs:  shape (n, n_classes) — OOF predicted probabilities
    """
    n = len(y_noisy)
    n_classes = len(np.unique(y_noisy))
    probs = np.zeros((n, n_classes), dtype=float)
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, valid_idx in folds.split(X, y_noisy):
        model = model_factory()
        model.fit(X[train_idx], y_noisy[train_idx])
        probs[valid_idx] = model.predict_proba(X[valid_idx])
    losses = -np.log(np.clip(probs[np.arange(n), y_noisy], 1e-12, 1.0))
    return losses, probs


def out_of_fold_loss(...) -> np.ndarray:
    """Backward-compatible wrapper — returns only losses."""
    losses, _ = out_of_fold_scores(X, y_noisy, model_factory, n_splits, seed)
    return losses
```

### Step 2: `pipeline/cleaning/selectors.py`

Read current file (142 lines). Append `select_tbdc` after `select_crcc_m`.
File must stay ≤ 200 lines (142 + ~35 = 177 lines — fine).

```python
def select_tbdc(
    suspiciousness: np.ndarray,
    oof_probs: np.ndarray,
    y_noisy: np.ndarray,
    budget_count: int,
    minority_label: int = 1,
    damping: float = 1.0,
) -> np.ndarray:
    """Type-B-Directed Cleaning: routes budget by OOF noise density, not class frequency.

    Cap = round(budget × disagreement_rate × damping) where disagreement_rate is the
    fraction of minority-pool samples the OOF model predicts as majority class.
    This estimates Type B density (true majority mislabeled as minority) directly
    from the trained OOF classifier rather than using class frequency as a proxy.

    damping: multiplier to reduce cap if Phase 1 diagnostic shows overestimation.
    """
    if budget_count <= 0:
        return np.array([], dtype=int)

    minority_pool = np.where(y_noisy == minority_label)[0]
    if len(minority_pool) == 0:
        return select_global(suspiciousness, budget_count)

    oof_pred = np.argmax(oof_probs[minority_pool], axis=1)
    disagreement_rate = float(np.mean(oof_pred != minority_label))
    minority_cap = int(round(budget_count * disagreement_rate * damping))
    minority_cap = min(minority_cap, len(minority_pool), budget_count)
    majority_cap = budget_count - minority_cap

    majority_pool = np.where(y_noisy != minority_label)[0]
    class_caps = {minority_label: minority_cap}
    for lbl in np.unique(y_noisy):
        if lbl != minority_label:
            class_caps[lbl] = majority_cap

    deleted_count = {lbl: 0 for lbl in np.unique(y_noisy)}
    selected: list[int] = []
    for idx in np.argsort(suspiciousness)[::-1]:
        lbl = int(y_noisy[idx])
        if len(selected) >= budget_count:
            break
        if deleted_count[lbl] < class_caps[lbl]:
            selected.append(int(idx))
            deleted_count[lbl] += 1
    return np.array(selected, dtype=int)
```

### Step 3: `pipeline/core/config.py`

Add `tbdc_damping: float = 1.0` field to `BaseExperimentConfig`.
Add `"tbdc"` to `method_names` property.

### Step 4: `pipeline/core/experiment.py`

- Change import: add `out_of_fold_scores` (keep `out_of_fold_loss` import for compatibility)
- Change scoring call:
  ```python
  # Before:
  suspiciousness = out_of_fold_loss(X_tr, y_noisy, model_factory, ...)
  # After:
  suspiciousness, oof_probs = out_of_fold_scores(X_tr, y_noisy, model_factory, ...)
  ```
- Add to `selectors_map`:
  ```python
  from pipeline.cleaning.selectors import select_tbdc
  selectors_map["tbdc"] = lambda: select_tbdc(
      suspiciousness, oof_probs, y_noisy, budget_count,
      minority_label=minority_label,
      damping=cfg.tbdc_damping,
  )
  ```

### Step 5: Smoke test

```bash
/home/than-minh/miniconda3/bin/python3 scripts/run_crcc_smoke_test.py
```

Verify: output CSV contains a `tbdc` row for each (dataset, model, seed) combo. No NaN on normal datasets. Ecoli may NaN — acceptable.

## Success Criteria

- [ ] `out_of_fold_scores()` added to oof_loss.py; `out_of_fold_loss` still works as wrapper
- [ ] `select_tbdc()` added to selectors.py; file ≤ 200 lines
- [ ] `tbdc_damping` in config; `"tbdc"` in method_names
- [ ] experiment.py uses `out_of_fold_scores` (single CV pass, no double computation)
- [ ] Smoke test passes: `tbdc` appears in output CSV with valid BA values for ≥ 4/5 datasets
- [ ] Damping factor set per Phase 1 diagnostic result (default 1.0 if ratio < 2.0)

## Risk Assessment

- **experiment.py at 190 lines**: adding oof_probs plumbing may push past 200. Check line count
  after changes; if > 200, extract `_run_selectors()` helper into a separate module.
- **oof_loss.py backward compat**: `out_of_fold_loss` must remain callable with the same
  signature — all existing scripts (`run_full_experiment.py`, `run_pilot.py`, etc.) use it.
- **Memory**: storing `oof_probs` (n × 2 float64) adds ~0.1 MB for phoneme (n≈4000). Negligible.
