---
phase: 2
title: Implement Alternative Scorers + New Models + CUDA Guard
status: completed
priority: P1
effort: 3h
dependencies: []
---

# Phase 2: Implement Alternative Scorers + New Models + CUDA Guard

## Overview

Four parallel additions before any sweep runs:
1. **k-NN ratio scorer + cross-family OOF scorer** — new CWMS+MSBS method variants
2. **Plain SMOTE baseline** — noise-unaware oversampling, appears in all competitor papers
3. **New model: SVM** — add to factories; restore RF/ET from EXCLUDED_MODELS
4. **CUDA auto-detect + force-GPU** — startup check; abort if CUDA unavailable when `--gpu` passed

## Related Code Files

- **Create:** `pipeline/scoring/knn_ratio.py`
- **Modify:** `pipeline/models/factories.py` — add `svm` factory, remove RF/ET from exclusion list
- **Modify:** `scripts/run_cwms_msbs_deep_sweep.py` — add CUDA check, remove EXCLUDED_MODELS hardcoding
- **Modify:** `scripts/run_relabeling_viability_sweep.py` — add new method dispatchers, lazy scorer compute
- **Read:** `pipeline/scoring/balanced_oof.py` — interface contract to replicate for knn scorer

## Part A — CUDA Auto-Detection

### What to add to sweep scripts

Add a `_check_gpu()` function called at startup of any script that accepts `--gpu`:

```python
def _check_gpu():
    """Verify CUDA is available for boosting models. Abort if not."""
    results = {}
    try:
        import xgboost as xgb
        import numpy as np
        m = xgb.XGBClassifier(device="cuda", n_estimators=2, verbosity=0)
        m.fit(np.random.randn(20, 3), np.random.randint(0, 2, 20))
        results["xgboost"] = "OK"
    except Exception as e:
        results["xgboost"] = f"FAIL: {e}"
    try:
        import lightgbm as lgb
        m = lgb.LGBMClassifier(device="gpu", n_estimators=2, verbose=-1)
        m.fit(np.random.randn(20, 3), np.random.randint(0, 2, 20))
        results["lightgbm"] = "OK"
    except Exception as e:
        results["lightgbm"] = f"FAIL: {e}"
    try:
        from catboost import CatBoostClassifier
        m = CatBoostClassifier(task_type="GPU", devices="0", n_estimators=2, verbose=0)
        m.fit(np.random.randn(20, 3), np.random.randint(0, 2, 20))
        results["catboost"] = "OK"
    except Exception as e:
        results["catboost"] = f"FAIL: {e}"
    failed = [k for k, v in results.items() if not v.startswith("OK")]
    for k, v in results.items():
        print(f"  GPU check [{k}]: {v}", flush=True)
    if failed:
        raise RuntimeError(
            f"--gpu requested but CUDA unavailable for: {failed}. "
            "Remove --gpu or fix CUDA installation."
        )
    print("GPU check: all OK", flush=True)
```

Call in `main()` before the sweep loop:
```python
def main():
    args = sys.argv[1:]
    use_gpu = "--gpu" in args
    if use_gpu:
        print("Checking GPU availability...", flush=True)
        _check_gpu()
    ...
```

**Note:** HGB, LR, SVM, RF, ET have no GPU path — they silently run on CPU regardless of `--gpu`. This is correct behavior and requires no guard.

## Part B — Add SVM to Model Factories

### Why SVM is CWMS-compatible

- `SVC(probability=True).fit(X, y, sample_weight=sw)` — confirmed working
- `SVC.predict_proba()` available → balanced OOF scoring works
- `class_weight="balanced"` available → `bal_factory` works
- No GPU path (CPU only, but fast enough on tabular data)
- Slower than boosting at scale but legitimate baseline in classification literature

### Factory addition in `pipeline/models/factories.py`

Add imports at top:
```python
from sklearn.svm import SVC
```

Add to `list_publication_models()`:
```python
def list_publication_models(include_optional: bool = True) -> list[str]:
    models = ["lr", "svm", "calibrated_lr", "hgb",
              "random_forest", "extra_trees"]
    if include_optional:
        for name, package in [
            ("xgboost", "xgboost"),
            ("lightgbm", "lightgbm"),
            ("catboost", "catboost"),
        ]:
            if find_spec(package) is not None:
                models.append(name)
    return models
```

Add to `make_model_factory()` before the `raise ValueError`:
```python
if model_name == "svm":
    return lambda: _make_svm(seed, balanced)
```

Add factory function:
```python
def _make_svm(seed: int, balanced: bool) -> Pipeline:
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced" if balanced else None,
            random_state=seed,
        ),
    )
```

Add to `model_supports_sample_weight()`:
```python
def model_supports_sample_weight(model_name: str) -> bool:
    # SVC supports sample_weight via fit() — pass explicitly
    return model_name in ("hgb", "xgboost", "svm")
```

**Note on SVM + sample_weight:** SVC passes `sample_weight` directly to `fit()`. The CWMS dispatcher uses `_fit_kwargs()` which checks for pipeline steps. Add `svm` handling:
```python
def _fit_kwargs(model, sample_weight):
    last = model.steps[-1][0] if hasattr(model, "steps") else None
    return {f"{last}__sample_weight": sample_weight} if last else {"sample_weight": sample_weight}
```
This already handles pipeline step routing — no change needed, SVC is last step `"svc"`.

### Restore RF and ET from EXCLUDED_MODELS

In `scripts/run_cwms_msbs_deep_sweep.py`, change:
```python
EXCLUDED_MODELS = {"random_forest", "extra_trees"}
```
to:
```python
EXCLUDED_MODELS = set()  # all publication models included
```

RF/ET fully support `sample_weight` (RandomForestClassifier, ExtraTreesClassifier both accept it). They were excluded for performance reasons during early runs — include now for completeness.

**CWMS routing for RF/ET:** They use `class_weight="balanced"` (not `scale_pos_weight`) and support `sample_weight` → use the same LR/non-boosting path in the dispatcher (`confidence_weighted_sample_weights` + `minority_weight=1.0`).

Add to `model_supports_sample_weight()`:
```python
return model_name in ("hgb", "xgboost", "svm")
# RF/ET: use class_weight="balanced" in bal_factory, not explicit sample_weight
```

RF/ET do NOT need `use_sample_weight=True` for OOF scoring — they have native `class_weight`. Same as LR.

## Part C — k-NN Ratio Scorer

### Create `pipeline/scoring/knn_ratio.py`

```python
"""k-NN ratio scorer: fraction of minority neighbors among k nearest for majority samples."""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def knn_ratio_majority_scores(
    X: np.ndarray,
    y_noisy: np.ndarray,
    minority_label: int,
    majority_label: int,
    k: int = 5,
) -> np.ndarray:
    """Return minority-neighbor fraction for every MAJORITY-pool sample.

    score_i = (# of k nearest neighbors with y == minority_label) / k

    Higher score → surrounded by minority → suspicious (likely mislabeled minority).
    Same NaN contract as balanced_oof_majority_scores: NaN at minority positions.
    Deterministic — no seed needed.
    """
    n = len(y_noisy)
    scores = np.full(n, np.nan, dtype=float)
    maj_idx = np.where(y_noisy == majority_label)[0]
    if len(maj_idx) == 0:
        return scores

    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
    nn.fit(X)
    _, neighbor_indices = nn.kneighbors(X[maj_idx])
    neighbor_indices = neighbor_indices[:, 1:]  # exclude self

    minority_mask = y_noisy == minority_label
    for i, src_idx in enumerate(maj_idx):
        scores[src_idx] = float(np.sum(minority_mask[neighbor_indices[i]])) / k

    return scores
```

### Add dispatchers in `run_relabeling_viability_sweep.py`

Add import:
```python
from pipeline.scoring.knn_ratio import knn_ratio_majority_scores
```

In `run_single_viability()`, after `bal_scores` computation:
```python
knn_scores = (
    knn_ratio_majority_scores(X_tr, y_noisy, minority_label, majority_label, k=5)
    if any(m.startswith("cwms_msbs_knn") for m in methods_to_run)
    else None
)

hgb_scorer_factory = (
    make_model_factory("hgb", seed, cat_indices, balanced=True, use_gpu=use_gpu)
    if any(m.startswith("cwms_msbs_crossfamily") for m in methods_to_run)
    else None
)
crossfamily_scores = (
    balanced_oof_majority_scores(
        X_tr, y_noisy, hgb_scorer_factory,
        minority_label, majority_label, 5, seed, use_sample_weight=True,
    )
    if hgb_scorer_factory is not None else None
)
```

Pass to `_run_method()`. Add dispatchers for `cwms_msbs_knn` and `cwms_msbs_crossfamily` (same logic as `cwms_msbs` with different score arrays — details in phase-03).

## Part D — Plain SMOTE Baseline

Plain SMOTE (Chawla 2002) is the universal oversampling baseline in every competitor paper. Must be in Run A (all models) and Run C (competitor head-to-head). Zero noise handling — synthesizes directly from the noisy minority pool.

**Dispatcher entry in `run_relabeling_viability_sweep.py`:**

```python
if method == "smote":
    from imblearn.over_sampling import SMOTE as ImbSMOTE
    n_min = int(np.sum(y_noisy == min_label))
    n_maj = int(np.sum(y_noisy == maj_label))
    # Synthesize budget_count samples, cap at majority count so we don't overshoot balance
    target = min(n_min + budget, n_maj)
    if n_min < 2 or target <= n_min:
        return evaluate_augmented(X_tr, y_noisy, X_te, y_te, factory, min_label,
                                   n_synthetic=0, relabel_correctness=float("nan"))
    try:
        smote = ImbSMOTE(
            sampling_strategy={min_label: target},
            k_neighbors=min(5, n_min - 1),
            random_state=seed,
        )
        X_aug, y_aug = smote.fit_resample(X_tr, y_noisy)
        n_synth = len(y_aug) - len(y_noisy)
    except Exception:
        # e.g. too few minority samples after imbalance induction
        return evaluate_augmented(X_tr, y_noisy, X_te, y_te, factory, min_label,
                                   n_synthetic=0, relabel_correctness=float("nan"))
    return evaluate_augmented(X_aug, y_aug, X_te, y_te, factory, min_label,
                               n_synthetic=n_synth, relabel_correctness=float("nan"))
```

No new file needed — imblearn is already installed. No model exclusions: SMOTE is model-agnostic (returns X_aug, y_aug for any final classifier).

Add `"smote"` to `METHODS` list under a `# Oversampling baselines` section comment.

## Smoke Tests

```bash
# SVM CWMS compatibility
/home/than-minh/miniconda3/envs/dsp/bin/python -c "
from scripts.run_relabeling_viability_sweep import run_single_viability, NOISE_PROTOCOLS
mn, mj = NOISE_PROTOCOLS['hidden_minority_medium']
rows = run_single_viability('credit', 'svm', 42, 'hidden_minority_medium', mn, mj, 0.10, 0.15,
                             methods=['no_cleaning', 'class_proportional', 'cwms_msbs'])
import pandas as pd; df = pd.DataFrame(rows)
print(df[['method','balanced_accuracy','minority_recall']])
"

# k-NN ratio scorer
/home/than-minh/miniconda3/envs/dsp/bin/python -c "
from pipeline.scoring.knn_ratio import knn_ratio_majority_scores
import numpy as np
np.random.seed(0)
X = np.random.randn(200, 5); y = np.array([0]*160 + [1]*40)
s = knn_ratio_majority_scores(X, y, minority_label=1, majority_label=0, k=5)
print('NaN at minority:', np.all(np.isnan(s[y==1])))
print('Range:', np.nanmin(s), '-', np.nanmax(s))
"

# CUDA check
/home/than-minh/miniconda3/envs/dsp/bin/python scripts/run_cwms_msbs_deep_sweep.py --gpu --medium-only
# should print GPU check results then begin sweep
```

## Success Criteria

- [x] `pipeline/scoring/knn_ratio.py` created and smoke-tested
- [x] `svm` added to `list_publication_models()` and `make_model_factory()`
- [x] RF and ET removed from `EXCLUDED_MODELS`
- [x] `model_supports_sample_weight()` updated for SVM
- [x] CUDA `_check_gpu()` added to both sweep scripts; aborts with clear message if CUDA unavailable
- [x] `cwms_msbs_knn` and `cwms_msbs_crossfamily` dispatchers added
- [x] All existing sweep rows still reproducible (no regression in BA on smoke test)
