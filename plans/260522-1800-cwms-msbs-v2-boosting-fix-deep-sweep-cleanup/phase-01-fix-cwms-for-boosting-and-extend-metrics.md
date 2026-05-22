---
phase: 1
title: "Fix CWMS for Boosting and Extend Metrics"
status: pending
priority: P1
effort: "~1h"
dependencies: []
---

# Phase 1: Fix CWMS for Boosting and Extend Metrics

## Overview

Two code changes: (1) fix the double-correction bug where XGBoost's `scale_pos_weight` and
CWMS `sample_weight` stack on each other, and (2) add missing metrics to
`evaluate_augmented()` so the deep sweep can compare across all relevant dimensions.

## Root Cause: Double Correction in Boosting

In `run_relabeling_viability_sweep.py`, `std_factory` for XGBoost is built with
`scale_pos_weight=spw` (class ratio). When `cwms_msbs` then passes
`sample_weight = max(1 - bal_score, 0)` for majority, the model receives:

- minority gradient ∝ 1.0 × spw (sample_weight=1, spw applied globally)
- suspicious majority gradient ≈ 0 (sample_weight≈0)
- confident majority gradient ≈ 1.0 × 1.0 (sample_weight≈1, spw not applied to majority)

The result: minority is over-upscaled relative to confident majority — too aggressive a
correction. Fix: encode the class balance into the CWMS weights directly and set
`scale_pos_weight=1.0` in the factory used for cwms/cwms_msbs.

**New CWMS weight formula (class-balanced variant):**
```
weights[minority] = spw          # class balance signal
weights[majority] = max(1 - bal_score, 0)   # noise suppression (unchanged)
```

This way, when `scale_pos_weight=1.0`:
- minority effective gradient ∝ spw × 1 (from sample_weight)
- confident majority effective gradient ≈ 1 × 1
- suspicious majority effective gradient ≈ 0
Net: exactly the intended correction.

**calibrated_lr exclusion**: sklearn's `CalibratedClassifierCV` does not propagate
`sample_weight` to the base estimator (only uses it at calibration step). CWMS weights
distort calibration curves without fixing the decision boundary. Exclude `calibrated_lr`
from cwms and cwms_msbs dispatchers — run only `no_cleaning` and `class_proportional`
for this model. Document as known limitation.

## Related Code Files

- Modify: `pipeline/baselines/soft_weighting.py` — add `confidence_weighted_sample_weights_balanced()`
- Modify: `pipeline/evaluation/augment_metrics.py` — add accuracy, weighted_f1, minority_precision, majority_recall
- Modify: `scripts/run_relabeling_viability_sweep.py` — update cwms/cwms_msbs dispatchers + calibrated_lr skip
- Modify: `pipeline/models/factories.py` — add `make_cwms_factory()` helper (spw=1.0 variant)

## Implementation Steps

### Step 1 — Add `confidence_weighted_sample_weights_balanced()` to soft_weighting.py

```python
def confidence_weighted_sample_weights_balanced(
    y_noisy: np.ndarray,
    scores: np.ndarray,
    majority_label: int,
    scale_pos_weight: float = 1.0,
) -> np.ndarray:
    """CWMS weights with class balance folded in.

    minority weight = scale_pos_weight (replaces model-level spw)
    majority weight = max(1 - score, 0)  (suppress suspicious)

    Use this with scale_pos_weight=1.0 in the model factory so the two
    corrections don't compound.
    """
    n = len(y_noisy)
    maj_mask = y_noisy == majority_label
    min_mask = ~maj_mask
    weights = np.empty(n, dtype=float)
    weights[min_mask] = float(scale_pos_weight)
    valid = maj_mask & ~np.isnan(scores)
    weights[maj_mask] = 1.0  # default for majority with missing scores
    weights[valid] = np.maximum(1.0 - scores[valid], 0.0)
    return weights
```

Keep the original `confidence_weighted_sample_weights()` for backward compatibility
(used by lr which has no spw conflict).

### Step 2 — Add `make_cwms_factory()` to factories.py

```python
def make_cwms_factory(
    model_name: str,
    seed: int,
    cat_indices: list[int] | None = None,
    use_gpu: bool = False,
) -> ModelFactory:
    """Factory with scale_pos_weight=1.0, for use when CWMS weights carry the class balance signal.

    For LR: identical to std_factory (LR has no spw).
    For boosting (xgb, lgbm, catboost, hgb): disables built-in class correction
    so CWMS balanced weights are the sole correction mechanism.
    """
    return make_model_factory(
        model_name, seed, cat_indices,
        balanced=False, scale_pos_weight=1.0, use_gpu=use_gpu,
    )
```

### Step 3 — Update cwms/cwms_msbs dispatchers in run_relabeling_viability_sweep.py

In `run_single_viability()`, add:

```python
cwms_factory = make_cwms_factory(model_name, seed, cat_indices, use_gpu=use_gpu)
```

Then in `_run_method()` signature, add `cwms_factory=None` parameter and update callers.

Update cwms dispatcher:

```python
if method == "cwms":
    if model_name == "calibrated_lr":
        # sklearn routing bug: sample_weight doesn't reach base LR — skip cwms for this model
        return {"skipped": True, "balanced_accuracy": float("nan"), ...}
    use_balanced = model_name in ("xgboost", "lightgbm", "catboost", "hgb")
    if use_balanced:
        sw = confidence_weighted_sample_weights_balanced(y_noisy, bal_scores, maj_label, scale_pos_weight=spw)
        return evaluate_augmented(X_tr, y_noisy, X_te, y_te, cwms_factory, min_label, sample_weight=sw)
    else:  # lr
        sw = confidence_weighted_sample_weights(y_noisy, bal_scores, maj_label)
        return evaluate_augmented(X_tr, y_noisy, X_te, y_te, factory, min_label, sample_weight=sw)
```

Update cwms_msbs dispatcher similarly:

```python
if method == "cwms_msbs":
    if model_name == "calibrated_lr":
        return {"skipped": True, "balanced_accuracy": float("nan"), ...}
    X_aug, y_aug, n_synth = minority_side_boundary_synthesis(
        X_tr, y_noisy, bal_scores, budget, min_label, maj_label, seed=seed,
    )
    use_balanced = model_name in ("xgboost", "lightgbm", "catboost", "hgb")
    if use_balanced:
        sw_orig = confidence_weighted_sample_weights_balanced(y_noisy, bal_scores, maj_label, spw)
        sw_synth = np.full(n_synth, float(spw))  # synthetics get minority weight
    else:
        sw_orig = confidence_weighted_sample_weights(y_noisy, bal_scores, maj_label)
        sw_synth = np.ones(n_synth, dtype=float)
    sw_combined = np.concatenate([sw_orig, sw_synth])
    fact = cwms_factory if use_balanced else factory
    return evaluate_augmented(X_aug, y_aug, X_te, y_te, fact,
                               min_label, n_synthetic=n_synth,
                               relabel_correctness=float("nan"),
                               sample_weight=sw_combined)
```

### Step 4 — Extend evaluate_augmented() metrics

In `pipeline/evaluation/augment_metrics.py`, import and add:

```python
from sklearn.metrics import accuracy_score, precision_score

# Inside evaluate_augmented(), after y_pred:
return {
    "deleted": 0,
    "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
    "accuracy": accuracy_score(y_test, y_pred),
    "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
    "weighted_f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    "minority_recall": recall_score(y_test, y_pred, pos_label=minority_label, zero_division=0),
    "minority_precision": precision_score(y_test, y_pred, pos_label=minority_label, zero_division=0),
    "majority_recall": recall_score(y_test, y_pred, pos_label=1 - minority_label, zero_division=0),
    "noise_precision_deleted": float("nan"),
    "clean_minority_deletion_rate": float("nan"),
    "n_relabeled": int(n_relabeled),
    "n_synthetic": int(n_synthetic),
    "relabel_correctness": ...,
}
```

Update `_nan_result()` to match (all new metrics → `float("nan")`).

Also mirror these additions in `pipeline/evaluation/metrics.py` `evaluate()` function
so deletion-based baselines (`class_proportional`, `no_cleaning`) also return the
full metric set. Add `accuracy_score` and `precision_score` imports.

### Step 5 — Quick sanity check (do not skip)

Run a minimal smoke test before the full sweep to verify the fix works:

```bash
/home/than-minh/miniconda3/envs/dsp/bin/python -c "
from scripts.run_relabeling_viability_sweep import run_single_viability
rows = run_single_viability('pima', 'xgboost', 13, 'hidden_minority_medium',
                             0.30, 0.10, 0.10, 0.15, methods=['cwms_msbs', 'class_proportional'])
for r in rows:
    print(r['method'], r['balanced_accuracy'], r.get('accuracy'), r.get('minority_precision'))
"
```

Expected: cwms_msbs BA for xgboost should be positive vs class_proportional (was −1.93pp with old code).

## Success Criteria

- [ ] `confidence_weighted_sample_weights_balanced()` added to soft_weighting.py
- [ ] `make_cwms_factory()` helper added to factories.py
- [ ] cwms dispatcher uses class-balanced weights for hgb/xgboost/lightgbm/catboost
- [ ] cwms_msbs dispatcher uses class-balanced weights + spw for synthetic samples on boosting models
- [ ] calibrated_lr returns NaN row for cwms and cwms_msbs (not an error, a skip)
- [ ] `evaluate_augmented()` returns accuracy, weighted_f1, minority_precision, majority_recall
- [ ] Smoke test passes: xgboost cwms_msbs shows non-negative delta vs class_proportional

## Risk Assessment

- **spw value choice for synthetics**: synthetic MSBS samples are interpolations between
  real minority and nearby majority. Giving them `weight=spw` assumes they behave like
  minority. If synthesis quality is poor (e.g., very few minority seeds), this may
  over-weight low-quality synthetics. Mitigation: watch `n_synthetic` column; if low,
  synthetics have less influence anyway.
- **HGB spw=1.0**: HGB has no explicit spw; it was getting class correction purely through
  `make_cwms_factory()` setting spw=1.0, which is the same as its std_factory. So HGB
  behavior doesn't change — the new balanced weights (minority weight = spw) are the
  only change. This should help HGB's neutral result become positive.
- **calibrated_lr skipped**: the sweep CSV will have NaN BA for this model on cwms/cwms_msbs
  methods. Downstream analysis must handle NaN rows (already done via `dropna()` in pivot).
