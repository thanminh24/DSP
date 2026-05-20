"""Class-balanced out-of-fold scoring for Type A detection in the majority pool."""
from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight


def balanced_oof_majority_scores(
    X: np.ndarray,
    y_noisy: np.ndarray,
    model_factory: Callable,
    minority_label: int,
    majority_label: int,
    n_splits: int = 5,
    seed: int = 42,
    use_sample_weight: bool = False,
) -> np.ndarray:
    """Return P(minority | x) for every MAJORITY-pool sample via balanced OOF.

    Strategy: train a class-balanced model out-of-fold on the full (X, y_noisy);
    extract minority-class probability for samples whose y_noisy == majority_label.
    Returns array of shape (n,), with NaN at positions where y_noisy == minority_label.

    Args:
        use_sample_weight: True for HGB (no class_weight param); pipes balanced
            sample_weight via fit. False for LR which uses class_weight="balanced"
            internally (caller responsibility to configure factory).
    """
    n = len(y_noisy)
    probs = np.full(n, np.nan, dtype=float)
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr_idx, va_idx in folds.split(X, y_noisy):
        model = model_factory()
        if use_sample_weight:
            sw = compute_sample_weight("balanced", y_noisy[tr_idx])
            last = model.steps[-1][0] if hasattr(model, "steps") else None
            fit_kwargs = {f"{last}__sample_weight": sw} if last else {"sample_weight": sw}
            model.fit(X[tr_idx], y_noisy[tr_idx], **fit_kwargs)
        else:
            model.fit(X[tr_idx], y_noisy[tr_idx])
        classes = list(model.classes_)
        min_col = classes.index(minority_label)
        p = model.predict_proba(X[va_idx])[:, min_col]
        probs[va_idx] = p
    minority_mask = y_noisy == minority_label
    probs[minority_mask] = np.nan
    return probs
