"""Confidence-based relabeling baselines."""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight


def unbalanced_oof_majority_scores(
    X: np.ndarray,
    y_noisy: np.ndarray,
    model_factory: Callable,
    minority_label: int,
    majority_label: int,
    n_splits: int = 5,
    seed: int = 42,
    use_sample_weight: bool = False,
) -> np.ndarray:
    """OOF P(minority|x) for majority-labeled samples without class balancing."""
    n = len(y_noisy)
    probs = np.full(n, np.nan, dtype=float)
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr_idx, va_idx in folds.split(X, y_noisy):
        model = model_factory()
        if use_sample_weight:
            sw = np.ones(len(tr_idx), dtype=float)
            fit_kwargs = _fit_kwargs(model, sw)
            model.fit(X[tr_idx], y_noisy[tr_idx], **fit_kwargs)
        else:
            model.fit(X[tr_idx], y_noisy[tr_idx])
        classes = list(model.classes_)
        min_col = classes.index(minority_label)
        probs[va_idx] = model.predict_proba(X[va_idx])[:, min_col]
    probs[y_noisy == minority_label] = np.nan
    return probs


def weighted_oof_majority_scores(
    X: np.ndarray,
    y_noisy: np.ndarray,
    model_factory: Callable,
    minority_label: int,
    majority_label: int,
    n_splits: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """OOF P(minority|x) using explicit balanced sample weights."""
    n = len(y_noisy)
    probs = np.full(n, np.nan, dtype=float)
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr_idx, va_idx in folds.split(X, y_noisy):
        model = model_factory()
        sw = compute_sample_weight("balanced", y_noisy[tr_idx])
        model.fit(X[tr_idx], y_noisy[tr_idx], **_fit_kwargs(model, sw))
        classes = list(model.classes_)
        min_col = classes.index(minority_label)
        probs[va_idx] = model.predict_proba(X[va_idx])[:, min_col]
    probs[y_noisy == minority_label] = np.nan
    return probs


def naive_confidence_majority_scores(
    X: np.ndarray,
    y_noisy: np.ndarray,
    model_factory: Callable,
    minority_label: int,
    majority_label: int,
    seed: int = 42,
    use_sample_weight: bool = False,
) -> np.ndarray:
    """P(minority|x) for majority-labeled samples trained on the FULL dataset (no OOF).

    Confirmation-bias baseline: the scoring model sees every sample it scores during
    training. Comparing this to balanced_oof_relabel directly measures how much OOF
    prevents the 'Two Wrongs Don't Make a Right' failure mode.
    """
    model = model_factory()
    if use_sample_weight:
        sw = compute_sample_weight("balanced", y_noisy)
        model.fit(X, y_noisy, **_fit_kwargs(model, sw))
    else:
        model.fit(X, y_noisy)
    classes = list(model.classes_)
    min_col = classes.index(minority_label)
    probs = model.predict_proba(X)[:, min_col]
    probs[y_noisy == minority_label] = np.nan
    return probs


def select_confidence_relabels(
    y_noisy: np.ndarray,
    scores: np.ndarray,
    budget_count: int,
    majority_label: int,
) -> np.ndarray:
    """Return majority-pool indices with highest minority confidence."""
    maj_idx = np.where(y_noisy == majority_label)[0]
    valid = maj_idx[~np.isnan(scores[maj_idx])]
    if budget_count <= 0 or len(valid) == 0:
        return np.array([], dtype=int)
    ranked = valid[np.argsort(scores[valid])[::-1]]
    return ranked[: min(budget_count, len(ranked))]


def _fit_kwargs(model, sample_weight: np.ndarray) -> dict:
    last = model.steps[-1][0] if hasattr(model, "steps") else None
    return {f"{last}__sample_weight": sample_weight} if last else {"sample_weight": sample_weight}
