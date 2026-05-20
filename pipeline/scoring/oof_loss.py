"""Out-of-fold cross-entropy suspiciousness scoring."""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.model_selection import StratifiedKFold


def out_of_fold_loss(
    X: np.ndarray,
    y_noisy: np.ndarray,
    model_factory: Callable,
    n_splits: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Compute per-sample cross-entropy loss via stratified k-fold CV.

    Args:
        X: Feature matrix (n_samples × n_features).
        y_noisy: Noisy labels.
        model_factory: Callable returning a fresh sklearn-compatible model.
        n_splits: Number of CV folds.
        seed: Random seed for fold assignment.

    Returns:
        Array of per-sample cross-entropy loss (higher = more suspicious).
    """
    n = len(y_noisy)
    n_classes = len(np.unique(y_noisy))
    probabilities = np.zeros((n, n_classes), dtype=float)
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, valid_idx in folds.split(X, y_noisy):
        model = model_factory()
        model.fit(X[train_idx], y_noisy[train_idx])
        probabilities[valid_idx] = model.predict_proba(X[valid_idx])
    clipped = np.clip(probabilities[np.arange(n), y_noisy], 1e-12, 1.0)
    return -np.log(clipped)
