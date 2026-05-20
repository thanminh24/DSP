"""Evaluation metrics for cleaning methods."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    recall_score,
)


def evaluate(
    selected_idx: np.ndarray,
    X_train: np.ndarray,
    y_noisy: np.ndarray,
    y_clean: np.ndarray,
    noisy_mask: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_factory,
    minority_label: int = 1,
) -> dict:
    """Retrain on cleaned data and evaluate on clean test set.

    Returns dict with: deleted, balanced_accuracy, macro_f1, minority_recall,
    noise_precision_deleted, clean_minority_deletion_rate.
    """
    n = len(y_noisy)
    keep_mask = np.ones(n, dtype=bool)
    keep_mask[selected_idx] = False

    model = model_factory()
    model.fit(X_train[keep_mask], y_noisy[keep_mask])
    y_pred = model.predict(X_test)

    n_deleted = int(len(selected_idx))
    if n_deleted > 0:
        noise_precision = float(noisy_mask[selected_idx].mean())
        clean_minority_deleted = (y_clean[selected_idx] == minority_label) & ~noisy_mask[selected_idx]
        cmdr = float(clean_minority_deleted.mean())
    else:
        noise_precision = 0.0
        cmdr = 0.0

    return {
        "deleted": n_deleted,
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "minority_recall": recall_score(y_test, y_pred, pos_label=minority_label, zero_division=0),
        "noise_precision_deleted": noise_precision,
        "clean_minority_deletion_rate": cmdr,
    }
