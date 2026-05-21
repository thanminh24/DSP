"""Class-weight-only evaluation helpers."""

from __future__ import annotations

import numpy as np

from pipeline.evaluation.augment_metrics import evaluate_augmented


def evaluate_class_weight_only(
    X_train: np.ndarray,
    y_noisy: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_factory,
    minority_label: int,
) -> dict:
    """Train a balanced model without relabeling or deletion."""
    row = evaluate_augmented(X_train, y_noisy, X_test, y_test, model_factory, minority_label)
    row["n_relabeled"] = 0
    row["n_synthetic"] = 0
    row["relabel_correctness"] = float("nan")
    return row
