"""Evaluation wrapper for relabel/augment methods (training-set mutation paradigm)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score


def evaluate_augmented(
    X_train_aug: np.ndarray,
    y_train_aug: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_factory,
    minority_label: int,
    n_relabeled: int = 0,
    n_synthetic: int = 0,
    relabel_correctness: float | None = None,
) -> dict:
    """Train on augmented data; report metrics matching evaluate() schema.

    Schema mirrors pipeline.evaluation.metrics.evaluate output, with:
      - deleted = 0 (no deletion in augment paradigm)
      - noise_precision_deleted = NaN
      - clean_minority_deletion_rate = NaN
      - extra: n_relabeled, n_synthetic, relabel_correctness
    """
    if len(np.unique(y_train_aug)) < 2:
        return _nan_result(n_relabeled, n_synthetic, relabel_correctness)
    model = model_factory()
    model.fit(X_train_aug, y_train_aug)
    y_pred = model.predict(X_test)
    return {
        "deleted": 0,
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "minority_recall": recall_score(y_test, y_pred, pos_label=minority_label, zero_division=0),
        "noise_precision_deleted": float("nan"),
        "clean_minority_deletion_rate": float("nan"),
        "n_relabeled": int(n_relabeled),
        "n_synthetic": int(n_synthetic),
        "relabel_correctness": (
            float("nan") if relabel_correctness is None else float(relabel_correctness)
        ),
    }


def _nan_result(n_relabeled, n_synthetic, relabel_correctness):
    return {
        "deleted": 0,
        "balanced_accuracy": float("nan"),
        "macro_f1": float("nan"),
        "minority_recall": float("nan"),
        "noise_precision_deleted": float("nan"),
        "clean_minority_deletion_rate": float("nan"),
        "n_relabeled": int(n_relabeled),
        "n_synthetic": int(n_synthetic),
        "relabel_correctness": (
            float("nan") if relabel_correctness is None else float(relabel_correctness)
        ),
    }
