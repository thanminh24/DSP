"""Train-fitted feature encoding helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def encode_train_test(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    cat_cols: list[str],
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Encode categorical columns using categories fitted on training data only.

    Unseen test categories map to -1, which downstream tree models treat as a
    missing/unknown-like value and linear models receive as a stable numeric code.
    """
    train = X_train.copy()
    test = X_test.copy()
    cat_indices: list[int] = []
    for col in cat_cols:
        cat_indices.append(list(train.columns).index(col))
        categories = pd.Categorical(train[col]).categories
        train[col] = pd.Categorical(train[col], categories=categories).codes.astype(float)
        test[col] = pd.Categorical(test[col], categories=categories).codes.astype(float)
    return train.to_numpy(dtype=float), test.to_numpy(dtype=float), cat_indices


def encode_dataframe(
    X: pd.DataFrame,
    cat_cols: list[str],
) -> tuple[np.ndarray, list[int]]:
    """Encode one frame, fitting categories on that frame."""
    encoded, _, cat_indices = encode_train_test(X, X.iloc[:0].copy(), cat_cols)
    return encoded, cat_indices
