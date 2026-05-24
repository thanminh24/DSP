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
