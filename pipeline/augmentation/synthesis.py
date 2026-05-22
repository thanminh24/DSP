"""Confidence-Guided Minority Synthesis (CGMS).

Keep all original labels intact. Synthesize new minority samples from
high-confidence majority-pool seeds via cross-class interpolation with
real minority neighbors.
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def confidence_guided_synthesis(
    X: np.ndarray,
    y_noisy: np.ndarray,
    scores: np.ndarray,
    budget: int,
    minority_label: int,
    majority_label: int,
    threshold: float = 0.5,
    k_neighbors: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Synthesize new minority samples from high-confidence majority seeds.

    For each majority-pool sample with OOF P(minority|x) >= threshold,
    interpolate between it and a random real minority neighbor.

    Returns:
        X_aug, y_aug: original data + synthesized samples appended.
        n_synthesized: number of synthetic minority samples added.
        n_seeds_used: number of majority seeds that produced a sample.
    """
    rng = np.random.default_rng(seed)

    maj_mask = y_noisy == majority_label
    valid_mask = maj_mask & ~np.isnan(scores) & (scores >= threshold)
    seed_idx = np.where(valid_mask)[0]

    if len(seed_idx) == 0 or budget <= 0:
        return X.copy(), y_noisy.copy(), 0, 0

    if len(seed_idx) > budget:
        top_local = np.argsort(scores[seed_idx])[::-1][:budget]
        seed_idx = seed_idx[top_local]

    min_idx = np.where(y_noisy == minority_label)[0]
    if len(min_idx) == 0:
        return X.copy(), y_noisy.copy(), 0, 0

    k = min(k_neighbors, len(min_idx))
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto")
    nn.fit(X[min_idx])

    _, neighbor_locs = nn.kneighbors(X[seed_idx])
    new_X = []
    for i in range(len(seed_idx)):
        local_neighbor = rng.integers(0, k)
        m_idx = min_idx[neighbor_locs[i, local_neighbor]]
        u = rng.random()
        x_prime = X[seed_idx[i]] + u * (X[m_idx] - X[seed_idx[i]])
        new_X.append(x_prime)

    if not new_X:
        return X.copy(), y_noisy.copy(), 0, 0

    new_X = np.array(new_X)
    new_y = np.full(len(new_X), minority_label, dtype=y_noisy.dtype)
    X_aug = np.vstack([X, new_X])
    y_aug = np.concatenate([y_noisy, new_y])
    return X_aug, y_aug, len(new_X), len(seed_idx)
