"""Minority-Side Boundary Synthesis (MSBS).

Seeds from confirmed minority samples, interpolates toward high-confidence majority
neighbors. Avoids CGMS's local label contradiction by keeping both seed and synthetic
as minority-labeled.
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def minority_side_boundary_synthesis(
    X: np.ndarray,
    y_noisy: np.ndarray,
    scores: np.ndarray,          # bal_scores: P(minority|x), NaN for minority-labeled
    budget_count: int,
    minority_label: int,
    majority_label: int,
    k_neighbors: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Synthesize minority samples by interpolating from confirmed minority toward boundary.

    For each confirmed minority sample, finds its k nearest high-confidence majority
    neighbors (by bal_score), picks one, interpolates:
        x' = x_min + u * (x_high_conf_maj - x_min),  u ~ U(0, 1)
    x' receives minority_label. The majority neighbor keeps its label — no contradiction.

    Returns:
        X_aug, y_aug: original data + synthetic samples appended.
        n_synthesized: number of synthetic minority samples added.
    """
    rng = np.random.default_rng(seed)

    min_idx = np.where(y_noisy == minority_label)[0]
    if len(min_idx) == 0 or budget_count <= 0:
        return X, y_noisy, 0

    maj_mask = (y_noisy == majority_label) & ~np.isnan(scores)
    maj_idx = np.where(maj_mask)[0]
    if len(maj_idx) == 0:
        return X, y_noisy, 0

    pool_size = min(len(maj_idx), max(budget_count * 3, k_neighbors + 1))
    top_maj = maj_idx[np.argsort(scores[maj_idx])[::-1][:pool_size]]

    k = min(k_neighbors, len(top_maj))
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto")
    nn.fit(X[top_maj])

    _, neighbor_locs = nn.kneighbors(X[min_idx])

    per_seed, remainder = divmod(budget_count, len(min_idx))
    counts = [per_seed + (1 if i < remainder else 0) for i in range(len(min_idx))]

    new_X = []
    for i, s_idx in enumerate(min_idx):
        for _ in range(counts[i]):
            local_neighbor = rng.integers(0, k)
            m_idx = top_maj[neighbor_locs[i, local_neighbor]]
            u = rng.random()
            x_prime = X[s_idx] + u * (X[m_idx] - X[s_idx])
            new_X.append(x_prime)

    if not new_X:
        return X, y_noisy, 0

    new_X = np.array(new_X)
    new_y = np.full(len(new_X), minority_label, dtype=y_noisy.dtype)
    return np.vstack([X, new_X]), np.concatenate([y_noisy, new_y]), len(new_X)
