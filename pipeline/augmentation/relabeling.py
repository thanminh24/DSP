"""Type A relabeling: flip top-k majority samples to minority based on scores."""
from __future__ import annotations

import numpy as np


def relabel_typeA(
    y_noisy: np.ndarray,
    majority_scores: np.ndarray,
    k: int,
    minority_label: int,
    majority_label: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Relabel top-k majority samples (by P(minority|x)) to minority_label.

    Returns:
        y_relabel: copy of y_noisy with k entries flipped.
        relabeled_idx: indices that were flipped (shape (k,)).
    """
    y_relabel = y_noisy.copy()
    maj_idx = np.where(y_noisy == majority_label)[0]
    scores_in_maj = majority_scores[maj_idx]
    valid = ~np.isnan(scores_in_maj)
    maj_idx = maj_idx[valid]
    scores_in_maj = scores_in_maj[valid]
    if len(maj_idx) == 0 or k <= 0:
        return y_relabel, np.array([], dtype=int)
    k = min(k, len(maj_idx))
    top_local = np.argsort(scores_in_maj)[::-1][:k]
    relabeled_idx = maj_idx[top_local]
    y_relabel[relabeled_idx] = minority_label
    return y_relabel, relabeled_idx


def random_relabeling(
    y_noisy: np.ndarray,
    k: int,
    majority_label: int,
    minority_label: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Control: relabel k UNIFORM-RANDOM majority samples to minority.

    If the OOF signal is useless, Method A should match this baseline.
    """
    y_relabel = y_noisy.copy()
    maj_idx = np.where(y_noisy == majority_label)[0]
    if len(maj_idx) == 0 or k <= 0:
        return y_relabel, np.array([], dtype=int)
    k = min(k, len(maj_idx))
    chosen = rng.choice(maj_idx, size=k, replace=False)
    y_relabel[chosen] = minority_label
    return y_relabel, chosen
