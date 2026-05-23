"""Confidence-Weighted Majority Suppression (CWMS).

No relabeling, no synthetic samples. Downweights majority-labeled samples proportional
to their P(minority|x) score. Ambiguous majority samples lose influence on the majority
class boundary, causing it to shift outward without label corruption.
"""
from __future__ import annotations

import numpy as np


def confidence_weighted_sample_weights(
    y_noisy: np.ndarray,
    scores: np.ndarray,       # bal_scores: P(minority|x), NaN for minority-labeled
    majority_label: int,
) -> np.ndarray:
    """Return sample_weight array for CWMS training.

    Majority-labeled sample i: weight = max(1 - scores[i], 0.0)
    Minority-labeled sample i: weight = 1.0
    NaN scores (minority pool): treated as weight = 1.0

    Does NOT normalize — absolute weights preserve gradient scale relative
    to minority samples which all have weight=1.0.
    """
    n = len(y_noisy)
    weights = np.ones(n, dtype=float)
    maj_mask = y_noisy == majority_label
    valid = maj_mask & ~np.isnan(scores)
    weights[valid] = np.maximum(1.0 - scores[valid], 0.0)
    return weights


def confidence_weighted_sample_weights_balanced(
    y_noisy: np.ndarray,
    scores: np.ndarray,
    majority_label: int,
    scale_pos_weight: float = 1.0,
) -> np.ndarray:
    """CWMS weights with class balance folded in, for boosting models.

    minority weight = scale_pos_weight (replaces model-level spw)
    majority weight = max(1 - score, 0)  (suppress suspicious)

    Use with scale_pos_weight=1.0 in the model factory so the two
    corrections don't compound.
    """
    n = len(y_noisy)
    maj_mask = y_noisy == majority_label
    min_mask = ~maj_mask
    weights = np.empty(n, dtype=float)
    weights[min_mask] = float(scale_pos_weight)
    weights[maj_mask] = 1.0  # default for majority with missing scores
    valid = maj_mask & ~np.isnan(scores)
    weights[valid] = np.maximum(1.0 - scores[valid], 0.0)
    return weights
