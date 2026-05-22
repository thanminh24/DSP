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
