# cleanlab baseline: comparison method, not paper contribution
"""Optional cleanlab baselines."""

from __future__ import annotations

import numpy as np


def select_cleanlab_filter(
    y_noisy: np.ndarray,
    pred_probs: np.ndarray,
    budget_count: int,
) -> np.ndarray:
    """Select likely label issues with cleanlab if installed."""
    if budget_count <= 0:
        return np.array([], dtype=int)
    try:
        from cleanlab.filter import find_label_issues
    except ImportError as exc:
        raise ImportError("Install optional dependency: pip install cleanlab") from exc

    ranked = find_label_issues(
        labels=y_noisy,
        pred_probs=pred_probs,
        return_indices_ranked_by="self_confidence",
    )
    return np.asarray(ranked[:budget_count], dtype=int)
