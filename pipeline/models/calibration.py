"""Probability diagnostics for relabeling confidence scores."""

from __future__ import annotations

import numpy as np


def score_decile_table(scores: np.ndarray, is_true_target: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Summarize candidate precision by descending score decile."""
    mask = ~(np.isnan(scores) | np.isnan(is_true_target.astype(float)))
    scores = scores[mask]
    labels = is_true_target[mask]
    if len(scores) == 0:
        return []
    order = np.argsort(scores)[::-1]
    chunks = np.array_split(order, min(n_bins, len(order)))
    rows = []
    for rank, idx in enumerate(chunks, 1):
        rows.append({
            "decile": rank,
            "n": int(len(idx)),
            "score_min": float(scores[idx].min()),
            "score_max": float(scores[idx].max()),
            "precision": float(labels[idx].mean()),
        })
    return rows


def expected_calibration_error(scores: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Compute a simple binary ECE for candidate scores."""
    mask = ~(np.isnan(scores) | np.isnan(labels.astype(float)))
    scores = scores[mask]
    labels = labels[mask]
    if len(scores) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        bin_mask = (scores >= lo) & (scores < hi if hi < 1.0 else scores <= hi)
        if not bin_mask.any():
            continue
        weight = float(bin_mask.mean())
        ece += weight * abs(float(scores[bin_mask].mean()) - float(labels[bin_mask].mean()))
    return float(ece)
