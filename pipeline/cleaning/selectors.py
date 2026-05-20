"""Cleaning method selectors — each returns indices of samples to delete."""

from __future__ import annotations

import numpy as np


def select_none(n_samples: int) -> np.ndarray:
    """No cleaning — return empty selection."""
    return np.array([], dtype=int)


def select_random(
    n_samples: int, budget_count: int, rng: np.random.Generator
) -> np.ndarray:
    """Random deletion baseline."""
    return rng.choice(np.arange(n_samples), size=budget_count, replace=False)


def select_global(suspiciousness: np.ndarray, budget_count: int) -> np.ndarray:
    """Global top-loss deletion — no class constraints."""
    if budget_count <= 0:
        return np.array([], dtype=int)
    return np.argsort(suspiciousness)[::-1][:budget_count]


def select_class_proportional(
    suspiciousness: np.ndarray,
    y_noisy: np.ndarray,
    budget_count: int,
) -> np.ndarray:
    """Class-proportional top-loss — budget split by class frequency, caps only."""
    if budget_count <= 0:
        return np.array([], dtype=int)
    selected: list[int] = []
    for label in np.unique(y_noisy):
        label_idx = np.where(y_noisy == label)[0]
        class_budget = int(round(budget_count * len(label_idx) / len(y_noisy)))
        if class_budget == 0:
            class_budget = 1
        ranked = label_idx[np.argsort(suspiciousness[label_idx])[::-1]]
        selected.extend(ranked[:class_budget].tolist())
    return np.array(selected[:budget_count], dtype=int)


def select_oracle(noisy_mask: np.ndarray, budget_count: int) -> np.ndarray:
    """Oracle deletion — knows true noisy labels (upper bound)."""
    if budget_count <= 0:
        return np.array([], dtype=int)
    idx = np.where(noisy_mask)[0]
    return idx[:budget_count]


def select_crcc_p(
    suspiciousness: np.ndarray,
    y_noisy: np.ndarray,
    budget_count: int,
    lambda_risk: float,
    minority_label: int = 1,
) -> np.ndarray:
    """CRCC-P: proportional per-class caps with risk-adjusted scoring.

    Each class gets proportional deletion budget (based on frequency).
    Lambda controls how much minority risk penalizes deletion score.
    lambda=0 → risk penalty disabled; caps still enforced via greedy global pass
    (differs from class_proportional which fills each class independently).
    """
    if not 0.0 <= lambda_risk <= 1.0:
        raise ValueError(f"lambda_risk must be in [0, 1], got {lambda_risk}")
    if budget_count <= 0:
        return np.array([], dtype=int)

    class_caps = {
        label: max(1, int(round(budget_count * np.mean(y_noisy == label))))
        for label in np.unique(y_noisy)
    }
    class_risk = {label: 0.0 for label in np.unique(y_noisy)}
    class_risk[minority_label] = 1.0
    adjusted = np.array(
        [
            score - lambda_risk * class_risk[int(label)]
            for score, label in zip(suspiciousness, y_noisy)
        ]
    )
    deleted_count = {label: 0 for label in np.unique(y_noisy)}
    selected: list[int] = []
    for idx in np.argsort(adjusted)[::-1]:
        label = int(y_noisy[idx])
        if len(selected) >= budget_count:
            break
        if deleted_count[label] < class_caps[label]:
            selected.append(int(idx))
            deleted_count[label] += 1
    return np.array(selected, dtype=int)


def select_crcc_m(
    suspiciousness: np.ndarray,
    y_noisy: np.ndarray,
    budget_count: int,
    lambda_risk: float,
    minority_label: int = 1,
    minority_cap_factor: float = 0.5,
) -> np.ndarray:
    """CRCC-M: stricter minority cap with risk-adjusted scoring.

    Minority cap = floor(budget * minority_ratio * minority_cap_factor).
    Default minority_cap_factor=0.5 gives minority half its proportional share.
    """
    if not 0.0 <= lambda_risk <= 1.0:
        raise ValueError(f"lambda_risk must be in [0, 1], got {lambda_risk}")
    if budget_count <= 0:
        return np.array([], dtype=int)

    minority_ratio = np.mean(y_noisy == minority_label)
    minority_cap = max(1, int(np.floor(budget_count * minority_ratio * minority_cap_factor)))
    majority_cap = budget_count - minority_cap

    class_caps = {}
    for label in np.unique(y_noisy):
        if label == minority_label:
            class_caps[label] = minority_cap
        else:
            class_caps[label] = majority_cap

    class_risk = {label: 0.0 for label in np.unique(y_noisy)}
    class_risk[minority_label] = 1.0
    adjusted = np.array(
        [
            score - lambda_risk * class_risk[int(label)]
            for score, label in zip(suspiciousness, y_noisy)
        ]
    )
    deleted_count = {label: 0 for label in np.unique(y_noisy)}
    selected: list[int] = []
    for idx in np.argsort(adjusted)[::-1]:
        label = int(y_noisy[idx])
        if len(selected) >= budget_count:
            break
        if deleted_count[label] < class_caps[label]:
            selected.append(int(idx))
            deleted_count[label] += 1
    return np.array(selected, dtype=int)
