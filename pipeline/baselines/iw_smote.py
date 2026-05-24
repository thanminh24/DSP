"""IW-SMOTE: Instance Weighted SMOTE by Indirectly Exploring the Data Distribution.

Original authors: Aimin Zhang, Hualong Yu, Shanlin Zhou, Zhangjun Huan, Xibei Yang
Original source: https://github.com/... (Jiangsu University of Science and Technology)
Paper: Pattern Recognition 2022

Adaptations from original Algorithms.py:
- Replaced deprecated DataFrame.append() with pd.concat() (pandas 2.x compatibility)
- Removed unused smote_variants import (not in our env)
- Remapped arbitrary (minority_label, majority_label) → internal (1, -1) convention
- Added seed parameter for reproducibility (replaced random.choice with seeded numpy)
- Exposed standard pipeline interface: (X, y, minority_label, majority_label, budget_count, seed)
  → (X_aug, y_aug, n_synthetic)
- budget_count translates to gen_times = budget_count / max(1, n_reserve_maj)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


def _cart(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
          seed: int) -> np.ndarray:
    model = DecisionTreeClassifier(random_state=seed)
    model.fit(X_train, y_train)
    return model.predict(X_test)


def iw_smote(
    X: np.ndarray,
    y_noisy: np.ndarray,
    minority_label: int,
    majority_label: int,
    budget_count: int,
    lamda: int = 100,
    thres: float = 0.5,
    divide_times: int = 2,
    k_neighbor: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, int]:
    """IW-SMOTE oversampling with under-bagging noise filtering.

    Args:
        X: feature matrix (n_samples, n_features)
        y_noisy: noisy labels using caller's convention (minority_label / majority_label)
        minority_label: integer value that denotes minority class
        majority_label: integer value that denotes majority class
        budget_count: approximate number of synthetic minority samples to generate
        lamda: ensemble size multiplier (IR * lamda = number of CART classifiers)
        thres: noise filtering threshold — samples with error_rate >= thres are removed
        divide_times: under-bagging ratio denominator
        k_neighbor: k for k-NN in synthesis step
        seed: random seed

    Returns:
        (X_aug, y_aug, n_synthetic) — augmented dataset and number of new samples added.
        Labels use caller's original convention (minority_label / majority_label).
    """
    rng = np.random.default_rng(seed)

    # Remap labels to internal convention: minority=1, majority=-1
    y_internal = np.where(y_noisy == minority_label, 1, -1)

    data = pd.DataFrame(X)
    data["__label__"] = y_internal

    z = data[data["__label__"] == 1].reset_index(drop=True)   # minority
    p = data[data["__label__"] == -1].reset_index(drop=True)  # majority
    m1 = len(z)
    m2 = len(p)
    n_feat = X.shape[1]

    if m1 < 2 or m2 < 1:
        return X, y_noisy, 0

    IR = m2 / m1
    n_carts = max(1, int(IR * lamda))

    z_arr = z.iloc[:, :n_feat].values
    p_arr = p.iloc[:, :n_feat].values
    z_labels = z["__label__"].values
    p_labels = p["__label__"].values

    # Under-bagging CART ensemble: predict minority and majority samples
    predict_min = np.empty((m1, n_carts), dtype=int)
    predict_maj = np.empty((m2, n_carts), dtype=int)

    for i in range(n_carts):
        cart_seed = int(rng.integers(0, 10_000))
        n_sub = max(1, int(m1 / divide_times))
        min_idx = rng.choice(m1, size=n_sub, replace=False)
        maj_idx = rng.choice(m2, size=n_sub, replace=True)
        X_sub = np.vstack([z_arr[min_idx], p_arr[maj_idx]])
        y_sub = np.concatenate([z_labels[min_idx], p_labels[maj_idx]])
        predict_min[:, i] = _cart(X_sub, y_sub, z_arr, cart_seed)
        predict_maj[:, i] = _cart(X_sub, y_sub, p_arr, cart_seed)

    # Error rate per sample = fraction of ensemble that disagrees with noisy label
    err_min = (predict_min != z_labels[:, None]).mean(axis=1)
    err_maj = (predict_maj != p_labels[:, None]).mean(axis=1)

    # Filter: keep samples with error_rate < thres (likely clean)
    keep_min_mask = err_min < thres
    keep_maj_mask = err_maj < thres
    reserve_min = z_arr[keep_min_mask]
    reserve_maj = p_arr[keep_maj_mask]
    err_min_kept = err_min[keep_min_mask]
    n_reserve_min = len(reserve_min)
    n_reserve_maj = len(reserve_maj)

    if n_reserve_min < 2:
        # Not enough clean minority after filtering — return filtered only, no synthesis
        X_clean = np.vstack([reserve_maj, reserve_min]) if n_reserve_min > 0 else reserve_maj
        y_clean = np.concatenate([
            np.full(n_reserve_maj, majority_label),
            np.full(n_reserve_min, minority_label),
        ])
        return X_clean, y_clean, 0

    # Synthesis weights: higher error rate → higher weight (more synthetic neighbors)
    # Clip to avoid zero-weight (original clips at 1/n_carts)
    min_err_clip = np.maximum(err_min_kept, 1.0 / n_carts)
    weight = min_err_clip / min_err_clip.sum()

    # Translate budget_count into num_need_generate
    # Original uses: gen_times * n_reserve_maj - n_reserve_min
    # We override: use budget_count directly, clipped so we don't overshoot balance
    num_to_generate = min(budget_count, max(0, n_reserve_maj - n_reserve_min))
    if num_to_generate <= 0:
        X_aug = np.vstack([reserve_maj, reserve_min])
        y_aug = np.concatenate([
            np.full(n_reserve_maj, majority_label),
            np.full(n_reserve_min, minority_label),
        ])
        return X_aug, y_aug, 0

    # Weighted synthesis: each minority sample generates proportional to its weight
    per_sample_count = (weight * num_to_generate).astype(int)
    # Distribute remainder to highest-weight samples
    remainder = num_to_generate - per_sample_count.sum()
    if remainder > 0:
        top_idx = np.argsort(weight)[::-1][:remainder]
        per_sample_count[top_idx] += 1

    # k-NN distance-based synthesis (original algorithm)
    synthetic_X = []
    for i in range(n_reserve_min):
        n_gen = per_sample_count[i]
        if n_gen == 0:
            continue
        dists = np.linalg.norm(reserve_min - reserve_min[i], axis=1)
        dists[i] = np.inf  # exclude self
        nn_idx = np.argsort(dists)[:k_neighbor]
        for _ in range(n_gen):
            neighbor = reserve_min[rng.choice(nn_idx)]
            alpha = rng.random()
            x_new = reserve_min[i] + alpha * (neighbor - reserve_min[i])
            synthetic_X.append(x_new)

    n_synthetic = len(synthetic_X)
    if n_synthetic == 0:
        X_aug = np.vstack([reserve_maj, reserve_min])
        y_aug = np.concatenate([
            np.full(n_reserve_maj, majority_label),
            np.full(n_reserve_min, minority_label),
        ])
        return X_aug, y_aug, 0

    X_synth = np.array(synthetic_X)
    X_aug = np.vstack([reserve_maj, reserve_min, X_synth])
    y_aug = np.concatenate([
        np.full(n_reserve_maj, majority_label),
        np.full(n_reserve_min, minority_label),
        np.full(n_synthetic, minority_label),
    ])
    return X_aug, y_aug, n_synthetic
