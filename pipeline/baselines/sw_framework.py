"""SW Framework approximation via k-NN label inconsistency.

Original (KBS 2022): RF hypergraph partition → hyperedge chaos → weighted SMOTE.
No public code. Our approximation: RF leaf co-occurrence → label inconsistency score
→ weighted SMOTE where suspicious minority samples get lower synthesis weight.

Labeled "SW-approx" in paper.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors


def sw_framework_oversample(
    X: np.ndarray,
    y_noisy: np.ndarray,
    minority_label: int,
    majority_label: int,
    budget_count: int,
    seed: int = 42,
    k: int = 5,
) -> tuple[np.ndarray, np.ndarray, int]:
    """SW-style chaos-weighted SMOTE oversampling.

    1. Train RF to get leaf assignments (proxy for hypergraph partition)
    2. For each minority sample, measure label inconsistency among k-NN
       majority neighbors in RF proximity space → chaos score
    3. Low-chaos minority samples get higher synthesis weight (cleaner)
    4. SMOTE synthesis weighted by (1 - chaos)

    Returns (X_aug, y_aug, n_synthetic).
    """
    rng = np.random.default_rng(seed)
    n_min = int(np.sum(y_noisy == minority_label))
    n_maj = int(np.sum(y_noisy == majority_label))

    if n_min < 2:
        return X, y_noisy, 0

    # Step 1: RF leaf assignments as proximity proxy
    rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=4)
    rf.fit(X, y_noisy)
    leaf_indices = rf.apply(X)  # (n_samples, n_estimators)

    # Step 2: For each minority sample, check k-NN majority neighbors in leaf space
    # Higher fraction of opposite-label neighbors → higher risk of being noisy
    min_idx = np.where(y_noisy == minority_label)[0]
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(X)), algorithm="auto")
    nn.fit(leaf_indices)
    _, neighbor_idx = nn.kneighbors(leaf_indices[min_idx])

    chaos = np.zeros(n_min, dtype=float)
    for i, src in enumerate(min_idx):
        neighbors = neighbor_idx[i, 1:]  # exclude self
        chaos[i] = np.mean(y_noisy[neighbors] != minority_label)

    # Step 3: Weight = 1 - chaos (cleaner samples get higher weight)
    # Clip so all samples get some chance
    weights = np.maximum(1.0 - chaos, 0.05)
    if weights.sum() == 0:
        return X, y_noisy, 0
    weights = weights / weights.sum()

    # Step 4: Weighted synthesis budget allocation
    target = min(n_min + budget_count, n_maj)
    num_to_generate = max(0, target - n_min)
    if num_to_generate <= 0:
        return X, y_noisy, 0

    per_sample = (weights * num_to_generate).astype(int)
    remainder = num_to_generate - per_sample.sum()
    if remainder > 0:
        top_idx = np.argsort(weights)[::-1][:remainder]
        per_sample[top_idx] += 1

    # Step 5: SMOTE synthesis
    X_min = X[min_idx]
    synthetic = []
    for i in range(n_min):
        n_gen = per_sample[i]
        if n_gen == 0:
            continue
        dists = np.linalg.norm(X_min - X_min[i], axis=1)
        dists[i] = np.inf
        nn_synth = np.argsort(dists)[:min(k, n_min - 1)]
        for _ in range(n_gen):
            neighbor = X_min[rng.choice(nn_synth)]
            alpha = rng.random()
            x_new = X_min[i] + alpha * (neighbor - X_min[i])
            synthetic.append(x_new)

    n_synthetic = len(synthetic)
    if n_synthetic == 0:
        return X, y_noisy, 0

    X_synth = np.array(synthetic)
    X_aug = np.vstack([X, X_synth])
    y_aug = np.concatenate([y_noisy, np.full(n_synthetic, minority_label)])
    return X_aug, y_aug, n_synthetic
