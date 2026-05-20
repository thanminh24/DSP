"""OOF-filtered SMOTE: drop top-frac suspicious minority before synthesis."""
from __future__ import annotations

import numpy as np
from imblearn.over_sampling import SMOTE


def filtered_smote_augment(
    X: np.ndarray,
    y_noisy: np.ndarray,
    suspiciousness: np.ndarray,
    minority_label: int,
    majority_label: int,
    filter_frac: float = 0.05,
    seed: int = 42,
    sampling_strategy: float | str = "auto",
    k_neighbors: int = 5,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Drop top-`filter_frac` of minority pool by OOF loss; SMOTE-augment the rest.

    Returns:
        X_aug, y_aug: augmented training data.
        info: {"n_filtered", "n_synthetic", "minority_kept", "filter_indices"}.
    """
    n = len(y_noisy)
    min_idx = np.where(y_noisy == minority_label)[0]
    n_min = len(min_idx)
    n_filter = int(np.floor(filter_frac * n_min))
    if n_filter > 0 and n_min - n_filter >= max(k_neighbors + 1, 2):
        susp_min = suspiciousness[min_idx]
        top_local = np.argsort(susp_min)[::-1][:n_filter]
        filter_idx = min_idx[top_local]
    else:
        filter_idx = np.array([], dtype=int)

    keep_mask = np.ones(n, dtype=bool)
    keep_mask[filter_idx] = False
    X_keep, y_keep = X[keep_mask], y_noisy[keep_mask]

    n_min_keep = int((y_keep == minority_label).sum())
    if n_min_keep < k_neighbors + 1 or len(np.unique(y_keep)) < 2:
        return X_keep, y_keep, {
            "n_filtered": int(len(filter_idx)),
            "n_synthetic": 0,
            "minority_kept": n_min_keep,
            "filter_indices": filter_idx,
            "smote_applied": False,
        }

    sm = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=min(k_neighbors, n_min_keep - 1),
        random_state=seed,
    )
    X_aug, y_aug = sm.fit_resample(X_keep, y_keep)
    return X_aug, y_aug, {
        "n_filtered": int(len(filter_idx)),
        "n_synthetic": int(len(y_aug) - len(y_keep)),
        "minority_kept": n_min_keep,
        "filter_indices": filter_idx,
        "smote_applied": True,
    }


def plain_smote_augment(
    X: np.ndarray,
    y_noisy: np.ndarray,
    minority_label: int,
    seed: int = 42,
    sampling_strategy: float | str = "auto",
    k_neighbors: int = 5,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Unfiltered SMOTE baseline."""
    n_min = int((y_noisy == minority_label).sum())
    if n_min < k_neighbors + 1 or len(np.unique(y_noisy)) < 2:
        return X, y_noisy, {"n_synthetic": 0, "smote_applied": False}
    sm = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=min(k_neighbors, n_min - 1),
        random_state=seed,
    )
    X_aug, y_aug = sm.fit_resample(X, y_noisy)
    return X_aug, y_aug, {"n_synthetic": int(len(y_aug) - len(y_noisy)), "smote_applied": True}
