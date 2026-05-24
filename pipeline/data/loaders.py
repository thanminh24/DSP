from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

DATASETS: dict[str, dict] = {
    "pima":          {"file": "pima.parquet"},
    "credit-g":      {"file": "credit-g.parquet"},
    "yeast":         {"file": "yeast.parquet"},
    "ecoli":         {"file": "ecoli.parquet"},
    "phoneme":       {"file": "phoneme.parquet"},
    "breast_cancer": {"file": "breast_cancer.parquet"},
    "ilpd":          {"file": "ilpd.parquet"},
    "blood":         {"file": "blood.parquet"},
    "haberman":      {"file": "haberman.parquet"},
    "ionosphere":    {"file": "ionosphere.parquet"},
    "vehicle_bus":   {"file": "vehicle_bus.parquet"},
    "glass_float":   {"file": "glass_float.parquet"},
    "abalone":       {"file": "abalone.parquet"},
    "spambase":      {"file": "spambase.parquet"},
    "kc1":           {"file": "kc1.parquet"},
}

MINORITY_LABELS: dict[str, str] = {
    "pima":          "tested_positive",
    "credit-g":      "bad",
    "yeast":         "MIT",
    "ecoli":         "im",
    "phoneme":       "nasal",
    "breast_cancer": "malignant",
    "ilpd":          "no_disease",
    "blood":         "donated",
    "haberman":      "died",
    "ionosphere":    "bad",
    "vehicle_bus":   "bus",
    "glass_float":   "window_float",
    "abalone":       "rings_gt_10",
    "spambase":      "spam",
    "kc1":           "defective",
}

ALL_15_DATASETS = list(DATASETS.keys())

TARGET_MINORITY_RATIO = 0.15
MIN_TO_MAJ_NOISE = 0.30
MAJ_TO_MIN_NOISE = 0.10


def _get_categorical_columns(df_no_target: pd.DataFrame) -> list[str]:
    cat_cols = df_no_target.select_dtypes(include=["category", "object"]).columns
    return list(cat_cols)


def load_dataset(
    name: str,
) -> tuple[pd.DataFrame, np.ndarray, list[str], list[str]]:
    """Load a dataset by name. Returns (X_df, y_binary, cat_cols, feature_names)."""
    file = DATASETS[name]["file"]
    df = pd.read_parquet(DATA_DIR / file)
    minority_str = MINORITY_LABELS[name]

    y_raw = df["target"]
    y = np.where(y_raw == minority_str, 1, 0).astype(int)

    X = df.drop(columns=["target"])
    cat_cols = _get_categorical_columns(X)
    feature_names = list(X.columns)

    return X, y, cat_cols, feature_names


def induce_imbalance(
    X: pd.DataFrame,
    y: np.ndarray,
    minority_label: int = 1,
    target_ratio: float = TARGET_MINORITY_RATIO,
    rng: np.random.Generator | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Subsample majority class to achieve target minority ratio."""
    if rng is None:
        rng = np.random.default_rng()
    current_ratio = np.mean(y == minority_label)
    if current_ratio <= target_ratio:
        return X, y

    minority_idx = np.where(y == minority_label)[0]
    majority_idx = np.where(y != minority_label)[0]
    target_minority_count = int(
        (target_ratio / (1 - target_ratio)) * len(majority_idx)
    )
    keep_count = min(len(minority_idx), max(2, target_minority_count))
    keep_minority = rng.choice(minority_idx, size=keep_count, replace=False)
    keep_idx = np.concatenate([majority_idx, keep_minority])
    rng.shuffle(keep_idx)
    X_out = X.iloc[keep_idx].reset_index(drop=True)
    return X_out, y[keep_idx].copy()


def inject_noise(
    y_clean: np.ndarray,
    minority_label: int = 1,
    min_to_maj: float = MIN_TO_MAJ_NOISE,
    maj_to_min: float = MAJ_TO_MIN_NOISE,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Inject class-dependent label noise. Returns (y_noisy, noisy_mask)."""
    if rng is None:
        rng = np.random.default_rng()
    classes = np.unique(y_clean)
    if len(classes) != 2:
        raise ValueError("Expected binary labels.")
    majority_label = int([c for c in classes if c != minority_label][0])

    y_noisy = y_clean.copy()
    noisy_mask = np.zeros_like(y_clean, dtype=bool)
    for i, label in enumerate(y_clean):
        flip_prob = min_to_maj if label == minority_label else maj_to_min
        flipped_label = majority_label if label == minority_label else minority_label
        if rng.random() < flip_prob:
            y_noisy[i] = flipped_label
            noisy_mask[i] = True
    return y_noisy, noisy_mask
