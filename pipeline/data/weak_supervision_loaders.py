"""Load local weak-supervision tabular datasets for realistic noisy-label tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

WEAK_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "weak-supervision"


def load_weak_supervision_dataset(name: str):
    """Return (X, y_noisy, y_true, cat_cols, metadata) from a cached CSV/Parquet.

    Expected columns:
    - `label`: noisy/weak label used for training
    - `gold`: clean label used only for evaluation
    """
    path = _resolve_dataset_path(name)
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    _validate_columns(df, path)
    y_noisy = df["label"].to_numpy()
    y_true = df["gold"].to_numpy()
    X = df.drop(columns=["label", "gold"])
    cat_cols = list(X.select_dtypes(include=["category", "object"]).columns)
    metadata = {
        "name": name,
        "path": str(path),
        "rows": int(len(df)),
        "features": int(X.shape[1]),
        "minority_label": _minority_label(y_true),
    }
    return X, y_noisy, y_true, cat_cols, metadata


def list_cached_weak_supervision_datasets() -> list[str]:
    """List cached weak-supervision datasets by basename."""
    if not WEAK_DATA_DIR.exists():
        return []
    names = []
    for path in WEAK_DATA_DIR.iterdir():
        if path.suffix.lower() in {".csv", ".parquet"}:
            names.append(path.stem)
    return sorted(names)


def _resolve_dataset_path(name: str) -> Path:
    for suffix in (".parquet", ".csv"):
        path = WEAK_DATA_DIR / f"{name}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(
        f"No cached weak-supervision dataset found for {name!r} in {WEAK_DATA_DIR}"
    )


def _validate_columns(df: pd.DataFrame, path: Path) -> None:
    missing = {"label", "gold"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    if df["label"].isna().any() or df["gold"].isna().any():
        raise ValueError(f"{path} has missing labels in `label` or `gold`")


def _minority_label(y: np.ndarray):
    values, counts = np.unique(y, return_counts=True)
    return values[np.argmin(counts)].item()
