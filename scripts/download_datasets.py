"""Download Yeast, Ecoli, Phoneme datasets from OpenML and cache as Parquet.

Run once — never called during experiments. Requires network.
Outputs data/yeast.parquet, data/ecoli.parquet, data/phoneme.parquet
Each parquet has a 'target' column with string labels matching MINORITY_LABELS in data_loader.py
"""
from __future__ import annotations

from pathlib import Path

import openml
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def fetch_yeast() -> None:
    """OpenML dataset 181 — yeast multi-class, binarize MIT vs rest."""
    dataset = openml.datasets.get_dataset(181)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    y_binary = (y == "MIT").astype(int)
    df = pd.DataFrame(X)
    df["target"] = y_binary.map({1: "MIT", 0: "other"})
    df.to_parquet(DATA_DIR / "yeast.parquet", index=False)
    print(f"yeast: {df.shape}, minority={y_binary.mean():.3f}")


def fetch_ecoli() -> None:
    """OpenML dataset 39 — binarize 'im' vs rest (~22.9% minority)."""
    dataset = openml.datasets.get_dataset(39)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    y_binary = (y == "im").astype(int)
    df = pd.DataFrame(X)
    df["target"] = y_binary.map({1: "im", 0: "other"})
    df.to_parquet(DATA_DIR / "ecoli.parquet", index=False)
    print(f"ecoli: {df.shape}, minority={y_binary.mean():.3f}")


def fetch_phoneme() -> None:
    """OpenML dataset 1489 — class 2=nasal=minority (~29.3%)."""
    dataset = openml.datasets.get_dataset(1489)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    y_binary = (y == "2").astype(int)  # '2' = nasal (minority class)
    df = pd.DataFrame(X)
    df["target"] = y_binary.map({1: "nasal", 0: "other"})
    df.to_parquet(DATA_DIR / "phoneme.parquet", index=False)
    print(f"phoneme: {df.shape}, minority={y_binary.mean():.3f}")


if __name__ == "__main__":
    fetch_yeast()
    fetch_ecoli()
    fetch_phoneme()
    print("Done — 3 datasets saved to data/")
