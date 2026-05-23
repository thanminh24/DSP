"""Download all benchmark datasets from OpenML and cache as Parquet.

Run once — never called during experiments. Requires network.
Each parquet has a 'target' column with string labels matching MINORITY_LABELS in loaders.py.
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


def fetch_breast_cancer() -> None:
    """OpenML 1510 — Breast Cancer Wisconsin. class '2'=malignant=minority (~37%)."""
    dataset = openml.datasets.get_dataset(1510)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    # OpenML 1510 labels: '2'=malignant (minority), '1'=benign (majority)
    minority = (y == "2")
    df = pd.DataFrame(X)
    df["target"] = minority.map({True: "malignant", False: "benign"})
    df.to_parquet(DATA_DIR / "breast_cancer.parquet", index=False)
    print(f"breast_cancer: {df.shape}, minority={minority.mean():.3f}")


def fetch_ilpd() -> None:
    """OpenML 1480 — Indian Liver Patient Dataset. diseased=minority (~71% → flip so minority<50%)."""
    dataset = openml.datasets.get_dataset(1480)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    # Class 1=liver disease (71%=majority), class 2=no disease (29%=minority) → minority=class 2
    vals, counts = np.unique(y, return_counts=True)
    minority_val = vals[np.argmin(counts)]
    minority = (y == minority_val)
    df = pd.DataFrame(X)
    df["target"] = minority.map({True: "no_disease", False: "liver_disease"})
    df.to_parquet(DATA_DIR / "ilpd.parquet", index=False)
    print(f"ilpd: {df.shape}, minority={minority.mean():.3f}")


def fetch_blood() -> None:
    """OpenML 1464 — Blood Transfusion Service. donated=minority (~24%)."""
    dataset = openml.datasets.get_dataset(1464)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    vals, counts = np.unique(y, return_counts=True)
    minority_val = vals[np.argmin(counts)]
    minority = (y == minority_val)
    df = pd.DataFrame(X)
    df["target"] = minority.map({True: "donated", False: "not_donated"})
    df.to_parquet(DATA_DIR / "blood.parquet", index=False)
    print(f"blood: {df.shape}, minority={minority.mean():.3f}")


def fetch_haberman() -> None:
    """OpenML 43 — Haberman Survival. died_within_5yrs=minority (~27%)."""
    dataset = openml.datasets.get_dataset(43)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    vals, counts = np.unique(y, return_counts=True)
    minority_val = vals[np.argmin(counts)]
    minority = (y == minority_val)
    df = pd.DataFrame(X)
    df["target"] = minority.map({True: "died", False: "survived"})
    df.to_parquet(DATA_DIR / "haberman.parquet", index=False)
    print(f"haberman: {df.shape}, minority={minority.mean():.3f}")


def fetch_ionosphere() -> None:
    """OpenML 59 — Ionosphere. bad=minority (~36%)."""
    dataset = openml.datasets.get_dataset(59)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    vals, counts = np.unique(y, return_counts=True)
    minority_val = vals[np.argmin(counts)]
    minority = (y == minority_val)
    df = pd.DataFrame(X)
    df["target"] = minority.map({True: "bad", False: "good"})
    df.to_parquet(DATA_DIR / "ionosphere.parquet", index=False)
    print(f"ionosphere: {df.shape}, minority={minority.mean():.3f}")


def fetch_vehicle_bus() -> None:
    """OpenML 54 — Vehicle Silhouettes. bus=minority vs rest."""
    dataset = openml.datasets.get_dataset(54)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    minority = (y == "bus")
    df = pd.DataFrame(X)
    df["target"] = minority.map({True: "bus", False: "other"})
    df.to_parquet(DATA_DIR / "vehicle_bus.parquet", index=False)
    print(f"vehicle_bus: {df.shape}, minority={minority.mean():.3f}")


def fetch_glass_float() -> None:
    """OpenML 41 — Glass Identification. 'build wind float'=minority (~33%) vs rest."""
    dataset = openml.datasets.get_dataset(41)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    # OpenML 41 uses full string labels; 'build wind float' is the largest single class (70/214)
    minority = (y == "build wind float")
    df = pd.DataFrame(X)
    df["target"] = minority.map({True: "window_float", False: "other"})
    df.to_parquet(DATA_DIR / "glass_float.parquet", index=False)
    print(f"glass_float: {df.shape}, minority={minority.mean():.3f}")


def fetch_abalone() -> None:
    """OpenML 183 — Abalone. rings>10=minority, stratified subsample to 2000."""
    import numpy as np
    dataset = openml.datasets.get_dataset(183)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    y_int = pd.to_numeric(y, errors="coerce")
    minority = (y_int > 10).fillna(False)
    df = pd.DataFrame(X)
    df["target"] = minority.map({True: "rings_gt_10", False: "rings_le_10"})
    # Stratified subsample to 2000 rows (SVM tractability)
    if len(df) > 2000:
        from sklearn.model_selection import train_test_split
        _, df = train_test_split(df, test_size=2000, stratify=df["target"], random_state=42)
        df = df.reset_index(drop=True)
    min_frac = (df["target"] == "rings_gt_10").mean()
    df.to_parquet(DATA_DIR / "abalone.parquet", index=False)
    print(f"abalone: {df.shape}, minority={min_frac:.3f}")


def fetch_spambase() -> None:
    """OpenML 44 — Spambase. spam=minority (~39%)."""
    dataset = openml.datasets.get_dataset(44)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    vals, counts = np.unique(y, return_counts=True)
    minority_val = vals[np.argmin(counts)]
    minority = (y == minority_val)
    df = pd.DataFrame(X)
    df["target"] = minority.map({True: "spam", False: "not_spam"})
    df.to_parquet(DATA_DIR / "spambase.parquet", index=False)
    print(f"spambase: {df.shape}, minority={minority.mean():.3f}")


def fetch_kc1() -> None:
    """OpenML 1067 — KC1 software defect prediction. defective=minority (~15.5%)."""
    dataset = openml.datasets.get_dataset(1067)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    # True = defective (minority)
    minority = (y == True) | (y == "true") | (y == "True")
    df = pd.DataFrame(X)
    df["target"] = minority.map({True: "defective", False: "clean"})
    df.to_parquet(DATA_DIR / "kc1.parquet", index=False)
    print(f"kc1: {df.shape}, minority={minority.mean():.3f}")


if __name__ == "__main__":
    import numpy as np
    fetch_yeast()
    fetch_ecoli()
    fetch_phoneme()
    fetch_breast_cancer()
    fetch_ilpd()
    fetch_blood()
    fetch_haberman()
    fetch_ionosphere()
    fetch_vehicle_bus()
    fetch_glass_float()
    fetch_abalone()
    fetch_spambase()
    fetch_kc1()
    print("Done — 13 datasets saved to data/")
