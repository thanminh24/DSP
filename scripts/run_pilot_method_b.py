"""Pilot for Method B -- OOF-filtered SMOTE."""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.core.config import BaseExperimentConfig
from scripts.run_augment_experiment import run_single_augment

DATASETS = ["pima", "credit-g", "yeast", "phoneme"]
MODELS = ["lr", "hgb"]
SEEDS = [13, 17, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101]
METHODS = ["no_cleaning", "class_proportional", "plain_smote", "oof_filtered_smote"]
OUT_CSV = "outputs/pilot-method-b-results.csv"


def main():
    cfg = BaseExperimentConfig()
    rows = []
    total = len(DATASETS) * len(MODELS) * len(SEEDS)
    done = 0
    for ds, mdl, sd in itertools.product(DATASETS, MODELS, SEEDS):
        try:
            rows.extend(run_single_augment(ds, mdl, sd, METHODS, cfg))
        except Exception as e:
            print(f"FAIL {ds}/{mdl}/{sd}: {e}", flush=True)
        done += 1
        if done % 4 == 0:
            pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
            print(f"  progress: {done}/{total}  ({len(rows)} rows saved)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"wrote {len(df)} rows -> {OUT_CSV}", flush=True)
    _print_summary(df)


def _print_summary(df: pd.DataFrame):
    pivot = df.pivot_table(
        index=["dataset", "model"], columns="method",
        values="balanced_accuracy", aggfunc="mean",
    )
    pivot["d_cp"] = pivot["oof_filtered_smote"] - pivot["class_proportional"]
    pivot["d_plain"] = pivot["oof_filtered_smote"] - pivot["plain_smote"]
    print("\n-- Method B BA summary (mean over 20 seeds) --")
    print(pivot.round(4).to_string())
    wins_cp = (pivot["d_cp"] > 0).sum()
    print(f"\nMethod B wins vs class_proportional: {wins_cp} / {len(pivot)}")
    print(f"Mean delta vs class_proportional: {pivot['d_cp'].mean():+.4f}")
    print(f"Mean delta vs plain_smote:        {pivot['d_plain'].mean():+.4f}")


if __name__ == "__main__":
    main()
