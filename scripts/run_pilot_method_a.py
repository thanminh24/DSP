"""Pilot for Method A -- balanced-OOF Type A relabeling."""
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
METHODS = ["no_cleaning", "class_proportional", "balanced_oof_relabel", "random_relabel"]
OUT_CSV = "outputs/pilot-method-a-results.csv"


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
    pivot["delta_vs_cp"] = pivot["balanced_oof_relabel"] - pivot["class_proportional"]
    pivot["delta_vs_random"] = pivot["balanced_oof_relabel"] - pivot["random_relabel"]
    print("\n-- Method A BA summary (mean over 20 seeds) --")
    print(pivot.round(4).to_string())
    wins = (pivot["delta_vs_cp"] > 0).sum()
    print(f"\nMethod A wins vs class_proportional: {wins} / {len(pivot)} combos")
    print(f"Mean delta vs class_proportional: {pivot['delta_vs_cp'].mean():+.4f}")
    print(f"Mean delta vs random_relabel:    {pivot['delta_vs_random'].mean():+.4f}")

    rc = df[df["method"] == "balanced_oof_relabel"]["relabel_correctness"].dropna()
    if len(rc) > 0:
        print(f"\nrelabel_correctness (balanced_oof): mean={rc.mean():.4f}, median={rc.median():.4f}")
    rc_r = df[df["method"] == "random_relabel"]["relabel_correctness"].dropna()
    if len(rc_r) > 0:
        print(f"relabel_correctness (random):       mean={rc_r.mean():.4f}, median={rc_r.median():.4f}")


if __name__ == "__main__":
    main()
