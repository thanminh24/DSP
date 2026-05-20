"""Full sweep over surviving augmentation methods x 3 noise levels."""
from __future__ import annotations

import dataclasses
import itertools
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.core.config import BaseExperimentConfig
from scripts.run_augment_experiment import run_single_augment

DATASETS = ["pima", "credit-g", "yeast", "phoneme", "ecoli"]
MODELS = ["lr"]  # HGB too slow (250s/fit on this machine)
SEEDS = [13, 17, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101]
NOISE_LEVELS = {
    "low": (0.10, 0.05),
    "medium": (0.30, 0.10),
    "high": (0.40, 0.20),
}
OUT_CSV = "outputs/augment-sweep-results.csv"

BASE_METHODS = ["no_cleaning", "class_proportional"]
SURVIVING_DEFAULT = ["balanced_oof_relabel", "random_relabel", "oof_filtered_smote", "plain_smote"]


def parse_methods(argv):
    if len(argv) > 1:
        return BASE_METHODS + argv[1:]
    return BASE_METHODS + SURVIVING_DEFAULT


def main():
    methods = parse_methods(sys.argv)
    print(f"running methods: {methods}")
    rows = []
    combos = list(itertools.product(DATASETS, MODELS, SEEDS, NOISE_LEVELS.items()))
    total = len(combos)
    for i, (ds, mdl, sd, (lvl, (mn, mj))) in enumerate(combos, 1):
        cfg = dataclasses.replace(
            BaseExperimentConfig(),
            minority_to_majority_noise=mn,
            majority_to_minority_noise=mj,
        )
        try:
            batch = run_single_augment(ds, mdl, sd, methods, cfg)
            for r in batch:
                r["noise_level"] = lvl
                r["mn_to_maj"] = mn
                r["maj_to_min"] = mj
            rows.extend(batch)
        except Exception as e:
            print(f"FAIL {ds}/{mdl}/{sd}/{lvl}: {e}")
        if i % 20 == 0:
            print(f"  progress: {i}/{total}")
            pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"wrote {len(rows)} rows -> {OUT_CSV}")


if __name__ == "__main__":
    main()
