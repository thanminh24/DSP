"""Expanded competitor head-to-head: LR+SVM+HGB × 3 protocols × 15 datasets.

Run with:
  python scripts/run_expanded_competitor_headtohead.py
  python scripts/run_expanded_competitor_headtohead.py --datasets pima credit-g yeast phoneme ecoli
"""
from __future__ import annotations
import os
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "4")
import argparse
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_relabeling_viability_sweep import (
    NOISE_PROTOCOLS, QUICK_SEEDS, run_single_viability,
)
from scripts._sweep_utils import load_completed
from pipeline.data.loaders import ALL_15_DATASETS

COMPETITOR_METHODS = [
    "no_cleaning",
    "class_proportional",
    "smote",
    "iw_smote",
    "sw_framework",
    "cwms_msbs",
]
EXPANDED_MODELS = ["lr", "svm", "hgb"]
FULL_PROTOCOLS = ["hidden_minority_low", "hidden_minority_medium", "hidden_minority_high"]
POC_BUDGET, POC_RATIO = 0.10, 0.15
OUTPUT_CSV = PROJECT_ROOT / "outputs" / "competitor-headtohead-expanded.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Datasets to run (default: all 15)")
    args = parser.parse_args()

    datasets = args.datasets if args.datasets else ALL_15_DATASETS

    print(f"Models: {EXPANDED_MODELS}, Protocols: {FULL_PROTOCOLS}", flush=True)
    print(f"Datasets: {len(datasets)}, Output: {OUTPUT_CSV}", flush=True)

    completed = load_completed(OUTPUT_CSV)
    total_written = 0

    for proto_name in FULL_PROTOCOLS:
        mn, mj = NOISE_PROTOCOLS[proto_name]
        for model_name in EXPANDED_MODELS:
            for dataset in datasets:
                for seed in QUICK_SEEDS:
                    needed = [
                        m for m in COMPETITOR_METHODS
                        if (dataset, model_name, seed, proto_name,
                            float(POC_BUDGET), float(POC_RATIO), m) not in completed
                    ]
                    if not needed:
                        continue
                    try:
                        batch = run_single_viability(
                            dataset, model_name, seed, proto_name, mn, mj,
                            POC_BUDGET, POC_RATIO, use_gpu=False, methods=needed,
                        )
                    except Exception as exc:
                        batch = [{"dataset": dataset, "model": model_name, "seed": seed,
                                   "noise_protocol": proto_name, "method": m, "error": str(exc),
                                   "mn_to_maj": mn, "maj_to_min": mj,
                                   "budget": POC_BUDGET, "target_ratio": POC_RATIO}
                                  for m in needed]
                        print(f"FAIL {dataset}/{model_name}/{seed}/{proto_name}: {exc}", flush=True)
                    df_batch = pd.DataFrame(batch)
                    write_header = not OUTPUT_CSV.exists() or OUTPUT_CSV.stat().st_size == 0
                    df_batch.to_csv(OUTPUT_CSV, mode="a", header=write_header, index=False)
                    total_written += len(batch)
                    print(f"  {proto_name}/{model_name}/{dataset}/{seed}: {needed}", flush=True)

    print(f"\nDone. {total_written} rows -> {OUTPUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
