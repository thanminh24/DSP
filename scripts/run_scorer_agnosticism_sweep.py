"""Scorer comparison: CWMS+MSBS-compatible models x scorer variants x medium noise.

Run with: python scripts/run_scorer_agnosticism_sweep.py --gpu
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_relabeling_viability_sweep import (
    DATASETS, NOISE_PROTOCOLS, QUICK_SEEDS, run_single_viability,
)
from scripts._sweep_utils import check_gpu, load_completed

SCORER_METHODS = [
    "no_cleaning", "class_proportional",
    "cwms_msbs", "cwms_msbs_knn", "cwms_msbs_crossfamily",
]
SCORER_MODELS = ["lr", "svm", "hgb", "lightgbm", "catboost"]
TARGET_PROTOCOL = "hidden_minority_medium"
POC_BUDGET, POC_RATIO = 0.10, 0.15
OUTPUT_CSV = PROJECT_ROOT / "outputs" / "scorer-agnosticism-sweep.csv"


def main():
    args = sys.argv[1:]
    use_gpu = "--gpu" in args
    if use_gpu:
        print("Checking GPU availability...", flush=True)
        check_gpu()

    mn, mj = NOISE_PROTOCOLS[TARGET_PROTOCOL]
    completed = load_completed(OUTPUT_CSV)
    total_written = 0

    for model_name in SCORER_MODELS:
        for dataset in DATASETS:
            for seed in QUICK_SEEDS:
                needed = [
                    m for m in SCORER_METHODS
                    if (dataset, model_name, seed, TARGET_PROTOCOL,
                        float(POC_BUDGET), float(POC_RATIO), m) not in completed
                ]
                if not needed:
                    continue
                try:
                    batch = run_single_viability(
                        dataset, model_name, seed, TARGET_PROTOCOL, mn, mj,
                        POC_BUDGET, POC_RATIO, use_gpu=use_gpu, methods=needed,
                    )
                except Exception as exc:
                    batch = [{"dataset": dataset, "model": model_name, "seed": seed,
                               "noise_protocol": TARGET_PROTOCOL, "method": m, "error": str(exc),
                               "mn_to_maj": mn, "maj_to_min": mj,
                               "budget": POC_BUDGET, "target_ratio": POC_RATIO}
                              for m in needed]
                    print(f"FAIL {dataset}/{model_name}/{seed}/{TARGET_PROTOCOL}: {exc}", flush=True)
                df_batch = pd.DataFrame(batch)
                write_header = not OUTPUT_CSV.exists() or OUTPUT_CSV.stat().st_size == 0
                df_batch.to_csv(OUTPUT_CSV, mode="a", header=write_header, index=False)
                total_written += len(batch)
                print(f"  {dataset}/{model_name}/{seed}: {needed}", flush=True)

    print(f"\nDone. {total_written} rows -> {OUTPUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
