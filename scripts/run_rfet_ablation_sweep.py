"""Sweep B — H2: RF/ET ablation using existing cwms and msbs method keys.

cwms (line 336 of run_relabeling_viability_sweep.py): suppression only, no synthesis
msbs (line 329): synthesis only, no weight modification
cwms_msbs (line 352): full pipeline

Identifies which component causes harm to RF/ET.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_relabeling_viability_sweep import (
    NOISE_PROTOCOLS, QUICK_SEEDS, run_single_viability,
)
from scripts._sweep_utils import check_gpu, load_completed

ABLATION_METHODS = [
    "no_cleaning",
    "class_proportional",
    "cwms",        # suppression only — existing method key, line 336
    "msbs",        # synthesis only — existing method key, line 329
    "cwms_msbs",   # full pipeline — existing method key, line 352
]
ABLATION_MODELS = ["random_forest", "extra_trees"]
ABLATION_PROTOCOLS = [
    "hidden_minority_low",
    "hidden_minority_medium",
    "hidden_minority_high",
]
DATASETS = ["pima", "credit-g", "yeast", "phoneme", "ecoli"]
SEEDS = QUICK_SEEDS
BUDGET = 0.10
RATIO = 0.15
OUTPUT_CSV = PROJECT_ROOT / "outputs" / "rfet-ablation-sweep.csv"

# 2 models × 5 methods × 5 datasets × 10 seeds × 3 protocols = 1,500 rows


def main():
    args = sys.argv[1:]
    use_gpu = "--gpu" in args
    if use_gpu:
        check_gpu()

    completed = load_completed(OUTPUT_CSV)
    total_written = 0

    for proto_name in ABLATION_PROTOCOLS:
        mn, mj = NOISE_PROTOCOLS[proto_name]
        for model_name in ABLATION_MODELS:
            for dataset in DATASETS:
                for seed in SEEDS:
                    needed = [
                        m for m in ABLATION_METHODS
                        if (dataset, model_name, seed, proto_name,
                            float(BUDGET), float(RATIO), m) not in completed
                    ]
                    if not needed:
                        continue
                    try:
                        batch = run_single_viability(
                            dataset, model_name, seed, proto_name, mn, mj,
                            BUDGET, RATIO, use_gpu=use_gpu, methods=needed,
                        )
                    except Exception as exc:
                        batch = [{"dataset": dataset, "model": model_name, "seed": seed,
                                   "noise_protocol": proto_name, "method": m, "error": str(exc),
                                   "mn_to_maj": mn, "maj_to_min": mj,
                                   "budget": BUDGET, "target_ratio": RATIO}
                                  for m in needed]
                        print(f"FAIL {dataset}/{model_name}/{seed}/{proto_name}: {exc}", flush=True)
                    df_batch = pd.DataFrame(batch)
                    write_header = not OUTPUT_CSV.exists() or OUTPUT_CSV.stat().st_size == 0
                    df_batch.to_csv(OUTPUT_CSV, mode="a", header=write_header, index=False)
                    total_written += len(batch)
                    print(f"  {proto_name}/{dataset}/{model_name}/{seed}: {needed}", flush=True)

    print(f"\nDone. {total_written} rows -> {OUTPUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
