"""Sweep A — C4: Failure-mode protocols (symmetric + reverse_asymmetric) on 5 datasets.

Tests NoiSyn under noise conditions it was NOT designed for.
Expected: degradation, strengthening the operating-condition framing.
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

FAILURE_PROTOCOLS = {
    "symmetric":          NOISE_PROTOCOLS["symmetric"],
    "reverse_asymmetric": NOISE_PROTOCOLS["reverse_asymmetric"],
}
FAILURE_METHODS = ["no_cleaning", "class_proportional", "smote", "cwms_msbs"]
FAILURE_MODELS = ["lr"]
DATASETS = ["pima", "credit-g", "yeast", "phoneme", "ecoli"]
SEEDS = QUICK_SEEDS
BUDGET = 0.10
RATIO = 0.15
OUTPUT_CSV = PROJECT_ROOT / "outputs" / "failure-mode-sweep.csv"

# 2 protocols × 5 datasets × 10 seeds × 4 methods × 1 model = 400 rows


def main():
    args = sys.argv[1:]
    use_gpu = "--gpu" in args
    if use_gpu:
        check_gpu()

    completed = load_completed(OUTPUT_CSV)
    total_written = 0

    for proto_name, (mn, mj) in FAILURE_PROTOCOLS.items():
        for model_name in FAILURE_MODELS:
            for dataset in DATASETS:
                for seed in SEEDS:
                    needed = [
                        m for m in FAILURE_METHODS
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
