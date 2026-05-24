"""Full benchmark: all models x our solution methods x 3 protocols x CUDA.

Run with:
  python scripts/run_full_benchmark_solution.py --gpu
  python scripts/run_full_benchmark_solution.py --gpu --ratio 0.30 --output outputs/full-benchmark-ir030-solution.csv
  python scripts/run_full_benchmark_solution.py --gpu --datasets pima credit-g yeast phoneme ecoli
"""
from __future__ import annotations
import os
# Cap per-process thread count so concurrent sweeps don't over-subscribe CPU
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
from scripts._sweep_utils import check_gpu, load_completed
from pipeline.data.loaders import ALL_15_DATASETS
from pipeline.models.factories import list_publication_models

CWMS_FULL_METHODS = [
    "no_cleaning", "class_proportional", "smote",
    "msbs", "cwms", "cwms_msbs", "cwms_msbs_shuffled",
]
BASELINE_ONLY_METHODS = ["no_cleaning", "class_proportional", "smote"]
FULL_PROTOCOLS = ["hidden_minority_low", "hidden_minority_medium", "hidden_minority_high"]
POC_BUDGET = 0.10


def _methods_for(model_name: str) -> list[str]:
    if model_name in ("xgboost", "calibrated_lr"):
        return BASELINE_ONLY_METHODS
    return CWMS_FULL_METHODS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--ratio", type=float, default=0.15,
                        help="Minority target ratio (default: 0.15)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: auto-named by ratio)")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Datasets to run (default: all 15)")
    args = parser.parse_args()

    ratio = args.ratio
    if args.output:
        output_csv = Path(args.output)
    else:
        output_csv = PROJECT_ROOT / "outputs" / f"full-benchmark-ir{int(ratio*100):03d}-solution.csv"
        # Keep original name for default ratio=0.15 run
        if ratio == 0.15:
            output_csv = PROJECT_ROOT / "outputs" / "full-benchmark-solution.csv"

    datasets = args.datasets if args.datasets else ALL_15_DATASETS

    if args.gpu:
        print("Checking GPU availability...", flush=True)
        check_gpu()

    print(f"Ratio: {ratio}, Datasets: {len(datasets)}, Output: {output_csv}", flush=True)

    models = list_publication_models()
    completed = load_completed(output_csv)
    total_written = 0

    for proto_name in FULL_PROTOCOLS:
        mn, mj = NOISE_PROTOCOLS[proto_name]
        for model_name in models:
            methods_for_model = _methods_for(model_name)
            for dataset in datasets:
                for seed in QUICK_SEEDS:
                    needed = [
                        m for m in methods_for_model
                        if (dataset, model_name, seed, proto_name,
                            float(POC_BUDGET), float(ratio), m) not in completed
                    ]
                    if not needed:
                        continue
                    try:
                        batch = run_single_viability(
                            dataset, model_name, seed, proto_name, mn, mj,
                            POC_BUDGET, ratio, use_gpu=args.gpu, methods=needed,
                        )
                    except Exception as exc:
                        batch = [{"dataset": dataset, "model": model_name, "seed": seed,
                                   "noise_protocol": proto_name, "method": m, "error": str(exc),
                                   "mn_to_maj": mn, "maj_to_min": mj,
                                   "budget": POC_BUDGET, "target_ratio": ratio}
                                  for m in needed]
                        print(f"FAIL {dataset}/{model_name}/{seed}/{proto_name}: {exc}", flush=True)
                    df_batch = pd.DataFrame(batch)
                    write_header = not output_csv.exists() or output_csv.stat().st_size == 0
                    df_batch.to_csv(output_csv, mode="a", header=write_header, index=False)
                    total_written += len(batch)
                    print(f"  {proto_name}/{dataset}/{model_name}/{seed}: {needed}", flush=True)

    print(f"\nDone. {total_written} rows -> {output_csv}", flush=True)


if __name__ == "__main__":
    main()
