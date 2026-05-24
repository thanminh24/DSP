"""Sweep D — M2: Clean-data ablation (zero noise).

Answers: is NoiSyn's gain noise-specific, or does it help on clean imbalanced data?
- If ΔBA > 0 on clean data → reframe as 'boundary-aware synthesis' (still novel)
- If ΔBA ≈ 0 on clean data → noise-specific mechanism confirmed; stronger paper story
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_relabeling_viability_sweep import QUICK_SEEDS, run_single_viability
from scripts._sweep_utils import check_gpu, load_completed

CLEAN_PROTOCOL_NAME = "clean"
CLEAN_MN, CLEAN_MJ = 0.00, 0.00
CLEAN_METHODS = ["no_cleaning", "class_proportional", "smote", "cwms_msbs"]
CLEAN_MODELS = ["lr", "svm"]
DATASETS = ["pima", "credit-g", "yeast", "phoneme", "ecoli"]
SEEDS = QUICK_SEEDS
BUDGET = 0.10
RATIO = 0.15
OUTPUT_CSV = PROJECT_ROOT / "outputs" / "clean-data-ablation.csv"

# 2 models × 4 methods × 5 datasets × 10 seeds = 400 rows


def main():
    args = sys.argv[1:]
    use_gpu = "--gpu" in args
    if use_gpu:
        check_gpu()

    completed = load_completed(OUTPUT_CSV)
    total_written = 0

    for model_name in CLEAN_MODELS:
        for dataset in DATASETS:
            for seed in SEEDS:
                needed = [
                    m for m in CLEAN_METHODS
                    if (dataset, model_name, seed, CLEAN_PROTOCOL_NAME,
                        float(BUDGET), float(RATIO), m) not in completed
                ]
                if not needed:
                    continue
                try:
                    batch = run_single_viability(
                        dataset, model_name, seed, CLEAN_PROTOCOL_NAME,
                        CLEAN_MN, CLEAN_MJ,
                        BUDGET, RATIO, use_gpu=use_gpu, methods=needed,
                    )
                except Exception as exc:
                    batch = [{"dataset": dataset, "model": model_name, "seed": seed,
                               "noise_protocol": CLEAN_PROTOCOL_NAME, "method": m,
                               "error": str(exc), "mn_to_maj": CLEAN_MN,
                               "maj_to_min": CLEAN_MJ, "budget": BUDGET,
                               "target_ratio": RATIO}
                              for m in needed]
                    print(f"FAIL {dataset}/{model_name}/{seed}/clean: {exc}", flush=True)
                df_batch = pd.DataFrame(batch)
                write_header = not OUTPUT_CSV.exists() or OUTPUT_CSV.stat().st_size == 0
                df_batch.to_csv(OUTPUT_CSV, mode="a", header=write_header, index=False)
                total_written += len(batch)
                print(f"  clean/{dataset}/{model_name}/{seed}: {needed}", flush=True)

    print(f"\nDone. {total_written} rows -> {OUTPUT_CSV}", flush=True)

    if OUTPUT_CSV.exists():
        df = pd.read_csv(OUTPUT_CSV)
        df = df[df["balanced_accuracy"].notna()]
        print("\nClean-data ablation summary (mean BA over datasets × seeds):")
        for model_name in CLEAN_MODELS:
            mdf = df[df["model"] == model_name]
            print(f"\n  Model: {model_name}")
            for method in CLEAN_METHODS:
                vals = mdf[mdf["method"] == method]["balanced_accuracy"]
                if len(vals):
                    print(f"    {method:<25}: BA={vals.mean():.4f}")
            cp = mdf[mdf["method"] == "class_proportional"]["balanced_accuracy"].mean()
            cw = mdf[mdf["method"] == "cwms_msbs"]["balanced_accuracy"].mean()
            delta = (cw - cp) * 100
            print(f"  ΔBA(cwms_msbs vs class_prop) = {delta:+.2f}pp")
            if delta > 0.5:
                print("  → cwms_msbs helps on clean data: reframe as boundary-aware synthesis")
            elif delta < -0.5:
                print("  → cwms_msbs hurts on clean data: noise-specific mechanism confirmed")
            else:
                print("  → cwms_msbs neutral on clean data: noise-specific mechanism confirmed")


if __name__ == "__main__":
    main()
