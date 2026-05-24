"""Competitor head-to-head: LR x competitor methods x medium noise.

Run with: python scripts/run_competitor_headtohead.py
GPU flag not needed — LR is CPU-only.
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
from scripts._sweep_utils import load_completed

COMPETITOR_METHODS = [
    "no_cleaning",
    "class_proportional",
    "smote",
    "iw_smote",
    "sw_framework",
    "cwms_msbs",
]
COMPETITOR_MODELS = ["lr"]
COMPETITOR_PROTOCOLS = ["hidden_minority_medium"]
POC_BUDGET, POC_RATIO = 0.10, 0.15
OUTPUT_CSV = PROJECT_ROOT / "outputs" / "competitor-headtohead.csv"


def main():
    args = sys.argv[1:]
    use_gpu = "--gpu" in args
    if use_gpu:
        print("Note: LR is CPU-only — GPU flag ignored for this sweep.", flush=True)

    completed = load_completed(OUTPUT_CSV)
    total_written = 0

    for proto_name in COMPETITOR_PROTOCOLS:
        mn, mj = NOISE_PROTOCOLS[proto_name]
        for model_name in COMPETITOR_MODELS:
            for dataset in DATASETS:
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
                    print(f"  {dataset}/{seed}: {needed}", flush=True)

    print(f"\nDone. {total_written} rows -> {OUTPUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
