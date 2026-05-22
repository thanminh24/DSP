"""Full rigorous sweep for CWMS+MSBS boundary correction method.

Scope: lr, calibrated_lr, hgb (+ xgboost/catboost/lightgbm if installed)
       x 5 datasets x 10 seeds x 3 hidden_minority protocols x 5 methods
Output: outputs/cwms-msbs-full-sweep.csv  (resume-safe)
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
from pipeline.models.factories import list_publication_models

FULL_METHODS = ["no_cleaning", "class_proportional", "msbs", "cwms", "cwms_msbs"]
FULL_SEEDS = QUICK_SEEDS
FULL_PROTOCOLS = ["hidden_minority_medium", "hidden_minority_low", "hidden_minority_high"]
EXCLUDED_MODELS = {"random_forest", "extra_trees"}
POC_BUDGET = 0.10
POC_RATIO = 0.15
OUTPUT_CSV = PROJECT_ROOT / "outputs" / "cwms-msbs-full-sweep.csv"


def _load_completed(path: Path) -> set:
    if not path.exists():
        return set()
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"WARNING: could not read {path}, restarting from scratch ({exc})", flush=True)
        return set()
    # Exclude error rows so failed combos are retried on resume
    if 'error' in df.columns:
        df = df[df['error'].isna()]
    return {
        (r["dataset"], r["model"], int(r["seed"]), r["noise_protocol"],
         float(r["budget"]), float(r["target_ratio"]), r["method"])
        for _, r in df.iterrows()
    }


def main():
    args = sys.argv[1:]
    use_gpu = "--gpu" in args
    medium_only = "--medium-only" in args
    protocols = ["hidden_minority_medium"] if medium_only else FULL_PROTOCOLS

    models = [m for m in list_publication_models() if m not in EXCLUDED_MODELS]
    completed = _load_completed(OUTPUT_CSV)
    total_written = 0

    for proto_name in protocols:
        mn, mj = NOISE_PROTOCOLS[proto_name]
        for model_name in models:
            for dataset in DATASETS:
                for seed in FULL_SEEDS:
                    needed = [
                        m for m in FULL_METHODS
                        if (dataset, model_name, seed, proto_name,
                            float(POC_BUDGET), float(POC_RATIO), m) not in completed
                    ]
                    if not needed:
                        continue
                    try:
                        batch = run_single_viability(
                            dataset, model_name, seed, proto_name, mn, mj,
                            POC_BUDGET, POC_RATIO,
                            use_gpu=use_gpu, methods=needed,
                        )
                    except Exception as exc:
                        batch = [
                            {
                                "dataset": dataset, "model": model_name, "seed": seed,
                                "noise_protocol": proto_name, "method": m,
                                "error": str(exc), "mn_to_maj": mn, "maj_to_min": mj,
                                "budget": POC_BUDGET, "target_ratio": POC_RATIO,
                            }
                            for m in needed
                        ]
                        print(f"FAIL {dataset}/{model_name}/{seed}/{proto_name}: {exc}", flush=True)

                    df_batch = pd.DataFrame(batch)
                    write_header = not OUTPUT_CSV.exists() or OUTPUT_CSV.stat().st_size == 0
                    df_batch.to_csv(OUTPUT_CSV, mode="a", header=write_header, index=False)
                    total_written += len(batch)
                    print(
                        f"  {proto_name}/{dataset}/{model_name}/{seed}: {needed} -> {len(batch)} rows",
                        flush=True,
                    )

    print(f"\nDone. {total_written} rows -> {OUTPUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
