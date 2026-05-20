"""Noise rate ablation: 3 noise levels × 5 datasets × 2 models × 5 seeds × 10 methods.

Tests CRCC robustness across low/medium/high class-dependent noise.
Outputs: outputs/noise-ablation-results.csv + outputs/noise-ablation-summary.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.core.config import BaseExperimentConfig
from pipeline.core.experiment import run_single

NOISE_LEVELS = {
    "low":    {"min_to_maj": 0.10, "maj_to_min": 0.05},
    "medium": {"min_to_maj": 0.20, "maj_to_min": 0.10},
    "high":   {"min_to_maj": 0.40, "maj_to_min": 0.20},
}


def main() -> None:
    all_results: list[dict] = []
    for level_name, noise in NOISE_LEVELS.items():
        cfg = BaseExperimentConfig(
            minority_to_majority_noise=noise["min_to_maj"],
            majority_to_minority_noise=noise["maj_to_min"],
        )
        total = len(cfg.datasets) * 2 * len(cfg.seeds)
        n_done = 0
        for dataset in cfg.datasets:
            for model in ("lr", "hgb"):
                for seed in cfg.seeds:
                    n_done += 1
                    print(f"[{level_name}] [{n_done}/{total}] {dataset}/{model}/seed={seed}")
                    rows = run_single(dataset, model, seed, cfg)
                    for r in rows:
                        r["noise_level"] = level_name
                    all_results.extend(rows)

    df = pd.DataFrame(all_results)
    df.to_csv(PROJECT_ROOT / "outputs" / "noise-ablation-results.csv", index=False)

    metric_cols = ["balanced_accuracy", "macro_f1", "minority_recall",
                   "noise_precision_deleted", "clean_minority_deletion_rate"]
    summary = (df.groupby(["noise_level", "dataset", "model", "method"])[metric_cols]
               .agg(["mean", "std"]).round(4))
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary.reset_index().to_csv(
        PROJECT_ROOT / "outputs" / "noise-ablation-summary.csv", index=False)

    print(f"\nDone. {len(df)} rows saved.")

    # Verify key expectation
    for level in NOISE_LEVELS:
        sub = df[(df["noise_level"] == level) & (df["method"] == "crcc_p_l05")]
        cmdr = sub["clean_minority_deletion_rate"].mean()
        print(f"  {level:6s} crcc_p_l05 mean CMDR={cmdr:.4f}")

    nan_rows = df["balanced_accuracy"].isna().sum()
    if nan_rows > 0:
        print(f"  NaN rows (one-class guard): {nan_rows}")


if __name__ == "__main__":
    main()
