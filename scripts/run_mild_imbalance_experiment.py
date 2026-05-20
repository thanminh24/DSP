"""Mild-imbalance experiment: 70/30 ratio + 20% budget.

Tests whether lambda provides benefit over plain class-proportional caps
when imbalance is less extreme and the budget is larger.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.core.config import BaseExperimentConfig
from pipeline.core.experiment import run_single, print_lambda_sensitivity


@dataclass(frozen=True)
class MildConfig(BaseExperimentConfig):
    """Milder imbalance (70/30) + larger budget (20%)."""
    target_minority_ratio: float = 0.30
    cleaning_budget: float = 0.20


def main() -> None:
    cfg = MildConfig()
    print(f"Mild imbalance experiment: ratio={cfg.target_minority_ratio} budget={cfg.cleaning_budget}")
    print(f"Datasets: {cfg.datasets}  Models: lr, hgb  Seeds: {cfg.seeds}")

    all_results: list[dict] = []
    total = len(cfg.datasets) * 2 * len(cfg.seeds)
    n_done = 0

    for dataset_name in cfg.datasets:
        for model_name in ("lr", "hgb"):
            for seed in cfg.seeds:
                n_done += 1
                print(f"[{n_done}/{total}] {dataset_name} / {model_name} / seed={seed}")
                combo_results = run_single(dataset_name, model_name, seed, cfg)
                all_results.extend(combo_results)

    df = pd.DataFrame(all_results)
    output_dir = PROJECT_ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)

    raw_path = output_dir / "mild-imbalance-results.csv"
    df.to_csv(raw_path, index=False)
    print(f"\nSaved {len(df)} rows to {raw_path}")

    metric_cols = [
        "balanced_accuracy", "macro_f1", "minority_recall",
        "noise_precision_deleted", "clean_minority_deletion_rate",
    ]
    summary = (
        df.groupby(["dataset", "model", "method"])[metric_cols]
        .agg(["mean", "std"])
        .round(4)
    )
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary = summary.reset_index()
    summary_path = output_dir / "mild-imbalance-summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Summary saved to {summary_path}")

    print_lambda_sensitivity(df)

    print("\n── CMDR comparison: global vs CRCC-P (λ=0.5) ──")
    for dataset in df["dataset"].unique():
        for model in df["model"].unique():
            sub = df[(df["dataset"] == dataset) & (df["model"] == model)]
            glob_cmdr = sub[sub["method"] == "global_top_loss"]["clean_minority_deletion_rate"].mean()
            crcc_cmdr = sub[sub["method"] == "crcc_p_l05"]["clean_minority_deletion_rate"].mean()
            reduction = (glob_cmdr - crcc_cmdr) / glob_cmdr * 100 if glob_cmdr > 0 else 0
            print(f"  {dataset:10s}/{model:4s}  global={glob_cmdr:.3f}  crcc_p={crcc_cmdr:.3f}  reduction={reduction:.0f}%")


if __name__ == "__main__":
    main()
