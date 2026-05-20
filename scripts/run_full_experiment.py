"""Full experiment: main CRCC pipeline — 5 datasets × 2 models × 5 seeds × 10 methods."""

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
class Config(BaseExperimentConfig):
    """Full experiment uses all defaults from BaseExperimentConfig."""
    pass


def main() -> None:
    cfg = Config()
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
    output_path = output_dir / "full-experiment-results.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} rows to {output_path}")
    expected = total * len(cfg.method_names)
    print(f"Expected {expected} rows, got {len(df)}")
    print(f"Methods: {sorted(df['method'].unique())}")

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
