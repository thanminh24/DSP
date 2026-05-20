"""Budget ablation: 5 budget levels × 5 datasets × 2 models × 5 seeds × 10 methods.

Produces CMDR-vs-budget curve — key figure showing how cleaning budget
interacts with class protection.
Outputs: outputs/budget-ablation-results.csv + outputs/budget-ablation-summary.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.core.config import BaseExperimentConfig
from pipeline.core.experiment import run_single

BUDGET_LEVELS = [0.05, 0.10, 0.15, 0.20, 0.30]


def main() -> None:
    all_results: list[dict] = []
    for budget in BUDGET_LEVELS:
        cfg = BaseExperimentConfig(cleaning_budget=budget)
        total = len(cfg.datasets) * 2 * len(cfg.seeds)
        n_done = 0
        for dataset in cfg.datasets:
            for model in ("lr", "hgb"):
                for seed in cfg.seeds:
                    n_done += 1
                    print(f"[budget={budget}] [{n_done}/{total}] {dataset}/{model}/seed={seed}")
                    rows = run_single(dataset, model, seed, cfg)
                    for r in rows:
                        r["budget"] = budget
                    all_results.extend(rows)

    df = pd.DataFrame(all_results)
    df.to_csv(PROJECT_ROOT / "outputs" / "budget-ablation-results.csv", index=False)

    metric_cols = ["balanced_accuracy", "macro_f1", "minority_recall",
                   "noise_precision_deleted", "clean_minority_deletion_rate"]
    summary = (df.groupby(["budget", "dataset", "model", "method"])[metric_cols]
               .agg(["mean", "std"]).round(4))
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary.reset_index().to_csv(
        PROJECT_ROOT / "outputs" / "budget-ablation-summary.csv", index=False)

    print(f"\nDone. {len(df)} rows saved.")

    # Sanity: global CMDR should increase with budget
    print("\n── Global top-loss mean CMDR by budget ──")
    check = (df[df["method"] == "global_top_loss"]
             .groupby("budget")["clean_minority_deletion_rate"].mean())
    for budget, cmdr in check.items():
        print(f"  budget={budget:.2f}  CMDR={cmdr:.4f}")

    print("\n── CRCC-P (λ=0.5) mean CMDR by budget ──")
    check_crcc = (df[df["method"] == "crcc_p_l05"]
                  .groupby("budget")["clean_minority_deletion_rate"].mean())
    for budget, cmdr in check_crcc.items():
        print(f"  budget={budget:.2f}  CMDR={cmdr:.4f}")


if __name__ == "__main__":
    main()
