"""Oracle paradox analysis: why CRCC-P outperforms oracle deletion.

Oracle knows exactly which labels are noisy. CRCC-P does not. Yet CRCC-P
achieves higher balanced accuracy. This script analyzes the mechanism:
oracle deletes noisy minority samples whose feature vectors come from the
minority distribution — removing them impoverishes the model's view of the
minority feature space, even though the labels were wrong.

Outputs: outputs/oracle-paradox-analysis.csv
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.core.config import BaseExperimentConfig
from pipeline.core.experiment import (
    _encode_dataframe, _make_lr_factory, _make_hgb_factory,
)
from pipeline.data.loaders import load_dataset, induce_imbalance, inject_noise
from pipeline.scoring.oof_loss import out_of_fold_loss
from pipeline.cleaning.selectors import select_oracle, select_crcc_p
from pipeline.evaluation.metrics import evaluate


@dataclass(frozen=True)
class OracleParadoxConfig(BaseExperimentConfig):
    datasets: tuple[str, ...] = ("pima", "credit-g", "yeast", "ecoli", "phoneme")


def analyze_seed(
    dataset_name: str, model_name: str, seed: int, cfg: OracleParadoxConfig,
) -> dict:
    """Run one combo and return oracle vs CRCC-P comparison data."""
    rng = np.random.default_rng(seed)
    minority_label = cfg.get_minority_label(dataset_name)

    X_raw, y_raw, cat_cols, _ = load_dataset(dataset_name)

    from sklearn.model_selection import train_test_split
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=cfg.test_size, stratify=y_raw, random_state=seed,
    )
    X_train_df = X_train_df.reset_index(drop=True)
    X_test_df = X_test_df.reset_index(drop=True)

    X_train_imb, y_train_imb = induce_imbalance(
        X_train_df, y_train, minority_label=minority_label,
        target_ratio=cfg.target_minority_ratio, rng=rng,
    )

    X_tr, cat_indices = _encode_dataframe(X_train_imb, cat_cols)
    X_te, _ = _encode_dataframe(X_test_df, cat_cols)

    y_noisy, noisy_mask = inject_noise(
        y_train_imb, minority_label=minority_label,
        min_to_maj=cfg.minority_to_majority_noise,
        maj_to_min=cfg.majority_to_minority_noise,
        rng=rng,
    )

    if model_name == "lr":
        model_factory = _make_lr_factory(seed)
    else:
        model_factory = _make_hgb_factory(cat_indices, seed)

    suspiciousness = out_of_fold_loss(
        X_tr, y_noisy, model_factory, n_splits=cfg.n_cv_folds, seed=seed,
    )
    n_samples = len(y_noisy)
    budget_count = max(1, int(round(cfg.cleaning_budget * n_samples)))

    oracle_idx = select_oracle(noisy_mask, budget_count)
    crcc_idx = select_crcc_p(
        suspiciousness, y_noisy, budget_count, lambda_risk=0.5,
        minority_label=minority_label,
    )

    oracle_set = set(oracle_idx.tolist())
    crcc_set = set(crcc_idx.tolist())

    overlap = len(oracle_set & crcc_set)
    oracle_only = list(oracle_set - crcc_set)

    # oracle_only: samples oracle deletes but CRCC-P preserves
    # These are noisy minority samples (true label=minority, flipped to majority)
    # whose feature vectors come from the minority distribution
    oracle_only_true_minority = 0
    if oracle_only:
        oracle_only_true_minority = int(np.sum(y_train_imb[oracle_only] == minority_label))
    oracle_only_minority_frac = (
        oracle_only_true_minority / len(oracle_only) if oracle_only else float("nan")
    )

    # Evaluate both
    n = len(y_noisy)
    keep_oracle = np.ones(n, dtype=bool); keep_oracle[oracle_idx] = False
    keep_crcc = np.ones(n, dtype=bool); keep_crcc[crcc_idx] = False

    if len(np.unique(y_noisy[keep_oracle])) < 2 or len(np.unique(y_noisy[keep_crcc])) < 2:
        return {}

    try:
        metrics_oracle = evaluate(
            oracle_idx, X_tr, y_noisy, y_train_imb, noisy_mask,
            X_te, y_test, model_factory, minority_label=minority_label,
        )
        metrics_crcc = evaluate(
            crcc_idx, X_tr, y_noisy, y_train_imb, noisy_mask,
            X_te, y_test, model_factory, minority_label=minority_label,
        )
    except ValueError:
        return {}

    return {
        "dataset": dataset_name, "model": model_name, "seed": seed,
        "overlap": overlap,
        "oracle_only_count": len(oracle_only),
        "oracle_only_minority_frac": round(oracle_only_minority_frac, 4),
        "crcc_only_count": len(crcc_set - oracle_set),
        "budget_count": budget_count,
        "oracle_ba": round(metrics_oracle["balanced_accuracy"], 4),
        "crcc_ba": round(metrics_crcc["balanced_accuracy"], 4),
        "ba_gain_over_oracle": round(metrics_crcc["balanced_accuracy"] - metrics_oracle["balanced_accuracy"], 4),
        "oracle_minority_recall": round(metrics_oracle["minority_recall"], 4),
        "crcc_minority_recall": round(metrics_crcc["minority_recall"], 4),
    }


def main() -> None:
    cfg = OracleParadoxConfig()
    all_rows: list[dict] = []
    total = len(cfg.datasets) * 2 * len(cfg.seeds)
    n_done = 0

    for dataset in cfg.datasets:
        for model in ("lr", "hgb"):
            for seed in cfg.seeds:
                n_done += 1
                print(f"[{n_done}/{total}] {dataset}/{model}/seed={seed}")
                row = analyze_seed(dataset, model, seed, cfg)
                if row:
                    all_rows.append(row)

    df = pd.DataFrame(all_rows)
    out_path = PROJECT_ROOT / "outputs" / "oracle-paradox-analysis.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df)} rows to {out_path}")

    # Summary
    print("\n── Oracle Paradox Summary ──")
    print(f"Mean oracle_only_minority_frac: {df['oracle_only_minority_frac'].mean():.4f}")
    print(f"Mean ba_gain_over_oracle:      {df['ba_gain_over_oracle'].mean():.4f}")
    wins = (df["ba_gain_over_oracle"] > 0).sum()
    total_combos = len(df)
    print(f"CRCC-P beats oracle in {wins}/{total_combos} combos")

    print("\n── Per-dataset breakdown ──")
    for ds in df["dataset"].unique():
        sub = df[df["dataset"] == ds]
        print(f"  {ds}: minority_frac={sub['oracle_only_minority_frac'].mean():.3f}  "
              f"ba_gain={sub['ba_gain_over_oracle'].mean():.4f}  "
              f"wins={int((sub['ba_gain_over_oracle']>0).sum())}/{len(sub)}")


if __name__ == "__main__":
    main()
