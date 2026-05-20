"""Diagnostic: OOF disagreement rate vs true Type B count.

Measures whether the TBDC cap formula (minority_cap = budget × disagreement_rate)
is accurate enough to route budget to Type B without over-cleaning clean minority.

Two gates checked:
1. Overestimation ratio: oof_disagreement_count / true_type_b_count
   - < 2.0 → cap formula usable; > 2.0 → apply damping
2. FPR-at-depth: at what depth into the sorted minority pool does clean-minority FPR
   first exceed 10%? TBDC cap must stay below this depth.

Run: /home/than-minh/miniconda3/bin/python3 scripts/run_diagnostic_tbdc.py
"""
from __future__ import annotations
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.data.loaders import load_dataset, induce_imbalance, inject_noise
from pipeline.core.config import BaseExperimentConfig

SEEDS = [13, 29, 47, 61, 83]
DATASETS = ["pima", "credit-g", "yeast", "phoneme"]  # ecoli separate


def _oof_scores_and_probs(X, y_noisy, factory, n_splits=5, seed=42):
    """OOF cross-entropy loss AND full probability matrix (diagnostic only)."""
    n = len(y_noisy)
    n_cls = len(np.unique(y_noisy))
    probs = np.zeros((n, n_cls), dtype=float)
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr_idx, va_idx in folds.split(X, y_noisy):
        m = factory()
        m.fit(X[tr_idx], y_noisy[tr_idx])
        probs[va_idx] = m.predict_proba(X[va_idx])
    losses = -np.log(np.clip(probs[np.arange(n), y_noisy], 1e-12, 1.0))
    return losses, probs


def _encode(X_df, cat_cols):
    X = X_df.copy()
    for col in cat_cols:
        if hasattr(X[col], "cat"):
            X[col] = X[col].cat.codes.astype(float)
        else:
            X[col] = X[col].astype("category").cat.codes.astype(float)
    return X.to_numpy(dtype=float)


def _lr_factory(seed):
    def f():
        return Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
        ])
    return f


def _hgb_factory(cat_idx, seed):
    def f():
        return make_pipeline(HistGradientBoostingClassifier(
            categorical_features=cat_idx if cat_idx else None, random_state=seed,
        ))
    return f


def run_diagnostic():
    cfg = BaseExperimentConfig()
    rows = []

    for ds in DATASETS:
        X_raw, y_raw, cat_cols, _ = load_dataset(ds)
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            minority_label = cfg.get_minority_label(ds)
            X_tr_df, _, y_tr, _ = train_test_split(
                X_raw, y_raw, test_size=0.25, stratify=y_raw, random_state=seed,
            )
            X_tr_df = X_tr_df.reset_index(drop=True)
            X_imb, y_imb = induce_imbalance(
                X_tr_df, y_tr, minority_label=minority_label,
                target_ratio=0.15, rng=rng,
            )
            X_np = _encode(X_imb, cat_cols)
            cat_idx = [list(X_imb.columns).index(c) for c in cat_cols]

            y_noisy, noisy_mask = inject_noise(
                y_imb, minority_label=minority_label,
                min_to_maj=0.30, maj_to_min=0.10, rng=rng,
            )

            for model_name in ("lr", "hgb"):
                factory = _lr_factory(seed) if model_name == "lr" else _hgb_factory(cat_idx, seed)
                losses, probs = _oof_scores_and_probs(X_np, y_noisy, factory, seed=seed)

                minority_pool = np.where(y_noisy == minority_label)[0]
                n = len(y_noisy)
                budget_count = max(1, int(round(0.10 * n)))
                minority_freq = len(minority_pool) / n

                # True Type B: samples where y_noisy==minority_label AND noisy_mask==True
                # (these are true majority that got relabeled to minority)
                true_type_b = np.sum(noisy_mask[minority_pool])

                # OOF disagreement count in minority pool
                oof_pred = np.argmax(probs[minority_pool], axis=1)
                oof_disagree = np.sum(oof_pred != minority_label)
                disagree_rate = oof_disagree / max(len(minority_pool), 1)

                tbdc_cap = int(round(budget_count * disagree_rate))
                prop_cap = int(round(budget_count * minority_freq))
                overest = oof_disagree / max(true_type_b, 1)

                # FPR-at-depth: walk sorted minority pool by suspiciousness
                sorted_min = minority_pool[np.argsort(losses[minority_pool])[::-1]]
                is_clean = ~noisy_mask[sorted_min]
                fpr_curve = np.cumsum(is_clean) / (np.arange(len(sorted_min)) + 1)
                exceed_idx = np.where(fpr_curve > 0.10)[0]
                fpr_10pct_depth = (
                    exceed_idx[0] / len(sorted_min) if len(exceed_idx) > 0 else 1.0
                )
                tbdc_cap_frac = tbdc_cap / max(len(minority_pool), 1)
                safe = tbdc_cap_frac <= fpr_10pct_depth

                rows.append({
                    "dataset": ds, "model": model_name, "seed": seed,
                    "n_minority_pool": len(minority_pool),
                    "true_type_b": int(true_type_b),
                    "oof_disagree": int(oof_disagree),
                    "overest_ratio": round(overest, 3),
                    "tbdc_cap": tbdc_cap, "prop_cap": prop_cap,
                    "tbdc_cap_frac": round(tbdc_cap_frac, 3),
                    "fpr_10pct_depth": round(fpr_10pct_depth, 3),
                    "fpr_safe": safe,
                })

    df = pd.DataFrame(rows)
    df.to_csv("outputs/diagnostic-tbdc.csv", index=False)

    print("\n=== TBDC Diagnostic: Overestimation Ratio & FPR Safety ===")
    print(f"{'Dataset':<12} {'Model':<5} {'OverestRatio':>12} {'FPR10Depth':>10} {'TBDCCapFrac':>11} {'FPRSafe':>8}")
    summary = df.groupby(["dataset", "model"]).agg(
        overest_mean=("overest_ratio", "mean"),
        fpr_depth_mean=("fpr_10pct_depth", "mean"),
        tbdc_frac_mean=("tbdc_cap_frac", "mean"),
        fpr_safe_pct=("fpr_safe", "mean"),
    ).reset_index()
    for _, row in summary.iterrows():
        safe_str = f"{row.fpr_safe_pct*100:.0f}%"
        print(f"{row.dataset:<12} {row.model:<5} {row.overest_mean:>12.3f} "
              f"{row.fpr_depth_mean:>10.3f} {row.tbdc_frac_mean:>11.3f} {safe_str:>8}")

    overall_overest = df["overest_ratio"].mean()
    overall_safe = df["fpr_safe"].mean()
    print(f"\nOverall mean overest_ratio: {overall_overest:.3f}")
    print(f"Overall FPR-safe rate: {overall_safe*100:.0f}%")

    if overall_overest < 2.0 and overall_safe >= 0.70:
        print("\n→ GO: TBDC cap formula is usable (overest < 2.0, ≥70% FPR-safe)")
        print("   damping=1.0 (no adjustment needed)")
    elif overall_overest < 2.0:
        rec_damp = round(df["fpr_10pct_depth"].mean() / max(df["tbdc_cap_frac"].mean(), 1e-6), 2)
        rec_damp = min(rec_damp, 1.0)
        print(f"\n→ GO WITH DAMPING: overest OK but FPR-safe <70%. Recommended damping={rec_damp:.2f}")
    else:
        print("\n→ APPLY DAMPING: overest_ratio ≥ 2.0. Use damping=0.7 in Phase 2.")
        print("   minority_cap = round(budget × disagreement_rate × 0.7)")


if __name__ == "__main__":
    run_diagnostic()
