"""Experiment orchestration — run_single and shared helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from pipeline.core.config import BaseExperimentConfig, LAMBDA_NAMES
from pipeline.data.encoding import encode_train_test
from pipeline.data.loaders import load_dataset, induce_imbalance, inject_noise
from pipeline.scoring.oof_loss import out_of_fold_loss
from pipeline.cleaning.selectors import (
    select_none,
    select_random,
    select_global,
    select_class_proportional,
    select_oracle,
    select_crcc_p,
    select_crcc_m,
)
from pipeline.evaluation.metrics import evaluate


def _make_lr_factory(seed: int):
    """Logistic regression pipeline factory."""
    def factory():
        return Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
        ])
    return factory


def _make_hgb_factory(cat_indices: list[int], seed: int):
    """HistGradientBoosting pipeline factory."""
    def factory():
        return make_pipeline(
            HistGradientBoostingClassifier(
                categorical_features=cat_indices if cat_indices else None,
                random_state=seed,
            )
        )
    return factory


def _nan_metrics_dict(n_deleted: int) -> dict:
    """Return NaN-filled metrics when evaluation is not possible."""
    return {
        "deleted": n_deleted,
        "balanced_accuracy": float("nan"),
        "macro_f1": float("nan"),
        "minority_recall": float("nan"),
        "noise_precision_deleted": float("nan"),
        "clean_minority_deletion_rate": float("nan"),
    }


def run_single(
    dataset_name: str,
    model_name: str,
    seed: int,
    cfg: BaseExperimentConfig,
) -> list[dict]:
    """Run one experiment combo: dataset × model × seed × all methods.

    Returns one dict per cleaning method with metrics.
    """
    rng = np.random.default_rng(seed)
    minority_label = cfg.get_minority_label(dataset_name)

    X_raw, y_raw, cat_cols, _ = load_dataset(dataset_name)

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=cfg.test_size, stratify=y_raw, random_state=seed,
    )
    X_train_df = X_train_df.reset_index(drop=True)
    X_test_df = X_test_df.reset_index(drop=True)

    X_train_imb, y_train_imb = induce_imbalance(
        X_train_df, y_train, minority_label=minority_label,
        target_ratio=cfg.target_minority_ratio, rng=rng,
    )

    X_tr, X_te, cat_indices = encode_train_test(X_train_imb, X_test_df, cat_cols)

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

    selectors_map = {
        "no_cleaning": lambda: select_none(n_samples),
        "random_deletion": lambda: select_random(n_samples, budget_count, rng),
        "global_top_loss": lambda: select_global(suspiciousness, budget_count),
        "class_proportional": lambda: select_class_proportional(
            suspiciousness, y_noisy, budget_count,
        ),
        "oracle_deletion": lambda: select_oracle(noisy_mask, budget_count),
    }
    for lam in cfg.lambda_grid:
        selectors_map[LAMBDA_NAMES[lam]] = lambda lam=lam: select_crcc_p(
            suspiciousness, y_noisy, budget_count, lam,
            minority_label=minority_label,
        )
    selectors_map["crcc_m"] = lambda: select_crcc_m(
        suspiciousness, y_noisy, budget_count, lambda_risk=0.5,
        minority_label=minority_label,
        minority_cap_factor=cfg.minority_cap_factor_m,
    )

    results = []
    for method_name in cfg.method_names:
        selected = selectors_map[method_name]()

        # One-class guard: after deletion, training set must have both classes
        keep_mask = np.ones(n_samples, dtype=bool)
        keep_mask[selected] = False
        if len(np.unique(y_noisy[keep_mask])) < 2:
            metrics = _nan_metrics_dict(int(len(selected)))
        else:
            try:
                metrics = evaluate(
                    selected, X_tr, y_noisy, y_train_imb, noisy_mask,
                    X_te, y_test, model_factory, minority_label=minority_label,
                )
            except ValueError:
                metrics = _nan_metrics_dict(int(len(selected)))

        metrics["dataset"] = dataset_name
        metrics["model"] = model_name
        metrics["seed"] = seed
        metrics["method"] = method_name
        results.append(metrics)

    return results


def print_lambda_sensitivity(df: pd.DataFrame) -> None:
    """Report whether different lambda values produce different CMDR/recall."""
    print("\n── Lambda sensitivity check ──")
    lambda_methods = ["crcc_p_l0", "crcc_p_l025", "crcc_p_l05", "crcc_p_l10"]
    for dataset in df["dataset"].unique():
        for model in df["model"].unique():
            subset = df[(df["dataset"] == dataset) & (df["model"] == model)]
            cmdrs = []
            recalls = []
            for m in lambda_methods:
                rows = subset[subset["method"] == m]
                if rows.empty:
                    continue
                cmdrs.append(rows["clean_minority_deletion_rate"].mean())
                recalls.append(rows["minority_recall"].mean())
            cmdr_range = max(cmdrs) - min(cmdrs) if cmdrs else 0
            recall_range = max(recalls) - min(recalls) if recalls else 0
            sensitive = cmdr_range > 0.01 or recall_range > 0.01
            flag = "LAMBDA MATTERS" if sensitive else "lambda flat"
            print(f"  {dataset:10s} / {model:4s}  CMDR range={cmdr_range:.4f}  recall range={recall_range:.4f}  → {flag}")
