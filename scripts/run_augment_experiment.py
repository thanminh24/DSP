"""Experiment runner for augmentation methods (relabel + SMOTE).

Provides run_single_augment() — same shape as pipeline.core.experiment.run_single
but branches into relabeling, SMOTE, and deletion methods.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from pipeline.core.config import BaseExperimentConfig
from pipeline.data.encoding import encode_train_test
from pipeline.data.loaders import load_dataset, induce_imbalance, inject_noise
from pipeline.scoring.oof_loss import out_of_fold_loss
from pipeline.scoring.balanced_oof import balanced_oof_majority_scores
from pipeline.cleaning.selectors import select_class_proportional
from pipeline.augmentation.relabeling import relabel_typeA, random_relabeling
from pipeline.augmentation.filtered_smote import filtered_smote_augment, plain_smote_augment
from pipeline.evaluation.metrics import evaluate
from pipeline.evaluation.augment_metrics import evaluate_augmented


def _make_std_factory(model_name, cat_indices, seed):
    if model_name == "lr":
        def factory():
            return Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
            ])
    else:
        def factory():
            return make_pipeline(
                HistGradientBoostingClassifier(
                    categorical_features=cat_indices if cat_indices else None,
                    random_state=seed,
                )
            )
    return factory


def _make_bal_factory(model_name, cat_indices, seed):
    if model_name == "lr":
        def factory():
            return Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)),
            ])
    else:
        def factory():
            return make_pipeline(
                HistGradientBoostingClassifier(
                    categorical_features=cat_indices if cat_indices else None,
                    random_state=seed,
                )
            )
    return factory


def _tag(row, dataset_name, model_name, seed, method):
    row["dataset"] = dataset_name
    row["model"] = model_name
    row["seed"] = seed
    row["method"] = method
    return row


def run_single_augment(
    dataset_name: str,
    model_name: str,
    seed: int,
    methods: list[str],
    cfg: BaseExperimentConfig,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    minority_label = cfg.get_minority_label(dataset_name)

    X_raw, y_raw, cat_cols, _ = load_dataset(dataset_name)
    X_tr_df, X_te_df, y_tr, y_te = train_test_split(
        X_raw, y_raw, test_size=cfg.test_size, stratify=y_raw, random_state=seed,
    )
    X_tr_df = X_tr_df.reset_index(drop=True)
    X_te_df = X_te_df.reset_index(drop=True)

    X_tr_df, y_tr = induce_imbalance(
        X_tr_df, y_tr, minority_label=minority_label,
        target_ratio=cfg.target_minority_ratio, rng=rng,
    )

    X_tr, X_te, cat_indices = encode_train_test(X_tr_df, X_te_df, cat_cols)

    y_noisy, noisy_mask = inject_noise(
        y_tr, minority_label=minority_label,
        min_to_maj=cfg.minority_to_majority_noise,
        maj_to_min=cfg.majority_to_minority_noise,
        rng=rng,
    )

    std_factory = _make_std_factory(model_name, cat_indices, seed)
    bal_factory = _make_bal_factory(model_name, cat_indices, seed)
    use_sw = model_name == "hgb"

    susp = out_of_fold_loss(X_tr, y_noisy, std_factory, n_splits=cfg.n_cv_folds, seed=seed)
    n = len(y_noisy)
    budget_count = max(1, int(round(cfg.cleaning_budget * n)))
    majority_label = 1 if minority_label == 0 else 0

    results = []
    for method in methods:
        if method == "no_cleaning":
            row = evaluate_augmented(
                X_tr, y_noisy, X_te, y_te, std_factory, minority_label,
            )
            results.append(_tag(row, dataset_name, model_name, seed, method))
        elif method == "class_proportional":
            sel = select_class_proportional(susp, y_noisy, budget_count)
            row = evaluate(
                sel, X_tr, y_noisy, y_tr, noisy_mask,
                X_te, y_te, std_factory, minority_label=minority_label,
            )
            row["n_relabeled"] = 0
            row["n_synthetic"] = 0
            row["relabel_correctness"] = float("nan")
            results.append(_tag(row, dataset_name, model_name, seed, method))
        elif method == "balanced_oof_relabel":
            scores = balanced_oof_majority_scores(
                X_tr, y_noisy, bal_factory, minority_label, majority_label,
                n_splits=cfg.n_cv_folds, seed=seed, use_sample_weight=use_sw,
            )
            y_rel, rel_idx = relabel_typeA(
                y_noisy, scores, budget_count, minority_label, majority_label,
            )
            correctness = (
                float((y_tr[rel_idx] == minority_label).mean())
                if len(rel_idx) > 0 else float("nan")
            )
            row = evaluate_augmented(
                X_tr, y_rel, X_te, y_te, std_factory, minority_label,
                n_relabeled=len(rel_idx), relabel_correctness=correctness,
            )
            results.append(_tag(row, dataset_name, model_name, seed, method))
        elif method == "random_relabel":
            y_rel, rel_idx = random_relabeling(
                y_noisy, budget_count, majority_label, minority_label, rng,
            )
            correctness = (
                float((y_tr[rel_idx] == minority_label).mean())
                if len(rel_idx) > 0 else float("nan")
            )
            row = evaluate_augmented(
                X_tr, y_rel, X_te, y_te, std_factory, minority_label,
                n_relabeled=len(rel_idx), relabel_correctness=correctness,
            )
            results.append(_tag(row, dataset_name, model_name, seed, method))
        elif method == "oof_filtered_smote":
            X_aug, y_aug, sinfo = filtered_smote_augment(
                X_tr, y_noisy, susp, minority_label, majority_label,
                filter_frac=0.05, seed=seed,
            )
            row = evaluate_augmented(
                X_aug, y_aug, X_te, y_te, std_factory, minority_label,
                n_synthetic=sinfo["n_synthetic"],
            )
            results.append(_tag(row, dataset_name, model_name, seed, method))
        elif method == "plain_smote":
            X_aug, y_aug, sinfo = plain_smote_augment(
                X_tr, y_noisy, minority_label, seed=seed,
            )
            row = evaluate_augmented(
                X_aug, y_aug, X_te, y_te, std_factory, minority_label,
                n_synthetic=sinfo["n_synthetic"],
            )
            results.append(_tag(row, dataset_name, model_name, seed, method))
        else:
            raise ValueError(f"Unknown method: {method}")
    return results
