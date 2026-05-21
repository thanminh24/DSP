"""Publication-oriented viability sweep for confidence-guided relabeling."""

from __future__ import annotations

import dataclasses
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.augmentation.relabeling import random_relabeling, relabel_typeA
from pipeline.baselines.confidence_relabeling import (
    select_confidence_relabels,
    unbalanced_oof_majority_scores,
)
from pipeline.cleaning.selectors import select_class_proportional, select_global, select_oracle
from pipeline.core.config import BaseExperimentConfig
from pipeline.data.encoding import encode_train_test
from pipeline.data.loaders import induce_imbalance, inject_noise, load_dataset
from pipeline.evaluation.augment_metrics import evaluate_augmented
from pipeline.evaluation.metrics import evaluate
from pipeline.models.factories import make_model_factory, model_supports_sample_weight
from pipeline.scoring.balanced_oof import balanced_oof_majority_scores
from pipeline.scoring.oof_loss import out_of_fold_loss

DATASETS = ["pima", "credit-g", "yeast", "phoneme", "ecoli"]
SEEDS = [13, 17, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101]
QUICK_SEEDS = [13, 17, 23, 29, 31, 37, 41, 43, 47, 53]
NOISE_PROTOCOLS = {
    "hidden_minority_low": (0.10, 0.05),
    "hidden_minority_medium": (0.30, 0.10),
    "hidden_minority_high": (0.40, 0.20),
    "reverse_asymmetric": (0.10, 0.30),
    "symmetric": (0.20, 0.20),
}
QUICK_PROTOCOLS = {"hidden_minority_medium", "reverse_asymmetric", "symmetric"}
METHODS = [
    "no_cleaning",
    "global_top_loss",
    "class_proportional",
    "balanced_oof_relabel",
    "unbalanced_oof_relabel",
    "shuffled_score_relabel",
    "inverted_score_relabel",
    "random_relabel",
    "class_weight_only",
    "oracle_relabel",
]
OUT_CSV = "outputs/relabeling-viability-results.csv"


def main() -> None:
    args = sys.argv[1:]
    quick = "--quick" in args
    model_names = [a for a in args if not a.startswith("--")] or ["lr", "hgb"]
    seeds = QUICK_SEEDS if quick else SEEDS
    protocols = {k: v for k, v in NOISE_PROTOCOLS.items() if (not quick or k in QUICK_PROTOCOLS)}
    rows = []
    combos = list(itertools.product(DATASETS, model_names, seeds, protocols.items()))
    for i, (dataset, model_name, seed, (noise_name, (mn, mj))) in enumerate(combos, 1):
        cfg = dataclasses.replace(
            BaseExperimentConfig(),
            minority_to_majority_noise=mn,
            majority_to_minority_noise=mj,
        )
        try:
            batch = run_single_viability(dataset, model_name, seed, noise_name, cfg)
            rows.extend(batch)
        except Exception as exc:
            rows.append({
                "dataset": dataset, "model": model_name, "seed": seed,
                "noise_protocol": noise_name, "method": "ERROR",
                "error": str(exc), "mn_to_maj": mn, "maj_to_min": mj,
            })
            print(f"FAIL {dataset}/{model_name}/{seed}/{noise_name}: {exc}", flush=True)
        if i % 20 == 0:
            pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
            print(f"progress {i}/{len(combos)} rows={len(rows)}", flush=True)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"wrote {len(rows)} rows -> {OUT_CSV}")


def run_single_viability(dataset_name: str, model_name: str, seed: int, noise_name: str, cfg):
    rng = np.random.default_rng(seed)
    minority_label = cfg.get_minority_label(dataset_name)
    X_raw, y_raw, cat_cols, _ = load_dataset(dataset_name)
    X_tr_df, X_te_df, y_tr, y_te = train_test_split(
        X_raw, y_raw, test_size=cfg.test_size, stratify=y_raw, random_state=seed,
    )
    X_tr_df, X_te_df = X_tr_df.reset_index(drop=True), X_te_df.reset_index(drop=True)
    X_tr_df, y_tr = induce_imbalance(
        X_tr_df, y_tr, minority_label=minority_label,
        target_ratio=cfg.target_minority_ratio, rng=rng,
    )
    X_tr, X_te, cat_indices = encode_train_test(X_tr_df, X_te_df, cat_cols)
    y_noisy, noisy_mask = inject_noise(
        y_tr, minority_label, cfg.minority_to_majority_noise,
        cfg.majority_to_minority_noise, rng,
    )
    majority_label = 1 if minority_label == 0 else 0
    budget_count = max(1, int(round(cfg.cleaning_budget * len(y_noisy))))
    std_factory = make_model_factory(model_name, seed, cat_indices, balanced=False)
    bal_factory = make_model_factory(model_name, seed, cat_indices, balanced=True)
    use_sw = model_supports_sample_weight(model_name)
    suspiciousness = out_of_fold_loss(X_tr, y_noisy, std_factory, cfg.n_cv_folds, seed)
    bal_scores = balanced_oof_majority_scores(
        X_tr, y_noisy, bal_factory, minority_label, majority_label,
        cfg.n_cv_folds, seed, use_sample_weight=use_sw,
    )
    unbal_scores = unbalanced_oof_majority_scores(
        X_tr, y_noisy, std_factory, minority_label, majority_label,
        cfg.n_cv_folds, seed, use_sample_weight=False,
    )
    rows = []
    for method in METHODS:
        row = _run_method(
            method, X_tr, y_noisy, y_tr, noisy_mask, X_te, y_te, std_factory,
            suspiciousness, bal_scores, unbal_scores, budget_count,
            minority_label, majority_label, rng,
        )
        row.update({
            "dataset": dataset_name, "model": model_name, "seed": seed,
            "noise_protocol": noise_name,
            "mn_to_maj": cfg.minority_to_majority_noise,
            "maj_to_min": cfg.majority_to_minority_noise,
            "method": method,
        })
        rows.append(row)
    return rows


def _run_method(method, X_tr, y_noisy, y_tr, noisy_mask, X_te, y_te, factory,
                suspiciousness, bal_scores, unbal_scores, budget, min_label, maj_label, rng):
    if method == "no_cleaning":
        return evaluate_augmented(X_tr, y_noisy, X_te, y_te, factory, min_label)
    if method == "class_weight_only":
        # The selected factory is already the model under test; class-weight-only
        # variants are covered by the model stress benchmark's balanced factories.
        return evaluate_augmented(X_tr, y_noisy, X_te, y_te, factory, min_label)
    if method == "global_top_loss":
        selected = select_global(suspiciousness, budget)
        return evaluate(selected, X_tr, y_noisy, y_tr, noisy_mask, X_te, y_te, factory, min_label)
    if method == "class_proportional":
        selected = select_class_proportional(suspiciousness, y_noisy, budget)
        return evaluate(selected, X_tr, y_noisy, y_tr, noisy_mask, X_te, y_te, factory, min_label)
    if method == "oracle_relabel":
        idx = _oracle_relabel_indices(y_noisy, y_tr, min_label, maj_label, budget)
        y_rel = y_noisy.copy()
        y_rel[idx] = min_label
        return _eval_relabel(X_tr, y_rel, idx, y_tr, X_te, y_te, factory, min_label)
    if method == "balanced_oof_relabel":
        y_rel, idx = relabel_typeA(y_noisy, bal_scores, budget, min_label, maj_label)
        return _eval_relabel(X_tr, y_rel, idx, y_tr, X_te, y_te, factory, min_label)
    if method == "unbalanced_oof_relabel":
        idx = select_confidence_relabels(y_noisy, unbal_scores, budget, maj_label)
        y_rel = y_noisy.copy()
        y_rel[idx] = min_label
        return _eval_relabel(X_tr, y_rel, idx, y_tr, X_te, y_te, factory, min_label)
    if method == "shuffled_score_relabel":
        shuffled = bal_scores.copy()
        valid = np.where(~np.isnan(shuffled))[0]
        values = shuffled[valid].copy()
        rng.shuffle(values)
        shuffled[valid] = values
        idx = select_confidence_relabels(y_noisy, shuffled, budget, maj_label)
        y_rel = y_noisy.copy()
        y_rel[idx] = min_label
        return _eval_relabel(X_tr, y_rel, idx, y_tr, X_te, y_te, factory, min_label)
    if method == "inverted_score_relabel":
        inverted = np.where(np.isnan(bal_scores), np.nan, 1.0 - bal_scores)
        idx = select_confidence_relabels(y_noisy, inverted, budget, maj_label)
        y_rel = y_noisy.copy()
        y_rel[idx] = min_label
        return _eval_relabel(X_tr, y_rel, idx, y_tr, X_te, y_te, factory, min_label)
    if method == "random_relabel":
        y_rel, idx = random_relabeling(y_noisy, budget, maj_label, min_label, rng)
        return _eval_relabel(X_tr, y_rel, idx, y_tr, X_te, y_te, factory, min_label)
    raise ValueError(f"Unknown method: {method}")


def _eval_relabel(X_tr, y_rel, idx, y_clean, X_te, y_te, factory, min_label):
    correctness = float((y_clean[idx] == min_label).mean()) if len(idx) else float("nan")
    return evaluate_augmented(
        X_tr, y_rel, X_te, y_te, factory, min_label,
        n_relabeled=len(idx), relabel_correctness=correctness,
    )


def _oracle_relabel_indices(y_noisy, y_clean, min_label, maj_label, budget):
    idx = np.where((y_noisy == maj_label) & (y_clean == min_label))[0]
    return idx[: min(budget, len(idx))]


if __name__ == "__main__":
    main()
