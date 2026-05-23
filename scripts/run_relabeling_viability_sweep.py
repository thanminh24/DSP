"""Publication-oriented viability sweep for confidence-guided relabeling.

Incremental: appends rows on each combo, resumes from checkpoint on restart.
One process per output file; never share a CSV between concurrent processes.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.augmentation.relabeling import random_relabeling, relabel_typeA
from pipeline.augmentation.msbs import minority_side_boundary_synthesis
from pipeline.augmentation.synthesis import confidence_guided_synthesis
from pipeline.baselines.soft_weighting import (
    confidence_weighted_sample_weights,
    confidence_weighted_sample_weights_balanced,
)
from pipeline.baselines.iw_smote import iw_smote
from pipeline.baselines.sw_framework import sw_framework_oversample
from pipeline.baselines.cleanlab_baselines import select_cleanlab_filter
from pipeline.baselines.confidence_relabeling import (
    naive_confidence_majority_scores,
    select_confidence_relabels,
    unbalanced_oof_majority_scores,
)
from pipeline.cleaning.selectors import select_class_proportional, select_global, select_oracle
from pipeline.core.config import BaseExperimentConfig
from pipeline.data.encoding import encode_train_test
from pipeline.data.loaders import induce_imbalance, inject_noise, load_dataset
from pipeline.evaluation.augment_metrics import evaluate_augmented
from pipeline.evaluation.metrics import evaluate
from pipeline.models.factories import (
    make_cwms_factory,
    make_model_factory,
    model_supports_sample_weight,
)
from pipeline.scoring.balanced_oof import balanced_oof_majority_scores
from pipeline.scoring.knn_ratio import knn_ratio_majority_scores
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
BUDGETS = [0.05, 0.10, 0.20]
QUICK_BUDGETS = [0.10]
IMBALANCE_RATIOS = [0.15, 0.30]
QUICK_RATIOS = [0.15]
METHODS = [
    # === PAPER METHODS (primary contribution) ===
    "cwms_msbs",        # NoiSyn: confidence-weighted majority suppression + minority-side boundary synthesis
    "cwms_msbs_shuffled",  # shuffled-score ablation: proves OOF scores are load-bearing
    "msbs",             # MSBS standalone (ablation)
    "cwms",             # CWMS standalone (ablation)

    # === OVERSAMPLING BASELINES ===
    "smote",            # noise-unaware SMOTE (Chawla 2002) — universal paper baseline

    # === BASELINES (deletion-based) ===
    "no_cleaning",      # train on noisy data, no correction
    "class_proportional",  # delete top-budget by loss, class-proportional allocation

    # === HISTORICAL COMPARISON (not in paper main results) ===
    "class_weight_only",
    "global_top_loss",
    "oracle_relabel",
    "cleanlab_filter",
    "cleanlab_relabel",
    "unbalanced_oof_relabel",
    "naive_confidence_relabel",
    "shuffled_score_relabel",
    "inverted_score_relabel",
    "random_relabel",
    # "balanced_oof_relabel",  # DISCOURAGED: OOF circularity
    "cgms_t03",
    "cgms_t05",
    "cgms_t07",
]
out_csv_TEMPLATE = str(PROJECT_ROOT / "outputs/relabeling-viability-{models}.csv")


def _combo_key(dataset, model_name, seed, noise_name, budget, ratio):
    return (dataset, model_name, seed, noise_name, budget, ratio)


def _load_completed(out_csv: str) -> set:
    path = Path(out_csv)
    if not path.exists():
        return set()
    try:
        existing = pd.read_csv(path)
    except Exception:
        return set()
    completed = set()
    for _, r in existing.iterrows():
        completed.add(_combo_key(
            r["dataset"], r["model"], int(r["seed"]),
            r["noise_protocol"], float(r["budget"]), float(r["target_ratio"]),
        ))
    return completed


def main() -> None:
    args = sys.argv[1:]
    quick = "--quick" in args
    use_gpu = "--gpu" in args
    model_names = [a for a in args if not a.startswith("--")] or ["lr", "hgb"]
    model_key = "-".join(sorted(model_names))
    seeds = QUICK_SEEDS if quick else SEEDS
    protocols = {k: v for k, v in NOISE_PROTOCOLS.items() if (not quick or k in QUICK_PROTOCOLS)}
    budgets = QUICK_BUDGETS if quick else BUDGETS
    ratios = QUICK_RATIOS if quick else IMBALANCE_RATIOS
    out_csv = out_csv_TEMPLATE.format(models=model_key)

    combos = list(itertools.product(DATASETS, model_names, seeds, protocols.items(), budgets, ratios))
    completed = _load_completed(out_csv)
    remaining = [c for c in combos if _combo_key(c[0], c[1], c[2], c[3][0], c[4], c[5]) not in completed]
    n_total = len(remaining)
    n_done = len(completed)

    if n_total == 0:
        print(f"All {n_done} combos already completed in {out_csv}")
        return
    print(f"Resuming: {n_done} done, {n_total} remaining (GPU={'on' if use_gpu else 'off'})", flush=True)

    for i, (dataset, model_name, seed, (noise_name, (mn, mj)), budget, ratio) in enumerate(remaining, 1):
        try:
            batch = run_single_viability(
                dataset, model_name, seed, noise_name, mn, mj, budget, ratio, use_gpu=use_gpu,
            )
        except Exception as exc:
            batch = [{
                "dataset": dataset, "model": model_name, "seed": seed,
                "noise_protocol": noise_name, "method": "ERROR",
                "error": str(exc), "mn_to_maj": mn, "maj_to_min": mj,
                "budget": budget, "target_ratio": ratio,
            }]
            print(f"FAIL {dataset}/{model_name}/{seed}/{noise_name}/b{budget}/r{ratio}: {exc}", flush=True)

        df_batch = pd.DataFrame(batch)
        write_header = not Path(out_csv).exists()
        df_batch.to_csv(out_csv, mode="a", header=write_header, index=False)

        if i % 50 == 0:
            print(f"progress {i}/{n_total} (total done: {n_done + i}/{n_done + n_total})", flush=True)

    print(f"wrote {n_total} new combos -> {out_csv}", flush=True)


def run_single_viability(dataset_name, model_name, seed, noise_name, mn, mj, budget, ratio,
                         use_gpu=False, methods=None):
    rng = np.random.default_rng(seed)
    minority_label = 1
    X_raw, y_raw, cat_cols, _ = load_dataset(dataset_name)
    vals, counts = np.unique(y_raw, return_counts=True)
    minority_label = vals[np.argmin(counts)].item()
    X_tr_df, X_te_df, y_tr, y_te = train_test_split(
        X_raw, y_raw, test_size=0.25, stratify=y_raw, random_state=seed,
    )
    X_tr_df, X_te_df = X_tr_df.reset_index(drop=True), X_te_df.reset_index(drop=True)
    X_tr_df, y_tr = induce_imbalance(
        X_tr_df, y_tr, minority_label=minority_label,
        target_ratio=ratio, rng=rng,
    )
    X_tr, X_te, cat_indices = encode_train_test(X_tr_df, X_te_df, cat_cols)
    y_noisy, noisy_mask = inject_noise(
        y_tr, minority_label, mn, mj, rng,
    )
    majority_label = 1 if minority_label == 0 else 0
    budget_count = max(1, int(round(budget * len(y_noisy))))

    # Dynamic scale_pos_weight: majority / minority ratio from noisy training labels
    _vals, _counts = np.unique(y_noisy, return_counts=True)
    spw = float(max(_counts)) / float(min(_counts))
    is_boosting = model_name in ("xgboost", "lightgbm", "catboost")
    spw_val = spw if is_boosting else 1.0

    std_factory = make_model_factory(model_name, seed, cat_indices, balanced=False,
                                     scale_pos_weight=spw_val, use_gpu=use_gpu)
    bal_factory = make_model_factory(model_name, seed, cat_indices, balanced=True,
                                     scale_pos_weight=spw_val, use_gpu=use_gpu)
    cwms_factory = make_cwms_factory(model_name, seed, cat_indices, use_gpu=use_gpu)
    use_sw = model_supports_sample_weight(model_name)

    methods_to_run = set(methods or METHODS)

    suspiciousness = out_of_fold_loss(X_tr, y_noisy, std_factory, 5, seed)
    bal_scores = balanced_oof_majority_scores(
        X_tr, y_noisy, bal_factory, minority_label, majority_label, 5, seed, use_sample_weight=use_sw,
    )
    unbal_scores = (
        unbalanced_oof_majority_scores(
            X_tr, y_noisy, std_factory, minority_label, majority_label, 5, seed, use_sample_weight=False,
        )
        if any(m in methods_to_run for m in ["unbalanced_oof_relabel", "shuffled_score_relabel",
                                              "inverted_score_relabel"])
        else None
    )
    # Full-data (non-OOF) balanced scores for confirmation-bias baseline
    naive_scores = (
        naive_confidence_majority_scores(
            X_tr, y_noisy, bal_factory, minority_label, majority_label, seed, use_sw,
        )
        if "naive_confidence_relabel" in methods_to_run else None
    )

    # k-NN ratio scores for scorer comparison (lazy: only when needed)
    knn_scores = (
        knn_ratio_majority_scores(X_tr, y_noisy, minority_label, majority_label, k=5)
        if any(m.startswith("cwms_msbs_knn") for m in methods_to_run)
        else None
    )

    # Cross-family OOF scores: HGB scorer → any final model (lazy: only when needed)
    crossfamily_scores = None
    if any(m.startswith("cwms_msbs_crossfamily") for m in methods_to_run):
        hgb_factory = make_model_factory("hgb", seed, cat_indices, balanced=True, use_gpu=use_gpu)
        crossfamily_scores = balanced_oof_majority_scores(
            X_tr, y_noisy, hgb_factory, minority_label, majority_label, 5, seed,
            use_sample_weight=False,
        )

    rows = []
    for method in (methods or METHODS):
        row = _run_method(
            method, X_tr, y_noisy, y_tr, noisy_mask, X_te, y_te, std_factory,
            suspiciousness, bal_scores, unbal_scores, naive_scores, budget_count,
            minority_label, majority_label, rng, seed,
            bal_factory=bal_factory, use_sw=use_sw,
            model_name=model_name, cwms_factory=cwms_factory, spw=spw,
            knn_scores=knn_scores, crossfamily_scores=crossfamily_scores,
        )
        row.update({
            "dataset": dataset_name, "model": model_name, "seed": seed,
            "noise_protocol": noise_name,
            "mn_to_maj": mn, "maj_to_min": mj,
            "budget": budget, "target_ratio": ratio,
            "method": method,
        })
        rows.append(row)
    return rows


def _run_method(method, X_tr, y_noisy, y_tr, noisy_mask, X_te, y_te, factory,
                suspiciousness, bal_scores, unbal_scores, naive_scores, budget, min_label, maj_label, rng, seed,
                bal_factory=None, use_sw=False, model_name=None, cwms_factory=None, spw=1.0,
                knn_scores=None, crossfamily_scores=None):
    if method == "no_cleaning":
        return evaluate_augmented(X_tr, y_noisy, X_te, y_te, factory, min_label)
    if method == "class_weight_only":
        # Train the balanced variant of the model (class_weight="balanced" for sklearn,
        # sample_weight for HGB/XGBoost) on the noisy data without any cleaning.
        # This isolates the contribution of class-weighting alone vs. our relabeling.
        sw = compute_sample_weight("balanced", y_noisy) if use_sw else None
        return evaluate_augmented(X_tr, y_noisy, X_te, y_te, bal_factory or factory, min_label,
                                   sample_weight=sw)
    if method == "global_top_loss":
        selected = select_global(suspiciousness, budget)
        return evaluate(selected, X_tr, y_noisy, y_tr, noisy_mask, X_te, y_te, factory, min_label)
    if method == "class_proportional":
        selected = select_class_proportional(suspiciousness, y_noisy, budget)
        return evaluate(selected, X_tr, y_noisy, y_tr, noisy_mask, X_te, y_te, factory, min_label)
    if method == "cleanlab_filter":
        return _run_cleanlab_filter(X_tr, y_noisy, y_tr, noisy_mask, X_te, y_te, factory, min_label, budget)
    if method == "oracle_relabel":
        idx = _oracle_relabel_indices(y_noisy, y_tr, min_label, maj_label, budget)
        y_rel = y_noisy.copy()
        y_rel[idx] = min_label
        return _eval_relabel(X_tr, y_rel, idx, X_te, y_te, factory, min_label, y_clean=y_tr)
    if method == "balanced_oof_relabel":
        y_rel, idx = relabel_typeA(y_noisy, bal_scores, budget, min_label, maj_label)
        return _eval_relabel(X_tr, y_rel, idx, X_te, y_te, factory, min_label, y_clean=y_tr)
    if method == "naive_confidence_relabel":
        y_rel, idx = relabel_typeA(y_noisy, naive_scores, budget, min_label, maj_label)
        return _eval_relabel(X_tr, y_rel, idx, X_te, y_te, factory, min_label, y_clean=y_tr)
    if method == "cleanlab_relabel":
        return _run_cleanlab_relabel(X_tr, y_noisy, y_tr, noisy_mask, X_te, y_te, factory, min_label, maj_label, budget)
    if method == "unbalanced_oof_relabel":
        idx = select_confidence_relabels(y_noisy, unbal_scores, budget, maj_label)
        y_rel = y_noisy.copy()
        y_rel[idx] = min_label
        return _eval_relabel(X_tr, y_rel, idx, X_te, y_te, factory, min_label, y_clean=y_tr)
    if method == "shuffled_score_relabel":
        shuffled = bal_scores.copy()
        valid = np.where(~np.isnan(shuffled))[0]
        values = shuffled[valid].copy()
        rng.shuffle(values)
        shuffled[valid] = values
        idx = select_confidence_relabels(y_noisy, shuffled, budget, maj_label)
        y_rel = y_noisy.copy()
        y_rel[idx] = min_label
        return _eval_relabel(X_tr, y_rel, idx, X_te, y_te, factory, min_label, y_clean=y_tr)
    if method == "inverted_score_relabel":
        inverted = np.where(np.isnan(bal_scores), np.nan, 1.0 - bal_scores)
        idx = select_confidence_relabels(y_noisy, inverted, budget, maj_label)
        y_rel = y_noisy.copy()
        y_rel[idx] = min_label
        return _eval_relabel(X_tr, y_rel, idx, X_te, y_te, factory, min_label, y_clean=y_tr)
    if method == "random_relabel":
        y_rel, idx = random_relabeling(y_noisy, budget, maj_label, min_label, rng)
        return _eval_relabel(X_tr, y_rel, idx, X_te, y_te, factory, min_label, y_clean=y_tr)
    if method in ("cgms_t03", "cgms_t05", "cgms_t07"):
        threshold_map = {"cgms_t03": 0.3, "cgms_t05": 0.5, "cgms_t07": 0.7}
        tau = threshold_map[method]
        X_aug, y_aug, n_synth, n_seeds = confidence_guided_synthesis(
            X_tr, y_noisy, bal_scores, budget, min_label, maj_label,
            threshold=tau, seed=seed,
        )
        return evaluate_augmented(X_aug, y_aug, X_te, y_te, factory,
                                   min_label, n_synthetic=n_synth,
                                   relabel_correctness=float("nan"))
    if method == "msbs":
        X_aug, y_aug, n_synth = minority_side_boundary_synthesis(
            X_tr, y_noisy, bal_scores, budget, min_label, maj_label, seed=seed,
        )
        return evaluate_augmented(X_aug, y_aug, X_te, y_te, factory,
                                   min_label, n_synthetic=n_synth,
                                   relabel_correctness=float("nan"))
    if method == "cwms":
        # XGBoost: scale_pos_weight and CWMS sample_weight create conflicting imbalance
        # corrections even with scale_pos_weight=1.0 in cwms_factory. Excluded from paper.
        if model_name in ("calibrated_lr", "xgboost"):
            return _nan_skip_row()
        use_balanced = model_name in ("lightgbm", "catboost", "hgb")
        if use_balanced:
            sw = confidence_weighted_sample_weights_balanced(
                y_noisy, bal_scores, maj_label, scale_pos_weight=spw,
            )
            return evaluate_augmented(X_tr, y_noisy, X_te, y_te,
                                       cwms_factory or factory, min_label, sample_weight=sw)
        else:
            sw = confidence_weighted_sample_weights(y_noisy, bal_scores, maj_label)
            return evaluate_augmented(X_tr, y_noisy, X_te, y_te,
                                       factory, min_label, sample_weight=sw)
    if method in ("cwms_msbs", "cwms_msbs_shuffled"):
        if model_name in ("calibrated_lr", "xgboost"):
            return _nan_skip_row()
        scores_for_method = bal_scores.copy()
        if method == "cwms_msbs_shuffled":
            valid = ~np.isnan(scores_for_method)
            scores_for_method[valid] = rng.permutation(scores_for_method[valid])
        X_aug, y_aug, n_synth = minority_side_boundary_synthesis(
            X_tr, y_noisy, scores_for_method, budget, min_label, maj_label, seed=seed,
        )
        use_balanced = model_name in ("lightgbm", "catboost", "hgb")
        if use_balanced:
            # Boosting models: fold spw so CWMS and class correction don't double-apply.
            sw_orig = confidence_weighted_sample_weights_balanced(
                y_noisy, scores_for_method, maj_label, spw,
            )
            sw_synth = np.full(n_synth, float(spw))
            fact = cwms_factory or factory
        else:
            # LR: minority weight=1.0; CWMS majority suppression provides sufficient correction.
            # Adding spw to minority over-corrects (confirmed by smoke test: -1.5pp BA).
            sw_orig = confidence_weighted_sample_weights(y_noisy, scores_for_method, maj_label)
            sw_synth = np.ones(n_synth, dtype=float)
            fact = factory
        sw_combined = np.concatenate([sw_orig, sw_synth])
        return evaluate_augmented(X_aug, y_aug, X_te, y_te, fact,
                                   min_label, n_synthetic=n_synth,
                                   relabel_correctness=float("nan"),
                                   sample_weight=sw_combined)
    if method == "smote":
        from imblearn.over_sampling import SMOTE as ImbSMOTE
        n_min = int(np.sum(y_noisy == min_label))
        n_maj = int(np.sum(y_noisy == maj_label))
        target = min(n_min + budget, n_maj)
        if n_min < 2 or target <= n_min:
            return evaluate_augmented(X_tr, y_noisy, X_te, y_te, factory, min_label,
                                       n_synthetic=0, relabel_correctness=float("nan"))
        try:
            smote = ImbSMOTE(
                sampling_strategy={min_label: target},
                k_neighbors=min(5, n_min - 1),
                random_state=seed,
            )
            X_aug, y_aug = smote.fit_resample(X_tr, y_noisy)
            n_synth = len(y_aug) - len(y_noisy)
        except Exception:
            return evaluate_augmented(X_tr, y_noisy, X_te, y_te, factory, min_label,
                                       n_synthetic=0, relabel_correctness=float("nan"))
        return evaluate_augmented(X_aug, y_aug, X_te, y_te, factory, min_label,
                                   n_synthetic=n_synth, relabel_correctness=float("nan"))
    if method in ("cwms_msbs_knn", "cwms_msbs_crossfamily"):
        if model_name in ("calibrated_lr", "xgboost"):
            return _nan_skip_row()
        # Select score array based on scorer variant
        if method == "cwms_msbs_knn":
            scores_for_method = knn_scores.copy() if knn_scores is not None else bal_scores.copy()
        else:  # cwms_msbs_crossfamily
            if model_name == "hgb":
                return _nan_skip_row()  # same-family, not cross-family
            scores_for_method = crossfamily_scores.copy() if crossfamily_scores is not None else bal_scores.copy()
        X_aug, y_aug, n_synth = minority_side_boundary_synthesis(
            X_tr, y_noisy, scores_for_method, budget, min_label, maj_label, seed=seed,
        )
        use_balanced = model_name in ("lightgbm", "catboost", "hgb")
        if use_balanced:
            sw_orig = confidence_weighted_sample_weights_balanced(
                y_noisy, scores_for_method, maj_label, spw,
            )
            sw_synth = np.full(n_synth, float(spw))
            fact = cwms_factory or factory
        else:
            sw_orig = confidence_weighted_sample_weights(y_noisy, scores_for_method, maj_label)
            sw_synth = np.ones(n_synth, dtype=float)
            fact = factory
        sw_combined = np.concatenate([sw_orig, sw_synth])
        return evaluate_augmented(X_aug, y_aug, X_te, y_te, fact,
                                   min_label, n_synthetic=n_synth,
                                   relabel_correctness=float("nan"),
                                   sample_weight=sw_combined)
    if method == "iw_smote":
        X_aug, y_aug, n_synth = iw_smote(
            X_tr, y_noisy, min_label, maj_label, budget_count=budget,
            lamda=30, seed=seed,
        )
        return evaluate_augmented(X_aug, y_aug, X_te, y_te, factory, min_label,
                                   n_synthetic=n_synth, relabel_correctness=float("nan"))
    if method == "sw_framework":
        X_aug, y_aug, n_synth = sw_framework_oversample(
            X_tr, y_noisy, min_label, maj_label, budget_count=budget, seed=seed,
        )
        return evaluate_augmented(X_aug, y_aug, X_te, y_te, factory, min_label,
                                   n_synthetic=n_synth, relabel_correctness=float("nan"))
    raise ValueError(f"Unknown method: {method}")


def _nan_skip_row():
    """Return NaN row for methods skipped on a per-model basis (e.g. calibrated_lr + CWMS)."""
    return {
        "deleted": 0,
        "balanced_accuracy": float("nan"),
        "accuracy": float("nan"),
        "macro_f1": float("nan"),
        "weighted_f1": float("nan"),
        "minority_recall": float("nan"),
        "minority_precision": float("nan"),
        "majority_recall": float("nan"),
        "noise_precision_deleted": float("nan"),
        "clean_minority_deletion_rate": float("nan"),
        "n_relabeled": 0,
        "n_synthetic": 0,
        "relabel_correctness": float("nan"),
    }


def _compute_cleanlab_oof_probs(X_tr, y_noisy, factory):
    """OOF probability matrix (N, 2) used by both cleanlab_filter and cleanlab_relabel."""
    from sklearn.model_selection import StratifiedKFold
    n = len(y_noisy)
    probs = np.full((n, 2), 0.5, dtype=float)
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr_idx, va_idx in folds.split(X_tr, y_noisy):
        m = factory()
        m.fit(X_tr[tr_idx], y_noisy[tr_idx])
        probs[va_idx] = m.predict_proba(X_tr[va_idx])
    return probs


def _run_cleanlab_filter(X_tr, y_noisy, y_tr, noisy_mask, X_te, y_te, factory, min_label, budget):
    """Cleanlab filter baseline: find label issues via OOF self-confidence, then delete."""
    probs = _compute_cleanlab_oof_probs(X_tr, y_noisy, factory)
    try:
        selected = select_cleanlab_filter(y_noisy, probs, budget)
    except Exception:
        selected = np.array([], dtype=int)
    return evaluate(selected, X_tr, y_noisy, y_tr, noisy_mask, X_te, y_te, factory, min_label)


def _run_cleanlab_relabel(X_tr, y_noisy, y_tr, noisy_mask, X_te, y_te, factory,
                          min_label, maj_label, budget):
    """Cleanlab relabel: CleanLab identifies noisy majority samples, we relabel them.

    Same identification mechanism as cleanlab_filter but relabels instead of deletes.
    Tests whether OOF-based ranking (balanced_oof_relabel) is better than CleanLab's
    self-confidence ranking for the relabeling task specifically.
    """
    probs = _compute_cleanlab_oof_probs(X_tr, y_noisy, factory)
    try:
        from cleanlab.filter import find_label_issues
        ranked_noisy = find_label_issues(
            labels=y_noisy,
            pred_probs=probs,
            return_indices_ranked_by="self_confidence",
        )
        # Only relabel majority-labeled samples (minority→minority relabeling is meaningless)
        maj_noisy = [i for i in ranked_noisy if y_noisy[i] == maj_label]
        idx = np.array(maj_noisy[:budget], dtype=int)
    except Exception:
        idx = np.array([], dtype=int)
    y_rel = y_noisy.copy()
    if len(idx):
        y_rel[idx] = min_label
    return _eval_relabel(X_tr, y_rel, idx, X_te, y_te, factory, min_label, y_clean=y_tr)


def _eval_relabel(X_tr, y_rel, idx, X_te, y_te, factory, min_label, *, y_clean):
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
