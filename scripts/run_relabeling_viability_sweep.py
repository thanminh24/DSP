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
from pipeline.baselines.soft_weighting import confidence_weighted_sample_weights
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
BUDGETS = [0.05, 0.10, 0.20]
QUICK_BUDGETS = [0.10]
IMBALANCE_RATIOS = [0.15, 0.30]
QUICK_RATIOS = [0.15]
METHODS = [
    "no_cleaning",
    "global_top_loss",
    "class_proportional",
    # "balanced_oof_relabel",  # DISCOURAGED: OOF circularity — scorer same family as final model
    "unbalanced_oof_relabel",
    "naive_confidence_relabel",   # confirmation-bias baseline: no OOF, full-data scoring
    "cleanlab_relabel",           # CleanLab identifies noisy majority samples → relabel
    "cleanlab_filter",
    "shuffled_score_relabel",
    "inverted_score_relabel",
    "random_relabel",
    "class_weight_only",
    "oracle_relabel",
    "cgms_t03",
    "cgms_t05",
    "cgms_t07",
    "msbs",
    "cwms",
    "cwms_msbs",   # combined: MSBS synthesis + CWMS per-sample weights in one training pass
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
    use_sw = model_supports_sample_weight(model_name)

    methods_to_run = set(methods or METHODS)

    suspiciousness = out_of_fold_loss(X_tr, y_noisy, std_factory, 5, seed)
    bal_scores = balanced_oof_majority_scores(
        X_tr, y_noisy, bal_factory, minority_label, majority_label, 5, seed, use_sample_weight=use_sw,
    )
    unbal_scores = unbalanced_oof_majority_scores(
        X_tr, y_noisy, std_factory, minority_label, majority_label, 5, seed, use_sample_weight=False,
    )
    # Full-data (non-OOF) balanced scores for confirmation-bias baseline
    naive_scores = (
        naive_confidence_majority_scores(
            X_tr, y_noisy, bal_factory, minority_label, majority_label, seed, use_sw,
        )
        if "naive_confidence_relabel" in methods_to_run else None
    )

    rows = []
    for method in (methods or METHODS):
        row = _run_method(
            method, X_tr, y_noisy, y_tr, noisy_mask, X_te, y_te, std_factory,
            suspiciousness, bal_scores, unbal_scores, naive_scores, budget_count,
            minority_label, majority_label, rng, seed,
            bal_factory=bal_factory, use_sw=use_sw,
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
                bal_factory=None, use_sw=False):
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
        return _eval_relabel(X_tr, y_rel, idx, y_tr, X_te, y_te, factory, min_label)
    if method == "balanced_oof_relabel":
        y_rel, idx = relabel_typeA(y_noisy, bal_scores, budget, min_label, maj_label)
        return _eval_relabel(X_tr, y_rel, idx, y_tr, X_te, y_te, factory, min_label)
    if method == "naive_confidence_relabel":
        y_rel, idx = relabel_typeA(y_noisy, naive_scores, budget, min_label, maj_label)
        return _eval_relabel(X_tr, y_rel, idx, y_tr, X_te, y_te, factory, min_label)
    if method == "cleanlab_relabel":
        return _run_cleanlab_relabel(X_tr, y_noisy, y_tr, noisy_mask, X_te, y_te, factory, min_label, maj_label, budget)
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
        sw = confidence_weighted_sample_weights(y_noisy, bal_scores, maj_label)
        return evaluate_augmented(X_tr, y_noisy, X_te, y_te, factory,
                                   min_label, sample_weight=sw)
    if method == "cwms_msbs":
        # Combined: synthesize minority samples near boundary (MSBS) + suppress suspicious
        # majority in the same training pass (CWMS). Synthetic samples get weight=1.0.
        X_aug, y_aug, n_synth = minority_side_boundary_synthesis(
            X_tr, y_noisy, bal_scores, budget, min_label, maj_label, seed=seed,
        )
        # CWMS weights for original samples; synthetic appended at end → weight=1.0 by default
        sw_orig = confidence_weighted_sample_weights(y_noisy, bal_scores, maj_label)
        sw_synth = np.ones(n_synth, dtype=float)
        sw_combined = np.concatenate([sw_orig, sw_synth])
        return evaluate_augmented(X_aug, y_aug, X_te, y_te, factory,
                                   min_label, n_synthetic=n_synth,
                                   relabel_correctness=float("nan"),
                                   sample_weight=sw_combined)
    raise ValueError(f"Unknown method: {method}")


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
    return _eval_relabel(X_tr, y_rel, idx, y_tr, X_te, y_te, factory, min_label)


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
