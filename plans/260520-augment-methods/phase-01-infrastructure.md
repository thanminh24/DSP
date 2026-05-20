---
phase: 1
title: "Infrastructure"
status: pending
effort: "3h"
---

# Phase 1: Infrastructure

## Overview

Stand up balanced-OOF scoring, relabeling utilities, OOF-filtered SMOTE, and a new
experiment runner that mutates the training set (relabel/augment) rather than only
deleting indices. All new files; no existing files modified.

## Context Links

- Existing OOF: `pipeline/scoring/oof_loss.py:11`
- Existing evaluate: `pipeline/evaluation/metrics.py:13`
- Existing run loop: `pipeline/core/experiment.py:77`
- Baseline output schema: `outputs/full-experiment-results.csv`

## Requirements

- All new Python files ≤ 200 lines.
- Output CSV schema must match existing full-experiment-results.csv columns:
  `dataset, model, seed, method, deleted, balanced_accuracy, macro_f1, minority_recall,
  noise_precision_deleted, clean_minority_deletion_rate`.
- For relabel/SMOTE methods, set `deleted=0`, `noise_precision_deleted=NaN`,
  `clean_minority_deletion_rate=NaN`; add `n_added` and `n_relabeled` extra columns.
- Test set is NEVER touched. SMOTE/relabel applied only to (X_tr, y_noisy).

## Data Flow

```
load_dataset → induce_imbalance → encode → inject_noise →
  ├── (deletion methods) suspiciousness = out_of_fold_loss → selector → evaluate
  ├── balanced_oof_majority_scores → relabel_typeA → fit on relabeled (X_tr, y_relabel) → predict on X_te
  └── out_of_fold_loss (for filter) → filtered_smote_augment → fit on (X_aug, y_aug) → predict on X_te
```

## Related Code Files

**Create:**
- `pipeline/scoring/balanced_oof.py`
- `pipeline/augmentation/__init__.py` (empty)
- `pipeline/augmentation/relabeling.py`
- `pipeline/augmentation/filtered_smote.py`
- `pipeline/evaluation/augment_metrics.py`
- `scripts/run_augment_experiment.py`

**Modify:** none.
**Delete:** none.

## Implementation Steps

### Step 1 — install dependency

```bash
/home/than-minh/miniconda3/bin/pip install "imbalanced-learn==0.12.*"
```

Verify: `python -c "import imblearn; print(imblearn.__version__)"`.

### Step 2 — `pipeline/scoring/balanced_oof.py`

```python
"""Class-balanced out-of-fold scoring for Type A detection in the majority pool."""
from __future__ import annotations
from typing import Callable
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight


def balanced_oof_majority_scores(
    X: np.ndarray,
    y_noisy: np.ndarray,
    model_factory: Callable,
    minority_label: int,
    majority_label: int,
    n_splits: int = 5,
    seed: int = 42,
    use_sample_weight: bool = False,
) -> np.ndarray:
    """Return P(minority | x) for every MAJORITY-pool sample via balanced OOF.

    Strategy: train a class-balanced model out-of-fold on the full (X, y_noisy);
    extract minority-class probability for samples whose y_noisy == majority_label.
    Returns array of shape (n,), with NaN at positions where y_noisy == minority_label.

    Args:
        use_sample_weight: True for HGB (no class_weight param); pipes balanced
            sample_weight via fit. False for LR which uses class_weight="balanced"
            internally (caller responsibility to configure factory).
    """
    n = len(y_noisy)
    probs = np.full(n, np.nan, dtype=float)
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr_idx, va_idx in folds.split(X, y_noisy):
        model = model_factory()
        if use_sample_weight:
            sw = compute_sample_weight("balanced", y_noisy[tr_idx])
            # last step name must be the classifier; pipeline forwards sample_weight via __sample_weight
            last = model.steps[-1][0] if hasattr(model, "steps") else None
            fit_kwargs = {f"{last}__sample_weight": sw} if last else {"sample_weight": sw}
            model.fit(X[tr_idx], y_noisy[tr_idx], **fit_kwargs)
        else:
            model.fit(X[tr_idx], y_noisy[tr_idx])
        classes = list(model.classes_)
        min_col = classes.index(minority_label)
        p = model.predict_proba(X[va_idx])[:, min_col]
        probs[va_idx] = p
    # Mask out minority-pool entries: scores undefined there
    minority_mask = (y_noisy == minority_label)
    probs[minority_mask] = np.nan
    return probs
```

Notes:
- LR factory for this phase must set `class_weight="balanced"` in `LogisticRegression(...)`.
- HGB has no `class_weight`; use `use_sample_weight=True` and pass `sample_weight` through
  pipeline. Verify `model.steps[-1][0]` name corresponds to the classifier
  (`make_pipeline` names it `histgradientboostingclassifier`).

### Step 3 — `pipeline/augmentation/relabeling.py`

```python
"""Type A relabeling: flip top-k majority samples to minority based on scores."""
from __future__ import annotations
import numpy as np


def relabel_typeA(
    y_noisy: np.ndarray,
    majority_scores: np.ndarray,
    k: int,
    minority_label: int,
    majority_label: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Relabel top-k majority samples (by P(minority|x)) to minority_label.

    Returns:
        y_relabel: copy of y_noisy with k entries flipped.
        relabeled_idx: indices that were flipped (shape (k,)).
    """
    y_relabel = y_noisy.copy()
    maj_idx = np.where(y_noisy == majority_label)[0]
    # majority_scores has NaN outside majority; restrict
    scores_in_maj = majority_scores[maj_idx]
    valid = ~np.isnan(scores_in_maj)
    maj_idx = maj_idx[valid]
    scores_in_maj = scores_in_maj[valid]
    if len(maj_idx) == 0 or k <= 0:
        return y_relabel, np.array([], dtype=int)
    k = min(k, len(maj_idx))
    top_local = np.argsort(scores_in_maj)[::-1][:k]  # descending P(minority)
    relabeled_idx = maj_idx[top_local]
    y_relabel[relabeled_idx] = minority_label
    return y_relabel, relabeled_idx


def random_relabeling(
    y_noisy: np.ndarray,
    k: int,
    majority_label: int,
    minority_label: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Control: relabel k UNIFORM-RANDOM majority samples to minority.

    If the OOF signal is useless, Method A should match this baseline.
    """
    y_relabel = y_noisy.copy()
    maj_idx = np.where(y_noisy == majority_label)[0]
    if len(maj_idx) == 0 or k <= 0:
        return y_relabel, np.array([], dtype=int)
    k = min(k, len(maj_idx))
    chosen = rng.choice(maj_idx, size=k, replace=False)
    y_relabel[chosen] = minority_label
    return y_relabel, chosen
```

### Step 4 — `pipeline/augmentation/filtered_smote.py`

```python
"""OOF-filtered SMOTE: drop top-frac suspicious minority before synthesis."""
from __future__ import annotations
import numpy as np
from imblearn.over_sampling import SMOTE


def filtered_smote_augment(
    X: np.ndarray,
    y_noisy: np.ndarray,
    suspiciousness: np.ndarray,
    minority_label: int,
    majority_label: int,
    filter_frac: float = 0.05,
    seed: int = 42,
    sampling_strategy: float | str = "auto",
    k_neighbors: int = 5,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Drop top-`filter_frac` of minority pool by OOF loss; SMOTE-augment the rest.

    Returns:
        X_aug, y_aug: augmented training data.
        info: {"n_filtered", "n_synthetic", "minority_kept", "filter_indices"}.
    """
    n = len(y_noisy)
    min_idx = np.where(y_noisy == minority_label)[0]
    n_min = len(min_idx)
    n_filter = int(np.floor(filter_frac * n_min))
    if n_filter > 0 and n_min - n_filter >= max(k_neighbors + 1, 2):
        susp_min = suspiciousness[min_idx]
        top_local = np.argsort(susp_min)[::-1][:n_filter]
        filter_idx = min_idx[top_local]
    else:
        filter_idx = np.array([], dtype=int)

    keep_mask = np.ones(n, dtype=bool)
    keep_mask[filter_idx] = False
    X_keep, y_keep = X[keep_mask], y_noisy[keep_mask]

    # Guard: need both classes and ≥ k_neighbors+1 minority to SMOTE
    n_min_keep = int((y_keep == minority_label).sum())
    if n_min_keep < k_neighbors + 1 or len(np.unique(y_keep)) < 2:
        return X_keep, y_keep, {
            "n_filtered": int(len(filter_idx)),
            "n_synthetic": 0,
            "minority_kept": n_min_keep,
            "filter_indices": filter_idx,
            "smote_applied": False,
        }

    sm = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=min(k_neighbors, n_min_keep - 1),
        random_state=seed,
    )
    X_aug, y_aug = sm.fit_resample(X_keep, y_keep)
    return X_aug, y_aug, {
        "n_filtered": int(len(filter_idx)),
        "n_synthetic": int(len(y_aug) - len(y_keep)),
        "minority_kept": n_min_keep,
        "filter_indices": filter_idx,
        "smote_applied": True,
    }


def plain_smote_augment(
    X: np.ndarray, y_noisy: np.ndarray, minority_label: int,
    seed: int = 42, sampling_strategy: float | str = "auto", k_neighbors: int = 5,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Unfiltered SMOTE baseline."""
    n_min = int((y_noisy == minority_label).sum())
    if n_min < k_neighbors + 1 or len(np.unique(y_noisy)) < 2:
        return X, y_noisy, {"n_synthetic": 0, "smote_applied": False}
    sm = SMOTE(sampling_strategy=sampling_strategy,
               k_neighbors=min(k_neighbors, n_min - 1), random_state=seed)
    X_aug, y_aug = sm.fit_resample(X, y_noisy)
    return X_aug, y_aug, {"n_synthetic": int(len(y_aug) - len(y_noisy)), "smote_applied": True}
```

### Step 5 — `pipeline/evaluation/augment_metrics.py`

```python
"""Evaluation wrapper for relabel/augment methods (training-set mutation paradigm)."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score


def evaluate_augmented(
    X_train_aug: np.ndarray,
    y_train_aug: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_factory,
    minority_label: int,
    n_relabeled: int = 0,
    n_synthetic: int = 0,
    relabel_correctness: float | None = None,
) -> dict:
    """Train on augmented data; report metrics matching evaluate() schema.

    Schema mirrors pipeline.evaluation.metrics.evaluate output, with:
      - deleted = 0 (no deletion in augment paradigm)
      - noise_precision_deleted = NaN
      - clean_minority_deletion_rate = NaN
      - extra: n_relabeled, n_synthetic, relabel_correctness
    """
    if len(np.unique(y_train_aug)) < 2:
        return _nan_result(n_relabeled, n_synthetic, relabel_correctness)
    model = model_factory()
    model.fit(X_train_aug, y_train_aug)
    y_pred = model.predict(X_test)
    return {
        "deleted": 0,
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
        "minority_recall": recall_score(y_test, y_pred, pos_label=minority_label, zero_division=0),
        "noise_precision_deleted": float("nan"),
        "clean_minority_deletion_rate": float("nan"),
        "n_relabeled": int(n_relabeled),
        "n_synthetic": int(n_synthetic),
        "relabel_correctness": (float("nan") if relabel_correctness is None
                                else float(relabel_correctness)),
    }


def _nan_result(n_relabeled, n_synthetic, relabel_correctness):
    return {
        "deleted": 0, "balanced_accuracy": float("nan"),
        "macro_f1": float("nan"), "minority_recall": float("nan"),
        "noise_precision_deleted": float("nan"),
        "clean_minority_deletion_rate": float("nan"),
        "n_relabeled": int(n_relabeled), "n_synthetic": int(n_synthetic),
        "relabel_correctness": (float("nan") if relabel_correctness is None
                                else float(relabel_correctness)),
    }
```

### Step 6 — `scripts/run_augment_experiment.py`

Provides one `run_single_augment(dataset_name, model_name, seed, methods, cfg)` returning
a list of dicts. Same general shape as `pipeline/core/experiment.py:run_single`, but:

- Builds two model factories per run: a **balanced** factory for OOF scoring
  (`class_weight="balanced"` for LR; HGB uses sample-weight passthrough), and the
  **standard** factory for final training/evaluation (unchanged from baseline).
- Branch on method name:
  - `"no_cleaning"` → train on (X_tr, y_noisy).
  - `"class_proportional"` → re-use existing selector + evaluate (deletion).
  - `"balanced_oof_relabel"` → balanced_oof_majority_scores → relabel_typeA(k=budget_count)
    → evaluate_augmented.
  - `"random_relabel"` → random_relabeling(k=budget_count) → evaluate_augmented.
  - `"oof_filtered_smote"` → out_of_fold_loss → filtered_smote_augment(filter_frac=0.05)
    → evaluate_augmented.
  - `"plain_smote"` → plain_smote_augment → evaluate_augmented.
- For relabel methods, compute `relabel_correctness = mean(y_clean[relabeled_idx] == minority_label)`
  — this is the % of relabeled samples that were truly Type A (ground-truth check using `y_train_imb`).
- Output one row per (dataset, model, seed, method); 11 columns matching baseline + 3 extras.

Pseudocode (full implementation in the script, ≤ 200 lines):

```python
def run_single_augment(dataset_name, model_name, seed, methods, cfg):
    # ---- identical setup as run_single (lines 87-116) ----
    # build std_factory (LR/HGB) and bal_factory:
    #   LR: LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
    #   HGB: HistGradientBoostingClassifier(..., random_state=seed) with sample_weight at fit
    susp = out_of_fold_loss(X_tr, y_noisy, std_factory, cfg.n_cv_folds, seed)
    n = len(y_noisy); k = max(1, int(round(cfg.cleaning_budget * n)))
    majority_label = 1 - minority_label  # binary; or derive from np.unique
    results = []
    for method in methods:
        if method == "no_cleaning":
            X_aug, y_aug, info = X_tr, y_noisy, {}
        elif method == "class_proportional":
            sel = select_class_proportional(susp, y_noisy, k)
            row = evaluate(sel, X_tr, y_noisy, y_train_imb, noisy_mask,
                           X_te, y_test, std_factory, minority_label)
            row.update({"n_relabeled": 0, "n_synthetic": 0,
                        "relabel_correctness": float("nan")})
            results.append(_tag(row, dataset_name, model_name, seed, method))
            continue
        elif method == "balanced_oof_relabel":
            scores = balanced_oof_majority_scores(
                X_tr, y_noisy, bal_factory, minority_label, majority_label,
                cfg.n_cv_folds, seed, use_sample_weight=(model_name == "hgb"))
            y_rel, rel_idx = relabel_typeA(y_noisy, scores, k, minority_label, majority_label)
            correctness = float((y_train_imb[rel_idx] == minority_label).mean()) if len(rel_idx) else float("nan")
            X_aug, y_aug = X_tr, y_rel
            info = {"n_relabeled": len(rel_idx), "n_synthetic": 0,
                    "relabel_correctness": correctness}
        elif method == "random_relabel":
            y_rel, rel_idx = random_relabeling(y_noisy, k, majority_label, minority_label, rng)
            correctness = float((y_train_imb[rel_idx] == minority_label).mean()) if len(rel_idx) else float("nan")
            X_aug, y_aug = X_tr, y_rel
            info = {"n_relabeled": len(rel_idx), "n_synthetic": 0,
                    "relabel_correctness": correctness}
        elif method == "oof_filtered_smote":
            X_aug, y_aug, sinfo = filtered_smote_augment(
                X_tr, y_noisy, susp, minority_label, majority_label,
                filter_frac=0.05, seed=seed)
            info = {"n_relabeled": 0, "n_synthetic": sinfo["n_synthetic"],
                    "relabel_correctness": float("nan")}
        elif method == "plain_smote":
            X_aug, y_aug, sinfo = plain_smote_augment(X_tr, y_noisy, minority_label, seed=seed)
            info = {"n_relabeled": 0, "n_synthetic": sinfo["n_synthetic"],
                    "relabel_correctness": float("nan")}
        else:
            raise ValueError(method)

        row = evaluate_augmented(X_aug, y_aug, X_te, y_test, std_factory,
                                 minority_label, **info)
        results.append(_tag(row, dataset_name, model_name, seed, method))
    return results
```

`_tag(row, ds, mdl, sd, mth)` adds `dataset/model/seed/method` columns.

## Todo List

- [ ] Install imbalanced-learn 0.12.* via miniconda pip
- [ ] Write `pipeline/scoring/balanced_oof.py` (≤80 lines)
- [ ] Write `pipeline/augmentation/__init__.py` (empty)
- [ ] Write `pipeline/augmentation/relabeling.py` (≤80 lines)
- [ ] Write `pipeline/augmentation/filtered_smote.py` (≤120 lines)
- [ ] Write `pipeline/evaluation/augment_metrics.py` (≤80 lines)
- [ ] Write `scripts/run_augment_experiment.py` (≤200 lines)
- [ ] Smoke test: `python -c "from scripts.run_augment_experiment import run_single_augment; print(run_single_augment('pima','lr',13,['no_cleaning','balanced_oof_relabel','random_relabel','oof_filtered_smote','plain_smote'],BaseExperimentConfig()))"`
- [ ] Verify output dict has all 11 schema columns + 3 extras

## Success Criteria

- [ ] All new files compile (`python -c "import pipeline.augmentation.relabeling"` etc.)
- [ ] Smoke test on `pima/lr/seed=13` produces 5 rows with non-NaN BA for all augment methods
- [ ] No modification to any file outside the new file list above
- [ ] All files ≤ 200 lines
- [ ] `imblearn.__version__` starts with `0.12`

## Test Matrix

| Unit | Check |
|------|-------|
| `balanced_oof_majority_scores` | shape == n; minority entries NaN; majority probs ∈ [0,1] |
| `relabel_typeA(k=10)` | exactly 10 indices flipped; all originally majority |
| `random_relabeling` | reproducible under same rng seed |
| `filtered_smote_augment(filter_frac=0.05)` | `n_filtered == floor(0.05 * n_min)`; output X shape ≥ input |
| `evaluate_augmented` | returns dict with all 11+3 keys |
| `run_single_augment` smoke | returns list[dict] len == len(methods), all BA non-NaN |

## Risk Assessment

| Risk | L | I | Mitigation |
|------|---|---|----|
| HGB sample_weight pipeline name mismatch | M | M | Smoke test prints classifier step name first; fall back to direct estimator if pipeline keyword fails |
| SMOTE k_neighbors > n_minority_kept | M | H | Guard in code: `k_neighbors=min(k, n_min_keep-1)`; skip SMOTE if `n_min_keep < k+1` |
| Balanced LR converges differently → seed instability | L | L | `max_iter=1000` preserved from baseline factory |

## Rollback Plan

`git rm` new files. No existing code affected.

## Next Steps

Phase 2 (Method A pilot) and Phase 3 (Method B pilot) both depend on Phase 1.
