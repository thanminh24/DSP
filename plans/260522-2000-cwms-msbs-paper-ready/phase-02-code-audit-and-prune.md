---
phase: 2
title: "Code Audit and Prune"
status: pending
priority: P1
effort: "~1h"
dependencies: []
---

# Phase 2: Code Audit and Prune

## Overview

Prune dead pipeline modules and isolate the 15+ non-paper methods in
`run_relabeling_viability_sweep.py` into a clearly separated "baselines archive" section.
The goal: anyone reading the repo sees immediately what is the contribution vs. what is
historical scaffolding. No functional code is deleted — just reorganized.

## What Stays Active (Paper-Relevant)

### Pipeline modules — keep, actively used

| Module | Used by | Status |
|--------|---------|--------|
| `pipeline/augmentation/msbs.py` | cwms_msbs, msbs dispatchers | **KEEP** |
| `pipeline/baselines/soft_weighting.py` | cwms, cwms_msbs | **KEEP** |
| `pipeline/scoring/balanced_oof.py` | all methods (OOF scorer) | **KEEP** |
| `pipeline/scoring/oof_loss.py` | global_top_loss, class_proportional | **KEEP** |
| `pipeline/cleaning/selectors.py` → `select_class_proportional` | class_proportional | **KEEP** (function only) |
| `pipeline/evaluation/augment_metrics.py` | all augment methods | **KEEP** |
| `pipeline/evaluation/metrics.py` | deletion methods | **KEEP** |
| `pipeline/models/factories.py` | all | **KEEP** |
| `pipeline/data/loaders.py` | all | **KEEP** |
| `pipeline/data/encoding.py` | all | **KEEP** |
| `pipeline/core/config.py` | sweep config | **KEEP** |

### Scripts — keep

| Script | Status |
|--------|--------|
| `scripts/run_relabeling_viability_sweep.py` | **KEEP** (core infra) — prune internally |
| `scripts/run_cwms_msbs_deep_sweep.py` | **KEEP** (primary runner) |
| `scripts/analyze_cwms_msbs_deep_results.py` | **KEEP** |
| `scripts/analyze_cwms_msbs_results.py` | **KEEP** (v1 reference) |
| `scripts/download_datasets.py` | **KEEP** |
| `scripts/validate_environment.py` | **KEEP** |

## What to Prune or Isolate

### Pipeline modules — isolate (don't delete, but clearly mark as non-paper)

These modules are imported by the sweep for baseline comparison methods. They should
stay in the repo but be clearly marked:

| Module | Action |
|--------|--------|
| `pipeline/augmentation/synthesis.py` | Add header comment: `# CGMS (failed baseline): not part of paper contribution` |
| `pipeline/augmentation/relabeling.py` | Add header: `# relabeling baselines: OOF relabeling discouraged (see OOF circularity note)` |
| `pipeline/augmentation/filtered_smote.py` | **DELETE** — not imported anywhere in active codebase |
| `pipeline/baselines/cleanlab_baselines.py` | Add header: `# cleanlab baseline: comparison method, not paper contribution` |
| `pipeline/baselines/confidence_relabeling.py` | Add header: `# unbalanced OOF baseline: comparison only` |
| `pipeline/baselines/class_weight_baselines.py` | Check if imported. If not: **DELETE** |
| `pipeline/models/calibration.py` | Check if imported. If not: **DELETE** |
| `pipeline/data/weak_supervision_loaders.py` | **DELETE** — data removed, never imported |
| `pipeline/cleaning/selectors.py` | Keep, but add: `# select_class_proportional used in paper; select_global/oracle are comparison baselines` |

### `run_relabeling_viability_sweep.py` — restructure METHODS list

Current `METHODS` list has 18 entries. Restructure with clear sections:

```python
METHODS = [
    # === PAPER METHODS (primary contribution) ===
    "cwms_msbs",        # combined: MSBS synthesis + CWMS weights (our method)
    "msbs",             # MSBS standalone (ablation)
    "cwms",             # CWMS standalone (ablation)

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
```

The deep sweep runner (`run_cwms_msbs_deep_sweep.py`) only uses the paper methods +
baselines — 5 methods total. The full `METHODS` list in the viability sweep is kept
as a historical record.

### `run_relabeling_viability_sweep.py` — remove dead OOF scoring

Currently the viability sweep computes both `unbal_scores` and `naive_scores` on every
run. For the deep sweep these are never used. Add a guard:

```python
unbal_scores = (
    unbalanced_oof_majority_scores(...)
    if any(m in methods_to_run for m in ["unbalanced_oof_relabel", "shuffled_score_relabel",
                                          "inverted_score_relabel"])
    else None
)
naive_scores = (
    naive_confidence_majority_scores(...)
    if "naive_confidence_relabel" in methods_to_run
    else None
)
```

This is already partially done (`naive_scores` is guarded). Extend the same pattern to
`unbal_scores` so deep sweeps skip that OOF pass entirely (saves ~20% compute per combo).

## Implementation Steps

### Step 1 — Check which modules are actually imported anywhere active

```bash
cd /home/than-minh/project/DSP
grep -rn "filtered_smote\|class_weight_baselines\|calibration\|weak_supervision_loaders" \
  scripts/ pipeline/ | grep -v __pycache__ | grep "^[^#]"
```

### Step 2 — Delete confirmed-unused modules

```bash
git rm pipeline/augmentation/filtered_smote.py
git rm pipeline/data/weak_supervision_loaders.py
# Only after confirming no active import:
# git rm pipeline/models/calibration.py
# git rm pipeline/baselines/class_weight_baselines.py
```

### Step 3 — Add header comments to non-paper modules

Add a one-line module-level comment to: `synthesis.py`, `relabeling.py`,
`cleanlab_baselines.py`, `confidence_relabeling.py`.

### Step 4 — Restructure METHODS list in viability sweep

Reorder METHODS list as shown above with section comments.

### Step 5 — Add lazy guard for `unbal_scores`

In `run_single_viability()`, wrap `unbalanced_oof_majority_scores()` call with a
methods-to-run check (same pattern as existing `naive_scores` guard).

### Step 6 — Verify deep sweep still works

```bash
/home/than-minh/miniconda3/envs/dsp/bin/python -c "
from scripts.run_cwms_msbs_deep_sweep import main
from scripts.run_relabeling_viability_sweep import run_single_viability
rows = run_single_viability('pima', 'lr', 13, 'hidden_minority_medium', 0.30, 0.10, 0.10, 0.15,
                             methods=['no_cleaning', 'class_proportional', 'cwms_msbs'])
print('OK:', [r['method'] for r in rows], [r['balanced_accuracy'] for r in rows])
"
```

## Success Criteria

- [ ] `filtered_smote.py` and `weak_supervision_loaders.py` deleted
- [ ] `calibration.py` and `class_weight_baselines.py` checked — deleted if unused
- [ ] Non-paper modules have clear header comments
- [ ] METHODS list restructured with section comments
- [ ] `unbal_scores` lazy-guarded like `naive_scores`
- [ ] Deep sweep smoke test passes
- [ ] `git diff --stat` shows no logic changes, only cosmetic + deletions

## Risk Assessment

- **Import chain**: before deleting any module, grep for all import sites. Some modules
  may be transitively imported through `__init__.py` files.
- **`calibration.py`**: this may be used by `calibrated_lr` factory indirectly via sklearn.
  Only delete if confirmed not imported in pipeline or scripts.
- **METHODS reorder**: purely cosmetic — does not change behavior. Safe.
