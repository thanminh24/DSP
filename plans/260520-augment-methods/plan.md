---
title: "Augmentation Methods: Balanced-OOF Relabeling and OOF-Filtered SMOTE"
description: "Last-chance test of two augmentation methods (relabel Type A via balanced OOF, OOF-filtered SMOTE) before abandoning the noisy-label-cleaning direction."
status: completed
priority: P2
effort: 12h
branch: ""
tags: [research, noisy-labels, smote, relabeling, imbalanced]
blockedBy: []
blocks: []
created: "2026-05-20T16:10:19.921Z"
createdBy: "ck:plan"
source: skill
---

# Augmentation Methods: Balanced-OOF Relabeling and OOF-Filtered SMOTE

## Overview

All prior deletion-based methods (CRCC-P, CRCC-Adaptive, TBDC, OOF relabeling thresholds 0.65–0.95)
have failed to beat `class_proportional` under asymmetric class-dependent noise (min→maj=30%,
maj→min=10%) at 15% minority ratio. One signal worth chasing: balanced OOF at threshold 0.65 gave
+0.099 BA in a quick test (possibly accidental oversampling — must be controlled for).

This plan tests two augmentation paradigms as a **final attempt** before abandoning the
research direction:

- **Method A** — Balanced-OOF Type A relabeling: detect true-minority samples mislabeled as
  majority via class-balanced OOF; relabel top-k. Restores real (non-synthetic) data.
- **Method B** — OOF-filtered SMOTE: remove top-5% suspect Type B from minority seed pool
  before synthesizing, reducing contaminated synthetic samples.

Both methods break the deletion-only paradigm of `evaluate()`. A new training-set-mutating
runner is required.

**Abandonment clause (binding):** If Phase 2 AND Phase 3 both FAIL their pilot gates, write the
negative result summary, stop. Do not run Phases 4–5. The negative result to document:
> Under asymmetric class-dependent noise (30%/10%) at 15% minority ratio, neither
> deletion-based cleaning, balanced-OOF Type A relabeling, nor OOF-filtered SMOTE
> consistently outperforms class-proportional. The structural constraint is OOF
> unreliability at small minority pool size (n_minority ~ 75). Per-class budget
> allocation improves only when the noise-density estimator has sufficient calibration
> quality, which requires larger minority pools.

## Phases

| Phase | Name | Status | Effort |
|-------|------|--------|--------|
| 1 | [Infrastructure](./phase-01-infrastructure.md) | Complete | 3h |
| 2 | [Method A Pilot](./phase-02-method-a-pilot.md) | **GO** | 2h |
| 3 | [Method B Pilot](./phase-03-method-b-pilot.md) | **FAIL** | 2h |
| 4 | [Full Sweep](./phase-04-full-sweep.md) | Complete (Method A only) | 3h |
| 5 | [Statistical Analysis](./phase-05-statistical-analysis.md) | Complete | 2h |

## Decision Graph

```
Phase 1 (infrastructure)
   ├── Phase 2 (Method A pilot) ──┐
   │                              ├─ GO/PARTIAL on either? ── Phase 4 (full sweep, surviving methods only) ── Phase 5 (stats)
   ├── Phase 3 (Method B pilot) ──┘
   │
   └── Both FAIL? ── STOP. Write negative result. Do not run Phase 4 or 5.
```

## Pilot Gate (applied identically in Phase 2 and Phase 3)

For each method vs `class_proportional` baseline, across 4 datasets × 2 models = 8 combos
at 20 seeds (medium noise 30%/10%):

- **GO**: BA > class_proportional in ≥ 6/8 combos AND mean Cohen's d ≥ 0.3 AND
  Wilcoxon signed-rank p < 0.05 (paired by seed, pooled across combos).
- **PARTIAL GO**: ≥ 5/8 combos AND mean Cohen's d ≥ 0.2 → proceed to full sweep, but
  scoped to this method only; report as "moderate evidence".
- **FAIL**: neither threshold met.

## Scope

- **Datasets (pilot)**: pima, credit-g, yeast, phoneme. (ecoli excluded — minority pool too small.)
- **Datasets (full sweep)**: + ecoli → 5 datasets.
- **Models**: lr, hgb.
- **Seeds**: 20 fixed primes — [13,17,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101].
- **Noise (pilot)**: medium 30%/10% only.
- **Noise (full sweep)**: low 10%/5%, medium 30%/10%, high 40%/20%.
- **Budget**: 10% of n_samples (`cleaning_budget = 0.10`), same as prior work.
- **Minority ratio**: 15%.
- **Runtime**: `/home/than-minh/miniconda3/bin/python3`.

## Risk Matrix

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Balanced OOF gain is pure oversampling artifact | HIGH | HIGH | `random_relabel` control in Phase 2 |
| SMOTE introduces train/test leakage via synthesizing across test boundary | LOW | HIGH | SMOTE applied only to `X_tr, y_noisy`; test set untouched (verified in evaluation wrapper) |
| imbalanced-learn version drift | MED | LOW | pin `imbalanced-learn==0.12.*` in Phase 1 |
| New runner output schema diverges from existing CSV | MED | MED | Phase 1 explicitly produces same columns as `outputs/full-experiment-results.csv` |
| Both methods FAIL — wasted effort on Phases 4–5 | MED | HIGH | Hard abandonment gate after Phase 3 |
| HGB ignores `class_weight="balanced"` (different API) | HIGH | MED | Use `sample_weight` computed via `compute_sample_weight('balanced', y_noisy)` for HGB; documented in Phase 1 |
| Type B contamination still leaks past OOF filter at depth 5% | MED | MED | Phase 3 logs filter precision (% of removed that were truly Type B) |

## File Ownership

- New code: `pipeline/scoring/balanced_oof.py`, `pipeline/augmentation/*`,
  `scripts/run_augment_experiment.py`, `scripts/run_pilot_method_a.py`,
  `scripts/run_pilot_method_b.py`, `scripts/run_augment_sweep.py`,
  `scripts/analyze_augment_stats.py`.
- Untouched: `pipeline/core/experiment.py`, `pipeline/cleaning/selectors.py`,
  `pipeline/evaluation/metrics.py`, existing scripts.
- Outputs: `outputs/pilot-method-a-results.csv`, `outputs/pilot-method-b-results.csv`,
  `outputs/augment-sweep-results.csv`, `outputs/augment-statistical-tests.csv`,
  `outputs/negative-result-summary.md` (if abandonment triggered).

## Rollback Plan

- Phase 1 only adds new files; no existing code modified. Rollback = `git rm` new files.
- Phases 2–5 produce CSVs only; no schema change to existing data.
- Abandonment path: write `outputs/negative-result-summary.md`, leave new code in tree
  for reproducibility, mark plan `cancelled`.

## Dependencies

- External: `pip install imbalanced-learn==0.12.*` (Phase 1).
- Internal: relies on existing `load_dataset`, `induce_imbalance`, `inject_noise`,
  `out_of_fold_loss`, `evaluate` from `pipeline/`.

## Open Questions

None — abandonment gate, controls, and statistics are all specified.

## Final Results (2026-05-21)

**Method A (balanced-OOF Type A relabeling): GO**

| dataset | class_prop | balanced_oof_relabel | delta | Cohen's d |
|---------|-----------|---------------------|-------|-----------|
| credit-g | 0.610 | 0.632 | +0.021 | 0.564 |
| phoneme | 0.645 | 0.723 | +0.078 | 5.375 |
| pima | 0.670 | 0.703 | +0.033 | 0.881 |
| yeast | 0.740 | 0.761 | +0.021 | 0.835 |
| ecoli | 0.838 | 0.851 | +0.013 | 0.237 |

Wins: 15/15 combos (5 datasets × 3 noise levels). Cohen's d=1.465. Control passed (beats random_relabel, p=7.8e-15). Noise scaling: +0.024 → +0.033 → +0.036 (method gets stronger with more noise).

**Method B (OOF-filtered SMOTE): FAIL**

SMOTE beats class_proportional, but OOF filtering adds zero value (delta vs plain_smote = +0.002, p=0.198). Gain is from SMOTE synthesis, not Type B noise removal.

**Verdict: Continue research.** Method A is a real finding with proper statistical controls. Write the paper. Only Method A survived — but it survived robustly.
