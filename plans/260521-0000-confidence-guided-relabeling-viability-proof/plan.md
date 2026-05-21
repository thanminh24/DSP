---
title: "Confidence-Guided Relabeling Viability Proof"
description: "Validate balanced out-of-fold relabeling as the main project/paper contribution under leakage-safe synthetic and realistic noisy-label protocols."
status: pending
priority: P1
branch: ""
tags: [research, noisy-labels, imbalanced, relabeling, tabular, paper]
blockedBy: []
blocks: []
created: "2026-05-21T06:57:37.327Z"
createdBy: "ck:plan"
source: skill
---

# Confidence-Guided Relabeling Viability Proof

## Overview

This plan proves whether the current confidence-guided relabeling method is viable as both
a school project and a research-paper direction.

Current evidence is promising but incomplete. The earlier CRCC deletion story is not strong
enough as the main paper claim because CRCC-P matches `class_proportional` exactly in the
stored tests. The stronger result is `balanced_oof_relabel`: it wins against
`class_proportional` in the existing augmentation sweep and passes the random-relabel control.

The revised project framing:

> In imbalanced noisy tabular classification, deletion can discard scarce minority feature
> evidence. We study a lightweight post-detection intervention that uses class-balanced
> out-of-fold confidence to identify majority-labeled samples that likely belong to the
> minority class, then relabels them instead of deleting them.

This plan must answer four questions:

1. Is the method leakage-safe?
2. Does it beat strong, fair baselines, not just weak deletion baselines?
3. Does it work beyond artificial class-dependent label flips?
4. Is the gap narrow but real after surveying 2024+ Q1/Q2 and A*/A venue work?

## Scope Challenge

- Existing code: reuse `pipeline/scoring/balanced_oof.py`,
  `pipeline/augmentation/relabeling.py`, `scripts/run_augment_experiment.py`, and
  existing KEEL/UCI datasets.
- Minimum changes: add audit scripts, cleanlab/baseline adapters, weak-supervision data
  support, XGBoost/LightGBM/CatBoost adapters, and analysis/report generation. Do not
  rewrite the pipeline.
- Complexity: 9 phases are justified because the main risk is not implementation, it is
  paper-validity: leakage, novelty, baselines, model generality, realistic data, and claims.
- Selected mode: HOLD SCOPE, upgraded to publication rigor. LR is only a smoke-test model;
  publication evidence requires stronger tabular learners.

## Main Hypothesis

Balanced out-of-fold relabeling improves downstream minority-sensitive performance because
it repairs hidden minority examples mislabeled as majority while preserving their feature
vectors. It should beat deletion-based cleaning when minority-to-majority label noise is
common and should remain competitive under realistic weak-label noise.

## Negative Result Boundary

Stop pursuing a paper if any of these are true:

- Leakage audit finds train/test or clean-label leakage that explains the current gains.
- Method does not beat `cleanlab` or a confidence-threshold relabeling baseline on balanced
  accuracy and minority recall across meaningful datasets.
- Method only works on synthetic minority-to-majority flips and fails on weak-supervision
  tabular tasks.
- Relabel precision is not better than random relabeling by at least 2x in controlled data.
- Top-venue 2024+ survey finds an already-published method with the same core setting:
  tabular, imbalanced, noisy labels, budgeted majority-pool relabeling using class-balanced
  OOF confidence, with no substantial differentiator.
- The method only works for LR and fails on XGBoost/LightGBM/CatBoost or calibrated tree
  ensembles.

## Literature Positioning

- Confident Learning is a model-agnostic label-error framework based on pruning, counting,
  and ranking examples by label quality. It is a required baseline, not a strawman.
- Class-aware noisy-label learning already exists, especially in deep/long-tailed settings.
  The gap is not "class-aware noisy labels are unsolved."
- Active label cleaning studies budgeted relabeling, often with human annotation loops.
  Our gap is automated, lightweight, tabular, and post-detection.
- Recent weak-supervision noisy-label benchmarks include tabular datasets and emphasize
  realistic non-random noise. We need at least one such benchmark to make the paper credible.
- A separate survey gate must cover 2024+ Q1/Q2 journals and A*/A conferences before any
  novelty claim is allowed.

## Related Existing Plans

- `plans/260520-augment-methods/` is completed and provides the first positive evidence.
- `plans/plan.md` covers TBDC deletion and should be treated as secondary. Do not use TBDC
  as the main paper story unless the relabeling viability gate fails.
- `plans/260520-1604-crcc-paper-completion/` is completed and supplies reusable CRCC
  baselines, datasets, and ablation infrastructure.

## Phases

| Phase | Name | Status |
|-------|------|--------|
| 1 | [Framing and Gap Audit](./phase-01-framing-and-gap-audit.md) | Pending |
| 2 | [Top-Venue Gap Confirmation](./phase-02-top-venue-gap-confirmation.md) | Pending |
| 3 | [Leakage and Protocol Audit](./phase-03-leakage-and-protocol-audit.md) | Pending |
| 4 | [Synthetic Controlled Benchmark](./phase-04-synthetic-controlled-benchmark.md) | Pending |
| 5 | [Realistic Weak-Supervision Benchmark](./phase-05-realistic-weak-supervision-benchmark.md) | Pending |
| 6 | [Strong Baseline Comparison](./phase-06-strong-baseline-comparison.md) | Pending |
| 7 | [Publication-Grade Model Stress Benchmark](./phase-07-publication-grade-model-stress-benchmark.md) | Pending |
| 8 | [Robustness and Statistical Validation](./phase-08-robustness-and-statistical-validation.md) | Pending |
| 9 | [Paper and Project Packaging](./phase-09-paper-and-project-packaging.md) | Pending |

## Dependencies

- Phase 2 blocks any novelty claim.
- Phase 3 blocks all experiment claims.
- Phase 4 supplies controlled evidence and relabel precision.
- Phase 5 supplies real-world relevance.
- Phase 6 must include baselines before Phase 8 statistics are meaningful.
- Phase 7 must pass before paper submission; LR-only evidence is not publication-grade.

## Success Criteria

- Leakage audit produces a written pass/fail report with explicit clean-label access checks.
- Controlled benchmark shows `balanced_oof_relabel` beats `class_proportional`,
  `random_relabel`, and `cleanlab`-style handling on most dataset/noise/model combinations.
- Realistic benchmark includes at least one weak-supervision tabular dataset with noisy
  labels and ground-truth labels available for evaluation.
- Model stress benchmark includes at minimum LR, calibrated LR, Random Forest or ExtraTrees,
  HistGradientBoosting, XGBoost, and LightGBM or CatBoost.
- Statistical report uses paired tests with enough seeds or folds to avoid the previous
  `n=5` Wilcoxon floor problem.
- Gap report surveys 2024+ A*/A conferences and Q1/Q2 journals and records whether any
  method directly fills the proposed gap.
- Final report states a narrow claim and lists failed alternatives, especially CRCC-P
  versus class-proportional deletion.

## Out of Scope

- Claiming state of the art against deep noisy-label methods.
- Implementing DivideMix, Co-teaching, ITEM, CBS, or long-tailed image benchmarks.
- Using test labels or clean training labels to select relabel candidates.
- Treating synthetic noise alone as enough for paper viability.
- Publishing with LR-only evidence.

## Resolved Decisions

- **Venue ranking**: CORE for conferences (A*, A gate), Scimago/JCR for journals (Q1/Q2 gate).
- **Boosting stack**: All three (XGBoost, LightGBM, CatBoost) if installation and runtime
  allow; fallback order is XGBoost → LightGBM → CatBoost.

## Unresolved Questions

- Phase 5 WRENCH smoke test result is unknown. Must run a manual load test before writing
  any weak-supervision loader. If it fails on all candidates, paper scope narrows to
  synthetic-only evidence with a negative real-world transfer note.
