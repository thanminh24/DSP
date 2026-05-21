# Confidence-Guided Relabeling Report

## Summary

The current project direction is confidence-guided relabeling for hidden-minority label
noise. This replaces the older CRCC deletion framing as the main paper contribution.

## Method

1. Split clean held-out test data before training noise is injected.
2. Train class-balanced out-of-fold models on noisy training labels.
3. Score majority-labeled samples by `P(minority | x)`.
4. Relabel the top-k majority-labeled samples to minority.
5. Train the final model on relabeled training data.
6. Compare against deletion, random relabeling, unbalanced confidence relabeling,
   cleanlab, class-weight-only models, and boosting baselines.

## Why It Matters

Deletion can remove scarce minority feature evidence. Relabeling keeps the sample and
attempts to repair the noisy label, which is especially relevant when true minority
examples are mislabeled as majority.

## Validation Gates

- Top-venue gap confirmation.
- Leakage/protocol audit.
- Controlled synthetic benchmark.
- Realistic weak-supervision benchmark.
- Strong baselines.
- XGBoost/LightGBM/CatBoost stress testing.
- Paired statistics with enough observations.

## Current Smoke Results

The protocol audit passes and writes `outputs/relabeling-protocol-audit.md`.

Single-combo smoke checks on `pima`, seed `13`, hidden-minority medium noise:

| model | class_proportional BA | balanced_oof_relabel BA | unbalanced_oof_relabel BA |
|---|---:|---:|---:|
| lr | 0.6519 | 0.7105 | 0.7254 |
| hgb | 0.6111 | 0.5940 | 0.5775 |
| xgboost | 0.5610 | 0.5970 | 0.5786 |

Interpretation: the method path runs beyond LR, but the LR result also shows why the
unbalanced confidence baseline is mandatory. Publication claims must be based on the full
paired sweep, not these smoke checks.

## Unresolved Questions

- Weak-supervision transfer is not yet proven.
- Publication claim depends on non-LR model stress results.
