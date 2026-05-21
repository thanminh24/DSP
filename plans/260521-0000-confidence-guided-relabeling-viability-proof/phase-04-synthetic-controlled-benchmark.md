---
phase: 4
title: "Synthetic Controlled Benchmark"
status: pending
priority: P1
effort: "2d"
dependencies: [2, 3]
---

# Phase 4: Synthetic Controlled Benchmark

## Overview

Run a stronger controlled benchmark on existing KEEL/UCI datasets. Synthetic noise remains
necessary because relabel precision and true hidden-minority recovery require known clean
labels, but it cannot be the only evidence.

## Requirements

- Functional: compare balanced OOF relabeling under multiple noise directions, budgets,
  imbalance ratios, datasets, model families, and controls.
- Non-functional: results must be reproducible and paired by seed.

## Architecture

```text
cached tabular datasets
  -> train/test split
  -> induced imbalance
  -> controlled noise protocol
  -> OOF candidate scoring
  -> relabel/delete/baseline interventions
  -> final model training
  -> paired metrics and diagnostics
```

## Related Code Files

- Modify: `scripts/run_augment_sweep.py`
- Create: `scripts/run_relabeling_viability_sweep.py`
- Create: `scripts/analyze_relabeling_viability.py`
- Read: `pipeline/core/config.py`
- Read: `pipeline/data/loaders.py`
- Read: `pipeline/scoring/balanced_oof.py`
- Read: `outputs/augment-sweep-results.csv`

## Dataset Scope

- Primary: `pima`, `credit-g`, `yeast`, `ecoli`, `phoneme`.
- Existing rationale: these are already cached and used in prior imbalanced tabular
  literature.

## Experiment Grid

- Models: LR smoke test only. Publication runs must include HGB and at least one external
  boosting model through Phase 7.
- Seeds: at least 20, preferably the existing prime seed list.
- Imbalance ratios: 0.15 and 0.30 minority.
- Noise protocols:
  - asymmetric hidden-minority: min->maj 30%, maj->min 10%.
  - low noise: min->maj 10%, maj->min 5%.
  - high noise: min->maj 40%, maj->min 20%.
  - reverse asymmetric: min->maj 10%, maj->min 30% as failure-mode check.
  - symmetric: 20%/20% as non-specialized control.
- Budgets: 5%, 10%, 20%.

## Methods

- `no_cleaning`
- `class_proportional`
- `random_relabel`
- `balanced_oof_relabel`
- `global_top_loss`
- `plain_smote`
- cleanlab-style filter/relabel baseline from Phase 6 when available
- oracle relabel only as synthetic ceiling

## Metrics

- balanced accuracy
- macro-F1
- minority recall
- relabel precision: selected examples whose clean label is minority
- relabel false-positive rate: majority-clean examples flipped to minority
- candidate AUC/AP for identifying hidden-minority examples
- calibration bins for `P(minority | x)`

## Implementation Steps

1. Refactor sweep config so noise direction, imbalance ratio, and budget are parameters.
2. Add oracle relabel baseline:
   - relabel budgeted majority-labeled samples whose clean label is minority.
   - use only as ceiling, not practical baseline.
3. Add reverse-noise and symmetric-noise protocols.
4. Add candidate-detection metrics before final model training.
5. Run LR grid first as a protocol smoke test.
6. Run HGB reduced grid before Phase 7 expands to XGBoost/LightGBM/CatBoost.
7. Generate plots:
   - delta BA vs `class_proportional`.
   - relabel precision vs random.
   - minority recall by noise direction.
   - failure-mode plot for reverse asymmetric noise.

## Success Criteria

- [ ] Balanced OOF relabeling beats `class_proportional` in at least 70% of
      hidden-minority controlled combinations.
- [ ] Relabel precision is at least 2x random relabel precision in controlled hidden-minority
      settings.
- [ ] Method does not claim success under reverse asymmetric noise unless results support it.
- [ ] HGB runs successfully on the reduced grid before publication-grade model stress.
- [ ] All result rows include dataset, model, seed, noise protocol, imbalance ratio, budget,
      method, and metrics.

## Risk Assessment

If the method only works at 0.15 minority and fails at 0.30, the claim narrows to severe
imbalance. If reverse-noise fails, present that as expected: this method targets hidden
minority recovery, not arbitrary label correction.

## Security Considerations

None.

## Next Steps

Proceed to Phase 5 for realistic noisy-label evidence.

## Unresolved Questions

- Can HGB runtime be kept under a practical limit with 20 seeds before adding XGBoost?
