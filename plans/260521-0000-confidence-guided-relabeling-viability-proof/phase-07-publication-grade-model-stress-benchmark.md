---
phase: 7
title: "Publication-Grade Model Stress Benchmark"
status: pending
priority: P1
effort: "3d"
dependencies: [4, 5, 6]
---

# Phase 7: Publication-Grade Model Stress Benchmark

## Overview

Test whether the method survives beyond Logistic Regression. LR-only evidence is acceptable
for a school prototype but unacceptable for a serious tabular ML paper. This phase adds
strong tabular learners, probability calibration checks, runtime controls, and model-family
ablation.

## Requirements

- Functional: run the relabeling method and baselines across linear, bagging, boosting,
  and calibrated-probability model families.
- Non-functional: all models must use identical data splits, noise injections, candidate
  budgets, and final evaluation metrics.

## Required Model Families

Minimum publication set:

- Logistic Regression.
- Calibrated Logistic Regression.
- Random Forest or ExtraTrees.
- HistGradientBoostingClassifier.
- XGBoost.
- LightGBM or CatBoost.

Boosting target:

- Include all three: XGBoost, LightGBM, and CatBoost if installation and runtime permit.
- Fallback order if runtime becomes prohibitive: XGBoost first, LightGBM second, CatBoost third.
- Document any library excluded due to install failure or excessive runtime.

Stretch set:

- SVM with calibrated probabilities.
- TabPFN or TabPFN-like tabular foundation model if practical and license/runtime allow.

## Model Roles

For each family decide whether it is used as:

- scorer only,
- final classifier only,
- scorer and final classifier.

Primary matrix:

```text
scorer_model x final_model
  balanced_oof_relabel
  unbalanced_oof_relabel
  cleanlab/confident-learning baseline
  class_proportional
  no_cleaning
```

Start with diagonal pairs, then test cross-model transfer:

- LR scorer -> XGBoost final.
- XGBoost scorer -> LR final.
- HGB scorer -> HGB final.

## Related Code Files

- Create: `pipeline/models/factories.py`
- Create: `pipeline/models/calibration.py`
- Create: `scripts/run_model_stress_benchmark.py`
- Create: `outputs/model-stress-results.csv`
- Create: `outputs/model-stress-summary.md`
- Modify: `requirements.txt` for `xgboost`, and optionally `lightgbm` or `catboost`.

## Implementation Steps

1. Extract model factories from existing experiment scripts into `pipeline/models/factories.py`.
2. Add dependency-gated factories:
   - if XGBoost/LightGBM/CatBoost missing, fail clearly with install instructions.
3. Add probability calibration options:
   - uncalibrated.
   - isotonic or sigmoid calibration where runtime permits.
4. Run smoke tests on one dataset and two seeds.
5. Run full hidden-minority protocol on all cached datasets.
6. Run reduced non-hidden-minority protocols:
   - symmetric noise.
   - reverse asymmetric noise.
7. Record runtime and failure cases per model.
8. Analyze whether gains come from the scorer, final classifier, or both.
9. Report:
   - mean delta BA.
   - mean delta minority recall.
   - relabel precision.
   - calibration quality.
   - runtime cost.

## Success Criteria

- [ ] XGBoost is included unless installation/runtime is impossible and documented.
- [ ] At least four model families complete the main hidden-minority benchmark.
- [ ] Balanced OOF relabeling beats strong baselines in at least three model families.
- [ ] Results show whether class-balanced scoring matters independently of final classifier.
- [ ] Calibration analysis explains when confidence scores are trustworthy.
- [ ] Runtime table is included for reproducibility.

## Risk Assessment

Tree/boosting probabilities may be poorly calibrated. If calibration changes conclusions,
report the calibrated version as primary and uncalibrated as ablation. If gains only appear
with LR, downgrade the paper claim to a school project or a narrow linear-model result.

## Security Considerations

No secrets. Do not use private datasets or external paid services.

## Next Steps

Proceed to Phase 8 statistical validation after model stress results are complete.

## Unresolved Questions

None. Target is all three boosting libraries (XGBoost, LightGBM, CatBoost) with documented
fallback if runtime or installation is prohibitive.
