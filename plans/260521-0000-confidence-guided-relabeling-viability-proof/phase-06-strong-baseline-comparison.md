---
phase: 6
title: "Strong Baseline Comparison"
status: pending
priority: P1
effort: "2d"
dependencies: [4, 5]
---

# Phase 6: Strong Baseline Comparison

## Overview

Add baselines strong enough that a reviewer cannot dismiss the result as beating only weak
deletion methods. The target is not to reproduce deep noisy-label systems; it is to compare
against practical tabular-compatible interventions.

## Requirements

- Functional: implement or wrap practical baselines under the same budget and split.
- Non-functional: no baseline gets privileged access to clean labels or test labels.

## Architecture

```text
shared dataset split/noisy labels
  -> candidate scoring or baseline selection
  -> intervention with same budget
  -> final model training
  -> paired metrics
```

## Related Code Files

- Create: `pipeline/baselines/cleanlab_baselines.py`
- Create: `pipeline/baselines/confidence_relabeling.py`
- Create: `pipeline/baselines/class_weight_baselines.py`
- Modify: `scripts/run_relabeling_viability_sweep.py`
- Modify: `requirements.txt` only if adding `cleanlab`

## Required Baselines

1. **Cleanlab/CleanLearning filter**
   - Uses confident learning or label-quality scores.
   - Apply as either filtering or relabel suggestion if API supports it.
2. **Self-confidence relabel**
   - OOF model, no class balancing.
   - Relabel majority candidates by `P(minority | x)`.
   - This isolates the value of class balancing.
3. **In-sample confidence relabel**
   - Use only as a leakage/overfitting warning baseline.
   - It may perform well but should not be a deployable baseline.
4. **Random relabel**
   - Same budget, majority pool only.
5. **Class-proportional deletion**
   - Existing strong deletion baseline.
6. **Plain SMOTE**
   - Existing augmentation baseline; report separately because it changes sample count.
7. **Class-weight final training**
   - No relabeling; final LR with `class_weight="balanced"`.
8. **Boosting-native imbalance baselines**
   - XGBoost `scale_pos_weight` / sample weighting.
   - LightGBM `is_unbalance` or `scale_pos_weight` if LightGBM is selected.
   - CatBoost class weights if CatBoost is selected.

## Implementation Steps

1. Add baseline adapters with the same method output schema as existing runners.
2. Use the same train/test split, noise, budget, and final evaluator for all baselines.
3. Add `self_confidence_relabel`:
   - same as balanced OOF relabeling but scorer is unbalanced.
4. Add `cleanlab_filter`:
   - use OOF predicted probabilities when possible.
   - filter top budget suspicious samples or use CleanLearning's supported API.
5. Add `cleanlab_relabel_or_suggest` if cleanlab exposes label suggestions cleanly.
6. Add `class_weight_only` final training baseline.
7. Produce a baseline comparison table by dataset/noise/model.
8. Mark any baseline that is not deployment-safe.

## Success Criteria

- [ ] Balanced OOF relabeling beats unbalanced OOF relabeling often enough to justify the
      "class-balanced" part of the method.
- [ ] Balanced OOF relabeling beats cleanlab filter/relabel on hidden-minority settings or
      clearly trades slightly lower accuracy for better minority recall.
- [ ] Any baseline failure or API limitation is documented, not hidden.
- [ ] Baseline comparison includes effect sizes, not only means.

## Risk Assessment

Cleanlab may be strong enough to beat the method. If so, reposition as an ablation or
operating-condition study unless balanced OOF still wins on minority recall or relabel
precision.

## Security Considerations

No secrets expected. Do not upload data to external services.

## Next Steps

Proceed to Phase 7 for publication-grade model stress testing.

## Unresolved Questions

- Should cleanlab be added as a pinned dependency or isolated in an optional script?
