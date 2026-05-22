---
phase: 3
title: "Method Hardening from Research"
status: completed
priority: P1
effort: "~2h"
dependencies: [1]
---

# Phase 3: Method Hardening from Research

## Overview

Apply targeted patches to the CWMS+MSBS method and experimental design based on
Phase 1 gap audit findings. This phase is conditional — what gets patched depends on
what Phase 1 found. Section headers below describe the decision tree.

## Decision Tree Post-Phase-1

### Case A: Gap fully confirmed (no overlapping paper found)

No method changes required. Two documentation patches needed:

1. **Add cross-scorer experiment** to address reviewer G1 concern (circular scorer).
   Run 1 experiment where the OOF scorer is a different family than the final model:
   - OOF scorer: `hgb` → final model: `lr` (cross-family)
   - Compare to same-family baseline
   If cross-family result is similar, G1 is addressed empirically, not just as a limitation.

2. **Add budget sensitivity analysis** to address G3 (budget covers full noisy pool).
   Run cwms_msbs vs class_proportional at budget = 0.05, 0.10, 0.20.
   Show the method works across budget levels, not just the "saturated" one.

### Case B: SW (2022) overlaps (symmetric noise version is also ours)

Add one differentiating design element:

**Option B1 — Asymmetric noise framing** (cheapest):
SW (2022) targets symmetric/random noise. Reframe our paper explicitly around
*asymmetric hidden minority noise* and show SW performs poorly when noise is
one-directional (minority-to-majority only). Add SW as a baseline in the sweep.

**Option B2 — Confidence score differentiation** (medium):
SW uses space partitioning (geometric), we use instance-level OOF confidence scores
from the same model family that will train. Show this makes our weights more targeted.
Add a "random weight" ablation (shuffle CWMS weights → control for just having
sample_weight ≠ 1) to prove the scores are load-bearing.

Both options can be combined. B1 is the minimum required if SW overlaps.

### Case C: Direct conflict (a paper does exactly what we do)

Stop. Redesign contribution framing:
- Shift to "we apply existing technique to a new specific noise type" (narrow application)
- OR add a genuinely new element (e.g., adaptive budget, confidence calibration)
This case is unlikely but must be planned for.

## Unconditional Patches (regardless of Phase 1 outcome)

These address known weaknesses from the code review and mini-sweep analysis.

### Patch 1 — Non-boosting CWMS has no minority upweighting (finding M2)

For `lr` in `cwms` and `cwms_msbs`, `confidence_weighted_sample_weights` sets
minority weight = 1.0. But `cwms_factory` uses `balanced=False`, so LR has no class
correction. This means CWMS corrects boundary noise but doesn't address class imbalance.

Fix: in the non-boosting branch, set minority weight = `n_majority / n_minority` (same
formula as sklearn `class_weight="balanced"`). Call the updated function
`confidence_weighted_sample_weights_balanced` with the computed imbalance ratio.

```python
# In run_single_viability(), compute imbalance ratio for non-boosting use:
n_min = (y_noisy == minority_label).sum()
n_maj = (y_noisy == majority_label).sum()
lr_spw = n_maj / n_min  # same as sklearn class_weight="balanced"

# In cwms dispatcher for lr:
sw = confidence_weighted_sample_weights_balanced(y_noisy, bal_scores, maj_label, scale_pos_weight=lr_spw)
```

Use `make_cwms_factory` (scale_pos_weight=1.0) consistently for all models.
This unifies the non-boosting and boosting paths.

### Patch 2 — Add "shuffled CWMS" ablation for the deep sweep

Add method `cwms_msbs_shuffled` that uses randomly shuffled `bal_scores` for CWMS weights.
This proves the confidence scores are doing real work (not just adding any sample_weight).

```python
if method == "cwms_msbs_shuffled":
    shuffled = bal_scores.copy()
    valid = ~np.isnan(shuffled)
    shuffled[valid] = rng.permutation(shuffled[valid])
    # same as cwms_msbs but with shuffled weights
    ...
```

If `cwms_msbs` >> `cwms_msbs_shuffled`: the OOF confidence scores are load-bearing.
This directly addresses reviewer concern G1.

### Patch 3 — Explicit XGBoost incompatibility documentation

Add a code-level comment in the CWMS factory and dispatcher explaining why
`scale_pos_weight` + CWMS don't compose, and that this is a known design limitation:

```python
# XGBoost note: scale_pos_weight and CWMS sample_weight create conflicting corrections.
# Even with scale_pos_weight=1.0 (cwms_factory), XGBoost shows -2pp vs class_proportional
# in our experiments. Root cause: XGBoost's internal GBDT mechanism does not benefit
# from the same per-sample boundary suppression that directly scales LR gradients.
# Exclude XGBoost from paper main results; include as "out-of-scope" model.
```

Add XGBoost to the `calibrated_lr`-style skip logic in cwms/cwms_msbs dispatchers,
returning a NaN row with a skip reason. This prevents misleading negative results
in the deep sweep polluting the aggregate statistics.

## Related Code Files

- Modify: `scripts/run_relabeling_viability_sweep.py` — Patch 1, Patch 2, Patch 3
- Modify: `pipeline/baselines/soft_weighting.py` — Patch 1 (unified formula)
- Modify: `scripts/run_cwms_msbs_deep_sweep.py` — add shuffled ablation method
- Create: `scripts/run_cross_scorer_experiment.py` — Case A, cross-scorer validation

## Success Criteria

- [ ] Phase 1 report read and verdict determined (Case A / B / C)
- [ ] Case B action items executed if SW overlaps
- [ ] Patch 1: minority upweighting added to non-boosting CWMS path
- [ ] Patch 2: `cwms_msbs_shuffled` added to sweep
- [ ] Patch 3: XGBoost skip logic + documentation added
- [ ] Smoke test: `cwms_msbs` still gives +4.5pp+ for LR on pima/seed=13
- [ ] Smoke test: `cwms_msbs_shuffled` gives lower BA than `cwms_msbs` (confirms scores are load-bearing)

## Risk Assessment

- **Patch 1 may change LR results**: adding minority upweighting to the non-boosting
  CWMS path changes the effective sample weights for LR. The v1 result (+4.22pp) was
  WITHOUT this patch. Patch 1 may increase or decrease the LR delta. Run the smoke test
  before the full sweep to confirm direction.
- **Patch 2 (shuffled ablation) adds compute**: 1 extra method per combo = +20% sweep time.
  Worth it — the ablation directly answers the most likely reviewer objection.
- **Case C**: if a direct conflict is found, stop and consult before proceeding. Do not
  try to paper over a fundamental conflict.
