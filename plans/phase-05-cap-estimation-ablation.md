---
phase: 5
title: "Cap Estimation Ablation"
status: pending
priority: P2
effort: "1h"
dependencies: [3]
---

# Phase 5: Cap Estimation Ablation

## Overview

Compare three strategies for estimating the minority pool noise density (the cap-setting
signal). This shows WHICH estimation strategy drives TBDC's gains and whether disagreement
rate is the right choice vs Confident Learning thresholding or a simpler fixed multiplier.

## The Three Strategies

| Variant | Cap Formula | Notes |
|---|---|---|
| `tbdc_disagree` | `round(budget × mean(OOF_pred≠minority))` | Main TBDC method (Phase 2 implementation) |
| `tbdc_cl` | `round(budget × CL_estimated_noise_rate)` | Confident Learning threshold per Northcutt 2021 |
| `tbdc_fixed` | `round(budget × fixed_minority_frac × k)` | k tuned to match observed noise rate |

**CL threshold formula (tbdc_cl)**:
```
threshold_min = mean(P̂(y=minority | x_i) for i where ỹ_i = minority)
cl_noisy_count = count(ỹ_i = minority AND P̂(y=minority | x_i) < threshold_min)
minority_cap = min(cl_noisy_count, budget_count)
```

**Fixed multiplier (tbdc_fixed)**:
```
minority_cap = round(budget × minority_ratio × k)   where k = 3.0
```
(At 15% minority + 10% majority→minority noise, expected Type B / minority pool ≈ 45%,
so k=3.0 approximates routing 45% of budget to minority: 15% × 3.0 = 45%.)

## Related Code Files

- Modify: `pipeline/cleaning/selectors.py` (add `select_tbdc_cl`, `select_tbdc_fixed`)
- Modify: `pipeline/core/config.py` (add new method names)
- Create: `scripts/run_cap_ablation.py` (≤ 100 lines)

## Implementation Steps

1. Add `select_tbdc_cl()` to selectors.py:

```python
def select_tbdc_cl(
    suspiciousness: np.ndarray,
    oof_probs: np.ndarray,
    y_noisy: np.ndarray,
    budget_count: int,
    minority_label: int = 1,
) -> np.ndarray:
    """TBDC with Confident Learning noise rate estimate for minority cap.

    Uses per-class CL threshold (mean P̂(y=c|x)) to estimate noisy count.
    """
    if budget_count <= 0:
        return np.array([], dtype=int)
    minority_pool = np.where(y_noisy == minority_label)[0]
    if len(minority_pool) == 0:
        return select_global(suspiciousness, budget_count)
    min_probs = oof_probs[minority_pool, minority_label]
    threshold = float(np.mean(min_probs))
    cl_noisy_count = int(np.sum(min_probs < threshold))
    minority_cap = min(cl_noisy_count, budget_count, len(minority_pool))
    majority_cap = budget_count - minority_cap
    # same greedy loop as select_tbdc
    class_caps = {minority_label: minority_cap}
    for lbl in np.unique(y_noisy):
        if lbl != minority_label:
            class_caps[lbl] = majority_cap
    deleted_count = {lbl: 0 for lbl in np.unique(y_noisy)}
    selected: list[int] = []
    for idx in np.argsort(suspiciousness)[::-1]:
        lbl = int(y_noisy[idx])
        if len(selected) >= budget_count:
            break
        if deleted_count[lbl] < class_caps[lbl]:
            selected.append(int(idx))
            deleted_count[lbl] += 1
    return np.array(selected, dtype=int)
```

2. Add `select_tbdc_fixed(k=3.0)` — identical structure but cap = round(budget × minority_ratio × k).

3. **Check selectors.py line count after both additions.** If > 200 lines, extract the shared
   greedy-deletion loop into `_greedy_delete(suspiciousness, class_caps, budget_count)` helper.

4. Write `scripts/run_cap_ablation.py`:
   - Runs on medium noise (30%/10%) only
   - 5 datasets × 2 models × 20 seeds
   - Methods: class_proportional, tbdc_disagree, tbdc_cl, tbdc_fixed_k3
   - Output: `outputs/cap-ablation-results.csv`

5. Analysis: for each variant, compute:
   - Mean (estimated_noisy_count / true_type_b_count) — accuracy of cap estimate
   - BA win rate vs class_proportional
   - Minority recall win rate vs class_proportional

## Key Question

Does the accuracy of the noise estimate predict TBDC's performance gain?
If tbdc_cl has a more accurate cap estimate than tbdc_disagree but worse BA:
→ The cap accuracy is not the bottleneck; something else is (e.g., over-deletion of
  genuine hard-but-clean minority near decision boundary).

## Success Criteria

- [ ] `select_tbdc_cl()` and `select_tbdc_fixed()` added; selectors.py ≤ 200 lines
- [ ] `scripts/run_cap_ablation.py` runs and produces `outputs/cap-ablation-results.csv`
- [ ] Cap accuracy (estimated/true Type B) reported per variant
- [ ] BA and recall win rate vs class_proportional reported per variant
- [ ] Recommendation: which cap strategy to use in the paper's main table

## Risk Assessment

- If all three TBDC variants perform similarly: report as "cap estimation strategy is
  robust" — any reasonable noise density estimate produces similar results. This strengthens
  the method (not sensitive to exact formula).
- If tbdc_cl underperforms tbdc_disagree: confirms Phase 1 finding that CL thresholding
  is unreliable at n_minority~75; hard argmax is more robust.
- selectors.py line count risk: at 142 lines + ~35 (tbdc) + ~35 (tbdc_cl) + ~15 (tbdc_fixed)
  = 227 lines → OVER 200. Must extract `_greedy_delete` helper first.
