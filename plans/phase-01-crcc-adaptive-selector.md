---
phase: 1
title: "CRCC-Adaptive Selector"
status: pending
priority: P1
effort: "1h"
dependencies: [0]
---

# Phase 1: CRCC-Adaptive Selector

## Overview

Add `select_crcc_adaptive()` and `select_majority_only()` to `pipeline/cleaning/selectors.py`.
Also wire both into the experiment pipeline. Only run after Phase 0 shows GO.

## Cap Formula

```
IR = majority_count / minority_count   (from y_noisy)
k  = max(0.0, 1.0 - IR / ir_threshold)
minority_cap = int(np.floor(budget_count * minority_ratio * k))
majority_cap = budget_count - minority_cap
```

No `max(1,…)` floor. Zero is intentional at IR ≥ ir_threshold.

| ratio | IR    | k (thresh=20) | minority_cap (phoneme, budget=405) |
|-------|-------|---------------|------------------------------------|
| 5%    | 19.1  | 0.045         | 0 (zero deletion)                  |
| 7%    | 13.3  | 0.335         | 7                                  |
| 10%   | 9.0   | 0.550         | 16                                 |
| 15%   | 5.7   | 0.715         | 32                                 |

## ir_threshold Sensitivity

ir_threshold is NOT tuned to data — it is a design constant encoding "above IR=20 we protect
minority completely." Sensitivity run tests {10, 15, 20, 25} in Phase 2. Reported as robustness
check, not a hyperparameter grid.

## Related Code Files

- Modify: `pipeline/cleaning/selectors.py` (currently 142 lines → stays ≤ 200)
- Modify: `pipeline/core/config.py` (add `ir_threshold: float = 20.0`)
- Modify: `pipeline/core/experiment.py` (wire new methods into dispatch)

## Implementation Steps

1. Read `pipeline/cleaning/selectors.py`
2. Append after `select_crcc_m()`:

```python
def select_majority_only(
    suspiciousness: np.ndarray,
    y_noisy: np.ndarray,
    budget_count: int,
    minority_label: int = 1,
) -> np.ndarray:
    """Trivial baseline: never delete minority. Delete majority by suspiciousness rank only."""
    if budget_count <= 0:
        return np.array([], dtype=int)
    majority_idx = np.where(y_noisy != minority_label)[0]
    ranked = majority_idx[np.argsort(suspiciousness[majority_idx])[::-1]]
    return ranked[:budget_count]


def select_crcc_adaptive(
    suspiciousness: np.ndarray,
    y_noisy: np.ndarray,
    budget_count: int,
    lambda_risk: float,
    minority_label: int = 1,
    ir_threshold: float = 20.0,
) -> np.ndarray:
    """CRCC-Adaptive: IR-aware cap with no hard min floor.

    cap = floor(budget * minority_ratio * max(0, 1 - IR/ir_threshold))
    Above ir_threshold (default IR=20, ~5% minority): minority_cap = 0.
    Transitions smoothly below threshold — not a hard switch.
    """
    if budget_count <= 0:
        return np.array([], dtype=int)
    minority_count = int(np.sum(y_noisy == minority_label))
    majority_count = len(y_noisy) - minority_count
    ir = majority_count / max(minority_count, 1)
    minority_ratio = minority_count / len(y_noisy)
    k = max(0.0, 1.0 - ir / ir_threshold)
    minority_cap = int(np.floor(budget_count * minority_ratio * k))
    majority_cap = budget_count - minority_cap

    class_caps = {
        label: minority_cap if label == minority_label else majority_cap
        for label in np.unique(y_noisy)
    }
    class_risk = {label: 0.0 for label in np.unique(y_noisy)}
    class_risk[minority_label] = 1.0
    adjusted = np.array([
        score - lambda_risk * class_risk[int(label)]
        for score, label in zip(suspiciousness, y_noisy)
    ])
    deleted_count = {label: 0 for label in np.unique(y_noisy)}
    selected: list[int] = []
    for idx in np.argsort(adjusted)[::-1]:
        label = int(y_noisy[idx])
        if len(selected) >= budget_count:
            break
        if deleted_count[label] < class_caps[label]:
            selected.append(int(idx))
            deleted_count[label] += 1
    return np.array(selected, dtype=int)
```

3. In `pipeline/core/config.py`:
   - Add `ir_threshold: float = 20.0` to `BaseExperimentConfig`
   - Add `"crcc_adaptive"` and `"majority_only"` to `method_names` property

4. In `pipeline/core/experiment.py`:
   - Grep for `select_crcc_m` call to find dispatch pattern
   - Add `"crcc_adaptive"` and `"majority_only"` dispatch entries

5. Verify: run smoke test, confirm `crcc_adaptive` and `majority_only` appear in output

## Success Criteria

- [ ] Both selectors added; `selectors.py` ≤ 200 lines
- [ ] Phoneme at 5%: `select_crcc_adaptive` returns 0 minority indices, `majority_only` same
- [ ] `method_names` includes both new methods
- [ ] Smoke test passes with new methods visible in output

## Risk Assessment

- `majority_cap > len(majority_idx)`: greedy loop runs out of majority samples before hitting cap.
  Output len < budget_count. Acceptable — partial budget.
- At ratio=0.15 with ir_threshold=20: k=0.715, minority_cap=32 (phoneme). Behavioral check
  needed: does adaptive cap at 0.15 ratio equal CRCC-M at same ratio? They should be close but
  not identical (CRCC-M uses 0.5 multiplier; adaptive uses 1-IR/20 = 0.715).
