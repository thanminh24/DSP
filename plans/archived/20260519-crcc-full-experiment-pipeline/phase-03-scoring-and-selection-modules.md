---
phase: 3
title: "Scoring and Selection Modules"
status: completed
priority: P1
effort: "2h"
dependencies: [2]
---

# Phase 3: Scoring and Selection Modules

## Overview

Write two modules: `scripts/scoring.py` (out-of-fold cross-entropy suspiciousness) and `scripts/selectors.py` (all 7 cleaning-method selectors including oracle, CRCC-P, CRCC-M). These are pure functions with no side effects.

## Requirements

- Functional:
  - `scoring.py`: `out_of_fold_loss(X, y, model_factory, n_splits=5, seed=42) -> np.ndarray`
  - `selectors.py`: one `select_*` function per method, all returning `np.ndarray` of indices to delete
  - Methods: `no_cleaning`, `random_deletion`, `global_top_loss`, `class_proportional`, `oracle_deletion`, `crcc_p`, `crcc_m`
- Non-functional: each module ≤ 200 lines; no sklearn imports in selectors.py (pure numpy).

## Architecture

### scoring.py

```
out_of_fold_loss(X, y_noisy, model_factory, n_splits, seed) -> suspiciousness[n]
  └─ StratifiedKFold(n_splits, shuffle=True, random_state=seed)
  └─ for each fold: fit model_factory() on train split, predict_proba on valid split
  └─ collect OOF probabilities → cross-entropy loss per sample
  └─ suspiciousness[i] = -log(p_hat[i, y_noisy[i]])  clipped at 1e-12
```

`model_factory` is a callable → fitted sklearn Pipeline. The orchestrator passes the right factory per model family.

### selectors.py

All selectors share signature: `select_*(args...) -> np.ndarray[int]` (indices to delete, length ≤ budget_count).

```
select_none(n) -> empty array

select_random(n, budget_count, rng) -> rng.choice(n, budget_count, replace=False)

select_global(suspiciousness, budget_count)
  -> argsort(suspiciousness)[::-1][:budget_count]

select_class_proportional(suspiciousness, y_noisy, budget_count)
  -> for each class: top (budget × class_freq) by suspiciousness

select_oracle(noisy_mask, budget_count)
  -> indices where noisy_mask=True, up to budget_count
  -> ties broken by index order (deterministic)

select_crcc_p(suspiciousness, y_noisy, budget_count, lambda_risk, minority_label=1)
  -> class_risk = {majority: 0.0, minority: 1.0}
  -> adjusted = suspiciousness - lambda_risk × class_risk[y_noisy]
  -> class_cap[c] = round(budget_count × freq[c])  [proportional]
  -> greedy loop: delete top adjusted-score sample if class cap not exceeded

select_crcc_m(suspiciousness, y_noisy, budget_count, lambda_risk, minority_label=1,
              minority_cap_factor=0.5)
  -> minority class cap = floor(budget_count × minority_ratio × minority_cap_factor)
  -> majority class cap = budget_count - minority_cap
  -> same adjusted-score ranking as crcc_p
```

**Lambda grid for CRCC-P:** The orchestrator calls `select_crcc_p` once per lambda value in `{0.0, 0.25, 0.5, 1.0}` — selectors.py itself is stateless and lambda-agnostic.

**Note on CRCC-P vs class-proportional at λ=0:** When `lambda_risk=0`, adjusted scores equal raw suspiciousness. CRCC-P then ranks globally but still applies proportional caps — this is slightly different from `select_class_proportional` which ranks within each class. This preserves a meaningful λ=0 ablation point.

## Related Code Files

- Create: `scripts/scoring.py`
- Create: `scripts/selectors.py`
- Read for context: `scripts/run-crcc-smoke-test.py` (port `out_of_fold_suspiciousness`, `select_crcc`, `select_class_proportional`)

## Implementation Steps

1. Read `scripts/run-crcc-smoke-test.py` to extract existing selector logic.

2. Write `scripts/scoring.py`:
   - `out_of_fold_loss(X, y_noisy, model_factory, n_splits=5, seed=42)`:
     - StratifiedKFold, collect OOF probabilities
     - Return `-np.log(np.clip(p_oof[arange, y_noisy], 1e-12, 1.0))`
   - Add `if __name__ == "__main__":` smoke check using breast cancer + LR

3. Write `scripts/selectors.py`:
   - `select_none(n_samples)` → `np.array([], dtype=int)`
   - `select_random(n_samples, budget_count, rng)` → `rng.choice(arange(n), budget, replace=False)`
   - `select_global(suspiciousness, budget_count)` → argsort descending, take top
   - `select_class_proportional(suspiciousness, y_noisy, budget_count)`:
     - Per-class budget = `round(budget × freq[c])`
     - Rank within class by suspiciousness descending
     - Concatenate, truncate to budget_count
   - `select_oracle(noisy_mask, budget_count)`:
     - `np.where(noisy_mask)[0][:budget_count]`
   - `select_crcc_p(suspiciousness, y_noisy, budget_count, lambda_risk, minority_label=1)`:
     - Port from smoke test `select_crcc`, generalize for any lambda
     - Proportional cap: `round(budget × freq[c])` per class
     - Adjusted score: `suspiciousness - lambda_risk × (1.0 if label == minority_label else 0.0)`
   - `select_crcc_m(suspiciousness, y_noisy, budget_count, lambda_risk, minority_label=1, minority_cap_factor=0.5)`:
     - Minority cap: `floor(budget × minority_freq × minority_cap_factor)`
     - Majority cap: `budget - minority_cap`
     - Same adjusted-score ranking as crcc_p

4. Add `if __name__ == "__main__":` in selectors.py that runs all 7 selectors on random data and asserts output lengths ≤ budget.

## Success Criteria

- [ ] `out_of_fold_loss` returns array of length `n_train`, all values ≥ 0
- [ ] `select_none` returns empty array
- [ ] `select_random` returns exactly `budget_count` unique indices
- [ ] `select_global` returns top-loss indices in descending order
- [ ] `select_class_proportional` respects per-class budgets (no class over-represented)
- [ ] `select_oracle` returns only indices where `noisy_mask=True`, ≤ budget
- [ ] `select_crcc_p` with `lambda_risk=0` produces globally-ranked, cap-constrained result
- [ ] `select_crcc_m` minority cap is strictly smaller than `select_crcc_p` minority cap
- [ ] Both CRCC selectors: total deleted ≤ budget_count; per-class deleted ≤ cap
- [ ] Both modules ≤ 200 lines

## Risk Assessment

- Sick/Credit-G categorical columns don't affect selectors (selectors only see suspiciousness scores and labels).
- Class cap rounding edge case: if `round(budget × freq[minority])` = 0 for very small minority, clamp to `max(1, ...)`.
- Oracle selector may return fewer than `budget_count` if true noise count < budget — this is correct behavior; document in docstring.
