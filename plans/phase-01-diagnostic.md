---
phase: 1
title: "Diagnostic: OOF Disagreement Rate vs True Type B Count"
status: pending
priority: P1
effort: "30min"
dependencies: []
---

# Phase 1: Diagnostic

## Overview

Before writing any new selector code, verify that the OOF disagreement rate is a reliable
proxy for Type B count. This determines whether the TBDC cap formula will be accurate enough
to outperform class_proportional, or whether it systematically over/under-estimates noise.

## Why This First

The TBDC cap formula is `minority_cap = round(budget × disagreement_rate_minority)`.
If OOF disagreement_rate overestimates Type B (because hard-but-clean minority are also
predicted as majority), we over-clean minority pool and hurt minority recall — the exact
problem we're trying to fix. This diagnostic quantifies the overestimation before we
commit to the formula.

The diagnostic uses EXISTING pipeline infrastructure only — no new code added to the
production pipeline.

## Related Code Files

- Create: `scripts/run_diagnostic_tbdc.py` (≤ 80 lines, standalone)
- Read: `pipeline/scoring/oof_loss.py` (need to expose probabilities)
- Read: `pipeline/data/loaders.py` (inject_noise, induce_imbalance)
- Read: `outputs/full-experiment-results.csv` (existing results for context)

## Implementation Steps

1. Write `scripts/run_diagnostic_tbdc.py`:
   - For each of {pima, credit-g, yeast, phoneme} × {lr, hgb} × 5 seeds:
     a. Load data, induce 15% minority, inject medium noise (30%/10%)
     b. Run OOF (reuse `out_of_fold_loss` but also save the full probabilities matrix —
        modify the call temporarily OR duplicate the CV loop inline in the diagnostic)
     c. Compute for minority pool:
        - `true_type_b_count` = count(noisy_mask[minority_pool] == True) (these are majority relabeled as minority)
        - `oof_disagreement_count` = count(argmax(oof_probs[minority_pool]) != minority_label)
        - `overestimation_ratio` = oof_disagreement_count / max(true_type_b_count, 1)
        - `tbdc_cap` = round(budget × disagreement_rate) vs `prop_cap` = round(budget × minority_freq)
     d. Record: dataset, model, seed, true_type_b_count, oof_disagreement_count,
        overestimation_ratio, tbdc_cap, prop_cap, budget_count

2. Print summary table: mean overestimation_ratio per dataset. Flag datasets where ratio > 2.0.

3. GO/NO-GO for Phase 2:
   - Mean overestimation_ratio < 2.0 across all datasets → TBDC cap formula is usable
   - Overestimation ratio 1.0–1.5 → ideal (mild over-estimate, still cleanable)
   - Ratio > 2.0 on a majority of datasets → Phase 2 needs a damping floor:
     `minority_cap = round(budget × disagreement_rate × 0.7)` (40% damping)
   - Ratio < 1.0 → underestimate → TBDC will under-clean Type B; still better than class_proportional

4. FPR-at-depth measurement (addresses red-team circularity concern):
   For each (dataset, model, seed), rank minority pool samples by OOF loss descending.
   Walk from depth=1 to depth=full minority pool. At each depth k:
   - fpr_at_k = fraction of top-k minority pool samples that are clean (noisy_mask=False)
   Record fpr_10pct_depth = depth fraction at which fpr first exceeds 0.10.
   Compare to tbdc_cap / minority_pool_size — TBDC must stay below fpr_10pct_depth.

   Hard gate for Phase 2:
   mean(tbdc_cap / minority_pool_size) < mean(fpr_10pct_depth) across all datasets.
   If TBDC's estimated cap fraction exceeds fpr_10pct_depth → set damping so minority_cap
   stays at or below the depth where clean minority contamination begins.

   ```python
   sorted_minority = minority_pool[np.argsort(suspiciousness[minority_pool])[::-1]]
   is_clean = ~noisy_mask[sorted_minority]
   fpr_at_depth = np.cumsum(is_clean) / (np.arange(len(sorted_minority)) + 1)
   exceed_idx = np.where(fpr_at_depth > 0.10)[0]
   fpr_10pct_depth = exceed_idx[0] / len(sorted_minority) if len(exceed_idx) > 0 else 1.0
   ```

Note on probabilities: `out_of_fold_loss` computes but discards the full probability matrix.
For the diagnostic, duplicate the CV loop inline (don't modify oof_loss.py yet) to keep the diagnostic self-contained and not change production code.

```python
# Inline CV loop to get both loss AND probabilities (diagnostic only)
def _oof_scores_and_probs(X, y_noisy, factory, n_splits=5, seed=42):
    n = len(y_noisy)
    n_cls = len(np.unique(y_noisy))
    probs = np.zeros((n, n_cls), dtype=float)
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr_idx, va_idx in folds.split(X, y_noisy):
        m = factory()
        m.fit(X[tr_idx], y_noisy[tr_idx])
        probs[va_idx] = m.predict_proba(X[va_idx])
    losses = -np.log(np.clip(probs[np.arange(n), y_noisy], 1e-12, 1.0))
    return losses, probs
```

## Success Criteria

- [ ] `scripts/run_diagnostic_tbdc.py` runs in < 5 min, produces console summary table
- [ ] overestimation_ratio reported per dataset/model/seed
- [ ] Mean overestimation_ratio across all combos is documented
- [ ] GO/NO-GO decision documented in plan.md after running (update this phase status)
- [ ] If ratio > 2.0 on ≥ 3 datasets: document damping factor needed before Phase 2

## Risk Assessment

- If OOF probabilities at minority n~75 are uncalibrated (hard samples mislabeled as noisy):
  overestimation_ratio could be 2–4×. This is addressable by applying a 0.5–0.7 damping factor.
  The diagnostic quantifies this precisely before any production code changes.
- Ecoli (smallest dataset) may show extreme ratios — expected; it's an edge case and
  results should be reported separately.
