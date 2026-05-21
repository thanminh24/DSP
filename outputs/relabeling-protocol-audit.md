# Relabeling Protocol Audit

## Verdict

**PASS**

## Checks

- **PASS** `out-of-fold scoring isolation`: balanced scorer must fit on tr_idx and score va_idx only
- **PASS** `clean-label isolation`: clean labels may be used only after relabel indices are selected
- **PASS** `active runner balanced relabel selection inputs`: balanced relabel selection must use y_noisy, OOF scores, budget, and labels only
- **PASS** `split before imbalance/noise`: train/test split must happen before training-only imbalance and noise injection
- **PASS** `relabeler has no clean-label inputs`: relabeler must use y_noisy and scores only
- **PASS** `baseline fairness surface`: core baselines must share split, noisy labels, and budget
- **PASS** `poison-pill controls present`: shuffled/inverted score relabel controls must be present in active METHODS
- **PASS** `stale archive ignored`: archive folders should stay out of publication context

## Unresolved Questions

None.
