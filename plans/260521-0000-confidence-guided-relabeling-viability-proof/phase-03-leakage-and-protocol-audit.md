---
phase: 3
title: "Leakage and Protocol Audit"
status: pending
priority: P1
effort: "1d"
dependencies: [1, 2]
---

# Phase 3: Leakage and Protocol Audit

## Overview

Audit the existing balanced OOF relabeling protocol for label leakage, test leakage,
selection leakage, and unfair baseline comparisons. This phase is a hard gate: if it
fails, do not use current results as paper evidence.

## Requirements

- Functional: produce automated checks and a written audit report.
- Non-functional: audit must distinguish allowed synthetic-only diagnostics from
  information used by deployable methods.

## Architecture

```text
experiment runner
  -> split audit
  -> OOF audit
  -> clean-label access audit
  -> baseline fairness audit
  -> reproducibility audit
  -> pass/fail report
```

## Related Code Files

- Create: `scripts/audit_relabeling_protocol.py`
- Create: `outputs/relabeling-protocol-audit.md`
- Read: `scripts/run_augment_experiment.py`
- Read: `pipeline/scoring/balanced_oof.py`
- Read: `pipeline/augmentation/relabeling.py`
- Read: `pipeline/evaluation/augment_metrics.py`

## Implementation Steps

1. Verify train/test separation:
   - test labels are never passed into scorer, selector, relabeler, or augmenter.
   - train/test split happens before imbalance/noise injection.
2. Verify OOF separation:
   - each relabel candidate score is produced only from a model that did not train on that
     candidate.
   - no candidate is scored with an in-sample fitted final model.
3. Verify clean-label isolation:
   - `y_tr` is used only for synthetic metric reporting such as `relabel_correctness`.
   - `y_tr` is never used to choose `rel_idx`.
4. Verify baseline fairness:
   - all methods use the same train/test split, same noisy labels, same budget, and same
     final evaluator.
   - random relabel uses the same number of relabeled examples.
5. Verify preprocessing safety:
   - imputation/scaling for LR are fitted inside the model pipeline, not globally.
   - categorical encoding does not use target labels.
6. Add a poison-pill check:
   - shuffle OOF scores or invert them; method should lose advantage.
   - if shuffled scores still win, current gain is not from confidence signal.
7. Add a report section listing deployable inputs versus synthetic-only diagnostics.

## Success Criteria

- [ ] Audit script exits non-zero on any leakage finding.
- [ ] Report explicitly says PASS or FAIL.
- [ ] Shuffled-score control does not beat real balanced OOF relabeling.
- [ ] Clean labels are confirmed absent from selection logic.
- [ ] Any unavoidable synthetic-only metric is labeled as non-deployable.

## Risk Assessment

If leakage is found, fix the protocol before continuing. Do not reinterpret leaked results
as "still useful"; rerun all downstream experiments.

## Security Considerations

No secrets expected. Do not inspect environment files.

## Next Steps

Only continue to Phase 4 after an audit PASS.

## Unresolved Questions

None.
