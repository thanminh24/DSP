---
phase: 5
title: "Realistic Weak-Supervision Benchmark"
status: pending
priority: P1
effort: "2d"
dependencies: [3, 4]
---

# Phase 5: Realistic Weak-Supervision Benchmark

## Overview

Add at least one meaningful dataset where noisy labels come from imperfect labeling rules,
not artificial flips. This is the phase that turns the project from a synthetic experiment
into a paper-worthy empirical study.

## Requirements

- Functional: load realistic noisy labels and clean ground-truth labels for evaluation.
- Non-functional: no method may use ground truth for selection; ground truth is evaluation
  only.

## Architecture

```text
weak-supervision dataset
  -> features X
  -> noisy weak labels y_noisy
  -> ground-truth labels y_true for eval only
  -> same relabeling/baseline runner
  -> downstream clean-test evaluation
```

## Related Code Files

- Create: `pipeline/data/weak_supervision_loaders.py`
- Create: `scripts/download_weak_supervision_datasets.py`
- Create: `scripts/run_weak_supervision_relabeling.py`
- Create: `outputs/weak-supervision-relabeling-results.csv`
- Modify: `docs/research-foundation.md` or create a plan report, after results exist.

## Candidate Datasets

Priority order:

1. WRENCH-style tabular weak-label datasets:
   - bank-marketing
   - bioresponse
   - mushroom
   - phishing
   - spambase
2. Other weak-supervision datasets with explicit noisy labels and gold labels.
3. A real audit subset if available from coursework or public data, with human-reviewed
   labels.

Selection rule:

- Choose the first dataset that can be loaded reproducibly and has both weak/noisy labels
  and clean labels.
- Prefer binary tabular tasks with non-trivial minority class.

## Pre-condition: WRENCH Loader Smoke Test

**Before writing any loader code**, run a manual loading test on this machine:

1. Install `wrench` or download the raw WRENCH dataset files manually.
2. Attempt to load one dataset (e.g. `bank-marketing`) using the WRENCH Python API or
   direct parquet/CSV path.
3. Confirm Windows path separators and file encoding do not cause errors.
4. If loading succeeds → proceed with `pipeline/data/weak_supervision_loaders.py`.
5. If loading fails on all candidates → document the failure, narrow paper scope to
   controlled synthetic evidence with a negative real-world transfer note, and skip the
   remaining steps in this phase.

Do not write the loader or create the download script until the smoke test passes.

## Implementation Steps

1. Run the WRENCH smoke test above. Stop here if it fails.
2. Identify one or two reproducible weak-supervision tabular datasets.
3. Cache them under `data/weak-supervision/` or `data/` with clear provenance.
3. Implement loader returning:
   - `X`
   - `y_noisy`
   - `y_true`
   - categorical columns
   - metadata with source and label-generation notes.
4. Split using `y_true` only for stratified evaluation if necessary; do not expose it to
   relabel selection.
5. Run methods from Phase 4 using `y_noisy` for training and selection.
6. Evaluate final models on clean held-out `y_true`.
7. Compute relabel correctness using `y_true` only after selection.
8. Write a report comparing synthetic and realistic results.

## Success Criteria

- [ ] At least one realistic noisy-label tabular dataset is included.
- [ ] Dataset provenance and label semantics are documented.
- [ ] Ground truth is used only for evaluation and post-hoc correctness metrics.
- [ ] Balanced OOF relabeling beats at least two practical baselines on at least one
      meaningful metric: balanced accuracy, minority recall, or macro-F1.
- [ ] If it fails, the failure is documented and the paper scope is reduced to controlled
      synthetic evidence plus negative real-world transfer.

## Risk Assessment

Weak-supervision noise may not be minority-to-majority. If so, the method may fail. That is
still useful: it defines the method's valid operating condition.

## Security Considerations

Use public datasets only. Do not use private or personally identifiable data.

## Next Steps

Proceed to Phase 6 to ensure baselines are fair and strong.

## Unresolved Questions

- WRENCH smoke test result is unknown. Run before proceeding. If it fails, scope narrows
  to synthetic-only evidence.
