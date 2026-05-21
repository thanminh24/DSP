---
phase: 1
title: "Framing and Gap Audit"
status: pending
priority: P1
effort: "1d"
dependencies: []
---

# Phase 1: Framing and Gap Audit

## Overview

Define the defensible research gap and rewrite the project around confidence-guided
relabeling, not CRCC deletion. The deliverable is a short related-work matrix and a
claim boundary that can survive reviewer pushback.

## Requirements

- Functional: identify nearest competing methods and classify whether they are detection,
  deletion/filtering, relabeling, active cleaning, or deep integrated training.
- Non-functional: every claim must be backed by a source or by project output.

## Architecture

```text
literature -> gap matrix -> revised problem framing -> accepted/rejected claims
```

## Related Code Files

- Create: `plans/260521-0000-confidence-guided-relabeling-viability-proof/research/gap-matrix.md`
- Create: `plans/260521-0000-confidence-guided-relabeling-viability-proof/reports/framing-decision.md`
- Read: `docs/research-foundation.md`
- Read: `docs/experiment-report.md`
- Read: `outputs/augment-final-verdict.md`

## Implementation Steps

1. Build a related-work matrix with these rows: Confident Learning/cleanlab, CBS/CSA,
   IJCAI long-tailed noisy sample selection, ITEM/debiased sample selection, active label
   cleaning, weak-supervision noisy-label benchmarks, and tabular noisy-label detection.
2. For each row, record:
   - uses tabular data?
   - uses deep representation learning?
   - automated or human relabeling?
   - deletes, reweights, relabels, or semi-supervises?
   - handles class imbalance explicitly?
   - can be realistically implemented as a baseline?
3. Rewrite the contribution as:
   - primary: balanced OOF confidence-guided relabeling for hidden minority recovery.
   - secondary: deletion baselines show why relabeling is better than removal.
   - negative: CRCC-P collapses to class-proportional deletion under current setup.
4. Define unsafe claims and safe replacements.
5. Choose paper title candidates and one final working title.

## Success Criteria

- [ ] Gap matrix covers at least 8 nearby methods/papers/tools.
- [ ] Final framing does not claim "first class-aware noisy-label method."
- [ ] Final framing names at least one method that has done something similar.
- [ ] Final framing explains why our method is still different: lightweight, tabular,
      post-detection, OOF, asymmetric hidden-minority relabeling.

## Risk Assessment

The biggest risk is overclaiming novelty. Mitigation: position the paper as an empirical
intervention study, not a broad noisy-label-learning breakthrough.

## Security Considerations

None.

## Next Steps

Proceed to Phase 2 for top-venue gap confirmation before running any new experiments.

## Unresolved Questions

None.
