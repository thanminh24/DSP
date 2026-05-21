---
phase: 9
title: "Paper and Project Packaging"
status: pending
priority: P2
effort: "1d"
dependencies: [2, 8]
---

# Phase 9: Paper and Project Packaging

## Overview

Package the work into a school-project deliverable and, if Phase 8 passes, a research-paper
draft. The paper must be honest about what failed: CRCC-P did not beat class-proportional;
balanced OOF relabeling is the usable contribution.

## Requirements

- Functional: produce report, README, reproducibility commands, figures, and paper outline.
- Non-functional: avoid overclaiming and make limitations visible.

## Architecture

```text
validated results
  -> revised docs
  -> project README
  -> final report
  -> paper draft outline
  -> presentation material
```

## Related Code Files

- Create: `README.md`
- Create: `docs/confidence-guided-relabeling-report.md`
- Create: `docs/paper-outline-confidence-guided-relabeling.md`
- Create: `docs/reproducibility-guide.md`
- Modify: `docs/codebase-summary.md`
- Modify: `docs/research-foundation.md`

## Final Project Framing

Working title:

> Confidence-Guided Relabeling for Hidden-Minority Label Noise in Imbalanced Tabular Classification

Core claim:

> When minority examples are mislabeled as majority, deletion-based cleaning can remove
> scarce minority evidence. A class-balanced out-of-fold relabeling rule can recover some
> hidden minority examples and improve minority-sensitive performance over deletion and
> random relabeling baselines.

Narrow contribution:

- A simple post-detection intervention: class-balanced OOF majority-pool relabeling.
- A controlled benchmark showing when relabeling beats deletion.
- A realistic weak-supervision check.
- A top-venue novelty audit.
- A publication-grade model stress test beyond LR.
- A failure analysis showing when the method should not be used.

Do not claim:

- state of the art noisy-label learning.
- first class-aware noisy-label method.
- general label correction for all noise directions.
- success on real data unless Phase 5 supports it.
- publication readiness with LR-only experiments.

## Implementation Steps

1. Write `README.md` with setup, commands, and project summary.
2. Update `docs/codebase-summary.md` to reflect the relabeling focus.
3. Write final report:
   - problem.
   - gap.
   - method.
   - leakage audit.
   - synthetic results.
   - realistic noisy-label results.
   - baselines.
   - limitations.
4. Write paper outline:
   - abstract.
   - introduction.
   - related work.
   - method.
   - experiments.
   - results.
   - limitations.
5. Generate 4 required figures:
   - method diagram.
   - main delta BA/minority recall table.
   - relabel precision vs random.
   - operating-condition/failure-mode plot.
6. Prepare school presentation:
   - why deletion fails.
   - how OOF relabeling works.
   - what passed and failed.
   - real-life relevance.

## Success Criteria

- [ ] README exists and a new user can reproduce the core experiment.
- [ ] Final report clearly separates synthetic and realistic evidence.
- [ ] Paper outline includes limitations and competing methods.
- [ ] Docs mention unresolved questions at the end.
- [ ] No report says CRCC-P beats class-proportional deletion.

## Risk Assessment

The final writeup can still overstate the method. Mitigation: include a "Claim Boundary"
section and an "Operating Conditions" section in both project and paper docs.

## Security Considerations

No confidential data. Use public datasets and local CSV outputs only.

## Next Steps

If paper verdict is GO, choose target venue after reviewing page limits and baseline
expectations.

## Unresolved Questions

- Which target venue or class rubric should determine final report depth?
