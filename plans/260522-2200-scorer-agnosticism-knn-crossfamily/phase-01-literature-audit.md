---
phase: 1
title: Literature Audit
status: completed
priority: P1
effort: 1h
dependencies: []
---

# Phase 1: Literature Audit

## Overview

Determine whether k-NN ratio scoring (on majority-labeled samples) and cross-family OOF scoring have appeared as noise detection strategies in prior noisy-label + imbalanced classification literature (Q1/Q2/A*/A venues 2020–2025).

## Verdict: CONFIRMED NOVEL (both)

Report: [plans/reports/scorer-novelty-audit.md](../reports/scorer-novelty-audit.md)

## Key Findings

### Q1: k-NN Ratio Majority Scorer — NOVEL (92% confidence, LOW publish risk)

No published paper uses k-NN ratio (minority-neighbor frequency) on *majority-labeled* samples as a *continuous per-sample weight* simultaneously feeding both downstream reweighting AND minority synthesis direction.

**Near misses and why they differ:**
- **Borderline-SMOTE (2004, variants 2023+)**: k-NN on *minority* samples to find borderline cases — inverse direction.
- **ENDM (Sensors 2020)**: k-NN is a categorical threshold (>50% → noisy), not a continuous score.
- **NI-MWMOTE (ESWA 2020)**: Euclidean distance + density, not k-NN ratio; operates on all samples.
- **Deep k-NN (2020)**: Filters noisy samples entirely; no synthesis or continuous weighting.

**Novelty statement for paper:**
> While k-nearest neighbor methods serve as classical noise detectors, existing approaches either filter suspicious samples entirely (Deep k-NN) or use k-NN proximity to identify *minority-class* instances near decision boundaries (Borderline-SMOTE). CWMS scores *majority-labeled* samples by their minority-neighbor frequency, feeding this suspiciousness signal to both per-sample reweighting and downstream synthesis — inverting the conventional imbalanced-learning paradigm.

### Q2: Cross-Family OOF Scorer — NOVEL (78% confidence, MEDIUM publish risk)

No prior work explicitly uses a *different model family* for OOF noise detection versus the final model in an imbalanced noisy-label context.

**Near misses:**
- **ReCoV (MICCAI 2024), E-NKCVS (IEEE Trans. Cybernetics 2021)**: Cross-validation OOF but same family throughout.
- **Meta-Weight-Net (NeurIPS 2019)**: Different architectures but both neural; no tree-vs-linear heterogeneity.
- **SW Framework (KBS 2022)**: RF space partition → SMOTE, but final model often same family.

**Novelty statement for paper:**
> Ensemble and meta-learning methods have explored multiple classifier families, and model-agnostic noise detection has been demonstrated (ReCoV, 2023). However, no prior work explicitly evaluates *heterogeneous model families for out-of-fold noise scoring* — specifically, training a linear model to detect label noise that a separate tree-based model must overcome. We present this as a novel empirical pattern.

**Risk note:** Q2 is a supporting empirical pattern, not a primary contribution. Reviewers will ask "why not same-family?" — requires ablation to justify. Empirical validation in Phase 3 is mandatory before including in paper.

## Related Work Citations to Add

| Paper | Venue | Year | Differentiation |
|-------|-------|------|-----------------|
| Borderline-SMOTE | Pattern Recognition | 2004 | k-NN on minority; inverse direction |
| Deep k-NN (Bahri et al.) | ICML | 2020 | Filter-only; no continuous score |
| ENDM | Sensors | 2020 | k-NN threshold; 4-metric ensemble |
| NI-MWMOTE | ESWA | 2020 | Distance+density, not k-NN ratio |
| E-NKCVS | IEEE Trans. Cybernetics | 2021 | Same-family OOF; no cross-family |
| SW Framework | KBS | 2022 | RF partition; filter-first paradigm |
| ReCoV | MICCAI | 2024 | CV noise detection; same-family only |

## Success Criteria

- [x] Literature searched across Q1/Q2/A*/A venues 2020–2025
- [x] 40+ papers reviewed
- [x] Both Q1 and Q2 confirmed novel with differentiation statements
- [x] Novelty statements ready for paper Related Work section
- [x] Report saved to plans/reports/scorer-novelty-audit.md
