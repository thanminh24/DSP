---
phase: 7
title: "Experiment Report"
status: completed
priority: P2
effort: "2h"
dependencies: [6]
---

# Phase 7: Experiment Report

## Overview

Write `docs/experiment-report.md` — a concise technical report covering problem, method, experimental setup, results, and limitations. Report should be self-contained and suitable as a draft for a Springer LNCS-style conference submission.

## Requirements

- Functional:
  - Sections: Abstract, Problem, Related Work (brief), Method, Experiment Setup, Results, Limitations, Conclusion
  - Results section tables populated from `outputs/summary-table.csv`
  - 2-3 figure references (plots from phase 6)
- Non-functional: ≤ 2500 words (excluding tables); concise academic tone.

## Architecture / Report Structure

```markdown
# Class-Risk-Constrained Label Cleaning for Imbalanced Tabular ML

## Abstract
3-4 sentences: problem, method, key finding.

## 1. Problem
- Label noise + class imbalance in tabular ML
- Global top-loss cleaning harms minority-class recall
- Gap: no lightweight post-detection rule for this setting

## 2. Related Work
- Confident Learning (Northcutt 2021): detection foundation
- Deep noisy-label methods (DivideMix, Co-teaching): out of scope for tabular
- Class-aware methods (Liu 2024, Sheng 2024): deep-learning integrated, not post-detection rules
- Active label cleaning (Bernhardt 2022): budgeted setting, different intervention type

## 3. Method: CRCC
- Framing: separate detection from intervention
- CRCC score: adjusted_score = suspiciousness - λ × class_risk
- Deletion rule: greedy loop with per-class caps
- Variants: CRCC-P (proportional cap), CRCC-M (minority-protected cap)
- Clean-minority deletion rate: harm metric

## 4. Experimental Setup
- Datasets: Pima Indians (OpenML 37), Credit-G (OpenML 31), Sick/Thyroid (OpenML 38)
- Imbalance: 85/15 (or natural if stronger)
- Noise: class-dependent, 30% minority→majority, 10% majority→minority
- Budget: 10% of training set
- Models: LogisticRegression, HistGradientBoostingClassifier
- Seeds: 5 (results as mean ± std)
- Baselines: no cleaning, random deletion, global top-loss, class-proportional deletion, oracle deletion

## 5. Results
- Table 1: Main results — all methods × all datasets × LR model
- Table 2: Main results — HGB model
- Figure 1: Clean-minority deletion rate (main harm metric)
- Figure 2: Minority recall
- Key finding: CRCC-P (λ=0.5) reduces CMDR vs global top-loss on [N] of 3 datasets
- Lambda ablation: Table 3 or Figure 4

## 6. Limitations
- Synthetic noise: class-dependent but single noise rate across all datasets
- Single noise level (10% minority→majority base rate per class)
- Tabular scope only
- Oracle is an upper bound, not a practical method

## 7. Conclusion
- CRCC offers a simple, model-agnostic post-detection rule
- Reduces clean-minority deletion harm vs global top-loss while maintaining comparable balanced accuracy
- (Or: negative result if CRCC fails — document conditions)
```

## Related Code Files

- Create: `docs/experiment-report.md`
- Read: `outputs/summary-table.csv`, `outputs/plots/*.png`
- Reference: `docs/tabular-class-risk-capped-label-cleaning-proposal.md` (proposal)

## Implementation Steps

1. Read `outputs/summary-table.csv` to extract key numbers.

2. Check main success condition:
   - For each of 3 datasets: does `crcc_p_l05` have lower `clean_minority_deletion_rate` than `global_top_loss`?
   - Record count of datasets where condition holds.
   - If ≥ 2: frame as positive result. If < 2: frame as negative result per proposal.

3. Write `docs/experiment-report.md` following structure above:
   - Use actual numbers from summary table in results section
   - Reference figures as `![Figure 1](../outputs/plots/fig1-clean-minority-deletion-rate.png)`
   - Address the 3 unresolved questions from the proposal:
     a. Primary class-risk definition: binary (minority=1) used as primary; mean-loss version noted as future work
     b. CRCC-M: included as ablation with comparison vs CRCC-P
     c. Sick preprocessing: resolved (OpenML fetch + HGB native categoricals)

4. Add a "Reviewer Risk Mitigation" appendix (brief):
   - "If reviewer says this is stratified deletion": point to risk-adjusted ranking
   - "If reviewer says synthetic noise is artificial": necessity for clean oracle
   - "If reviewer says only tabular": scope is intentional

5. Proofread for:
   - No plan artifact references (no "phase 3", "F1 finding", etc.)
   - All numbers match the actual CSV output
   - Failure condition properly documented if applicable

## Success Criteria

- [ ] `docs/experiment-report.md` written and ≤ 2500 words
- [ ] Results section contains actual numbers from `outputs/summary-table.csv`
- [ ] Main success condition explicitly stated (positive or negative framing)
- [ ] All 3 unresolved questions from proposal addressed
- [ ] Figure references point to actual generated PNGs
- [ ] No references to plan artifacts in the report text

## Risk Assessment

- If CRCC fails on all 3 datasets: the proposal explicitly allows a negative result reframe. Document clearly: "under this noise model, risk-adjusted ranking did not reduce harm beyond class-proportional caps."
- Numbers must match plots — write report after plots are confirmed correct.
