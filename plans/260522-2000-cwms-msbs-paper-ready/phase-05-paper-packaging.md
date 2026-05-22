---
phase: 5
title: "Paper Packaging"
status: in-progress
priority: P1
effort: "~3h writing"
dependencies: [4]
---

# Phase 5: Paper Packaging

## Overview

Write the paper. Structure is an 8-section format targeting ECML-PKDD or AISTATS
(2026 submission). All tables come directly from Phase 4 analysis. This phase is
purely documentation — no new experiments.

## Target Venue (Tentative)

**Primary:** ECML-PKDD 2026 (A*, January deadline) — tabular ML + label noise is a
natural fit; strong representation of imbalanced learning work.

**Backup:** KDD 2026 Research Track, or AISTATS 2026.

**Alternative (if LR-only scope persists):** TMLR (rolling) or MLJ (Q1 journal) —
these allow narrower, complete empirical studies without requiring sweeping claims.

## Paper Structure

### Abstract (template — fill numbers from Phase 4)

> Hidden minority label noise — where true minority samples are systematically mislabeled
> as majority — degrades decision boundary quality in imbalanced tabular classification.
> Existing approaches either delete suspicious samples (losing scarce minority evidence)
> or relabel them (introducing label corruption). We propose **CWMS+MSBS**, which corrects
> the decision boundary without modifying any training labels: MSBS synthesizes new minority
> samples toward the contaminated boundary, while CWMS suppresses suspicious majority
> samples via confidence-derived per-sample weights. Both components reuse OOF confidence
> scores already computed for detection, requiring no extra model training. On 5 tabular
> benchmarks across [N] model families, CWMS+MSBS achieves **+[X]pp balanced accuracy**
> and **+[X]pp minority recall** over confidence-guided deletion (p=[X]), while preserving
> all original training labels. An ablation with shuffled confidence weights confirms the
> OOF scores are the active mechanism.

### Section 1 — Introduction

- Frame: imbalanced tabular + hidden minority noise is prevalent (citation needed)
- Problem: standard deletion-based cleaning removes minority evidence; relabeling corrupts labels
- Gap: no method does zero-label-modification boundary correction for this noise type
- Contribution: CWMS+MSBS, confirmed novel against 2020–2025 Q1/Q2/A*/A literature
- Scope: tested on 5 datasets × 4 model families; method requires sample_weight gradient scaling (LR + HGB family)

### Section 2 — Related Work (from Phase 1 gap audit)

Three subsections:
- **Label noise learning**: DivideMix, CORES, Co-teaching, etc. — all modify labels or require multiple models
- **Imbalanced learning**: SMOTE variants, class weighting — don't address noise
- **Combined (noise + imbalance)**: SW (2022) and related — different noise type or mechanism
- Position: "We are the first to..."

### Section 3 — Problem Formulation

- Define hidden minority label noise formally: ε_mn (minority→majority flip rate), ε_mj (majority→minority)
- Define hidden minority protocol: ε_mn >> ε_mj (one-directional)
- Objective: recover balanced accuracy / minority recall under this contamination

### Section 4 — Method (CWMS+MSBS)

Two subsections:
- **MSBS**: seeds, synthesis direction, why toward high-confidence majority (they mark the boundary)
- **CWMS**: weight formula, why minority weight = SPW, why this doesn't modify labels
- **Combined pass**: single training iteration, no extra model
- **Complexity**: O(N log N) for kNN + O(N·K·T) for model training — same as a standard model fit

### Section 5 — Experimental Setup

- Datasets table (5 datasets, key statistics)
- Noise protocol table (low/medium/high; ε_mn, ε_mj values)
- Model families table (in-scope vs out-of-scope with mechanistic reason)
- Metrics: BA (primary), minority recall, minority precision, weighted F1
- Baselines: no_cleaning, class_proportional, msbs (ablation), cwms (ablation)

### Section 6 — Results

- Table 1: summary across all 4 models (hidden_minority_medium)
- Table 2: per-model breakdown with p-values
- Table 3: multi-metric LR results (headline model)
- Table 4: ablation — cwms_msbs vs cwms_msbs_shuffled (proves scores load-bearing)
- Table 5: cross-protocol robustness (low/medium/high)

### Section 7 — Discussion

- Why LR benefits most: sample_weight is exact gradient scaling
- Why HGB/LightGBM benefit less: already partially robust to class imbalance
- Why XGBoost is excluded: scale_pos_weight conflict (mechanistic, not a failure)
- Limitation 1: scorer and final model same family (G1 concern; addressed by shuffled ablation)
- Limitation 2: budget defined relative to training set size, not noisy pool
- Limitation 3: 5 datasets; future work: larger benchmarks

### Section 8 — Conclusion

1-paragraph. State the finding, the mechanism, the venue (tabular + hidden minority).

## Files to Write / Update

| File | Action |
|------|--------|
| `README.md` | Update with Phase 4 numbers; keep as paper landing page |
| `docs/paper-outline.md` | Update with final structure from this phase |
| `docs/codebase-summary.md` | Final state: active scripts, outputs, env |
| `docs/reproducibility-guide.md` | Step-by-step to reproduce Phase 4 sweep |
| `paper/` directory (create) | LaTeX or markdown draft of the 8 sections |

## Success Criteria

- [ ] Abstract filled with Phase 4 numbers
- [ ] Related work section covers all papers from Phase 1 gap audit
- [ ] All 5 result tables present and consistent with Phase 4 CSV
- [ ] Shuffled ablation table included
- [ ] XGBoost exclusion explained mechanistically in methods/discussion
- [ ] Reproducibility guide updated with `run_cwms_msbs_deep_sweep.py` command
- [ ] README verdict table shows Phase 4 final numbers
- [ ] `paper/` directory created with draft skeleton

## Risk Assessment

- **LR-only scope**: if Phase 4 confirms catboost/HGB also don't benefit, paper becomes
  "LR + evidence from other models." Still publishable, but weaker claim. Adjust abstract
  scope accordingly — do not overstate.
- **Shuffled ablation inconclusive**: if `cwms_msbs_shuffled` is close to `cwms_msbs`,
  acknowledge in limitations. The method still works (LR result stands), but the mechanism
  argument weakens.
- **SW (2022) handled in Phase 3**: if differentiation was added in Phase 3, ensure the
  related work section clearly explains the distinction.
