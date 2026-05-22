---
phase: 1
title: "Literature Gap Audit (2020-2025 Q1/Q2 A*/A)"
status: completed
priority: P1
effort: "~2h"
dependencies: []
---

# Phase 1: Literature Gap Audit (2020-2025 Q1/Q2 A*/A)

## Overview

Systematic literature search to confirm our gap is real: no Q1/Q2/A*/A paper from
2020–2025 has published zero-label-modification boundary correction under hidden minority
label noise. Output is a research report that either confirms the gap or identifies
what must change to differentiate.

## Venue Scope

**Accepted venues (A*/A + Q1/Q2):**

| Tier | Venues |
|------|--------|
| A* conferences | NeurIPS, ICML, ICLR, KDD, AAAI, IJCAI |
| A conferences | ECML-PKDD, AISTATS, UAI, CIKM, SDM |
| Q1 journals | JMLR, TPAMI, TKDE, Information Sciences, Knowledge-Based Systems, Pattern Recognition, Expert Systems with Applications, Neurocomputing |
| Q2 journals | IJML, Applied Intelligence, Data Mining and Knowledge Discovery |

**Years:** 2020–2025 (push the cutoff back as requested to catch any earlier precedent).
**Primary focus for recency claims:** 2024–2025.

## One-Sentence Gap to Validate

> "No existing paper corrects the decision boundary under **hidden minority label noise**
> (minority samples mislabeled as majority) using per-sample weighting and minority
> boundary synthesis, **without modifying any training labels**."

## Search Strategy

Spawn **three parallel researcher agents** with distinct mandates:

### Agent 1 — Noisy Labels + Imbalanced Learning intersection

Search for papers combining label noise handling with class imbalance correction on
tabular data. Key terms:

- "hidden minority noise" OR "minority label noise" OR "mislabeled minority"
- "imbalanced + label noise" tabular classification
- "asymmetric label noise imbalanced"
- "noisy imbalanced tabular"

For each paper found: record (a) does it target hidden minority specifically or symmetric noise,
(b) does it modify labels, (c) does it use sample weighting + synthesis combined.

### Agent 2 — Sample Weighting under Label Noise

Search for papers using sample weighting / instance weighting specifically to handle
noisy majority-labeled samples in imbalanced settings. Key terms:

- "sample weight noisy label imbalanced"
- "instance weight label noise classification"
- "confidence weighted noisy majority"
- "reweighting noisy samples tabular"

Flag: SW (2022) — "*A weighted space division framework for imbalanced problems with
label noise*" (Knowledge-Based Systems) — determine if it targets hidden minority or
symmetric noise, and whether it uses synthesis.

### Agent 3 — Boundary Synthesis under Label Noise (SMOTE variants)

Search for papers combining SMOTE-type synthesis with label noise robustness. Key terms:

- "SMOTE label noise" OR "oversampling noisy labels"
- "synthetic minority noise robust"
- "boundary synthesis label noise imbalanced"
- "noise-robust oversampling tabular"

Focus: do any of these (a) generate samples specifically toward the noisy boundary,
(b) combine weighting + synthesis without relabeling?

## What Each Agent Must Report

For every paper identified as a potential overlap:

1. **Citation**: title, venue, year, authors
2. **Noise type**: hidden minority / symmetric / asymmetric / unspecified
3. **Method**: what exactly does it do? (relabel / filter / weight / synthesize)
4. **Label modification**: YES / NO — does the method change any training labels?
5. **Differentiation**: one sentence on how our method differs if there is overlap

## Output

Save report to: `plans/reports/gap-audit-q1q2-astar-2020-2025.md`

Report structure:
```markdown
# Gap Audit Report

## Verdict: [CONFIRMED NOVEL / PARTIAL OVERLAP / CONFLICT]

## Papers Checked

| Paper | Venue | Year | Noise Type | Modifies Labels | Overlap? | Differentiator |
|-------|-------|------|-----------|-----------------|----------|----------------|

## Papers Requiring Action

[For each paper with overlap: specific differentiation needed]

## Confirmed Gap Statement

[Final 1-2 sentence gap suitable for paper abstract]
```

## Success Criteria

- [ ] All three agents complete their searches
- [ ] SW (2022) scope confirmed (symmetric vs. hidden minority)
- [ ] No A*/A or Q1/Q2 paper 2020–2025 found with zero-label-modification boundary correction under hidden minority noise
- [ ] If overlap found: specific differentiation strategy identified
- [ ] Gap statement ready for paper abstract

## Risk Assessment

- **SW (2022) overlap**: most likely threat. Differentiation if it overlaps: (a) SW uses
  space division (global geometry), our method uses per-sample OOF confidence (instance-level);
  (b) SW doesn't target hidden minority specifically; (c) SW requires separate preprocessing,
  our method is single-pass and reuses already-computed OOF scores.
- **CrowdTeacher / co-teaching overlap**: these use two networks and require multiple models.
  Our method is single-model, single-pass, no auxiliary network.
- **Robust-GBDT (2024)**: targets gradient boosting, not LR. Different model family and
  different mechanism. Low overlap risk.
