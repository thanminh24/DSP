---
title: "CWMS+MSBS: Gap Validation, Code Prune, Method Hardening, Paper"
description: "End-to-end plan to take CWMS+MSBS from an empirical finding to a standalone paper: rigorous gap audit against Q1/Q2 A*/A venues (2020-2025), code pruning, method patching based on research, final deep sweep, and paper write-up."
status: in-progress
priority: P1
branch: "master"
tags: ["cwms-msbs", "paper", "gap-audit", "literature", "deep-sweep"]
blockedBy: []
blocks: []
created: "2026-05-22T09:23:47.574Z"
createdBy: "ck:plan"
source: skill
---

# CWMS+MSBS: Gap Validation, Code Prune, Method Hardening, Paper

## Problem Statement

We have a method — CWMS+MSBS — that corrects decision boundaries under **hidden minority label noise** (true minority samples mislabeled as majority) without modifying any training labels. Current empirical evidence:

- LR: +4.8pp BA, p<1e-6, 80% win rate (strong, robust)
- HGB/LightGBM: neutral
- XGBoost: −2pp (incompatible due to `scale_pos_weight` conflict)
- Catboost: untested (deep sweep pending)

The gap claim: *"No existing method corrects the decision boundary under hidden minority label noise without modifying any training labels."*

Before this can become a paper, three things need to happen:
1. Prove the gap is real (no Q1/Q2/A*/A paper 2020–2025 has done this)
2. Patch any weakness the research reveals (differentiation, metric coverage)
3. Run the definitive sweep and write the paper

## Current Codebase State

**Active sweep infrastructure** (`run_relabeling_viability_sweep.py`) still contains 15+ methods that are dead for the paper (relabeling variants, CGMS, cleanlab, etc.) and their supporting pipeline modules. These need to be isolated — not deleted (they inform baselines), but clearly separated so readers see only what matters.

**Dead pipeline modules** (still imported by the sweep):
- `pipeline/augmentation/synthesis.py` — CGMS (failed, −8.5pp)
- `pipeline/augmentation/relabeling.py` — OOF relabeling (discouraged)
- `pipeline/augmentation/filtered_smote.py` — unused
- `pipeline/baselines/cleanlab_baselines.py` — cleanlab wrapper
- `pipeline/baselines/confidence_relabeling.py` — unbalanced OOF baseline
- `pipeline/baselines/class_weight_baselines.py` — class weight only baseline
- `pipeline/models/calibration.py` — unused by active sweep
- `pipeline/data/weak_supervision_loaders.py` — weak supervision (removed from data/)
- `pipeline/cleaning/selectors.py` — contains `select_class_proportional` (KEEP), `select_global`, `select_oracle` (dead for paper)

## Phases

| Phase | Name | Status | Effort | Blocker |
|-------|------|--------|--------|---------|
| 1 | [Literature Gap Audit](./phase-01-literature-gap-audit-2020-2025-q1-q2-a-a.md) | **Completed** | ~2h | none |
| 2 | [Code Audit and Prune](./phase-02-code-audit-and-prune.md) | **Completed** | ~1h | none |
| 3 | [Method Hardening](./phase-03-method-hardening-from-research.md) | **Completed** (Case A, patches 2+3) | ~2h | Phase 1 |
| 4 | [Deep Sweep + Analysis](./phase-04-deep-sweep-and-statistical-analysis.md) | **Completed** (4350 rows) | ~3h compute | Phase 3 |
| 5 | [Paper Packaging](./phase-05-paper-packaging.md) | **In Progress** (draft done) | ~3h writing | Phase 4 |

Phases 1 and 2 are independent — run in parallel.
Phase 3 depends on Phase 1 findings.
Phases 4 and 5 are sequential.

## Key Numbers to Preserve

From v1 full sweep (5 models, hidden_minority_medium, 10 seeds):

| Method | BA | Δ vs cp | p |
|--------|----|---------|----|
| cwms_msbs (LR only) | 0.7454 | +4.22pp | 6.2e-07 |
| class_proportional | 0.7032 | — | — |

From mini-sweep v2 (3 seeds, confirms fix direction):
- LR: +4.86pp, 80% win rate — stable across fix iterations
- XGBoost: −2.06pp — structural incompatibility, not a bug

## Dependencies

Phase 2 is independent and can execute concurrently with Phase 1.
Phase 3 must wait for Phase 1 research report before patching.
