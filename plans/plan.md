---
title: "Type-B-Directed Cleaning: Noise-Density Budget Allocation"
description: "Test whether routing cleaning budget by OOF noise density (Type B estimation) beats class_proportional at asymmetric class-dependent noise"
status: pending
priority: P1
branch: ""
tags: ["research", "ml", "tbdc", "noise", "imbalance"]
blockedBy: []
blocks: []
created: "2026-05-20T15:43:53.296Z"
createdBy: "ck:plan"
source: skill
---

# Type-B-Directed Cleaning: Noise-Density Budget Allocation

## Core Hypothesis

Under asymmetric class-dependent noise, the optimal cleaning budget allocation routes
budget proportional to **noise density** (where Type B actually is), not class frequency
(what class_proportional uses). At 15% minority with 10% majority→minority noise, Type B
occupies ~45% of the minority pool. class_proportional routes only 15% of budget there.
TBDC routes budget proportional to the OOF disagreement rate per pool — getting it right.

## Theoretical Grounding (Three Fields)

**Noisy Label Learning gap**: Detection quality ≠ cleaning quality. Confident Learning
finds noisy samples but assumes uniform deletion. No paper allocates the deletion budget
by per-class noise density (confirmed: zero papers found in literature survey).

**Imbalanced Learning gap**: ENN/Tomek confound noisy with hard borderline samples.
Class_proportional accidentally works but uses class frequency as a proxy for noise density —
a proxy that fails under asymmetric noise rates.

**Joint NLL+Imbalance gap**: No method models noise directionality. Type A (minority→majority
mislabeling) carries minority feature geometry and should NOT be deleted (oracle paradox:
class_proportional beats oracle in 44/50 conditions, +0.060 BA). Type B (majority→minority
mislabeling) contaminates minority training distribution and MUST be deleted.

## Cap Formula (TBDC)

```
minority_pool     = {i : y_noisy[i] == minority_label}
oof_pred[i]       = argmax(oof_probs[i])   (predicted label for sample i)
disagreement_rate = mean(oof_pred[j] != minority_label for j in minority_pool)
minority_cap      = round(budget_count * disagreement_rate)
majority_cap      = budget_count - minority_cap
```

Key contrasts:
- class_proportional: minority_cap = round(budget × minority_frequency) ≈ 15% of budget
- TBDC: minority_cap = round(budget × disagreement_rate) ≈ 45% of budget (at 15% minority, 10% noise)
- CRCC-Adaptive (FAILED): minority_cap = 0 (all budget to majority) — wrong direction

## Paper Framing Note (red-team: Attack 3)

TBDC is NOT framed as a new method architecture. It is framed as:
"Noise-density-adaptive budget allocation: disagreement rate is a better budget-routing
signal than class frequency under asymmetric class-dependent noise."
The contribution is the ALLOCATION SIGNAL (what drives the cap), not the deletion mechanics.
This is a genuine theoretical departure from all existing methods (confirmed: no paper uses
per-class OOF disagreement rate to set deletion budget).

## What FAILED (do not repeat)

| Approach | Outcome | Reason |
|---|---|---|
| CRCC-Adaptive (zero minority deletion) | NO-GO pilot | Ignores Type B entirely; budget to majority = cleaning Type A = oracle paradox |
| KNN scoring | WORSE than OOF | OOF already 97-100% Type B precision; KNN adds noise |
| Lambda penalty | d=0.000 irrelevant | Cap is the only binding mechanism |

## Pilot Gate (Phase 3)

Phase 2 must be implemented before Phase 3 runs. Phase 3 is the go/no-go:
- TBDC minority recall > class_proportional in ≥ 60% of 10 combos (5 datasets × 2 models)
  at medium noise (default 30%/10%) over 20 seeds
- If FAIL → negative result paper; report TBDC and explain why Type B estimation is
  insufficient at n_minority~75 (OOF calibration failure)

## Phases

| Phase | Name | Status | Effort | Depends On |
|-------|------|--------|--------|------------|
| 1 | [Diagnostic](./phase-01-diagnostic.md) | Pending | 30min | — |
| 2 | [OOF Infrastructure + TBDC Selector](./phase-02-oof-infrastructure-tbdc-selector.md) | Pending | 1h | Phase 1 GO |
| 3 | [Pilot Go/No-Go](./phase-03-pilot-go-no-go.md) | Pending | 30min | Phase 2 |
| 4 | [Full Sweep + Noise Ablation](./phase-04-full-sweep-noise-ablation.md) | Pending | 3h | Phase 3 GO |
| 5 | [Cap Estimation Ablation](./phase-05-cap-estimation-ablation.md) | Pending | 1h | Phase 3 GO |
| 6 | [Statistical Analysis](./phase-06-statistical-analysis.md) | Pending | 1h | Phases 4+5 |

## Scope

- Datasets: pima, credit-g, yeast, ecoli, phoneme (ecoli included — small-sample edge case)
- Models: lr, hgb
- Seeds: 20 — [13,17,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101]
- Noise levels: low (10%/5%), medium (30%/10%), high (40%/20%)
- Minority ratio: 0.15 (existing setup; extreme imbalance tested only in ablation if Phase 3 passes)
- Methods compared: no_cleaning, global_top_loss, class_proportional, crcc_m, tbdc
- Budget: 10% (fixed)

## Key Constraints

- All Python files ≤ 200 lines (split if needed)
- Runtime: /home/than-minh/miniconda3/bin/python3
- Existing pipeline/ structure preserved; changes are additive
- oof_loss.py must stay ≤ 200 lines; if adding out_of_fold_scores pushes over, extract helper
