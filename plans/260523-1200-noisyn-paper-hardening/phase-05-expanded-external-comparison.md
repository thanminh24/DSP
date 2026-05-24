---
phase: 5
title: "Expanded External Comparison"
status: pending
priority: P1
effort: "2h"
dependencies: [2, 3]
---

# Phase 5: Expanded External Comparison

## Overview

Expand Table 2 from LR×medium to LR+SVM+HGB × all 3 noise protocols × 15 datasets. This directly fixes C1 (single condition cherry-pick) and uses the lamda sensitivity result from Phase 2 to set the correct IW-SMOTE lamda.

## Related Code Files

- Modify: `scripts/run_competitor_headtohead.py` — add models, protocols, datasets
- Create: `outputs/competitor-headtohead-v2.csv`
- Modify: `scripts/analyze_competitor_headtohead.py` — new aggregation tables

## Implementation Steps

### Step 1 — Decide IW-SMOTE lamda based on Phase 2 result

Before running, check Phase 2 (Sweep C) output:
- If IW-SMOTE at lamda=100 gives BA > lamda=30 by >0.5pp on average → use lamda=100
- Otherwise → keep lamda=30 and add sensitivity note to paper

### Step 2 — Update run_competitor_headtohead.py

```python
COMPETITOR_METHODS = [
    "no_cleaning",
    "class_proportional",
    "smote",
    "iw_smote",
    "sw_framework",
    "cwms_msbs",
]
# Expand from LR-only to 3 models
COMPETITOR_MODELS = ["lr", "svm", "hgb"]

# Expand from medium-only to all 3 protocols
COMPETITOR_PROTOCOLS = [
    "hidden_minority_low",
    "hidden_minority_medium",
    "hidden_minority_high",
]

# Use 15 datasets from Phase 3
DATASETS = ALL_15_DATASETS  # from updated loaders

OUTPUT_CSV = PROJECT_ROOT / "outputs" / "competitor-headtohead-v2.csv"

# Rows: 6 methods × 3 models × 15 datasets × 10 seeds × 3 protocols = 8,100 rows
```

**IW-SMOTE lamda**: Read Phase 2 Sweep C result (`outputs/iw-lamda-sweep.csv`) before running:
- If BA(lamda=100) > BA(lamda=30) by >0.5pp on average → pass `lamda=100` to `iw_smote()` call
- Otherwise → keep `lamda=30`, add footnote: "Sensitivity confirms lamda=30 ≡ lamda=100 (ΔBA<0.5pp)"

The `iw_smote()` signature: `iw_smote(X, y_noisy, minority_label, majority_label, budget_count, lamda=100, ...)` — current call passes `lamda=30`. Update to `lamda=CHOSEN_LAMDA`.

### Step 3 — Run

```bash
/home/than-minh/miniconda3/envs/dsp/bin/python scripts/run_competitor_headtohead.py \
  --gpu --output outputs/competitor-headtohead-v2.csv
```

Estimated runtime: 2h (LR and HGB fast; SVM moderate).

### Step 4 — Updated analysis: three aggregation levels

Update `scripts/analyze_competitor_headtohead.py` to produce:

**Table 2a — Aggregate (mean over models × protocols × datasets)**
```
Method | BA | G-mean | Min-F1 | Min-Rec | ΔBA vs IW-SMOTE | p-value
```

**Table 2b — Per-model breakdown**
```
               LR      SVM     HGB
no_cleaning   0.xxx   0.xxx   0.xxx
SMOTE         ...
IW-SMOTE      ...
NoiSyn        ...     ...     ...
```

**Table 2c — Per-protocol breakdown (LR only)**
```
           low    medium   high
IW-SMOTE  0.xxx  0.727   0.xxx
NoiSyn    0.xxx  0.745   0.xxx
Δ         +x.xx  +1.84   +x.xx
```

**Table 2d — Per-dataset: where does IW-SMOTE win?** (honest reporting)

Wilcoxon: CWMS+MSBS vs IW-SMOTE paired over (dataset, seed, protocol) per model. n = 15×10×3 = 450 pairs per model.

### Step 5 — Handle SW-approx decision

Based on Phase 2 result (whether SW-approx can be validated):
- If validation shows approximation is adequate: keep in Table 2 with dagger footnote
- If approximation is clearly weaker than original: remove from Table 2, move to Appendix as "approximate reproduction"

## Success Criteria

- [ ] `outputs/competitor-headtohead-v2.csv` has **8,100 rows** (6 methods × 3 models × 15 datasets × 10 seeds × 3 protocols)
- [ ] Zero NaN BA for LR, SVM, HGB rows
- [ ] Tables 2a–2d produced by updated analysis script
- [ ] NoiSyn vs IW-SMOTE Wilcoxon run at per-model level with n=450 pairs
- [ ] SW-approx decision documented
- [ ] IW-SMOTE lamda choice documented (30 or 100 per Phase 2 sensitivity), propagated to `iw_smote()` call
- [ ] `outputs/competitor-headtohead.csv` (300-row original, LR × medium) unchanged

## Risk Assessment

- **SVM × 15 datasets × 3 protocols**: SVM is the bottleneck. Estimate 3–5h for SVM alone. Mitigation: run SVM in a separate background process; HGB and LR finish fast in parallel.
- **IW-SMOTE × 15 datasets × lamda=100**: If lamda=100 is required, IW-SMOTE becomes 3× slower (450,000+ CART fits). Mitigation: run with lamda=50 as a compromise if lamda=100 times out.
- **Table 2 story**: If HGB+NoiSyn does not outperform IW-SMOTE, Table 2 narrows to LR+SVM only. That is a valid, honest narrowing — not a failure.
