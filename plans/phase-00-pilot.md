---
phase: 0
title: "Pilot Go/No-Go"
status: pending
priority: P1
effort: "20min"
dependencies: []
---

# Phase 0: Pilot Go/No-Go

## Overview

Confirm signal exists BEFORE writing any code. Simulate CRCC-Adaptive by patching CRCC-M's cap
to zero for phoneme + yeast at ratio=0.05, 20 seeds. If recall gain ≥ 0.02 → proceed to Phase 1.
If < 0.02 → stop and pivot.

## Why Phase 0 First

At phoneme 5% minority: class_proportional deletes 15 minority samples, adaptive deletes 0.
That's 10% of minority protected. If this doesn't show up in recall with 20 seeds, the method
fails even in the best-case dataset. No point writing 200 lines of new code.

## What "Simulate" Means

We don't need the new selector yet. We just need:
- class_proportional: uses current `select_class_proportional` (selects 15 minority from phoneme)
- simulated_adaptive: `select_majority_only` — exists after we add it to selectors.py, OR
  we hardcode: select from majority only using global top-loss on majority indices

Actually we can implement this as a one-function script using existing code:

```python
# Simulated adaptive: never delete minority
# All budget goes to majority by suspiciousness rank
def _sim_adaptive(suspiciousness, y_noisy, budget_count, minority_label):
    majority_idx = np.where(y_noisy != minority_label)[0]
    ranked = majority_idx[np.argsort(suspiciousness[majority_idx])[::-1]]
    return ranked[:budget_count]
```

This is exactly what CRCC-Adaptive does at IR ≥ ir_threshold (cap=0). It's also the
`majority_only` baseline we plan to implement. They're identical at ratio=5%, phoneme (IR=19,
adaptive cap=0).

## Script

`scripts/run_pilot.py` — standalone, ≤ 80 lines, runs in < 5 min:

```python
"""Pilot: simulate CRCC-Adaptive (zero minority cap) vs class_proportional.
Tests phoneme + yeast at ratio=0.05. Go/no-go for full implementation.

Run: python3 scripts/run_pilot.py
Expected: recall gain ≥ 0.02 on both datasets (else pivot)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from pipeline.data.loaders import load_dataset, induce_imbalance, inject_noise
from pipeline.scoring.oof_loss import out_of_fold_loss
from pipeline.cleaning.selectors import select_class_proportional

SEEDS = [13,17,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101]
RATIO = 0.05
BUDGET = 0.10
DATASETS = ["phoneme", "yeast"]
MINORITY_LABEL = 1

def make_lr(seed):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))

def sim_adaptive(suspiciousness, y_noisy, budget_count):
    """Zero minority deletion: rank majority by suspiciousness, take top budget."""
    maj = np.where(y_noisy != MINORITY_LABEL)[0]
    return maj[np.argsort(suspiciousness[maj])[::-1]][:budget_count]

rows = []
for ds in DATASETS:
    X_raw, y_raw, cat_cols, _ = load_dataset(ds)
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_raw.values if hasattr(X_raw, "values") else X_raw,
            y_raw, test_size=0.25, stratify=y_raw, random_state=seed
        )
        # induce_imbalance expects DataFrame; skip for simplicity — use raw split
        X_tr_f = X_tr.astype(float) if not isinstance(X_tr, np.ndarray) else X_tr
        X_te_f = X_te.astype(float) if not isinstance(X_te, np.ndarray) else X_te
        # subsample to target ratio
        minority_idx = np.where(y_tr == MINORITY_LABEL)[0]
        majority_idx = np.where(y_tr != MINORITY_LABEL)[0]
        target_n = int((RATIO / (1 - RATIO)) * len(majority_idx))
        keep = rng.choice(minority_idx, size=min(len(minority_idx), max(2, target_n)), replace=False)
        keep_idx = np.concatenate([majority_idx, keep])
        rng.shuffle(keep_idx)
        X_tr_f, y_tr_sub = X_tr_f[keep_idx], y_tr[keep_idx]
        if int(np.sum(y_tr_sub == MINORITY_LABEL)) < 20:
            continue
        y_noisy, noisy_mask = inject_noise(y_tr_sub, minority_label=MINORITY_LABEL, rng=rng)
        n = len(y_noisy)
        budget_count = max(1, int(round(BUDGET * n)))
        factory = lambda s=seed: make_lr(s)
        susp = out_of_fold_loss(X_tr_f, y_noisy, factory, n_splits=5, seed=seed)
        for method, sel in [
            ("class_prop", select_class_proportional(susp, y_noisy, budget_count)),
            ("sim_adaptive", sim_adaptive(susp, y_noisy, budget_count)),
        ]:
            keep_mask = np.ones(n, dtype=bool); keep_mask[sel] = False
            model = make_lr(seed); model.fit(X_tr_f[keep_mask], y_noisy[keep_mask])
            pred = model.predict(X_te_f)
            rows.append({"dataset": ds, "seed": seed, "method": method,
                "ba": balanced_accuracy_score(y_te, pred),
                "recall": recall_score(y_te, pred, pos_label=MINORITY_LABEL)})

df = pd.DataFrame(rows)
print("\n=== PILOT RESULTS: class_prop vs sim_adaptive at ratio=5% ===")
for ds in DATASETS:
    sub = df[df["dataset"] == ds]
    cp = sub[sub["method"]=="class_prop"]["recall"].mean()
    sa = sub[sub["method"]=="sim_adaptive"]["recall"].mean()
    print(f"  {ds}: class_prop recall={cp:.4f}  sim_adaptive={sa:.4f}  gain={sa-cp:+.4f}")
    go = "GO" if (sa - cp) >= 0.02 else "NO-GO"
    print(f"  --> {go} (threshold: +0.02)")
```

## Go/No-Go Decision

| Result | Decision |
|--------|----------|
| Both datasets gain ≥ 0.02 recall | GO — implement Phase 1 |
| One dataset gains, one doesn't | PARTIAL GO — note which and investigate why |
| Neither gains | NO-GO — adaptive cap doesn't work; pivot to purely analysis paper |

## Related Code Files

- Create: `scripts/run_pilot.py` (≤ 80 lines)

## Success Criteria

- [ ] Script runs in < 5 minutes
- [ ] Both phoneme and yeast show recall gain direction (positive delta)
- [ ] At least phoneme gain ≥ 0.02 (expected ~0.04–0.08 based on 15-sample cap protection)
- [ ] Clear go/no-go decision documented in output

## Risk Assessment

- If phoneme gain < 0.02: the issue is the majority noise signal — adaptive is deleting majority
  samples that are harder to identify correctly. Check if global_top_loss on majority is also
  hurting BA. This would mean the suspiciousness score is not reliably detecting majority noise
  at extreme imbalance either.
- Pilot uses LR only (not HGB) to save time. Phase 2 tests both.
