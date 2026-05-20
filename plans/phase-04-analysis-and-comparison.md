---
phase: 4
title: "Analysis and Comparison"
status: pending
priority: P1
effort: "1h"
dependencies: [2]
---

# Phase 4: Analysis and Comparison

## Overview

Evaluate all pre-registered hypotheses (H1–H5) from Phase 2. Produce the paper's main results
table, the IR-threshold transition plot narrative, and the final method comparison.

## Analysis Script

`scripts/analyze_imbalance_sweep.py` (≤ 100 lines):

```python
import pandas as pd, numpy as np
from pathlib import Path

df = pd.read_csv("outputs/imbalance-sweep-results.csv")

# H1: crcc_adaptive recall > class_proportional at ratio <= 0.07
print("=== H1: crcc_adaptive vs class_proportional recall by ratio ===")
methods = ["no_cleaning","global_top_loss","class_proportional","crcc_m","crcc_adaptive","majority_only"]
for ratio in sorted(df["minority_ratio"].unique()):
    sub = df[df["minority_ratio"] == ratio]
    cp = sub[sub["method"]=="class_proportional"]["minority_recall"]
    ca = sub[sub["method"]=="crcc_adaptive"]["minority_recall"]
    mo = sub[sub["method"]=="majority_only"]["minority_recall"]
    n_combos = len(cp)
    pct_better = (ca.values[:len(cp)] > cp.values).mean() * 100
    print(f"  ratio={ratio:.2f} IR~{sub['ir'].mean():.1f}: "
          f"cp={cp.mean():.4f} crcc_a={ca.mean():.4f} maj_only={mo.mean():.4f} "
          f"delta={ca.mean()-cp.mean():+.4f} better_in={pct_better:.0f}%")

# H2: majority_only vs crcc_adaptive
print("\n=== H2: majority_only vs crcc_adaptive ===")
for ratio in sorted(df["minority_ratio"].unique()):
    sub = df[df["minority_ratio"] == ratio]
    ca = sub[sub["method"]=="crcc_adaptive"]["minority_recall"].mean()
    mo = sub[sub["method"]=="majority_only"]["minority_recall"].mean()
    verdict = "adaptive BETTER" if ca > mo else ("SAME" if abs(ca-mo)<0.001 else "majority_only BETTER")
    print(f"  ratio={ratio:.2f}: adaptive={ca:.4f} majority_only={mo:.4f} --> {verdict}")

# H5: synthetic
print("\n=== H5: synthetic dataset ===")
syn = df[df["dataset"]=="synthetic"]
if not syn.empty:
    for ratio in sorted(syn["minority_ratio"].unique()):
        s = syn[syn["minority_ratio"]==ratio]
        cp = s[s["method"]=="class_proportional"]["minority_recall"].mean()
        ca = s[s["method"]=="crcc_adaptive"]["minority_recall"].mean()
        print(f"  synthetic ratio={ratio:.2f}: cp={cp:.4f} adaptive={ca:.4f} gain={ca-cp:+.4f}")

# Cohen's d
def cohens_d(a, b):
    pooled = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else float("nan")

print("\n=== Cohen's d: crcc_adaptive vs class_proportional (minority_recall) ===")
for ratio in sorted(df["minority_ratio"].unique()):
    sub = df[df["minority_ratio"] == ratio]
    # Pair by seed+dataset+model for paired comparison
    merged = sub[sub["method"]=="crcc_adaptive"][["seed","dataset","model","minority_recall"]].merge(
        sub[sub["method"]=="class_proportional"][["seed","dataset","model","minority_recall"]],
        on=["seed","dataset","model"], suffixes=("_a","_cp")
    )
    if len(merged) < 5:
        continue
    d = cohens_d(merged["minority_recall_a"].values, merged["minority_recall_cp"].values)
    print(f"  ratio={ratio:.2f}: d={d:.3f} (n_pairs={len(merged)})")
```

## H1 Pass/Fail Verdicts

After running, fill in this table:

| Hypothesis | Criterion | Result | Verdict |
|---|---|---|---|
| H1 | ≥60% combos recall gain at ratio≤0.07 | TBD | TBD |
| H2 | majority_only ≤ crcc_adaptive everywhere | TBD | TBD |
| H3 | no recall regression at ratio≥0.10 | TBD | TBD |
| H4 | ir_threshold std < 0.005 | TBD | TBD |
| H5 | synthetic recall gain at ratio≤0.07 | TBD | TBD |

## Paper Table Template

| Ratio (IR) | class_proportional | crcc_adaptive | majority_only | Cohen's d |
|---|---|---|---|---|
| 2% (49) | recall=X | recall=X | recall=X | d=X |
| 5% (19) | recall=X | recall=X | recall=X | d=X |
| 7% (13) | recall=X | recall=X | recall=X | d=X |
| 10% (9) | recall=X | recall=X | recall=X | d=X |
| 15% (6) | recall=X | recall=X | recall=X | d=X |

## If H1 Fails

If crcc_adaptive ≤ class_proportional at ratio ≤ 0.07:
- Check why: is majority noise deletion hurting? Is suspiciousness signal too weak at extreme IR?
- Inspect which samples crcc_adaptive deletes vs class_proportional
- Consider: noise injection rate may need to be higher (40%/20%) to create enough noisy majority
  samples for the majority-only path to find clean signal

Document as honest negative result. The method is theoretically sound but empirically doesn't
activate under the tested conditions.

## Related Code Files

- Create: `scripts/analyze_imbalance_sweep.py` (≤ 100 lines)
- Update: `docs/experiment-report.md`

## Success Criteria

- [ ] All 5 hypotheses evaluated with numeric results
- [ ] Verdict table filled in (pass/fail with numbers)
- [ ] Paper table produced
- [ ] Honest negative result documented if H1 fails
