---
phase: 2
title: "Method A Pilot — Balanced-OOF Type A Relabeling"
status: pending
effort: "2h"
---

# Phase 2: Method A Pilot

## Overview

Test whether **balanced-OOF Type A relabeling** beats `class_proportional` on
4 datasets × 2 models × 20 seeds at medium noise (30%/10%). The `random_relabel`
control distinguishes a real OOF signal from an oversampling artifact.

## Context Links

- Method A implementation: phase-01 → `pipeline/augmentation/relabeling.py`
- Runner: phase-01 → `scripts/run_augment_experiment.py`
- Baseline behavior: `outputs/full-experiment-results.csv` (class_proportional rows)

## Hypothesis

H1: Under asymmetric noise (30% min→maj, 10% maj→min), top-k majority samples by
balanced-OOF `P(minority|x)` contain Type A noise with sufficient precision that
relabeling them yields BA > `class_proportional`.

H0 (control): Balanced-OOF signal is uninformative; any gain from relabeling is
pure oversampling. If `balanced_oof_relabel ≈ random_relabel` in BA, accept H0.

## Methods Under Test

| Method | Role |
|--------|------|
| `no_cleaning` | Lower baseline |
| `class_proportional` | Strong baseline (the bar to beat) |
| `balanced_oof_relabel` | **Method A** |
| `random_relabel` | **Control** — same number of relabels, random indices |

## Requirements

- Use Phase 1 `run_single_augment` runner.
- Fixed 20 seeds: `[13,17,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101]`.
- Datasets: `["pima","credit-g","yeast","phoneme"]`.
- Models: `["lr","hgb"]`.
- Budget: 10% (default `BaseExperimentConfig.cleaning_budget`).
- Noise: 30%/10% (default).
- Output: `outputs/pilot-method-a-results.csv`.

## Related Code Files

**Create:**
- `scripts/run_pilot_method_a.py` (≤ 100 lines)
- `outputs/pilot-method-a-results.csv` (data)
- `outputs/pilot-method-a-decision.md` (GO/PARTIAL/FAIL writeup)

**Modify:** none.

## Implementation Steps

### Step 1 — `scripts/run_pilot_method_a.py`

```python
"""Pilot for Method A — balanced-OOF Type A relabeling."""
from __future__ import annotations
import itertools
import pandas as pd
from pipeline.core.config import BaseExperimentConfig
from scripts.run_augment_experiment import run_single_augment

DATASETS = ["pima", "credit-g", "yeast", "phoneme"]
MODELS = ["lr", "hgb"]
SEEDS = [13,17,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101]
METHODS = ["no_cleaning", "class_proportional", "balanced_oof_relabel", "random_relabel"]
OUT_CSV = "outputs/pilot-method-a-results.csv"


def main():
    cfg = BaseExperimentConfig()  # default 30%/10% noise, 10% budget, 15% min ratio
    rows = []
    total = len(DATASETS) * len(MODELS) * len(SEEDS)
    done = 0
    for ds, mdl, sd in itertools.product(DATASETS, MODELS, SEEDS):
        try:
            rows.extend(run_single_augment(ds, mdl, sd, METHODS, cfg))
        except Exception as e:
            print(f"FAIL {ds}/{mdl}/{sd}: {e}")
        done += 1
        if done % 8 == 0:
            print(f"  progress: {done}/{total}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"wrote {len(df)} rows → {OUT_CSV}")
    _print_summary(df)


def _print_summary(df: pd.DataFrame):
    pivot = df.pivot_table(
        index=["dataset", "model"], columns="method",
        values="balanced_accuracy", aggfunc="mean")
    pivot["delta_vs_cp"] = pivot["balanced_oof_relabel"] - pivot["class_proportional"]
    pivot["delta_vs_random"] = pivot["balanced_oof_relabel"] - pivot["random_relabel"]
    print("\n── Method A BA summary (mean over 20 seeds) ──")
    print(pivot.round(4).to_string())
    wins = (pivot["delta_vs_cp"] > 0).sum()
    print(f"\nMethod A wins vs class_proportional: {wins} / {len(pivot)} combos")
    print(f"Mean Δ vs class_proportional: {pivot['delta_vs_cp'].mean():+.4f}")
    print(f"Mean Δ vs random_relabel:    {pivot['delta_vs_random'].mean():+.4f}")


if __name__ == "__main__":
    main()
```

### Step 2 — run

```bash
cd /home/than-minh/project/DSP
/home/than-minh/miniconda3/bin/python3 scripts/run_pilot_method_a.py 2>&1 | tee outputs/pilot-method-a-log.txt
```

Expected runtime: ~25–40 min (160 combos, ~10–15s each).

### Step 3 — apply hard gate

```python
# in outputs/pilot-method-a-decision.md analysis section
from scipy.stats import wilcoxon
import numpy as np, pandas as pd

df = pd.read_csv("outputs/pilot-method-a-results.csv")
pivot = df.pivot_table(index=["dataset","model","seed"], columns="method",
                       values="balanced_accuracy").reset_index()
diff = pivot["balanced_oof_relabel"] - pivot["class_proportional"]
W, p = wilcoxon(diff.dropna())
combo_wins = pivot.groupby(["dataset","model"]).apply(
    lambda g: g["balanced_oof_relabel"].mean() > g["class_proportional"].mean()).sum()

def cohens_d(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return (a.mean() - b.mean()) / np.sqrt(((a.std(ddof=1)**2 + b.std(ddof=1)**2)/2))
ds_d = [cohens_d(g["balanced_oof_relabel"], g["class_proportional"])
        for _, g in pivot.groupby(["dataset","model"])]
mean_d = float(np.mean(ds_d))
print(f"wins {combo_wins}/8  p={p:.4g}  mean_d={mean_d:.3f}")

# CONTROL
diff_ctrl = pivot["balanced_oof_relabel"] - pivot["random_relabel"]
W2, p2 = wilcoxon(diff_ctrl.dropna())
print(f"vs random_relabel: p={p2:.4g}  delta={diff_ctrl.mean():+.4f}")
```

**Gate logic:**

```
GO       : wins ≥ 6/8 AND mean_d ≥ 0.3 AND p < 0.05
           AND  BA(balanced_oof_relabel) > BA(random_relabel)  AND  p2 < 0.05
PARTIAL  : wins ≥ 5/8 AND mean_d ≥ 0.2 AND control delta > 0
FAIL     : otherwise
```

The **control condition is non-negotiable**: if Method A does not beat
`random_relabel`, declare FAIL even when beating `class_proportional` — gain is oversampling.

### Step 4 — writeup `outputs/pilot-method-a-decision.md`

Must contain:
- Pivot table (4×2×4 mean BA ± std)
- Wilcoxon stats vs class_proportional and vs random_relabel
- Cohen's d table per (dataset, model)
- Final verdict: GO / PARTIAL / FAIL
- `relabel_correctness` summary
  - Target signal: `balanced_oof_relabel` correctness > 0.30 (real Type A detection)
  - Random baseline expected: ~0.03 (Type A is ~3% of majority pool)
- One-paragraph justification of verdict

## Todo List

- [ ] Write `scripts/run_pilot_method_a.py`
- [ ] Run pilot; produce `outputs/pilot-method-a-results.csv`
- [ ] Compute Wilcoxon + Cohen's d + win counts
- [ ] Compute control comparison (vs random_relabel)
- [ ] Write `outputs/pilot-method-a-decision.md` with verdict
- [ ] If FAIL: still proceed to Phase 3 (independent)
- [ ] If GO/PARTIAL: tag method for inclusion in Phase 4 sweep

## Success Criteria

- [ ] CSV has 4 × 2 × 20 × 4 = 640 rows (< 5% loss tolerated)
- [ ] Pivot summary printed to stdout
- [ ] Decision document written with explicit verdict + numbers
- [ ] Control comparison reported

## Test Matrix

| Check | Pass condition |
|-------|----------------|
| Schema | All rows have 14 columns incl. `relabel_correctness` |
| No NaN BA | < 5% rows NaN |
| `n_relabeled == budget_count` | for both relabel methods |
| `relabel_correctness` random | ≈ 0.03 |

## Risk Assessment

| Risk | L | I | Mitigation |
|------|---|---|----|
| Pilot beats class_proportional but fails control | H | H | Control mandatory in gate |
| Some combos fail (one-class after relabel) | L | L | NaN handled by `evaluate_augmented` |
| credit-g HGB cat_indices issue | L | M | Existing `_encode_dataframe` handles it |
| Runtime > 1h | L | L | Progress log every 8 combos |

## Rollback Plan

Pilot is read-only on existing code; discard CSV.

## Next Steps

- Phase 3 (Method B) runs regardless of Phase 2 outcome (independent).
- Phase 4 inclusion: only if Phase 2 GO or PARTIAL.
- Phase 5 always reads any CSVs that exist.
