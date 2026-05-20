---
phase: 4
title: "Full Sweep — Surviving Methods Across 3 Noise Levels × 5 Datasets"
status: pending
effort: "3h"
---

# Phase 4: Full Sweep

## Overview

Run surviving method(s) from Phase 2/3 pilots on the **full grid**: 5 datasets
× 2 models × 20 seeds × 3 noise levels. Includes `ecoli` (added back) and noise
extremes to test method robustness.

**BLOCKED BY**: At least one of Phase 2 or Phase 3 must return GO or PARTIAL.
If both FAIL, Phase 4 is **cancelled** — do not run.

## Context Links

- Pilot verdicts: `outputs/pilot-method-a-decision.md`, `outputs/pilot-method-b-decision.md`
- Pilot results: `outputs/pilot-method-{a,b}-results.csv`

## Requirements

- Read pilot verdicts; method set = those tagged GO or PARTIAL.
- Always include `no_cleaning` and `class_proportional` as baselines.
- Always include direct controls for any surviving augmentation:
  - if `balanced_oof_relabel` survives → also run `random_relabel`
  - if `oof_filtered_smote` survives → also run `plain_smote`
- 5 datasets: pima, credit-g, yeast, phoneme, **ecoli**.
- 2 models: lr, hgb.
- 20 seeds (same primes).
- 3 noise levels:
  - low: min→maj=0.10, maj→min=0.05
  - medium: min→maj=0.30, maj→min=0.10
  - high: min→maj=0.40, maj→min=0.20
- Budget 10%; minority ratio 15%.
- Output: `outputs/augment-sweep-results.csv`.

## Related Code Files

**Create:**
- `scripts/run_augment_sweep.py` (≤ 130 lines)
- `outputs/augment-sweep-results.csv`

**Modify:** `pipeline/core/config.py` is read but NOT modified — noise is
overridden per-call via a small config-copy helper inside the sweep script.

## Implementation Steps

### Step 1 — config override helper

`BaseExperimentConfig` has fields `minority_to_majority_noise`,
`majority_to_minority_noise` (used at `pipeline/data/loaders.py` via
`inject_noise` in `run_single`). The sweep script clones the base config per
noise level using `dataclasses.replace`.

### Step 2 — `scripts/run_augment_sweep.py`

```python
"""Full sweep over surviving augmentation methods × 3 noise levels."""
from __future__ import annotations
import itertools, dataclasses, sys
import pandas as pd
from pipeline.core.config import BaseExperimentConfig
from scripts.run_augment_experiment import run_single_augment

DATASETS = ["pima", "credit-g", "yeast", "phoneme", "ecoli"]
MODELS = ["lr", "hgb"]
SEEDS = [13,17,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101]
NOISE_LEVELS = {
    "low":    (0.10, 0.05),
    "medium": (0.30, 0.10),
    "high":   (0.40, 0.20),
}
OUT_CSV = "outputs/augment-sweep-results.csv"

# Methods derived from pilot verdicts; passed via CLI or hardcoded after pilots.
# Defaults shown for the "both survive" case.
BASE_METHODS = ["no_cleaning", "class_proportional"]
SURVIVING = ["balanced_oof_relabel", "random_relabel",
             "oof_filtered_smote", "plain_smote"]


def parse_methods(argv):
    if len(argv) > 1:
        return BASE_METHODS + argv[1:]
    return BASE_METHODS + SURVIVING


def main():
    methods = parse_methods(sys.argv)
    print(f"running methods: {methods}")
    rows = []
    combos = list(itertools.product(DATASETS, MODELS, SEEDS, NOISE_LEVELS.items()))
    total = len(combos)
    for i, (ds, mdl, sd, (lvl, (mn, mj))) in enumerate(combos, 1):
        cfg = dataclasses.replace(
            BaseExperimentConfig(),
            minority_to_majority_noise=mn,
            majority_to_minority_noise=mj,
        )
        try:
            batch = run_single_augment(ds, mdl, sd, methods, cfg)
            for r in batch:
                r["noise_level"] = lvl
                r["mn_to_maj"] = mn
                r["maj_to_min"] = mj
            rows.extend(batch)
        except Exception as e:
            print(f"FAIL {ds}/{mdl}/{sd}/{lvl}: {e}")
        if i % 20 == 0:
            print(f"  progress: {i}/{total}")
            # periodic checkpoint
            pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"wrote {len(rows)} rows → {OUT_CSV}")


if __name__ == "__main__":
    main()
```

### Step 3 — run

If Method A survives only:
```bash
python scripts/run_augment_sweep.py balanced_oof_relabel random_relabel
```
If Method B survives only:
```bash
python scripts/run_augment_sweep.py oof_filtered_smote plain_smote
```
If both survive:
```bash
python scripts/run_augment_sweep.py
```

Expected runtime: ~1.5–2.5 h (5 × 2 × 20 × 3 = 600 combos × ~10s each).
Checkpoint every 20 combos.

### Step 4 — sanity validation

After sweep completes:
- Verify row count: `len(df) == n_combos × len(methods)` (modulo failures).
- Confirm `noise_level ∈ {low, medium, high}` distribution is uniform.
- Print mean BA per (method, noise_level) — proceed to Phase 5 for stats.

## Todo List

- [ ] Read pilot verdicts; determine surviving methods
- [ ] Write `scripts/run_augment_sweep.py`
- [ ] Run sweep with appropriate method list
- [ ] Verify CSV row count and schema
- [ ] Print noise-level pivot for visual sanity check
- [ ] Hand off to Phase 5

## Success Criteria

- [ ] Sweep runs to completion (or completes with < 5% failures)
- [ ] `outputs/augment-sweep-results.csv` contains all combos × methods
- [ ] Has columns: dataset, model, seed, method, noise_level, mn_to_maj,
      maj_to_min, balanced_accuracy, macro_f1, minority_recall, deleted,
      noise_precision_deleted, clean_minority_deletion_rate, n_relabeled,
      n_synthetic, relabel_correctness
- [ ] No method present that did NOT pass pilot gate (except baselines/controls)

## Test Matrix

| Check | Pass |
|-------|------|
| Row count | combos × methods, ±5% |
| `noise_level` uniqueness | exactly 3 unique values |
| `mn_to_maj, maj_to_min` mapping consistent | yes per level |
| Surviving methods only | no orphan method values |

## Risk Assessment

| Risk | L | I | Mitigation |
|------|---|---|----|
| Sweep crashes midway | M | M | Checkpoint every 20 combos to CSV |
| High-noise level breaks one-class guard | M | L | Existing NaN handling propagates; counted as failure but doesn't abort |
| ecoli minority pool too small at high noise | H | M | `evaluate_augmented` returns NaN; documented in Phase 5 |
| Pilot tagged GO but full sweep collapses at high noise | M | H | Phase 5 will report per-noise-level breakdown; verdict can downgrade to "PARTIAL — medium noise only" |

## File Ownership

Only `scripts/run_augment_sweep.py` and `outputs/augment-sweep-results.csv` are
touched. No conflicts with Phases 1–3 outputs.

## Rollback Plan

Discard CSV. No code outside the sweep script affected.

## Next Steps

Phase 5 (statistical analysis) reads the sweep CSV and renders final verdict.
