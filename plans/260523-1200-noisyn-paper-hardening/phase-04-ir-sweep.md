---
phase: 4
title: "IR Sweep"
status: pending
priority: P2
effort: "4h"
dependencies: [3]
---

# Phase 4: IR Sweep

## Overview

Run the full benchmark at IR=0.30 (target_ratio=0.30) to show the method works under less severe imbalance. IR=0.15 is the current single point; reviewers need to see performance is not specific to one imbalance level.

## Why IR=0.30 Specifically

At IR=0.30 (30% minority, 70% majority):
- The label noise problem is less extreme (larger minority pool → fewer corrupted samples in absolute terms)
- The SPW (scale_pos_weight) drops from ~5.7 to ~2.3 — a meaningful change to the CWMS boosting path
- MSBS synthesis budget of 10% = more synthetic points relative to minority pool size
- Expected: smaller gains (less noise damage to correct), but method should remain positive

If IR=0.30 shows significant gains, it argues for broad applicability. If it shows near-zero gains, it correctly scopes the method to high-imbalance settings — both are defensible outcomes.

## Related Code Files

- Modify: `scripts/run_full_benchmark_solution.py` — add `--ratio 0.30` flag
- Create: `outputs/full-benchmark-ir030-solution.csv`
- Modify: `scripts/analyze_full_benchmark.py` — add IR breakdown table

## Implementation Steps

### Step 1 — Add ratio flag to run_full_benchmark_solution.py

```python
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--ratio", type=float, default=0.15,
                        help="Minority target ratio (default: 0.15)")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    
    ratio = args.ratio
    output_csv = Path(args.output) if args.output else (
        PROJECT_ROOT / "outputs" / f"full-benchmark-ir{int(ratio*100):03d}-solution.csv"
    )
    ...
```

### Step 2 — Run IR=0.30 on all 15 datasets

```bash
/home/than-minh/miniconda3/envs/dsp/bin/python scripts/run_full_benchmark_solution.py \
  --gpu --ratio 0.30 --output outputs/full-benchmark-ir030-solution.csv
```

**Correct expected row count:**
- 7 CWMS-full models × 7 methods × 3 protocols × 10 seeds × 15 datasets = 22,050
- 2 baseline-only models × 3 methods × 3 protocols × 10 seeds × 15 datasets = 2,700
- **Total: 24,750 rows** — identical structure to Phase 3 v2 run, just at ratio=0.30

Runtime: 4–6h (SVM faster at higher IR due to smaller class imbalance, faster convergence).

**Pilot option (recommended before full run):** Run on original 5 datasets first to check direction:

```bash
/home/than-minh/miniconda3/envs/dsp/bin/python scripts/run_full_benchmark_solution.py \
  --gpu --ratio 0.30 --datasets pima credit-g yeast phoneme ecoli \
  --output outputs/full-benchmark-ir030-pilot.csv
```

Pilot row count: 8,250 (1,650 per dataset × 5 datasets). Takes ~1h. If direction is unexpected (ΔBA < -2pp or no signal), investigate before committing the 4–6h full run.

### Step 3 — Add IR breakdown to analysis

In `scripts/analyze_full_benchmark.py`, add a comparison table:

```
Table S_IR — NoiSyn ΔBA vs class_prop at different imbalance ratios (LR, mean over datasets×seeds×protocols)
IR=0.15 (current):  ΔBA = +3.47pp  p=6.1e-15
IR=0.30 (new):      ΔBA = ???pp    p=???
```

Also run per-model analysis at IR=0.30 and compare to IR=0.15 results.

### Step 4 — Interpret and document

Expected outcomes:
- If ΔBA(IR=0.30) > 0 and significant: "method generalises across imbalance levels"
- If ΔBA(IR=0.30) ≈ 0: "method most effective at high imbalance; at IR=0.30 the noise damage is reduced and the boundary disruption less severe"
- If ΔBA(IR=0.30) < 0: "method requires IR < 0.25 — add as explicit scope condition"

## Success Criteria

- [ ] `outputs/full-benchmark-ir030-solution.csv` has **24,750 rows** (or 8,250 for pilot only)
- [ ] Zero NaN BA for CWMS-compatible methods
- [ ] IR breakdown table added to `analyze_full_benchmark.py` output
- [ ] IR sensitivity result documented in `plans/reports/ir-sensitivity.md`
- [ ] Paper Limitations section updated with IR scope

## Risk Assessment

- **Decision dependency**: Phase 5 (external comparison) should also run at IR=0.30 to match. If resources are tight, run Phase 5 at both ratios simultaneously.
- **SVM at IR=0.30**: Less severe imbalance → SVM may be faster than at IR=0.15.
- **Pilot option**: If 15 datasets × IR=0.30 is too slow, run on the original 5 first to validate the direction, then expand.
