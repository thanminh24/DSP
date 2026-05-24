---
phase: 2
title: "Fast New Sweeps"
status: pending
priority: P1
effort: "2.5h"
dependencies: [1]
---

# Phase 2: Fast New Sweeps

## Overview

Four targeted sweeps on the existing 5 datasets addressing critical red-team issues without dataset expansion. All use existing infrastructure. Can run concurrently where CSVs don't conflict.

**Key correction from plan review:** The RF/ET ablation uses existing method keys `cwms` and `msbs` (already implemented at `run_relabeling_viability_sweep.py:329,336`). No new dispatchers needed.

## Related Code Files

- Create: `scripts/run_failure_mode_sweep.py`
- Create: `scripts/run_rfet_ablation_sweep.py`
- Create: `scripts/run_iw_smote_lamda_sweep.py`
- Create: `scripts/run_clean_data_ablation.py`
- Create: `scripts/analyze_rfet_ablation.py`
- Create: `scripts/analyze_failure_mode.py`
- Output: `outputs/failure-mode-sweep.csv`, `outputs/rfet-ablation-sweep.csv`, `outputs/iw-lamda-sweep.csv`, `outputs/clean-data-ablation.csv`

## Sweep A — C4: Failure-Mode Protocols (symmetric + reverse_asymmetric)

Run the method under noise conditions it was NOT designed for. Expected: degradation, which strengthens the operating-condition claim.

```python
# scripts/run_failure_mode_sweep.py
FAILURE_PROTOCOLS = {
    "symmetric":          (0.20, 0.20),   # equal flip both ways
    "reverse_asymmetric": (0.02, 0.25),   # majority→minority dominates
}
FAILURE_METHODS = ["no_cleaning", "class_proportional", "smote", "cwms_msbs"]
FAILURE_MODELS  = ["lr"]
DATASETS        = ["pima", "credit-g", "yeast", "phoneme", "ecoli"]
SEEDS           = QUICK_SEEDS  # 10 seeds
# 2 protocols × 5 datasets × 10 seeds × 4 methods × 1 model = 400 rows
OUTPUT_CSV = PROJECT_ROOT / "outputs" / "failure-mode-sweep.csv"
```

Expected findings:
- Symmetric: NoiSyn ≈ class_proportional (OOF scores carry weak directional signal)
- Reverse_asymmetric: NoiSyn ≤ class_proportional (OOF scores point wrong direction)

Document as "operating conditions" in Section 7 of paper.

## Sweep B — H2: RF/ET Ablation (CWMS-only vs MSBS-only)

Use **existing method keys** `cwms` and `msbs` — no new dispatchers required.

- `cwms` (line 336): confidence-weighted suppression only, no synthesis
- `msbs` (line 329): synthesis only with `bal_scores`, no weight modification  
- `cwms_msbs` (line 352): full method (existing)

```python
# scripts/run_rfet_ablation_sweep.py
ABLATION_METHODS = [
    "no_cleaning",
    "class_proportional",
    "cwms",       # suppression only — existing method
    "msbs",       # synthesis only — existing method
    "cwms_msbs",  # full pipeline — existing method
]
ABLATION_MODELS    = ["random_forest", "extra_trees"]
ABLATION_PROTOCOLS = ["hidden_minority_low", "hidden_minority_medium", "hidden_minority_high"]
DATASETS           = ["pima", "credit-g", "yeast", "phoneme", "ecoli"]
SEEDS              = QUICK_SEEDS
# 2 models × 5 methods × 5 datasets × 10 seeds × 3 protocols = 1,500 rows
OUTPUT_CSV = PROJECT_ROOT / "outputs" / "rfet-ablation-sweep.csv"
```

Analysis script `analyze_rfet_ablation.py` produces:

```
RF/ET Ablation — ΔBA vs class_proportional (n=150 pairs per row)
Model         | cwms_only | msbs_only | cwms_msbs
Random Forest | +/-x.xx   | +/-x.xx   | -4.64pp
Extra Trees   | +/-x.xx   | +/-x.xx   | -3.80pp
```

This replaces the unsupported "bagging robustness" rationalisation with empirical attribution of harm.

## Sweep C — M3: IW-SMOTE Lamda Sensitivity

```python
# scripts/run_iw_smote_lamda_sweep.py
LAMDA_VALUES = [10, 20, 30, 50, 100]   # 30 = current; 100 = original paper default
PROTOCOL     = "hidden_minority_medium"
MODEL        = "lr"
DATASETS     = ["pima", "credit-g", "yeast", "phoneme", "ecoli"]
SEEDS        = QUICK_SEEDS
# 5 lamda × 5 datasets × 10 seeds = 250 rows
OUTPUT_CSV = PROJECT_ROOT / "outputs" / "iw-lamda-sweep.csv"
```

**Decision gate for Phase 5:** If IW-SMOTE BA at lamda=100 exceeds lamda=30 by >0.5pp on average, Phase 5 must use lamda=100. Otherwise keep lamda=30 and add footnote: "Sensitivity analysis confirms lamda=30 equivalent to lamda=100 (mean ΔBA<0.5pp) while reducing compute 3×."

## Sweep D — M2: Clean-Data Ablation

Answers: is NoiSyn's gain noise-specific, or does it help on clean imbalanced data too?

```python
# scripts/run_clean_data_ablation.py
CLEAN_PROTOCOL = {"clean": (0.00, 0.00)}  # mn=0, mj=0 — zero noise
CLEAN_METHODS  = ["no_cleaning", "class_proportional", "smote", "cwms_msbs"]
CLEAN_MODELS   = ["lr", "svm"]
DATASETS       = ["pima", "credit-g", "yeast", "phoneme", "ecoli"]
SEEDS          = QUICK_SEEDS
# 2 models × 4 methods × 5 datasets × 10 seeds = 400 rows
OUTPUT_CSV = PROJECT_ROOT / "outputs" / "clean-data-ablation.csv"
```

Interpretation:
- If ΔBA > 0 on clean data → reframe contribution as "boundary-aware synthesis" (still novel, different angle)
- If ΔBA ≈ 0 on clean data → noise-specific mechanism confirmed; stronger paper story

## Run Order

```bash
# A and B have independent output CSVs — run concurrently
/home/than-minh/miniconda3/envs/dsp/bin/python scripts/run_failure_mode_sweep.py &
/home/than-minh/miniconda3/envs/dsp/bin/python scripts/run_rfet_ablation_sweep.py &
wait
# C determines Phase 5 lamda — run before Phase 5
/home/than-minh/miniconda3/envs/dsp/bin/python scripts/run_iw_smote_lamda_sweep.py
/home/than-minh/miniconda3/envs/dsp/bin/python scripts/run_clean_data_ablation.py
```

## Success Criteria

- [ ] `outputs/failure-mode-sweep.csv` — 400 rows, zero NaN BA
- [ ] `outputs/rfet-ablation-sweep.csv` — 1,500 rows; `cwms` and `msbs` rows populated (no new dispatchers added)
- [ ] `outputs/iw-lamda-sweep.csv` — 250 rows; lamda gate decision documented
- [ ] `outputs/clean-data-ablation.csv` — 400 rows; clean-data mechanism documented
- [ ] `scripts/analyze_rfet_ablation.py` produces decomposition table
- [ ] `scripts/analyze_failure_mode.py` produces protocol comparison table

## Risk Assessment

- **lamda=100 consequence**: If IW-SMOTE at lamda=100 closes the gap, Phase 5 re-run at lamda=100 adds ~2h. Gate decision must happen before Phase 5 starts.
- **RF OOF overhead for `cwms`/`msbs`**: These use `bal_scores` which requires balanced OOF of RF itself. With n_estimators=300, 5-fold RF OOF ≈ 1,500 tree fits per (dataset, seed). For 150 rows, this is ~225,000 tree fits — may take 2–3h for RF rows. Consider reducing RF/ET n_estimators to 100 for this ablation sweep only (add parameter override).
