---
phase: 3
title: Structured 4-Category Benchmark Sweep
status: completed
priority: P1
effort: 4h
dependencies:
  - 2
---

# Phase 3: Structured 4-Category Benchmark Sweep

## Overview

Run four separately scoped sweeps (separate CSVs, resume-safe) covering all models × all methods. GPU forced for boosting models. All 3 noise protocols × 5 datasets × 10 seeds.

## Run Structure

| Run | Script | Output CSV | Scope | GPU |
|-----|--------|-----------|-------|-----|
| A | `run_full_benchmark_solution.py` | `outputs/full-benchmark-solution.csv` | All models × all our methods + SMOTE | ✅ forced |
| B | `run_scorer_agnosticism_sweep.py` | `outputs/scorer-agnosticism-sweep.csv` | CWMS-compatible models × scorer variants | ✅ forced |
| C | (Phase 4) | `outputs/competitor-headtohead.csv` | Competitor methods vs ours (LR only) | ✅ forced |

## Run A — `scripts/run_full_benchmark_solution.py`

**Scope:**
- Models: `lr, svm, hgb, lightgbm, catboost, random_forest, extra_trees` (CWMS full); `xgboost, calibrated_lr` (baselines only)
- Methods per model:

```python
CWMS_FULL_METHODS = [
    # Baselines (noise-unaware)
    "no_cleaning", "class_proportional", "smote",
    # Our method family
    "msbs", "cwms", "cwms_msbs", "cwms_msbs_shuffled",
]
BASELINE_ONLY_METHODS = ["no_cleaning", "class_proportional", "smote"]

def _methods_for(model_name):
    if model_name in ("xgboost", "calibrated_lr"):
        return BASELINE_ONLY_METHODS
    return CWMS_FULL_METHODS
```

- Protocols: `hidden_minority_low`, `hidden_minority_medium`, `hidden_minority_high`
- Seeds: 10 (QUICK_SEEDS)
- GPU: `--gpu` enforced at startup via `_check_gpu()` (Phase 2 Part A)
- Budget: 0.10, ratio: 0.15
- **Resume-safe**: skip completed rows
- **Total rows**: ~8,500 rows (9 models × 7 methods avg × 5 datasets × 10 seeds × 3 protocols)

**Script skeleton:**
```python
"""Full benchmark: all models × our solution methods × 3 protocols × CUDA.

Run with: python scripts/run_full_benchmark_solution.py --gpu
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_relabeling_viability_sweep import (
    DATASETS, NOISE_PROTOCOLS, QUICK_SEEDS, run_single_viability,
)
from pipeline.models.factories import list_publication_models

CWMS_FULL_METHODS = [
    "no_cleaning", "class_proportional",
    "msbs", "cwms", "cwms_msbs", "cwms_msbs_shuffled",
]
BASELINE_ONLY_METHODS = ["no_cleaning", "class_proportional"]
FULL_PROTOCOLS = ["hidden_minority_low", "hidden_minority_medium", "hidden_minority_high"]
POC_BUDGET, POC_RATIO = 0.10, 0.15
OUTPUT_CSV = PROJECT_ROOT / "outputs" / "full-benchmark-solution.csv"


def _methods_for(model_name: str) -> list[str]:
    if model_name in ("xgboost", "calibrated_lr"):
        return BASELINE_ONLY_METHODS
    return CWMS_FULL_METHODS


def _check_gpu():
    # ... (same function as defined in Phase 2 Part A)
    pass


def _load_completed(path):
    # ... (same resume-safe loader pattern)
    pass


def main():
    args = sys.argv[1:]
    use_gpu = "--gpu" in args
    if use_gpu:
        _check_gpu()

    models = list_publication_models()
    completed = _load_completed(OUTPUT_CSV)
    total_written = 0

    for proto_name in FULL_PROTOCOLS:
        mn, mj = NOISE_PROTOCOLS[proto_name]
        for model_name in models:
            methods_for_model = _methods_for(model_name)
            for dataset in DATASETS:
                for seed in QUICK_SEEDS:
                    needed = [
                        m for m in methods_for_model
                        if (dataset, model_name, seed, proto_name,
                            float(POC_BUDGET), float(POC_RATIO), m) not in completed
                    ]
                    if not needed:
                        continue
                    try:
                        batch = run_single_viability(
                            dataset, model_name, seed, proto_name, mn, mj,
                            POC_BUDGET, POC_RATIO, use_gpu=use_gpu, methods=needed,
                        )
                    except Exception as exc:
                        batch = [{"dataset": dataset, "model": model_name, "seed": seed,
                                   "noise_protocol": proto_name, "method": m, "error": str(exc),
                                   "mn_to_maj": mn, "maj_to_min": mj,
                                   "budget": POC_BUDGET, "target_ratio": POC_RATIO}
                                  for m in needed]
                        print(f"FAIL {dataset}/{model_name}/{seed}/{proto_name}: {exc}", flush=True)
                    df_batch = pd.DataFrame(batch)
                    write_header = not OUTPUT_CSV.exists() or OUTPUT_CSV.stat().st_size == 0
                    df_batch.to_csv(OUTPUT_CSV, mode="a", header=write_header, index=False)
                    total_written += len(batch)
                    print(f"  {proto_name}/{dataset}/{model_name}/{seed}: {needed}", flush=True)

    print(f"\nDone. {total_written} rows -> {OUTPUT_CSV}", flush=True)
```

## Run B — `scripts/run_scorer_agnosticism_sweep.py`

**Scope:** LR, SVM, HGB, LightGBM, CatBoost (CWMS-compatible) × scorer variants × medium protocol only
**Methods:** `no_cleaning, class_proportional, cwms_msbs, cwms_msbs_knn, cwms_msbs_crossfamily`

**Note on `cwms_msbs_crossfamily` routing:**
- `cwms_msbs_crossfamily` skips HGB final model (same-family, not cross-family)
- Skip calibrated_lr, xgboost (not CWMS-compatible)
- For remaining (lr, svm, lightgbm, catboost): HGB OOF scores → final model is whatever `model_name` is

**Total rows:** 5 models × 5 methods × 5 datasets × 10 seeds = 250 rows

```python
SCORER_METHODS = ["no_cleaning", "class_proportional",
                   "cwms_msbs", "cwms_msbs_knn", "cwms_msbs_crossfamily"]
SCORER_MODELS = ["lr", "svm", "hgb", "lightgbm", "catboost"]
TARGET_PROTOCOL = "hidden_minority_medium"
OUTPUT_CSV = PROJECT_ROOT / "outputs" / "scorer-agnosticism-sweep.csv"
```

## Analysis Scripts

### `scripts/analyze_full_benchmark.py`

Produces the following tables for Run A:

**Paper Table 1 — Per-model benchmark (mean BA, 150 pairs per row)**
```
Model     | no_clean | class_prop | smote | cwms_msbs | Δ(vs cp) | p-value | wins/150
LR        | ...      | 0.7032     | ...   | 0.7454    | +4.22pp  | 6.2e-7  | 143/150
SVM       | ...      | ...        | ...   | ...       | ...      | ...     | ...
HGB       | ...
LightGBM  | ...
CatBoost  | ...
RF        | ...
ET        | ...
```
- G-mean column added: `df["g_mean"] = np.sqrt(df["minority_recall"] * df["majority_recall"])`
- XGBoost / calibrated_lr in footnote only

**Paper Table 1b — Shuffled ablation sub-table**
For each CWMS-compatible model: `cwms_msbs vs cwms_msbs_shuffled` ΔBA + p-value
Confirms OOF scores are load-bearing (not random weighting)

**Supplementary Table S1 — Best method per model**
Which of msbs / cwms / cwms_msbs / cwms_msbs_knn / cwms_msbs_crossfamily wins per model

**Supplementary Table S2 — Protocol breakdown: low/medium/high**
Does cwms_msbs advantage hold at all noise levels? (links to paper Section 7.4 limitations)

### `scripts/analyze_scorer_agnosticism.py`

From Run B:

**Table 1 — Mean BA per scorer** (already planned in old phase-03)
**Table 2 — Gains vs class_proportional with p-values** (n=50 pairs per scorer)
**Table 3 — Head-to-head: cwms_msbs vs knn vs crossfamily**
**Table 4 — Per-dataset breakdown**

## Run Command

```bash
# Run A: full benchmark (CUDA forced)
/home/than-minh/miniconda3/envs/dsp/bin/python scripts/run_full_benchmark_solution.py --gpu

# Run B: scorer comparison (CUDA forced)  
/home/than-minh/miniconda3/envs/dsp/bin/python scripts/run_scorer_agnosticism_sweep.py --gpu

# Analysis
/home/than-minh/miniconda3/envs/dsp/bin/python scripts/analyze_full_benchmark.py
/home/than-minh/miniconda3/envs/dsp/bin/python scripts/analyze_scorer_agnosticism.py
```

## Success Criteria

- [x] `scripts/run_full_benchmark_solution.py` created
- [x] `scripts/run_scorer_agnosticism_sweep.py` created (updated from old Phase 3)
- [x] `scripts/analyze_full_benchmark.py` created with Tables 1–5
- [x] `scripts/analyze_scorer_agnosticism.py` created with Tables 1–4
- [ ] Run A completes: `outputs/full-benchmark-solution.csv` has ~7,000+ rows, zero NaN BA for CWMS-compatible methods
- [ ] Run B completes: `outputs/scorer-agnosticism-sweep.csv` has 250 rows
- [ ] Statistical conclusions written up (one scenario from A/B/C in original phase-03)

## Risk Assessment

- **SVM runtime**: SVC is O(n²) in training. With n≈1000–3000 training samples per fold, this is 5–30s per seed. 10 seeds × 5 datasets × 3 protocols × 5 methods = 750 SVM fits. Estimate 1–2h for SVM alone.
- **RF/ET runtime**: With n_estimators=300, these are moderate. Expect similar wall time to LightGBM (GPU saves LGB/XGB/CB time, compensated by RF/ET CPU time).
- **GPU memory**: CatBoost+GPU on tabular data uses <1GB VRAM. All three boosting models can run concurrently if parallelized — but the current sweep is sequential per model. No memory risk.
- **crossfamily HGB OOF per seed**: Adds one HGB 5-fold OOF fit per (dataset, seed) when `cwms_msbs_crossfamily` is in methods. Lazy: only computed when needed.
