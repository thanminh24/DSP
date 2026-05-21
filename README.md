# DSP

Confidence-guided relabeling experiments for imbalanced noisy tabular classification.

## Overview

This project tests whether class-balanced out-of-fold confidence can recover hidden
minority examples that were mislabeled as majority. Earlier deletion-only CRCC experiments
are archived; the active research direction is relabeling, not deletion.

## Active Plan

See:

`plans/260521-0000-confidence-guided-relabeling-viability-proof/plan.md`

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The local machine currently uses Python 3.14. `cleanlab` and `catboost` are guarded with
Python-version markers in `requirements.txt`; XGBoost and LightGBM are the practical
boosting baselines on this interpreter.

## Core Commands

Run the static leakage/protocol audit:

```powershell
.\.venv\Scripts\python.exe scripts\audit_relabeling_protocol.py
```

Run controlled viability sweep:

```powershell
.\.venv\Scripts\python.exe scripts\run_relabeling_viability_sweep.py lr hgb
```

Run publication-grade model stress benchmark:

```powershell
.\.venv\Scripts\python.exe scripts\run_model_stress_benchmark.py lr hgb xgboost
```

Analyze result CSVs:

```powershell
.\.venv\Scripts\python.exe scripts\analyze_relabeling_statistics.py
```

## Claim Boundary

Do not claim state of the art. The defensible claim is narrower:

> For hidden-minority label noise in imbalanced tabular data, class-balanced out-of-fold
> relabeling can preserve minority feature evidence better than deletion-based cleaning.

## Unresolved Questions

- Weak-supervision dataset loading still needs smoke testing.
- XGBoost/LightGBM/CatBoost runtime must be verified locally.
