# Reproducibility Guide

## Environment

Use a local virtual environment. The default system Python may not include project
dependencies.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On Python 3.14, `cleanlab` and `catboost` are skipped by environment markers because their
current dependency wheels are not compatible in this environment. XGBoost and LightGBM are
installed and should be used as the publication-grade boosting baselines.

## Required Data

Cached datasets live under `data/`:

- `pima.parquet`
- `credit-g.parquet`
- `yeast.parquet`
- `ecoli.parquet`
- `phoneme.parquet`

Weak-supervision datasets are optional and should be cached under
`data/weak-supervision/` with columns `label` and `gold`.

## Audit

```powershell
.\.venv\Scripts\python.exe scripts\audit_relabeling_protocol.py
```

Expected output:

- `outputs/relabeling-protocol-audit.md`

## Experiments

```powershell
.\.venv\Scripts\python.exe scripts\run_relabeling_viability_sweep.py lr hgb
.\.venv\Scripts\python.exe scripts\run_model_stress_benchmark.py lr hgb xgboost
```

## Analysis

```powershell
.\.venv\Scripts\python.exe scripts\analyze_relabeling_statistics.py
```

## Unresolved Questions

- Whether all optional boosting libraries can be installed and run within local time limits.
