---
phase: 1
title: "Environment and Dependency Validation"
status: completed
priority: P1
effort: "30m"
dependencies: []
---

# Phase 1: Environment and Dependency Validation

## Overview

Confirm the full Python environment is ready for the experiment pipeline. All packages are already installed in the conda env; all 3 datasets are already cached as local Parquet files in `data/`. This phase writes a validation script, runs it, and pins versions in a requirements file. Also sets up the output directory tree.

## Requirements

- Functional: validation script exits 0 only when all imports succeed and dataset IDs resolve.
- Non-functional: no new package installs should be needed (environment confirmed).

## Architecture

Single validation script that imports every dependency used downstream, fetches dataset metadata (no full download), and prints a version table. Creates `outputs/plots/` directory.

**Confirmed environment:**
- Python: `/home/than-minh/miniconda3/bin/python3`
- scikit-learn 1.8.0, numpy 2.2.6, pandas 2.3.3, openml 0.15.1, seaborn 0.13.2, matplotlib 3.10.8

## Related Code Files

- Create: `scripts/validate-environment.py`
- Create: `requirements.txt`
- Create: `outputs/plots/` (directory)

## Implementation Steps

1. Create `scripts/validate-environment.py`:
   - Import: `sklearn`, `numpy`, `pandas`, `openml`, `seaborn`, `matplotlib`
   - Print version of each
   - Verify local parquet files exist and load cleanly:
     - `pd.read_parquet("data/pima.parquet")` → assert shape (768, 9)
     - `pd.read_parquet("data/credit-g.parquet")` → assert shape (1000, 21)
     - `pd.read_parquet("data/sick.parquet")` → assert shape (3772, 30)
   - Create `outputs/plots/` via `Path.mkdir(parents=True, exist_ok=True)`
   - Print `VALIDATION OK` on success

2. Run with: `/home/than-minh/miniconda3/bin/python3 scripts/validate-environment.py`

3. Create `requirements.txt` pinning confirmed versions:
   ```
   scikit-learn==1.8.0
   numpy==2.2.6
   pandas==2.3.3
   openml==0.15.1
   seaborn==0.13.2
   matplotlib==3.10.8
   ```

## Success Criteria

- [ ] `validate-environment.py` runs without error
- [ ] All 3 local parquet files load with correct shapes
- [ ] `outputs/plots/` directory exists
- [ ] `requirements.txt` written

## Risk Assessment

Very low. All packages confirmed installed. All datasets cached locally — zero network dependency.
