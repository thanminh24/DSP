---
phase: 2
title: "Dataset Correction"
status: pending
priority: P1
effort: "2h"
dependencies: []
---

# Phase 2: Dataset Correction

## Overview

Replace the current 3 datasets (Pima, Credit-G, Sick) with the 5 literature-verified KEEL/UCI benchmarks that appear in comparative imbalanced learning papers. This gives CRCC direct comparability to prior work and justifies dataset choice by citation.

**Current → Target:**
| Current | Status |
|---------|--------|
| Pima | Keep — appears in Chawla 2002, He 2009 |
| Credit-G (German) | Keep — appears in He 2009, Fernández 2018 |
| Sick | Remove — not cited in comparative imbalanced ML literature |
| *(new)* Yeast | Add — KEEL, appears in ADASYN 2008, Krawczyk 2016, Fernández 2018 |
| *(new)* Ecoli | Add — KEEL, appears in ADASYN 2008, Krawczyk 2016, Fernández 2018 |
| *(new)* Phoneme | Add — KEEL, appears in KEEL paper 2011, Fernández 2018 |

Final 5 datasets: Pima, German Credit, Yeast, Ecoli, Phoneme.

## Dataset Specifications

### Yeast (KEEL)
- **Rows:** 1,484 | **Features:** 8 numeric | **Binary:** MIT class (CYT=0, MIT=1)
- **Minority ratio:** ~28% MIT class (binary conversion from multi-class: MIT vs rest)
- **KEEL OpenML ID:** 181 (yeast) or fetch from KEEL directly
- **cat_cols:** []
- **Target column:** `Class` (MIT=1 minority, all others=0)
- Note: KEEL distributes a pre-binarized version (yeast-2_vs_4, yeast-0-5-6-7-9_vs_4 etc.). Use the standard binary `yeast` task from OpenML (ID 181): `Class` has 10 values; binarize MIT vs rest for ~28% minority.

### Ecoli (KEEL)
- **Rows:** 336 | **Features:** 7 numeric | **Binary:** im class (im=1 minority ~10.7%, rest=0)
- **KEEL OpenML ID:** 40536 (ecoli) — use the pre-binarized binary version from OpenML: ecoli-0_vs_1 has balanced ratio; use `ecoli` (ID 40536) with target `class` binarized (im vs rest)
- **Minority ratio:** ~10.7%
- **cat_cols:** []
- Note: Alternatively fetch from KEEL URL directly as CSV and process.

### Phoneme (KEEL / OpenML)
- **Rows:** 5,404 | **Features:** 5 numeric | **Binary:** 0=nasal (minority ~29%), 1=oral
- **OpenML ID:** 1489
- **cat_cols:** []
- **Target column:** `Class` (0=minority/nasal, 1=majority/oral) — minority_label=0

## Related Code Files

- Modify: `scripts/data_loader.py` — replace `_load_sick()` with `_load_yeast()`, `_load_ecoli()`, `_load_phoneme()`; keep `_load_pima()`, `_load_credit_g()`
- Create: `scripts/download_datasets.py` — fetches Yeast, Ecoli, Phoneme via openml and saves as parquet
- Create: `data/yeast.parquet`, `data/ecoli.parquet`, `data/phoneme.parquet`
- Delete: `data/sick.parquet` (if it exists)
- Update: any hardcoded `datasets=("pima","credit-g","sick")` tuples in experiment scripts → `("pima","credit-g","yeast","ecoli","phoneme")`

## Implementation Steps

1. **Create `scripts/download_datasets.py`** (≤ 80 lines):
   ```python
   import openml
   import pandas as pd
   from pathlib import Path

   DATA_DIR = Path(__file__).resolve().parent.parent / "data"
   DATA_DIR.mkdir(exist_ok=True)

   def fetch_yeast():
       # OpenML task 181 — yeast multi-class, binarize MIT vs rest
       dataset = openml.datasets.get_dataset(181)
       X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
       y_binary = (y == "MIT").astype(int)
       df = pd.DataFrame(X)
       df["target"] = y_binary.map({1: "MIT", 0: "other"})  # string targets for generic loader
       df.to_parquet(DATA_DIR / "yeast.parquet", index=False)
       print(f"yeast: {df.shape}, minority={y_binary.mean():.3f}")

   def fetch_ecoli():
       # OpenML dataset 40536
       dataset = openml.datasets.get_dataset(40536)
       X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
       y_binary = (y == "im").astype(int)
       df = pd.DataFrame(X)
       df["target"] = y_binary.map({1: "im", 0: "other"})
       df.to_parquet(DATA_DIR / "ecoli.parquet", index=False)
       print(f"ecoli: {df.shape}, minority={y_binary.mean():.3f}")

   def fetch_phoneme():
       # OpenML dataset 1489 — 0=nasal=minority
       dataset = openml.datasets.get_dataset(1489)
       X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
       y_binary = (y.astype(int) == 0).astype(int)  # 1 = nasal (minority)
       df = pd.DataFrame(X)
       df["target"] = y_binary.map({1: "nasal", 0: "other"})
       df.to_parquet(DATA_DIR / "phoneme.parquet", index=False)
       print(f"phoneme: {df.shape}, minority={y_binary.mean():.3f}")

   if __name__ == "__main__":
       fetch_yeast(); fetch_ecoli(); fetch_phoneme()
   ```

   Run: `python3 scripts/download_datasets.py`

2. **Extend generic dispatch in `scripts/data_loader.py`** — keep the existing dict-based pattern (generic `load_dataset()` using `DATASETS` + `MINORITY_LABELS` dicts). Add entries for the 3 new datasets:
   ```python
   DATASETS: dict[str, dict] = {
       "pima":     {"file": "pima.parquet"},
       "credit-g": {"file": "credit-g.parquet"},
       "yeast":    {"file": "yeast.parquet"},
       "ecoli":    {"file": "ecoli.parquet"},
       "phoneme":  {"file": "phoneme.parquet"},
   }

   MINORITY_LABELS: dict[str, str] = {
       "pima": "tested_positive",
       "credit-g": "bad",
       "yeast": "MIT",
       "ecoli": "im",
       "phoneme": "nasal",
   }
   ```
   The existing `load_dataset()` function handles all datasets generically — reads parquet, binarizes target via `MINORITY_LABELS[name]`. No per-dataset loader functions needed.
   
   Note: `load_dataset()` returns `(X, y, cat_cols, feature_names)` — the download script must ensure parquet files have a `target` column and the target strings match `MINORITY_LABELS`.

3. **Update download script to use `target` column** — the generic `load_dataset()` reads `df["target"]`. Ensure `download_datasets.py` names the target column `target`, not `Class`:
   ```python
   # In fetch_yeast():
   df["target"] = y_binary.astype(str)  # "MIT" (1) vs "0" (others)
   
   # In fetch_ecoli():
   df["target"] = y_binary.astype(str)  # "im" (1) vs "0" (others)
   
   # In fetch_phoneme():
   df["target"] = y_binary.astype(str)  # "nasal" (1) vs "0" (others)
   # Note: phoneme minority=0 (nasal), so y_binary=1 means nasal
   ```
   Also set `MINORITY_LABELS["phoneme"] = "nasal"` and note that Phoneme has `minority_label=0` in the experiment (handled by Phase 3's `minority_label_map`).

4. **Remove `"sick"` entry** from `DATASETS` and `MINORITY_LABELS` dicts. Delete `data/sick.parquet` if present.

5. **Update default `datasets` tuples** in `run_full_experiment.py` and `run_mild_imbalance_experiment.py`:
   ```python
   datasets: tuple = ("pima", "credit-g", "yeast", "ecoli", "phoneme")
   ```

6. **Smoke-test all 5 loaders**:
   ```bash
   python3 -c "
   import sys; sys.path.insert(0, 'scripts')
   from data_loader import load_dataset
   for name in ['pima','credit-g','yeast','ecoli','phoneme']:
       X, y, cats, feats = load_dataset(name)
       print(f'{name}: X={X.shape}, minority={y.mean():.3f}')
   "
   ```
   Expected minority ratios: pima~0.35, credit-g~0.30, yeast~0.28, ecoli~0.11, phoneme~0.29

7. **Verify `data_loader.py` ≤ 200 lines** after edits.

8. **Smoke experiment on new datasets** (seed=13 only):
   ```bash
   python3 -c "
   import sys; sys.path.insert(0, 'scripts')
   from run_full_experiment import run_single, Config
   cfg = Config(seeds=(13,), datasets=('yeast','ecoli','phoneme'))
   for ds in cfg.datasets:
       for model in ('lr','hgb'):
           rows = run_single(ds, model, 13, cfg)
           cmdr = next(r['clean_minority_deletion_rate'] for r in rows if r['method']=='crcc_p_l05')
           print(ds, model, 'CMDR:', round(cmdr, 4))
   "
   ```

## Success Criteria

- [ ] `data/yeast.parquet`, `data/ecoli.parquet`, `data/phoneme.parquet` exist
- [ ] `load_dataset("yeast")` returns shape ~(1484, 8), minority ~0.28
- [ ] `load_dataset("ecoli")` returns shape ~(336, 7), minority ~0.11
- [ ] `load_dataset("phoneme")` returns shape ~(5404, 5), minority ~0.29
- [ ] `_load_sick()` / `"sick"` entries removed from `DATASETS` and `MINORITY_LABELS`
- [ ] Smoke experiment on new 3 datasets completes without crash
- [ ] `data_loader.py` ≤ 200 lines

## Risk Assessment

- **Ecoli at 10.7% minority + 10% budget**: budget_count ≈ 3 samples. Selector may select 0 or crash. Phase 3 one-class guard must handle this. Document as edge case in experiment report.
- **OpenML API during download**: run once, commit parquets; never call openml in experiments.
- **Yeast binarization**: MIT class is ~28% of 1484 rows. This is balanced enough — `induce_imbalance(target=0.15)` will subsample majority from 72% down to 85%, which is feasible.
- **Phoneme minority_label=0** (nasal class): ensure all selectors receive `minority_label=0` not the default=1. Check `select_crcc_p()` and `select_crcc_m()` parameter passing.
