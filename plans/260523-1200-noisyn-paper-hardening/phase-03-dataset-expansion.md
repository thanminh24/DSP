---
phase: 3
title: "Dataset Expansion"
status: pending
priority: P1
effort: "8h"
dependencies: [1, 2]
---

# Phase 3: Dataset Expansion

## Overview

Expand from 5 to 15 datasets to meet the 15–27 venue standard. Pattern: extend `download_datasets.py` (run once, network required), cache as parquet in `data/`, register in `loaders.py`. No live OpenML calls at experiment time — identical to the existing pattern for yeast/ecoli/phoneme.

**Correct loader interface** (confirmed from `pipeline/data/loaders.py:36`):
```python
def load_dataset(name: str) -> tuple[pd.DataFrame, np.ndarray, list[str], list[str]]:
    """Returns (X_df, y_binary, cat_cols, feature_names)"""
```

**Correct expected row count** for 15-dataset full benchmark:
- 7 CWMS-full models × 7 methods × 3 protocols × 10 seeds × 15 datasets = 22,050
- 2 baseline-only models × 3 methods × 3 protocols × 10 seeds × 15 datasets = 2,700
- **Total: 24,750 rows** (verified: existing 5-dataset run = 8,250 = 1,650 per dataset × 5)

## Dataset Selection Criteria

- Binary classification only
- Tabular (no image/text/time-series)
- 200–5,000 samples (SVM tractability ceiling)
- Imbalance ratio 1.5–15 at original labels
- Available on OpenML with clean `target` attribute
- Used in ≥1 published imbalanced learning paper

## 10 New Datasets

| Dataset | OpenML ID | n (approx) | Original IR | Binarisation |
|---------|-----------|------------|-------------|--------------|
| Breast Cancer Wisconsin | 1510 | 569 | 1.68 | default (malignant=minority) |
| ILPD (Indian Liver) | 1480 | 583 | 2.51 | default (diseased=minority) |
| Blood Transfusion | 1464 | 748 | 3.20 | default (donated=minority) |
| Haberman Survival | 43 | 306 | 2.78 | default (died=minority) |
| Ionosphere | 59 | 351 | 1.79 | default (bad=minority) |
| Vehicle Silhouettes (bus) | 54 | 846 | 3.25 | bus vs rest |
| Glass (window float) | 41 | 214 | 3.20 | window-float vs rest |
| Abalone (rings>10) | 183 | 2000* | 3.06 | rings>10 vs rest; subsample to 2000 |
| Spambase | 44 | 4601 | 1.54 | default (spam=minority) |
| Sonar | 40 | 208 | 1.14 | default (mine=minority) |

*Abalone full n=4177; stratified subsample to 2000 to keep SVM tractable.

## Related Code Files

- Modify: `scripts/download_datasets.py` — add 10 `fetch_*()` functions
- Modify: `pipeline/data/loaders.py` — add entries to `DATASETS` and `MINORITY_LABELS` dicts
- Modify: `scripts/run_full_benchmark_solution.py` — update DATASETS import source
- Modify: `scripts/run_relabeling_viability_sweep.py` — update DATASETS list
- Create: `outputs/full-benchmark-solution-v2.csv`

## Implementation Steps

### Step 1 — Extend download_datasets.py

Follow the existing `fetch_yeast()` / `fetch_ecoli()` pattern exactly. Each function:
1. Calls `openml.datasets.get_dataset(id)`
2. Binarises `y` to minority=1 / majority=0
3. Maps back to string labels matching `MINORITY_LABELS`
4. Saves as `data/{name}.parquet`

```python
def fetch_breast_cancer() -> None:
    """OpenML 1510 — malignant=minority."""
    dataset = openml.datasets.get_dataset(1510)
    X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
    minority = (y == "malignant")
    df = pd.DataFrame(X)
    df["target"] = minority.map({True: "malignant", False: "benign"})
    df.to_parquet(DATA_DIR / "breast_cancer.parquet", index=False)
    print(f"breast_cancer: {df.shape}, minority={minority.mean():.3f}")

# ... same pattern for each dataset
```

For Vehicle (bus vs rest) and Glass (window-float vs rest): filter to binary and label minority accordingly.

### Step 2 — Register in loaders.py

Add to `DATASETS` dict:
```python
"breast_cancer": {"file": "breast_cancer.parquet"},
"ilpd":          {"file": "ilpd.parquet"},
"blood":         {"file": "blood.parquet"},
"haberman":      {"file": "haberman.parquet"},
"ionosphere":    {"file": "ionosphere.parquet"},
"vehicle_bus":   {"file": "vehicle_bus.parquet"},
"glass_float":   {"file": "glass_float.parquet"},
"abalone":       {"file": "abalone.parquet"},
"spambase":      {"file": "spambase.parquet"},
"sonar":         {"file": "sonar.parquet"},
```

Add to `MINORITY_LABELS` dict with the string labels used in parquet `target` column.

### Step 3 — Download all 10 new datasets

```bash
/home/than-minh/miniconda3/envs/dsp/bin/python scripts/download_datasets.py
```

Verify each parquet file exists in `data/` and has correct shape.

### Step 4 — Smoke-test each loader

```bash
/home/than-minh/miniconda3/envs/dsp/bin/python -c "
from pipeline.data.loaders import load_dataset
for name in ['breast_cancer','ilpd','blood','haberman','ionosphere',
             'vehicle_bus','glass_float','abalone','spambase','sonar']:
    X, y, cat, feat = load_dataset(name)
    minority_frac = y.mean()
    print(f'{name}: n={len(X)}, features={X.shape[1]}, minority_frac={minority_frac:.3f}')
    assert minority_frac > 0.05 and minority_frac < 0.5, f'{name} minority frac out of range'
"
```

### Step 5 — Re-run full benchmark (v2, 15 datasets)

Update `DATASETS` in `run_relabeling_viability_sweep.py` to the 15-dataset list. Run to new output file — keep original `full-benchmark-solution.csv` untouched.

```bash
/home/than-minh/miniconda3/envs/dsp/bin/python scripts/run_full_benchmark_solution.py \
  --gpu --output outputs/full-benchmark-solution-v2.csv
```

Expected: **24,750 rows** (1,650 per dataset × 15 datasets). Resume-safe.

### Step 6 — Phoneme sensitivity check

After v2 completes, run Wilcoxon for LR with and without phoneme:

```python
# In analyze_full_benchmark.py or a one-off script
df_nophoneme = df[df["dataset"] != "phoneme"]
result_with = per_dataset_wilcoxon_stouffer(df, "cwms_msbs", "class_proportional")
result_without = per_dataset_wilcoxon_stouffer(df_nophoneme, "cwms_msbs", "class_proportional")
```

Document in `plans/reports/phoneme-sensitivity.md`.

## Success Criteria

- [ ] 10 new `fetch_*()` functions in `download_datasets.py`; all produce valid parquet files
- [ ] `pipeline/data/loaders.py` has all 15 datasets in `DATASETS` and `MINORITY_LABELS`
- [ ] Smoke test passes: all 10 new datasets load cleanly, minority_frac in (0.05, 0.50)
- [ ] `outputs/full-benchmark-solution-v2.csv` has **24,750 rows**, zero NaN BA for CWMS-compatible methods
- [ ] `outputs/full-benchmark-solution.csv` (5-dataset original) unchanged
- [ ] Phoneme sensitivity report at `plans/reports/phoneme-sensitivity.md`

## Risk Assessment

- **SVM on Abalone/Spambase (n≈2000–4601)**: SVM is O(n²). Abalone capped at 2000 mitigates this. Spambase at 4601 may take 10–30min per seed for SVM OOF. If SVM exceeds 4h total, consider excluding spambase from SVM rows only (add to `_methods_for()` override).
- **OpenML download reliability**: Cached at download time only. Network dependency isolated to one manual step before experiments.
- **IR mismatch post-induction**: Sonar (n=208, IR=1.14) will have very few minority samples after target_ratio=0.15 induction. Run the smoke test with `induce_imbalance` included; if minority < 20 samples, replace sonar with another dataset (e.g., Glass2, OpenML 42).
