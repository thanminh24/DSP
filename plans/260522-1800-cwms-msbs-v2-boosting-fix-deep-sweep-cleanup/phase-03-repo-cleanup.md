---
phase: 3
title: "Repo Cleanup"
status: pending
priority: P2
effort: "~30m"
dependencies: []
---

# Phase 3: Repo Cleanup

## Overview

Delete dead scripts, superseded analysis files, and stale pipeline modules that are no
longer part of the active method. Keep only what is needed to run or understand CWMS+MSBS.
This is a permanent deletion — files will be removed from the repo.

## What to Keep (Active)

### Scripts (keep)
| File | Reason |
|------|--------|
| `scripts/run_relabeling_viability_sweep.py` | Core infrastructure (`run_single_viability`, dispatchers) — all sweeps import from here |
| `scripts/run_cwms_msbs_deep_sweep.py` | New main sweep runner (created in Phase 2) |
| `scripts/analyze_cwms_msbs_deep_results.py` | New main analysis (created in Phase 2) |
| `scripts/analyze_cwms_msbs_results.py` | Previous analysis — keep as reference for old sweep |
| `scripts/download_datasets.py` | Required to re-download OpenML data |
| `scripts/validate_environment.py` | Environment smoke test |
| `scripts/run_cwms_msbs_full_sweep.py` | Superseded by deep sweep but keep until deep sweep is validated |

### Pipeline (keep all of `pipeline/`)
All modules under `pipeline/` are live infrastructure — they are imported by the sweep.
No pipeline module is deleted. Dead code within modules (e.g., any lingering CRCC logic)
is cleaned up in Phase 5 docs update if found.

## What to Delete

### Scripts (dead — delete)

All of these were for old methods, old framings (CRCC deletion, relabeling viability,
CGMS, augmentation pilot), or superseded one-off analyses.

```
scripts/analyze_augment_stats.py
scripts/analyze_cgms_poc.py
scripts/analyze_msbs_cwms_poc.py
scripts/analyze_relabeling_statistics.py
scripts/analyze_relabeling_viability.py
scripts/audit_relabeling_protocol.py
scripts/combine_relabeling_results.py
scripts/generate_figures.py
scripts/plot_results.py
scripts/run_augment_experiment.py
scripts/run_augment_sweep.py
scripts/run_budget_ablation.py
scripts/run_cgms_experiment.py
scripts/run_crcc_smoke_test.py
scripts/run_cwms_fixed_experiment.py
scripts/run_cwms_msbs_combined_prelim.py
scripts/run_diagnostic_tbdc.py
scripts/run_full_experiment.py
scripts/run_mild_imbalance_experiment.py
scripts/run_model_stress_benchmark.py
scripts/run_msbs_cwms_experiment.py
scripts/run_new_baselines_sweep.py
scripts/run_noise_ablation.py
scripts/run_oracle_paradox_analysis.py
scripts/run_pilot.py
scripts/run_pilot_method_a.py
scripts/run_pilot_method_b.py
scripts/run_statistical_tests.py
scripts/run_weak_supervision_relabeling.py
scripts/download_weak_supervision_datasets.py
```

### Other directories (delete)
```
catboost_info/       # training logs from a previous catboost run, not part of any result
data/weak-supervision/  # weak supervision datasets, not relevant to CWMS+MSBS paper
```

## Implementation Steps

### Step 1 — Delete dead scripts

```bash
# Run from project root
cd /home/than-minh/project/DSP

git rm \
  scripts/analyze_augment_stats.py \
  scripts/analyze_cgms_poc.py \
  scripts/analyze_msbs_cwms_poc.py \
  scripts/analyze_relabeling_statistics.py \
  scripts/analyze_relabeling_viability.py \
  scripts/audit_relabeling_protocol.py \
  scripts/combine_relabeling_results.py \
  scripts/generate_figures.py \
  scripts/plot_results.py \
  scripts/run_augment_experiment.py \
  scripts/run_augment_sweep.py \
  scripts/run_budget_ablation.py \
  scripts/run_cgms_experiment.py \
  scripts/run_crcc_smoke_test.py \
  scripts/run_cwms_fixed_experiment.py \
  scripts/run_cwms_msbs_combined_prelim.py \
  scripts/run_diagnostic_tbdc.py \
  scripts/run_full_experiment.py \
  scripts/run_mild_imbalance_experiment.py \
  scripts/run_model_stress_benchmark.py \
  scripts/run_msbs_cwms_experiment.py \
  scripts/run_new_baselines_sweep.py \
  scripts/run_noise_ablation.py \
  scripts/run_oracle_paradox_analysis.py \
  scripts/run_pilot.py \
  scripts/run_pilot_method_a.py \
  scripts/run_pilot_method_b.py \
  scripts/run_statistical_tests.py \
  scripts/run_weak_supervision_relabeling.py \
  scripts/download_weak_supervision_datasets.py
```

### Step 2 — Delete dead data and logs

```bash
git rm -r catboost_info/
git rm -r data/weak-supervision/
```

### Step 3 — Verify imports still work

```bash
/home/than-minh/miniconda3/envs/dsp/bin/python -c "
from scripts.run_relabeling_viability_sweep import run_single_viability, DATASETS, QUICK_SEEDS
from scripts.run_cwms_msbs_deep_sweep import main
print('imports OK')
"
```

### Step 4 — Commit cleanup

```bash
git add -A
git commit -m "refactor: remove dead scripts and stale data from pre-CWMS-MSBS experiments"
```

## Success Criteria

- [ ] 29 dead scripts removed
- [ ] `catboost_info/` removed
- [ ] `data/weak-supervision/` removed
- [ ] All remaining scripts import cleanly
- [ ] `run_relabeling_viability_sweep.py` still importable and functional (core infrastructure)

## Risk Assessment

- **Import chains**: `run_relabeling_viability_sweep.py` imports from pipeline modules,
  not from any of the deleted scripts. Safe to delete the scripts without breaking the sweep.
- **Weak supervision data**: these parquet files are not used by any active script. Safe
  to remove. The download script for them (`download_weak_supervision_datasets.py`) is also deleted.
- **catboost_info/**: this is catboost's automatic training log directory. It has no
  dependencies — any future catboost run will re-create it.
