---
phase: 5
title: "Update Docs"
status: pending
priority: P2
effort: "~30m"
dependencies: [2]
---

# Phase 5: Update Docs

## Overview

Update all project documentation to reflect: (1) CWMS+MSBS v2 is the primary method,
(2) v2 numbers from the deep sweep, (3) conda `dsp` as the development environment,
(4) cleaned repo structure. Populate the paper-outline with updated results framing.

## Related Code Files

- Modify: `README.md` — update verdict table with v2 numbers + model scope table
- Modify: `docs/codebase-summary.md` — reflect cleaned repo structure and active modules
- Modify: `docs/paper-outline.md` — update claims with v2 results and per-model scope
- Modify: `docs/reproducibility-guide.md` — replace venv instructions with conda `dsp`
- Modify: `docs/research-foundation.md` — note OOF relabeling discouraged section (brief)

## Implementation Steps

### Step 1 — Update README.md

After the deep sweep (Phase 2) analysis is complete, update:

1. **Verdict table**: replace v1 numbers (5 models) with v2 numbers (6 models including catboost).
   Use hidden_minority_medium, primary method = cwms_msbs.

2. **Model scope table** — replace existing per-model breakdown with v2 results:

```markdown
### Per-Model Results (v2: linear + all boosting families)

| Model | cwms_msbs BA | class_prop BA | Δ | Win Rate | p-value |
|-------|-------------|---------------|---|----------|---------|
| lr | X.XXXX | X.XXXX | +Xpp | X% | X |
| hgb | X.XXXX | X.XXXX | +Xpp | X% | X |
| xgboost | X.XXXX | X.XXXX | +Xpp | X% | X |
| lightgbm | X.XXXX | X.XXXX | +Xpp | X% | X |
| catboost | X.XXXX | X.XXXX | +Xpp | X% | X |

calibrated_lr: shown only for no_cleaning vs class_proportional — sample_weight routing
limitation in sklearn's CalibratedClassifierCV prevents CWMS from reaching the base estimator.
```

3. **Multi-metric section** (new): add a table showing BA, recall, precision, F1 for
   cwms_msbs vs class_proportional to make the paper claim multi-dimensional.

4. **Environment note**: replace any venv references with:
   ```
   conda activate dsp
   # or: /home/than-minh/miniconda3/envs/dsp/bin/python ...
   ```

5. **Operating condition**: keep the existing "hidden_minority_* only" warning.

### Step 2 — Update docs/codebase-summary.md

Rewrite the **Active Modules → Scripts** section:

```markdown
### Scripts (`scripts/`) — active only

| Script | Purpose |
|--------|---------|
| `run_relabeling_viability_sweep.py` | Core sweep infrastructure; imports `run_single_viability()` |
| `run_cwms_msbs_deep_sweep.py` | Primary experiment runner (v2: all 6 models, all metrics) |
| `analyze_cwms_msbs_deep_results.py` | Primary analysis: Wilcoxon tests, per-model tables |
| `analyze_cwms_msbs_results.py` | Legacy analysis for v1 full sweep |
| `download_datasets.py` | Fetch OpenML datasets |
| `validate_environment.py` | Environment smoke test |
```

Rewrite the **Outputs** section:

```markdown
### Outputs (`outputs/`)

| File | Contents |
|------|---------|
| `cwms-msbs-deep-sweep.csv` | PRIMARY — v2 deep sweep (6 models, all metrics) |
| `cwms-msbs-full-sweep.csv` | v1 sweep (5 models, BA+recall only); reference only |
| `archive/superseded-results.tar.gz` | All pre-pivot experiment results |
| `relabeling-viability-verdict.md` | Historical: why relabeling was discouraged |
| `relabeling-protocol-audit.md` | Historical: OOF circularity audit |
| `crcc-academic-references-25-papers.md` | Literature references |
```

Add **Conda Environment** section:

```markdown
### Environment

Python: `/home/than-minh/miniconda3/envs/dsp/bin/python`  
Packages: sklearn 1.6.1, xgboost 3.1.2, lightgbm 4.6.0, catboost 1.2.8, pandas 2.3.3

Run sweeps:
```bash
/home/than-minh/miniconda3/envs/dsp/bin/python scripts/run_cwms_msbs_deep_sweep.py --medium-only
```
```

### Step 3 — Update docs/paper-outline.md

In the abstract / contribution section, update to reflect v2 findings:

- Contribution: "CWMS+MSBS achieves significant improvement for linear classifiers (LR: +Xpp BA,
  p<0.001) and gradient boosting families, without any label modification."
- Model scope: "We evaluate on LR, HGB, XGBoost, LightGBM, CatBoost. CalibratedLR is excluded
  from CWMS methods due to a known sklearn sample_weight routing limitation."
- Drop any reference to `balanced_oof_relabel` as the main contribution (it's superseded).

### Step 4 — Update docs/reproducibility-guide.md

Replace all `source .venv/bin/activate` or `pip install -r requirements.txt` instructions with:

```bash
# Activate the conda environment
conda activate dsp
# OR use the full path:
/home/than-minh/miniconda3/envs/dsp/bin/python scripts/run_cwms_msbs_deep_sweep.py
```

Update the "Data download" step to use `dsp` python.

Update the "Run sweep" command to point to `run_cwms_msbs_deep_sweep.py`.

Update the "Expected outputs" section to reference `outputs/cwms-msbs-deep-sweep.csv`.

### Step 5 — Archive the completed old plan

Mark `plans/260522-1600-cwms-msbs-full-sweep-paper/plan.md` status as `complete`:

```bash
cd plans/260522-1600-cwms-msbs-full-sweep-paper
ck plan check 1
ck plan check 2
```

## Success Criteria

- [ ] README.md verdict table shows v2 numbers (post-deep-sweep)
- [ ] README.md per-model table includes catboost; notes calibrated_lr limitation
- [ ] `docs/codebase-summary.md` scripts section lists only active scripts
- [ ] `docs/reproducibility-guide.md` references conda `dsp` env, not venv
- [ ] `docs/paper-outline.md` contribution framing is CWMS+MSBS, not relabeling
- [ ] Old plan marked complete

## Risk Assessment

- **Numbers not available**: Phase 5 depends on Phase 2 completing. If Phase 2 is still
  running, write the structural updates first (env instructions, script list) and leave
  placeholder `X.XXXX` in tables to fill after the sweep finishes.
- **Codebase-summary stale modules**: `pipeline/augmentation/filtered_smote.py` and
  `pipeline/augmentation/synthesis.py` (CGMS code) are still in the repo but not used
  by the active sweep. Note them as "unused, candidate for removal in future cleanup."
  Do not delete pipeline modules in this phase — risk of breaking import chains.
