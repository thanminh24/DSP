---
phase: 4
title: "Archive Outputs and Drop venv"
status: pending
priority: P2
effort: "~15m"
dependencies: []
---

# Phase 4: Archive Outputs and Drop venv

## Overview

Compress superseded result CSVs into `outputs/archive/superseded-results.tar.gz`,
delete the 1.6 GB `.venv/` folder, and remove the leftover `catboost_info/` training logs.
All active results (`cwms-msbs-full-sweep.csv`, `cwms-msbs-deep-sweep.csv`) remain in place.

## What to Archive (compress, keep in archive)

These CSVs are superseded by the CWMS+MSBS sweep but retain historical value for reference.
Move to `outputs/archive/` then compress:

```
outputs/relabeling-viability-lr.csv
outputs/relabeling-viability-lr-old.csv
outputs/relabeling-viability-calibrated_lr.csv
outputs/relabeling-viability-hgb-complete.csv
outputs/relabeling-viability-xgboost-broken-oof.csv
outputs/relabeling-viability-random_forest.csv
outputs/relabeling-viability-extra_trees.csv
outputs/relabeling-viability-lightgbm.csv
outputs/relabeling-viability-catboost.csv
outputs/relabeling-viability-results.csv
outputs/relabeling-all-results-combined.csv
outputs/relabeling-statistical-tests.csv
outputs/budget-ablation-results.csv
outputs/budget-ablation-summary.csv
outputs/noise-ablation-results.csv
outputs/noise-ablation-summary.csv
outputs/augment-sweep-results.csv
outputs/augment-statistical-tests.csv
outputs/augment-main-table.md
outputs/augment-noise-scaling.md
outputs/augment-final-verdict.md
outputs/cgms-poc-results.csv
outputs/crcc-smoke-test-results.csv
outputs/cwms-msbs-combined-prelim.csv
outputs/diagnostic-tbdc.csv
outputs/full-experiment-results.csv
outputs/mild-imbalance-results.csv
outputs/mild-imbalance-summary.csv
outputs/model-stress-results.csv
outputs/msbs-cwms-poc-results.csv
outputs/noise-ablation-results.csv
outputs/oracle-paradox-analysis.csv
outputs/pilot-method-a-results.csv
outputs/pilot-method-a-log.txt
outputs/pilot-method-a-decision.md
outputs/pilot-method-b-results.csv
outputs/pilot-method-b-log.txt
outputs/pilot-method-b-decision.md
outputs/statistical-tests-results.csv
outputs/summary-table.csv
outputs/weak-supervision-relabeling-results.csv
outputs/plots/  (archive the old figures too)
```

## What to Keep in outputs/ (active)

```
outputs/cwms-msbs-full-sweep.csv          # v1 full sweep, reference
outputs/cwms-msbs-deep-sweep.csv          # v2 deep sweep (Phase 2 output)
outputs/crcc-academic-references-25-papers.md   # literature references
outputs/relabeling-viability-verdict.md   # historical verdict doc
outputs/relabeling-protocol-audit.md      # historical audit doc
```

## Implementation Steps

### Step 1 — Create archive and compress

```bash
cd /home/than-minh/project/DSP/outputs

mkdir -p archive

# Move superseded files
mv relabeling-viability-*.csv archive/
mv relabeling-all-results-combined.csv relabeling-statistical-tests.csv archive/
mv budget-ablation-*.csv noise-ablation-*.csv archive/
mv augment-sweep-results.csv augment-statistical-tests.csv archive/
mv augment-main-table.md augment-noise-scaling.md augment-final-verdict.md archive/
mv cgms-poc-results.csv crcc-smoke-test-results.csv archive/
mv cwms-msbs-combined-prelim.csv archive/
mv diagnostic-tbdc.csv full-experiment-results.csv archive/
mv mild-imbalance-*.csv model-stress-results.csv archive/
mv msbs-cwms-poc-results.csv oracle-paradox-analysis.csv archive/
mv pilot-method-*.csv pilot-method-*.txt pilot-method-*.md archive/
mv statistical-tests-results.csv summary-table.csv archive/
mv weak-supervision-relabeling-results.csv archive/
mv plots/ archive/

# Compress the archive
tar -czf archive/superseded-results.tar.gz -C archive \
  --exclude="superseded-results.tar.gz" .

# After verifying the tar is valid, remove the uncompressed archived files
tar -tzf archive/superseded-results.tar.gz | head -20  # verify
find archive/ -not -name "superseded-results.tar.gz" -type f -delete
find archive/ -not -name "superseded-results.tar.gz" -mindepth 1 -type d -delete
```

### Step 2 — Delete .venv (1.6 GB)

The `.venv/` folder is the old project virtualenv. All runs now use conda `dsp`.

```bash
cd /home/than-minh/project/DSP
rm -rf .venv/
```

Note: `.venv/` is typically gitignored so this doesn't need a git commit.
Verify: `grep -r "\.venv" .gitignore || cat .gitignore`

### Step 3 — Update requirements.txt

`requirements.txt` currently reflects the venv. Update it to document the conda env packages:

```
# Install via: conda activate dsp
# or: /home/than-minh/miniconda3/envs/dsp/bin/pip install -r requirements.txt
scikit-learn>=1.6
xgboost>=3.0
lightgbm>=4.0
catboost>=1.2
pandas>=2.3
scipy>=1.16
numpy>=1.26
openml
```

### Step 4 — Verify disk savings

```bash
du -sh /home/than-minh/project/DSP/.venv 2>/dev/null || echo ".venv deleted OK"
du -sh /home/than-minh/project/DSP/outputs/
ls /home/than-minh/project/DSP/outputs/
```

## Success Criteria

- [ ] `outputs/archive/superseded-results.tar.gz` exists and contains all archived files
- [ ] `outputs/` contains only: `cwms-msbs-full-sweep.csv`, `cwms-msbs-deep-sweep.csv`,
  `crcc-academic-references-25-papers.md`, `relabeling-viability-verdict.md`,
  `relabeling-protocol-audit.md`, and `archive/`
- [ ] `.venv/` deleted; 1.6 GB recovered
- [ ] `requirements.txt` updated to document conda env
- [ ] `outputs/archive/superseded-results.tar.gz` verifies with `tar -tzf`

## Risk Assessment

- **Archive corruption**: compress first, verify with `tar -tzf`, then delete. Never delete
  originals before confirming the tar is valid.
- **.venv in .gitignore**: if `.venv/` is gitignored, deleting it doesn't require a git commit.
  If it was accidentally committed, use `git rm -r --cached .venv/` first.
- **Relabeling verdict docs**: `relabeling-viability-verdict.md` and `relabeling-protocol-audit.md`
  are kept in active `outputs/` since they document the historical decision to pivot to CWMS+MSBS.
  These are markdown, tiny, and contextually valuable. Do not archive them.
