# Codebase Summary

## Overview

The active codebase supports confidence-guided relabeling experiments for imbalanced noisy
tabular classification. Historical CRCC deletion docs and plans are archived under ignored
archive folders.

## Active Modules

- `pipeline/data/loaders.py`: cached KEEL/UCI dataset loading, imbalance induction, and
  synthetic class-dependent label noise.
- `pipeline/scoring/balanced_oof.py`: class-balanced out-of-fold minority-confidence scores.
- `pipeline/augmentation/relabeling.py`: top-k majority-pool relabeling and random control.
- `pipeline/models/factories.py`: publication-grade model factories, including optional
  XGBoost/LightGBM/CatBoost.
- `pipeline/baselines/`: cleanlab and confidence-relabeling baselines.
- `scripts/audit_relabeling_protocol.py`: leakage and protocol audit.
- `scripts/run_relabeling_viability_sweep.py`: controlled synthetic benchmark.
- `scripts/run_model_stress_benchmark.py`: model-family stress benchmark.
- `scripts/analyze_relabeling_statistics.py`: paired statistics and verdict output.

## Active Research Plan

`plans/260521-0000-confidence-guided-relabeling-viability-proof/plan.md`

## Unresolved Questions

- Weak-supervision dataset source not finalized.
- Full boosting runtime not yet measured.
