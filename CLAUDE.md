# CLAUDE.md — DSP / NoiSyn Project

## Project Summary

**NoiSyn** (Noise-Aware OOF Synthesis): a label-corruption-free method for imbalanced tabular classification under *hidden minority label noise* (minority samples mislabeled as majority). Two components:

- **CWMS** (Confidence-Weighted Majority Suppression): down-weight suspicious majority-labeled samples via OOF P(minority|x) scores
- **MSBS** (Minority-Side Boundary Synthesis): synthesize minority samples near the contaminated boundary using confirmed minority seeds

Zero label modification. OOF confidence scores feed both components — no extra training cost.

**Claim boundary**: works for LR and SVM (strong, significant). Marginal/neutral for HGB/LGB. Actively hurts RF/ET (OOF signal diluted by bootstrap). Only for hidden-minority asymmetric noise, not symmetric or reverse-asymmetric.

## Python Environment

```bash
/home/than-minh/miniconda3/envs/dsp/bin/python
# Packages: sklearn 1.6.1, xgboost 3.1.2, lightgbm 4.6.0, catboost 1.2.8, pandas 2.3.3
# GPU: CUDA confirmed for XGBoost, LightGBM, CatBoost
```

## Key Code Locations

| File | Purpose |
|------|---------|
| `scripts/run_relabeling_viability_sweep.py` | Core sweep dispatcher — all method dispatchers live here |
| `scripts/run_full_benchmark_solution.py` | Full benchmark runner (all models × methods × protocols × seeds) |
| `scripts/run_competitor_headtohead.py` | External comparison vs IW-SMOTE, SMOTE, SW-framework |
| `scripts/download_datasets.py` | One-time OpenML download → `data/*.parquet` |
| `scripts/analyze_full_benchmark.py` | Table 1 analysis (BA, p-values, effect sizes) |
| `scripts/analyze_competitor_headtohead.py` | Table 2 analysis |
| `pipeline/data/loaders.py` | Dataset registry; `load_dataset(name) → (X_df, y_binary, cat_cols, feature_names)` |
| `pipeline/evaluation/metrics.py` | `evaluate()` — cleaning/weighting methods evaluator |
| `pipeline/evaluation/augment_metrics.py` | `evaluate_augmented()` — synthesis methods evaluator |
| `pipeline/models/factories.py` | `make_model_factory(model_name, seed, cat_indices, balanced=False)` |
| `pipeline/baselines/iw_smote.py` | IW-SMOTE baseline; `lamda=30` in current calls (original paper default: 100) |
| `docs/paper-draft.md` | Full 8-section paper draft |
| `docs/results-reference.md` | All key numbers consolidated |

## Critical Code Facts

**Self-family OOF scorer (cwms_msbs):** `bal_scores` computed at `run_relabeling_viability_sweep.py:205-207` using `bal_factory = make_model_factory(model_name, ..., balanced=True)` — each model uses its OWN balanced OOF variant. This is already self-family; no new runs needed.

**Method keys** (all existing, no new dispatchers):
- `cwms` — line 336: suppression only, no synthesis
- `msbs` — line 329: synthesis only, no weight modification
- `cwms_msbs` — line 352: full pipeline (paper's main method)
- `oracle_relabel` — line 282: uses ground-truth labels (appendix only, not deployable)

**Dataset loader interface:** `load_dataset(name) -> (X_df, y_binary, cat_cols, feature_names)` — returns parquet-cached data. No live OpenML at experiment time.

**Baseline-only models:** `xgboost` and `calibrated_lr` get `BASELINE_ONLY_METHODS` in `run_full_benchmark_solution.py:29` (`_methods_for()`). Do NOT add cwms/msbs keys for these.

**Row count math (15 datasets):**
- 7 CWMS-full models × 7 methods × 3 protocols × 10 seeds × 15 datasets = 22,050
- 2 baseline-only models × 3 methods × 3 protocols × 10 seeds × 15 datasets = 2,700
- **Total: 24,750 rows** (existing 5-dataset run = 8,250 = 1,650/dataset × 5 ✓)

## Current Datasets (5)

`["pima", "credit-g", "yeast", "phoneme", "ecoli"]` — all cached in `data/*.parquet`

## Active Plan

`plans/260523-1200-noisyn-paper-hardening/` — 6-phase hardening to reach PR/KBS/TNNLS tier:

| Phase | Focus | Key output |
|-------|-------|-----------|
| 1 | Quick fixes (no experiments) | Fix "1,050 pairs", PR-AUC in both evaluators, per-dataset Wilcoxon+Stouffer, binary label assertion |
| 2 | Fast sweeps on 5 datasets | Failure-mode protocols, RF/ET ablation (cwms vs msbs), IW-SMOTE lamda sensitivity, clean-data ablation |
| 3 | Dataset expansion to 15 | Download 10 new OpenML datasets, register in loaders.py, re-run benchmark → 24,750 rows |
| 4 | IR=0.30 sweep | Same benchmark at target_ratio=0.30 → 24,750 rows |
| 5 | Expanded external comparison | LR+SVM+HGB × 3 protocols × 15 datasets → 8,100 rows |
| 6 | Paper rewrite | Hardened tables, honest framing, per-dataset p-values |

## Statistical Approach

**Do NOT use aggregate Wilcoxon with 150/450 pairs treated as i.i.d.** — observations share seeds/datasets, not independent.

**Correct approach:** `per_dataset_wilcoxon_stouffer()` in `analyze_full_benchmark.py`:
- Per-dataset Wilcoxon over (seed × protocol) pairs (30 per dataset = 10 seeds × 3 protocols)
- Stouffer's Z combination across datasets (fully independent between datasets)
- Report: "X/15 datasets individually significant" + "Stouffer Z=X.X, p=X.Xe-X"

## Key Decisions (do not reverse without flagging)

- **Method name**: NoiSyn (accepted by user)
- **Oracle relabel**: Appendix only — not a Table 1 row. Reference as "upper bound, see Appendix A."
- **IW-SMOTE lamda**: Currently 30; Phase 2 Sweep C gates Phase 5. Change lamda in `iw_smote()` call only if lamda=100 wins by >0.5pp.
- **SW-framework**: Keep in Table 2 with dagger footnote if validation shows approximation is adequate; remove to Appendix if clearly weaker.
- **RF/ET**: Not claimed to benefit from NoiSyn. Discussed honestly in Discussion section with ablation evidence from Phase 2.
- **calibrated_lr**: Excluded from cwms/msbs methods due to sklearn#21134 sample_weight routing bug. Baseline-only.

## Existing Outputs (do not overwrite)

| File | Rows | Description |
|------|------|-------------|
| `outputs/full-benchmark-solution.csv` | 8,250 | Original 5-dataset benchmark |
| `outputs/competitor-headtohead.csv` | 300 | LR × medium × 5 datasets external comparison |
| `outputs/cwms-msbs-deep-sweep.csv` | — | Per-model deep sweep (oracle reference) |
| `outputs/scorer-agnosticism-sweep.csv` | — | Cross-family OOF scorer ablation |
