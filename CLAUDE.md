# CLAUDE.md — DSP / COINS Project

## Project Summary

**COINS** (Confidence Out-of-fold Imbalanced Noise Synthesis): a label-corruption-free method for imbalanced tabular classification under *hidden minority label noise* (minority samples mislabeled as majority). Two components:

- **CWMS** (Confidence-Weighted Majority Suppression): down-weight suspicious majority-labeled samples via OOF P(minority|x) scores
- **MSBS** (Minority-Side Boundary Synthesis): synthesize minority samples near the contaminated boundary using confirmed minority seeds

Zero label modification. OOF confidence scores feed both components — no extra training cost.

**Claim boundary**: works for LR and SVM (strong, significant). Marginal/neutral for HGB/LGB. Actively hurts RF/ET (OOF signal diluted by bootstrap). Only for hidden-minority asymmetric noise, not symmetric or reverse-asymmetric.

**Paper title**: "COINS: Out-of-Fold Confidence Scoring for Noise-Robust Synthesis in Imbalanced Classification"

## Python Environment

```bash
/home/than-minh/miniconda3/envs/dsp/bin/python
# Packages: sklearn 1.6.1, xgboost 3.1.2, lightgbm 4.6.0, catboost 1.2.8, pandas 2.3.3
# GPU: CUDA confirmed for XGBoost, LightGBM, CatBoost
```

## Key Code Locations

| File | Purpose |
|------|---------|
| `pipeline/experiments/run_relabeling_viability_sweep.py` | Core sweep dispatcher — all method dispatchers live here |
| `pipeline/experiments/run_full_benchmark_solution.py` | Full benchmark runner (all models × methods × protocols × seeds) |
| `pipeline/experiments/run_expanded_competitor_headtohead.py` | External comparison vs IW-SMOTE, SMOTE, SW-framework (15 datasets) |
| `pipeline/experiments/download_datasets.py` | One-time OpenML download → `data/*.parquet` |
| `pipeline/experiments/analyze_full_benchmark.py` | Table 1 analysis (BA, PR-AUC, per-dataset Wilcoxon+Stouffer) |
| `pipeline/experiments/analyze_expanded_competitor_headtohead.py` | Table 2 analysis (15-dataset competitor comparison) |
| `pipeline/data/loaders.py` | Dataset registry; `load_dataset(name) → (X_df, y_binary, cat_cols, feature_names)` |
| `pipeline/evaluation/metrics.py` | `evaluate()` — cleaning/weighting methods evaluator |
| `pipeline/evaluation/augment_metrics.py` | `evaluate_augmented()` — synthesis methods evaluator |
| `pipeline/models/factories.py` | `make_model_factory(model_name, seed, cat_indices, balanced=False)` |
| `pipeline/baselines/iw_smote.py` | IW-SMOTE baseline; `lamda=30` (original paper default: 100; gated by iw-lamda-sweep) |
| `docs/paper/paper-draft.md` | Full 8-section paper draft |
| `docs/experiments/results-reference.md` | All key numbers consolidated |
| `docs/research/COINS-literature-review.xlsx` | 25-paper literature review (2-sheet xlsx) |

## Critical Code Facts

**Self-family OOF scorer (cwms_msbs):** `bal_scores` computed at `pipeline/experiments/run_relabeling_viability_sweep.py:205-207` using `bal_factory = make_model_factory(model_name, ..., balanced=True)` — each model uses its OWN balanced OOF variant. This is already self-family; no new runs needed.

**Method keys** (all existing, no new dispatchers):
- `cwms` — line 336: suppression only, no synthesis
- `msbs` — line 329: synthesis only, no weight modification
- `cwms_msbs` — line 352: full pipeline (paper's main method = COINS)
- `oracle_relabel` — line 282: uses ground-truth labels (appendix only, not deployable)

**Dataset loader interface:** `load_dataset(name) -> (X_df, y_binary, cat_cols, feature_names)` — returns parquet-cached data. No live OpenML at experiment time.

**Baseline-only models:** `xgboost` and `calibrated_lr` get `BASELINE_ONLY_METHODS` in `pipeline/experiments/run_full_benchmark_solution.py:29` (`_methods_for()`). Do NOT add cwms/msbs keys for these.

**Row count math (15 datasets):**
- 7 CWMS-full models × 7 methods × 3 protocols × 10 seeds × 15 datasets = 22,050
- 2 baseline-only models × 3 methods × 3 protocols × 10 seeds × 15 datasets = 2,700
- **Total: 24,750 rows** ✓ (confirmed in `docs/experiments/raw/full-benchmark-solution-v2.csv`)

## Datasets (15)

All cached in `data/*.parquet`:
`abalone, blood, breast_cancer, credit-g, ecoli, glass_float, haberman, ilpd, ionosphere, kc1, phoneme, pima, spambase, vehicle_bus, yeast`

## Output Files

| File | Rows | Purpose |
|------|------|---------|
| `docs/experiments/COINS-all-results.xlsx` | 49,250 | **All experiments** in one xlsx (9 tabs + README) |
| `docs/experiments/COINS-all-results.csv` | 49,250 | Same data flat CSV with `experiment` tag column |
| `docs/experiments/raw/full-benchmark-solution-v2.csv` | 24,750 | Main benchmark — 15 datasets, Table 1 |
| `docs/experiments/raw/full-benchmark-ir030-solution.csv` | 8,250 | IR=0.30 sensitivity sweep, Table 6 |
| `docs/experiments/raw/competitor-headtohead-expanded.csv` | 8,100 | External comparison, Table 2 |
| `docs/experiments/raw/cwms-msbs-deep-sweep.csv` | 4,350 | Deep sweep + oracle reference, Appendix |
| `docs/experiments/raw/rfet-ablation-sweep.csv` | 1,500 | RF/ET failure-mode ablation |
| `docs/experiments/raw/scorer-agnosticism-sweep.csv` | 1,250 | Self-family OOF scorer ablation |
| `docs/experiments/raw/clean-data-ablation.csv` | 400 | Zero-noise ablation |
| `docs/experiments/raw/failure-mode-sweep.csv` | 400 | Symmetric/reverse-asymmetric noise |
| `docs/experiments/raw/iw-lamda-sweep.csv` | 250 | IW-SMOTE lambda sensitivity |

Superseded 5-dataset v1 runs archived in `docs/experiments/raw/archive/superseded-results.tar.gz`.

## Statistical Approach

**Do NOT use aggregate Wilcoxon with 150/450 pairs treated as i.i.d.** — observations share seeds/datasets, not independent.

**Correct approach:** `per_dataset_wilcoxon_stouffer()` in `analyze_full_benchmark.py`:
- Per-dataset Wilcoxon over (seed × protocol) pairs (30 per dataset = 10 seeds × 3 protocols)
- Stouffer's Z combination across datasets (fully independent between datasets)
- Report: "X/15 datasets individually significant" + "Stouffer Z=X.X, p=X.Xe-X"

## Key Decisions (do not reverse without flagging)

- **Algorithm name**: COINS (Confidence Out-of-fold Imbalanced Noise Synthesis)
- **Oracle relabel**: Appendix only — not a Table 1 row. Reference as "upper bound, see Appendix A."
- **IW-SMOTE lamda**: 30 (gated by iw-lamda-sweep; change only if lamda=100 wins by >0.5pp).
- **SW-framework**: Keep in Table 2 with dagger footnote if validation shows approximation is adequate; remove to Appendix if clearly weaker.
- **RF/ET**: Not claimed to benefit from COINS. Discussed honestly in Discussion with ablation evidence.
- **calibrated_lr**: Excluded from cwms/msbs methods due to sklearn#21134 sample_weight routing bug. Baseline-only.
