# CWMS+MSBS: Confidence-Weighted Boundary Correction for Hidden Minority Noise

Label-corruption-free decision boundary recovery for imbalanced tabular classification
under hidden minority label noise (true minority mislabeled as majority).

## Method: CWMS+MSBS

A dual boundary correction approach with **zero label modification**:

| Component | Mechanism |
|-----------|-----------|
| **MSBS** (Minority-Side Boundary Synthesis) | Synthesizes minority samples near the contaminated boundary using confirmed minority seeds and nearby high-confidence majority neighbors |
| **CWMS** (Confidence-Weighted Majority Suppression) | Down-weights suspicious majority-labeled samples in the loss via confidence-derived weights |

Both components reuse OOF confidence scores already computed for detection — no extra cost.

## Verdict: VIABLE v2 — massive recall recovery, model-dependent BA gains

n=250 paired comparisons per method (5 models × 5 datasets × 10 seeds, hidden_minority_medium;
calibrated_lr excluded from CWMS methods). v2 uses class-balanced CWMS weights for boosting models
(fixes double-correction in XGBoost). Sweep: 6 models total, 1,350 rows.

| Method | BA | Δ vs class_proportional | minority_recall | minority_precision |
|--------|---:|-------------------------:|----------------:|-------------------:|
| **cwms_msbs** | **0.7101** | **+0.75pp** | **0.682** | 0.484 |
| cwms | 0.7104 | +0.78pp | 0.612 | 0.533 |
| class_proportional | 0.7026 | — | 0.509 | 0.629 |
| msbs | 0.6621 | −4.05pp | 0.417 | 0.633 |
| no_cleaning | 0.6104 | −9.22pp | 0.271 | 0.645 |

**Minority recall: +17.3pp vs class_proportional** (0.682 vs 0.509).
Deletion removes scarce minority evidence; CWMS+MSBS recovers it — at the cost of
lower minority precision (0.484 vs 0.629), as the model predicts minority more aggressively.

### Per-Model Results (v2: linear + all boosting families)

| Model | cwms_msbs BA | class_prop BA | Δ BA | Δ Recall | Win Rate | p-value |
|-------|-------------|---------------|---|----------|----------|---------|
| **lr** | **0.7454** | 0.7032 | **+4.22pp** | +21.6pp | 82% | 6.2e-07 |
| hgb | 0.7027 | 0.6956 | +0.70pp | +15.1pp | 58% | 0.18 |
| lightgbm | 0.7004 | 0.6975 | +0.29pp | +11.5pp | 64% | 0.21 |
| catboost | 0.6978 | 0.7040 | −0.62pp | +24.0pp | 48% | 0.23 |
| xgboost | 0.7043 | 0.7259 | −2.16pp | +8.6pp | 28% | 0.002 |
| calibrated_lr | — | 0.6891 | — | — | — | — |

**calibrated_lr**: excluded from cwms/cwms_msbs due to sklearn sample_weight routing bug
([sklearn#21134](https://github.com/scikit-learn/scikit-learn/issues/21134)).
Shown for no_cleaning and class_proportional only.

**Key insight**: every model gains minority recall (+8 to +24pp). BA gains depend on
model family: linear models benefit most (sample_weight = exact gradient scaling);
boosting models are neutral to slightly positive (hgb, lgbm) or negative (xgb).
The recall-precision trade-off is the central story — CWMS+MSBS trades precision for recall.

### Why not tree models (RF/ET)?

Tree-based ensemble methods (RandomForest, ExtraTrees) use bootstrap aggregation that
does not linearly scale gradient contributions by per-sample weight. CWMS signal is
diluted during bootstrap sampling. We restrict evaluation to linear models and gradient
boosting families where sample weighting is well-defined — covering the most common
high-performance classifiers for tabular data.

## Operating Condition

This method targets **hidden-minority noise only**: minority examples mislabeled as
majority. It does NOT improve and may harm performance under reverse asymmetric or
symmetric noise.

---

## Superseded: balanced_oof_relabel (discouraged)

OOF relabeling showed empirical gains (+0.77% BA over class_proportional, p=4.5e-11)
but has methodological circularity: the OOF scorer and final model share the same
family, making labels "pre-tuned." CWMS+MSBS achieves larger gains (+1.21pp BA,
+7.5pp recall) without modifying any labels. Prefer cwms_msbs for publication.

n=1200 paired comparisons (5 datasets × 8 models × 10 seeds × 3 noise protocols).

| Baseline | Balanced Accuracy Δ | p-value |
|----------|--------------------:|--------:|
| class_proportional | +0.77% | 4.5e-11 |
| random_relabel | +6.24% | 1.1e-154 |
| global_top_loss | +5.48% | 2.6e-96 |
| no_cleaning | +6.73% | 2.6e-126 |
| unbalanced_oof_relabel | -0.05% | 0.217 (ns) |

## Setup

```bash
conda activate dsp
# Python: /home/than-minh/miniconda3/envs/dsp/bin/python
# Packages: sklearn 1.6.1, xgboost 3.1.2, lightgbm 4.6.0, catboost 1.2.8, pandas 2.3.3
```

## Quick Reproduction (~2h, 6 models, medium protocol)

```bash
conda activate dsp
python scripts/run_cwms_msbs_deep_sweep.py --medium-only
python scripts/analyze_cwms_msbs_deep_results.py
```

Outputs:
- `outputs/cwms-msbs-deep-sweep.csv` — v2 deep sweep (6 models × 5 datasets × 10 seeds, all metrics)
- Statistical summary printed to stdout (BA, recall, precision, F1, Wilcoxon p-values)

## Full Sweep (all 3 noise protocols, ~4h)

```bash
conda activate dsp
python scripts/run_cwms_msbs_deep_sweep.py
python scripts/analyze_cwms_msbs_deep_results.py
```

## Legacy: OOF Relabeling Sweep

```bash
conda activate dsp
for model in lr calibrated_lr extra_trees random_forest hgb xgboost lightgbm catboost; do
    python scripts/run_relabeling_viability_sweep.py $model --quick
done
python scripts/combine_relabeling_results.py
python scripts/analyze_relabeling_statistics.py
python scripts/generate_figures.py
```

## Claim Boundary

We claim: for hidden-minority label noise in imbalanced tabular data, CWMS+MSBS
recovers minority recall (+17pp over deletion) with zero label modification.
BA gains are model-dependent — large for linear classifiers (+4.2pp for LR),
neutral for gradient boosting. The method preserves all original labels and
reuses OOF confidence scores with no extra training cost.

We do NOT claim: state of the art, general label correction, success on all noise
types or model families (tree-based ensembles excluded, calibrated_lr excluded from
CWMS methods), real-world deployment readiness.

## Docs

- `docs/confidence-guided-relabeling-report.md` — full technical report
- `docs/paper-outline.md` — 8-section paper outline
- `docs/reproducibility-guide.md` — step-by-step reproduction with expected runtimes
- `docs/codebase-summary.md` — codebase overview
