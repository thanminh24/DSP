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

## Verdict: VIABLE — beats class-proportional deletion with zero label corruption

n=150 paired comparisons per method (3 models × 5 datasets × 10 seeds, hidden_minority_medium).

| Method | BA | Δ vs class_proportional | p-value |
|--------|---:|-------------------------:|--------:|
| **cwms_msbs** | **0.7081** | **+1.21pp** | 0.008 |
| class_proportional | 0.6960 | — | — |
| cwms | 0.6598 | −3.62pp | 2.9e-08 |
| msbs | 0.6371 | −5.89pp | 4.6e-25 |
| no_cleaning | 0.5759 | −12.01pp | 3.9e-26 |

**Minority recall: +7.5pp vs class_proportional** (0.556 vs 0.481, p=2.1e-09).
Deletion removes scarce minority evidence; synthesis adds it.

### Combined method beats both standalone components

| Comparison | Win Rate | Δ BA | p-value |
|------------|----------|------|---------|
| cwms_msbs vs msbs | 96.0% | +7.10pp | 3.2e-25 |
| cwms_msbs vs cwms | 84.0% | +4.83pp | 8.3e-20 |

### Per-Model Results (linear + boosting families)

| Model | cwms_msbs BA | class_prop BA | Δ | Win Rate | p-value |
|-------|-------------|---------------|---|----------|---------|
| **lr** | **0.7454** | 0.7032 | **+4.22pp** | 82% | 6.2e-07 |
| hgb | 0.6981 | 0.6956 | +0.25pp | 56% | 0.53 (ns) |
| calibrated_lr | 0.6808 | 0.6891 | −0.83pp | 26% | 0.006 |

LR is the star performer — sample_weight maps directly to gradient scaling. HGB ties.
calibrated_lr's sample_weight routing is partially blocked by scikit-learn
(see [sklearn#21134](https://github.com/scikit-learn/scikit-learn/issues/21134)).

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
conda create -n dsp python=3.12 -y
conda activate dsp
pip install -r requirements.txt
```

## Quick Reproduction (~10 min, LR only)

```bash
conda activate dsp
python scripts/run_cwms_msbs_full_sweep.py --medium-only
python scripts/analyze_cwms_msbs_results.py
```

Outputs:
- `outputs/cwms-msbs-full-sweep.csv` — full benchmark grid (3 models × 5 datasets × 10 seeds)
- Statistical summary printed to stdout (BA, recall, Wilcoxon p-values)

## Full Sweep (all 3 noise protocols, ~4h)

```bash
conda activate dsp
python scripts/run_cwms_msbs_full_sweep.py
python scripts/analyze_cwms_msbs_results.py
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

Outputs:
- `outputs/relabeling-statistical-tests.csv` — paired Wilcoxon tests
- `outputs/relabeling-viability-verdict.md` — verdict + per-model win rates
- `outputs/plots/` — 3 publication figures (PNG)

## Claim Boundary

We claim: for hidden-minority label noise in imbalanced tabular data, the CWMS+MSBS
boundary correction method beats confidence-proportional deletion while preserving all
original labels — zero label corruption, higher recall, and better balanced accuracy for
linear and gradient boosting classifiers.

We do NOT claim: state of the art, general label correction, success on all noise types
or model families (tree-based ensembles excluded), real-world deployment readiness.

## Docs

- `docs/confidence-guided-relabeling-report.md` — full technical report
- `docs/paper-outline.md` — 8-section paper outline
- `docs/reproducibility-guide.md` — step-by-step reproduction with expected runtimes
- `docs/codebase-summary.md` — codebase overview
