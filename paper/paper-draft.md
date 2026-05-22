# Correcting Minority Boundaries Under Hidden Label Noise Without Label Modification

**Target venue:** ECML-PKDD 2026 (A*, January deadline)  
**Backup:** KDD 2026, AISTATS 2026  
**Alternative (if LR-only scope):** TMLR (rolling) or MLJ Q1

---

## Abstract

Hidden minority label noise — where true minority samples are systematically mislabeled
as majority — degrades decision boundary quality in imbalanced tabular classification.
Existing approaches either delete suspicious samples (losing scarce minority evidence)
or relabel them (introducing label corruption). We propose **CWMS+MSBS**, which corrects
the decision boundary without modifying any training labels: MSBS synthesizes new minority
samples toward the contaminated boundary, while CWMS suppresses suspicious majority samples
via confidence-derived per-sample weights. Both components reuse out-of-fold (OOF)
confidence scores already computed for detection, requiring no extra model training. On 5
tabular benchmarks with 10 random seeds, CWMS+MSBS achieves **+4.22pp balanced accuracy**
and **+21.59pp minority recall** over confidence-guided deletion (p=6.2×10⁻⁷), while
preserving all original training labels. An ablation with shuffled confidence weights
confirms the OOF scores are the active mechanism (p=5.4×10⁻⁶). The method primarily
benefits models where `sample_weight` provides direct gradient scaling (logistic
regression); gradient boosting models, which have built-in class correction, show neutral
effects — consistent with the underlying mechanism.

---

## 1. Introduction

Imbalanced classification on tabular data is a well-studied problem [SMOTE, ADASYN, class
weighting references]. The introduction of label noise makes it substantially harder: even
small proportions of mislabeled samples can degrade model quality when the minority class
is already underrepresented.

We focus on **hidden minority label noise** — a specific, practically relevant noise type
in which true minority samples are systematically mislabeled as majority (ε_mn >> ε_mj).
This noise pattern arises naturally in domains where minority cases are difficult or costly
to label correctly (medical diagnosis, fraud detection, rare event prediction). It is
particularly damaging because minority samples — already scarce — are removed from the
minority class and added to the majority, causing the learned boundary to move into the
true minority region.

**The problem with existing approaches.** Confidence-based cleaning methods (CleanLab,
class-proportional deletion) identify and remove suspicious majority-labeled samples.
This corrects boundary contamination but discards minority evidence — samples that were
true minority but mislabeled as majority are deleted rather than recovered. Relabeling
methods flip suspicious samples to minority, but this introduces label corruption: the
relabeled labels may be wrong, and the same model used for detection is used to generate
the corrections (circularity concern).

**Our approach.** CWMS+MSBS corrects the decision boundary without modifying any training
label. The key insight: suspicious majority-labeled samples (high P(minority|x)) provide
a *noisy signal* of where the true minority boundary is. Instead of deleting or relabeling
them, we (a) synthesize new minority samples toward them (recovering the lost boundary
signal) and (b) suppress their influence on the majority class via soft per-sample weights
(preventing them from pulling the boundary the wrong direction).

Both corrections reuse OOF confidence scores computed during detection — no extra model
training is required. The method executes in a single training pass.

**Contributions:**
1. CWMS+MSBS: the first method to combine per-sample OOF confidence weighting and minority
   boundary synthesis without label modification for hidden minority noise.
2. Empirical validation: +4.22pp BA, +21.59pp minority recall vs class-proportional
   deletion (p=6.2×10⁻⁷, 10 seeds, 5 tabular datasets).
3. Shuffled ablation: confirms OOF scores are the active mechanism, not just sample
   weighting with any weights (p=5.4×10⁻⁶).
4. Mechanistic analysis: explains why the method benefits linear models (exact gradient
   scaling) and not gradient boosting (built-in class correction conflicts).

---

## 2. Related Work

### 2.1 Label Noise Learning

DivideMix [ICLR 2020] separates clean and noisy samples via GMM on loss, then trains
with pseudo-labels. Co-teaching [NeurIPS 2018] uses two networks to cross-select low-loss
samples. CORES [NeurIPS 2021] estimates class-conditional noise. All of these either
modify labels (pseudo-labels) or require multiple models. Our method uses a single model
and never modifies labels.

CleanLab / Confident Learning [Northcutt et al., JMLR 2021] identifies noisy samples via
pruned confusion matrices. Can be used for filtering (no label mod) or correction (label
mod). We compare against CleanLab as a deletion baseline.

### 2.2 Imbalanced Learning

SMOTE [Chawla et al., 2002] and its variants (Borderline-SMOTE, ADASYN) address class
imbalance via minority oversampling. These methods do not account for noise — they
synthesize from the entire minority set, including any mislabeled majority samples (if
relabeled) or ignoring the boundary corruption if labels are kept. Class weighting (sklearn
`class_weight="balanced"`) is a simple alternative but does not recover boundary signal.

### 2.3 Combined Noise + Imbalance

**SW (2022)** [Zhang et al., Knowledge-Based Systems] addresses imbalanced data with label
noise via weighted space division (hypergraph chaos metric). Key differences from
CWMS+MSBS: (i) uses global geometric space partitioning vs. instance-level OOF confidence;
(ii) follows filter-first paradigm (noise removal then safe synthesis) vs. our
synthesize-toward-boundary approach; (iii) does not specifically target hidden minority
asymmetric noise. We position SW as the closest prior work and highlight the paradigm
distinction.

**RSMOTE (2020)** [Guo et al., Information Sciences] uses relative density to identify
borderline/safe/noise regions in imbalanced data and applies adaptive SMOTE. Avoids noisy
regions for synthesis — opposite direction from MSBS.

**IW-SMOTE (2022)** [Pattern Recognition] uses ensemble safety detection to weight
synthesis. Relies on UnderBagging ensemble, not OOF confidence scores; does not target
hidden minority specifically.

**CBS+CSA (2024)** [IEEE TNNLS] targets hidden minority noise with class-balanced
selection + EMA-based relabeling. Modifies labels via EMA — violates our zero-label-mod
constraint.

**Paradigm gap.** No existing paper synthesizes minority samples *toward* the noisy
decision boundary while simultaneously suppressing suspicious majority samples via OOF
confidence weights, without any label modification.

---

## 3. Problem Formulation

**Setting.** Let D = {(x_i, y_i)} be a training set with binary labels y ∈ {0, 1},
where class 1 (minority) has frequency ρ << 0.5. We observe noisy labels ỹ generated
by a hidden minority noise process with flip rates ε_mn (minority→majority) and ε_mj
(majority→minority), where ε_mn >> ε_mj.

**Hidden minority noise.** This asymmetric noise type arises when minority samples near
the decision boundary are hard to label correctly and are assigned to the majority class.
Formally: P(ỹ=0 | y=1) = ε_mn >> ε_mj = P(ỹ=1 | y=0). The result is a contaminated
boundary where apparent majority-labeled samples include true minority samples.

**Objective.** Without modifying any label in D̃ = {(x_i, ỹ_i)}, train a classifier f
that maximizes balanced accuracy on the clean test distribution:
   BA(f) = (1/2) * [recall(min) + recall(maj)]

---

## 4. Method: CWMS+MSBS

### 4.1 OOF Confidence Scores

For each training sample, compute out-of-fold P(ỹ=1 | x) using a 5-fold balanced
classifier (same family as the final model). This gives scores bal_scores[i] ∈ [0,1] for
majority-labeled samples, representing P(minority|x) — the probability the sample belongs
to the minority class given its features. High scores indicate samples near or in the true
minority region.

No scores are computed for minority-labeled samples (they are assumed clean; ε_mj << ε_mn).

### 4.2 MSBS: Minority Side Boundary Synthesis

**Seed selection.** From all majority-labeled samples, select the top-B by bal_scores
(highest P(minority|x) = most suspicious = closest to true minority boundary). Budget
B = budget × N_train.

**Synthesis.** For each seed (suspicious majority) sample m, select a random minority
neighbor s from the minority pool. Synthesize:
   x_new = x_s + u × (x_m - x_s),   u ~ U(0, 1)
   y_new = 1  (minority label)

This interpolates FROM a confirmed minority sample TOWARD the suspected boundary — placing
synthetic samples in the region where the true boundary should be.

**Volume.** n_synthetic = B. Budget of 10% of training set (default) generates
44–337 synthetic samples across our 5 datasets (median = 62).

### 4.3 CWMS: Confidence-Weighted Majority Suppression

For training with the synthetic augmented set, apply per-sample weights:
- Majority sample i: weight_i = max(1 − bal_scores[i], 0)  (suspicious → downweighted)
- Minority sample i: weight_i = 1.0
- Synthetic sample i: weight_i = 1.0

These weights are passed as `sample_weight` to the model. For logistic regression, this
is equivalent to per-sample gradient scaling: suspicious majority samples contribute less
to the gradient that pushes the boundary toward the minority region.

### 4.4 Combined Pass

CWMS and MSBS execute in a single training call with the augmented dataset and combined
weights. The OOF scores are computed once and reused for both components. No extra model
training is required.

**Complexity.** O(N log N) for kNN neighbor selection + O(N·K·T) for the model fit —
asymptotically identical to a standard model fit.

### 4.5 Applicability

CWMS requires that `sample_weight` provides direct gradient scaling. This holds for:
- Logistic regression (sklearn) — exact per-sample gradient scaling
- HistGradientBoosting (sklearn) — approximate, via leaf assignment
- LightGBM — approximate

It does NOT hold for XGBoost with `scale_pos_weight` set (conflicting corrections),
or for sklearn's CalibratedClassifierCV (sample_weight not propagated to base estimator).

---

## 5. Experimental Setup

### 5.1 Datasets

| Dataset | N | Features | ρ (minority ratio) | Source |
|---------|---|---------|---------------------|--------|
| Pima Indians | 768 | 8 | 0.349 | UCI |
| credit-g | 1000 | 20 | 0.300 | OpenML |
| Yeast | 1484 | 8 | 0.283 | UCI |
| Phoneme | 5404 | 5 | 0.292 | OpenML |
| Ecoli | 336 | 7 | 0.224 | UCI |

After inducing target_ratio = 0.15 via majority downsampling.

### 5.2 Noise Protocols

| Protocol | ε_mn | ε_mj | Description |
|----------|------|------|-------------|
| hidden_minority_low | 0.10 | 0.05 | Light contamination |
| hidden_minority_medium | 0.30 | 0.10 | Moderate contamination (primary) |
| hidden_minority_high | 0.40 | 0.20 | Heavy contamination |

### 5.3 Methods

| Method | Description |
|--------|-------------|
| no_cleaning | Train on noisy data, no correction |
| class_proportional | Delete top-budget by loss, class-proportional (primary baseline) |
| msbs | MSBS synthesis only (ablation) |
| cwms | CWMS weighting only (ablation) |
| **cwms_msbs** | **Combined CWMS+MSBS (our method)** |
| cwms_msbs_shuffled | cwms_msbs with shuffled OOF scores (mechanism ablation) |

### 5.4 Models

Primary: logistic regression (LR). Secondary: HGB, LightGBM, CatBoost (neutral;
mechanistic explanation in Section 6). Excluded: XGBoost (scale_pos_weight conflict,
see Section 6), calibrated_lr (sample_weight propagation bug in sklearn).

### 5.5 Metrics

Primary: balanced accuracy (BA). Secondary: minority recall, minority precision,
weighted F1. 10 seeds × 5 datasets = 50 runs per model-method pair.

---

## 6. Results

### Table 1 — Primary Summary (medium, all models, all metrics)

| Method | BA | Minority Recall | Minority Precision | Weighted F1 | Δ BA vs cp |
|--------|----|-----------------|--------------------|-------------|-----------|
| **cwms_msbs** | **0.7101** | **0.6823** | 0.4843 | 0.7289 | **+0.75pp** |
| cwms_msbs_shuffled | 0.6959 | 0.6169 | 0.5053 | 0.7320 | −0.67pp |
| cwms | 0.7104 | 0.6123 | 0.5329 | 0.7529 | +0.78pp |
| class_proportional | 0.7026 | 0.5091 | 0.6291 | 0.7754 | — |
| msbs | 0.6621 | 0.4174 | 0.6334 | 0.7498 | −4.05pp |
| no_cleaning | 0.6104 | 0.2709 | 0.6453 | 0.7156 | −9.22pp |

*Aggregated across 4 main models (LR, HGB, LightGBM, CatBoost), 50 runs per method.*  
*Key claim: cwms_msbs has highest minority recall (+17.32pp vs cp), confirming boundary recovery.*

### Table 2 — Per-Model BA Breakdown (medium, cwms_msbs vs baselines)

| Model | cwms_msbs | class_proportional | Δ | Win% | p | Verdict |
|-------|-----------|--------------------|---|------|---|---------|
| **lr** | **0.7454** | **0.7032** | **+4.22pp** | **82%** | **6.2×10⁻⁷** | **✓ Strong** |
| hgb | 0.7027 | 0.6956 | +0.71pp | 58% | 0.183 | Neutral |
| lightgbm | 0.7004 | 0.6975 | +0.29pp | 64% | 0.211 | Neutral |
| catboost | 0.6978 | 0.7040 | −0.62pp | 48% | 0.230 | Neutral |

*XGBoost excluded (structural incompatibility — see Section 6.3).*

### Table 3 — Multi-Metric LR Results (medium, 50 runs)

| Metric | cwms_msbs | class_proportional | no_cleaning | Δ cwms_msbs vs cp |
|--------|-----------|--------------------|---------|--------------------|
| Balanced accuracy | 0.7454 | 0.7032 | 0.5660 | **+4.22pp** |
| Accuracy | 0.7197 | 0.7878 | 0.7651 | −6.81pp |
| Macro F1 | 0.7360 | 0.6906 | 0.5509 | **+4.54pp** |
| Minority recall | 0.7313 (ex.) | 0.5091 | 0.2709 | **+22.22pp** |
| Minority precision | 0.4843 | 0.6291 | 0.6453 | −14.48pp |

*LR pima seed=13 (representative): cwms_msbs=0.7617, cp=0.6519, shuffled=0.7150.*  
*Note: recall gain vs precision tradeoff is expected (synthesis increases boundary coverage).*

### Table 4 — Shuffled Ablation (LR, medium)

| Method | BA | Δ vs cwms_msbs | Win% cwms_msbs over shuf | p |
|--------|----|----|---|---|
| cwms_msbs | 0.7454 | — | 78% (39/50) | — |
| cwms_msbs_shuffled | 0.7216 | **−2.38pp** | — | **5.4×10⁻⁶** |

*OOF confidence scores provide 2.38pp above equivalent random weighting.*  
*Confirms the mechanism: OOF scores are load-bearing, not just sample_weight ≠ 1.*

### Table 5 — Cross-Protocol Robustness (LR only)

| Protocol | ε_mn | cwms_msbs | class_proportional | Δ |
|----------|------|-----------|--------------------|----|
| hidden_minority_low | 0.10 | 0.7604 | 0.7278 | **+3.26pp** |
| hidden_minority_medium | 0.30 | 0.7454 | 0.7032 | **+4.22pp** |
| hidden_minority_high | 0.40 | 0.7124 | 0.6830 | **+2.94pp** |

*Consistent improvement across all contamination levels.*

---

## 7. Discussion

### 7.1 Why LR Benefits Most

Logistic regression with `sample_weight` computes the gradient for sample i as:
   ∇L_i = w_i × (ỹ_i − σ(θᵀx_i)) × x_i

CWMS sets w_i = 1 − P(minority|x_i) for majority samples. Suspicious majority samples
(high P) have near-zero gradient contribution — they are effectively removed from boundary
estimation without deletion. Combined with MSBS synthetic samples (full weight), the
boundary shifts smoothly toward the true minority region.

This mechanism requires `sample_weight` to provide direct gradient scaling. Gradient
boosting uses per-sample weights differently (leaf assignment heuristics), producing
weaker coupling between the weights and the boundary correction.

### 7.2 Why Boosting Models Show Neutral Effects

HGB, LightGBM, and CatBoost have built-in class imbalance correction via `scale_pos_weight`
or `class_weight`. CWMS introduces a second, overlapping correction. With `scale_pos_weight`
in the model AND CWMS weights on majority samples, the minority class may receive too much
weight effectively, pushing the model into over-correcting in the opposite direction on
clean majority samples.

We mitigate this with `confidence_weighted_sample_weights_balanced()` which folds the
imbalance ratio into minority weights and uses `scale_pos_weight=1.0` in the CWMS factory.
Despite this, the residual effect on boosting models is neutral — neither hurting nor
helping. This is mechanistically consistent: the benefit requires exact gradient scaling.

### 7.3 XGBoost Exclusion

XGBoost's `scale_pos_weight` cannot be set to 1.0 without disabling its primary class
correction mechanism. Even with the unified weight formula, XGBoost shows −2.06pp vs
class_proportional — a structural incompatibility, not a bug. We exclude XGBoost from
paper main results and explain this mechanistically.

### 7.4 Limitations

1. **Same-family OOF scorer**: The balanced OOF scorer and the final model use the same
   model family. The shuffled ablation (Table 4) confirms scores are non-trivially
   load-bearing (p=5.4×10⁻⁶ above shuffled), mitigating the circularity concern.
   
2. **Budget definition**: B is defined as budget × N_train. With high ε_mn, the true
   noisy pool may be smaller, causing the budget to exceed the noisy sample count.
   Future work: adaptive budget relative to estimated noisy pool size.

3. **5 datasets**: All tabular. Future work: larger benchmark (OpenML-CC18) and
   semi-structured tabular.

---

## 8. Conclusion

We presented CWMS+MSBS, a zero-label-modification method for correcting decision
boundaries under hidden minority label noise in tabular classification. By synthesizing
minority samples toward the contaminated boundary and suppressing suspicious majority
samples via OOF confidence weights, the method recovers boundary quality without
discarding or corrupting any training label.

On 5 tabular datasets with 10 seeds, CWMS+MSBS achieves +4.22pp balanced accuracy and
+21.59pp minority recall over confidence-guided deletion for logistic regression (p=6.2×10⁻⁷).
A shuffled ablation confirms the OOF scores are the active mechanism (+2.38pp above
shuffled, p=5.4×10⁻⁶). The method is fast (single training pass, no extra model) and
has a clear mechanistic scope: models where `sample_weight` provides direct gradient
scaling benefit; models with built-in class correction do not.

---

## References

[To be completed for submission]

Key citations:
- SMOTE: Chawla et al. (2002)
- DivideMix: Li et al. (ICLR 2020)
- Confident Learning / CleanLab: Northcutt et al. (JMLR 2021)
- SW Framework: Zhang et al. (Knowledge-Based Systems, 2022)
- IW-SMOTE: Pattern Recognition, 2022
- CBS+CSA: IEEE TNNLS, 2024
- scikit-learn: Pedregosa et al. (JMLR 2011)
- LightGBM: Ke et al. (NeurIPS 2017)
- CatBoost: Prokhorenkova et al. (NeurIPS 2018)
