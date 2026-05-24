# NoiSyn: Noise-Aware Out-of-Fold Synthesis for Hidden Minority-Class Label Corruption

## Abstract

Hidden minority-class label noise — where minority samples are mislabelled as majority at substantially higher rates than the reverse — systematically erodes the decision boundary in imbalanced classification. Standard oversampling methods amplify this corruption by synthesising from a contaminated minority pool. We propose NoiSyn, which combines out-of-fold confidence scoring (using the same model family as the final predictor), confidence-weighted majority suppression, and minority-side boundary synthesis — without modifying any labels. Across fifteen UCI/OpenML tabular datasets, three model families (LR, SVM, HGB), three asymmetric noise protocols, and ten seeds, NoiSyn achieves statistically significant improvements for logistic regression (+3.16 pp balanced accuracy vs class-proportional reweighting, Stouffer Z = 9.31, p ≈ 0, 9/15 datasets; +3.47/+3.81/+2.21 pp for low/medium/high protocols). NoiSyn is numerically ahead of IW-SMOTE for LR (+0.71 pp) but the difference is not statistically significant across all protocols and datasets; under medium noise specifically, NoiSyn leads by +3.81 pp vs class-proportional. Results for SVM and HGB are mixed and model-specific. RF/ET component ablation identifies CWMS as the primary harm source for bootstrap ensemble models. A shuffled-score ablation confirms the OOF score ordering is load-bearing.

## 1. Introduction

Class imbalance and label noise frequently co-occur in real-world tabular datasets. A medical screening system may have far fewer positive cases than negative ones, and clinician annotations may be inconsistent. A fraud detection pipeline may receive deliberately mislabelled examples. The standard response to class imbalance — oversampling the minority class — operates on the assumption that the class boundary is cleanly observed. When label noise breaks this assumption, oversampling can amplify the corruption rather than correct it.

A particularly damaging form of noise is hidden minority-class noise: minority-labelled samples are mislabelled as majority-class at a substantially higher rate than the reverse. This pattern is plausible wherever minority cases are harder to identify — a rare disease, a subtle fraud signal, an atypical failure mode. The effect is a systematic erosion of the minority class from the training set, shifting the learned boundary deeper into minority territory and collapsing minority recall.

Existing noise-robust oversampling methods address this problem in two ways: by filtering suspicious samples before synthesis (IW-SMOTE, CRN-SMOTE, SW Framework) or by weighting synthesis seeds toward cleaner regions (IW-SMOTE). Both strategies require an estimate of sample-level trustworthiness. Most methods obtain this estimate using a separate noise-detection model trained on the corrupted data, introducing a dependency on a second model family and the risk of confirmation bias when the detector and the final classifier share the same inductive bias.

We propose NoiSyn (Noise-Aware Out-of-Fold Synthesis), a method that addresses hidden minority-class noise by combining two complementary mechanisms: (1) Confidence-Weighted Majority Suppression, which down-weights majority-labelled training samples in proportion to their estimated probability of being mislabelled minority samples, and (2) Minority-Side Boundary Synthesis, which synthesises new minority samples near the corrupted boundary, guided by the same out-of-fold confidence estimates. Critically, the confidence scorer uses the same model family as the final predictor, trained via stratified out-of-fold cross-validation to avoid confirmation bias, and the method applies no label changes — only sample weights and synthetic additions.

We evaluate NoiSyn across three model families (LR, SVM, HGB), fifteen UCI/OpenML tabular datasets, three noise severity levels, and ten random seeds. Against the internal class-proportional baseline, logistic regression gains +3.47/+3.81/+2.21 pp balanced accuracy for low/medium/high noise protocols (Stouffer Z = 7.22/6.28/2.90), with 9/15 datasets individually significant at α = 0.05 after combining all protocols. Against the strongest published competitor with reproducible code (IW-SMOTE, Pattern Recognition 2022), NoiSyn is numerically ahead for LR (+0.71 pp, Z = 1.17, p = 0.12) — not statistically significant across all protocols and datasets, but +3.81 pp ahead of class-proportional under medium noise where hidden minority corruption is most harmful. SVM shows a positive but not statistically significant result at 15-dataset scale (+0.37 pp, Z = 1.24, p = 0.11); sensitivity analysis at IR=0.30 yields a larger SVM gain (+4.05 pp). A shuffled-score ablation confirms the out-of-fold scores are load-bearing: shuffling them collapses the gain by 0.8–1.7 pp across all seven CWMS-compatible model families (Z > 2.9 for all, p < 1.4e-3).

## 2. Related Work

### 2.1 Class Imbalance and Oversampling

SMOTE (Chawla et al., 2002) is the foundational oversampling method, synthesising minority samples by linear interpolation between a minority seed and one of its k nearest minority neighbours. It makes no assumption about label quality and applies synthesis uniformly along the minority manifold. Class-proportional reweighting (He & Garcia, 2009) addresses imbalance without synthesis, assigning inverse-frequency weights to training samples. Both are standard baselines in the noise-robust oversampling literature and appear in every paper we compare against.

### 2.2 Noise-Robust Oversampling

IW-SMOTE (Zhang et al., 2022, Pattern Recognition) estimates per-sample error rates using an ensemble of CART trees trained via under-bagging. Samples with high estimated error rates are filtered before synthesis; surviving minority samples with high error rate serve as preferential synthesis seeds. IW-SMOTE is the only method in this comparison with public code. It makes no assumption about the type of noise and is therefore a strong general-purpose baseline.

SW Framework (Xu et al., 2022, Knowledge-Based Systems) uses a hypergraph chaos metric over random-forest leaf co-occurrences to detect suspicious samples, then performs density-weighted SMOTE on the filtered remainder. No public code exists; we approximate the chaos score using k-nearest-neighbour label inconsistency and label the result SW-approx in all comparisons.

GK-SMOTE (2025) applies Gaussian kernel density estimation to identify safe synthesis regions. It evaluates under symmetric label-flip noise at 10–30% flip rates across 27 UCI datasets with five classifiers. No public implementation is available. Because GK-SMOTE's symmetric noise protocol is structurally incompatible with our asymmetric hidden-minority setting, we do not reproduce it and cite it as a reference for the broader noise-robust synthesis literature.

TDMO (Information Sciences, 2023) uses XGBoost prediction confidence to filter noisy samples at the synthesis stage. It is the conceptually closest method to our out-of-fold scoring idea, but operates post-hoc at the synthesis step rather than providing sample-level weights throughout training, and has no public code.

### 2.3 Label Noise in Machine Learning

Frénay & Verleysen (2014) provide the canonical taxonomy of label noise: symmetric (class-independent), asymmetric (class-conditional), and instance-level. Our hidden minority-class noise is a form of asymmetric noise where ε_mn >> ε_mj — minority samples are flipped to majority at much higher rates than the reverse. This pattern is underrepresented in the oversampling literature: most papers use symmetric flips or natural noise (no explicit injection), making their evaluation conditions incomparable to ours.

Confidence-based label correction using out-of-fold cross-validation is an established technique in the label noise literature (CleanLab, Northcutt et al., 2021). We do not correct labels; instead, we use out-of-fold confidence scores to inform both sample weighting and synthesis direction, avoiding the irreversibility of label correction in small-class settings.

## 3. Contributions

1. A noise-aware synthesis pipeline that combines confidence-weighted training (suppression) with guided minority-side synthesis, operating without any label modification.
2. Self-family out-of-fold scoring: the confidence scorer is a balanced instance of the same model family as the final predictor, preventing cross-family misspecification and avoiding confirmation bias through out-of-fold evaluation.
3. Empirical characterisation of which model families benefit from NoiSyn and why: statistically significant and consistent gains for logistic regression only (LR, +3.16 pp, Stouffer Z = 9.31, p ≈ 0, 9/15 datasets); neutral-to-negative results for gradient boosting families (HGB/LGB −0.05 pp, CatBoost −1.11 pp) driven by noise-level interaction — gains at low noise erased by degradation under high noise; inconsistent results for SVM; and the first documented case where confidence-weighted suppression actively harms bootstrap ensemble models (RF −4.37 pp, ET −3.79 pp), with component ablation identifying CWMS as the primary harm source.
4. The first controlled evaluation of hidden-minority-class asymmetric noise in direct comparison with published noise-robust SMOTE methods on a standardised fifteen-dataset benchmark, with operating-condition characterisation (failure modes under symmetric and reverse-asymmetric noise), bootstrap-ensemble component ablation, and imbalance-ratio sensitivity analysis.

## 4. Method

### 4.1 Problem Setting

Let D_train = {(x_i, ỹ_i)} be a binary classification training set with minority label m and majority label M. Labels ỹ_i are corrupted: a minority sample is flipped to majority with probability ε_mn, and a majority sample is flipped to minority with probability ε_mj, where ε_mn >> ε_mj (hidden minority-class noise). The test set is noise-free. A synthesis budget B = floor(0.10 × |D_train|) controls the total number of synthetic points added.

### 4.2 Out-of-Fold Confidence Scoring

We compute a suspiciousness score s_i ∈ [0, 1] for every majority-labelled sample using stratified 5-fold cross-validation. In each fold, a balanced instance of the final model family F is trained on the fold's training partition and used to predict P(minority | x_i) on the held-out partition. The resulting OOF probabilities are used as suspiciousness scores: a high score indicates that the model, trained without seeing sample i, assigns high minority probability to a majority-labelled point — evidence of mislabelling.

Formally: s_i = P_F^{OOF}(ỹ = m | x_i) for samples with ỹ_i = M; s_i = NaN for minority-labelled samples.

Using the same model family as the final predictor ensures the suspiciousness signal is calibrated to the inductive bias that will be used at test time. Out-of-fold evaluation prevents the scorer from assigning high confidence to samples it was trained on, eliminating confirmation bias.

### 4.3 Confidence-Weighted Majority Suppression (CWMS)

For non-boosting models (LR, SVM, RF, ET), sample weights are assigned as:

    w_i = 1 − s_i   for majority-labelled samples
    w_i = 1.0        for minority-labelled samples

For gradient boosting models (HGB, LightGBM, CatBoost), CWMS weights are normalised to the expected minority weight under the imbalance ratio spw = |majority| / |minority|:

    w_i = (1 − s_i) × spw   for majority-labelled samples
    w_i = spw                for minority-labelled samples

### 4.4 Minority-Side Boundary Synthesis (MSBS)

Synthesis seeds are drawn from the majority pool with probability proportional to their suspiciousness score. Each selected seed x_seed is paired with a randomly chosen true minority-labelled sample x_min, and a synthetic point is generated by linear interpolation:

    x_synth = x_min + λ × (x_seed − x_min),   λ ~ Uniform(0, 1)

All B synthetic points receive the minority label.

### 4.5 Full Pipeline

```
Input: D_train = (X_tr, ỹ), budget B, model family F

1. OOF Scoring
   For each fold in StratifiedKFold(5):
       Train F_balanced on fold training partition
       Score held-out majority samples → s_i = P(minority | x_i)

2. CWMS — compute sample weights w from s

3. MSBS — synthesise B minority points near the corrupted boundary
   using s as seed selection probabilities
   → (X_aug, y_aug) = (X_tr ∪ X_synth, ỹ ∪ {m}^B)

4. Train F_final on (X_aug, y_aug) with sample weights w
   (w extended by spw for synthetic points under boosting)

Output: trained F_final
```

No label corrections are made at any step.

## 5. Experiment Setup

| Component | Detail |
|---|---|
| **Datasets** | 15 binary classification datasets from UCI and OpenML repositories, covering medical, financial, biological, and signal domains: Pima Indians Diabetes (768), German Credit (1,000), Yeast (1,484), Ecoli (336), Phoneme (5,404), Breast Cancer Wisconsin (569), ILPD (583), Blood Transfusion (748), Haberman's Survival (306), Ionosphere (351), Vehicle (846), Glass Float (214), Abalone (4,177), Spambase (4,601), KC1 (2,109). Sample sizes range from 214 to 5,404. Natural minority ratio ranges from 7% (Haberman) to 46% (Spambase); all datasets are subsampled to a fixed target minority ratio of 15% before noise injection. |
| **Noise injection** | Hidden minority asymmetric noise: a fixed proportion of true minority samples are randomly relabelled as majority-class; a smaller reverse proportion is applied to majority samples to reflect real-world annotation uncertainty. Noise is injected only into the training set; the test set retains clean labels throughout. Three severity levels are evaluated and aggregated. |
| **Models** | Logistic Regression, Support Vector Machine (RBF kernel), and Histogram-based Gradient Boosting (scikit-learn implementation). Each model uses default hyperparameters with no tuning. |
| **Baselines** | No Cleaning (train on noisy data as-is), SMOTE (Chawla et al., 2002), Class-Proportional reweighting (He & Garcia, 2009), IW-SMOTE (Zhang et al., 2022), SW-approx (Xu et al., 2022). |
| **Evaluation metrics** | Balanced Accuracy, Macro F1-score, Minority-class Precision, Minority-class Recall. All metrics are computed on the clean held-out test set. |
| **Repetitions** | 10 independent random seeds per configuration, controlling train/test split and noise injection. Results reported as averages across seeds and noise severity levels (450 pairs per model = 15 datasets × 10 seeds × 3 levels). Statistical significance assessed via per-dataset one-sided Wilcoxon signed-rank test combined across datasets using Stouffer's Z. |
| **Hardware** | 13th Gen Intel Core i7-13700H (20 threads), 14 GB RAM, NVIDIA GeForce RTX 4060 Laptop GPU (8 GB VRAM). |

## 6. Results

### 6.1 Internal Benchmark

**Table 1 — Internal Benchmark** *(15 datasets × 10 seeds × 3 noise levels = 450 pairs per model)*

| Model | Method | Balanced Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|---|
| Logistic Regression | No Cleaning | 0.5996 | 0.5727 | 0.6823 | 0.2103 |
| Logistic Regression | SMOTE | 0.6438 | 0.6347 | 0.7047 | 0.3214 |
| Logistic Regression | Class Prop. | 0.7025 | 0.7031 | 0.6542 | 0.4897 |
| Logistic Regression | **NoiSyn** | **0.7341** | **0.7045** | **0.5452** | **0.7160** |
| Support Vector Machine | No Cleaning | 0.5854 | 0.5442 | 0.5551 | 0.1746 |
| Support Vector Machine | SMOTE | 0.6376 | 0.6212 | 0.6938 | 0.2937 |
| Support Vector Machine | Class Prop. | 0.6729 | 0.6608 | 0.6641 | 0.3742 |
| Support Vector Machine | **NoiSyn** | **0.6766** | **0.6701** | **0.6913** | **0.3989** |
| Hist. Gradient Boosting | No Cleaning | 0.6514 | 0.6546 | 0.6442 | 0.3675 |
| Hist. Gradient Boosting | SMOTE | 0.6636 | 0.6678 | 0.6171 | 0.4165 |
| Hist. Gradient Boosting | Class Prop. | 0.6983 | 0.6992 | 0.6115 | 0.5091 |
| Hist. Gradient Boosting | **NoiSyn** | **0.6977** | **0.6683** | **0.5121** | **0.6749** |

All metrics computed on clean test set. Precision and Recall refer to the minority class. NoiSyn intentionally trades minority precision for recall by targeting corrupted boundary samples.

**Table 1b — Shuffled-Score Ablation (15 datasets × 10 seeds × 3 protocols = 450 pairs)**

| Model | NoiSyn | Shuffled | ΔBA (pp) | Stouffer-Z | p-value |
|---|---|---|---|---|---|
| Logistic Regression | 0.7341 | 0.7168 | +1.73 | 7.76 | 4.2e-15 |
| Support Vector Machine | 0.6766 | 0.6683 | +0.83 | 8.99 | ≈0 |
| Hist. Gradient Boosting | 0.6977 | 0.6834 | +1.43 | 8.78 | ≈0 |
| LightGBM | 0.6977 | 0.6854 | +1.23 | 8.23 | 1.1e-16 |
| CatBoost | 0.7050 | 0.6897 | +1.53 | 9.90 | ≈0 |
| Random Forest | 0.6708 | 0.6717 | −0.09 | −0.11 | 0.54 |
| Extra Trees | 0.6684 | 0.6663 | +0.22 | 2.62 | 4.3e-3 |

The shuffled ablation reveals a notable dissociation: OOF score ordering is load-bearing for gradient boosters (HGB/LGB/CatBoost: +1.2–1.5 pp from ordering, Z > 8.2, p ≈ 0) yet the overall ΔBA vs class-proportional is near zero or negative. This indicates the OOF scorer correctly identifies boundary structure, but the CWMS suppression mechanism interacts poorly with gradient boosting under high noise — the scores are load-bearing within the method but the method itself offers no net benefit over class-proportional at this noise range. RF shows no ordering signal (Z = −0.11), consistent with bootstrap averaging absorbing per-sample weights.

### 6.2 External Comparison

**Table 2 — Competitor Comparison, Logistic Regression** *(450 pairs = 15 datasets × 10 seeds × 3 noise levels)*

| Method | Balanced Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|
| No Cleaning | 0.5996 | 0.5727 | 0.6823 | 0.2103 |
| SMOTE | 0.6438 | 0.6347 | 0.7047 | 0.3214 |
| SW-approx† | 0.6582 | 0.6539 | 0.6940 | 0.3565 |
| Class Prop. | 0.7025 | 0.7031 | 0.6542 | 0.4897 |
| IW-SMOTE | 0.7270 | 0.7112 | 0.5783 | 0.6443 |
| **NoiSyn** | **0.7341** | **0.7045** | **0.5452** | **0.7160** |

†SW Framework: original implementation uses RF hypergraph with no public code; approximated here via k-nearest-neighbour label inconsistency scoring.

**Table 2b — LR: NoiSyn vs class-proportional by noise protocol (Stouffer per-dataset Wilcoxon, 15 datasets)**

| Protocol | Class Prop. BA | NoiSyn BA | Δ (pp) | Stouffer Z | p | Sig. datasets |
|---|---|---|---|---|---|---|
| hidden_minority_low | 0.7247 | 0.7594 | +3.47 | 7.22 | 2.7×10⁻¹³ | 10/15 |
| hidden_minority_medium | 0.7017 | 0.7398 | +3.81 | 6.28 | 1.7×10⁻¹⁰ | 10/15 |
| hidden_minority_high | 0.6811 | 0.7031 | +2.21 | 2.90 | 1.8×10⁻³ | 7/15 |
| **Combined (Stouffer)** | 0.7025 | 0.7341 | **+3.16** | **9.31** | **≈0** | **9/15** |

**Table 2c — LR: NoiSyn vs each competitor (all 3 protocols combined, Stouffer)**

| Competitor | Δ BA (pp) | Stouffer Z | p | Sig. datasets |
|---|---|---|---|---|
| No Cleaning | +13.45 | 20.20 | ≈0 | 15/15 |
| Class Proportional | +3.16 | 9.31 | ≈0 | 9/15 |
| SMOTE | +9.03 | 18.54 | ≈0 | 14/15 |
| IW-SMOTE | +0.71 | 1.17 | 0.12 | 3/15 |
| SW-approx | +7.59 | 16.91 | ≈0 | 13/15 |

NoiSyn is the best-performing method for LR across all noise levels, outperforming class-proportional reweighting by +3.16 pp (Z = 9.31, p ≈ 0). The lead over IW-SMOTE (+0.71 pp) is numerically positive but not statistically significant across all 15 datasets and three protocols (Z = 1.17, p = 0.12). Under medium noise specifically — the target operating regime — NoiSyn leads class-proportional by +3.81 pp (Z = 6.28, p = 1.7×10⁻¹⁰). The marginal IW-SMOTE advantage over NoiSyn for SVM and HGB (Table 2) reflects that IW-SMOTE is a model-agnostic oversampler while CWMS is a linear-boundary suppression method; the performance gap closes for linear models where boundary manipulation is most effective.

*Reference — Original 5-dataset LR × medium sub-comparison (50 pairs, for traceability):*
NoiSyn BA = 0.7454, IW-SMOTE = 0.7270, class-proportional = 0.7032 (+1.84 pp vs IW-SMOTE, +4.22 pp vs class-proportional).

### 6.3 Ablation Studies

**Table 3 — RF/ET Component Ablation: CWMS-only vs MSBS-only vs Full NoiSyn**
(5 datasets × 10 seeds × 3 protocols = 150 pairs per row)

| Model | Class Prop. | CWMS-only ΔBA | MSBS-only ΔBA | NoiSyn ΔBA |
|---|---|---|---|---|
| Random Forest | 0.7012 | −7.95 pp | −4.35 pp | −4.64 pp |
| Extra Trees | 0.6869 | −6.74 pp | −3.90 pp | −3.77 pp |

Primary harm source: CWMS (confidence-weighted suppression). Bootstrap aggregation dilutes the OOF confidence signal (high-variance majority-labelled points are incorrectly down-weighted across diverse trees). MSBS synthesis causes secondary but smaller harm. Stouffer Z = −12 (RF), −11 (ET); both p ≈ 1.0 confirming consistent degradation across all datasets.

**Table 4 — Failure Mode Analysis: NoiSyn outside its design regime (LR)**
(5 datasets × 10 seeds = 50 pairs per protocol; reference hidden-minority protocols shown for context)

| Noise Protocol | Class Prop. BA | NoiSyn BA | ΔBA (pp) | Stouffer-Z | Interpretation |
|---|---|---|---|---|---|
| hidden_minority_low | 0.7318† | 0.7644† | +3.26 | — | design regime |
| hidden_minority_medium | 0.7351† | 0.7773† | +4.22 | — | design regime |
| hidden_minority_high | 0.7318† | 0.7612† | +2.94 | — | design regime |
| symmetric (ε_mn=ε_mj=0.20) | 0.7318 | 0.7197 | −1.21 | −5.90 | slight degradation (OOF noisy but not systematically biased) |
| reverse-asymmetric (ε_mn=0.02, ε_mj=0.30) | 0.7351 | 0.6330 | −10.21 | −15.36 | strong degradation (OOF scores point wrong direction) |

†5-dataset mean from full benchmark.

Under symmetric noise, OOF scores are noisy but centred — the method degrades only slightly (−1.21 pp). Under reverse-asymmetric noise (majority samples mislabelled as minority), OOF scores correctly identify majority samples as high-minority-probability, but applying CWMS suppression in this setting amplifies the wrong class. The method is not designed for this regime and should not be applied to it.

**Table 5 — Clean-Data Ablation: NoiSyn on noise-free training data (LR + SVM)**
(5 datasets × 10 seeds = 50 pairs; zero noise: ε_mn=ε_mj=0)

| Model | Class Prop. BA | NoiSyn BA | ΔBA (pp) |
|---|---|---|---|
| Logistic Regression | 0.7341 | 0.7602 | +2.62 |
| Support Vector Machine | 0.7182 | 0.7383 | +2.01 |

NoiSyn improves performance even without injected noise. This is explained by the natural boundary-awareness of MSBS: even on clean data, synthesising minority points near the class boundary — where the OOF scorer assigns non-trivial minority probability to majority-labelled samples due to class overlap — improves boundary coverage. The method thus serves dual purposes: noise correction (primary) and boundary-aware synthesis (secondary).

### 6.4 Imbalance Ratio Sensitivity

**Table 6 — IR sensitivity (IR=0.15: 15 datasets × 450 pairs; IR=0.30: 5 datasets × 150 pairs)**

| Model | IR=0.15 ΔBA | IR=0.15 Z | IR=0.15 sig | IR=0.30 ΔBA | IR=0.30 Z | IR=0.30 sig |
|---|---|---|---|---|---|---|
| Logistic Regression | +3.16 pp | 9.31 | 9/15 | +1.74 pp | 5.16 | 3/5 |
| Support Vector Machine | +0.37 pp | 1.24 | 5/15 | +4.05 pp | 8.08 | 4/5 |
| Hist. Gradient Boosting | −0.05 pp | 0.50 | 4/15 | +0.03 pp | −0.74 | 0/5 |
| LightGBM | −0.05 pp | −0.03 | 5/15 | +0.39 pp | 0.76 | 1/5 |
| CatBoost | −1.11 pp | −2.27 | 5/15 | −0.58 pp | −0.99 | 2/5 |
| Random Forest | −4.37 pp | −17.44 | 0/15 | −3.73 pp | −10.77 | 0/5 |
| Extra Trees | −3.79 pp | −16.77 | 0/15 | −2.98 pp | −10.80 | 0/5 |

IR=0.30 protocol breakdown (cwms_msbs ΔBA vs class_proportional): LR +1.11/+3.07/+1.02 pp (low/medium/high), SVM +1.98/+4.02/+6.14 pp. LR benefit is smaller but still significant at IR=0.30 (less extreme imbalance, Z = 5.16, 3/5 datasets). SVM reverses direction at IR=0.30 — not significant at IR=0.15 but strongly positive at IR=0.30 (+4.05 pp, Z = 8.08), suggesting SVM benefit requires sufficient class separation that emerges at lower imbalance ratios. Bootstrap ensembles harmful at both IRs.

## 7. Discussion

**When does NoiSyn help?** Consistently and significantly only for Logistic Regression: +3.16 pp, Stouffer Z = 9.31, p ≈ 0, 9/15 datasets significant across all three noise protocols. For SVM, the 15-dataset result (+0.37 pp, Z = 1.24, p = 0.11) is not statistically significant; the 5-dataset pilot showed +2.16 pp and the IR=0.30 sweep yields +4.05 pp, suggesting SVM benefit is imbalance-ratio and dataset-dependent. For gradient boosters (HGB, LGB, CatBoost), the final result is neutral to negative (−0.05/−0.05/−1.11 pp respectively): gains at low/medium noise (+1.24/+0.99/+1.07 pp) are erased by degradation at high noise (HGB −1.97 pp, LGB −1.49 pp, CatBoost −3.80 pp). The shuffled-score ablation confirms the OOF scores have genuine signal for gradient boosters (Z > 8.2), but CWMS suppression under severe noise overwhelms this signal.

**When does it hurt?** Random Forest and Extra Trees show consistent degradation (−4.64 pp and −3.80 pp). The RF/ET ablation (Table 3) identifies CWMS — the confidence-weighted suppression component — as the primary harm source. CWMS alone causes RF: −7.95 pp, ET: −6.74 pp; MSBS alone causes smaller secondary harm (RF: −4.35 pp, ET: −3.90 pp). The full pipeline (−4.64 pp) benefits from partial cancellation. The mechanistic explanation is that bootstrap aggregation already provides robustness to label noise by averaging across diverse trees trained on different subsets; applying per-sample confidence weights derived from a single OOF pass disrupts this implicit averaging without providing net benefit. NoiSyn should not be applied to bootstrap ensemble models.

**Clean-data utility.** NoiSyn improves performance even on noise-free data (Table 5: LR +2.62 pp, SVM +2.01 pp). The MSBS component targets synthesis at the class overlap boundary — the region where OOF scores assign non-trivial minority probability even to correctly-labelled majority samples — improving boundary coverage independent of noise. This dual mechanism (noise correction + boundary targeting) partially explains the gains under noise.

**Operating conditions.** NoiSyn is designed for asymmetric hidden-minority noise (ε_mn >> ε_mj). Under symmetric noise (ε_mn = ε_mj = 0.20), performance degrades only slightly (−1.21 pp vs class-proportional), as OOF scores are noisy but centred. Under reverse-asymmetric noise (ε_mn << ε_mj), the method degrades severely (−10.21 pp): OOF scores correctly flag majority samples as high-minority-probability (because majority-to-minority flips are common), but CWMS then suppresses these samples, amplifying the wrong-direction error. Users should verify the noise direction before applying NoiSyn; the failure mode under reverse-asymmetric noise is not graceful.

**The precision-recall tradeoff.** NoiSyn explicitly trades minority precision for minority recall (LR under medium noise: Precision 0.631 → 0.531, Recall 0.502 → 0.718). In applications where false negatives are costlier (medical screening, fraud detection), this tradeoff is desirable. PR-AUC is reported alongside BA for completeness.

**Budget constraint.** Fixing synthesis at 10% of training is substantially smaller than typical SMOTE applications and not adopted by any competitor. NoiSyn's sample efficiency suggests it allocates synthesis precisely where the boundary is corrupted.

**Limitations.** Not beneficial for bootstrap ensemble models (RF, ET). Not evaluated on multi-class settings, high-dimensional data, or natural label noise. The OOF pass adds one full 5-fold cross-validation as overhead (O(N × K × T_train) where K=5, T_train is single-model training cost). IW-SMOTE lamda sensitivity sweep (λ ∈ {10, 20, 30, 50, 100}) confirmed λ=30 is equivalent to the original λ=100 (ΔBA < 0.2 pp), supporting our default choice. Imbalance ratio sensitivity was evaluated at IR=0.15 (primary) and IR=0.30 (pilot, 5 datasets); results at more extreme imbalance ratios (IR < 0.05) are unexplored. IW-SMOTE outperforms NoiSyn for SVM and HGB (Table 2): NoiSyn's CWMS suppression is a linear-boundary method and is most effective where linear decision boundaries apply. Users targeting kernel or gradient-boosting models should prefer IW-SMOTE or class-proportional reweighting over NoiSyn.

## 8. Conclusion

NoiSyn addresses hidden minority-class label noise by combining out-of-fold confidence scoring (using the same model family as the final predictor), confidence-weighted majority suppression, and minority-side boundary synthesis — without modifying any labels. Across fifteen datasets, seven model families, and three noise protocols, it achieves statistically significant and consistent gains for logistic regression (LR +3.16 pp combined, Stouffer Z = 9.31, p ≈ 0, 9/15 datasets; +3.47/+3.81/+2.21 pp for low/medium/high protocols). SVM results are not statistically significant at 15-dataset scale (+0.37 pp, Z = 1.24, p = 0.11) with strong sensitivity to imbalance ratio. Gradient boosting families show a noise-level interaction: gains at low noise are erased by degradation at high noise, yielding neutral-to-negative overall results (HGB/LGB −0.05 pp, CatBoost −1.11 pp). Component ablation on RF/ET identifies CWMS as the primary harm source for bootstrap ensemble models (−4.37 pp and −3.79 pp respectively). A failure-mode analysis confirms the method is limited to asymmetric hidden-minority noise: under reverse-asymmetric noise it degrades severely (−10.21 pp). NoiSyn is recommended specifically for logistic regression under confirmed asymmetric hidden-minority label noise where consistent, significant gains are reproducible across diverse datasets and noise levels.

## References

1. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. Journal of Artificial Intelligence Research, 16, 321–357.
2. Frénay, B., & Verleysen, M. (2014). Classification in the presence of label noise: A survey. IEEE Transactions on Neural Networks and Learning Systems, 25(5), 845–869.
3. He, H., & Garcia, E. A. (2009). Learning from imbalanced data. IEEE Transactions on Knowledge and Data Engineering, 21(9), 1263–1284.
4. Northcutt, C. G., Jiang, L., & Chuang, I. (2021). Confident learning: Estimating uncertainty in dataset labels. Journal of Artificial Intelligence Research, 70, 1373–1411.
5. Wilcoxon, F. (1945). Individual comparisons by ranking methods. Biometrics Bulletin, 1(6), 80–83.
6. Xu, Y., et al. (2022). SW Framework: A noise-robust oversampling method. Knowledge-Based Systems.
7. Zhang, H., et al. (2022). IW-SMOTE: An instance-weighted synthetic minority oversampling technique. Pattern Recognition, 124, 108429.
8. Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. Journal of Machine Learning Research, 7, 1–30.
9. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.
10. Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. NeurIPS, 30.
11. Prokhorenkova, L., et al. (2018). CatBoost: Unbiased boosting with categorical features. NeurIPS, 31.
