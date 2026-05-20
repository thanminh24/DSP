# Class-Risk-Constrained Label Cleaning for Imbalanced Tabular ML

## Problem

- Tabular classification datasets often contain both label noise and class imbalance.
- Standard noisy-label workflows usually:
  - Train a model.
  - Score samples by suspiciousness, often loss, margin, or prediction disagreement.
  - Remove, downweight, or relabel the most suspicious samples.
- The risky part is the intervention step.
- In imbalanced tabular data, minority-class samples are often harder to learn.
- Hard but correctly labeled minority samples can receive high loss.
- Global top-k cleaning may therefore delete clean minority samples, not just mislabeled samples.
- This can improve average performance while damaging minority recall.
- Standard metrics such as accuracy and macro-F1 may not directly expose this cleaning harm.

## Motivation

- This project focuses on the practitioner setting:
  - Dataset is tabular.
  - Model is classical ML or small gradient boosting, not a deep vision pipeline.
  - Cleaning budget is small.
  - Experiment time is limited.
  - Method must be simple enough to implement and explain.
- Existing noisy-label research has strong work on detection and robust training.
- Less practical guidance exists for the narrow post-detection question:
  - Given suspiciousness scores, which samples are safe to clean?
- Class-aware noisy-label methods already exist, so the novelty claim must be narrow.
- The defensible gap is:
  - Existing class-aware noisy-label methods mostly integrate selection, relabeling, representation learning, and training inside deep-learning pipelines.
  - This project studies a lightweight, model-agnostic post-detection rule for imbalanced tabular ML.
- The paper value:
  - Small but explicit method.
  - Clear harm metric.
  - Compact tabular evaluation.
  - No need for large benchmark-style evaluation.

## Contribution

- Propose **Class-Risk-Constrained Cleaning (CRCC)**:
  - A lightweight post-detection cleaning rule for imbalanced tabular ML.
  - Works after any suspiciousness scorer.
  - Combines risk-adjusted suspiciousness scoring with class-level deletion caps.
  - Prevents over-cleaning classes that are more likely to contain hard-but-correct samples.
- Separate detection from intervention:
  - Detection: score each sample by suspiciousness.
  - Intervention: decide whether deletion is safe under class-risk constraints.
- Add a harm-aware evaluation metric:
  - **Clean-minority deletion rate**: fraction of selected/deleted samples that were actually clean minority-class samples.
- Validate the method across a compact tabular setting:
  - Three tabular datasets.
  - Two model families.
  - One controlled class-dependent noise protocol.
  - Goal is a normal conference paper, not a large benchmark paper.

## Method

- Input:
  - Training data: `(X_train, y_noisy)`.
  - Known or estimated cleaning budget `B`, e.g. 10% of training set.
  - Base classifier.
  - Suspiciousness scoring function.
- Step 1: Create noisy imbalanced training data.
  - Start from clean tabular labels.
  - Induce class imbalance if needed.
  - Inject controlled synthetic label noise.
- Step 2: Score samples.
  - Train classifier with k-fold cross-validation.
  - Collect out-of-fold predicted probabilities.
  - Compute suspiciousness using cross-entropy loss or negative margin.
- Step 3: Estimate class risk.
  - Compute class-level risk from observed labels and model behavior.
  - Simple option: minority class has higher protection risk.
  - Stronger option: class risk is proportional to mean class loss or inverse class frequency.
- Step 4: Compute risk-adjusted cleaning score.
  - Penalize suspicious samples from high-risk classes before deletion.
- Step 5: Apply baseline global deletion.
  - Rank all samples by suspiciousness.
  - Delete top `B` samples globally.
- Step 6: Apply proposed CRCC deletion.
  - Rank samples by risk-adjusted score.
  - Before deleting a sample, check whether its observed class has exceeded its class deletion cap.
  - If cap is exceeded, skip that sample and move to the next suspicious sample.
- Step 7: Retrain.
  - Train the same classifier on the cleaned training set.
  - Evaluate on a clean held-out test set.

### CRCC Score

```text
adjusted_score_i = suspiciousness_i - lambda * class_risk[observed_class_i]
```

- `suspiciousness_i`:
  - Cross-entropy loss under out-of-fold predicted probability.
  - Higher means more suspicious.
- `class_risk[c]`:
  - Risk that deleting from class `c` removes hard-but-correct samples.
  - For the initial version:
    - Majority risk = 0.
    - Minority risk = 1.
  - For the stronger version:
    - `class_risk[c] = mean_loss[c] / max_class_mean_loss`.
- `lambda`:
  - Controls protection strength.
  - Use small grid: `{0, 0.25, 0.5, 1.0}`.
  - `lambda = 0` reduces to cap-only cleaning.

### CRCC Deletion Rule

```text
for sample i in adjusted_score_ranked_samples:
    c = observed_class(i)
    if total_deleted < B and deleted_count[c] < class_cap[c]:
        delete i
```

### Class Cap Options

- CRCC-P: Proportional cap.
  - Each class can contribute deletions proportional to its observed training frequency.
  - Simple and least controversial.
- CRCC-M: Minority-protected cap.
  - Minority class receives a stricter deletion cap than proportional allocation.
  - More directly protects minority samples.
  - Use as ablation, not as the only method.
- Recommended paper version:
  - Main method: CRCC-P with risk-adjusted scoring.
  - Ablation: cap-only CRCC-P with `lambda = 0`.
  - Optional ablation: CRCC-M.

### Why This Is More Than Stratified Deletion

- Plain stratified deletion only changes budget allocation.
- CRCC changes both:
  - Which samples are ranked highly for cleaning.
  - How many samples each observed class can lose.
- The method is still simple, but the paper contribution is the full package:
  - Post-detection intervention framing.
  - Risk-adjusted cleaning score.
  - Class deletion caps.
  - Clean-minority deletion metric.
  - Tabular ML evaluation.

## Dataset Used

- Dataset 1:
  - **Pima Indians Diabetes**.
  - 768 rows.
  - 8 predictive features plus binary outcome.
  - Natural class split often reported around 500 negative / 268 positive.
  - All features are numeric medical measurements, so preprocessing is simple.
  - Strong fit for tabular healthcare ML.
  - Weakness: small dataset, so use repeated seeds and avoid overclaiming.
- Dataset 2:
  - **OpenML Credit-G**.
  - 1,000 rows.
  - 20 attributes.
  - Binary target: good or bad credit risk.
  - Mixed numeric/categorical features.
  - Strong fit for tabular ML and fairness-style discussion.
  - Weakness: needs one-hot encoding or categorical preprocessing.
- Dataset 3:
  - **OpenML Sick / Thyroid**.
  - 3,772 rows.
  - 30 features.
  - Highly imbalanced binary thyroid disease classification variant.
  - Strong fit for minority-class harm.
  - Weakness: more missing/categorical handling than Pima.
- Avoid for main experiment:
  - **Bank Marketing**.
  - 45,211 rows.
  - Useful tabular dataset, but too large for quick repeated noisy-label runs.
  - Better as future-work mention.
- Emergency local fallback:
  - `sklearn.datasets.load_breast_cancer`.
  - 569 rows.
  - 30 numeric features.
  - 2 classes.
  - Available directly in scikit-learn, no network dependency.
  - Use only if OpenML/Kaggle access becomes annoying.

## Dataset Decision

- Paper-target setup:
  - Use Pima Indians Diabetes, Credit-G, and Sick/Thyroid.
- Class-project fallback:
  - Use Pima Indians Diabetes first.
  - Add Credit-G second.
  - Add Sick/Thyroid only if preprocessing time allows.
- Do not start with Bank Marketing:
  - It pushes the project toward benchmark engineering.
  - It increases runtime and preprocessing for little class-project benefit.

## Model Used

- Model 1:
  - Logistic regression.
  - Reason: fast, interpretable, tabular-friendly, easy to cross-validate.
  - Use `class_weight=None` for the main experiment.
  - Add `class_weight="balanced"` only as a small diagnostic if time allows.
- Model 2:
  - HistGradientBoostingClassifier.
  - Reason: stronger tabular baseline.
  - Use after logistic regression pipeline is complete.
  - Prefer scikit-learn HistGradientBoosting over XGBoost for course simplicity and dependency control.
- Avoid as required model:
  - XGBoost.
  - Reason: strong tabular model, but extra dependency and class-weight settings can distract from the cleaning method.
- Do not use deep neural networks.
- Do not implement SED, CBS, CPC, DivideMix, or Co-teaching.
- These are related work, not implementation targets.

## Model Decision

- Paper-target setup:
  - Logistic regression with standardized numeric features.
  - HistGradientBoostingClassifier.
- Class-project fallback:
  - Logistic regression only.
- Rationale:
  - Logistic regression makes the suspiciousness score easy to explain.
  - It is fast enough for repeated cross-validation scoring.
  - HistGradientBoosting tests whether the effect survives a stronger tabular learner.
  - Tree-based models are strong for tabular data, but the paper still studies cleaning policy, not model benchmarking.

## Experiments & Setup

- Task:
  - Binary tabular classification.
- Train/test split:
  - Clean held-out test set.
  - Only training labels are corrupted.
- Dataset:
  - Pima Indians Diabetes.
  - Credit-G.
  - Sick/Thyroid.
- Imbalance:
  - Use induced 85/15 imbalance for the main experiment.
  - For naturally imbalanced datasets, preserve natural imbalance if it is already near or stronger than 85/15.
  - Otherwise, subsample to 85/15.
- Noise type:
  - Class-dependent synthetic label noise.
  - Example:
    - Minority to majority flip: 30%.
    - Majority to minority flip: 10%.
  - Reason: matches the minority-harm motivation better than symmetric noise.
- Cleaning budget:
  - 10% of training set.
- Seeds:
  - 5 seeds for paper-target setup.
  - 3 seeds for class-project fallback.
- Compared methods:
  - No cleaning.
  - Random deletion.
  - Global top-loss deletion.
  - Class-proportional top-loss deletion.
  - CRCC-P with risk-adjusted scoring.
  - Oracle deletion as synthetic upper bound.
- Implementation size:
  - Paper-target: 3 datasets x 2 models x 1 noise type x 1 budget x 5 seeds x 6 methods.
  - Total: 180 final training runs plus scoring runs.
  - Class-project fallback: 1 dataset x 1 model x 1 noise type x 1 budget x 3 seeds x 4 methods.
  - Total: 12 final training runs plus scoring runs.
- Metrics:
  - Balanced accuracy.
  - Macro-F1.
  - Minority recall.
  - Noise precision among deleted samples.
  - Clean-minority deletion rate.
- Main success condition:
  - CRCC reduces clean-minority deletion rate versus global top-loss deletion on at least 2 of 3 datasets.
  - CRCC keeps balanced accuracy within 1-2 percentage points of global deletion or improves it.
- Secondary success condition:
  - CRCC improves minority recall versus global deletion.
- Failure condition:
  - If CRCC does not reduce clean-minority deletion, reframe as a negative result:
    - Class-risk-constrained cleaning did not improve harm under this detector/dataset.

## Dataset and Model Research Notes

- OpenML supports direct Python loading and metadata access through `openml.datasets.get_dataset`, including `X`, `y`, categorical indicators, and feature names.
- OpenML's own Python intro uses `credit-g` as the minimal example dataset, which makes it a safe backup for reproducible class work.
- Scikit-learn documents Breast Cancer Wisconsin as 569 samples, 30 numeric features, and 2 classes. This makes it a good local fallback, but less ideal than Pima because its classes are not strongly imbalanced without subsampling.
- OpenML dataset collection metadata lists `sick` as 3,772 rows, 30 features, and 2 classes, with a strong class imbalance. This makes it scientifically aligned but slightly less convenient than Pima.
- Grinsztajn, Oyallon, and Varoquaux show tree-based models remain very strong on typical tabular data. This supports adding HistGradientBoosting as optional robustness, but not as the first implementation target.
- Scikit-learn logistic regression is a standard probabilistic classifier, suitable for loss-based suspiciousness scoring through predicted probabilities.

## Expected Result Pattern

- No cleaning:
  - Keeps noisy labels.
  - May preserve minority samples but model learns corrupted signal.
- Random deletion:
  - Weak baseline.
  - Should not reliably improve performance.
- Global deletion:
  - May remove actual noisy labels.
  - May also remove hard clean minority samples.
- Class-proportional deletion:
  - Should reduce class-level over-deletion compared with global deletion.
  - May still delete hard minority samples if within-class loss ranking is biased.
- CRCC:
  - May remove slightly fewer noisy labels than global deletion.
  - Should delete fewer clean minority samples.
  - Should preserve minority recall better under class-dependent noise.
  - Should perform at least competitively with class-proportional deletion if risk adjustment helps.

## Scope Boundary

- This is not a benchmark paper.
- This is not a new noisy-label detector.
- This is not a deep noisy-label learning method.
- This is not a state-of-the-art claim.
- This is a compact method paper about post-detection intervention for imbalanced tabular ML.

## Novelty Boundary

- Unsafe claim:
  - "No one has done class-aware noisy-label cleaning."
- Safer claim:
  - "Prior work has studied class-aware noisy-label sample selection, mostly inside integrated deep-learning training pipelines. This project studies a model-agnostic post-detection intervention rule for imbalanced tabular ML."
- Strongest defensible gap:
  - "Given a fixed suspiciousness score and fixed deletion budget, there is limited practical guidance on how to intervene without over-deleting clean minority-class samples in tabular ML."

## Paper Positioning for Springer LNCS-Style Conference

- Viable paper angle:
  - Lightweight method for tabular ML.
  - Clear post-detection cleaning setting.
  - Compact but not tiny evaluation across 3 datasets and 2 model families.
- Main claim:
  - CRCC reduces clean-minority deletion harm compared with global top-loss cleaning while preserving comparable balanced accuracy.
- Avoid claiming:
  - State-of-the-art noisy-label learning.
  - First class-aware noisy-label method.
  - General solution for all label-noise settings.
- Reviewer-risk mitigation:
  - If reviewer says "this is stratified deletion":
    - Point to risk-adjusted ranking plus caps, not caps alone.
    - Include class-proportional deletion as an ablation.
  - If reviewer says "synthetic noise is artificial":
    - Explain synthetic noise is required to know which samples are truly clean/noisy and compute clean-minority deletion.
  - If reviewer says "only tabular":
    - State tabular ML is the intended scope, where lightweight pipelines are more practical than deep noisy-label methods.

## Required Literature

### Core Detection and Label-Noise Foundations

- Northcutt, Jiang, and Chuang, "Confident Learning: Estimating Uncertainty in Dataset Labels," JAIR, 2021.  
  https://www.jair.org/index.php/jair/article/view/12125
- Northcutt, Athalye, and Mueller, "Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks," 2021.  
  https://labelerrors.com/paper.pdf
- Song et al., "Learning from Noisy Labels with Deep Neural Networks: A Survey," 2022.  
  https://arxiv.org/abs/2007.08199
- Pickler, Kamassury, and Silva, "Benchmarking Noisy Label Detection Methods," arXiv, 2025.  
  https://arxiv.org/abs/2510.16211

### Closest Class-Aware / Imbalanced Noisy-Label Work

- Liu et al., "Learning with Imbalanced Noisy Data by Preventing Bias in Sample Selection," IEEE Transactions on Multimedia, 2024.  
  https://arxiv.org/abs/2402.11242
- Sheng et al., "Foster Adaptivity and Balance in Learning with Noisy Labels," ECCV, 2024.  
  https://arxiv.org/abs/2407.02778
- Huang et al., "Class Prototype-based Cleaner for Label Noise Learning," arXiv, 2022.  
  https://arxiv.org/abs/2212.10766
- Karim et al., "UNICON: Combating Label Noise Through Uniform Selection and Contrastive Learning," CVPR, 2022.  
  https://openaccess.thecvf.com/content/CVPR2022/html/Karim_UniCon_Combating_Label_Noise_Through_Uniform_Selection_and_Contrastive_Learning_CVPR_2022_paper.html
- Yao et al., "Jo-SRC: A Contrastive Approach for Combating Noisy Labels," CVPR, 2021.  
  https://openaccess.thecvf.com/content/CVPR2021/html/Yao_Jo-SRC_A_Contrastive_Approach_for_Combating_Noisy_Labels_CVPR_2021_paper.html

### Active Label Cleaning and Budgeted Relabeling

- Bernhardt et al., "Active Label Cleaning for Improved Dataset Quality Under Resource Constraints," Nature Communications, 2022.  
  https://www.nature.com/articles/s41467-022-28818-3
- Khanal et al., "Active Label Refinement for Robust Training of Imbalanced Medical Image Classification Tasks in the Presence of High Label Noise," MICCAI, 2024.  
  https://papers.miccai.org/miccai-2024/044-Paper3963.html
- Heidari, Zhang, and Guo, "Learning to Clean: Reinforcement Learning for Noisy Label Correction," NeurIPS, 2025.  
  https://openreview.net/forum?id=v8InI8hobW

### Robust Noisy-Label Learning Baselines for Context

- Patrini et al., "Making Deep Neural Networks Robust to Label Noise: A Loss Correction Approach," CVPR, 2017.  
  https://openaccess.thecvf.com/content_cvpr_2017/html/Patrini_Making_Deep_Neural_CVPR_2017_paper.html
- Zhang and Sabuncu, "Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels," NeurIPS, 2018.  
  https://arxiv.org/abs/1805.07836
- Jiang et al., "MentorNet: Learning Data-Driven Curriculum for Very Deep Neural Networks on Corrupted Labels," ICML, 2018.  
  https://proceedings.mlr.press/v80/jiang18c.html
- Han et al., "Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels," NeurIPS, 2018.  
  https://arxiv.org/abs/1804.06872
- Yu et al., "How Does Disagreement Help Generalization Against Label Corruption?" ICML, 2019.  
  https://proceedings.mlr.press/v97/yu19b.html
- Wei et al., "Combating Noisy Labels by Agreement: A Joint Training Method with Co-Regularization," CVPR, 2020.  
  https://openaccess.thecvf.com/content_CVPR_2020/html/Wei_Combating_Noisy_Labels_by_Agreement_A_Joint_Training_Method_With_Co-Regularization_CVPR_2020_paper.html
- Li, Socher, and Hoi, "DivideMix: Learning with Noisy Labels as Semi-Supervised Learning," ICLR, 2020.  
  https://openreview.net/forum?id=HJgExaVtwr
- Tanaka et al., "Joint Optimization Framework for Learning with Noisy Labels," CVPR, 2018.  
  https://openaccess.thecvf.com/content_cvpr_2018/html/Tanaka_Joint_Optimization_Framework_CVPR_2018_paper.html
- Yi and Wu, "PENCIL: Probabilistic End-to-End Noise Correction for Learning with Noisy Labels," CVPR, 2019.  
  https://openaccess.thecvf.com/content_CVPR_2019/html/Yi_Probabilistic_End-To-End_Noise_Correction_for_Learning_With_Noisy_Labels_CVPR_2019_paper.html

### Tabular / Classical ML Relevance

- Salekshahrezaee, Leevy, and Khoshgoftaar, "A Reconstruction Error-Based Framework for Label Noise Detection," Journal of Big Data, 2021.  
  https://journalofbigdata.springeropen.com/articles/10.1186/s40537-021-00447-5
- Ali et al., "Adaptive Label Cleaning for Error Detection on Tabular Data," 2024.  
  Use as tabular data-cleaning context if accessible through institutional/library search.
- Zhang et al., "TabMentor: Detect Errors on Tabular Data with Noisy Labels," 2023.  
  Use as tabular data-quality context if accessible through institutional/library search.

## Minimal Final Deliverable

- One notebook or script:
  - Loads tabular dataset.
  - Injects imbalance and noise.
  - Scores suspicious samples.
  - Runs no cleaning, random deletion, global deletion, class-proportional deletion, CRCC, and oracle deletion.
  - Outputs metric table and 1-2 plots.
- One final report:
  - Problem.
  - Related work.
  - Method.
  - Experiment setup.
  - Results.
  - Limitations.
- One presentation:
  - Motivation.
  - Method diagram.
  - Key result plot.
  - Takeaway.

## Implementation Status

The CRCC experiment pipeline has been fully implemented (May 2026). See `docs/experiment-report.md` for complete results. Key findings: CRCC-P reduces clean-minority deletion rate by 86-99% versus global top-loss deletion. Class-proportional caps alone dominate under extreme imbalance (lambda provides negligible additional benefit). CRCC-M was included and further reduces CMDR at comparable accuracy.

## Unresolved Questions (Resolved by Implementation)

- Which CRCC class-risk definition should be primary: binary minority risk or mean-loss class risk? **Resolved: binary minority risk was used; caps dominated over risk-adjusted scoring under 85/15 imbalance.**
- Should CRCC-M be included as an ablation, or is CRCC-P plus class-proportional deletion enough? **Resolved: CRCC-M was included; it further reduces CMDR with minor accuracy tradeoff.**
- Can Sick/Thyroid preprocessing be finished quickly enough for the 3-dataset paper-target setup? **Resolved: Yes, Sick was loaded from cached Parquet and processed successfully.**
