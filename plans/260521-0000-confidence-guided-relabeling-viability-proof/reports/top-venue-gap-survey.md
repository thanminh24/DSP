# Top-Venue Gap Survey

## Summary

No exact 2024+ A*/Q1 paper was found that matches all target axes together:

- tabular data
- imbalanced labels
- noisy labels
- lightweight post-detection relabeling via class-balanced out-of-fold confidence

The nearest practical blocker is cleanlab/Confident Learning. It already handles
out-of-fold probabilities, per-class self-confidence, and tabular label-issue detection,
but it does not provide the proposed minority-aware majority-pool relabeling policy.

## Closest Works

| Rank | Work | Venue / year | Fit | Distinction |
|---|---|---:|---|---|
| 1 | Cleanlab / Confident Learning | tool + 2021 paper | Blocker | OOF label issue detection exists; missing minority-aware relabel policy. |
| 2 | Mislabeled examples detection survey/benchmark | TMLR 2024 | Context | Tabular-oriented detection benchmark; not relabeling. |
| 3 | Confidence-Based Sieving Strategy | TMLR 2023 | Partial | Per-class confidence sieving; not tabular-specific OOF relabeling. |
| 4 | Learning with Imbalanced Noisy Data by Preventing Bias in Sample Selection | IEEE TMM 2024 | Partial | Imbalance-aware noisy sample selection; heavier train-time method. |
| 5 | Data-centric insights improve pseudo-labeling | DMLR @ ICLR 2024 | Partial | Data-centric tabular pseudo-labeling; not label repair. |
| 6 | ULAREF | ICML 2024 Spotlight | Partial | Label refinement; not hidden-minority tabular OOF confidence. |
| 7 | Collaborative Refining for Inaccurate Labels | NeurIPS 2024 | Partial | Multiple annotators; not single noisy-label tabular repair. |
| 8 | Pi-DUAL | ICML 2024 | Partial | Requires privileged information; not lightweight tabular deployment. |
| 9 | Validation Sets for Imbalanced Noisy-label Meta-learning | TMLR 2025 | Partial | Validation-set construction; not relabeling intervention. |
| 10 | Data Glitches Discovery | KDD 2025 | Context | Tabular repair but influence-based and heavier. |

## Novelty Verdict

**NARROW BUT DEFENSIBLE**

The paper claim must be narrow:

> tabular minority-aware relabeling using class-balanced out-of-fold confidence.

Do not claim first class-aware noisy-label method or first confidence-based label repair.

## Unresolved Questions

- Verify final publisher metadata for 2025 OpenReview-listed works before submission.
