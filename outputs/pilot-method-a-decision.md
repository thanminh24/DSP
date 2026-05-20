# Method A Pilot Decision: GO

## Results (4 datasets × LR × 20 seeds, medium noise 30%/10%)

| dataset | class_prop | balanced_oof_relabel | random_relabel | delta_vs_cp | delta_vs_random |
|---------|-----------|---------------------|----------------|-------------|-----------------|
| credit-g | 0.610 | 0.632 | 0.553 | **+0.021** | **+0.079** |
| phoneme | 0.645 | 0.723 | 0.518 | **+0.078** | **+0.205** |
| pima | 0.670 | 0.703 | 0.569 | **+0.033** | **+0.133** |
| yeast | 0.740 | 0.761 | 0.603 | **+0.021** | **+0.158** |

## Statistical Tests

- Wilcoxon signed-rank vs class_proportional: W=85.5, p=1.8e-13
- Mean Cohen's d: 1.914 (per-dataset: credit-g=0.564, phoneme=5.375, pima=0.881, yeast=0.835)
- Control (vs random_relabel): p=7.8e-15, delta=+0.144 → **PASS**
- relabel_correctness: balanced_oof=19.1%, random=5.5% → OOF signal provides ~3.5× precision over random

## Verdict: GO

Balanced-OOF relabeling beats class_proportional on all 4 datasets with large effect sizes.
The control comparison confirms the gain is real (OOF signal) not an oversampling artifact.
Proceed to Phase 4 full sweep with Method A.

## Limitation

HGB model excluded from pilot due to pathological runtime (250s/fit). Full sweep will attempt
HGB if time permits; otherwise report LR-only as primary evidence.
