# Method B Pilot Decision: FAIL

## Results (4 datasets × LR × 20 seeds, medium noise 30%/10%)

| dataset | class_prop | oof_filtered_smote | plain_smote | delta_vs_cp | delta_vs_plain |
|---------|-----------|-------------------|-------------|-------------|----------------|
| credit-g | 0.610 | 0.653 | 0.653 | +0.043 | +0.001 |
| phoneme | 0.645 | 0.750 | 0.750 | +0.105 | 0.000 |
| pima | 0.670 | 0.700 | 0.697 | +0.030 | +0.003 |
| yeast | 0.740 | 0.742 | 0.740 | +0.002 | +0.002 |

## Statistical Tests

- Wilcoxon signed-rank vs class_proportional: W=308.5, p=3.2e-10
- Mean Cohen's d: 2.827
- Control (vs plain_smote): p=0.198, delta=+0.002 → **FAIL**
- OOF-filtered SMOTE ≈ plain SMOTE in all datasets

## Verdict: FAIL

While SMOTE (both filtered and plain) significantly outperforms class_proportional, the
OOF filtering step adds zero practical value (mean delta +0.0016, p=0.20). The gain is
from SMOTE itself, not from Type B noise filtering.

Method B is EXCLUDED from Phase 4. Only Method A proceeds to full sweep.
