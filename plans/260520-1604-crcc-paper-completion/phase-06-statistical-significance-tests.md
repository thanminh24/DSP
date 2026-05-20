---
phase: 6
title: "Statistical Significance Tests"
status: pending
priority: P1
effort: "1h"
dependencies: [4, 5]
---

# Phase 6: Statistical Significance Tests

## Overview

Run Wilcoxon signed-rank tests on 5-seed distributions for key pairwise comparisons. Required for any conference paper making quantitative performance claims. Also computes Cohen's d effect size (more credible than p-value at n=5).

Key comparisons (per dataset/model):
1. `global_top_loss` vs `crcc_p_l05` on CMDR — primary claim
2. `global_top_loss` vs `crcc_p_l05` on `balanced_accuracy` — secondary claim
3. `no_cleaning` vs `crcc_p_l05` on `balanced_accuracy` — cleaning benefit
4. `class_proportional` vs `crcc_p_l05` on CMDR — lambda contribution (expected flat)

## Related Code Files

- Create: `scripts/run_statistical_tests.py` (≤ 150 lines)
- Read: `outputs/full-experiment-results.csv`
- Write: `outputs/statistical-tests-results.csv`

## Implementation Steps

1. **Create `scripts/run_statistical_tests.py`**:
   ```python
   import sys
   from pathlib import Path
   import pandas as pd
   import numpy as np
   from scipy.stats import wilcoxon
   
   PROJECT_ROOT = Path(__file__).resolve().parent.parent
   
   COMPARISONS = [
       ("global_top_loss",    "crcc_p_l05", "clean_minority_deletion_rate"),
       ("global_top_loss",    "crcc_p_l05", "balanced_accuracy"),
       ("no_cleaning",        "crcc_p_l05", "balanced_accuracy"),
       ("class_proportional", "crcc_p_l05", "clean_minority_deletion_rate"),
   ]
   
   def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
       """Effect size: difference of means / pooled std."""
       pooled = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
       return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 0 else float("nan")
   
   def run_tests(df: pd.DataFrame) -> pd.DataFrame:
       rows = []
       for method_a, method_b, metric in COMPARISONS:
           for dataset in df["dataset"].unique():
               for model in df["model"].unique():
                   sub = df[(df["dataset"]==dataset) & (df["model"]==model)]
                   a = sub[sub["method"]==method_a][metric].dropna().values
                   b = sub[sub["method"]==method_b][metric].dropna().values
                   n = min(len(a), len(b))
                   if n < 3:
                       continue
                   try:
                       stat, p = wilcoxon(a[:n], b[:n], alternative="two-sided")
                   except ValueError:
                       stat, p = float("nan"), float("nan")
                   rows.append({
                       "dataset": dataset, "model": model,
                       "method_a": method_a, "method_b": method_b,
                       "metric": metric, "n_seeds": n,
                       "statistic": round(stat, 4), "p_value": round(p, 4),
                       "effect_size_d": round(cohens_d(a[:n], b[:n]), 3),
                       "significant_p05": p < 0.05,
                   })
       return pd.DataFrame(rows)
   
   def main():
       df = pd.read_csv(PROJECT_ROOT / "outputs" / "full-experiment-results.csv")
       results = run_tests(df)
       out = PROJECT_ROOT / "outputs" / "statistical-tests-results.csv"
       results.to_csv(out, index=False)
       print(f"Saved {len(results)} test rows to {out}")
       # Print summary
       print("\n── Effect sizes (CMDR: global vs crcc_p) ──")
       sub = results[(results["method_a"]=="global_top_loss") &
                     (results["method_b"]=="crcc_p_l05") &
                     (results["metric"]=="clean_minority_deletion_rate")]
       print(sub[["dataset","model","effect_size_d","p_value"]].to_string(index=False))
       print("\nNote: n=5 seeds. Wilcoxon minimum p=0.0625 (two-sided). Effect size d is primary evidence.")
   
   if __name__ == "__main__":
       main()
   ```

2. **Run**:
   ```bash
   python3 scripts/run_statistical_tests.py
   ```

3. **Verify**: effect size |d| > 1.0 for CMDR comparison on all datasets (expected: very large, CRCC-P achieves near-zero CMDR vs global's 30–70%).

4. **Handle zero-CMDR case**: if all 5 seeds show crcc_p CMDR = 0.0 exactly, Wilcoxon raises ValueError (zero differences). The try/except handles this; add annotation in output: "CRCC-P achieves zero CMDR in all seeds — Wilcoxon not applicable, effect size is infinite."

## Success Criteria

- [ ] `scripts/run_statistical_tests.py` exists, ≤ 150 lines
- [ ] `outputs/statistical-tests-results.csv` exists with `effect_size_d` and `p_value` columns
- [ ] All CMDR comparisons (global vs crcc_p) show |effect_size_d| > 1.0
- [ ] Script handles NaN rows and zero-variance distributions without crashing

## Risk Assessment

- n=5 is too small for reliable Wilcoxon significance. Effect sizes (Cohen's d) are more important. Frame in paper: "given n=5 seeds, we report effect sizes as primary evidence; p-values are indicative."
- `class_proportional` vs `crcc_p_l05` on CMDR: expected to show d ≈ 0 (lambda insensitivity confirmed finding). This is the validation of the key negative finding.
