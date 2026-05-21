---
phase: 8
title: "Robustness and Statistical Validation"
status: pending
priority: P1
effort: "1d"
dependencies: [4, 5, 6, 7]
---

# Phase 8: Robustness and Statistical Validation

## Overview

Turn experiment CSVs into evidence. The previous CRCC stats used 5 seeds, producing a
Wilcoxon p-value floor of 0.0625. This phase requires enough paired observations to support
claims without leaning only on effect sizes.

## Requirements

- Functional: run paired significance tests, effect sizes, confidence intervals, and
  subgroup analyses.
- Non-functional: report practical significance separately from statistical significance.

## Architecture

```text
result CSVs
  -> validation filters
  -> paired comparisons
  -> effect sizes
  -> confidence intervals
  -> robustness tables/plots
  -> go/no-go verdict
```

## Related Code Files

- Create: `scripts/analyze_relabeling_statistics.py`
- Create: `outputs/relabeling-statistical-tests.csv`
- Create: `outputs/relabeling-viability-verdict.md`
- Create: `outputs/plots/relabeling-*`

## Implementation Steps

1. Validate result schemas and missing values.
2. Pair rows by dataset, scorer model, final model, seed, noise protocol, imbalance
   ratio, and budget.
3. Run Wilcoxon signed-rank tests for:
   - balanced OOF relabeling vs class-proportional deletion.
   - balanced OOF relabeling vs random relabel.
   - balanced OOF relabeling vs unbalanced confidence relabel.
   - balanced OOF relabeling vs cleanlab baseline.
4. Compute Cohen's d or paired standardized mean difference.
5. Bootstrap 95% confidence intervals for deltas.
6. Analyze operating conditions:
   - hidden-minority noise.
   - symmetric noise.
   - reverse asymmetric noise.
   - 0.15 vs 0.30 minority ratio.
   - small vs larger datasets.
   - LR vs tree ensembles vs gradient boosting.
   - calibrated vs uncalibrated confidence.
7. Compute calibration diagnostics:
   - relabel precision by score decile.
   - failure cases where high confidence flips are wrong.
8. Write final viability verdict:
   - GO paper.
   - GO school project only.
   - NO-GO, negative result.

## Success Criteria

- [ ] No main p-value is limited by too few pairs.
- [ ] Report includes both effect sizes and confidence intervals.
- [ ] Verdict explicitly names the method's valid operating condition.
- [ ] Failure modes are reported, especially reverse asymmetric noise.
- [ ] Results are reproducible from documented commands.
- [ ] Statistical conclusions are not based only on LR.
- [ ] At least one XGBoost or LightGBM/CatBoost comparison is included in the primary table.

## Risk Assessment

Multiple comparisons can make weak wins look stronger than they are. Mitigation: define
primary comparisons before looking at results and report secondary analyses as exploratory.

## Security Considerations

None.

## Next Steps

Proceed to Phase 9 only if verdict is GO or school-project-only GO.

## Unresolved Questions

None.
