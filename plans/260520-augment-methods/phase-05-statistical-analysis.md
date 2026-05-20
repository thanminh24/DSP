---
phase: 5
title: "Statistical Analysis — Wilcoxon, Cohen's d, Noise Scaling"
status: pending
effort: "2h"
---

# Phase 5: Statistical Analysis

## Overview

Compute final statistics from the Phase 4 sweep CSV. Produce:
- Wilcoxon signed-rank paired by seed (per dataset × model × noise_level × method-vs-baseline)
- Cohen's d per (dataset, model, noise_level) × comparison
- Main results table: mean ± std BA at medium noise (paper-ready)
- Noise-scaling table: ΔBA by noise level
- Final verdict: GO (submit) / NO-GO (publish negative) / PARTIAL (moderate evidence)

**BLOCKED BY**: Phase 4 must have completed. If both pilots failed and Phase 4
was cancelled, Phase 5 is also skipped — the negative-result summary from
Phase 3 is the final artifact.

## Context Links

- Sweep CSV: `outputs/augment-sweep-results.csv`
- Prior baseline CSV (for reference): `outputs/full-experiment-results.csv`

## Requirements

- Stats computed for every surviving augmentation method vs:
  - `class_proportional` (primary)
  - `no_cleaning` (sanity)
  - Direct control (`random_relabel` for Method A; `plain_smote` for Method B)
- Tables rendered both to CSV and to Markdown for paper inclusion.
- Final verdict explicitly stated with numbers.

## Related Code Files

**Create:**
- `scripts/analyze_augment_stats.py` (≤ 180 lines)
- `outputs/augment-statistical-tests.csv` (long-format table)
- `outputs/augment-main-table.md` (paper-ready)
- `outputs/augment-noise-scaling.md` (paper-ready)
- `outputs/augment-final-verdict.md` (decision document)

**Modify:** none.

## Implementation Steps

### Step 1 — `scripts/analyze_augment_stats.py`

```python
"""Statistical analysis for the augmentation sweep."""
from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

IN_CSV   = "outputs/augment-sweep-results.csv"
OUT_LONG = "outputs/augment-statistical-tests.csv"
OUT_MAIN = "outputs/augment-main-table.md"
OUT_NOISE = "outputs/augment-noise-scaling.md"
OUT_VERDICT = "outputs/augment-final-verdict.md"

BASELINES = ["class_proportional", "no_cleaning"]
CONTROL_MAP = {
    "balanced_oof_relabel": "random_relabel",
    "oof_filtered_smote":   "plain_smote",
}


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) < 2: return float("nan")
    s = np.sqrt(((a.std(ddof=1)**2 + b.std(ddof=1)**2) / 2))
    return float("nan") if s == 0 else (a.mean() - b.mean()) / s


def paired_wilcoxon(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 5: return float("nan"), float("nan")
    diff = a[mask] - b[mask]
    if (diff == 0).all(): return float("nan"), 1.0
    W, p = wilcoxon(diff)
    return float(W), float(p)


def main():
    df = pd.read_csv(IN_CSV)
    augments = [m for m in df["method"].unique()
                if m not in BASELINES + list(CONTROL_MAP.values())]

    # Long-form pivoted by (dataset, model, noise_level, seed) × method
    pv = df.pivot_table(index=["dataset","model","noise_level","seed"],
                        columns="method", values="balanced_accuracy").reset_index()

    rows = []
    for aug in augments:
        comparisons = BASELINES.copy()
        if aug in CONTROL_MAP and CONTROL_MAP[aug] in pv.columns:
            comparisons.append(CONTROL_MAP[aug])
        for (ds, mdl, lvl), g in pv.groupby(["dataset","model","noise_level"]):
            for base in comparisons:
                if base not in g.columns or aug not in g.columns: continue
                a_arr, b_arr = g[aug].values, g[base].values
                W, p = paired_wilcoxon(a_arr, b_arr)
                d = cohens_d(a_arr, b_arr)
                rows.append({
                    "dataset": ds, "model": mdl, "noise_level": lvl,
                    "method": aug, "baseline": base,
                    "mean_method": float(np.nanmean(a_arr)),
                    "mean_baseline": float(np.nanmean(b_arr)),
                    "delta_BA": float(np.nanmean(a_arr) - np.nanmean(b_arr)),
                    "cohens_d": d, "wilcoxon_W": W, "wilcoxon_p": p,
                    "n_pairs": int((~(np.isnan(a_arr)|np.isnan(b_arr))).sum()),
                })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_LONG, index=False)
    print(f"wrote {len(out)} comparison rows → {OUT_LONG}")

    _write_main_table(df, augments)
    _write_noise_scaling(out, augments)
    _write_verdict(out, augments)


def _write_main_table(df, augments):
    """Mean ± std BA per (dataset, model) at medium noise; rows = method."""
    med = df[df["noise_level"] == "medium"]
    mu = med.groupby(["dataset","model","method"])["balanced_accuracy"].mean().unstack("method")
    sd = med.groupby(["dataset","model","method"])["balanced_accuracy"].std().unstack("method")
    keep = BASELINES + augments + [CONTROL_MAP[a] for a in augments if a in CONTROL_MAP]
    keep = [c for c in keep if c in mu.columns]
    lines = ["# Main Results — Mean ± Std BA at medium noise (30%/10%), 20 seeds\n"]
    lines.append("| dataset | model | " + " | ".join(keep) + " |")
    lines.append("|" + "---|" * (2 + len(keep)))
    for (ds, mdl), _ in mu.iterrows():
        cells = [ds, mdl]
        for c in keep:
            m = mu.loc[(ds, mdl), c]; s = sd.loc[(ds, mdl), c]
            cells.append(f"{m:.3f} ± {s:.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    open(OUT_MAIN, "w").write("\n".join(lines) + "\n")
    print(f"wrote {OUT_MAIN}")


def _write_noise_scaling(out, augments):
    """ΔBA by noise level vs class_proportional, averaged across (dataset, model)."""
    sub = out[(out["baseline"] == "class_proportional")]
    pivot = sub.pivot_table(index="method", columns="noise_level",
                            values="delta_BA", aggfunc="mean")
    pivot = pivot.loc[[m for m in augments if m in pivot.index]]
    pivot = pivot[[c for c in ["low","medium","high"] if c in pivot.columns]]
    lines = ["# Noise scaling — mean ΔBA vs class_proportional (over datasets × models)\n"]
    lines.append("| method | low (10/5) | medium (30/10) | high (40/20) |")
    lines.append("|---|---|---|---|")
    for m, row in pivot.iterrows():
        cells = [m] + [f"{row.get(c, float('nan')):+.4f}" for c in ["low","medium","high"]]
        lines.append("| " + " | ".join(cells) + " |")
    open(OUT_NOISE, "w").write("\n".join(lines) + "\n")
    print(f"wrote {OUT_NOISE}")


def _write_verdict(out, augments):
    """Decision logic mirroring pilot gates but applied to the full sweep."""
    lines = ["# Final Verdict\n"]
    overall = {}
    for aug in augments:
        sub = out[(out["method"] == aug) & (out["baseline"] == "class_proportional")]
        wins = int((sub["delta_BA"] > 0).sum())
        total = len(sub)
        mean_d = float(sub["cohens_d"].mean())
        pooled_p = float(sub["wilcoxon_p"].median())  # conservative summary
        # control check
        ctrl = CONTROL_MAP.get(aug)
        ctrl_ok = True
        if ctrl:
            cs = out[(out["method"] == aug) & (out["baseline"] == ctrl)]
            ctrl_ok = (cs["delta_BA"].mean() > 0) and (cs["wilcoxon_p"].median() < 0.05)
        if wins >= 0.75 * total and mean_d >= 0.3 and pooled_p < 0.05 and ctrl_ok:
            verdict = "GO"
        elif wins >= 0.6 * total and mean_d >= 0.2 and ctrl_ok:
            verdict = "PARTIAL"
        else:
            verdict = "NO-GO"
        overall[aug] = verdict
        lines.append(f"## {aug}")
        lines.append(f"- wins vs class_proportional: {wins}/{total}")
        lines.append(f"- mean Cohen's d: {mean_d:.3f}")
        lines.append(f"- median Wilcoxon p: {pooled_p:.4g}")
        lines.append(f"- control check passed: {ctrl_ok}")
        lines.append(f"- **VERDICT: {verdict}**\n")
    lines.append("## Overall")
    if any(v == "GO" for v in overall.values()):
        lines.append("**Submit paper**. At least one method has strong evidence.")
    elif any(v == "PARTIAL" for v in overall.values()):
        lines.append("**Report as moderate evidence**. Frame as exploratory contribution.")
    else:
        lines.append("**Publish negative result**. No method beat class_proportional with controls.")
    open(OUT_VERDICT, "w").write("\n".join(lines) + "\n")
    print(f"wrote {OUT_VERDICT}")


if __name__ == "__main__":
    main()
```

### Step 2 — run

```bash
cd /home/than-minh/project/DSP
/home/than-minh/miniconda3/bin/python3 scripts/analyze_augment_stats.py
```

### Step 3 — interpret outputs

- `outputs/augment-statistical-tests.csv` — long-form, machine-readable
- `outputs/augment-main-table.md` — paper Table 1
- `outputs/augment-noise-scaling.md` — paper Table 2
- `outputs/augment-final-verdict.md` — decision

### Final Decision Logic

Applied per surviving augmentation method (using full-sweep data, NOT pilot):

```
GO       : wins ≥ 75% of (dataset × model × noise_level) cells vs class_proportional
           AND mean Cohen's d ≥ 0.3
           AND median Wilcoxon p < 0.05
           AND control (random_relabel / plain_smote) is also beaten with p < 0.05
PARTIAL  : wins ≥ 60% AND mean_d ≥ 0.2 AND control beaten
NO-GO    : otherwise
```

**Project outcome:**
- ≥ 1 method GO → submit paper
- only PARTIAL methods → frame as moderate evidence; consider workshop submission
- all NO-GO → publish negative result (use Phase 3 negative summary, augmented with
  full-sweep numbers)

## Todo List

- [ ] Write `scripts/analyze_augment_stats.py`
- [ ] Run analysis on Phase 4 sweep CSV
- [ ] Verify all four output files produced
- [ ] Review verdict; cross-check pilot vs full-sweep consistency
- [ ] If full sweep contradicts pilot (e.g. pilot GO but sweep NO-GO at high noise),
      document in verdict
- [ ] Submit / negative-result / moderate-evidence path chosen and documented

## Success Criteria

- [ ] All 4 output files present and non-empty
- [ ] Long-form CSV has rows for every (method × baseline × dataset × model × noise_level) combo
- [ ] Main table renders correctly in Markdown viewer
- [ ] Verdict file states GO / PARTIAL / NO-GO per method + overall decision
- [ ] Control comparisons explicitly reported (not omitted)

## Test Matrix

| Check | Pass |
|-------|------|
| Wilcoxon handles NaN | yes — `paired_wilcoxon` masks |
| Cohen's d handles zero variance | yes — returns NaN |
| Control map only adds known controls | yes |
| Verdict logic deterministic | yes — pure numeric thresholds |

## Risk Assessment

| Risk | L | I | Mitigation |
|------|---|---|----|
| Wilcoxon fails on near-zero diff | L | L | guarded: returns (NaN, 1.0) |
| Sweep CSV has unexpected method values | L | M | code derives augments dynamically from CSV |
| Verdict disagrees with pilot | M | M | full-sweep is authoritative; pilot discrepancy documented |
| n_pairs < 5 in some cells | M | L | reported but doesn't crash |

## Rollback Plan

Discard outputs; rerun. No source code modified.

## Open Questions

None. Verdict thresholds are pre-registered; all comparison sets pre-specified.
