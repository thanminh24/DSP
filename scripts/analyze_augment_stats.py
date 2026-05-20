"""Statistical analysis for the augmentation sweep."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

IN_CSV = "outputs/augment-sweep-results.csv"
OUT_LONG = "outputs/augment-statistical-tests.csv"
OUT_MAIN = "outputs/augment-main-table.md"
OUT_NOISE = "outputs/augment-noise-scaling.md"
OUT_VERDICT = "outputs/augment-final-verdict.md"

BASELINES = ["class_proportional", "no_cleaning"]
CONTROL_MAP = {
    "balanced_oof_relabel": "random_relabel",
    "oof_filtered_smote": "plain_smote",
}


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) < 2:
        return float("nan")
    s = np.sqrt(((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2))
    return float("nan") if s == 0 else (a.mean() - b.mean()) / s


def paired_wilcoxon(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 5:
        return float("nan"), float("nan")
    diff = a[mask] - b[mask]
    if (diff == 0).all():
        return float("nan"), 1.0
    W, p = wilcoxon(diff)
    return float(W), float(p)


def main():
    df = pd.read_csv(IN_CSV)
    augments = [
        m for m in df["method"].unique()
        if m not in BASELINES + list(CONTROL_MAP.values())
    ]

    pv = df.pivot_table(
        index=["dataset", "model", "noise_level", "seed"],
        columns="method", values="balanced_accuracy",
    ).reset_index()

    rows = []
    for aug in augments:
        comparisons = BASELINES.copy()
        if aug in CONTROL_MAP and CONTROL_MAP[aug] in pv.columns:
            comparisons.append(CONTROL_MAP[aug])
        for (ds, mdl, lvl), g in pv.groupby(["dataset", "model", "noise_level"]):
            for base in comparisons:
                if base not in g.columns or aug not in g.columns:
                    continue
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
                    "n_pairs": int((~(np.isnan(a_arr) | np.isnan(b_arr))).sum()),
                })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_LONG, index=False)
    print(f"wrote {len(out)} comparison rows -> {OUT_LONG}")

    _write_main_table(df, augments)
    _write_noise_scaling(out, augments)
    _write_verdict(out, augments)


def _write_main_table(df, augments):
    med = df[df["noise_level"] == "medium"]
    mu = med.groupby(["dataset", "model", "method"])["balanced_accuracy"].mean().unstack("method")
    sd = med.groupby(["dataset", "model", "method"])["balanced_accuracy"].std().unstack("method")
    keep = BASELINES + augments + [
        CONTROL_MAP[a] for a in augments if a in CONTROL_MAP
    ]
    keep = [c for c in keep if c in mu.columns]
    lines = ["# Main Results --- Mean +/- Std BA at medium noise (30%/10%), 20 seeds\n"]
    lines.append("| dataset | model | " + " | ".join(keep) + " |")
    lines.append("|" + "---|" * (2 + len(keep)))
    for (ds, mdl), _ in mu.iterrows():
        cells = [ds, mdl]
        for c in keep:
            m = mu.loc[(ds, mdl), c]
            s = sd.loc[(ds, mdl), c]
            cells.append(f"{m:.3f} +/- {s:.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    with open(OUT_MAIN, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {OUT_MAIN}")


def _write_noise_scaling(out, augments):
    sub = out[out["baseline"] == "class_proportional"]
    pivot = sub.pivot_table(
        index="method", columns="noise_level",
        values="delta_BA", aggfunc="mean",
    )
    pivot = pivot.loc[[m for m in augments if m in pivot.index]]
    pivot = pivot[[c for c in ["low", "medium", "high"] if c in pivot.columns]]
    lines = ["# Noise scaling --- mean delta BA vs class_proportional (over datasets x models)\n"]
    lines.append("| method | low (10/5) | medium (30/10) | high (40/20) |")
    lines.append("|---|---|---|---|")
    for m, row in pivot.iterrows():
        cells = [m] + [f"{row.get(c, float('nan')):+.4f}" for c in ["low", "medium", "high"]]
        lines.append("| " + " | ".join(cells) + " |")
    with open(OUT_NOISE, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {OUT_NOISE}")


def _write_verdict(out, augments):
    lines = ["# Final Verdict\n"]
    overall = {}
    for aug in augments:
        sub = out[(out["method"] == aug) & (out["baseline"] == "class_proportional")]
        total = len(sub)
        wins = int((sub["delta_BA"] > 0).sum())
        mean_d = float(sub["cohens_d"].mean())
        pooled_p = float(sub["wilcoxon_p"].median())
        ctrl = CONTROL_MAP.get(aug)
        ctrl_ok = True
        if ctrl:
            cs = out[(out["method"] == aug) & (out["baseline"] == ctrl)]
            ctrl_delta = float(cs["delta_BA"].mean()) if len(cs) > 0 else 0.0
            ctrl_p = float(cs["wilcoxon_p"].median()) if len(cs) > 0 else 1.0
            ctrl_ok = ctrl_delta > 0 and ctrl_p < 0.05
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
        lines.append(
            "**Publish negative result**. No method beat class_proportional with controls."
        )
    with open(OUT_VERDICT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {OUT_VERDICT}")


if __name__ == "__main__":
    main()
