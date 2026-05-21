"""Analyze relabeling viability and model-stress result CSVs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

IN_CSVS = [
    Path("outputs/relabeling-viability-results.csv"),
    Path("outputs/model-stress-results.csv"),
    Path("outputs/augment-sweep-results.csv"),
]
OUT_CSV = Path("outputs/relabeling-statistical-tests.csv")
OUT_MD = Path("outputs/relabeling-viability-verdict.md")
PRIMARY = "balanced_oof_relabel"
BASELINES = [
    "class_proportional",
    "random_relabel",
    "unbalanced_oof_relabel",
    "global_top_loss",
    "no_cleaning",
]


def main() -> None:
    df = _load_results()
    rows = _paired_tests(df)
    out = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    _write_verdict(df, out)
    print(f"wrote {OUT_CSV}")
    print(f"wrote {OUT_MD}")


def _load_results() -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in IN_CSVS if path.exists()]
    if not frames:
        OUT_CSV.parent.mkdir(exist_ok=True)
        pd.DataFrame().to_csv(OUT_CSV, index=False)
        OUT_MD.write_text(
            "# Relabeling Viability Verdict\n\n"
            "## Evidence Summary\n\n"
            "No relabeling result CSVs found. Run a sweep first.\n\n"
            "## Unresolved Questions\n\n"
            "- Generate `outputs/relabeling-viability-results.csv` or "
            "`outputs/model-stress-results.csv`.\n",
            encoding="utf-8",
        )
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _paired_tests(df: pd.DataFrame) -> list[dict]:
    if df.empty or "method" not in df.columns:
        return []
    _fail_on_error_rows(df)
    index_cols = [
        c for c in [
            "dataset", "model", "scorer_model", "final_model", "noise_protocol",
            "seed", "mn_to_maj", "maj_to_min",
        ] if c in df.columns
    ]
    metric_cols = ["balanced_accuracy", "minority_recall", "macro_f1", "relabel_correctness"]
    rows = []
    for metric in metric_cols:
        if metric not in df.columns:
            continue
        pv = df.pivot_table(index=index_cols, columns="method", values=metric).reset_index()
        for baseline in BASELINES:
            if PRIMARY not in pv.columns or baseline not in pv.columns:
                continue
            a = pv[PRIMARY].to_numpy(float)
            b = pv[baseline].to_numpy(float)
            rows.append(_comparison_row(metric, baseline, a, b))
    return rows


def _fail_on_error_rows(df: pd.DataFrame) -> None:
    if "method" in df.columns and (df["method"] == "ERROR").any():
        errors = df[df["method"] == "ERROR"].head(10)
        raise RuntimeError(f"Experiment CSV contains failed combinations:\n{errors}")


def _comparison_row(metric: str, baseline: str, a: np.ndarray, b: np.ndarray) -> dict:
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    diff = a - b
    if len(diff) >= 5 and not np.all(diff == 0):
        stat, p_value = wilcoxon(diff)
    else:
        stat, p_value = float("nan"), float("nan")
    return {
        "method": PRIMARY,
        "baseline": baseline,
        "metric": metric,
        "n_pairs": int(len(diff)),
        "mean_method": float(np.mean(a)) if len(a) else float("nan"),
        "mean_baseline": float(np.mean(b)) if len(b) else float("nan"),
        "mean_delta": float(np.mean(diff)) if len(diff) else float("nan"),
        "ci95_low": _bootstrap_ci(diff)[0],
        "ci95_high": _bootstrap_ci(diff)[1],
        "cohens_d": _cohens_d(diff),
        "wilcoxon_stat": float(stat),
        "wilcoxon_p": float(p_value),
    }


def _cohens_d(diff: np.ndarray) -> float:
    if len(diff) < 2 or np.std(diff, ddof=1) == 0:
        return float("nan")
    return float(np.mean(diff) / np.std(diff, ddof=1))


def _bootstrap_ci(diff: np.ndarray, n_boot: int = 2000) -> tuple[float, float]:
    if len(diff) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(42)
    means = [rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _write_verdict(df: pd.DataFrame, tests: pd.DataFrame) -> None:
    lines = ["# Relabeling Viability Verdict\n"]

    lines.append("## Evidence Summary\n")
    if tests.empty:
        lines.append("- No paired tests available.")
    else:
        primary = tests[tests["metric"] == "balanced_accuracy"]
        for _, row in primary.iterrows():
            sig = "(*)" if (not np.isnan(row["wilcoxon_p"]) and row["wilcoxon_p"] < 0.05) else ""
            lines.append(
                f"- `{PRIMARY}` vs `{row['baseline']}`: "
                f"delta={row['mean_delta']:+.4f}, p={row['wilcoxon_p']:.4g}, "
                f"n={int(row['n_pairs'])}{sig}"
            )

    models = sorted(df["model"].dropna().unique()) if "model" in df else []
    lines.append("\n## Model Coverage\n")
    lines.append(", ".join(models) if models else "No model column found.")

    lines.append("\n## Per-Model Win Rate vs class_proportional (balanced_accuracy)\n")
    if "model" in df.columns and "method" in df.columns and not df.empty:
        index_cols = [c for c in ["dataset", "model", "seed", "noise_protocol", "noise_level"] if c in df.columns]
        pv = df.pivot_table(index=index_cols, columns="method", values="balanced_accuracy").reset_index()
        for m in models:
            sub = pv[pv["model"] == m] if "model" in pv.columns else pv
            col = PRIMARY
            base = "class_proportional"
            if col not in sub.columns or base not in sub.columns:
                continue
            diff = (sub[col] - sub[base]).dropna().values
            if len(diff) == 0:
                continue
            wins = (diff > 0).sum()
            try:
                _, p = wilcoxon(diff)
                pstr = f"p={p:.3e}"
            except Exception:
                pstr = "n/a"
            lines.append(f"- {m}: {wins}/{len(diff)}={wins/len(diff):.0%} mean_delta={diff.mean():+.4f} {pstr}")

    # Overall verdict logic
    sig_wins = 0
    if not tests.empty:
        ba_tests = tests[tests["metric"] == "balanced_accuracy"]
        sig_wins = ((ba_tests["wilcoxon_p"] < 0.05) & (ba_tests["mean_delta"] > 0)).sum()
    verdict_line = "**VIABLE** — significant wins on majority of baseline comparisons." if sig_wins >= 3 \
        else ("**MARGINAL** — fewer than 3 significant baseline wins." if sig_wins >= 1 \
        else "**INSUFFICIENT DATA** — run more seeds/models before claiming viability.")
    lines.append(f"\n## Verdict\n\n{verdict_line}")

    lines.append("\n## Unresolved Questions\n")
    lines.append("- Confirm weak-supervision transfer after Phase 5 data is available.")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
