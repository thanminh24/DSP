"""Statistical significance tests: Wilcoxon signed-rank + Cohen's d effect size.

Key comparisons: global_top_loss vs crcc_p_l05 on CMDR and balanced accuracy.
n=5 seeds → effect sizes (Cohen's d) are primary evidence; p-values are indicative.

Outputs: outputs/statistical-tests-results.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parent.parent

COMPARISONS: list[tuple[str, str, str]] = [
    ("global_top_loss",    "crcc_p_l05", "clean_minority_deletion_rate"),
    ("global_top_loss",    "crcc_p_l05", "balanced_accuracy"),
    ("no_cleaning",        "crcc_p_l05", "balanced_accuracy"),
    ("class_proportional", "crcc_p_l05", "clean_minority_deletion_rate"),
]


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Effect size: difference of means / pooled standard deviation."""
    pooled = np.sqrt((np.std(a, ddof=1) ** 2 + np.std(b, ddof=1) ** 2) / 2)
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 0 else float("nan")


def run_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for method_a, method_b, metric in COMPARISONS:
        for dataset in df["dataset"].unique():
            for model in df["model"].unique():
                sub = df[(df["dataset"] == dataset) & (df["model"] == model)]
                sub_a = sub[sub["method"] == method_a][["seed", metric]].dropna()
                sub_b = sub[sub["method"] == method_b][["seed", metric]].dropna()
                merged = sub_a.merge(sub_b, on="seed", suffixes=("_a", "_b"))
                n = len(merged)
                if n < 3:
                    continue
                a = merged[f"{metric}_a"].values
                b = merged[f"{metric}_b"].values
                try:
                    stat, p = wilcoxon(a, b, alternative="two-sided")
                except ValueError:
                    stat, p = float("nan"), float("nan")
                rows.append({
                    "dataset": dataset, "model": model,
                    "method_a": method_a, "method_b": method_b,
                    "metric": metric, "n_seeds": n,
                    "statistic": round(stat, 4), "p_value": round(p, 4),
                    "effect_size_d": round(cohens_d(a, b), 3),
                    "significant_p05": p < 0.05,
                })
    return pd.DataFrame(rows)


def main() -> None:
    results_path = PROJECT_ROOT / "outputs" / "full-experiment-results.csv"
    if not results_path.exists():
        print(f"ERROR: {results_path} not found. Run full experiment first.")
        return

    df = pd.read_csv(results_path)
    results = run_tests(df)
    out_path = PROJECT_ROOT / "outputs" / "statistical-tests-results.csv"
    results.to_csv(out_path, index=False)
    print(f"Saved {len(results)} test rows to {out_path}")

    # Print key findings
    print("\n── Effect sizes: CMDR (global vs crcc_p) ──")
    sub = results[(results["method_a"] == "global_top_loss") &
                  (results["method_b"] == "crcc_p_l05") &
                  (results["metric"] == "clean_minority_deletion_rate")]
    if not sub.empty:
        print(sub[["dataset", "model", "effect_size_d", "p_value"]].to_string(index=False))
        print(f"\nMean |d| = {sub['effect_size_d'].abs().mean():.3f}")

    print("\n── Effect sizes: BA (global vs crcc_p) ──")
    sub_ba = results[(results["method_a"] == "global_top_loss") &
                     (results["method_b"] == "crcc_p_l05") &
                     (results["metric"] == "balanced_accuracy")]
    if not sub_ba.empty:
        print(sub_ba[["dataset", "model", "effect_size_d", "p_value"]].to_string(index=False))

    print("\n── Lambda insensitivity: CMDR (class_proportional vs crcc_p_l05) ──")
    sub_lambda = results[(results["method_a"] == "class_proportional") &
                         (results["method_b"] == "crcc_p_l05") &
                         (results["metric"] == "clean_minority_deletion_rate")]
    if not sub_lambda.empty:
        print(sub_lambda[["dataset", "model", "effect_size_d", "p_value"]].to_string(index=False))
        print(f"\nMean |d| = {sub_lambda['effect_size_d'].abs().mean():.3f} (expected near zero)")

    print("\nNote: n=5 seeds → Wilcoxon minimum p=0.0625 (two-sided). Effect size d is primary.")


if __name__ == "__main__":
    main()
