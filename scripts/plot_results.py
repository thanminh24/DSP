from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"

MAIN_METHODS = [
    "no_cleaning", "random_deletion", "global_top_loss",
    "class_proportional", "crcc_p_l05", "oracle_deletion",
]
METHOD_LABELS = {
    "no_cleaning": "No Clean",
    "random_deletion": "Random",
    "global_top_loss": "Global Top-Loss",
    "class_proportional": "Class-Prop",
    "crcc_p_l05": "CRCC-P",
    "oracle_deletion": "Oracle",
}
LAMBDA_METHODS = ["crcc_p_l0", "crcc_p_l025", "crcc_p_l05", "crcc_p_l10"]
LAMBDA_MAP = {"crcc_p_l0": 0.0, "crcc_p_l025": 0.25, "crcc_p_l05": 0.5, "crcc_p_l10": 1.0}

METRIC_COLS = [
    "balanced_accuracy", "macro_f1", "minority_recall",
    "noise_precision_deleted", "clean_minority_deletion_rate",
    "deleted",
]

def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["dataset", "model", "method"])[METRIC_COLS].agg(["mean", "std"])
    grouped.columns = [f"{m}_{s}" for m, s in grouped.columns]
    summary = grouped.reset_index()
    summary.to_csv(OUTPUT_DIR / "summary-table.csv", index=False)
    print(f"Summary table: {len(summary)} rows → outputs/summary-table.csv")
    return summary


def _plot_metric(
    df: pd.DataFrame, metric: str, ylabel: str, filename: str, title: str,
) -> None:
    plot_df = df[df["method"].isin(MAIN_METHODS)].copy()
    plot_df["method_label"] = plot_df["method"].map(METHOD_LABELS)
    plot_df["Model"] = plot_df["model"].str.upper()

    method_order = [METHOD_LABELS[m] for m in MAIN_METHODS]

    g = sns.catplot(
        data=plot_df, kind="bar",
        x="method_label", y=metric, hue="Model",
        col="dataset", col_order=["pima", "credit-g", "sick"],
        order=method_order,
        errorbar="sd", capsize=0.1,
        height=4, aspect=1.1,
        palette="Set2",
    )
    g.set_axis_labels("", ylabel)
    for ax in g.axes.flat:
        ax.tick_params(axis="x", rotation=35)
    g.figure.suptitle(title, y=1.02)
    g.figure.tight_layout()
    path = PLOTS_DIR / filename
    g.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  Saved {path}")
    plt.close(g.figure)


def plot_cmdr(df: pd.DataFrame) -> None:
    _plot_metric(
        df, "clean_minority_deletion_rate",
        "Clean-Minority Deletion Rate",
        "fig1-clean-minority-deletion-rate.png",
        "Fig 1: Clean-Minority Deletion Rate by Method and Dataset",
    )


def plot_minority_recall(df: pd.DataFrame) -> None:
    _plot_metric(
        df, "minority_recall",
        "Minority Recall",
        "fig2-minority-recall.png",
        "Fig 2: Minority Recall by Method and Dataset",
    )


def plot_balanced_accuracy(df: pd.DataFrame) -> None:
    _plot_metric(
        df, "balanced_accuracy",
        "Balanced Accuracy",
        "fig3-balanced-accuracy.png",
        "Fig 3: Balanced Accuracy by Method and Dataset",
    )


def plot_lambda_ablation(df: pd.DataFrame) -> None:
    plot_df = df[df["method"].isin(LAMBDA_METHODS)].copy()
    plot_df["lambda"] = plot_df["method"].map(LAMBDA_MAP)

    agg = plot_df.groupby(["dataset", "model", "lambda"])["clean_minority_deletion_rate"].mean().reset_index()
    agg["Model"] = agg["model"].str.upper()

    g = sns.relplot(
        data=agg, kind="line",
        x="lambda", y="clean_minority_deletion_rate",
        hue="dataset", style="Model",
        col="dataset", col_order=["pima", "credit-g", "sick"],
        markers=True, dashes=False,
        height=3.5, aspect=1.0,
    )
    g.set_axis_labels("λ (Risk Weight)", "Clean-Minority Deletion Rate")
    for ax in g.axes.flat:
        ax.set_xticks([0.0, 0.25, 0.5, 1.0])
    g.figure.suptitle("Fig 4: Lambda Ablation — Effect of λ on Clean-Minority Harm", y=1.03)
    g.figure.tight_layout()
    path = PLOTS_DIR / "fig4-lambda-ablation.png"
    g.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  Saved {path}")
    plt.close(g.figure)


def print_key_findings(df: pd.DataFrame) -> None:
    print("\nKey Findings — CMDR Reduction (crcc_p_l05 vs global_top_loss):")
    for ds in ["pima", "credit-g", "sick"]:
        for mdl in ["lr", "hgb"]:
            sub = df[(df["dataset"] == ds) & (df["model"] == mdl)]
            gtl = sub[sub["method"] == "global_top_loss"]["clean_minority_deletion_rate"].mean()
            crcc = sub[sub["method"] == "crcc_p_l05"]["clean_minority_deletion_rate"].mean()
            reduction = (gtl - crcc) / gtl * 100 if gtl > 0 else 0
            print(f"  {ds:10s} {mdl:4s}: {gtl:.4f} → {crcc:.4f}  ({reduction:.0f}% reduction)")


def main() -> None:
    sns.set_theme(style="whitegrid", font_scale=1.1)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(OUTPUT_DIR / "full-experiment-results.csv")

    compute_summary(df)
    plot_cmdr(df)
    plot_minority_recall(df)
    plot_balanced_accuracy(df)
    plot_lambda_ablation(df)
    print_key_findings(df)
    print("\nplot_results done.")


if __name__ == "__main__":
    main()
