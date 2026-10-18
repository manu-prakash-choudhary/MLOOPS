"""Generate exploratory plots from the cleaned heart-disease dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_CSV = ROOT / "working-data" / "heart_disease_clean.csv"
DEFAULT_OUTPUT = ROOT / "artifacts" / "eda"


def save_class_balance(df: pd.DataFrame, output_dir: Path) -> None:
    counts = df["target"].value_counts().reindex([0, 1], fill_value=0)
    labels = ["No disease (0)", "Disease (1)"]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, counts.values, color=["#4C78A8", "#E45756"])
    ax.set_title("Binary Target Class Balance")
    ax.set_ylabel("Records")
    ax.bar_label(
        bars,
        labels=[
            f"{count:,} ({count / len(df):.1%})" for count in counts.values
        ],
        padding=4,
    )
    ax.set_ylim(0, max(counts.values) * 1.18)
    fig.tight_layout()
    fig.savefig(output_dir / "01_class_balance.png", dpi=160)
    plt.close(fig)


def save_numeric_distributions(df: pd.DataFrame, output_dir: Path) -> None:
    numeric = [
        col for col in df.select_dtypes(include="number").columns
        if col != "target"
    ]
    if not numeric:
        return
    columns = 3
    rows = (len(numeric) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(14, 4 * rows))
    flat_axes = axes.ravel() if hasattr(axes, "ravel") else [axes]
    for ax, column in zip(flat_axes, numeric):
        ax.hist(df[column].dropna(), bins=20, color="#4C78A8", edgecolor="white")
        ax.set_title(column)
        ax.set_ylabel("Records")
        ax.grid(axis="y", alpha=0.2)
    for ax in flat_axes[len(numeric):]:
        ax.set_visible(False)
    fig.suptitle("Numeric Feature Distributions", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_dir / "02_numeric_distributions.png", dpi=160)
    plt.close(fig)


def save_correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return
    corr = numeric.corr()
    width = max(9, corr.shape[1] * 0.8)
    fig, ax = plt.subplots(figsize=(width, width * 0.78))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)), corr.index)
    ax.set_title("Numeric Feature and Target Correlations")
    fig.colorbar(image, ax=ax, label="Pearson correlation")
    fig.tight_layout()
    fig.savefig(output_dir / "03_numeric_correlations.png", dpi=160)
    plt.close(fig)


def save_missingness(df: pd.DataFrame, output_dir: Path) -> None:
    missing = df.isna().mean().mul(100).sort_values()
    missing = missing[missing > 0]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.38 * len(missing))))
    if missing.empty:
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center")
        ax.set_axis_off()
    else:
        bars = ax.barh(missing.index, missing.values, color="#F2A541")
        ax.bar_label(bars, fmt="%.1f%%", padding=3)
        ax.set_xlim(0, missing.max() * 1.2)
        ax.set_xlabel("Missing values (%)")
        ax.set_title("Missingness by Feature (After Cleaning)")
        ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "04_missingness.png", dpi=160)
    plt.close(fig)


def save_target_by_source(df: pd.DataFrame, output_dir: Path) -> None:
    if "dataset" not in df:
        return
    grouped = df.groupby("dataset", observed=True)["target"].agg(["mean", "count"])
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(grouped.index, grouped["mean"] * 100, color="#72B7B2")
    ax.bar_label(
        bars,
        labels=[
            f"{rate:.1f}% (n={count})"
            for rate, count in zip(grouped["mean"] * 100, grouped["count"])
        ],
        padding=4,
    )
    ax.set_title("Positive Target Rate by Data Source")
    ax.set_ylabel("Records with disease target (%)")
    ax.set_ylim(0, 118)
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "05_target_by_source.png", dpi=160)
    plt.close(fig)


def create_eda(csv_path: Path, output_dir: Path) -> None:
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {csv_path}. "
            "Run `python prepare_dataset.py` first."
        )
    df = pd.read_csv(csv_path)
    if "target" not in df.columns:
        raise ValueError(f"{csv_path} must contain the binary `target` column.")
    if df.empty:
        raise ValueError(f"{csv_path} contains no records.")

    output_dir.mkdir(parents=True, exist_ok=True)
    save_class_balance(df, output_dir)
    save_numeric_distributions(df, output_dir)
    save_correlation_heatmap(df, output_dir)
    save_missingness(df, output_dir)
    save_target_by_source(df, output_dir)

    print(f"EDA dataset: {csv_path}")
    print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Target counts:\n{df['target'].value_counts().sort_index().to_string()}")
    print(f"Plots saved to: {output_dir}")
    for chart in sorted(output_dir.glob("*.png")):
        print(f"  {chart.name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create EDA charts from the cleaned heart-disease CSV."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Cleaned CSV path (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output directory for charts (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()
    create_eda(args.csv, args.output)


if __name__ == "__main__":
    main()
