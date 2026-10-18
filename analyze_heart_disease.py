"""Print a descriptive summary of the UCI heart-disease CSV dataset."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from statistics import mean, median

MISSING_VALUES = {"", "na", "n/a", "nan", "null", "?"}


def is_missing(value: str) -> bool:
    return value.strip().lower() in MISSING_VALUES


def parse_number(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def format_number(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def summarize(path: Path) -> None:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames
        if not columns:
            raise ValueError(f"{path} does not contain a CSV header.")
        rows = list(reader)

    if not rows:
        raise ValueError(f"{path} does not contain any data rows.")

    print(f"Dataset: {path}")
    print(f"Rows: {len(rows):,} | Columns: {len(columns)}")

    print("\nColumn overview")
    print(f"{'Column':<12} {'Type':<12} {'Missing':>10} {'Unique':>10}")
    print("-" * 48)

    numeric_values: dict[str, list[float]] = {}
    categorical_values: dict[str, list[str]] = {}

    for column in columns:
        values = [row.get(column, "") or "" for row in rows]
        present = [value.strip() for value in values if not is_missing(value)]
        parsed = [parse_number(value) for value in present]
        is_numeric = bool(present) and all(value is not None for value in parsed)
        missing_count = len(values) - len(present)

        if is_numeric:
            numeric_values[column] = [value for value in parsed if value is not None]
            kind = "numeric"
            unique_count = len(set(numeric_values[column]))
        else:
            categorical_values[column] = present
            kind = "categorical"
            unique_count = len(set(present))

        print(
            f"{column:<12} {kind:<12} "
            f"{missing_count:>5} ({missing_count / len(rows):>5.1%}) "
            f"{unique_count:>10,}"
        )

    if numeric_values:
        print("\nNumeric summary")
        print(f"{'Column':<12} {'Mean':>10} {'Median':>10} {'Min':>10} {'Max':>10}")
        print("-" * 54)
        for column, values in numeric_values.items():
            if values:
                print(
                    f"{column:<12} {format_number(mean(values)):>10} "
                    f"{format_number(median(values)):>10} "
                    f"{format_number(min(values)):>10} "
                    f"{format_number(max(values)):>10}"
                )

    if categorical_values:
        print("\nCategorical values (up to 5 most common)")
        for column, values in categorical_values.items():
            if values:
                top_values = Counter(values).most_common(5)
                formatted = ", ".join(
                    f"{value}: {count:,}" for value, count in top_values
                )
                print(f"{column}: {formatted}")

    if "num" in columns:
        labels = [
            row["num"].strip()
            for row in rows
            if not is_missing(row.get("num", ""))
        ]
        distribution = Counter(labels)
        print("\nHeart-disease label (num)")
        for label, count in sorted(
            distribution.items(),
            key=lambda item: (
                parse_number(item[0]) is None,
                parse_number(item[0]) or 0,
                item[0],
            ),
        ):
            print(f"  {label}: {count:,} ({count / len(labels):.1%})")
        if labels:
            positive_count = sum(
                (parse_number(label) or 0) > 0 for label in labels
            )
            print(
                f"  Any disease (num > 0): {positive_count:,} "
                f"({positive_count / len(labels):.1%})"
            )


def main() -> None:
    default_path = (
        Path(__file__).resolve().parent
        / "heart-disease-data"
        / "heart_disease_uci.csv"
    )
    parser = argparse.ArgumentParser(
        description="Show a descriptive summary of the heart-disease CSV."
    )
    parser.add_argument(
        "csv_path",
        nargs="?",
        type=Path,
        default=default_path,
        help=f"CSV file to summarize (default: {default_path})",
    )
    args = parser.parse_args()
    summarize(args.csv_path)


if __name__ == "__main__":
    main()
