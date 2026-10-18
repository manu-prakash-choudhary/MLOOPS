"""Download and assemble the four processed files from the UCI Heart Disease data."""

from __future__ import annotations

import argparse
import ssl
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

import certifi
import pandas as pd

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "heart-disease-data" / "heart_disease_uci.csv"
UCI_BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease"
UCI_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalch",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]
SOURCES = {
    "processed.cleveland.data": "Cleveland",
    "processed.hungarian.data": "Hungary",
    "processed.switzerland.data": "Switzerland",
    "processed.va.data": "VA Long Beach",
}
VALUE_MAPS = {
    "sex": {0: "Female", 1: "Male"},
    "cp": {
        1: "typical angina",
        2: "atypical angina",
        3: "non-anginal",
        4: "asymptomatic",
    },
    "fbs": {0: "FALSE", 1: "TRUE"},
    "restecg": {
        0: "normal",
        1: "st-t abnormality",
        2: "lv hypertrophy",
    },
    "exang": {0: "FALSE", 1: "TRUE"},
    "slope": {1: "upsloping", 2: "flat", 3: "downsloping"},
    "thal": {3: "normal", 6: "fixed defect", 7: "reversable defect"},
}


def read_source_file(filename: str, dataset_name: str) -> pd.DataFrame:
    url = f"{UCI_BASE_URL}/{filename}"
    with urlopen(
        url,
        timeout=30,
        context=ssl.create_default_context(cafile=certifi.where()),
    ) as response:
        frame = pd.read_csv(
            BytesIO(response.read()),
            header=None,
            names=UCI_COLUMNS,
            na_values=["?"],
        )
    if frame.shape[1] != len(UCI_COLUMNS):
        raise ValueError(f"Unexpected column count downloaded from {url}")
    frame.insert(2, "dataset", dataset_name)

    for column, mapping in VALUE_MAPS.items():
        raw_values = frame[column]
        mapped = raw_values.map(mapping)
        unknown = raw_values.notna() & mapped.isna()
        if unknown.any():
            unexpected = sorted(raw_values[unknown].unique().tolist())
            raise ValueError(f"Unexpected {column} codes in {filename}: {unexpected}")
        frame[column] = mapped
    return frame


def download_dataset(output_path: Path) -> None:
    frames = [
        read_source_file(filename, dataset_name)
        for filename, dataset_name in SOURCES.items()
    ]
    combined = pd.concat(frames, ignore_index=True)
    combined.insert(0, "id", range(1, len(combined) + 1))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    print(f"Downloaded {len(combined)} rows to {output_path}")
    print("Rows by source:")
    print(combined["dataset"].value_counts().to_string())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the UCI heart-disease files and combine them as CSV."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()
    download_dataset(args.output)


if __name__ == "__main__":
    main()
