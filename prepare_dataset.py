"""Clean the working copy of the heart-disease CSV for binary classification.

The raw file in heart-disease-data/ is never modified; all work happens on
working-data/heart_disease_uci_copy.csv and the output is written next to it.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "heart-disease-data" / "heart_disease_uci.csv"
WORK_DIR = ROOT / "working-data"
COPY = WORK_DIR / "heart_disease_uci_copy.csv"
CLEAN = WORK_DIR / "heart_disease_clean.csv"

# Zero is physiologically impossible; the source uses it as a missing placeholder.
ZERO_IS_MISSING = ["trestbps", "chol"]


def main() -> None:
    WORK_DIR.mkdir(exist_ok=True)
    if not COPY.exists():
        shutil.copyfile(RAW, COPY)
    df = pd.read_csv(COPY)
    print(f"Loaded copy: {df.shape[0]} rows, {df.shape[1]} columns")

    feature_cols = [c for c in df.columns if c != "id"]
    dupes = int(df.duplicated(subset=feature_cols).sum())
    df = df.drop_duplicates(subset=feature_cols).copy()
    print(f"Dropped duplicate rows (ignoring id): {dupes}")

    for col in ["fbs", "exang"]:
        df[col] = (
            df[col].astype("string").str.upper().map({"TRUE": 1, "FALSE": 0}).astype("Float64")
        )

    for col in ZERO_IS_MISSING:
        zeros = int((df[col] == 0).sum())
        df.loc[df[col] == 0, col] = float("nan")
        print(f"{col}: {zeros} zero placeholders set to missing")

    print(f"oldpeak negatives kept (clinically possible): {int((df['oldpeak'] < 0).sum())}")

    # ca is 66% missing and thal 53%, but missingness is informative; keep them and let
    # the model pipeline impute / add missing indicators.
    df["target"] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num", "id"])
    print("Target distribution:\n" + df["target"].value_counts().sort_index().to_string())

    missing = df.isna().sum()
    print("Missing values after cleaning:\n" + missing[missing > 0].to_string())

    df.to_csv(CLEAN, index=False)
    print(f"Saved cleaned data: {CLEAN} ({len(df)} rows, {df.shape[1]} columns)")


if __name__ == "__main__":
    main()
