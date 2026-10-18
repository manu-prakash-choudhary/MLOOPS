from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "heart-disease-data" / "heart_disease_uci.csv"
CLEAN = ROOT / "working-data" / "heart_disease_clean.csv"
MODEL = ROOT / "working-data" / "final_model_no_source.joblib"


def test_source_dataset_schema_and_target_values():
    df = pd.read_csv(RAW)
    assert len(df) == 920
    assert {"id", "dataset", "num", "age", "chol"}.issubset(df.columns)
    assert set(df["num"].dropna().unique()) == {0, 1, 2, 3, 4}


def test_cleaned_dataset_invariants():
    df = pd.read_csv(CLEAN)
    assert len(df) == 918
    assert "id" not in df.columns
    assert "num" not in df.columns
    assert set(df["target"].unique()) == {0, 1}
    assert df["target"].value_counts().to_dict() == {1: 508, 0: 410}
    assert not df.duplicated().any()
    assert not (df["trestbps"] == 0).any()
    assert not (df["chol"] == 0).any()
    for column in ("fbs", "exang"):
        assert set(df[column].dropna().unique()).issubset({0, 1})


def test_saved_model_predicts_finite_probabilities():
    artifact = joblib.load(MODEL)
    model = artifact["model"]
    threshold = artifact["threshold"]
    assert 0 <= threshold <= 1

    row = pd.DataFrame(
        [
            {
                "age": 62,
                "sex": "Female",
                "cp": "asymptomatic",
                "trestbps": 140,
                "chol": 268,
                "fbs": 0,
                "restecg": "lv hypertrophy",
                "thalch": 160,
                "exang": 0,
                "oldpeak": 3.6,
                "slope": "downsloping",
                "ca": 2,
                "thal": "normal",
                "dataset": "Cleveland",
            }
        ]
    )
    probabilities = model.predict_proba(row)
    assert probabilities.shape == (1, 2)
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1)
