"""Train and compare baseline binary classifiers on the cleaned heart-disease data."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
CLEAN = ROOT / "working-data" / "heart_disease_clean.csv"
MODEL_PATH = ROOT / "working-data" / "best_model.joblib"
SEED = 42


def build_pipeline(model, scale: bool, use_source: bool = True) -> Pipeline:
    numeric = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca", "fbs", "exang"]
    categorical = ["sex", "cp", "restecg", "slope", "thal"]
    if use_source:
        categorical.insert(1, "dataset")
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    pre = ColumnTransformer(
        [
            ("num", Pipeline(num_steps), numeric),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            ),
        ]
    )
    return Pipeline([("pre", pre), ("model", model)])


def main() -> None:
    df = pd.read_csv(CLEAN)
    X, y = df.drop(columns="target"), df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)} | Positive rate: {y.mean():.1%}")

    candidates = {
        "baseline (majority)": build_pipeline(DummyClassifier(strategy="most_frequent"), True),
        "logistic regression": build_pipeline(
            LogisticRegression(max_iter=2000, random_state=SEED), True
        ),
        "random forest": build_pipeline(
            RandomForestClassifier(n_estimators=300, random_state=SEED), False
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores: dict[str, float] = {}
    print("\n5-fold CV on training set")
    for name, pipe in candidates.items():
        res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=["accuracy", "f1", "roc_auc"])
        scores[name] = res["test_roc_auc"].mean()
        print(
            f"{name:<22} acc={res['test_accuracy'].mean():.3f} "
            f"f1={res['test_f1'].mean():.3f} auc={res['test_roc_auc'].mean():.3f}"
        )

    best_name = max(scores, key=scores.get)
    best = candidates[best_name].fit(X_train, y_train)
    pred = best.predict(X_test)
    proba = best.predict_proba(X_test)[:, 1]
    print(f"\nBest by CV AUC: {best_name}")
    print(f"Held-out test ROC AUC: {roc_auc_score(y_test, proba):.3f}")
    print(classification_report(y_test, pred, target_names=["no disease", "disease"]))

    joblib.dump(best, MODEL_PATH)
    print(f"Saved model: {MODEL_PATH}")


if __name__ == "__main__":
    main()
