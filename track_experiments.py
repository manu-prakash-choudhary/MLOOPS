"""Compare and track heart-disease models with MLflow."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)

from train_binary_classifier import CLEAN, SEED, build_pipeline

ROOT = Path(__file__).resolve().parent
TRACKING_DB = ROOT / "mlflow.db"
FINAL_MODEL_PATH = ROOT / "working-data" / "final_model_no_source.joblib"
EXPERIMENT_NAME = "heart-disease-binary-classification"
THRESHOLDS = np.linspace(0.2, 0.8, 61)


def build_searches(cv: StratifiedKFold) -> list[tuple[str, Any, dict[str, list[Any]] | None]]:
    return [
        (
            "majority-baseline",
            build_pipeline(DummyClassifier(strategy="most_frequent"), True),
            None,
        ),
        (
            "logistic-regression-with-source",
            build_pipeline(LogisticRegression(max_iter=5000, random_state=SEED), True),
            {"model__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10]},
        ),
        (
            "random-forest-with-source",
            build_pipeline(RandomForestClassifier(random_state=SEED), False),
            {
                "model__n_estimators": [200, 500],
                "model__max_depth": [3, 5, 8, None],
                "model__min_samples_leaf": [1, 3, 5],
            },
        ),
        (
            "logistic-regression-no-source",
            build_pipeline(
                LogisticRegression(max_iter=5000, random_state=SEED),
                True,
                use_source=False,
            ),
            {"model__C": [0.01, 0.03, 0.1, 0.3, 1, 3]},
        ),
    ]


def choose_f1_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    scores = [
        f1_score(y_true, probabilities >= threshold, zero_division=0)
        for threshold in THRESHOLDS
    ]
    return float(THRESHOLDS[int(np.argmax(scores))])


def log_run(
    name: str,
    estimator: Any,
    param_grid: dict[str, list[Any]] | None,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv: StratifiedKFold,
) -> dict[str, Any]:
    with mlflow.start_run(run_name=name):
        mlflow.log_params(
            {
                "model_name": name,
                "seed": SEED,
                "test_size": 0.2,
                "cv_folds": cv.n_splits,
                "features_include_dataset": "no-source" not in name,
            }
        )

        cv_results: pd.DataFrame | None = None
        if param_grid:
            search = GridSearchCV(
                estimator,
                param_grid,
                scoring="roc_auc",
                cv=cv,
                n_jobs=-1,
                refit=True,
            ).fit(X_train, y_train)
            fitted = search.best_estimator_
            cv_results = pd.DataFrame(search.cv_results_)
            mlflow.log_params(
                {
                    f"best_{key.replace('model__', '')}": str(value)
                    for key, value in search.best_params_.items()
                }
            )
            mlflow.log_metric("tuning_best_cv_roc_auc", float(search.best_score_))
            mlflow.log_param(
                "search_space",
                json.dumps(
                    {key: [str(value) for value in values] for key, values in param_grid.items()}
                ),
            )
        else:
            fitted = clone(estimator)

        if param_grid is None:
            validation = cross_validate(
                fitted,
                X_train,
                y_train,
                cv=cv,
                scoring=["accuracy", "f1", "roc_auc"],
            )
            fitted.fit(X_train, y_train)
        else:
            validation = cross_validate(
                fitted,
                X_train,
                y_train,
                cv=cv,
                scoring=["accuracy", "f1", "roc_auc"],
            )

        for metric_name in ("accuracy", "f1", "roc_auc"):
            values = validation[f"test_{metric_name}"]
            mlflow.log_metric(f"cv_{metric_name}_mean", float(values.mean()))
            mlflow.log_metric(f"cv_{metric_name}_std", float(values.std()))

        probabilities = fitted.predict_proba(X_test)[:, 1]
        threshold = (
            choose_f1_threshold(
                y_train,
                cross_val_predict(fitted, X_train, y_train, cv=cv, method="predict_proba")[:, 1],
            )
            if "no-source" in name
            else 0.5
        )
        predictions = (probabilities >= threshold).astype(int)
        metrics = {
            "test_accuracy": accuracy_score(y_test, predictions),
            "test_precision": precision_score(y_test, predictions, zero_division=0),
            "test_recall": recall_score(y_test, predictions, zero_division=0),
            "test_f1": f1_score(y_test, predictions, zero_division=0),
            "test_roc_auc": roc_auc_score(y_test, probabilities),
        }
        mlflow.log_params({"decision_threshold": threshold})
        mlflow.log_metrics({key: float(value) for key, value in metrics.items()})

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            model_path = temp_path / "model.joblib"
            joblib.dump({"model": fitted, "threshold": threshold}, model_path)
            mlflow.log_artifact(str(model_path), artifact_path="model")

            if cv_results is not None:
                results_path = temp_path / "cv_results.csv"
                cv_results.to_csv(results_path, index=False)
                mlflow.log_artifact(str(results_path), artifact_path="tuning")

            fig, ax = plt.subplots(figsize=(6, 5))
            ConfusionMatrixDisplay.from_predictions(
                y_test,
                predictions,
                display_labels=["no disease", "disease"],
                ax=ax,
                colorbar=False,
            )
            ax.set_title(f"{name}: held-out test confusion matrix")
            fig.tight_layout()
            confusion_path = temp_path / "confusion_matrix.png"
            fig.savefig(confusion_path, dpi=150)
            plt.close(fig)
            mlflow.log_artifact(str(confusion_path), artifact_path="plots")

            fig, ax = plt.subplots(figsize=(6, 5))
            RocCurveDisplay.from_predictions(y_test, probabilities, ax=ax)
            ax.set_title(f"{name}: held-out test ROC curve")
            fig.tight_layout()
            roc_path = temp_path / "roc_curve.png"
            fig.savefig(roc_path, dpi=150)
            plt.close(fig)
            mlflow.log_artifact(str(roc_path), artifact_path="plots")

        print(
            f"{name:<38} cv_auc={validation['test_roc_auc'].mean():.3f} "
            f"test_auc={metrics['test_roc_auc']:.3f} "
            f"test_f1={metrics['test_f1']:.3f} threshold={threshold:.2f}"
        )
        return {
            "name": name,
            "estimator": fitted,
            "threshold": threshold,
            **metrics,
        }


def main() -> None:
    if not CLEAN.is_file():
        raise FileNotFoundError(f"{CLEAN} not found; run `python prepare_dataset.py` first.")
    mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB.as_posix()}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = pd.read_csv(CLEAN)
    X, y = df.drop(columns="target"), df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    print(f"Training rows: {len(X_train)} | Held-out rows: {len(X_test)}")

    results = [
        log_run(name, estimator, grid, X_train, y_train, X_test, y_test, cv)
        for name, estimator, grid in build_searches(cv)
    ]

    final = next(
        result for result in results if result["name"] == "logistic-regression-no-source"
    )
    FINAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": final["estimator"], "threshold": final["threshold"]},
        FINAL_MODEL_PATH,
    )
    print(f"\nFinal API artifact saved: {FINAL_MODEL_PATH}")
    print(f"MLflow tracking database: {TRACKING_DB}")
    print("Start the UI with: mlflow ui --backend-store-uri sqlite:///mlflow.db")


if __name__ == "__main__":
    main()
