"""Compare models with and without the `dataset` (source) feature, then tune the threshold."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)

from train_binary_classifier import CLEAN, SEED, build_pipeline

ROOT = Path(__file__).resolve().parent
FINAL_PATH = ROOT / "working-data" / "final_model_no_source.joblib"


def make_search(use_source: bool, cv) -> GridSearchCV:
    pipe = build_pipeline(LogisticRegression(max_iter=5000, random_state=SEED), True, use_source)
    return GridSearchCV(pipe, {"model__C": [0.01, 0.03, 0.1, 0.3, 1, 3]}, scoring="roc_auc", cv=cv)


def main() -> None:
    df = pd.read_csv(CLEAN)
    X, y = df.drop(columns="target"), df["target"]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    sources = sorted(df["dataset"].unique())

    print("Leave-one-source-out AUC (train on other sources, test on held-out source)")
    print(f"{'source':<14} {'with source':>12} {'without':>10}")
    means: dict[bool, list[float]] = {True: [], False: []}
    for source in sources:
        row = []
        for use_source in (True, False):
            train, test = df[df["dataset"] != source], df[df["dataset"] == source]
            search = make_search(use_source, cv).fit(train.drop(columns="target"), train["target"])
            auc = roc_auc_score(
                test["target"], search.predict_proba(test.drop(columns="target"))[:, 1]
            )
            means[use_source].append(auc)
            row.append(auc)
        print(f"{source:<14} {row[0]:>12.3f} {row[1]:>10.3f}")
    print(f"{'mean':<14} {np.mean(means[True]):>12.3f} {np.mean(means[False]):>10.3f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    search = make_search(False, cv).fit(X_train, y_train)
    final = search.best_estimator_
    print(f"\nFinal model without source: {search.best_params_}, cv_auc={search.best_score_:.3f}")
    print(f"Held-out test AUC: {roc_auc_score(y_test, final.predict_proba(X_test)[:, 1]):.3f}")

    # Threshold chosen on out-of-fold training predictions to avoid test leakage.
    oof = cross_val_predict(final, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    thresholds = np.linspace(0.2, 0.8, 61)

    def f1_at(t: float) -> float:
        pred = oof >= t
        tp = np.sum(pred & (y_train == 1))
        return 2 * tp / (pred.sum() + (y_train == 1).sum())

    best_t = float(max(thresholds, key=f1_at))
    proba = final.predict_proba(X_test)[:, 1]
    print(f"\nThreshold tuned for F1 on out-of-fold predictions: {best_t:.2f}")
    for t in (0.5, best_t):
        pred = proba >= t
        tp = int(np.sum(pred & (y_test == 1)))
        fp = int(np.sum(pred & (y_test == 0)))
        fn = int(np.sum(~pred & (y_test == 1)))
        tn = int(np.sum(~pred & (y_test == 0)))
        print(
            f"threshold={t:.2f} acc={(tp + tn) / len(y_test):.3f} "
            f"precision={tp / max(tp + fp, 1):.3f} recall={tp / max(tp + fn, 1):.3f} "
            f"(tp={tp} fp={fp} fn={fn} tn={tn})"
        )

    joblib.dump({"model": final, "threshold": best_t}, FINAL_PATH)
    print(f"\nSaved: {FINAL_PATH}")


if __name__ == "__main__":
    main()
