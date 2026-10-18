"""Tune binary classifiers, then report feature importance and per-source performance."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from train_binary_classifier import CLEAN, SEED, build_pipeline

ROOT = Path(__file__).resolve().parent
TUNED_PATH = ROOT / "working-data" / "tuned_model.joblib"


def main() -> None:
    df = pd.read_csv(CLEAN)
    X, y = df.drop(columns="target"), df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    searches = {
        "logistic regression": GridSearchCV(
            build_pipeline(LogisticRegression(max_iter=5000, random_state=SEED), True),
            {"model__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10]},
            scoring="roc_auc",
            cv=cv,
        ),
        "random forest": GridSearchCV(
            build_pipeline(RandomForestClassifier(random_state=SEED), False),
            {
                "model__n_estimators": [200, 500],
                "model__max_depth": [3, 5, 8, None],
                "model__min_samples_leaf": [1, 3, 5],
            },
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
        ),
    }

    print("Hyperparameter tuning (5-fold CV AUC, training set)")
    for name, search in searches.items():
        search.fit(X_train, y_train)
        test_auc = roc_auc_score(y_test, search.predict_proba(X_test)[:, 1])
        print(f"{name:<20} cv_auc={search.best_score_:.3f} test_auc={test_auc:.3f} {search.best_params_}")

    best_name = max(searches, key=lambda n: searches[n].best_score_)
    best = searches[best_name].best_estimator_
    print(f"\nSelected: {best_name}")
    joblib.dump(best, TUNED_PATH)

    proba = best.predict_proba(X_test)[:, 1]
    print("\nPer-source performance on held-out test set")
    for source, idx in X_test.groupby("dataset").groups.items():
        yt = y_test.loc[idx]
        pr = proba[X_test.index.get_indexer(idx)]
        auc = roc_auc_score(yt, pr) if yt.nunique() > 1 else float("nan")
        acc = accuracy_score(yt, pr > 0.5)
        print(f"{source:<14} n={len(idx):<4} pos={yt.mean():.0%} acc={acc:.3f} auc={auc:.3f}")

    print("\nLeave-one-source-out (train on other sources, test on held-out source)")
    for source in sorted(df["dataset"].unique()):
        train, test = df[df["dataset"] != source], df[df["dataset"] == source]
        model = clone(best)
        model.fit(train.drop(columns="target"), train["target"])
        p = model.predict_proba(test.drop(columns="target"))[:, 1]
        print(
            f"{source:<14} n={len(test):<4} acc={accuracy_score(test['target'], p > 0.5):.3f} "
            f"auc={roc_auc_score(test['target'], p):.3f}"
        )

    print("\nPermutation importance (test set, AUC drop)")
    result = permutation_importance(
        best, X_test, y_test, scoring="roc_auc", n_repeats=30, random_state=SEED
    )
    ranked = sorted(zip(X.columns, result.importances_mean, result.importances_std), key=lambda t: -t[1])
    for col, mean, std in ranked:
        print(f"{col:<10} {mean:+.4f} ± {std:.4f}")
    print(f"\nSaved tuned model: {TUNED_PATH}")


if __name__ == "__main__":
    main()
