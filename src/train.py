from __future__ import annotations

import json
import os
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    DATA_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.features import create_features


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical = [
        c for c in X.columns
        if c not in categorical
    ]

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median"),
        ),
        (
            "scaler",
            StandardScaler(),
        ),
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=2,
            ),
        ),
    ])

    return ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            numerical,
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical,
        ),
    ])


def build_models(preprocessor):
    return {
        "logistic_regression": Pipeline([
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    max_iter=3000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]),

        "random_forest": Pipeline([
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=500,
                    max_depth=10,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]),
    }


def optimize_threshold(y_true, probabilities):
    """
    Choose threshold using only training/OOF predictions.
    The final test set remains untouched.
    """
    best = {
        "threshold": 0.50,
        "f1": -1.0,
    }

    for threshold in np.arange(0.20, 0.701, 0.005):
        pred = (
            probabilities >= threshold
        ).astype(int)

        score = f1_score(
            y_true,
            pred,
            zero_division=0,
        )

        if score > best["f1"]:
            best = {
                "threshold": float(threshold),
                "f1": float(score),
            }

    return best


def evaluate(
    name,
    y_true,
    probabilities,
    threshold,
):
    predictions = (
        probabilities >= threshold
    ).astype(int)

    result = {
        "model": name,
        "threshold": threshold,
        "accuracy": accuracy_score(
            y_true,
            predictions,
        ),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),
    }

    print("\n" + "=" * 65)
    print(name.upper())
    print("=" * 65)

    for key, value in result.items():
        if key != "model":
            print(f"{key:12}: {value:.4f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, predictions))

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            predictions,
            target_names=["Stay", "Churn"],
            zero_division=0,
        )
    )

    return result


def main():
    start = time.perf_counter()

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print(
        f"Dataset: {df.shape[0]:,} rows x "
        f"{df.shape[1]} columns"
    )

    if "Churn" not in df.columns:
        raise ValueError(
            "Target column 'Churn' is missing."
        )

    y = df["Churn"].map(
        {"No": 0, "Yes": 1}
    )

    if y.isna().any():
        raise ValueError(
            "Churn contains unexpected values."
        )

    # Never use customerID as a predictive feature.
    X = create_features(
        df.drop(columns=["Churn"])
    )

    if "customerID" in X.columns:
        X = X.drop(columns=["customerID"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    preprocessor = build_preprocessor(
        X_train
    )

    models = build_models(
        preprocessor
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    benchmark = []

    # Threshold/model selection happens on OOF predictions.
    for name, model in models.items():

        print(
            f"\nRunning 5-fold OOF benchmark: {name}"
        )

        oof_probability = cross_val_predict(
            model,
            X_train,
            y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=None,
        )[:, 1]

        threshold_info = optimize_threshold(
            y_train,
            oof_probability,
        )

        print(
            f"OOF threshold: "
            f"{threshold_info['threshold']:.3f}"
        )

        benchmark.append({
            "name": name,
            "oof_f1": threshold_info["f1"],
            "threshold": threshold_info["threshold"],
        })

    # Select based on OOF F1, not final test F1.
    selected = max(
        benchmark,
        key=lambda x: x["oof_f1"],
    )

    selected_name = selected["name"]
    selected_threshold = selected["threshold"]

    print(
        f"\nSelected model: {selected_name}"
    )
    print(
        f"Selected threshold: "
        f"{selected_threshold:.3f}"
    )

    final_model = models[
        selected_name
    ]

    train_start = time.perf_counter()

    final_model.fit(
        X_train,
        y_train,
    )

    train_time = (
        time.perf_counter()
        - train_start
    )

    probabilities = final_model.predict_proba(
        X_test
    )[:, 1]

    final_metrics = evaluate(
        selected_name,
        y_test,
        probabilities,
        selected_threshold,
    )

    artifact = {
        "model": final_model,
        "model_name": selected_name,
        "threshold": selected_threshold,
        "features": X.columns.tolist(),
        "metrics": final_metrics,
        "train_time_seconds": train_time,
        "random_state": RANDOM_STATE,
        "version": "1.0.0",
    }

    os.makedirs(
        os.path.dirname(MODEL_PATH),
        exist_ok=True,
    )

    joblib.dump(
        artifact,
        MODEL_PATH,
        compress=3,
    )

    with open(
        "reports/model_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "benchmark": benchmark,
                "selected_model": selected_name,
                "final_test_metrics": final_metrics,
            },
            file,
            indent=2,
        )

    elapsed = (
        time.perf_counter()
        - start
    )

    print(
        f"\nModel saved: {MODEL_PATH}"
    )
    print(
        f"Training pipeline time: "
        f"{elapsed:.2f} seconds"
    )


if __name__ == "__main__":
    main()
