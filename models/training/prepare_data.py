"""Prepare the AI4I dataset and evaluate initial failure-classification baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "datasets" / "ai4i2020.csv"
TARGET = "Machine failure"
CATEGORICAL_FEATURES = ["Type"]
NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def load_dataset(dataset_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load selected input features and the machine-failure target."""
    data = pd.read_csv(dataset_path)
    required_columns = FEATURES + [TARGET]
    missing_columns = sorted(set(required_columns) - set(data.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")
    if data[required_columns].isna().any().any():
        raise ValueError("Selected features and target contain missing values.")

    labels = data[TARGET]
    if not labels.isin([0, 1]).all():
        raise ValueError(f"{TARGET} must contain only 0 and 1.")

    return data[FEATURES].copy(), labels.astype(int)


def build_preprocessor() -> ColumnTransformer:
    """Scale numeric features and one-hot encode the product type."""
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def evaluate_baseline(
    name: str,
    classifier: DummyClassifier | LogisticRegression,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """Fit preprocessing on training data and report held-out metrics."""
    model = Pipeline(
        steps=[
            ("preprocessing", build_preprocessor()),
            ("classifier", classifier),
        ]
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"\n{name}")
    print(f"  Accuracy:          {accuracy_score(y_test, predictions):.3f}")
    print(f"  Balanced accuracy: {balanced_accuracy_score(y_test, predictions):.3f}")
    print(f"  Failure precision: {precision_score(y_test, predictions, zero_division=0):.3f}")
    print(f"  Failure recall:    {recall_score(y_test, predictions, zero_division=0):.3f}")
    print(f"  Failure F1:        {f1_score(y_test, predictions, zero_division=0):.3f}")
    print("  Confusion matrix (rows=actual [0, 1], columns=predicted [0, 1]):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect the AI4I data split and evaluate initial baselines."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the AI4I CSV (defaults to datasets/ai4i2020.csv).",
    )
    args = parser.parse_args()

    X, y = load_dataset(args.data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"Dataset: {args.data_path}")
    print(f"Rows: {len(X)}; training rows: {len(X_train)}; test rows: {len(X_test)}")
    print(f"Input features ({len(FEATURES)}): {', '.join(FEATURES)}")
    print("Excluded: UDI, Product ID, and all failure-mode labels (TWF/HDF/PWF/OSF/RNF).")
    print(
        "Failure rate: "
        f"overall={y.mean():.2%}, train={y_train.mean():.2%}, test={y_test.mean():.2%}"
    )

    evaluate_baseline(
        "Majority-class baseline",
        DummyClassifier(strategy="most_frequent"),
        X_train,
        X_test,
        y_train,
        y_test,
    )
    evaluate_baseline(
        "Logistic-regression baseline",
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        X_train,
        X_test,
        y_train,
        y_test,
    )

    print(
        "\nLimitations: AI4I is synthetic; these are initial baselines, not a certified "
        "safety system or a guarantee of future failure prediction."
    )


if __name__ == "__main__":
    main()
