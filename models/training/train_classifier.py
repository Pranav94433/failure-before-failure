"""Train, evaluate, and save the AI4I machine-failure classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from models.training.prepare_data import (
    DEFAULT_DATASET_PATH,
    FEATURES,
    PROJECT_ROOT,
    TARGET,
    build_preprocessor,
    load_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a Random Forest on the AI4I dataset."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the AI4I CSV (defaults to datasets/ai4i2020.csv).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=PROJECT_ROOT / "models" / "saved" / "failure_classifier.joblib",
        help="Where to save the fitted preprocessing and classifier pipeline.",
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

    model = Pipeline(
        steps=[
            ("preprocessing", build_preprocessor()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    failure_probabilities = model.predict_proba(X_test)[:, 1]

    print(f"Dataset: {args.data_path}")
    print(
        f"Rows: {len(X)}; training rows: {len(X_train)}; "
        f"held-out test rows: {len(X_test)}"
    )
    print(f"Features: {', '.join(FEATURES)}")
    print("The 0.5 classifier threshold is unchanged; the test set was not used for tuning.")
    print(f"Accuracy:          {accuracy_score(y_test, predictions):.3f}")
    print(f"Balanced accuracy: {balanced_accuracy_score(y_test, predictions):.3f}")
    print(f"Failure precision: {precision_score(y_test, predictions, zero_division=0):.3f}")
    print(f"Failure recall:    {recall_score(y_test, predictions, zero_division=0):.3f}")
    print(f"Failure F1:        {f1_score(y_test, predictions, zero_division=0):.3f}")
    print(f"Average precision: {average_precision_score(y_test, failure_probabilities):.3f}")
    print(f"ROC AUC:           {roc_auc_score(y_test, failure_probabilities):.3f}")
    print("Confusion matrix (rows=actual [0, 1], columns=predicted [0, 1]):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "target": TARGET,
            "features": FEATURES,
            "positive_label": 1,
            "decision_threshold": 0.5,
            "training_dataset": str(args.data_path),
        },
        args.output_path,
    )
    print(f"Saved fitted pipeline: {args.output_path}")
    print(
        "Limitations: AI4I is synthetic and this model is a learning prototype, "
        "not a certified safety system or a guarantee of future failure prediction."
    )


if __name__ == "__main__":
    main()
