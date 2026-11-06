"""Train and evaluate a normal-pattern Autoencoder for anomaly detection."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from backend.services.anomaly_model import Autoencoder
from models.training.prepare_data import (
    DEFAULT_DATASET_PATH,
    FEATURES,
    PROJECT_ROOT,
    TARGET,
    build_preprocessor,
    load_dataset,
)

RANDOM_STATE = 42
VALIDATION_FALSE_ALARM_TARGET = 0.05


def to_float_tensor(values: object) -> torch.Tensor:
    """Convert transformed features to a dense float32 tensor."""
    to_array = getattr(values, "toarray", None)
    if callable(to_array):
        values = to_array()
    return torch.from_numpy(np.asarray(values, dtype=np.float32))


def mean_reconstruction_loss(
    model: Autoencoder,
    values: torch.Tensor,
    loss_function: nn.Module,
    device: torch.device,
) -> float:
    """Calculate average row reconstruction loss without updating the model."""
    model.eval()
    loader = DataLoader(TensorDataset(values), batch_size=512)
    total_loss = 0.0
    row_count = 0
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            loss = loss_function(model(batch), batch)
            total_loss += loss.item() * len(batch)
            row_count += len(batch)
    if row_count == 0:
        raise ValueError("Cannot calculate reconstruction loss for an empty dataset.")
    return total_loss / row_count


def reconstruction_errors(
    model: Autoencoder,
    values: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Return one mean-squared reconstruction error per input row."""
    model.eval()
    scores: list[np.ndarray] = []
    loader = DataLoader(TensorDataset(values), batch_size=512)
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            reconstructed = model(batch)
            errors = torch.mean((batch - reconstructed) ** 2, dim=1)
            scores.append(errors.cpu().numpy())
    return np.concatenate(scores)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an Autoencoder on normal AI4I examples and evaluate anomalies."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to the AI4I CSV (defaults to datasets/ai4i2020.csv).",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=PROJECT_ROOT / "models" / "saved" / "anomaly_autoencoder.pt",
        help="Where to save the trained PyTorch weights.",
    )
    parser.add_argument(
        "--preprocessor-output",
        type=Path,
        default=PROJECT_ROOT / "models" / "saved" / "anomaly_preprocessor.joblib",
        help="Where to save preprocessing and the validation threshold.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1:
        parser.error("--epochs and --patience must both be positive.")

    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y = load_dataset(args.data_path)
    X_development, X_test, y_development, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_normal = X_development.loc[y_development == 0]
    X_train_normal, X_validation_normal = train_test_split(
        X_normal,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    preprocessor = build_preprocessor()
    train_values = to_float_tensor(preprocessor.fit_transform(X_train_normal))
    validation_values = to_float_tensor(preprocessor.transform(X_validation_normal))
    test_values = to_float_tensor(preprocessor.transform(X_test))
    train_loader = DataLoader(
        TensorDataset(train_values),
        batch_size=128,
        shuffle=True,
    )

    model = Autoencoder(train_values.shape[1]).to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_validation_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = loss_function(model(batch), batch)
            loss.backward()
            optimizer.step()

        validation_loss = mean_reconstruction_loss(
            model,
            validation_values,
            loss_function,
            device,
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model checkpoint.")
    model.load_state_dict(best_state)
    model.to(device)

    validation_errors = reconstruction_errors(model, validation_values, device)
    test_errors = reconstruction_errors(model, test_values, device)
    threshold = float(np.quantile(validation_errors, 1 - VALIDATION_FALSE_ALARM_TARGET))
    anomaly_predictions = (test_errors > threshold).astype(int)

    print(f"Dataset: {args.data_path}")
    print(f"Device: {device}; best epoch: {best_epoch}")
    print(
        f"Normal-only training rows: {len(X_train_normal)}; "
        f"normal validation rows: {len(X_validation_normal)}; "
        f"held-out test rows: {len(X_test)}"
    )
    print(
        "Threshold: "
        f"{threshold:.6f} (95th percentile of normal validation reconstruction errors)"
    )
    print(f"Balanced accuracy: {balanced_accuracy_score(y_test, anomaly_predictions):.3f}")
    print(
        f"Anomaly precision: {precision_score(y_test, anomaly_predictions, zero_division=0):.3f}"
    )
    print(f"Failure recall:    {recall_score(y_test, anomaly_predictions, zero_division=0):.3f}")
    print(f"Average precision: {average_precision_score(y_test, test_errors):.3f}")
    print(f"ROC AUC:           {roc_auc_score(y_test, test_errors):.3f}")
    print("Confusion matrix (rows=actual [0, 1], columns=predicted [0, 1]):")
    print(confusion_matrix(y_test, anomaly_predictions, labels=[0, 1]))
    normal_test_count = int((y_test == 0).sum())
    false_alarm_count = int(((y_test == 0) & (anomaly_predictions == 1)).sum())
    print(
        f"Held-out normal false-alarm rate: "
        f"{false_alarm_count}/{normal_test_count} "
        f"({false_alarm_count / normal_test_count:.2%})"
    )

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.preprocessor_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.model_output)
    joblib.dump(
        {
            "preprocessor": preprocessor,
            "features": FEATURES,
            "target": TARGET,
            "threshold": threshold,
            "validation_false_alarm_target": VALIDATION_FALSE_ALARM_TARGET,
            "training_dataset": str(args.data_path),
        },
        args.preprocessor_output,
    )
    print(f"Saved PyTorch weights: {args.model_output}")
    print(f"Saved preprocessing and threshold: {args.preprocessor_output}")
    print(
        "Limitations: AI4I is synthetic; an anomaly is not proof of failure, "
        "and this is not a certified safety system."
    )


if __name__ == "__main__":
    main()
