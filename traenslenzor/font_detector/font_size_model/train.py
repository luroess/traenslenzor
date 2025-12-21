"""Training pipeline for font size regression."""

import argparse
import csv
from pathlib import Path
from typing import Tuple

import numpy as np

from .features import FeatureNormalizer
from .model import AdamOptimizer, FontSizeRegressorMLP, MSELoss


def load_dataset(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset from CSV.

    Args:
        csv_path: Path to CSV file

    Returns:
        (features, targets) tuple
    """
    features_list = []
    targets_list = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Extract features (last 34 columns)
            features = np.array([float(row[f"feat_{i}"]) for i in range(34)], dtype=np.float32)
            features_list.append(features)

            # Extract target
            target = float(row["font_size_pt"])
            targets_list.append(target)

    features = np.array(features_list, dtype=np.float32)
    targets = np.array(targets_list, dtype=np.float32).reshape(-1, 1)

    return features, targets


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """
    Compute evaluation metrics.

    Args:
        pred: Predictions
        target: Ground truth

    Returns:
        Dictionary with MAE and RMSE
    """
    mae = np.mean(np.abs(pred - target))
    rmse = np.sqrt(np.mean((pred - target) ** 2))

    return {
        "mae": float(mae),
        "rmse": float(rmse),
    }


def train_epoch(
    model: FontSizeRegressorMLP,
    features: np.ndarray,
    targets: np.ndarray,
    optimizer: AdamOptimizer,
    criterion: MSELoss,
    batch_size: int = 32,
) -> float:
    """
    Train for one epoch.

    Args:
        model: Model to train
        features: Training features
        targets: Training targets
        optimizer: Optimizer
        criterion: Loss function
        batch_size: Batch size

    Returns:
        Average loss
    """
    n_samples = features.shape[0]
    indices = np.random.permutation(n_samples)

    total_loss = 0.0
    n_batches = 0

    for i in range(0, n_samples, batch_size):
        batch_indices = indices[i : i + batch_size]
        batch_features = features[batch_indices]
        batch_targets = targets[batch_indices]

        # Forward pass
        pred = model.forward(batch_features, training=True)
        loss = criterion.forward(pred, batch_targets)

        # Backward pass
        optimizer.zero_grad()
        dout = criterion.backward()
        model.backward(dout)

        # Update weights
        optimizer.step()

        total_loss += loss
        n_batches += 1

    return total_loss / n_batches


def evaluate(
    model: FontSizeRegressorMLP,
    features: np.ndarray,
    targets: np.ndarray,
    criterion: MSELoss,
) -> Tuple[float, dict]:
    """
    Evaluate model.

    Args:
        model: Model to evaluate
        features: Features
        targets: Targets
        criterion: Loss function

    Returns:
        (loss, metrics) tuple
    """
    pred = model.forward(features, training=False)
    loss = criterion.forward(pred, targets)
    metrics = compute_metrics(pred, targets)

    return loss, metrics


def train_model(
    font_name: str,
    data_dir: str = "data",
    output_dir: str = "checkpoints",
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 0.001,
    early_stopping_patience: int = 10,
    verbose: bool = True,
) -> dict:
    """
    Train model for a single font.

    Args:
        font_name: Font name
        data_dir: Directory containing CSV files
        output_dir: Directory to save checkpoints
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        early_stopping_patience: Patience for early stopping
        verbose: Whether to print progress

    Returns:
        Dictionary with training results
    """
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Training model for {font_name}")
        print(f"{'=' * 60}")

    # Load datasets
    data_path = Path(data_dir) / font_name

    if verbose:
        print("Loading datasets...")

    train_features, train_targets = load_dataset(str(data_path / "train.csv"))
    val_features, val_targets = load_dataset(str(data_path / "val.csv"))
    test_features, test_targets = load_dataset(str(data_path / "test.csv"))

    if verbose:
        print(f"  Train: {train_features.shape[0]} samples")
        print(f"  Val: {val_features.shape[0]} samples")
        print(f"  Test: {test_features.shape[0]} samples")

    # Fit normalizer on training data
    if verbose:
        print("Fitting feature normalizer...")

    normalizer = FeatureNormalizer.fit(list(train_features))

    # Normalize features
    train_features_norm = normalizer.normalize(train_features)
    val_features_norm = normalizer.normalize(val_features)
    test_features_norm = normalizer.normalize(test_features)

    # Create model
    if verbose:
        print("Initializing model...")

    model = FontSizeRegressorMLP(input_dim=34, hidden1=64, hidden2=32)
    criterion = MSELoss()
    optimizer = AdamOptimizer(model.get_parameters(), lr=lr)

    # Training loop
    if verbose:
        print(f"\nTraining for {epochs} epochs...")

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(
            model, train_features_norm, train_targets, optimizer, criterion, batch_size
        )

        # Validate
        val_loss, val_metrics = evaluate(model, val_features_norm, val_targets, criterion)

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            # Save best model
            output_path = Path(output_dir) / font_name
            output_path.mkdir(parents=True, exist_ok=True)
            model.save(str(output_path / "best.json"))
            normalizer.save(str(output_path / "norm.json"))
        else:
            patience_counter += 1

        # Print progress
        if verbose and (epoch + 1) % 10 == 0:
            print(
                f"  Epoch {epoch + 1}/{epochs}: "
                f"train_loss={train_loss:.4f}, "
                f"val_loss={val_loss:.4f}, "
                f"val_mae={val_metrics['mae']:.2f}, "
                f"val_rmse={val_metrics['rmse']:.2f}"
            )

        # Early stopping
        if patience_counter >= early_stopping_patience:
            if verbose:
                print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    # Load best model and evaluate on test set
    if verbose:
        print(f"\nLoading best model from epoch {best_epoch + 1}...")

    output_path = Path(output_dir) / font_name
    model = FontSizeRegressorMLP.load(str(output_path / "best.json"))

    test_loss, test_metrics = evaluate(model, test_features_norm, test_targets, criterion)

    if verbose:
        print("\nTest Results:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  MAE: {test_metrics['mae']:.2f} pt")
        print(f"  RMSE: {test_metrics['rmse']:.2f} pt")

    # Save calibration data (predictions vs ground truth on test set)
    pred_test = model.forward(test_features_norm, training=False)

    calibration_path = output_path / "calibration.csv"
    with open(calibration_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["y_true", "y_pred"])
        for true_val, pred_val in zip(test_targets.flatten(), pred_test.flatten()):
            writer.writerow([true_val, pred_val])

    if verbose:
        print(f"\nSaved calibration data to {calibration_path}")
        print(f"{'=' * 60}\n")

    return {
        "font_name": font_name,
        "best_epoch": best_epoch,
        "test_loss": test_loss,
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
    }


def main():
    """CLI for training."""
    parser = argparse.ArgumentParser(description="Train font size regression model")
    parser.add_argument(
        "--font",
        type=str,
        required=True,
        help="Font name",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Data directory (default: module's data directory)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for checkpoints (default: module's checkpoints directory)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience",
    )

    args = parser.parse_args()

    # Set default paths if not provided
    if args.data_dir is None:
        args.data_dir = str(Path(__file__).parent.parent / "data")
    if args.output_dir is None:
        args.output_dir = str(Path(__file__).parent.parent / "checkpoints")

    train_model(
        font_name=args.font,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        early_stopping_patience=args.patience,
        verbose=True,
    )


if __name__ == "__main__":
    main()
