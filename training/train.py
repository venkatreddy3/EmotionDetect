"""Reproducible model training pipeline for Facial Emotion Recognition."""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Set backend for universal Keras 3 compatibility
os.environ.setdefault("KERAS_BACKEND", "torch")
import keras
from sklearn.utils.class_weight import compute_class_weight

from training.dataset import (
    CANONICAL_CLASSES,
    InMemoryFERDataset,
    parse_csv_dataset,
    scan_directory_dataset,
)
from training.model import get_model
from training.preprocessing import preprocess_image

def set_reproducibility_seeds(seed: int = 42):
    """Sets random seeds across python, numpy, and framework runtimes."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)

def compute_balanced_class_weights(y_train: np.ndarray) -> Dict[int, float]:
    """
    Computes balanced class weights only when justified by class frequencies.
    Weight formula: n_samples / (n_classes * np.bincount(y))
    """
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return {int(cls): float(w) for cls, w in zip(classes, weights)}

def plot_training_curves(history_dict: Dict, output_path: Union[str, Path]):
    """Plots and saves loss and accuracy learning curves."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    epochs_range = range(1, len(history_dict["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#111317")

    for ax in (ax1, ax2):
        ax.set_facecolor("#16191E")
        ax.grid(True, linestyle="--", alpha=0.3, color="#3D4656")
        ax.tick_params(colors="#E5E7EB")
        for spine in ax.spines.values():
            spine.set_color("#2A303C")

    # Loss Curve
    ax1.plot(epochs_range, history_dict["loss"], label="Train Loss", color="#2DD4BF", linewidth=2)
    if "val_loss" in history_dict:
        ax1.plot(epochs_range, history_dict["val_loss"], label="Val Loss", color="#F87171", linewidth=2)
    ax1.set_title("Categorical Cross-Entropy Loss", color="#FAFBFC", fontsize=12, fontweight="600")
    ax1.set_xlabel("Epoch", color="#E5E7EB")
    ax1.set_ylabel("Loss", color="#E5E7EB")
    ax1.legend(facecolor="#1D2128", edgecolor="#2A303C", labelcolor="#FAFBFC")

    # Accuracy Curve
    ax2.plot(epochs_range, history_dict["accuracy"], label="Train Accuracy", color="#2DD4BF", linewidth=2)
    if "val_accuracy" in history_dict:
        ax2.plot(epochs_range, history_dict["val_accuracy"], label="Val Accuracy", color="#60A5FA", linewidth=2)
    ax2.set_title("Classification Accuracy", color="#FAFBFC", fontsize=12, fontweight="600")
    ax2.set_xlabel("Epoch", color="#E5E7EB")
    ax2.set_ylabel("Accuracy", color="#E5E7EB")
    ax2.legend(facecolor="#1D2128", edgecolor="#2A303C", labelcolor="#FAFBFC")

    plt.tight_layout()
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()

def load_or_preprocess_training_data(
    data_dir: Path,
    val_split: float = 0.15,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads images and labels into memory arrays for training and validation."""
    csv_candidates = list(data_dir.glob("*.csv"))
    if csv_candidates:
        X_train, y_train, X_val, y_val, _, _, _ = parse_csv_dataset(csv_candidates[0], val_split=val_split, seed=seed)
        return X_train, y_train, X_val, y_val

    samples, _ = scan_directory_dataset(data_dir, val_split=val_split, seed=seed)
    train_samples = [s for s in samples if s.split == "train" and s.is_valid]
    val_samples = [s for s in samples if s.split == "val" and s.is_valid]

    if not train_samples:
        raise ValueError(f"No valid training samples found in {data_dir}")

    X_train = np.array([preprocess_image(s.file_path) for s in train_samples], dtype=np.float32)
    y_train = np.array([s.class_idx for s in train_samples], dtype=np.int32)

    if val_samples:
        X_val = np.array([preprocess_image(s.file_path) for s in val_samples], dtype=np.float32)
        y_val = np.array([s.class_idx for s in val_samples], dtype=np.int32)
    else:
        # Fallback split if val_samples was 0
        n_val = int(len(X_train) * val_split)
        X_val, y_val = X_train[:n_val], y_train[:n_val]
        X_train, y_train = X_train[n_val:], y_train[n_val:]

    return X_train, y_train, X_val, y_val

def run_training(
    data_dir: str = "data/raw",
    checkpoint_dir: str = "models/checkpoints",
    artifacts_dir: str = "artifacts",
    model_type: str = "custom_cnn",
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    val_split: float = 0.15,
    seed: int = 42,
    use_class_weight: bool = True,
    dry_run: bool = False,
) -> Dict:
    """Orchestrates full reproducible model training."""
    set_reproducibility_seeds(seed)

    chk_path = Path(checkpoint_dir)
    art_path = Path(artifacts_dir)
    plots_dir = art_path / "plots"
    reports_dir = art_path / "reports"

    chk_path.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {data_dir}...")
    X_train, y_train, X_val, y_val = load_or_preprocess_training_data(Path(data_dir), val_split=val_split, seed=seed)

    if dry_run:
        print("[DRY-RUN] Truncating dataset for fast smoke-test validation...")
        X_train, y_train = X_train[:32], y_train[:32]
        X_val, y_val = X_val[:16], y_val[:16]
        epochs = 2
        batch_size = 16

    print(f"Train samples: {len(X_train)} | Validation samples: {len(X_val)}")
    y_train_cat = keras.utils.to_categorical(y_train, num_classes=7)
    y_val_cat = keras.utils.to_categorical(y_val, num_classes=7)

    # Class weighting
    class_weights = None
    if use_class_weight:
        class_weights = compute_balanced_class_weights(y_train)
        print("Computed balanced class weights:")
        for idx, w in class_weights.items():
            print(f"  - {CANONICAL_CLASSES[idx]} (Class {idx}): {w:.4f}")

    print(f"Instantiating model: {model_type}...")
    model = get_model(model_type, input_shape=(48, 48, 1), num_classes=7)

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    best_model_path = chk_path / "best_model.keras"
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print(f"Beginning training ({epochs} epochs, batch_size={batch_size})...")
    history = model.fit(
        X_train,
        y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # Clean history dictionary (convert floats)
    history_dict = {
        k: [float(val) for val in v]
        for k, v in history.history.items()
    }

    # Save artifacts
    history_file = reports_dir / "training_history.json"
    with open(history_file, "w") as f:
        json.dump(history_dict, f, indent=2)

    config_payload = {
        "model_type": model_type,
        "epochs_trained": len(history_dict["loss"]),
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "val_split": val_split,
        "class_weighting_used": use_class_weight,
        "best_val_accuracy": float(max(history_dict.get("val_accuracy", [0]))),
        "min_val_loss": float(min(history_dict.get("val_loss", [0]))),
    }
    with open(reports_dir / "training_config.json", "w") as f:
        json.dump(config_payload, f, indent=2)

    # Plot learning curves
    plot_file = plots_dir / "training_curves.png"
    plot_training_curves(history_dict, plot_file)

    print("\nTraining Run Complete.")
    print(f"Artifacts saved:")
    print(f"  - Checkpoint: {best_model_path}")
    print(f"  - History: {history_file}")
    print(f"  - Config: {reports_dir / 'training_config.json'}")
    print(f"  - Curves: {plot_file}")

    return {
        "history": history_dict,
        "config": config_payload,
        "model_path": str(best_model_path),
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train custom CNN for Facial Emotion Recognition")
    parser.add_argument("--data-dir", type=str, default="data/raw")
    parser.add_argument("--checkpoint-dir", type=str, default="models/checkpoints")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")
    parser.add_argument("--model-type", type=str, default="custom_cnn", choices=["custom_cnn", "residual_compact"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-class-weight", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()
    run_training(
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        artifacts_dir=args.artifacts_dir,
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_split=args.val_split,
        seed=args.seed,
        use_class_weight=not args.no_class_weight,
        dry_run=args.dry_run,
    )
