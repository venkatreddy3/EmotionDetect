"""Model training orchestrator."""
import argparse
from pathlib import Path
from training.dataset import load_directory_dataset
from training.models import build_baseline_cnn, build_mini_xception

def train(
    data_dir: str,
    output_dir: str = "models/checkpoints",
    model_type: str = "mini_xception",
    epochs: int = 50,
    batch_size: int = 64,
    learning_rate: float = 0.001,
):
    import tensorflow as tf

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {data_dir}...")
    train_ds, val_ds = load_directory_dataset(Path(data_dir), batch_size=batch_size)

    print(f"Instantiating model architecture: {model_type}...")
    if model_type == "baseline":
        model = build_baseline_cnn()
    else:
        model = build_mini_xception()

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(out_path / "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print("Starting model training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    print(f"Training complete. Best model checkpoint saved to {out_path / 'best_model.keras'}")
    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Facial Emotion Recognition CNN")
    parser.add_argument("--data-dir", type=str, default="data/raw/train", help="Path to training dataset directory")
    parser.add_argument("--output-dir", type=str, default="models/checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--model-type", type=str, default="mini_xception", choices=["baseline", "mini_xception"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)

    args = parser.parse_args()
    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
