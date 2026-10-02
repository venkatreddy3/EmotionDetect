"""CLI wrapper for reproducible model training."""
import argparse
import sys
from pathlib import Path
from training.train import run_training

def main():
    parser = argparse.ArgumentParser(description="Train Facial Emotion Recognition CNN")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Path to raw dataset directory")
    parser.add_argument("--checkpoint-dir", type=str, default="models/checkpoints", help="Output directory for model checkpoints")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="Output directory for plots and reports")
    parser.add_argument("--model-type", type=str, default="custom_cnn", choices=["custom_cnn", "residual_compact"])
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial Adam learning rate")
    parser.add_argument("--val-split", type=float, default=0.15, help="Validation split proportion")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--no-class-weight", action="store_true", help="Disable class weighting")
    parser.add_argument("--dry-run", action="store_true", help="Run quick 2-epoch smoke test with small batch")

    args = parser.parse_args()

    data_path = Path(args.data_dir)
    if not data_path.exists() or not any(data_path.iterdir()):
        print(f"\n[ERROR] Dataset not found in '{args.data_dir}'.", file=sys.stderr)
        print("Please follow data/README.md instructions to download FER-2013 before running training.", file=sys.stderr)
        sys.exit(1)

    try:
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
    except Exception as exc:
        print(f"\n[ERROR] Training failed: {str(exc)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
