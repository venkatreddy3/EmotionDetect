"""CLI wrapper for independent model evaluation."""
import argparse
import sys
from pathlib import Path
from training.evaluate import run_evaluation

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained Facial Emotion Recognition CNN")
    parser.add_argument("--model-path", type=str, default="models/checkpoints/best_model.keras", help="Path to model checkpoint")
    parser.add_argument("--data-dir", type=str, default="data/raw", help="Path to test dataset directory")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts", help="Output directory for evaluation plots and reports")

    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"\n[ERROR] Model file not found at '{args.model_path}'.", file=sys.stderr)
        print("Please train a model first using: python -m scripts.train_model", file=sys.stderr)
        sys.exit(1)

    data_path = Path(args.data_dir)
    if not data_path.exists():
        print(f"\n[ERROR] Data directory not found at '{args.data_dir}'.", file=sys.stderr)
        sys.exit(1)

    try:
        run_evaluation(
            model_path=args.model_path,
            data_dir=args.data_dir,
            artifacts_dir=args.artifacts_dir,
        )
    except Exception as exc:
        print(f"\n[ERROR] Evaluation failed: {str(exc)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
