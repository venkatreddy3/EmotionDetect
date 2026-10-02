"""Data preparation CLI orchestrator."""
import argparse
from pathlib import Path
import sys
import numpy as np

from training.dataset import (
    CANONICAL_CLASSES,
    parse_csv_dataset,
    save_dataset_metadata,
    scan_directory_dataset,
)
from training.preprocessing import preprocess_image

def format_distribution_table(report) -> str:
    """Formats a clean terminal table of split distributions."""
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("DATASET DISTRIBUTION REPORT")
    lines.append("=" * 70)
    lines.append(f"Source: {report.dataset_source}")
    lines.append(f"Total Samples: {report.total_samples_found} | Valid: {report.valid_samples} | Corrupted: {report.corrupted_samples}")
    lines.append("-" * 70)

    header = f"{'Emotion Class':<12} | " + " | ".join([f"{s.upper():<14}" for s in report.splits.keys()])
    lines.append(header)
    lines.append("-" * 70)

    for cls_name in CANONICAL_CLASSES:
        row_parts = [f"{cls_name:<12}"]
        for split_name, dist in report.splits.items():
            cnt = dist.class_counts.get(cls_name, 0)
            pct = dist.class_percentages.get(cls_name, 0.0)
            row_parts.append(f"{cnt:>6} ({pct:>5.1f}%)")
        lines.append(" | ".join(row_parts))

    lines.append("-" * 70)
    total_parts = [f"{'TOTAL':<12}"]
    for split_name, dist in report.splits.items():
        total_parts.append(f"{dist.total_samples:>6} (100.0%)")
    lines.append(" | ".join(total_parts))
    lines.append("=" * 70 + "\n")

    return "\n".join(lines)

def run_preparation(
    raw_dir: str,
    output_dir: str,
    report_dir: str,
    val_split: float,
    seed: int,
    save_npz: bool = False,
):
    raw_path = Path(raw_dir)
    out_path = Path(output_dir)
    rep_path = Path(report_dir)

    out_path.mkdir(parents=True, exist_ok=True)
    rep_path.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        print(f"\n[ERROR] Raw data directory not found: {raw_path}", file=sys.stderr)
        print("Please download and place the FER-2013 dataset into data/raw/ as described in data/README.md\n", file=sys.stderr)
        return False

    # Check for CSV or directory layout
    csv_candidates = list(raw_path.glob("*.csv"))
    has_subdirs = any(d.is_dir() for d in raw_path.iterdir()) if raw_path.exists() else False

    if csv_candidates and not has_subdirs:
        csv_file = csv_candidates[0]
        print(f"Detected CSV dataset: {csv_file}")
        X_train, y_train, X_val, y_val, X_test, y_test, report = parse_csv_dataset(
            csv_file, val_split=val_split, seed=seed
        )
        if save_npz:
            print("Saving preprocessed arrays to .npz files...")
            np.savez_compressed(out_path / "train_arrays.npz", images=X_train, labels=y_train)
            np.savez_compressed(out_path / "val_arrays.npz", images=X_val, labels=y_val)
            np.savez_compressed(out_path / "test_arrays.npz", images=X_test, labels=y_test)
    elif has_subdirs:
        print(f"Scanning directory dataset structure in {raw_path}...")
        samples, report = scan_directory_dataset(raw_path, val_split=val_split, seed=seed)

        if report.valid_samples == 0:
            print(f"\n[WARNING] No valid image files found in {raw_path}.", file=sys.stderr)
            print("Expected structure: data/raw/train/<emotion>/*.png and data/raw/test/<emotion>/*.png\n", file=sys.stderr)
            return False

        if save_npz:
            print("Extracting and preprocessing images into .npz caches...")
            for split in ["train", "val", "test"]:
                split_samples = [s for s in samples if s.split == split and s.is_valid]
                if not split_samples:
                    continue
                imgs = np.array([preprocess_image(s.file_path) for s in split_samples], dtype=np.float32)
                lbls = np.array([s.class_idx for s in split_samples], dtype=np.int32)
                np.savez_compressed(out_path / f"{split}_arrays.npz", images=imgs, labels=lbls)
                print(f"  -> Saved {split}_arrays.npz ({len(imgs)} samples)")
    else:
        print(f"\n[ERROR] No valid dataset files (train/test folders or .csv) found in {raw_path}.", file=sys.stderr)
        print("Please check data/README.md for download instructions.\n", file=sys.stderr)
        return False

    # Display report
    print(format_distribution_table(report))

    # Save metadata
    meta_path1 = save_dataset_metadata(report, out_path / "dataset_metadata.json")
    meta_path2 = save_dataset_metadata(report, rep_path / "dataset_metadata.json")
    print(f"Dataset metadata successfully saved to:")
    print(f"  - {meta_path1}")
    print(f"  - {meta_path2}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare, validate, and summarize FER-2013 dataset")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Path to raw dataset directory")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Path to save processed metadata and arrays")
    parser.add_argument("--report-dir", type=str, default="artifacts/reports", help="Path to save distribution report")
    parser.add_argument("--val-split", type=float, default=0.15, help="Validation split proportion")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible split")
    parser.add_argument("--save-npz", action="store_true", help="Cache preprocessed images into compressed .npz files")

    args = parser.parse_args()
    success = run_preparation(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        report_dir=args.report_dir,
        val_split=args.val_split,
        seed=args.seed,
        save_npz=args.save_npz,
    )
    sys.exit(0 if success else 1)
