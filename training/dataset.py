"""Dataset loading, splitting, validation, and metadata generation pipeline."""
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from training.preprocessing import (
    CANONICAL_CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    augment_image,
    normalize_class_name,
    preprocess_image,
    validate_image_file,
)

@dataclass
class DatasetSample:
    file_path: Optional[str]
    class_name: str
    class_idx: int
    split: str  # "train", "val", "test"
    is_valid: bool = True
    error_message: Optional[str] = None

@dataclass
class SplitDistribution:
    total_samples: int
    class_counts: Dict[str, int]
    class_percentages: Dict[str, float]

@dataclass
class DatasetReport:
    dataset_source: str
    total_samples_found: int
    valid_samples: int
    corrupted_samples: int
    splits: Dict[str, SplitDistribution] = field(default_factory=dict)
    label_mapping: Dict[str, int] = field(default_factory=lambda: CLASS_TO_IDX)
    corrupted_files: List[Dict[str, str]] = field(default_factory=list)

def scan_directory_dataset(
    root_dir: Union[str, Path],
    val_split: float = 0.15,
    seed: int = 42,
) -> Tuple[List[DatasetSample], DatasetReport]:
    """
    Scans a folder layout (e.g. data/raw/train and data/raw/test, or data/raw/<class_name>).
    Preserves official train/test splits if present, and stratifies validation split reproducibly.
    """
    root_path = Path(root_dir)
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset root directory does not exist: {root_path}")

    samples: List[DatasetSample] = []
    corrupted_list: List[Dict[str, str]] = []

    # Check for predefined train/test subdirectories
    has_train = (root_path / "train").is_dir()
    has_test = (root_path / "test").is_dir()

    if has_train or has_test:
        sub_splits = []
        if has_train:
            sub_splits.append(("train", root_path / "train"))
        if has_test:
            sub_splits.append(("test", root_path / "test"))

        for split_name, split_dir in sub_splits:
            for item in split_dir.iterdir():
                if item.is_dir():
                    try:
                        canonical_cls = normalize_class_name(item.name)
                        cls_idx = CLASS_TO_IDX[canonical_cls]
                    except ValueError:
                        continue  # Skip non-emotion directories

                    for img_file in item.glob("*.*"):
                        if img_file.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]:
                            is_valid, err = validate_image_file(img_file)
                            if not is_valid:
                                corrupted_list.append({"path": str(img_file), "error": err or "Corrupted"})
                            samples.append(
                                DatasetSample(
                                    file_path=str(img_file),
                                    class_name=canonical_cls,
                                    class_idx=cls_idx,
                                    split=split_name,
                                    is_valid=is_valid,
                                    error_message=err,
                                )
                            )
    else:
        # Flat structure: data/raw/<class_name>/*.png
        for item in root_path.iterdir():
            if item.is_dir():
                try:
                    canonical_cls = normalize_class_name(item.name)
                    cls_idx = CLASS_TO_IDX[canonical_cls]
                except ValueError:
                    continue

                for img_file in item.glob("*.*"):
                    if img_file.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]:
                        is_valid, err = validate_image_file(img_file)
                        if not is_valid:
                            corrupted_list.append({"path": str(img_file), "error": err or "Corrupted"})
                        samples.append(
                            DatasetSample(
                                file_path=str(img_file),
                                class_name=canonical_cls,
                                class_idx=cls_idx,
                                split="train",  # initially all train before stratification
                                is_valid=is_valid,
                                error_message=err,
                            )
                        )

    # If we have a 'train' split and val_split > 0, carve out a reproducible stratified validation set
    valid_train_samples = [s for s in samples if s.split == "train" and s.is_valid]
    if val_split > 0 and len(valid_train_samples) > 0:
        rng = np.random.default_rng(seed)
        # Stratify by class
        for cls_name in CANONICAL_CLASSES:
            cls_samples = [s for s in valid_train_samples if s.class_name == cls_name]
            n_val = int(len(cls_samples) * val_split)
            if n_val > 0:
                val_indices = rng.choice(len(cls_samples), size=n_val, replace=False)
                for idx in val_indices:
                    cls_samples[idx].split = "val"

    # Compute split distributions
    valid_samples = [s for s in samples if s.is_valid]
    splits_summary: Dict[str, SplitDistribution] = {}

    for split in ["train", "val", "test"]:
        split_items = [s for s in valid_samples if s.split == split]
        total = len(split_items)
        if total > 0:
            counts = {cls: sum(1 for s in split_items if s.class_name == cls) for cls in CANONICAL_CLASSES}
            percentages = {cls: round((count / total) * 100, 2) for cls, count in counts.items()}
            splits_summary[split] = SplitDistribution(
                total_samples=total,
                class_counts=counts,
                class_percentages=percentages,
            )

    report = DatasetReport(
        dataset_source=str(root_path),
        total_samples_found=len(samples),
        valid_samples=len(valid_samples),
        corrupted_samples=len(corrupted_list),
        splits=splits_summary,
        label_mapping=CLASS_TO_IDX,
        corrupted_files=corrupted_list,
    )

    return samples, report

def parse_csv_dataset(
    csv_path: Union[str, Path],
    val_split: float = 0.15,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, DatasetReport]:
    """
    Parses original FER2013 CSV format (columns: 'emotion', 'pixels', optional 'Usage').
    Returns (X_train, y_train, X_val, y_val, X_test, y_test, report).
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {path}")

    df = pd.read_csv(path)
    if "emotion" not in df.columns or "pixels" not in df.columns:
        raise ValueError("CSV must contain 'emotion' and 'pixels' columns.")

    # Convert pixel strings to (N, 48, 48, 1) float32 arrays
    pixel_arrays = []
    labels = []
    usages = df["Usage"].tolist() if "Usage" in df.columns else []

    for _, row in df.iterrows():
        raw_pixels = [float(p) for p in str(row["pixels"]).split()]
        if len(raw_pixels) != 48 * 48:
            continue
        arr = np.array(raw_pixels, dtype=np.float32).reshape(48, 48, 1) / 255.0
        pixel_arrays.append(arr)
        labels.append(int(row["emotion"]))

    X_all = np.array(pixel_arrays, dtype=np.float32)
    y_all = np.array(labels, dtype=np.int32)

    if usages:
        train_mask = [u.lower() == "training" for u in usages]
        val_mask = [u.lower() in ["publictest", "validation"] for u in usages]
        test_mask = [u.lower() in ["privatetest", "test"] for u in usages]

        X_train, y_train = X_all[train_mask], y_all[train_mask]
        X_val, y_val = X_all[val_mask], y_all[val_mask]
        X_test, y_test = X_all[test_mask], y_all[test_mask]
    else:
        # Random stratified split
        rng = np.random.default_rng(seed)
        indices = np.arange(len(X_all))
        rng.shuffle(indices)

        n_val = int(len(indices) * val_split)
        n_test = int(len(indices) * val_split)

        val_idx = indices[:n_val]
        test_idx = indices[n_val : n_val + n_test]
        train_idx = indices[n_val + n_test :]

        X_train, y_train = X_all[train_idx], y_all[train_idx]
        X_val, y_val = X_all[val_idx], y_all[val_idx]
        X_test, y_test = X_all[test_idx], y_all[test_idx]

    splits_summary = {}
    for name, split_y in [("train", y_train), ("val", y_val), ("test", y_test)]:
        total = len(split_y)
        if total > 0:
            counts = {IDX_TO_CLASS[i]: int(np.sum(split_y == i)) for i in range(7)}
            percentages = {cls: round((c / total) * 100, 2) for cls, c in counts.items()}
            splits_summary[name] = SplitDistribution(
                total_samples=total,
                class_counts=counts,
                class_percentages=percentages,
            )

    report = DatasetReport(
        dataset_source=str(path),
        total_samples_found=len(X_all),
        valid_samples=len(X_all),
        corrupted_samples=0,
        splits=splits_summary,
        label_mapping=CLASS_TO_IDX,
    )

    return X_train, y_train, X_val, y_val, X_test, y_test, report

def save_dataset_metadata(report: DatasetReport, output_path: Union[str, Path]):
    """Serializes dataset report and label mapping to JSON."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "dataset_source": report.dataset_source,
        "total_samples": report.total_samples_found,
        "valid_samples": report.valid_samples,
        "corrupted_samples": report.corrupted_samples,
        "label_mapping": report.label_mapping,
        "splits": {
            k: {
                "total_samples": v.total_samples,
                "class_counts": v.class_counts,
                "class_percentages": v.class_percentages,
            }
            for k, v in report.splits.items()
        },
        "corrupted_files": report.corrupted_files,
    }

    with open(out_file, "w") as f:
        json.dump(data, f, indent=2)
    return out_file

class InMemoryFERDataset:
    """In-memory dataset container with batched generator and augmentation support."""

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        augment: bool = False,
        seed: int = 42,
    ):
        self.images = images
        self.labels = labels
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.images)

    def get_batch(self, batch_size: int = 32, shuffle: bool = True):
        indices = np.arange(len(self.images))
        if shuffle:
            self.rng.shuffle(indices)

        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i : i + batch_size]
            batch_x = self.images[batch_idx].copy()
            batch_y = self.labels[batch_idx]

            if self.augment:
                batch_x = np.array([augment_image(img, rng=self.rng) for img in batch_x])

            yield batch_x, batch_y
