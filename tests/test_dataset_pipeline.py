"""Unit tests for dataset acquisition, preprocessing, and preparation pipeline.
Uses synthetic temporary fixtures only.
"""
from pathlib import Path
import json
import numpy as np
from PIL import Image
import pytest

from training.dataset import (
    CANONICAL_CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    InMemoryFERDataset,
    parse_csv_dataset,
    save_dataset_metadata,
    scan_directory_dataset,
)
from training.preprocessing import (
    augment_image,
    normalize_class_name,
    preprocess_image,
    validate_image_file,
)

# ---------------------------------------------------------------------------
# Fixtures for Small Temporary Synthetic Datasets
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_mock_image_path(tmp_path: Path) -> Path:
    """Creates a single valid synthetic test image fixture."""
    img_path = tmp_path / "synthetic_test_face.png"
    # Create a 64x64 random synthetic grayscale image
    arr = np.random.randint(0, 256, size=(64, 64), dtype=np.uint8)
    Image.fromarray(arr, mode="L").save(img_path)
    return img_path

@pytest.fixture
def synthetic_mock_dataset_dir(tmp_path: Path) -> Path:
    """
    Creates a temporary synthetic mock dataset with standard train/test folders
    and one corrupted file to verify validation handling.
    """
    root = tmp_path / "synthetic_mock_dataset"
    for split in ["train", "test"]:
        for cls_name in ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]:
            folder = root / split / cls_name
            folder.mkdir(parents=True, exist_ok=True)
            # Create 4 synthetic images per class
            for i in range(4):
                img_file = folder / f"sample_{i}.png"
                arr = np.random.randint(0, 256, size=(48, 48), dtype=np.uint8)
                Image.fromarray(arr, mode="L").save(img_file)

    # Inject a corrupted (0-byte) file and a garbage text file
    corrupt_file = root / "train" / "angry" / "corrupted_0byte.png"
    corrupt_file.write_bytes(b"")

    corrupt_text_file = root / "train" / "happy" / "corrupted_text.jpg"
    corrupt_text_file.write_text("NOT_A_VALID_IMAGE_CONTENT")

    return root

@pytest.fixture
def synthetic_mock_csv_path(tmp_path: Path) -> Path:
    """Creates a small synthetic CSV fixture resembling FER format."""
    csv_file = tmp_path / "synthetic_mock_dataset.csv"
    lines = ["emotion,pixels,Usage"]
    # 7 classes x 2 samples = 14 samples
    usages = ["Training", "PublicTest"]
    for emo_idx in range(7):
        for usage in usages:
            # 48x48 = 2304 pixels
            pixels_str = " ".join(str(np.random.randint(0, 256)) for _ in range(2304))
            lines.append(f"{emo_idx},{pixels_str},{usage}")

    csv_file.write_text("\n".join(lines))
    return csv_file

# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

def test_canonical_class_taxonomy():
    """Validates the 7 canonical classes and indexing order."""
    expected = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
    assert CANONICAL_CLASSES == expected
    assert len(CANONICAL_CLASSES) == 7
    for idx, name in enumerate(expected):
        assert CLASS_TO_IDX[name] == idx
        assert IDX_TO_CLASS[idx] == name

def test_normalize_class_name():
    """Validates case-insensitive canonical class normalization."""
    assert normalize_class_name("angry") == "Angry"
    assert normalize_class_name("HAPPY") == "Happy"
    assert normalize_class_name("Neutral") == "Neutral"
    assert normalize_class_name("  surprise  ") == "Surprise"

    with pytest.raises(ValueError):
        normalize_class_name("unknown_emotion")

def test_preprocess_image_normalization(synthetic_mock_image_path: Path):
    """Verifies that preprocessing outputs (48, 48, 1) float32 in range [0, 1]."""
    processed = preprocess_image(synthetic_mock_image_path, target_size=(48, 48), normalize=True)
    assert processed.shape == (48, 48, 1)
    assert processed.dtype == np.float32
    assert processed.min() >= 0.0
    assert processed.max() <= 1.0

def test_preprocess_image_numpy_rgb_conversion():
    """Verifies preprocessing can take 3-channel RGB numpy arrays and convert to grayscale."""
    rgb_arr = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
    processed = preprocess_image(rgb_arr, target_size=(48, 48), normalize=True)
    assert processed.shape == (48, 48, 1)
    assert processed.dtype == np.float32

def test_data_augmentation_properties():
    """Verifies training augmentation preserves shape and value range [0, 1]."""
    img = np.random.uniform(0.0, 1.0, size=(48, 48, 1)).astype(np.float32)
    aug = augment_image(img)
    assert aug.shape == (48, 48, 1)
    assert aug.dtype == np.float32
    assert aug.min() >= 0.0
    assert aug.max() <= 1.0

def test_validate_image_file_corruption_detection(tmp_path: Path):
    """Verifies that corrupted and non-image files are detected as invalid."""
    # 0-byte file
    empty_file = tmp_path / "empty.png"
    empty_file.write_bytes(b"")
    is_valid, err = validate_image_file(empty_file)
    assert is_valid is False
    assert "empty" in err.lower()

    # Garbage file
    garbage_file = tmp_path / "garbage.jpg"
    garbage_file.write_text("THIS IS NOT A VALID JPEG")
    is_valid, err = validate_image_file(garbage_file)
    assert is_valid is False

def test_scan_synthetic_mock_dataset(synthetic_mock_dataset_dir: Path, tmp_path: Path):
    """Verifies full directory scanning, validation split creation, and corrupted file skipping."""
    samples, report = scan_directory_dataset(
        synthetic_mock_dataset_dir,
        val_split=0.25,
        seed=42
    )

    assert report.total_samples_found == 58  # (7*4 in train + 2 corrupt) + (7*4 in test) = 30 + 28 = 58
    assert report.corrupted_samples == 2
    assert report.valid_samples == 56

    assert "train" in report.splits
    assert "val" in report.splits
    assert "test" in report.splits

    assert report.splits["test"].total_samples == 28

    # Verify metadata JSON serialization
    meta_path = tmp_path / "metadata.json"
    save_dataset_metadata(report, meta_path)
    assert meta_path.exists()

    with open(meta_path, "r") as f:
        meta_content = json.load(f)

    assert meta_content["valid_samples"] == 56
    assert meta_content["corrupted_samples"] == 2
    assert "Angry" in meta_content["label_mapping"]

def test_parse_csv_synthetic_dataset(synthetic_mock_csv_path: Path):
    """Verifies CSV parsing, pixel array extraction, and split separation."""
    X_train, y_train, X_val, y_val, X_test, y_test, report = parse_csv_dataset(
        synthetic_mock_csv_path, val_split=0.5, seed=42
    )

    assert len(X_train) == 7
    assert len(X_val) == 7
    assert X_train.shape[1:] == (48, 48, 1)
    assert X_train.dtype == np.float32
    assert y_train.shape == (7,)
    assert report.valid_samples == 14

def test_in_memory_dataset_batch_generator():
    """Verifies InMemoryFERDataset batching and augmentation."""
    images = np.random.uniform(0.0, 1.0, size=(20, 48, 48, 1)).astype(np.float32)
    labels = np.random.randint(0, 7, size=(20,), dtype=np.int32)

    dataset = InMemoryFERDataset(images, labels, augment=True, seed=42)
    assert len(dataset) == 20

    batches = list(dataset.get_batch(batch_size=8, shuffle=True))
    assert len(batches) == 3  # 8 + 8 + 4 = 20
    assert batches[0][0].shape == (8, 48, 48, 1)
    assert batches[0][1].shape == (8,)
    assert batches[2][0].shape == (4, 48, 48, 1)
