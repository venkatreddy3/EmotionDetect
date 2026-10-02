"""Smoke and integration tests for training and evaluation pipeline using synthetic fixtures."""
import os
os.environ.setdefault("KERAS_BACKEND", "torch")
from pathlib import Path
import json
import numpy as np
from PIL import Image
import pytest

from training.evaluate import run_evaluation
from training.train import compute_balanced_class_weights, run_training

@pytest.fixture
def synthetic_training_fixture_dir(tmp_path: Path) -> Path:
    """Creates a temporary synthetic mock dataset with train and test folders."""
    root = tmp_path / "synthetic_training_data"
    for split in ["train", "test"]:
        for cls_name in ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]:
            folder = root / split / cls_name
            folder.mkdir(parents=True, exist_ok=True)
            # Create 3 images per class
            for i in range(3):
                img_file = folder / f"img_{i}.png"
                arr = np.random.randint(0, 256, size=(48, 48), dtype=np.uint8)
                Image.fromarray(arr, mode="L").save(img_file)
    return root

def test_compute_balanced_class_weights():
    # Imbalanced synthetic labels (10 zeros, 2 ones)
    labels = np.array([0] * 10 + [1] * 2)
    weights = compute_balanced_class_weights(labels)
    assert 0 in weights and 1 in weights
    assert weights[1] > weights[0]

def test_training_and_evaluation_smoke_run(synthetic_training_fixture_dir: Path, tmp_path: Path):
    """Executes a 2-epoch smoke training run and evaluation on synthetic fixture."""
    chk_dir = tmp_path / "models" / "checkpoints"
    art_dir = tmp_path / "artifacts"

    # 1. Run training (dry_run / 2 epochs)
    train_results = run_training(
        data_dir=str(synthetic_training_fixture_dir),
        checkpoint_dir=str(chk_dir),
        artifacts_dir=str(art_dir),
        model_type="custom_cnn",
        epochs=2,
        batch_size=8,
        learning_rate=0.005,
        val_split=0.2,
        seed=42,
        use_class_weight=True,
        dry_run=True,
    )

    best_model_file = chk_dir / "best_model.keras"
    assert best_model_file.exists()
    assert (art_dir / "reports" / "training_history.json").exists()
    assert (art_dir / "reports" / "training_config.json").exists()
    assert (art_dir / "plots" / "training_curves.png").exists()

    # 2. Run independent evaluation
    eval_results = run_evaluation(
        model_path=str(best_model_file),
        data_dir=str(synthetic_training_fixture_dir),
        artifacts_dir=str(art_dir),
    )

    assert "test_accuracy" in eval_results
    assert "macro_metrics" in eval_results
    assert (art_dir / "reports" / "evaluation_results.json").exists()
    assert (art_dir / "plots" / "confusion_matrix.png").exists()
    assert (art_dir / "plots" / "per_class_f1.png").exists()
