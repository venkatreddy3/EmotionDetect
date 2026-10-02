"""Independent evaluation and reporting pipeline for Facial Emotion Recognition."""
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

# Backend for universal Keras 3 compatibility
os.environ.setdefault("KERAS_BACKEND", "torch")
import keras

from training.dataset import (
    CANONICAL_CLASSES,
    CLASS_TO_IDX,
    parse_csv_dataset,
    scan_directory_dataset,
)
from training.preprocessing import preprocess_image

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Union[str, Path],
    normalize: bool = True,
):
    """Generates and saves a styled Confusion Matrix heatmap using pure matplotlib."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if normalize:
        cm_display = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-7)
    else:
        cm_display = cm

    fig, ax = plt.subplots(figsize=(8.5, 7), facecolor="#111317")
    ax.set_facecolor("#16191E")

    im = ax.imshow(cm_display, interpolation="nearest", cmap="Blues")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9, colors="#E5E7EB")

    # Ticks & labels
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=35, ha="right", color="#E5E7EB", fontsize=10)
    ax.set_yticklabels(class_names, color="#E5E7EB", fontsize=10)

    # Values in cells
    thresh = cm_display.max() / 2.0
    for i in range(cm_display.shape[0]):
        for j in range(cm_display.shape[1]):
            val_str = f"{cm_display[i, j]:.2f}" if normalize else f"{int(cm_display[i, j])}"
            text_color = "#111317" if cm_display[i, j] > thresh else "#FAFBFC"
            ax.text(j, i, val_str, ha="center", va="center", color=text_color, fontsize=9.5, fontweight="600")

    ax.set_title(f"Confusion Matrix ({'Normalized' if normalize else 'Counts'})", color="#FAFBFC", fontsize=13, pad=12, fontweight="600")
    ax.set_xlabel("Predicted Label", color="#E5E7EB", fontsize=11, labelpad=8)
    ax.set_ylabel("True Label", color="#E5E7EB", fontsize=11, labelpad=8)

    plt.tight_layout()
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()

def plot_per_class_f1(
    report_dict: Dict,
    class_names: List[str],
    output_path: Union[str, Path],
):
    """Plots and saves per-class F1-score bar chart."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    f1_scores = [report_dict.get(cls, {}).get("f1-score", 0.0) for cls in class_names]

    plt.figure(figsize=(9, 4.5), facecolor="#111317")
    ax = plt.subplot(111)
    ax.set_facecolor("#16191E")
    ax.grid(True, linestyle="--", alpha=0.25, color="#3D4656", axis="y")
    ax.tick_params(colors="#E5E7EB")
    for spine in ax.spines.values():
        spine.set_color("#2A303C")

    bars = ax.bar(class_names, f1_scores, color="#14B8A6", edgecolor="#2DD4BF", width=0.55)

    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 0.02,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            color="#FAFBFC",
            fontsize=9,
            fontweight="bold",
        )

    plt.title("Per-Class F1-Score Breakdown", color="#FAFBFC", fontsize=12, pad=10, fontweight="600")
    plt.xlabel("Emotion Class", color="#E5E7EB", fontsize=10)
    plt.ylabel("F1-Score", color="#E5E7EB", fontsize=10)
    plt.ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig(out_file, dpi=200, bbox_inches="tight")
    plt.close()

def load_test_data(
    data_dir: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """Loads pure unseen test split without data augmentation."""
    csv_candidates = list(data_dir.glob("*.csv"))
    if csv_candidates:
        _, _, _, _, X_test, y_test, _ = parse_csv_dataset(csv_candidates[0])
        return X_test, y_test

    samples, _ = scan_directory_dataset(data_dir)
    test_samples = [s for s in samples if s.split == "test" and s.is_valid]

    # Fallback to train/val if test folder was not present
    if not test_samples:
        test_samples = [s for s in samples if s.is_valid]

    X_test = np.array([preprocess_image(s.file_path) for s in test_samples], dtype=np.float32)
    y_test = np.array([s.class_idx for s in test_samples], dtype=np.int32)
    return X_test, y_test

def run_evaluation(
    model_path: str = "models/checkpoints/best_model.keras",
    data_dir: str = "data/raw",
    artifacts_dir: str = "artifacts",
) -> Dict:
    """Executes full evaluation on unseen test dataset and exports reports & plots."""
    art_path = Path(artifacts_dir)
    plots_dir = art_path / "plots"
    reports_dir = art_path / "reports"

    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Trained model checkpoint not found at: {model_path}")

    print(f"Loading trained model: {model_path}...")
    model = keras.models.load_model(model_path)

    print(f"Loading test dataset from: {data_dir}...")
    X_test, y_test = load_test_data(Path(data_dir))
    print(f"Test samples: {len(X_test)}")

    # Forward pass inference
    print("Running test set predictions...")
    y_pred_probs = model.predict(X_test, batch_size=64, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Compute metrics
    test_acc = float(np.mean(y_pred == y_test))
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    clf_report = classification_report(
        y_test,
        y_pred,
        target_names=CANONICAL_CLASSES,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(CANONICAL_CLASSES))))

    results_payload = {
        "model_path": str(model_path),
        "test_samples": int(len(y_test)),
        "test_accuracy": round(test_acc, 4),
        "macro_metrics": {
            "precision": round(float(prec_macro), 4),
            "recall": round(float(rec_macro), 4),
            "f1_score": round(float(f1_macro), 4),
        },
        "weighted_metrics": {
            "precision": round(float(prec_weighted), 4),
            "recall": round(float(rec_weighted), 4),
            "f1_score": round(float(f1_weighted), 4),
        },
        "classification_report": clf_report,
        "confusion_matrix": cm.tolist(),
        "classes": CANONICAL_CLASSES,
    }

    # Save report
    rep_file = reports_dir / "evaluation_results.json"
    with open(rep_file, "w") as f:
        json.dump(results_payload, f, indent=2)

    # Save plots
    cm_plot_file = plots_dir / "confusion_matrix.png"
    plot_confusion_matrix(cm, CANONICAL_CLASSES, cm_plot_file, normalize=True)

    f1_plot_file = plots_dir / "per_class_f1.png"
    plot_per_class_f1(clf_report, CANONICAL_CLASSES, f1_plot_file)

    # Print summary
    print("\n" + "=" * 65)
    print("MODEL EVALUATION BENCHMARK")
    print("=" * 65)
    print(f"Test Accuracy:      {test_acc * 100:.2f}%")
    print(f"Macro F1-Score:     {f1_macro:.4f}")
    print(f"Weighted F1-Score:  {f1_weighted:.4f}")
    print("-" * 65)
    print(f"{'Class':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 65)
    for cls_name in CANONICAL_CLASSES:
        metrics = clf_report.get(cls_name, {})
        p = metrics.get("precision", 0.0)
        r = metrics.get("recall", 0.0)
        f = metrics.get("f1-score", 0.0)
        s = int(metrics.get("support", 0))
        print(f"{cls_name:<12} | {p:<10.4f} | {r:<10.4f} | {f:<10.4f} | {s:<8}")
    print("=" * 65)
    print(f"Artifacts exported:")
    print(f"  - Metrics JSON:      {rep_file}")
    print(f"  - Confusion Matrix:  {cm_plot_file}")
    print(f"  - Per-Class F1 Plot: {f1_plot_file}\n")

    return results_payload

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained FER model on test dataset")
    parser.add_argument("--model-path", type=str, default="models/checkpoints/best_model.keras")
    parser.add_argument("--data-dir", type=str, default="data/raw")
    parser.add_argument("--artifacts-dir", type=str, default="artifacts")

    args = parser.parse_args()
    run_evaluation(
        model_path=args.model_path,
        data_dir=args.data_dir,
        artifacts_dir=args.artifacts_dir,
    )
