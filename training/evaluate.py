"""Model evaluation and metrics report generator."""
import argparse
from pathlib import Path
import json

def evaluate_model(
    model_path: str,
    test_data_dir: str,
    output_dir: str = "artifacts/reports"
):
    """
    Evaluates trained model on test dataset and generates metrics, confusion matrix, and reports.
    """
    import numpy as np
    from sklearn.metrics import classification_report, confusion_matrix
    import tensorflow as tf

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading trained model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_data_dir,
        image_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        shuffle=False,
    )
    rescaling = tf.keras.layers.Rescaling(1.0 / 255)
    test_ds = test_ds.map(lambda x, y: (rescaling(x), y))

    y_true = []
    y_pred = []

    print("Running evaluation inference...")
    for images, labels in test_ds:
        preds = model.predict(images)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())

    class_names = test_ds.class_names
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    matrix = confusion_matrix(y_true, y_pred).tolist()

    report_payload = {
        "model_path": model_path,
        "classification_report": report,
        "confusion_matrix": matrix,
        "class_names": class_names,
    }

    report_file = out_path / "evaluation_report.json"
    with open(report_file, "w") as f:
        json.dump(report_payload, f, indent=2)

    print(f"Evaluation report successfully saved to {report_file}")
    return report_payload

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Facial Emotion Recognition Model")
    parser.add_argument("--model-path", type=str, default="models/checkpoints/best_model.keras")
    parser.add_argument("--test-data-dir", type=str, default="data/raw/test")
    parser.add_argument("--output-dir", type=str, default="artifacts/reports")
    args = parser.parse_args()

    evaluate_model(args.model_path, args.test_data_dir, args.output_dir)
