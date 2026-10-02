"""Model export and serialization utilities (ONNX, TFLite, SavedModel)."""
import argparse
from pathlib import Path

def export_to_tflite(model_path: str, output_path: str):
    """Converts a Keras model to TensorFlow Lite format."""
    import tensorflow as tf

    print(f"Loading Keras model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "wb") as f:
        f.write(tflite_model)
    print(f"Exported TFLite model to {out_file} ({len(tflite_model) / 1024:.2f} KB)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export trained model")
    parser.add_argument("--model-path", type=str, default="models/checkpoints/best_model.keras")
    parser.add_argument("--output-path", type=str, default="models/exported/model.tflite")
    args = parser.parse_args()

    export_to_tflite(args.model_path, args.output_path)
