"""Unit tests for custom CNN model architectures."""
import os
os.environ.setdefault("KERAS_BACKEND", "torch")
import keras
import numpy as np
import pytest
from training.model import build_custom_cnn, build_residual_compact_cnn, get_model

def test_custom_cnn_shape_and_forward_pass():
    model = build_custom_cnn(input_shape=(48, 48, 1), num_classes=7)
    assert model.input_shape == (None, 48, 48, 1)
    assert model.output_shape == (None, 7)

    # Synthetic batch of 4 images
    dummy_input = np.random.uniform(0.0, 1.0, size=(4, 48, 48, 1)).astype(np.float32)
    output = model(dummy_input)

    assert output.shape == (4, 7)
    # Check probabilities sum to 1
    output_np = keras.ops.convert_to_numpy(output)
    sums = np.sum(output_np, axis=-1)
    np.testing.assert_allclose(sums, [1.0, 1.0, 1.0, 1.0], rtol=1e-4)

def test_residual_compact_cnn_shape_and_forward_pass():
    model = build_residual_compact_cnn(input_shape=(48, 48, 1), num_classes=7)
    assert model.input_shape == (None, 48, 48, 1)
    assert model.output_shape == (None, 7)

    dummy_input = np.random.uniform(0.0, 1.0, size=(2, 48, 48, 1)).astype(np.float32)
    output = model(dummy_input)
    assert output.shape == (2, 7)

def test_model_factory_get_model():
    model1 = get_model("custom_cnn")
    model2 = get_model("residual_compact")
    assert model1.name == "Custom_FER_DeepCNN"
    assert model2.name == "Residual_Compact_FER"

    with pytest.raises(ValueError):
        get_model("unsupported_model")
