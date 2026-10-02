"""Custom Convolutional Neural Network architectures for Facial Emotion Recognition.
No pretrained classification models are used.
"""
import os
from typing import Tuple

# Default to torch backend if not set, for universal Keras 3 compatibility
os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
from keras import layers, models

def build_custom_cnn(
    input_shape: Tuple[int, int, int] = (48, 48, 1),
    num_classes: int = 7,
    dropout_rate: float = 0.25,
    dense_units: int = 256,
) -> keras.Model:
    """
    Constructs a custom Deep Convolutional Neural Network from scratch.
    
    Architecture:
    - Input: 48x48x1 grayscale face crops
    - Block 1: [Conv2D(32) + BatchNorm + Conv2D(32) + BatchNorm + MaxPool(2x2) + Dropout]
    - Block 2: [Conv2D(64) + BatchNorm + Conv2D(64) + BatchNorm + MaxPool(2x2) + Dropout]
    - Block 3: [Conv2D(128) + BatchNorm + Conv2D(128) + BatchNorm + MaxPool(2x2) + Dropout]
    - Block 4: [Conv2D(256) + BatchNorm + MaxPool(2x2) + Dropout]
    - Classification Head: Flatten / GAP + Dense(256) + BatchNorm + Dropout(0.5) + Dense(7, Softmax)
    """
    model = models.Sequential([
        layers.Input(shape=input_shape, name="input_face_image"),

        # Block 1
        layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1_1"),
        layers.BatchNormalization(name="bn1_1"),
        layers.Conv2D(32, (3, 3), padding="same", activation="relu", name="conv1_2"),
        layers.BatchNormalization(name="bn1_2"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool1"),
        layers.Dropout(dropout_rate, name="drop1"),

        # Block 2
        layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2_1"),
        layers.BatchNormalization(name="bn2_1"),
        layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv2_2"),
        layers.BatchNormalization(name="bn2_2"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool2"),
        layers.Dropout(dropout_rate, name="drop2"),

        # Block 3
        layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3_1"),
        layers.BatchNormalization(name="bn3_1"),
        layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv3_2"),
        layers.BatchNormalization(name="bn3_2"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool3"),
        layers.Dropout(dropout_rate, name="drop3"),

        # Block 4
        layers.Conv2D(256, (3, 3), padding="same", activation="relu", name="conv4_1"),
        layers.BatchNormalization(name="bn4_1"),
        layers.MaxPooling2D(pool_size=(2, 2), name="pool4"),
        layers.Dropout(dropout_rate, name="drop4"),

        # Dense Head
        layers.Flatten(name="flatten"),
        layers.Dense(dense_units, activation="relu", name="dense_features"),
        layers.BatchNormalization(name="bn_dense"),
        layers.Dropout(0.5, name="drop_dense"),
        layers.Dense(num_classes, activation="softmax", name="emotion_probabilities"),
    ], name="Custom_FER_DeepCNN")

    return model

def build_residual_compact_cnn(
    input_shape: Tuple[int, int, int] = (48, 48, 1),
    num_classes: int = 7,
) -> keras.Model:
    """
    Constructs a lightweight residual CNN with Depthwise Separable convolutions
    and Global Average Pooling for edge and low-latency inference.
    """
    inputs = layers.Input(shape=input_shape, name="input_face_image")

    # Entry Flow
    x = layers.Conv2D(16, (3, 3), padding="same", use_bias=False, name="entry_conv")(inputs)
    x = layers.BatchNormalization(name="entry_bn")(x)
    x = layers.Activation("relu", name="entry_act")(x)

    # Residual Blocks with SeparableConv2D
    for i, filters in enumerate([32, 64, 128]):
        residual = layers.Conv2D(filters, (1, 1), strides=(2, 2), padding="same", use_bias=False, name=f"res_proj_{i}")(x)
        residual = layers.BatchNormalization(name=f"res_bn_{i}")(residual)

        x = layers.SeparableConv2D(filters, (3, 3), padding="same", use_bias=False, name=f"sep1_{i}")(x)
        x = layers.BatchNormalization(name=f"bn1_{i}")(x)
        x = layers.Activation("relu", name=f"act1_{i}")(x)

        x = layers.SeparableConv2D(filters, (3, 3), padding="same", use_bias=False, name=f"sep2_{i}")(x)
        x = layers.BatchNormalization(name=f"bn2_{i}")(x)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same", name=f"pool_{i}")(x)

        x = layers.add([x, residual], name=f"add_{i}")

    # Output projection with Global Average Pooling
    x = layers.Conv2D(num_classes, (3, 3), padding="same", name="conv_head")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    outputs = layers.Activation("softmax", name="emotion_probabilities")(x)

    return keras.Model(inputs=inputs, outputs=outputs, name="Residual_Compact_FER")

def get_model(
    model_name: str = "custom_cnn",
    input_shape: Tuple[int, int, int] = (48, 48, 1),
    num_classes: int = 7,
    **kwargs,
) -> keras.Model:
    """Factory helper to retrieve model architectures."""
    if model_name.lower() in ["custom_cnn", "baseline", "deep_cnn"]:
        return build_custom_cnn(input_shape=input_shape, num_classes=num_classes, **kwargs)
    elif model_name.lower() in ["residual_compact", "minixception", "compact"]:
        return build_residual_compact_cnn(input_shape=input_shape, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown architecture '{model_name}'. Available: 'custom_cnn', 'residual_compact'")
