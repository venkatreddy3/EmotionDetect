"""Convolutional Neural Network architectures for Facial Emotion Recognition."""
from typing import Tuple

def build_baseline_cnn(
    input_shape: Tuple[int, int, int] = (48, 48, 1),
    num_classes: int = 7
):
    """
    Builds a 4-block standard Convolutional Neural Network baseline.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    model = models.Sequential([
        # Block 1
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 4
        layers.Conv2D(256, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Classification Head
        layers.Flatten(),
        layers.Dense(512, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax", name="emotion_probabilities"),
    ], name="FER_Baseline_CNN")

    return model

def build_mini_xception(
    input_shape: Tuple[int, int, int] = (48, 48, 1),
    num_classes: int = 7
):
    """
    Builds a residual Mini-Xception architecture with depthwise separable convolutions.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    inputs = layers.Input(shape=input_shape)

    # Initial convolution
    x = layers.Conv2D(8, (3, 3), strides=(1, 1), use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(8, (3, 3), strides=(1, 1), use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # Residual Blocks
    for filters in [16, 32, 64, 128]:
        residual = layers.Conv2D(filters, (1, 1), strides=(2, 2), padding="same", use_bias=False)(x)
        residual = layers.BatchNormalization()(residual)

        x = layers.SeparableConv2D(filters, (3, 3), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, (3, 3), padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

        x = layers.add([x, residual])

    # Global Average Pooling and Softmax
    x = layers.Conv2D(num_classes, (3, 3), padding="same")(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Activation("softmax", name="emotion_probabilities")(x)

    return models.Model(inputs=inputs, outputs=outputs, name="FER_Mini_Xception")
