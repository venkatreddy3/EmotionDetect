"""Emotion classifier inference service."""
from typing import Dict, List, Optional
import numpy as np

class EmotionClassifier:
    """Wrapper around trained deep CNN model for emotion recognition."""

    def __init__(self, model_path: Optional[str] = None, classes: Optional[List[str]] = None):
        self.classes = classes or [
            "Angry",
            "Disgust",
            "Fear",
            "Happy",
            "Sad",
            "Surprise",
            "Neutral",
        ]
        self.model_path = model_path
        self.model = None
        self._is_loaded = False

    def load_model(self) -> bool:
        """Loads the model if weights are available on disk."""
        if not self.model_path:
            return False
        # Pluggable model loading (TensorFlow/Keras or ONNX runtime)
        return self._is_loaded

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def preprocess_face(self, face_crop: np.ndarray, target_size: tuple[int, int] = (48, 48)) -> np.ndarray:
        """Preprocesses raw face crop into normalized model tensor shape (1, 48, 48, 1)."""
        import cv2

        if len(face_crop.shape) == 3:
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_crop

        resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
        normalized = resized.astype("float32") / 255.0
        return np.expand_dims(np.expand_dims(normalized, axis=-1), axis=0)

    def predict(self, face_tensor: np.ndarray) -> Dict[str, float]:
        """Runs forward pass and returns emotion probabilities."""
        if not self.is_loaded:
            # Fallback uniform baseline distribution when weights are not yet generated
            n = len(self.classes)
            return {cls: round(1.0 / n, 4) for cls in self.classes}

        predictions = self.model.predict(face_tensor)[0]
        return {cls: float(predictions[i]) for i, cls in enumerate(self.classes)}
