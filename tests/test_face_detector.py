"""Unit tests for FaceDetector service."""
import numpy as np
from backend.services.face_detector import FaceDetector

def test_face_detector_instantiation():
    detector = FaceDetector()
    assert detector.cascade is not None

def test_face_detector_blank_image():
    detector = FaceDetector()
    # Create black image (300 x 300)
    blank_image = np.zeros((300, 300, 3), dtype=np.uint8)
    faces = detector.detect_faces(blank_image)
    assert isinstance(faces, list)
    assert len(faces) == 0
