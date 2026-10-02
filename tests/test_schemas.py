"""Unit tests for Pydantic API schemas."""
from backend.schemas import BoundingBox, FaceEmotionPrediction, PredictionResponse

def test_bounding_box_valid():
    bbox = BoundingBox(x=10, y=20, width=100, height=120)
    assert bbox.x == 10
    assert bbox.y == 20
    assert bbox.width == 100
    assert bbox.height == 120

def test_prediction_response_valid():
    bbox = BoundingBox(x=0, y=0, width=50, height=50)
    pred = FaceEmotionPrediction(
        face_id=1,
        bounding_box=bbox,
        dominant_emotion="Happy",
        confidence=0.95,
        probabilities={"Happy": 0.95, "Neutral": 0.05}
    )
    response = PredictionResponse(
        success=True,
        num_faces_detected=1,
        predictions=[pred],
        inference_time_ms=12.4
    )
    assert response.success is True
    assert response.num_faces_detected == 1
    assert response.predictions[0].dominant_emotion == "Happy"
    assert response.predictions[0].confidence == 0.95
