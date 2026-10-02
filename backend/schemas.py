"""Pydantic schemas for API requests and responses."""
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

class BoundingBox(BaseModel):
    x: int = Field(..., description="Top-left x coordinate")
    y: int = Field(..., description="Top-left y coordinate")
    width: int = Field(..., description="Bounding box width")
    height: int = Field(..., description="Bounding box height")

class FaceEmotionPrediction(BaseModel):
    face_id: int
    bounding_box: BoundingBox
    dominant_emotion: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: Dict[str, float]

class PredictionResponse(BaseModel):
    success: bool
    num_faces_detected: int
    predictions: List[FaceEmotionPrediction]
    inference_time_ms: float

class HealthCheckResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    model_path: Optional[str] = None
