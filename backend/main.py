"""FastAPI application entrypoint for Facial Emotion Recognition."""
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io

from backend.config import settings
from backend.schemas import (
    BoundingBox,
    FaceEmotionPrediction,
    HealthCheckResponse,
    PredictionResponse,
)
from backend.services.emotion_classifier import EmotionClassifier
from backend.services.face_detector import FaceDetector

detector = FaceDetector()
classifier = EmotionClassifier(model_path=str(settings.MODEL_PATH), classes=settings.EMOTION_CLASSES)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Lifecycle startup
    classifier.load_model()
    yield
    # Lifecycle shutdown cleanup if needed

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="High-performance real-time facial emotion recognition API.",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check():
    """Health check and model readiness probe."""
    return HealthCheckResponse(
        status="healthy",
        version=settings.APP_VERSION,
        model_loaded=classifier.is_loaded,
        model_path=str(settings.MODEL_PATH) if classifier.is_loaded else None,
    )

@app.post("/api/v1/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_emotions(file: UploadFile = File(...)):
    """
    Accepts an uploaded image frame, detects faces, and performs emotion classification.
    """
    start_time = time.perf_counter()

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image format.")

    try:
        image_bytes = await file.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(pil_image)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {str(exc)}")

    # Detect faces
    faces = detector.detect_faces(image_np)
    predictions: list[FaceEmotionPrediction] = []

    for idx, (x, y, w, h) in enumerate(faces):
        # Extract face crop
        face_crop = image_np[y : y + h, x : x + w]
        face_tensor = classifier.preprocess_face(face_crop, target_size=settings.IMAGE_SIZE)
        probabilities = classifier.predict(face_tensor)

        dominant_emotion = max(probabilities, key=probabilities.get)
        confidence = probabilities[dominant_emotion]

        predictions.append(
            FaceEmotionPrediction(
                face_id=idx + 1,
                bounding_box=BoundingBox(x=x, y=y, width=w, height=h),
                dominant_emotion=dominant_emotion,
                confidence=confidence,
                probabilities=probabilities,
            )
        )

    latency_ms = (time.perf_counter() - start_time) * 1000.0

    return PredictionResponse(
        success=True,
        num_faces_detected=len(predictions),
        predictions=predictions,
        inference_time_ms=round(latency_ms, 2),
    )
