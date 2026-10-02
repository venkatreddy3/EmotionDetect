"""Integration tests for FastAPI endpoints."""
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
import io
from backend.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "model_loaded" in data

def test_predict_endpoint_valid_image():
    # Create simple RGB test image
    img = Image.new("RGB", (100, 100), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/api/v1/predict",
        files={"file": ("test.jpg", buf, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "num_faces_detected" in data
    assert "inference_time_ms" in data
    assert isinstance(data["predictions"], list)

def test_predict_endpoint_invalid_file_type():
    response = client.post(
        "/api/v1/predict",
        files={"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")}
    )
    assert response.status_code == 400
