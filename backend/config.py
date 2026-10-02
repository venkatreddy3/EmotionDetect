"""Backend configuration settings."""
from pathlib import Path
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseModel):
    APP_NAME: str = "Facial Emotion Recognition API"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Model configuration
    MODEL_PATH: Path = BASE_DIR / "models" / "checkpoints" / "best_model.keras"
    CASCADE_PATH: Path = BASE_DIR / "backend" / "assets" / "haarcascade_frontalface_default.xml"
    EMOTION_CLASSES: list[str] = [
        "Angry",
        "Disgust",
        "Fear",
        "Happy",
        "Sad",
        "Surprise",
        "Neutral",
    ]
    IMAGE_SIZE: tuple[int, int] = (48, 48)

    # CORS settings
    CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ]

settings = Settings()
