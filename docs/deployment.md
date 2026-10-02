# Deployment Architecture & Strategy

## Production Deployment Targets

The project supports decoupled deployment of both the frontend UI and backend inference engine:

```
[ Client Browser / Mobile ]
         │
         ▼ (HTTPS)
[ Vercel CDN Edge Network ] ─── Hosts React + Vite Single Page Application (SPA)
         │
         ▼ (REST API / WebSocket - JSON / Base64 Frame)
[ Containerized FastAPI Backend ] ─── Hosted on Railway / Render / Fly.io / AWS ECS
         │
         ├── Haar Cascade / Mediapipe Face Detector
         └── ONNX Runtime / Keras FER Model Engine
```

---

## 1. Frontend (Vercel)
- **Framework**: Vite + React SPA.
- **Build Output**: `frontend/dist`.
- **Environment Variables**:
  - `VITE_API_BASE_URL`: Base URL for the FastAPI backend (e.g. `https://api.emotiondetect.example.com`).
- **Configuration**: Standard `vercel.json` rewrite routing for single-page applications.

## 2. Backend (FastAPI Inference Service)
- **Container**: `Dockerfile` utilizing lightweight Python base image.
- **Runtime**: Uvicorn ASGI server with configurable worker processes.
- **Optimization**: Model weights loaded into memory on application startup via FastAPI lifespan handlers.
- **Health Checks**: `/health` endpoint returning server uptime, model load status, and hardware acceleration capabilities.
