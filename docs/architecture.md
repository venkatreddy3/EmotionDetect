# System Architecture & Design Specification

## Overview

The **Real-Time Facial Emotion Recognition (FER)** system is engineered as a modular, production-grade deep learning solution. The system captures video frames from a client application, detects and normalizes human faces in real time, classifies emotions using a custom deep Convolutional Neural Network (CNN), and renders predictions with confidence scores and low-latency feedback.

```
+-----------------------------------------------------------------------------------+
|                                Client Layer                                       |
|  React 19 + TypeScript + Tailwind CSS (Vite)                                      |
|  - Camera Stream (Webcam / Canvas frame capture @ target 30 FPS)                  |
|  - Real-time Bounding Box & Emotion Confidence Overlay                             |
|  - Minimal Graphite & Muted Teal UI Dashboard                                     |
+----------------------------------------+------------------------------------------+
                                         | HTTP / WebSocket Frame Stream
                                         v
+-----------------------------------------------------------------------------------+
|                                Backend Service                                    |
|  FastAPI + OpenCV + ONNX Runtime / Keras Inference Engine                         |
|  - Frame Ingestion & Validation                                                   |
|  - Face Detection Pipeline (Haar Cascade / Mediapipe / YuNet)                     |
|  - Face Alignment & ROI Preprocessing (48x48 Grayscale / Normalization)           |
|  - Batched / Single-Face CNN Emotion Classifier                                    |
|  - Temporal Smoothing & Probability Distribution Output                           |
+----------------------------------------+------------------------------------------+
                                         ^
                                         | Model Weights / Exported Artifacts (.keras / .onnx)
+----------------------------------------+------------------------------------------+
|                                Training Pipeline                                  |
|  TensorFlow / Keras + scikit-learn + OpenCV                                       |
|  - Dataset Loaders & Augmentation (FER2013 / AffectNet subset)                   |
|  - Custom Deep CNN (Residual / Separable Conv blocks + Dropout + BatchNorm)       |
|  - Class Imbalance Handling (Focal Loss / Class Weights)                          |
|  - Hyperparameter Tuning & Callbacks (EarlyStopping, ReduceLROnPlateau)           |
|  - Evaluation & Benchmarking (Confusion Matrix, F1-Score, Classification Report)  |
+-----------------------------------------------------------------------------------+
```

---

## Component Boundaries

### 1. Training Module (`training/`)
- **Dataset Ingestion**: Automated loaders for FER2013 / custom datasets with validation splits.
- **Preprocessing Pipeline**: Grayscale conversion, contrast normalization (CLAHE / min-max standardizer), data augmentation (rotations, zooms, flips).
- **Model Architectures**:
  - Baseline CNN (4-layer ConvNet with MaxPool and Dropout).
  - Advanced FER-ResNet / Mini-Xception architecture for high-accuracy emotion classification.
- **Export Pipeline**: Exporters for native Keras `.keras`, TensorFlow Lite `.tflite`, and ONNX `.onnx` for high-throughput CPU/GPU inference.

### 2. Backend Service (`backend/`)
- **Web Framework**: FastAPI with asynchronous endpoints.
- **Face Detector**: Multi-engine face detection (OpenCV Haar Cascade baseline, extensible to SSD / Mediapipe / YuNet).
- **Inference Runtime**: Standardized inference wrapper decoupling model format from API contracts.
- **Endpoints**:
  - `GET /health` - Service health and model status check.
  - `POST /api/v1/predict` - Single image face detection + emotion classification.
  - `POST /api/v1/predict/batch` - Batch frame processing.
  - `WebSocket /api/v1/ws/stream` - Real-time low-latency frame streaming.

### 3. Frontend Application (`frontend/`)
- **Framework**: React 19 with TypeScript, powered by Vite.
- **Design System**: Tailored dark mode palette with graphite backgrounds, off-white text, and muted teal accents.
- **Features**:
  - Interactive webcam feed with canvas overlays.
  - Real-time probability bar charts for 7 standard emotions (*Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise*).
  - Performance metrics HUD (FPS, inference latency, detector latency).
  - Snapshot / Image upload analysis mode.

---

## Emotion Taxonomy

The system standardizes on the 7 canonical Ekman universal emotion classes:
0. **Angry**
1. **Disgust**
2. **Fear**
3. **Happy**
4. **Sad**
5. **Surprise**
6. **Neutral**
