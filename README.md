# Real-Time Facial Emotion Recognition Using Deep Convolutional Neural Networks

[![CI Pipeline](https://github.com/venkatreddy3/EmotionDetect/actions/workflows/ci.yml/badge.svg)](https://github.com/venkatreddy3/EmotionDetect/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-teal.svg)](pyproject.toml)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](backend/)
[![React + Vite](https://img.shields.io/badge/Frontend-React%20%7C%20Vite%20%7C%20TS-61DAFB.svg)](frontend/)

A modular, production-ready deep learning system designed for real-time facial expression analysis and emotion classification. Built as a Deep Learning capstone and production portfolio project.

---

## 🏗️ Architecture Overview

The system features an end-to-end multi-stage pipeline:

```
[ Video / Webcam Stream ]
           │
           ▼
[ Face Detection & Alignment ] ──── OpenCV Frontal Cascade / DNN
           │
           ▼
[ ROI Normalization (48x48) ] ───── Grayscale, CLAHE / Min-Max Scaling
           │
           ▼
[ Deep CNN Classifier ] ────────── Residual Mini-Xception / Custom ConvNet
           │
           ▼
[ Inference API & Client ] ─────── FastAPI Async Service + React 18 / Tailwind Dashboard
```

Detailed architectural diagrams and pipeline trade-offs are documented in [docs/architecture.md](docs/architecture.md).

---

## 📁 Repository Structure

```
├── frontend/                # React, Vite, TypeScript, Tailwind CSS client
│   ├── src/
│   │   ├── components/      # UI components (WebcamStream, ConfidenceBars, etc.)
│   │   ├── services/        # Backend API service integration
│   │   └── types/           # TypeScript interfaces & types
├── backend/                 # FastAPI real-time inference server
│   ├── main.py              # Application entrypoint & routes
│   ├── config.py            # Pydantic settings & paths
│   ├── schemas.py           # Request / Response schemas
│   └── services/            # Face detection & model inference services
├── training/                # Deep learning model development
│   ├── models.py            # Baseline CNN & Mini-Xception architectures
│   ├── dataset.py           # Data loaders, tf.data pipeline & augmentation
│   ├── train.py             # Training loop with callbacks & early stopping
│   ├── evaluate.py          # Metrics, confusion matrix & classification reports
│   └── export.py            # Format conversion (ONNX, TFLite, SavedModel)
├── scripts/                 # Developer commands and automation scripts
├── tests/                   # Pytest suite & integration tests
├── docs/                    # Architecture, dataset, and deployment documentation
├── data/                    # Dataset directory structure (.gitignore excluded)
├── models/                  # Checkpoints and serialized weights (.gitignore excluded)
└── artifacts/               # Evaluation plots, confusion matrices & logs
```

---

## 🎯 Emotion Classification Taxonomy

The model classifies facial regions into the 7 universal emotion categories:
1. **Angry**
2. **Disgust**
3. **Fear**
4. **Happy**
5. **Sad**
6. **Surprise**
7. **Neutral**

---

## 📊 Evaluation & Benchmarking Protocol

> **Note on Evaluation Integrity**: Exact empirical metrics (test accuracy, per-class F1-scores, inference latency) will be populated here after completing the training phase on the target dataset (FER2013). No fabricated numbers are presented.

See [docs/experiments.md](docs/experiments.md) for planned benchmark methodologies and tracking.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Node.js 18+ and npm
- Git

### 1. Backend Setup
```bash
# Create and activate virtual environment
python -m venv .venv
# On Windows:
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements-dev.txt

# Launch FastAPI server
uvicorn backend.main:app --reload --port 8000
```
API Documentation will be available at: `http://localhost:8000/docs`.

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
Client dashboard will be running at: `http://localhost:5173`.

### 3. Running Tests
```bash
pytest tests/ -v
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
