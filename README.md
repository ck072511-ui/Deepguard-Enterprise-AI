# 🔍 DeepGuard — Deepfake Detection System using Vision Transformer

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.x-0194E2?logo=mlflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-24.x-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/License-Apache_2.0-green)

**A production-ready, end-to-end deepfake detection pipeline powered by Vision Transformers (ViT),  
served via a high-performance FastAPI backend with full MLOps observability.**

</div>

---

## 📌 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Training](#training)
- [API Reference](#api-reference)
- [MLflow Tracking](#mlflow-tracking)
- [Docker Deployment](#docker-deployment)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

---

## 🧠 Overview

**DeepGuard** is a state-of-the-art deepfake detection system that leverages Vision Transformers (ViT)
to classify images and video frames as real or synthesized/manipulated. Built with a clean, layered
architecture and production-grade engineering practices, this system is designed to be:

- **Accurate**: Fine-tuned ViT models trained on FaceForensics++ and DFDC datasets.
- **Fast**: ONNX-exported inference with hardware-accelerated serving.
- **Observable**: Full MLflow experiment tracking, metric logging, and model registry.
- **Scalable**: Dockerized microservices with async FastAPI endpoints.
- **Maintainable**: Clean Architecture, SOLID principles, comprehensive test coverage.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          CLIENT LAYER                            │
│              (Browser / API Consumer / CLI Tools)                │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP/REST
┌──────────────────────────▼──────────────────────────────────────┐
│                        API LAYER (FastAPI)                        │
│         /api/v1/detect  |  /api/v1/health  |  /api/v1/models    │
└──────────┬──────────────┬──────────────────────────┬────────────┘
           │              │                          │
┌──────────▼──────┐ ┌─────▼──────────┐ ┌────────────▼───────────┐
│  Detection Svc  │ │  Model Service │ │   Experiment Service    │
│  (Inference)    │ │  (Registry)    │ │   (MLflow Tracking)     │
└──────────┬──────┘ └─────┬──────────┘ └────────────────────────┘
           │              │
┌──────────▼──────────────▼──────────────────────────────────────┐
│                       CORE LAYER                                 │
│   ViT Model │ Preprocessing Pipeline │ Feature Extraction       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                    REPOSITORY LAYER                              │
│         SQLite DB │ Model Weights Store │ Result Cache           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer              | Technology                                           |
|--------------------|------------------------------------------------------|
| Language           | Python 3.11                                          |
| Deep Learning      | PyTorch 2.x, Torchvision, timm                       |
| Model Architecture | Vision Transformer (ViT-B/16, ViT-L/16)              |
| Augmentation       | Albumentations                                       |
| Image Processing   | OpenCV, Pillow                                       |
| API Framework      | FastAPI, Uvicorn, Pydantic v2                        |
| Database           | SQLite + SQLAlchemy 2.x (async)                      |
| Experiment Track.  | MLflow                                               |
| Model Export       | ONNX, ONNX Runtime                                   |
| Containerization   | Docker, Docker Compose                               |
| Testing            | pytest, pytest-asyncio, pytest-cov                   |
| Code Quality       | black, ruff, mypy, pre-commit                        |
| Data Analysis      | NumPy, Pandas                                        |
| Visualization      | Matplotlib, Seaborn                                  |

---

## 📁 Project Structure

```
deepguard/
├── api/                    # FastAPI route handlers and middleware
├── assets/                 # Static assets (icons, sample images)
├── backend/                # Backend application entry point
├── configs/                # YAML configuration files
├── core/                   # Domain models, use cases, interfaces
├── database/               # SQLAlchemy models and migrations
├── datasets/               # Dataset loaders and preprocessing scripts
├── deployment/             # Kubernetes, Nginx, production configs
├── docs/                   # Technical documentation
├── frontend/               # Web UI (React/HTML)
├── logs/                   # Application and training logs
├── mlflow/                 # MLflow server configuration
├── models/                 # ViT model definitions
├── notebooks/              # Jupyter notebooks for EDA and experiments
├── repositories/           # Data access layer (DB queries)
├── schemas/                # Pydantic request/response schemas
├── scripts/                # Utility scripts (data prep, export)
├── services/               # Business logic services
├── tests/                  # Unit, integration, and E2E tests
├── training/               # Training loops, callbacks, schedulers
├── utils/                  # Shared utilities (logging, metrics)
├── weights/                # Saved model weights (.pt, .onnx)
├── .env.example            # Environment variable template
├── .gitignore              # Git ignore rules
├── .pre-commit-config.yaml # Pre-commit hooks configuration
├── docker-compose.yml      # Multi-service Docker composition
├── Dockerfile              # Production Docker image
├── LICENSE                 # Apache 2.0 License
├── Makefile                # Developer convenience commands
├── pyproject.toml          # Build system and tool configuration
├── README.md               # This file
└── requirements.txt        # Python dependencies
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- Docker 24+ & Docker Compose v2
- CUDA 11.8+ (optional, for GPU inference)
- Git

### Local Development Setup

```bash
# 1. Clone the repository
git clone https://github.com/your-org/deepguard.git
cd deepguard

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# 5. Install pre-commit hooks
pre-commit install

# 6. Run database migrations
make db-migrate

# 7. Start the development server
make dev
```

### Docker Setup

```bash
# Build and start all services
docker-compose up --build

# Services will be available at:
# API:     http://localhost:8000
# MLflow:  http://localhost:5000
# Docs:    http://localhost:8000/docs
```

---

## ⚙️ Configuration

All configuration is managed via YAML files in `configs/` and environment variables in `.env`.

| Config File                    | Purpose                              |
|--------------------------------|--------------------------------------|
| `configs/model_config.yaml`    | ViT architecture and hyperparameters |
| `configs/training_config.yaml` | Training loop settings               |
| `configs/api_config.yaml`      | FastAPI server settings              |
| `configs/logging_config.yaml`  | Logging handlers and levels          |
| `configs/database_config.yaml` | Database connection settings         |

---

## 🏋️ Training

```bash
# Prepare dataset
make prepare-dataset DATASET=ff++

# Run training experiment
make train CONFIG=configs/training_config.yaml

# Evaluate a checkpoint
make evaluate CHECKPOINT=weights/best_model.pt

# Export to ONNX
make export-onnx CHECKPOINT=weights/best_model.pt
```

---

## 🌐 API Reference

Interactive documentation available at `http://localhost:8000/docs`.

| Endpoint                  | Method | Description                        |
|---------------------------|--------|------------------------------------|
| `/api/v1/health`          | GET    | Health check                       |
| `/api/v1/detect/image`    | POST   | Detect deepfake in a single image  |
| `/api/v1/detect/video`    | POST   | Detect deepfakes in a video file   |
| `/api/v1/detect/batch`    | POST   | Batch image detection              |
| `/api/v1/models`          | GET    | List available models              |
| `/api/v1/models/{id}`     | GET    | Get model details and metrics      |
| `/api/v1/experiments`     | GET    | List MLflow experiments            |
| `/api/v1/results/{id}`    | GET    | Get detection result by ID         |

---

## 📊 MLflow Tracking

```bash
# Start MLflow UI
make mlflow-ui

# Access at http://localhost:5000
```

Tracked metrics include: train/val loss, accuracy, AUC-ROC, F1-score, precision, recall, inference latency.

---

## 🐳 Docker Deployment

```bash
# Production build
make docker-build

# Push to registry
make docker-push REGISTRY=your-registry.io

# Deploy with compose
make docker-up
```

---

## 🧪 Testing

```bash
# Run all tests
make test

# Run with coverage
make test-coverage

# Run specific test category
make test-unit
make test-integration
make test-e2e
```

---

## 🤝 Contributing

Please read [CONTRIBUTING.md](docs/CONTRIBUTING.md) before submitting pull requests.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the **Apache License 2.0** — see [LICENSE](LICENSE) for details.

---

<div align="center">
Built with ❤️ by the DeepGuard Team | Portfolio-grade AI Engineering
</div>
