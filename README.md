# 🔍 DeepGuard — Deepfake Detection System using Vision Transformer

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-005CED?logo=onnx&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.x-0194E2?logo=mlflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-24.x-2496ED?logo=docker&logoColor=white)
![CI](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF?logo=githubactions&logoColor=white)
![Coverage](https://img.shields.io/badge/Coverage-62%25-brightgreen)
![Tests](https://img.shields.io/badge/Tests-187_passing-success)
![License](https://img.shields.io/badge/License-Apache_2.0-green)

**A production-ready, end-to-end deepfake detection pipeline powered by Vision Transformers (ViT),
served via a high-performance FastAPI backend with Explainable AI, full MLOps observability, and
ONNX-optimized inference.**

[📖 Documentation](docs/) · [🚀 Quick Start](#-getting-started) · [🌐 API Docs](#-api-reference) · [🤝 Contribute](CONTRIBUTING.md)

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Key Features](#-key-features)
- [Tech Stack](#️-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Configuration](#️-configuration)
- [Training](#-training)
- [API Reference](#-api-reference)
- [Explainable AI](#-explainable-ai)
- [MLflow Tracking](#-mlflow-tracking)
- [Docker Deployment](#-docker-deployment)
- [Testing](#-testing)
- [Documentation](#-documentation)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🧠 Overview

**DeepGuard** is a state-of-the-art deepfake detection system that leverages Vision Transformers (ViT)
to classify images and video frames as real or synthesized/manipulated. Built with a clean, layered
architecture and production-grade engineering practices, this system is designed to be:

- **Accurate** — Fine-tuned ViT models trained on FaceForensics++ and DFDC datasets with AUC-ROC > 0.95
- **Fast** — ONNX-exported inference achieving **1.15× speedup** over baseline PyTorch on CPU
- **Explainable** — GradCAM, Attention Rollout, and heatmap overlays for every prediction
- **Observable** — Full MLflow experiment tracking, Prometheus metrics, and Grafana dashboards
- **Scalable** — Dockerized microservices with async FastAPI endpoints and load-tested throughput
- **Maintainable** — Clean Architecture, SOLID principles, 187 tests at 62%+ coverage

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                                │
│           Browser Dashboard │ REST API Client │ CLI Tools          │
└──────────────────────────┬─────────────────────────────────────────┘
                           │ HTTP/REST (Nginx :80)
┌──────────────────────────▼─────────────────────────────────────────┐
│                      API GATEWAY (FastAPI)                          │
│  Rate Limiting │ CORS │ GZip │ Prometheus Metrics │ JWT Auth       │
│  /api/v1/detect  |  /api/v1/health  |  /api/v1/history            │
└─────────┬────────────────┬──────────────────────┬──────────────────┘
          │                │                      │
┌─────────▼───────┐ ┌──────▼──────────┐ ┌────────▼────────────────┐
│ Detection       │ │  Model Registry │ │  History / Analytics    │
│ Service         │ │  Service        │ │  Service                │
│ (ONNX / PyTorch)│ │  (MLflow)       │ │  (SQLite async)         │
└─────────┬───────┘ └─────────────────┘ └─────────────────────────┘
          │
┌─────────▼───────────────────────────────────────────────────────┐
│                    CORE ML PIPELINE                              │
│   Face Extraction │ Preprocessing │ ViT Inference │ XAI Engine  │
│   GradCAM │ Attention Rollout │ Heatmap Overlay │ Explanations  │
└─────────────────────────────────────────────────────────────────┘
          │
┌─────────▼───────────────────────────────────────────────────────┐
│                    STORAGE LAYER                                 │
│   SQLite DB │ Model Weights (.pt / .onnx) │ MLflow Artifacts    │
└─────────────────────────────────────────────────────────────────┘
```

For detailed architecture documentation, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## ✨ Key Features

| Feature | Details |
|---|---|
| 🤖 **Vision Transformer** | ViT-Tiny/Small/Base/Large via `timm`, with LoRA fine-tuning |
| ⚡ **ONNX Inference** | 1.15× speedup on CPU; dynamic batch axes support |
| 🔍 **Explainable AI** | GradCAM, Attention Rollout, heatmap overlays, text explanations |
| 📊 **MLflow Tracking** | Experiment logging, model registry, artifact store |
| 🌐 **Modern Dashboard** | HTML/CSS/JS SPA with dark mode, charts, webcam scanner |
| 🐳 **Full Docker Stack** | API + MLflow + Prometheus + Grafana + Nginx |
| 🔄 **CI/CD Pipeline** | GitHub Actions: lint, test, coverage, Docker build |
| 🧪 **187 Tests** | Unit, integration, E2E; 62%+ coverage |
| 📈 **Load Testing** | Locust-based stress tests for API throughput |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| Deep Learning | PyTorch 2.x, Torchvision, timm |
| Model Architecture | Vision Transformer (ViT-B/16, LoRA) |
| Model Export | ONNX, ONNX Runtime |
| Augmentation | Albumentations |
| Image Processing | OpenCV, Pillow |
| API Framework | FastAPI, Uvicorn, Pydantic v2 |
| Database | SQLite + SQLAlchemy 2.x (async) |
| Experiment Tracking | MLflow |
| Containerization | Docker, Docker Compose |
| Monitoring | Prometheus, Grafana |
| Testing | pytest, pytest-asyncio, pytest-cov |
| Load Testing | Locust |
| Code Quality | black, ruff, mypy, pre-commit |
| CI/CD | GitHub Actions |

---

## 📁 Project Structure

```
deepguard/
├── .github/
│   └── workflows/ci.yml       # GitHub Actions CI/CD pipeline
├── api/                       # FastAPI route handlers and middleware
│   ├── middleware/            # Logging, metrics, rate limiting
│   └── v1/endpoints/          # All REST endpoints
├── backend/
│   └── main.py                # Application factory
├── configs/                   # YAML configuration files
│   ├── model_config.yaml      # ViT architecture + ONNX settings
│   ├── training_config.yaml   # Training hyperparameters
│   └── api_config.yaml        # Server, CORS, rate limit settings
├── core/                      # Domain entities and interfaces
├── database/                  # SQLAlchemy ORM models + session
├── datasets/                  # Dataset loaders (FF++, DFDC, CelebDF)
│   ├── loaders/               # Per-dataset loader classes
│   └── preprocessors/         # Face extraction, augmentation
├── deployment/                # Nginx, Prometheus, Grafana configs
├── docs/                      # All documentation
│   ├── ARCHITECTURE.md
│   ├── API_REFERENCE.md
│   ├── INSTALLATION.md
│   ├── DEPLOYMENT.md
│   ├── USER_MANUAL.md
│   ├── DEVELOPER_MANUAL.md
│   ├── PROJECT_REPORT.md
│   ├── FUTURE_IMPROVEMENTS.md
│   └── slides/presentation.html
├── frontend/                  # Modern HTML/CSS/JS dashboard
│   ├── index.html
│   ├── styles/main.css
│   └── src/                   # JS modules (dashboard, detection, etc.)
├── models/                    # ViT model definitions
│   ├── architectures/vit.py
│   ├── backbones/lora.py
│   └── heads/                 # Classification heads
├── repositories/              # Data access layer
├── schemas/                   # Pydantic request/response schemas
├── scripts/
│   ├── train.py               # Training entry point
│   ├── export_onnx.py         # PyTorch → ONNX converter
│   └── benchmark.py           # Inference latency benchmark
├── services/
│   └── detection/service.py   # Core detection logic (PyTorch + ONNX)
├── tests/
│   ├── unit/                  # 150+ unit tests
│   ├── integration/           # Integration tests
│   ├── e2e/                   # End-to-end tests
│   └── load/locustfile.py     # Locust load testing
├── training/                  # Trainer, callbacks, schedulers
├── utils/
│   └── explainability.py      # GradCAM + Attention Rollout engine
├── weights/                   # Saved model weights
├── .env.example               # Environment variable template
├── CONTRIBUTING.md            # Contribution guidelines
├── docker-compose.yml         # Multi-service Docker composition
├── Dockerfile                 # Multi-stage production image
├── LICENSE                    # Apache 2.0
├── Makefile                   # Developer convenience commands
├── pyproject.toml             # Build system and tool config
└── requirements.txt           # Python dependencies
```

---

## 🚀 Getting Started

For a detailed guide see [docs/INSTALLATION.md](docs/INSTALLATION.md).

### Prerequisites

- Python 3.11+
- Docker 24+ & Docker Compose v2
- Git
- CUDA 11.8+ *(optional, for GPU inference)*

### Quick Start — Local Development

```bash
# 1. Clone the repository
git clone https://github.com/your-org/deepguard.git
cd deepguard

# 2. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # Linux/macOS

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env with your settings

# 5. Start the API server
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# 6. Open interactive docs
# http://localhost:8000/docs
```

### Quick Start — Docker

```bash
# Start all core services
docker-compose up --build

# Services available at:
#   API:        http://localhost:8000
#   API Docs:   http://localhost:8000/docs
#   MLflow UI:  http://localhost:5000
#   Dashboard:  http://localhost:80  (with nginx profile)

# With monitoring (Prometheus + Grafana):
docker-compose --profile monitoring up -d
```

---

## ⚙️ Configuration

All configuration is managed via YAML files in `configs/` and environment variables in `.env`.

| Config File | Purpose |
|---|---|
| `configs/model_config.yaml` | ViT architecture, inference settings, ONNX config |
| `configs/training_config.yaml` | Training loop, optimizer, scheduler settings |
| `configs/api_config.yaml` | FastAPI server, CORS, rate limiting |
| `configs/logging_config.yaml` | Logging handlers and levels |
| `configs/database_config.yaml` | Database connection settings |

**Key environment variables** (see `.env.example` for full list):

```bash
APP_ENV=development              # development | production
DATABASE_URL=sqlite+aiosqlite:///./database/deepguard.db
MODEL_WEIGHTS_PATH=weights/best_model.pt
INFERENCE_USE_ONNX=false         # Set true to use ONNX Runtime
MLFLOW_TRACKING_URI=http://localhost:5000
SECRET_KEY=your-secret-key-here
```

---

## 🏋️ Training

```bash
# 1. Prepare dataset
make prepare-dataset DATASET=ff++

# 2. Run training
make train CONFIG=configs/training_config.yaml

# 3. Evaluate checkpoint
make evaluate CHECKPOINT=weights/best_model.pt

# 4. Export to ONNX
make export-onnx CHECKPOINT=weights/best_model.pt

# 5. Benchmark inference speed
make benchmark
```

Training metrics are tracked automatically in MLflow. Access at `http://localhost:5000`.

---

## 🌐 API Reference

Interactive Swagger docs: `http://localhost:8000/docs`
ReDoc: `http://localhost:8000/redoc`

For full reference, see [docs/API_REFERENCE.md](docs/API_REFERENCE.md).

| Endpoint | Method | Description |
|---|---|---|
| `/api/v1/health` | GET | System health check |
| `/api/v1/detect/image` | POST | Detect deepfake in an image |
| `/api/v1/detect/video` | POST | Detect deepfakes in a video |
| `/api/v1/detect/batch` | POST | Batch image detection (up to 32) |
| `/api/v1/history` | GET | List detection history |
| `/api/v1/history/stats` | GET | Aggregate detection statistics |
| `/api/v1/models` | GET | List registered model versions |
| `/api/v1/models` | POST | Register a new model version |
| `/api/v1/upload` | POST | Stage file upload |
| `/metrics` | GET | Prometheus metrics scrape |

**Quick example:**

```bash
# Detect deepfake in an image
curl -X POST http://localhost:8000/api/v1/detect/image \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_image.jpg"
```

---

## 🔬 Explainable AI

Every prediction includes explainability output:

- **GradCAM**: Class-activation map highlighting regions that influenced the classification
- **Attention Rollout**: Vision Transformer self-attention aggregation across all layers
- **Heatmap Overlay**: Jet colormap blended onto the detected face region
- **Text Explanation**: Natural language description of detected artifacts

The frontend dashboard renders these as interactive tabs in the Detection page.

---

## 📊 MLflow Tracking

```bash
# Start MLflow tracking UI
make mlflow-ui
# Access at http://localhost:5000
```

**Tracked metrics per experiment:**
- Training/validation loss, accuracy, AUC-ROC, F1-score
- Precision, recall, confusion matrix
- Inference latency (ms), throughput (imgs/sec)
- Model artifact: weights, ONNX export, config

---

## 🐳 Docker Deployment

For detailed instructions see [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

```bash
# Production build
make docker-build

# Start all services
make docker-up

# Stop services
make docker-down

# Push to registry
make docker-push REGISTRY=your-registry.io
```

**Services in the stack:**

| Service | Port | Description |
|---|---|---|
| `api` | 8000 | FastAPI backend |
| `mlflow` | 5000 | Experiment tracking UI |
| `prometheus` | 9090 | Metrics collection |
| `grafana` | 3001 | Monitoring dashboards |
| `nginx` | 80/443 | Reverse proxy + static files |

---

## 🧪 Testing

```bash
# Run all 187 tests
make test

# With coverage report
make test-coverage

# Individual suites
make test-unit
make test-integration
make test-e2e

# Load testing (requires running server)
make load-test
```

**Current test results:**
```
187 passed, 14 warnings
Total coverage: 62.84% (threshold: 55% ✅)
```

---

## 📚 Documentation

| Document | Description |
|---|---|
| [docs/INSTALLATION.md](docs/INSTALLATION.md) | Step-by-step installation guide |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | Production deployment guide |
| [docs/API_REFERENCE.md](docs/API_REFERENCE.md) | Full REST API reference |
| [docs/USER_MANUAL.md](docs/USER_MANUAL.md) | End-user dashboard guide |
| [docs/DEVELOPER_MANUAL.md](docs/DEVELOPER_MANUAL.md) | Developer reference |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture diagrams |
| [docs/PROJECT_REPORT.md](docs/PROJECT_REPORT.md) | Academic project report |
| [docs/FUTURE_IMPROVEMENTS.md](docs/FUTURE_IMPROVEMENTS.md) | Development roadmap |
| [docs/slides/presentation.html](docs/slides/presentation.html) | HTML presentation slides |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Contribution guidelines |

---

## 🤝 Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting pull requests.

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
