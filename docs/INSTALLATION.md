# DeepGuard — Installation Guide

## System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| OS | Windows 10, Ubuntu 20.04, macOS 12 | Ubuntu 22.04 LTS |
| Python | 3.11 | 3.11.x |
| RAM | 8 GB | 16 GB |
| Disk | 10 GB free | 50 GB free |
| CPU | 4 cores | 8+ cores |
| GPU *(optional)* | NVIDIA GTX 1060 | NVIDIA RTX 3080+ |
| CUDA *(GPU only)* | 11.8 | 12.1 |
| Docker | 24.0 | 25.0 |

---

## Option A: Local Python Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/your-org/deepguard.git
cd deepguard
```

### Step 2: Create Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install PyTorch

**CPU-only (works on any machine):**
```bash
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cpu
```

**GPU (CUDA 11.8):**
```bash
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
```

**GPU (CUDA 12.1):**
```bash
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```

### Step 4: Install All Dependencies

```bash
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
```

### Step 5: System Dependencies

**Ubuntu / Debian:**
```bash
sudo apt-get update
sudo apt-get install -y ffmpeg libsm6 libxext6 libgl1-mesa-glx libglib2.0-0
```

**macOS (Homebrew):**
```bash
brew install ffmpeg
```

**Windows:**
- Download FFmpeg from https://ffmpeg.org/download.html
- Add FFmpeg `bin/` directory to your system PATH

### Step 6: Configure Environment

```bash
# Copy the example configuration
cp .env.example .env
```

Edit `.env` and set the required values:

```bash
# Application
APP_ENV=development
SECRET_KEY=your-secret-key-change-this-in-production

# Database
DATABASE_URL=sqlite+aiosqlite:///./database/deepguard.db

# Model
MODEL_WEIGHTS_PATH=weights/best_model.pt
INFERENCE_USE_ONNX=false

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000
```

### Step 7: Install Pre-commit Hooks (Optional)

```bash
pip install pre-commit
pre-commit install
pre-commit install --hook-type commit-msg
```

### Step 8: Initialize Database

The database is automatically initialized on first API startup. You can also initialize manually:

```bash
python -c "
import asyncio
from database import Base, engine
asyncio.run(engine.begin().__aenter__().__class__.run_sync(lambda conn: Base.metadata.create_all(conn)))
"
```

### Step 9: Start the API Server

```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Visit:
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/v1/health

---

## Option B: Docker Installation (Recommended for Production)

### Prerequisites
- Docker 24+ installed: https://docs.docker.com/get-docker/
- Docker Compose v2: included with Docker Desktop

### Step 1: Clone and Configure

```bash
git clone https://github.com/your-org/deepguard.git
cd deepguard
cp .env.example .env
# Edit .env as needed
```

### Step 2: Start Core Services

```bash
# Build and start API + MLflow
docker-compose up --build -d

# Check service status
docker-compose ps

# View API logs
docker-compose logs -f api
```

### Step 3: Access Services

| Service | URL |
|---|---|
| API Docs (Swagger) | http://localhost:8000/docs |
| Health Check | http://localhost:8000/api/v1/health |
| MLflow UI | http://localhost:5000 |

### Step 4: Start Monitoring (Optional)

```bash
# Add Prometheus + Grafana
docker-compose --profile monitoring up -d

# Access:
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3001  (admin / deepguard123)
```

---

## Option C: Using Make Commands

```bash
# Full setup (creates venv + installs deps + pre-commit)
make setup

# Or just install deps into current environment
make install

# Start development server
make dev

# Run all tests
make test
```

---

## Verifying the Installation

### Quick Health Check

```bash
curl http://localhost:8000/api/v1/health
```

Expected response:
```json
{"status": "healthy", "version": "1.0.0"}
```

### Test Image Detection

```bash
# Using a sample image
curl -X POST http://localhost:8000/api/v1/detect/image \
  -F "file=@assets/sample_face.jpg"
```

### Run Test Suite

```bash
python -m pytest tests/ -v
```

Expected: **187 tests passed**

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'cv2'`
```bash
pip install opencv-python-headless
```

### `RuntimeError: CUDA error: no kernel image is available`
Your PyTorch CUDA version doesn't match your GPU driver. Reinstall PyTorch with the correct CUDA version or use CPU mode:
```bash
# Force CPU mode
echo "INFERENCE_DEVICE=cpu" >> .env
```

### `OSError: [WinError 193] %1 is not a valid Win32 application`
This is usually a dlib / face detection binary mismatch. Reinstall:
```bash
pip uninstall dlib
pip install dlib
```

### Port 8000 Already in Use
```bash
# Use a different port
python -m uvicorn backend.main:app --port 8080 --reload
```

### Docker Build Fails on `pip install`
This is usually a network timeout. Try:
```bash
docker-compose build --no-cache api
```

### Database is Locked (SQLite)
Only one process can write to SQLite at a time. If running multiple worker processes:
```bash
# Single-process development mode
uvicorn backend.main:app --workers 1 --reload
```
