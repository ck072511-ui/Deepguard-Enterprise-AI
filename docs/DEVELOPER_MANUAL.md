# DeepGuard — Developer Manual

## Overview

This manual is for developers who want to contribute to, extend, or customize the DeepGuard system.

---

## Development Environment Setup

See [INSTALLATION.md](INSTALLATION.md) for full setup instructions. Quick start:

```bash
git clone https://github.com/your-org/deepguard.git
cd deepguard
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
pre-commit install
cp .env.example .env
```

---

## Project Architecture

DeepGuard follows **Clean Architecture** with four concentric layers:

```
┌────────────────────────────────────────────┐
│  Presentation (API endpoints, schemas)     │
├────────────────────────────────────────────┤
│  Application (Services, use cases)         │
├────────────────────────────────────────────┤
│  Domain (Entities, interfaces, exceptions) │
├────────────────────────────────────────────┤
│  Infrastructure (DB, ML models, file I/O)  │
└────────────────────────────────────────────┘
```

Dependencies only point inward. Domain has zero external dependencies.

---

## Module Reference

### `backend/main.py` — Application Factory
```python
# The FastAPI app is created via a factory function
from backend.main import create_application
app = create_application()
```

Middleware is registered in this order (outer → inner):
1. `RequestLoggingMiddleware`
2. `PrometheusMetricsMiddleware`
3. `RateLimitMiddleware`
4. `GZipMiddleware`
5. `TrustedHostMiddleware`
6. `CORSMiddleware`

### `services/detection/service.py` — DetectionService

The central service class:

```python
from services.detection.service import DetectionService

service = DetectionService(db_session)

# Detect deepfake in image bytes
result = await service.detect_image(file_bytes, filename)

# Detect deepfake in video
result = await service.detect_video(video_path, filename)

# Get or initialize the ML model
model = await service.get_model()

# Get ONNX inference session (if enabled)
session = service.get_onnx_session()
```

**Key properties:**
- `service._model`: Cached PyTorch model (None until first inference)
- `service._onnx_session`: Cached ONNX session (None until first ONNX call)
- `service._model_config`: Loaded model config dict

### `utils/explainability.py` — ExplainabilityEngine

```python
from utils.explainability import ExplainabilityEngine

engine = ExplainabilityEngine(model)

# Generate full explanation for a face tensor
result = engine.explain(face_tensor, predicted_class)
# Returns: ExplainabilityResult with gradcam_map, attention_map, overlay images, text

# Individual components
gradcam = engine.generate_gradcam(face_tensor, target_class=1)
attention = engine.generate_attention_rollout(face_tensor)
heatmap = engine.apply_heatmap_overlay(face_np, attention)
text = engine.generate_text_explanation(label, confidence, gradcam)
```

### `models/architectures/vit.py` — ViT Model

```python
from models.factory import ModelFactory
from models.config import ModelConfig

config = ModelConfig(name="vit_base_patch16_224", num_classes=2)
model = ModelFactory.create_model(config)
```

### `datasets/preprocessors/face_extractor.py` — FaceExtractor

```python
from datasets.preprocessors.face_extractor import FaceExtractor

extractor = FaceExtractor(backend="mediapipe")  # or "mtcnn", "retinaface"
faces = extractor.extract_faces(image_np)  # Returns List[np.ndarray]
```

---

## Adding a New API Endpoint

### Step 1: Create Schema

`schemas/requests/my_feature.py`:
```python
from pydantic import BaseModel

class MyRequest(BaseModel):
    input_data: str
```

`schemas/responses/my_feature.py`:
```python
from pydantic import BaseModel

class MyResponse(BaseModel):
    result: str
    confidence: float
```

### Step 2: Create Endpoint

`api/v1/endpoints/my_endpoint.py`:
```python
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from database.session import get_db
from schemas.requests.my_feature import MyRequest
from schemas.responses.my_feature import MyResponse

router = APIRouter(prefix="/my-feature", tags=["My Feature"])

@router.post("/", response_model=MyResponse)
async def my_endpoint(
    request: MyRequest,
    db: AsyncSession = Depends(get_db),
) -> MyResponse:
    # Implementation
    return MyResponse(result="...", confidence=0.9)
```

### Step 3: Register Router

`api/v1/__init__.py`:
```python
from api.v1.endpoints.my_endpoint import router as my_router
v1_router.include_router(my_router)
```

### Step 4: Write Tests

`tests/unit/test_my_endpoint.py`:
```python
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_my_endpoint(client: AsyncClient) -> None:
    response = await client.post("/api/v1/my-feature/", json={"input_data": "test"})
    assert response.status_code == 200
    assert "result" in response.json()
```

---

## Adding a New Model Architecture

### Step 1: Create Model Class

`models/architectures/my_model.py`:
```python
import torch
import torch.nn as nn

class MyCustomModel(nn.Module):
    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        # Define layers
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)
```

### Step 2: Register in Factory

`models/factory.py`:
```python
from models.architectures.my_model import MyCustomModel

class ModelFactory:
    @staticmethod
    def create_model(config: ModelConfig) -> nn.Module:
        if config.name == "my_custom_model":
            return MyCustomModel(num_classes=config.num_classes)
        # ... existing logic
```

### Step 3: Add Config Variant

`configs/model_config.yaml`:
```yaml
variants:
  my_custom:
    name: my_custom_model
    embed_dim: 768
    depth: 6
    num_heads: 8
```

---

## Writing Tests

### Test Directory Structure

```
tests/
├── conftest.py          # Shared fixtures (DB, client, etc.)
├── unit/                # Fast, isolated unit tests
│   ├── test_api.py
│   ├── test_model_pipeline.py
│   └── services/
│       └── test_detection_service.py
├── integration/         # Tests with real DB and service interactions
├── e2e/                 # Full HTTP stack tests
└── load/
    └── locustfile.py    # Load testing scenarios
```

### Shared Fixtures (conftest.py)

```python
# Available fixtures in all tests:
test_database_engine  # Async SQLite engine (in-memory)
db_session            # Clean session per test
client                # AsyncClient for HTTP tests
```

### Running Tests

```bash
# All tests
pytest tests/

# Unit tests only (fastest)
pytest tests/unit/ -v

# With coverage
pytest tests/ --cov=. --cov-report=html

# Single test file
pytest tests/unit/test_api.py -v

# Single test function
pytest tests/unit/test_api.py::test_health_endpoint -v

# With detailed output on failure
pytest tests/ --tb=long -v
```

---

## Code Style

### Formatting

```bash
# Auto-format code
black .
ruff check . --fix

# Check without modifying
black --check .
ruff check .
```

### Type Checking

```bash
mypy . --config-file pyproject.toml
```

### Pre-commit

All checks run automatically on every `git commit`:
- `black` formatting
- `ruff` linting
- `mypy` type checking
- Trailing whitespace removal
- YAML syntax validation
- Conventional commit message format

---

## Configuration System

All configuration uses Pydantic models loaded from YAML files.

### Model Config Access

```python
from models.config import ModelConfig

# Loads from configs/model_config.yaml
config = ModelConfig.from_yaml("configs/model_config.yaml")
print(config.model.name)          # vit_tiny_patch16_224
print(config.inference.use_onnx)  # False
```

### Environment Variable Override

Any config value can be overridden via environment variables. Environment vars take precedence over YAML files.

```bash
INFERENCE_USE_ONNX=true    # Enables ONNX Runtime
INFERENCE_DEVICE=cuda      # Forces GPU inference
MODEL_WEIGHTS_PATH=weights/v2.pt
```

---

## Database

### ORM Models

```python
# database/models.py

class DetectionResultDB(Base):
    __tablename__ = "detection_results"

    id: Mapped[str]           # UUID primary key
    filename: Mapped[str]
    media_type: Mapped[str]   # "image" | "video"
    status: Mapped[str]       # "completed" | "failed"
    label: Mapped[int]        # 0=real, 1=fake
    confidence: Mapped[float]
    faces_count: Mapped[int]
    error_message: Mapped[Optional[str]]
    created_at: Mapped[datetime]
    completed_at: Mapped[Optional[datetime]]
```

### Repository Pattern

```python
from repositories.sqlite.detection import DetectionRepository

repo = DetectionRepository(db_session)

# Create
result = await repo.create(detection_data)

# Read
result = await repo.get_by_id(result_id)
results = await repo.list_all(page=1, page_size=20)

# Delete
await repo.delete(result_id)
```

---

## Logging

```python
import logging
logger = logging.getLogger("deepguard.my_module")

logger.debug("Detailed debug info")
logger.info("Normal operation event")
logger.warning("Something unexpected but recoverable")
logger.error("Error that needs attention")
logger.critical("System cannot continue")
```

Log format: `YYYY-MM-DD HH:MM:SS [LEVEL   ] module — message`

---

## Performance Profiling

```bash
# ONNX vs PyTorch latency benchmark
make benchmark

# Detailed profiling with py-spy
pip install py-spy
py-spy record -o profile.svg -- python -m uvicorn backend.main:app

# Load testing
locust -f tests/load/locustfile.py --host http://localhost:8000 --users 10 --spawn-rate 2 --run-time 60s
```

---

## Common Development Patterns

### Dependency Injection

```python
# Always inject DB session via FastAPI Depends
from database.session import get_db

@router.post("/")
async def endpoint(db: AsyncSession = Depends(get_db)) -> ...:
    service = DetectionService(db)
    ...
```

### Async Patterns

```python
# All DB operations are async
async with db.begin():
    result = await db.execute(select(DetectionResultDB))

# Service methods are async
result = await detection_service.detect_image(bytes_data, filename)
```

### Error Handling

```python
from core.exceptions.api_exceptions import DeepGuardBaseException

# Raise structured exceptions that become JSON responses
raise DeepGuardBaseException(
    code="NO_FACE_DETECTED",
    message="No faces found in the uploaded image",
    status_code=400,
)
```
