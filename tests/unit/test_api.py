"""
DeepGuard — tests/unit/test_api.py

Unit tests for FastAPI REST endpoints (health, models, detect) and middleware.
"""

import pytest
import io
from fastapi.testclient import TestClient
from database import Base, engine
from database.models import ModelVersionDB, DetectionResultDB


@pytest.fixture(autouse=True)
async def setup_api_database() -> None:
    """Ensure database tables exist on the app's default engine during API tests."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


def test_api_health_endpoint(test_client: TestClient) -> None:
    """Verify that the GET /health endpoint returns correct structure and status."""
    response = test_client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "database" in data
    assert "model" in data
    assert "mlflow" in data
    assert data["database"] == "connected"


def test_api_models_workflow(test_client: TestClient) -> None:
    """Test registering, listing, and activating model versions."""
    # 1. List models (should be empty initially)
    response = test_client.get("/api/v1/models")
    assert response.status_code == 200
    assert len(response.json()) == 0

    # 2. Register a new model
    payload = {
        "name": "vit_tiny_patch16_224",
        "version": "1.0.0",
        "registry_path": "/weights/vit_tiny_1.0.0.pt"
    }
    response = test_client.post("/api/v1/models", json=payload)
    assert response.status_code == 201
    model_data = response.json()
    assert model_data["name"] == "vit_tiny_patch16_224"
    assert model_data["active"] is False
    model_id = model_data["id"]

    # 3. Activate the registered model
    response = test_client.post(f"/api/v1/models/{model_id}/activate")
    assert response.status_code == 200
    activated_data = response.json()
    assert activated_data["active"] is True

    # 4. Confirm it is active in list
    response = test_client.get("/api/v1/models")
    assert response.status_code == 200
    models_list = response.json()
    assert len(models_list) == 1
    assert models_list[0]["active"] is True


def test_api_detect_image(test_client: TestClient) -> None:
    """Verify image upload and mock deepfake detection response."""
    # Create mock JPEG file in memory
    import numpy as np
    import cv2

    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    _, img_encoded = cv2.imencode(".jpg", img)
    img_bytes = img_encoded.tobytes()

    files = {"file": ("test_face.jpg", io.BytesIO(img_bytes), "image/jpeg")}
    response = test_client.post("/api/v1/detect", files=files)
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "test_face.jpg"
    assert data["media_type"] == "image"
    assert data["status"] == "completed"
    assert data["label"] in [0, 1]
    assert data["confidence"] is not None


def test_api_detect_unsupported_file(test_client: TestClient) -> None:
    """Verify that uploading unsupported file types returns HTTP 400."""
    files = {"file": ("test.txt", io.BytesIO(b"hello world"), "text/plain")}
    response = test_client.post("/api/v1/detect", files=files)
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["error"]["message"]


def test_rate_limiting_middleware(test_client: TestClient) -> None:
    """Verify rate-limiting middleware triggers HTTP 429 when threshold exceeded."""
    # We send multiple requests in quick succession to trigger the default rate limit of 60 req/min
    # For testing, we can simply call /health inside a loop.
    # Note: test_client shares IP, so history is shared.
    triggered = False
    for _ in range(70):
        response = test_client.get("/api/v1/health")
        if response.status_code == 429:
            triggered = True
            assert "Rate limit exceeded" in response.text
            break
    assert triggered, "Rate limiting was not triggered after 70 requests."
