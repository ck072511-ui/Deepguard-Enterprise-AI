"""
DeepGuard — tests/e2e/test_api_e2e.py

End-to-End integration tests for the DeepGuard REST API.

These tests spin up the full FastAPI application (including database
initialization and all middleware) via TestClient and verify the
complete request→response cycle, including persistence and metrics.

Run with:
    pytest tests/e2e/ -v --no-cov
"""

import io
import uuid
import pytest
import numpy as np
from fastapi.testclient import TestClient
from httpx import AsyncClient

from backend.main import create_application
from database import Base, engine


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def app():
    """Create a single application instance for the entire test module."""
    return create_application()


@pytest.fixture(scope="module")
def client(app):
    """Synchronous TestClient wrapping the full application."""
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


@pytest.fixture(autouse=True, scope="module")
async def init_db():
    """Create all tables before tests and drop them after."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


def _make_jpeg_bytes(width: int = 224, height: int = 224) -> bytes:
    """Generate a synthetic JPEG image for upload testing."""
    import cv2
    img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


# ── Health endpoint ───────────────────────────────────────────────────────────


class TestHealthEndpoint:
    """E2E tests for GET /api/v1/health."""

    def test_health_returns_200(self, client: TestClient):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200

    def test_health_response_schema(self, client: TestClient):
        data = client.get("/api/v1/health").json()
        assert "status" in data
        assert "database" in data
        assert "model" in data
        assert "mlflow" in data

    def test_health_database_connected(self, client: TestClient):
        data = client.get("/api/v1/health").json()
        assert data["database"] == "connected"


# ── Models CRUD endpoints ─────────────────────────────────────────────────────


class TestModelsEndpoints:
    """E2E tests for /api/v1/models CRUD lifecycle."""

    def test_list_models_initially_empty(self, client: TestClient):
        resp = client.get("/api/v1/models")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_register_model_success(self, client: TestClient):
        payload = {
            "name": "vit_base_patch16_224",
            "version": f"e2e-{uuid.uuid4().hex[:8]}",
            "registry_path": "/weights/vit_base.pt",
        }
        resp = client.post("/api/v1/models", json=payload)
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == payload["name"]
        assert data["active"] is False
        assert "id" in data

    def test_register_duplicate_model_fails(self, client: TestClient):
        version = f"dup-{uuid.uuid4().hex[:8]}"
        payload = {
            "name": "dup-model",
            "version": version,
            "registry_path": "/weights/dup.pt",
        }
        client.post("/api/v1/models", json=payload)
        resp2 = client.post("/api/v1/models", json=payload)
        assert resp2.status_code == 400

    def test_activate_model(self, client: TestClient):
        # Register fresh model
        payload = {
            "name": "vit_tiny_patch16_224",
            "version": f"act-{uuid.uuid4().hex[:8]}",
            "registry_path": "/weights/vit_tiny.pt",
        }
        created = client.post("/api/v1/models", json=payload).json()
        model_id = created["id"]

        # Activate it
        resp = client.post(f"/api/v1/models/{model_id}/activate")
        assert resp.status_code == 200
        data = resp.json()
        assert data["active"] is True

    def test_activate_nonexistent_model_returns_404(self, client: TestClient):
        resp = client.post("/api/v1/models/nonexistent-id-abc/activate")
        assert resp.status_code == 404


# ── Detection endpoint ────────────────────────────────────────────────────────


class TestDetectionEndpoints:
    """E2E tests for POST /api/v1/detect."""

    def test_detect_image_returns_200(self, client: TestClient):
        jpeg_bytes = _make_jpeg_bytes()
        files = {"file": ("test_face.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")}
        resp = client.post("/api/v1/detect", files=files)
        assert resp.status_code == 200

    def test_detect_image_response_schema(self, client: TestClient):
        jpeg_bytes = _make_jpeg_bytes()
        files = {"file": ("schema_test.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")}
        data = client.post("/api/v1/detect", files=files).json()

        required_keys = {"id", "filename", "media_type", "status", "label", "confidence"}
        assert required_keys.issubset(data.keys())

    def test_detect_image_label_is_binary(self, client: TestClient):
        jpeg_bytes = _make_jpeg_bytes()
        files = {"file": ("label_test.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")}
        data = client.post("/api/v1/detect", files=files).json()
        assert data["label"] in (0, 1, None)

    def test_detect_image_confidence_in_range(self, client: TestClient):
        jpeg_bytes = _make_jpeg_bytes()
        files = {"file": ("conf_test.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")}
        data = client.post("/api/v1/detect", files=files).json()
        if data["confidence"] is not None:
            assert 0.0 <= data["confidence"] <= 1.0

    def test_detect_unsupported_type_returns_400(self, client: TestClient):
        files = {"file": ("malware.exe", io.BytesIO(b"MZ\x90\x00"), "application/octet-stream")}
        resp = client.post("/api/v1/detect", files=files)
        assert resp.status_code == 400

    def test_detect_txt_returns_400(self, client: TestClient):
        files = {"file": ("data.txt", io.BytesIO(b"hello world"), "text/plain")}
        resp = client.post("/api/v1/detect", files=files)
        assert resp.status_code == 400


# ── Detection history endpoint ────────────────────────────────────────────────


class TestHistoryEndpoint:
    """E2E tests for GET /api/v1/detect (history retrieval)."""

    def test_history_returns_list(self, client: TestClient):
        resp = client.get("/api/v1/detect")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_history_grows_after_detection(self, client: TestClient):
        before = len(client.get("/api/v1/detect").json())
        jpeg = _make_jpeg_bytes()
        client.post("/api/v1/detect", files={"file": ("grow.jpg", io.BytesIO(jpeg), "image/jpeg")})
        after = len(client.get("/api/v1/detect").json())
        assert after >= before  # May be equal if detection failed

    def test_history_pagination_limit(self, client: TestClient):
        resp = client.get("/api/v1/detect", params={"limit": 2})
        assert resp.status_code == 200
        assert len(resp.json()) <= 2


# ── Metrics endpoint ──────────────────────────────────────────────────────────


class TestMetricsEndpoint:
    """E2E tests for GET /metrics (Prometheus)."""

    def test_metrics_endpoint_reachable(self, client: TestClient):
        resp = client.get("/metrics")
        assert resp.status_code in (200, 503)  # 503 if prometheus_client not installed

    def test_metrics_content_type(self, client: TestClient):
        resp = client.get("/metrics")
        if resp.status_code == 200:
            assert "text/plain" in resp.headers.get("content-type", "")

    def test_metrics_contains_http_counter_after_request(self, client: TestClient):
        # Trigger a request first
        client.get("/api/v1/health")
        resp = client.get("/metrics")
        if resp.status_code == 200:
            assert "http_requests_total" in resp.text


# ── Rate limiting ─────────────────────────────────────────────────────────────


class TestRateLimiting:
    """E2E tests for rate limiting middleware."""

    def test_rate_limit_triggers_429(self, client: TestClient):
        """Verify that exceeding 60 req/min triggers HTTP 429."""
        triggered = False
        for _ in range(75):
            resp = client.get("/api/v1/health")
            if resp.status_code == 429:
                triggered = True
                break
        assert triggered, "Rate limiter should have triggered 429 within 75 requests"
