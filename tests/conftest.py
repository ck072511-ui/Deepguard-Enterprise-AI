"""
DeepGuard — pytest conftest.py

Provides shared fixtures, test utilities, and configuration
for the entire test suite. Loaded automatically by pytest.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# ---------------------------------------------------------------------------
# pytest configuration
# ---------------------------------------------------------------------------

def pytest_configure(config: pytest.Config) -> None:
    """Register custom pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests (fast, no I/O)")
    config.addinivalue_line("markers", "integration: Integration tests (requires DB/filesystem)")
    config.addinivalue_line("markers", "e2e: End-to-end tests (requires running server)")
    config.addinivalue_line("markers", "slow: Tests that take more than 5 seconds")
    config.addinivalue_line("markers", "gpu: Tests that require a CUDA GPU")


# ---------------------------------------------------------------------------
# Event Loop
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def event_loop_policy() -> asyncio.DefaultEventLoopPolicy:
    """Use the default event loop policy for the session."""
    return asyncio.DefaultEventLoopPolicy()


# ---------------------------------------------------------------------------
# Database Fixtures
# ---------------------------------------------------------------------------

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture(scope="session")
def test_database_engine():
    """Create an in-memory SQLite engine for the test session."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        echo=False,
        future=True,
    )
    yield engine
    # Engine cleanup happens automatically with in-memory DB


@pytest.fixture
async def test_db_session(test_database_engine) -> AsyncGenerator[AsyncSession, None]:
    """Provide an async database session for tests, rolled back after each test."""
    async_session = async_sessionmaker(
        bind=test_database_engine,
        expire_on_commit=False,
        class_=AsyncSession,
    )
    async with async_session() as session:
        async with session.begin():
            yield session
            await session.rollback()


# ---------------------------------------------------------------------------
# File System Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory that is automatically cleaned up."""
    with tempfile.TemporaryDirectory(prefix="deepguard_test_") as tmp:
        yield Path(tmp)


@pytest.fixture
def sample_image_path(temp_dir: Path) -> Path:
    """Create a minimal test JPEG image file."""
    import cv2

    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img_path = temp_dir / "test_image.jpg"
    cv2.imwrite(str(img_path), img)
    return img_path


@pytest.fixture
def sample_image_rgb() -> np.ndarray:
    """Return a random RGB numpy image array (224x224x3)."""
    rng = np.random.default_rng(seed=42)
    return rng.integers(0, 255, (224, 224, 3), dtype=np.uint8)


@pytest.fixture
def sample_batch_images() -> np.ndarray:
    """Return a batch of 4 random RGB numpy image arrays."""
    rng = np.random.default_rng(seed=42)
    return rng.integers(0, 255, (4, 224, 224, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Environment Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def set_test_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force test environment variables for all tests."""
    monkeypatch.setenv("APP_ENV", "testing")
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv("DATABASE_URL", TEST_DATABASE_URL)
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("SECRET_KEY", "test-secret-key-minimum-32-characters-long")
    monkeypatch.setenv("JWT_SECRET_KEY", "test-jwt-secret-key-minimum-32-characters")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:///tmp/mlruns_test")


# ---------------------------------------------------------------------------
# Mock Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_model() -> MagicMock:
    """Return a mocked PyTorch model that returns deterministic predictions."""
    import torch

    model = MagicMock()
    model.eval.return_value = model
    model.return_value = torch.tensor([[0.2, 0.8]])  # Fake prediction
    return model


@pytest.fixture
def mock_mlflow_client() -> Generator[MagicMock, None, None]:
    """Patch MLflow client to prevent actual tracking during tests."""
    with patch("mlflow.start_run"), patch("mlflow.log_metric"), patch(
        "mlflow.log_params"
    ), patch("mlflow.log_artifact"), patch("mlflow.set_experiment"):
        yield MagicMock()


# ---------------------------------------------------------------------------
# API Client Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def test_client() -> Generator[TestClient, None, None]:
    """Synchronous FastAPI test client."""
    # Import lazily to avoid circular imports during collection
    from backend.main import create_application

    app = create_application()
    with TestClient(app, raise_server_exceptions=True) as client:
        yield client


@pytest_asyncio.fixture
async def async_test_client() -> AsyncGenerator[AsyncClient, None]:
    """Async FastAPI test client for async endpoint tests."""
    from backend.main import create_application

    app = create_application()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
