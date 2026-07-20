"""
DeepGuard — tests/unit/services/test_detection_service.py

Unit tests for the DetectionService.
"""

from datetime import datetime
from pathlib import Path
from typing import AsyncGenerator
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest
import torch
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from database.models import Base, DetectionResultDB, ModelVersionDB
from services.detection.service import DetectionService


@pytest.fixture(autouse=True)
async def setup_test_tables(test_database_engine) -> None:
    """Fixture to create and drop tables for each database test session."""
    async with test_database_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_database_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture
async def db_session(test_database_engine) -> AsyncGenerator[AsyncSession, None]:
    """Provide a clean session without an automatic begin block for repository testing."""
    async_session = async_sessionmaker(
        bind=test_database_engine,
        expire_on_commit=False,
        class_=AsyncSession,
    )
    async with async_session() as session:
        yield session
        await session.rollback()


class DummyModel(torch.nn.Module):
    """Dummy PyTorch model for testing inference without loading actual ViT weights."""
    def __init__(self) -> None:
        super().__init__()
        self.param = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns logits of shape (B, 2)
        batch_size = x.shape[0]
        logits = torch.zeros(batch_size, 2)
        logits[:, 1] = 2.0  # high probability of fake class
        return logits


@pytest.fixture
def mock_model_factory() -> MagicMock:
    """Patch ModelFactory to return our DummyModel."""
    with patch("services.detection.service.ModelFactory") as mock:
        dummy_model = DummyModel()
        mock.create_model.return_value = dummy_model
        yield mock


class MockVideoCapture:
    """Mock for cv2.VideoCapture."""
    def __init__(self, is_opened: bool = True, frames: list[np.ndarray] | None = None) -> None:
        self._is_opened = is_opened
        self._frames = frames if frames is not None else [np.zeros((224, 224, 3), dtype=np.uint8)]
        self._idx = 0

    def isOpened(self) -> bool:
        return self._is_opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._idx < len(self._frames):
            frame = self._frames[self._idx]
            self._idx += 1
            return True, frame
        return False, None

    def get(self, propId: int) -> float:
        return 25.0

    def release(self) -> None:
        pass


@pytest.mark.asyncio
async def test_detection_service_get_model(
    db_session: AsyncSession,
    mock_model_factory: MagicMock,
) -> None:
    """Test get_model instantiates and caches the model."""
    service = DetectionService(db_session)

    # 1. Verify get_model returns the model
    model = await service.get_model()
    assert isinstance(model, DummyModel)
    assert service._model is model
    assert service._model_config is not None

    # 2. Verify subsequent calls return the cached model
    model_cached = await service.get_model()
    assert model_cached is model
    mock_model_factory.create_model.assert_called_once()


@pytest.mark.asyncio
async def test_detect_image_success(
    db_session: AsyncSession,
    mock_model_factory: MagicMock,
) -> None:
    """Test successful image detection with mock face extractor and dummy model."""
    service = DetectionService(db_session)

    # Mock extract_faces to return 1 face
    dummy_face = np.zeros((224, 224, 3), dtype=np.uint8)
    with patch.object(service.face_extractor, "extract_faces", return_value=[dummy_face]) as mock_extract:
        # Create fake image bytes
        img = np.ones((10, 10, 3), dtype=np.uint8) * 255
        _, img_encoded = cv2.imencode(".jpg", img)
        file_bytes = img_encoded.tobytes()

        # Execute detection
        record = await service.detect_image(file_bytes, "test_face.jpg")

        assert record.status == "completed"
        assert record.media_type == "image"
        assert record.filename == "test_face.jpg"
        assert record.faces_count == 1
        assert record.label == 1  # DummyModel outputs fake probability ~0.88 which is >= threshold 0.5
        assert record.confidence > 0.5
        assert record.error_message is None
        assert record.completed_at is not None

        mock_extract.assert_called_once()


@pytest.mark.asyncio
async def test_detect_image_failure_invalid_bytes(
    db_session: AsyncSession,
) -> None:
    """Test detect_image exception handling with empty/invalid bytes."""
    service = DetectionService(db_session)

    record = await service.detect_image(b"invalid_bytes_here", "test_invalid.jpg")

    assert record.status == "failed"
    assert "Failed to decode image bytes" in record.error_message
    assert record.completed_at is not None


@pytest.mark.asyncio
async def test_detect_video_success(
    db_session: AsyncSession,
    mock_model_factory: MagicMock,
) -> None:
    """Test successful video detection using mock video capture."""
    service = DetectionService(db_session)

    # Mock extract_faces to return a face
    dummy_face = np.zeros((224, 224, 3), dtype=np.uint8)
    mock_video_frames = [np.ones((224, 224, 3), dtype=np.uint8) * 128]

    with patch.object(service.face_extractor, "extract_faces", return_value=[dummy_face]), \
         patch("services.detection.service.cv2.VideoCapture", return_value=MockVideoCapture(is_opened=True, frames=mock_video_frames)):
        
        record = await service.detect_video(Path("fake_video.mp4"), "fake_video.mp4")

        assert record.status == "completed"
        assert record.media_type == "video"
        assert record.filename == "fake_video.mp4"
        assert record.faces_count == 1
        assert record.label == 1
        assert record.confidence > 0.5
        assert record.error_message is None
        assert record.completed_at is not None


@pytest.mark.asyncio
async def test_detect_video_failure_cannot_open(
    db_session: AsyncSession,
) -> None:
    """Test detect_video failure when video cannot be opened by cv2."""
    service = DetectionService(db_session)

    with patch("services.detection.service.cv2.VideoCapture", return_value=MockVideoCapture(is_opened=False)):
        record = await service.detect_video(Path("bad_video.mp4"), "bad_video.mp4")

        assert record.status == "failed"
        assert "Failed to open video file stream" in record.error_message


@pytest.mark.asyncio
async def test_detect_video_failure_no_faces(
    db_session: AsyncSession,
) -> None:
    """Test detect_video failure when no frames/faces are extracted."""
    service = DetectionService(db_session)

    with patch.object(service.face_extractor, "extract_faces", return_value=[]), \
         patch("services.detection.service.cv2.VideoCapture", return_value=MockVideoCapture(is_opened=True, frames=[])):
        
        record = await service.detect_video(Path("empty_video.mp4"), "empty_video.mp4")

        assert record.status == "failed"
        assert "No frames or faces were extracted from the video" in record.error_message


@pytest.mark.asyncio
async def test_onnx_inference_route(
    db_session: AsyncSession,
) -> None:
    """Test that detection service routes correctly through ONNX session when config is set."""
    service = DetectionService(db_session)
    
    # Mock get_onnx_session to return a mock session
    mock_onnx_session = MagicMock()
    mock_input = MagicMock()
    mock_input.name = "input"
    mock_onnx_session.get_inputs.return_value = [mock_input]
    # Make session.run return a numpy array of logits representing high probability FAKE
    mock_onnx_session.run.return_value = [np.array([[0.0, 2.0]], dtype=np.float32)]
    
    with patch.object(service, "get_onnx_session", return_value=mock_onnx_session) as mock_get_onnx:
        dummy_face = np.zeros((224, 224, 3), dtype=np.uint8)
        with patch.object(service.face_extractor, "extract_faces", return_value=[dummy_face]):
            img = np.ones((10, 10, 3), dtype=np.uint8) * 255
            _, img_encoded = cv2.imencode(".jpg", img)
            file_bytes = img_encoded.tobytes()

            record = await service.detect_image(file_bytes, "test_onnx.jpg")

            assert record.status == "completed"
            assert record.label == 1
            assert record.confidence > 0.5
            mock_get_onnx.assert_called_once()
            mock_onnx_session.run.assert_called_once()

