"""
DeepGuard — tests/unit/test_database.py

Unit tests for database session, models, and repositories.
"""

from datetime import datetime
from typing import AsyncGenerator
import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from database.models import Base, DetectionResultDB, ModelVersionDB
from repositories.sqlite.detection import SQLiteDetectionRepository
from repositories.sqlite.model import SQLiteModelRepository



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
    from sqlalchemy.ext.asyncio import async_sessionmaker
    from typing import AsyncGenerator
    async_session = async_sessionmaker(
        bind=test_database_engine,
        expire_on_commit=False,
        class_=AsyncSession,
    )
    async with async_session() as session:
        yield session
        await session.rollback()


@pytest.mark.asyncio
async def test_detection_repository_operations(db_session: AsyncSession) -> None:
    """Verify add, retrieve, and paginate operations on DetectionResultDB."""
    repo = SQLiteDetectionRepository(db_session)

    # 1. Create a detection result record
    record = DetectionResultDB(
        id="result-uuid-1234",
        filename="test_face.jpg",
        media_type="image",
        status="completed",
        label=1,
        confidence=0.92,
        faces_count=1,
        meta_info={"detector": "retinaface"},
        completed_at=datetime.utcnow()
    )

    # 2. Add to database
    added = await repo.add(record)
    assert added.id == "result-uuid-1234"
    assert added.filename == "test_face.jpg"
    assert added.confidence == 0.92

    # 3. Retrieve by ID
    retrieved = await repo.get_by_id("result-uuid-1234")
    assert retrieved is not None
    assert retrieved.id == "result-uuid-1234"
    assert retrieved.status == "completed"

    # 4. Paginated retrieval
    all_results = await repo.get_all(limit=10, offset=0)
    assert len(all_results) == 1
    assert all_results[0].id == "result-uuid-1234"


@pytest.mark.asyncio
async def test_model_repository_operations(db_session: AsyncSession) -> None:

    """Verify add, retrieve, list, and activation operations on ModelVersionDB."""
    repo = SQLiteModelRepository(db_session)

    # 1. Register two models (one active, one inactive)
    model1 = ModelVersionDB(
        id="model-uuid-1",
        name="vit_tiny_patch16_224",
        version="1.0.0",
        registry_path="/weights/vit_tiny_1.0.0.pt",
        active=True
    )
    model2 = ModelVersionDB(
        id="model-uuid-2",
        name="vit_small_patch16_224",
        version="2.0.0",
        registry_path="/weights/vit_small_2.0.0.pt",
        active=False
    )

    await repo.add(model1)
    await repo.add(model2)

    # 2. Verify all models are listed
    models = await repo.get_all()
    assert len(models) == 2

    # 3. Verify active model retrieval
    active = await repo.get_active()
    assert active is not None
    assert active.id == "model-uuid-1"

    # 4. Activate model2 and check side effects (model1 should be deactivated)
    updated = await repo.set_active("model-uuid-2")
    assert updated is not None
    assert updated.active is True

    # Refresh session to load updated states
    active = await repo.get_active()
    assert active is not None
    assert active.id == "model-uuid-2"

    retrieved1 = await repo.get_by_id("model-uuid-1")
    assert retrieved1 is not None
    assert retrieved1.active is False

    # Check non-existent activation returns None
    invalid = await repo.set_active("non-existent-id")
    assert invalid is None
