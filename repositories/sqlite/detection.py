"""
DeepGuard — repositories/sqlite/detection.py

SQLAlchemy-based SQLite implementation of the IDetectionRepository interface.
"""

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from database.models import DetectionResultDB
from repositories.interfaces.detection import IDetectionRepository


class SQLiteDetectionRepository(IDetectionRepository):
    """SQLite repository for detection results using async SQLAlchemy."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(self, result: DetectionResultDB) -> DetectionResultDB:
        self.session.add(result)
        await self.session.commit()
        await self.session.refresh(result)
        return result

    async def get_by_id(self, result_id: str) -> DetectionResultDB | None:
        return await self.session.get(DetectionResultDB, result_id)

    async def get_all(self, limit: int = 100, offset: int = 0) -> list[DetectionResultDB]:
        stmt = (
            select(DetectionResultDB)
            .order_by(DetectionResultDB.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        res = await self.session.execute(stmt)
        return list(res.scalars().all())
