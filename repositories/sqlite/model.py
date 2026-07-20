"""
DeepGuard — repositories/sqlite/model.py

SQLAlchemy-based SQLite implementation of the IModelRepository interface.
"""

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from database.models import ModelVersionDB
from repositories.interfaces.model import IModelRepository


class SQLiteModelRepository(IModelRepository):
    """SQLite repository for model versions using async SQLAlchemy."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def add(self, model: ModelVersionDB) -> ModelVersionDB:
        self.session.add(model)
        await self.session.commit()
        await self.session.refresh(model)
        return model

    async def get_by_id(self, model_id: str) -> ModelVersionDB | None:
        return await self.session.get(ModelVersionDB, model_id)

    async def get_active(self) -> ModelVersionDB | None:
        stmt = select(ModelVersionDB).where(ModelVersionDB.active == True).limit(1)
        res = await self.session.execute(stmt)
        return res.scalars().first()

    async def set_active(self, model_id: str) -> ModelVersionDB | None:
        # Verify model exists
        model = await self.get_by_id(model_id)
        if not model:
            return None

        # Deactivate all models
        stmt_deactivate = update(ModelVersionDB).values(active=False)
        await self.session.execute(stmt_deactivate)

        # Activate the chosen model
        model.active = True
        await self.session.commit()
        await self.session.refresh(model)
        return model

    async def get_all(self) -> list[ModelVersionDB]:
        stmt = select(ModelVersionDB).order_by(ModelVersionDB.created_at.desc())
        res = await self.session.execute(stmt)
        return list(res.scalars().all())
