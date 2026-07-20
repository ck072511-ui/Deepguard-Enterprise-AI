"""
DeepGuard — api/v1/endpoints/models.py

FastAPI endpoint for model version registry tracking and activation control.
"""

import uuid
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from database.session import get_db_session
from database.models import ModelVersionDB
from repositories.sqlite.model import SQLiteModelRepository
from schemas.responses.models import ModelVersionResponse

router = APIRouter()


class ModelCreateRequest(BaseModel):
    """Schema representing request payload to register a new model version."""

    name: str = Field(..., example="vit_tiny_patch16_224")
    version: str = Field(..., example="1.0.0")
    registry_path: str = Field(..., example="/weights/vit_tiny_1.0.0.pt")


@router.get("/models", response_model=list[ModelVersionResponse])
async def list_models(db: AsyncSession = Depends(get_db_session)) -> list[ModelVersionResponse]:
    """Retrieve all model versions registered in DeepGuard."""
    repo = SQLiteModelRepository(db)
    models = await repo.get_all()
    return [
        ModelVersionResponse(
            id=m.id,
            name=m.name,
            version=m.version,
            registry_path=m.registry_path,
            active=m.active,
            created_at=m.created_at,
        )
        for m in models
    ]


@router.post("/models", response_model=ModelVersionResponse, status_code=status.HTTP_201_CREATED)
async def register_model(
    payload: ModelCreateRequest, db: AsyncSession = Depends(get_db_session)
) -> ModelVersionResponse:
    """Register a new model version in the database."""
    repo = SQLiteModelRepository(db)

    # Check if a model with the same version already exists
    all_models = await repo.get_all()
    for m in all_models:
        if m.name == payload.name and m.version == payload.version:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Model '{payload.name}' with version '{payload.version}' is already registered.",
            )

    model = ModelVersionDB(
        id=str(uuid.uuid4()),
        name=payload.name,
        version=payload.version,
        registry_path=payload.registry_path,
        active=False,
        created_at=datetime.utcnow(),
    )
    added = await repo.add(model)
    return ModelVersionResponse(
        id=added.id,
        name=added.name,
        version=added.version,
        registry_path=added.registry_path,
        active=added.active,
        created_at=added.created_at,
    )


@router.post("/models/{model_id}/activate", response_model=ModelVersionResponse)
async def activate_model(
    model_id: str, db: AsyncSession = Depends(get_db_session)
) -> ModelVersionResponse:
    """Designate a registered model as the active inference engine."""
    repo = SQLiteModelRepository(db)
    activated = await repo.set_active(model_id)
    if not activated:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model version with ID '{model_id}' was not found.",
        )
    return ModelVersionResponse(
        id=activated.id,
        name=activated.name,
        version=activated.version,
        registry_path=activated.registry_path,
        active=activated.active,
        created_at=activated.created_at,
    )
