"""
DeepGuard — api/v1/endpoints/health.py

FastAPI endpoint for system health verification.
"""

import logging
import yaml
from pathlib import Path
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from database.session import get_db_session
from schemas.responses.health import HealthCheckResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthCheckResponse)
async def health_check(db: AsyncSession = Depends(get_db_session)) -> HealthCheckResponse:
    """Check API, database, and system dependency availability."""
    db_status = "connected"
    model_status = "loaded"
    mlflow_status = "connected"

    # 1. Verify Database Connection
    try:
        # Load ping query from config
        project_root = Path(__file__).resolve().parents[3]
        config_path = project_root / "configs" / "database_config.yaml"
        ping_query = "SELECT 1"
        if config_path.exists():
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            ping_query = cfg.get("database", {}).get("health", {}).get("ping_query", "SELECT 1")

        await db.execute(text(ping_query))
    except Exception as e:
        logger.error("Database health check failed: %s", str(e))
        db_status = "disconnected"

    # 2. Check Model Status
    try:
        project_root = Path(__file__).resolve().parents[3]
        model_config_path = project_root / "configs" / "model_config.yaml"
        if model_config_path.exists():
            from services.detection.service import DetectionService
            service = DetectionService(db)
            await service.get_model()
        else:
            model_status = "config_missing"
    except Exception as e:
        logger.error("Model health check failed: %s", str(e))
        model_status = "failed"

    # 3. MLflow Check
    try:
        import mlflow
        tracking_uri = mlflow.get_tracking_uri()
        if not tracking_uri:
            mlflow_status = "disconnected"
    except Exception:
        mlflow_status = "unavailable"

    overall_status = "healthy"
    if db_status == "disconnected" or model_status == "failed":
        overall_status = "unhealthy"

    return HealthCheckResponse(
        status=overall_status,
        database=db_status,
        model=model_status,
        mlflow=mlflow_status,
    )
