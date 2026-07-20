"""
DeepGuard — api/v1/endpoints/model_info.py

Model information and architecture inspection endpoints.

GET /api/v1/model-info/active     — Active model capabilities & config
GET /api/v1/model-info/configs    — List available model configuration files
GET /api/v1/model-info/training   — Latest training run summary (MLflow)

Complements the model registry CRUD in api/v1/endpoints/models.py.
"""

import logging
import yaml
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Security, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth.security import require_any_auth
from database.session import get_db_session
from repositories.sqlite.model import SQLiteModelRepository

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/model-info", tags=["Model Information"])

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


# ── Response schemas ──────────────────────────────────────────────────────────

class ArchitectureInfo(BaseModel):
    """ViT architecture summary."""
    backbone: str = Field(..., description="Backbone model name (e.g. vit_tiny_patch16_224)")
    pretrained: bool = Field(..., description="Whether ImageNet pretraining was used")
    img_size: int = Field(..., description="Input image resolution")
    patch_size: int = Field(..., description="ViT patch size in pixels")
    num_classes: int = Field(..., description="Number of output classes (2 for binary deepfake)")
    head_type: str = Field(..., description="Classification head type")
    lora_enabled: bool = Field(..., description="Whether LoRA adapters are injected")
    lora_rank: int | None = Field(None, description="LoRA rank (r) if enabled")
    lora_alpha: float | None = Field(None, description="LoRA alpha scaling factor if enabled")
    dropout: float | None = Field(None, description="Head dropout rate")


class ActiveModelInfo(BaseModel):
    """Full information about the currently active model."""
    model_id: str | None = Field(None, description="Database ID of the active model record")
    name: str | None = Field(None, description="Model name")
    version: str | None = Field(None, description="Model version")
    registry_path: str | None = Field(None, description="Path to model weights")
    architecture: ArchitectureInfo | None = Field(None, description="Model architecture details")
    registered_at: datetime | None = Field(None, description="When the model was registered")
    config_file: str | None = Field(None, description="Source model configuration file")


class ConfigFileSummary(BaseModel):
    """Short summary of an available model config."""
    filename: str
    path: str
    backbone: str | None = None
    last_modified: datetime | None = None


class TrainingRunSummary(BaseModel):
    """Latest training run metadata from MLflow."""
    run_id: str | None = None
    experiment_name: str | None = None
    status: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    mlflow_ui_url: str | None = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get(
    "/active",
    response_model=ActiveModelInfo,
    summary="Active model information",
    description=(
        "Returns full details about the currently active inference model, including "
        "its ViT architecture configuration, LoRA settings, and registry metadata."
    ),
)
async def get_active_model_info(
    db: AsyncSession = Depends(get_db_session),
    _auth: str = Security(require_any_auth),
) -> ActiveModelInfo:
    """Return detailed information about the active model."""
    repo = SQLiteModelRepository(db)
    active = await repo.get_active()

    arch: ArchitectureInfo | None = None
    config_file: str | None = None

    # Load model config
    config_path = _PROJECT_ROOT / "configs" / "model_config.yaml"
    if config_path.exists():
        config_file = str(config_path)
        with open(config_path) as f:
            cfg: dict = yaml.safe_load(f) or {}

        model_cfg = cfg.get("model", {})
        head_cfg = model_cfg.get("head", {})
        fine_tuning_cfg = model_cfg.get("fine_tuning", {})

        arch = ArchitectureInfo(
            backbone=model_cfg.get("name", "unknown"),
            pretrained=model_cfg.get("pretrained", False),
            img_size=model_cfg.get("input_size", 224),
            patch_size=model_cfg.get("patch_size", 16),
            num_classes=model_cfg.get("num_classes", 2),
            head_type=head_cfg.get("type", "linear"),
            lora_enabled=(fine_tuning_cfg.get("strategy", "") == "lora"),
            lora_rank=fine_tuning_cfg.get("lora_r") if fine_tuning_cfg.get("strategy", "") == "lora" else None,
            lora_alpha=fine_tuning_cfg.get("lora_alpha") if fine_tuning_cfg.get("strategy", "") == "lora" else None,
            dropout=head_cfg.get("dropout"),
        )

    if active is None:
        return ActiveModelInfo(architecture=arch, config_file=config_file)

    return ActiveModelInfo(
        model_id=active.id,
        name=active.name,
        version=active.version,
        registry_path=active.registry_path,
        architecture=arch,
        registered_at=active.created_at,
        config_file=config_file,
    )


@router.get(
    "/configs",
    response_model=list[ConfigFileSummary],
    summary="List model configuration files",
    description="Return a list of all YAML model configuration files found in the `configs/` directory.",
)
async def list_model_configs(
    _auth: str = Security(require_any_auth),
) -> list[ConfigFileSummary]:
    """Scan the configs directory and return available model config files."""
    configs_dir = _PROJECT_ROOT / "configs"
    results: list[ConfigFileSummary] = []

    for yaml_file in sorted(configs_dir.glob("*.yaml")):
        if "model" not in yaml_file.stem:
            continue
        backbone_name: str | None = None
        try:
            with open(yaml_file) as f:
                data: dict = yaml.safe_load(f) or {}
            backbone_name = data.get("backbone", {}).get("name")
        except Exception:
            pass
        results.append(
            ConfigFileSummary(
                filename=yaml_file.name,
                path=str(yaml_file),
                backbone=backbone_name,
                last_modified=datetime.fromtimestamp(
                    yaml_file.stat().st_mtime, tz=timezone.utc
                ),
            )
        )
    return results


@router.get(
    "/training",
    response_model=TrainingRunSummary,
    summary="Latest training run summary",
    description=(
        "Retrieve the most recent training run metadata from MLflow, "
        "including final metrics (accuracy, F1, AUC), hyperparameters, "
        "and a link to the MLflow UI."
    ),
)
async def get_training_summary(
    _auth: str = Security(require_any_auth),
) -> TrainingRunSummary:
    """Return the latest training run data from MLflow (if available)."""
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
        from fastapi.concurrency import run_in_threadpool
        import socket
        from urllib.parse import urlparse

        # 1. Quick socket connection check to prevent blocking if MLflow is offline
        tracking_uri = mlflow.get_tracking_uri() or "http://localhost:5000"
        try:
            parsed_url = urlparse(tracking_uri)
            host = parsed_url.hostname or "localhost"
            port = parsed_url.port or 5000
            # Attempt socket connection with a short timeout (0.5 seconds)
            with socket.create_connection((host, port), timeout=0.5):
                pass
        except Exception:
            logger.warning("MLflow tracking server at %s is unreachable. Bypassing MLflow summary.", tracking_uri)
            return TrainingRunSummary()

        client = MlflowClient()
        experiments = await run_in_threadpool(client.search_experiments)
        if not experiments:
            return TrainingRunSummary()

        # Pick the first experiment and find the latest completed run
        exp = experiments[0]
        runs = await run_in_threadpool(
            client.search_runs,
            experiment_ids=[exp.experiment_id],
            order_by=["start_time DESC"],
            max_results=1,
        )
        if not runs:
            return TrainingRunSummary()

        run = runs[0]
        start = (
            datetime.fromtimestamp(run.info.start_time / 1000, tz=timezone.utc)
            if run.info.start_time
            else None
        )
        end = (
            datetime.fromtimestamp(run.info.end_time / 1000, tz=timezone.utc)
            if run.info.end_time
            else None
        )

        mlflow_uri = mlflow.get_tracking_uri()
        ui_url = f"{mlflow_uri}/#/experiments/{exp.experiment_id}/runs/{run.info.run_id}"

        return TrainingRunSummary(
            run_id=run.info.run_id,
            experiment_name=exp.name,
            status=run.info.status,
            start_time=start,
            end_time=end,
            metrics=dict(run.data.metrics),
            params=dict(run.data.params),
            mlflow_ui_url=ui_url,
        )

    except Exception as exc:
        logger.warning("Could not fetch MLflow training summary: %s", exc)
        return TrainingRunSummary()
