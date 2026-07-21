"""
DeepGuard — backend/main.py

Main FastAPI application setup and middleware pipeline wiring.

Middleware stack (outer → inner, execution order on request):
  1. RequestLoggingMiddleware  — attaches X-Request-ID, logs every req/resp
  2. PrometheusMetricsMiddleware — records http_requests_total etc.
  3. RateLimitMiddleware       — sliding-window IP rate limiting
  4. GZipMiddleware            — response compression
  5. TrustedHostMiddleware     — allowlisted Host headers only
  6. CORSMiddleware            — CORS preflight + headers

Exception handlers (registered globally):
  - DeepGuardBaseException     → structured JSON error envelope
  - StarletteHTTPException     → structured JSON error envelope
  - RequestValidationError     → 422 with per-field details
  - Exception (catch-all)      → 500 without traceback leakage
"""

import logging
import logging.config
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.openapi.utils import get_openapi

from api.exception_handlers import register_exception_handlers
from api.middleware.logging import RequestLoggingMiddleware
from api.middleware.metrics import PrometheusMetricsMiddleware
from api.middleware.rate_limit import RateLimitMiddleware
from api.v1 import v1_router


# ── Logging setup ─────────────────────────────────────────────────────────────

def _configure_logging(log_level: str = "INFO") -> None:
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {"level": log_level.upper(), "handlers": ["console"]},
            "loggers": {
                "deepguard.access": {"level": "INFO", "propagate": True},
                "uvicorn.access": {"level": "WARNING", "propagate": False},
            },
        }
    )


# ── Application factory ───────────────────────────────────────────────────────

def create_application() -> FastAPI:
    """Build and configure the FastAPI application instance."""

    # 1. Load config
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "configs" / "api_config.yaml"

    app_config: dict = {}
    cors_config: dict = {}
    rl_config: dict = {}
    gzip_config: dict = {}
    th_config: list = []

    if config_path.exists():
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        app_config = cfg.get("app", {})
        cors_config = cfg.get("cors", {})
        rl_config = cfg.get("rate_limiting", {})
        gzip_config = cfg.get("compression", {})
        th_config = cfg.get("trusted_hosts", [])
        server_cfg = cfg.get("server", {})
        _configure_logging(server_cfg.get("log_level", "INFO"))
    else:
        _configure_logging()

    logger = logging.getLogger(__name__)
    logger.info("Initialising DeepGuard API v%s", app_config.get("version", "1.0.0"))

    # 2. Build FastAPI with rich Swagger metadata
    contact_cfg = app_config.get("contact", {})
    license_cfg = app_config.get("license", {})

    app = FastAPI(
        title=app_config.get("title", "DeepGuard API"),
        description=app_config.get(
            "description",
            "Production-ready Deepfake Detection System powered by Vision Transformers.",
        ),
        version=app_config.get("version", "1.0.0"),
        openapi_url=app_config.get("openapi_url", "/openapi.json"),
        docs_url=app_config.get("docs_url", "/docs"),
        redoc_url=app_config.get("redoc_url", "/redoc"),
        contact={
            "name": contact_cfg.get("name", "DeepGuard Team"),
            "email": contact_cfg.get("email", "deepguard@example.com"),
        },
        license_info={
            "name": license_cfg.get("name", "Apache 2.0"),
            "url": license_cfg.get("url", "https://www.apache.org/licenses/LICENSE-2.0"),
        },
        openapi_tags=[
            {
                "name": "Health",
                "description": "Service health and readiness probes.",
            },
            {
                "name": "Authentication",
                "description": "JWT token issuance and identity endpoints. "
                               "Use `POST /auth/token` to obtain a Bearer token.",
            },
            {
                "name": "Image Detection",
                "description": "Deepfake analysis for individual image files (JPEG/PNG/WebP/BMP).",
            },
            {
                "name": "Video Detection",
                "description": "Deepfake analysis for video files via sampled-frame inference.",
            },
            {
                "name": "Batch Detection",
                "description": "Analyse up to 32 images in a single request.",
            },
            {
                "name": "Prediction History",
                "description": "Browse, filter, and delete stored detection records.",
            },
            {
                "name": "Upload Management",
                "description": "Stage file uploads before triggering detection.",
            },
            {
                "name": "Model Registry",
                "description": "Register, list, and activate model versions.",
            },
            {
                "name": "Model Information",
                "description": "Inspect active model architecture, configs, and training runs.",
            },
            {
                "name": "Detection (Legacy)",
                "description": "Original unified detect endpoint — kept for backwards compatibility.",
            },
        ],
        # Security scheme definitions for Swagger UI
        swagger_ui_parameters={
            "persistAuthorization": True,
            "displayRequestDuration": True,
            "filter": True,
            "tryItOutEnabled": True,
        },
    )

    # 3. Register global exception handlers (before middleware!)
    register_exception_handlers(app)

    # 4. Request logging (outermost layer — wraps everything)
    app.add_middleware(RequestLoggingMiddleware)

    # 5. Prometheus metrics
    metrics_enabled = app_config.get("metrics_enabled", True)
    app.add_middleware(PrometheusMetricsMiddleware, enabled=metrics_enabled)

    # 6. Rate limiting
    if rl_config.get("enabled", True):
        app.add_middleware(
            RateLimitMiddleware,
            requests_per_minute=rl_config.get("requests_per_minute", 60),
            enabled=True,
        )

    # 7. GZip compression
    if gzip_config.get("enabled", True):
        app.add_middleware(
            GZipMiddleware,
            minimum_size=gzip_config.get("min_size_bytes", 1024),
        )

    # 8. Trusted hosts
    if th_config:
        allowed_hosts = list(th_config)
        for always_allow in ("testserver", "localhost", "127.0.0.1"):
            if always_allow not in allowed_hosts:
                allowed_hosts.append(always_allow)
        allowed_hosts.extend(["0.0.0.0", ".railway.app", ".render.com", ".app.github.dev"])
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)

    # 9. CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_config.get("allow_origins", ["*"]),
        allow_credentials=cors_config.get("allow_credentials", True),
        allow_methods=cors_config.get("allow_methods", ["*"]),
        allow_headers=cors_config.get("allow_headers", ["*"]),
        max_age=cors_config.get("max_age", 3600),
    )

    # 10. Include all API routers
    app.include_router(v1_router, prefix="/api/v1")

    # 11. Prometheus /metrics scrape endpoint
    @app.get("/metrics", include_in_schema=False)
    async def metrics_endpoint() -> Response:
        """Prometheus metrics scrape endpoint (not shown in Swagger)."""
        try:
            from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

            data = generate_latest()
            return Response(content=data, media_type=CONTENT_TYPE_LATEST)
        except ImportError:
            return Response(
                content="prometheus_client not installed",
                media_type="text/plain",
                status_code=503,
            )

    # 12. Root redirect to docs
    @app.get("/", include_in_schema=False)
    async def root() -> Response:
        """Redirect root to interactive API documentation."""
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/docs")

    # 13. Startup: ensure DB tables exist
    @app.on_event("startup")
    async def startup_event() -> None:
        from database import Base, engine
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables verified / created.")

    # 14. Shutdown hook
    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        from database import engine
        await engine.dispose()
        logger.info("Database connection pool closed.")

    return app


app = create_application()
