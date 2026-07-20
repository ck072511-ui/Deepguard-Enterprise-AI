"""
DeepGuard — api/middleware/metrics.py

Prometheus metrics middleware for FastAPI.
Exposes /metrics endpoint and tracks:
  - HTTP request counts by method, endpoint, status
  - Request duration histogram
  - In-flight requests gauge
  - DeepGuard-specific detection counters and inference latency
"""

import time
import logging
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.routing import Match

try:
    from prometheus_client import (
        Counter,
        Histogram,
        Gauge,
        CollectorRegistry,
        generate_latest,
        CONTENT_TYPE_LATEST,
        REGISTRY,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = logging.getLogger(__name__)

# ── Metric definitions ────────────────────────────────────
if PROMETHEUS_AVAILABLE:
    HTTP_REQUESTS_TOTAL = Counter(
        "http_requests_total",
        "Total HTTP requests received",
        ["method", "endpoint", "status"],
    )

    HTTP_REQUEST_DURATION = Histogram(
        "http_request_duration_seconds",
        "HTTP request duration in seconds",
        ["method", "endpoint"],
        buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    )

    HTTP_REQUESTS_IN_FLIGHT = Gauge(
        "http_requests_in_flight",
        "Number of HTTP requests currently being processed",
        ["method"],
    )

    DEEPGUARD_DETECTIONS_TOTAL = Counter(
        "deepguard_detections_total",
        "Total deepfake detection requests processed",
        ["media_type", "label"],
    )

    DEEPGUARD_INFERENCE_DURATION = Histogram(
        "deepguard_inference_duration_seconds",
        "Model inference duration in seconds",
        buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
    )


def record_detection(media_type: str, label: str, duration_seconds: float) -> None:
    """Record a completed detection to Prometheus metrics.

    Call this from detection service after inference completes.

    Args:
        media_type: 'image' or 'video'
        label: 'real' or 'fake'
        duration_seconds: Inference wall-clock time in seconds
    """
    if not PROMETHEUS_AVAILABLE:
        return
    DEEPGUARD_DETECTIONS_TOTAL.labels(media_type=media_type, label=label).inc()
    DEEPGUARD_INFERENCE_DURATION.observe(duration_seconds)


class PrometheusMetricsMiddleware(BaseHTTPMiddleware):
    """ASGI middleware that records Prometheus HTTP metrics for every request.

    Exposes:
      - http_requests_total{method, endpoint, status}
      - http_request_duration_seconds{method, endpoint}
      - http_requests_in_flight{method}
    """

    def __init__(self, app, enabled: bool = True) -> None:
        super().__init__(app)
        self.enabled = enabled and PROMETHEUS_AVAILABLE
        if not PROMETHEUS_AVAILABLE and enabled:
            logger.warning(
                "prometheus_client is not installed — metrics middleware is disabled. "
                "Install it with: pip install prometheus-client"
            )

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if not self.enabled:
            return await call_next(request)

        method = request.method
        endpoint = self._get_route_path(request)

        HTTP_REQUESTS_IN_FLIGHT.labels(method=method).inc()
        start = time.perf_counter()
        status_code = 500

        try:
            response: Response = await call_next(request)
            status_code = response.status_code
            return response
        except Exception:
            raise
        finally:
            duration = time.perf_counter() - start
            HTTP_REQUESTS_TOTAL.labels(
                method=method, endpoint=endpoint, status=str(status_code)
            ).inc()
            HTTP_REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)
            HTTP_REQUESTS_IN_FLIGHT.labels(method=method).dec()

    @staticmethod
    def _get_route_path(request: Request) -> str:
        """Resolve the matched route template path (e.g. /api/v1/detect) to avoid high cardinality."""
        for route in request.app.routes:
            match, _ = route.matches(request.scope)
            if match == Match.FULL:
                return getattr(route, "path", request.url.path)
        return request.url.path
