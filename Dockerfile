# ============================================================
# DeepGuard — Dockerfile
# Multi-stage production Docker image
# ============================================================

# ── Stage 1: Builder ─────────────────────────────────────
FROM python:3.11-slim AS builder

LABEL maintainer="DeepGuard Team <deepguard@example.com>"
LABEL description="DeepGuard Deepfake Detection System — Builder Stage"

# Set environment variables for builder
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    POETRY_NO_INTERACTION=1

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    libopencv-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and wheel
RUN pip install --upgrade pip wheel setuptools

# Copy and install requirements in layers for better caching
WORKDIR /build
COPY requirements.txt .

# Install PyTorch CPU (replace with GPU variant if needed)
RUN pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining requirements
RUN pip install --no-deps -r requirements.txt || pip install -r requirements.txt


# ── Stage 2: Production Runner ────────────────────────────
FROM python:3.11-slim AS production

LABEL maintainer="DeepGuard Team <deepguard@example.com>"
LABEL description="DeepGuard Deepfake Detection System — Production Image"
LABEL version="1.0.0"

# Runtime environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PATH="/opt/venv/bin:$PATH" \
    APP_ENV=production \
    APP_HOME=/app

# Install only runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Create non-root user for security
RUN groupadd --gid 1001 deepguard && \
    useradd --uid 1001 --gid deepguard --shell /bin/bash --create-home deepguard

# Set working directory
WORKDIR $APP_HOME

# Copy application source code
COPY --chown=deepguard:deepguard . .

# Create required runtime directories
RUN mkdir -p \
    $APP_HOME/logs \
    $APP_HOME/uploads \
    $APP_HOME/weights \
    $APP_HOME/database \
    $APP_HOME/mlflow/artifacts \
    && chown -R deepguard:deepguard $APP_HOME

# Switch to non-root user
USER deepguard

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Default command — production uvicorn server
CMD ["uvicorn", "backend.main:app", \
    "--host", "0.0.0.0", \
    "--port", "8000", \
    "--workers", "4", \
    "--log-level", "info", \
    "--access-log", \
    "--no-use-colors"]
