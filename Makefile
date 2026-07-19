# ============================================================
# DeepGuard — Makefile
# Developer convenience commands for build, test, run, deploy
# ============================================================

.PHONY: help setup install lint format typecheck test test-unit \
        test-integration test-e2e test-coverage clean dev \
        docker-build docker-up docker-down docker-push \
        train evaluate export-onnx prepare-dataset \
        mlflow-ui db-migrate db-reset \
        pre-commit-install pre-commit-run docs-serve docs-build

# ── Variables ─────────────────────────────────────────────
PYTHON       := python
PIP          := pip
PYTEST       := pytest
BLACK        := black
RUFF         := ruff
MYPY         := mypy
DOCKER       := docker
COMPOSE      := docker compose
MLFLOW       := mlflow
REGISTRY     ?= ghcr.io/your-org
IMAGE_NAME   ?= deepguard
IMAGE_TAG    ?= latest
CONFIG       ?= configs/training_config.yaml
CHECKPOINT   ?= weights/best_model.pt
DATASET      ?= ff++

# ── Colors ────────────────────────────────────────────────
RESET  := \033[0m
BOLD   := \033[1m
GREEN  := \033[32m
YELLOW := \033[33m
CYAN   := \033[36m
RED    := \033[31m

# ── Default target ────────────────────────────────────────
.DEFAULT_GOAL := help

## help: Show this help message
help:
	@echo "$(BOLD)$(CYAN)DeepGuard — Developer Commands$(RESET)"
	@echo "$(CYAN)════════════════════════════════════════════$(RESET)"
	@grep -E '^## ' Makefile | awk 'BEGIN {FS = ": "}; {printf "  $(GREEN)%-30s$(RESET) %s\n", $$1, $$2}' | sed 's/## //'

# ── Environment Setup ─────────────────────────────────────
## setup: Create virtualenv and install all dependencies
setup:
	@echo "$(BOLD)Setting up development environment...$(RESET)"
	$(PYTHON) -m venv .venv
	.venv/Scripts/python -m pip install --upgrade pip wheel setuptools
	.venv/Scripts/pip install -r requirements.txt
	.venv/Scripts/pre-commit install
	@echo "$(GREEN)✓ Setup complete! Activate with: .venv\\Scripts\\activate$(RESET)"

## install: Install dependencies only
install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

## pre-commit-install: Install pre-commit hooks
pre-commit-install:
	pre-commit install
	pre-commit install --hook-type commit-msg

# ── Code Quality ──────────────────────────────────────────
## lint: Run all linters (ruff + bandit)
lint:
	@echo "$(BOLD)Running linters...$(RESET)"
	$(RUFF) check . --fix
	bandit -r . -c pyproject.toml
	@echo "$(GREEN)✓ Lint passed$(RESET)"

## format: Format code with black and ruff
format:
	@echo "$(BOLD)Formatting code...$(RESET)"
	$(BLACK) .
	$(RUFF) check . --fix
	@echo "$(GREEN)✓ Format complete$(RESET)"

## format-check: Check formatting without modifying files
format-check:
	$(BLACK) --check .
	$(RUFF) check .

## typecheck: Run mypy type checking
typecheck:
	@echo "$(BOLD)Running type checks...$(RESET)"
	$(MYPY) . --config-file pyproject.toml
	@echo "$(GREEN)✓ Type check passed$(RESET)"

## pre-commit-run: Run all pre-commit hooks
pre-commit-run:
	pre-commit run --all-files

# ── Testing ───────────────────────────────────────────────
## test: Run all tests
test:
	@echo "$(BOLD)Running all tests...$(RESET)"
	$(PYTEST) tests/ -v

## test-unit: Run unit tests only
test-unit:
	$(PYTEST) tests/unit/ -v -m unit

## test-integration: Run integration tests only
test-integration:
	$(PYTEST) tests/integration/ -v -m integration

## test-e2e: Run end-to-end tests only
test-e2e:
	$(PYTEST) tests/e2e/ -v -m e2e

## test-coverage: Run tests with coverage report
test-coverage:
	$(PYTEST) tests/ --cov=. --cov-report=html --cov-report=term-missing
	@echo "$(GREEN)✓ Coverage report generated in htmlcov/$(RESET)"

## test-fast: Run tests excluding slow/gpu tests
test-fast:
	$(PYTEST) tests/ -v -m "not slow and not gpu"

# ── Development Server ────────────────────────────────────
## dev: Start local development server with hot reload
dev:
	@echo "$(BOLD)Starting development server...$(RESET)"
	uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload --log-level debug

## dev-prod: Start local server in production mode
dev-prod:
	uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 4 --log-level info

# ── Docker ────────────────────────────────────────────────
## docker-build: Build production Docker image
docker-build:
	@echo "$(BOLD)Building Docker image...$(RESET)"
	$(DOCKER) build -t $(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG) \
		--target production \
		--build-arg BUILD_DATE=$(shell date -u +%Y-%m-%dT%H:%M:%SZ) \
		--build-arg VERSION=$(IMAGE_TAG) \
		.
	@echo "$(GREEN)✓ Image built: $(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)$(RESET)"

## docker-up: Start all services with docker compose
docker-up:
	$(COMPOSE) up -d
	@echo "$(GREEN)✓ Services started$(RESET)"
	@echo "  API:    http://localhost:8000"
	@echo "  Docs:   http://localhost:8000/docs"
	@echo "  MLflow: http://localhost:5000"

## docker-down: Stop all docker compose services
docker-down:
	$(COMPOSE) down
	@echo "$(GREEN)✓ Services stopped$(RESET)"

## docker-logs: Tail logs from all compose services
docker-logs:
	$(COMPOSE) logs -f

## docker-push: Push image to container registry
docker-push:
	$(DOCKER) push $(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)

## docker-clean: Remove containers, images, volumes
docker-clean:
	$(COMPOSE) down -v --remove-orphans
	$(DOCKER) image prune -f
	@echo "$(GREEN)✓ Docker cleanup complete$(RESET)"

# ── Database ──────────────────────────────────────────────
## db-migrate: Run database migrations
db-migrate:
	@echo "$(BOLD)Running database migrations...$(RESET)"
	alembic upgrade head
	@echo "$(GREEN)✓ Migrations applied$(RESET)"

## db-downgrade: Roll back last migration
db-downgrade:
	alembic downgrade -1

## db-reset: Drop and recreate database
db-reset:
	@echo "$(BOLD)$(RED)WARNING: This will delete all data!$(RESET)"
	alembic downgrade base
	alembic upgrade head
	@echo "$(GREEN)✓ Database reset complete$(RESET)"

## db-revision: Create a new migration revision
db-revision:
	alembic revision --autogenerate -m "$(MSG)"

# ── MLflow ────────────────────────────────────────────────
## mlflow-ui: Start MLflow tracking server UI
mlflow-ui:
	@echo "$(BOLD)Starting MLflow UI at http://localhost:5000$(RESET)"
	$(MLFLOW) server \
		--backend-store-uri sqlite:///./mlflow/mlflow.db \
		--default-artifact-root ./mlflow/artifacts \
		--host 0.0.0.0 \
		--port 5000

# ── Training ──────────────────────────────────────────────
## prepare-dataset: Download and preprocess dataset
prepare-dataset:
	@echo "$(BOLD)Preparing dataset: $(DATASET)$(RESET)"
	$(PYTHON) scripts/prepare_dataset.py --dataset $(DATASET)

## train: Run model training experiment
train:
	@echo "$(BOLD)Starting training with config: $(CONFIG)$(RESET)"
	$(PYTHON) scripts/train.py --config $(CONFIG)

## evaluate: Evaluate model checkpoint
evaluate:
	@echo "$(BOLD)Evaluating checkpoint: $(CHECKPOINT)$(RESET)"
	$(PYTHON) scripts/evaluate.py --checkpoint $(CHECKPOINT)

## export-onnx: Export PyTorch model to ONNX format
export-onnx:
	@echo "$(BOLD)Exporting to ONNX: $(CHECKPOINT)$(RESET)"
	$(PYTHON) scripts/export_onnx.py --checkpoint $(CHECKPOINT)
	@echo "$(GREEN)✓ ONNX model saved to weights/model.onnx$(RESET)"

## benchmark: Run inference benchmark
benchmark:
	$(PYTHON) scripts/benchmark.py --checkpoint $(CHECKPOINT)

# ── Documentation ─────────────────────────────────────────
## docs-serve: Serve documentation locally
docs-serve:
	mkdocs serve --dev-addr 0.0.0.0:8080

## docs-build: Build static documentation site
docs-build:
	mkdocs build --strict

# ── Cleanup ───────────────────────────────────────────────
## clean: Remove build artifacts and caches
clean:
	@echo "$(BOLD)Cleaning build artifacts...$(RESET)"
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	find . -type f -name "coverage.xml" -delete 2>/dev/null || true
	@echo "$(GREEN)✓ Clean complete$(RESET)"

## clean-all: Remove everything including virtualenv
clean-all: clean
	rm -rf .venv/
	@echo "$(GREEN)✓ Full clean complete$(RESET)"
