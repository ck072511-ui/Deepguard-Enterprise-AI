# DeepGuard — Deployment Guide

## Overview

This guide covers production deployment of the DeepGuard system using Docker Compose, environment hardening, CI/CD automation, and monitoring setup.

---

## Production Checklist

Before deploying to production:

- [x] Change `SECRET_KEY` to a strong random value
- [x] Set `APP_ENV=production`
- [x] Configure a real database (SQLite is acceptable for small deployments, PostgreSQL recommended for larger ones)
- [x] Set up SSL certificates for HTTPS
- [x] Configure proper CORS origins for the deployed frontend domain
- [x] Set up log aggregation (ELK, Loki, etc.)
- [x] Configure Grafana admin password
- [x] Review rate limiting settings
- [x] Restrict trusted hosts

### Required Runtime Environment Variables

Backend:
```bash
APP_ENV=production
DATABASE_URL=sqlite+aiosqlite:////app/database/deepguard.db
DEEPGUARD_API_URL=https://your-backend.example.com/api/v1
```

Frontend (Streamlit Community Cloud):
```bash
DEEPGUARD_API_URL=https://your-backend.example.com/api/v1
```

---

## Docker Compose Deployment

### Available Service Profiles

| Profile | Services |
|---|---|
| *(default)* | `api`, `mlflow` |
| `monitoring` | + `prometheus`, `grafana` |
| `production` | + `nginx` (SSL/reverse proxy) |
| `dev` | `api-dev` (hot reload) |

### Environment Configuration

Copy and customize:
```bash
cp .env.example .env
```

**Critical production settings:**
```bash
APP_ENV=production
SECRET_KEY=generate-with-openssl-rand-hex-32
DATABASE_URL=sqlite+aiosqlite:////app/database/deepguard.db

# For PostgreSQL (production recommended):
# DATABASE_URL=postgresql+asyncpg://user:pass@db:5432/deepguard

GRAFANA_PASSWORD=your-secure-grafana-password
CORS_ORIGINS=https://your-domain.com,https://app.your-domain.com
```

### Start Production Stack

### FastAPI Backend Deployment (Render or Railway)

1. Create a new web service for the backend using the repository root.
2. Set the build command to:
```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu && pip install -r backend/requirements.txt
```
3. Set the start command to:
```bash
uvicorn backend.main:app --host 0.0.0.0 --port $PORT
```
4. Add the environment variables:
```bash
APP_ENV=production
DATABASE_URL=sqlite+aiosqlite:///./database/deepguard.db
```
5. Confirm the health endpoint returns 200 at `/api/v1/health`.

### Streamlit Frontend Deployment (Streamlit Community Cloud)

1. Connect the repository to Streamlit Community Cloud.
2. Set the main file to `frontend/app.py`.
3. Add the environment variable:
```bash
DEEPGUARD_API_URL=https://your-backend.example.com/api/v1
```
4. Deploy and verify the UI reports the backend as online.


```bash
# Core: API + MLflow
docker-compose up -d

# With monitoring
docker-compose --profile monitoring up -d

# Full production (API + MLflow + Prometheus + Grafana + Nginx)
docker-compose --profile monitoring --profile production up -d
```

### Check Service Health

```bash
# View all service statuses
docker-compose ps

# API health
curl http://localhost:8000/api/v1/health

# View logs
docker-compose logs -f api
docker-compose logs -f mlflow

# Follow all logs
docker-compose logs -f
```

---

## SSL / HTTPS Setup

### Using Let's Encrypt (Recommended)

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal (cron)
echo "0 12 * * * /usr/bin/certbot renew --quiet" | sudo crontab -
```

Then place certificates in `deployment/nginx/ssl/`:
```
deployment/nginx/ssl/
├── fullchain.pem
└── privkey.pem
```

### Self-signed Certificate (Development)

```bash
mkdir -p deployment/nginx/ssl
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout deployment/nginx/ssl/privkey.pem \
  -out deployment/nginx/ssl/fullchain.pem \
  -subj "/C=US/ST=Dev/L=Dev/O=DeepGuard/CN=localhost"
```

---

## Scaling

### Horizontal Scaling (Multiple API Workers)

```bash
# Multiple containers behind Nginx
docker-compose up --scale api=3 -d
```

Update Nginx config to upstream load balance across containers.

### Vertical Scaling

Edit `docker-compose.yml` resource limits:
```yaml
deploy:
  resources:
    limits:
      memory: 8G      # Increase for larger models
      cpus: "4.0"
```

### GPU Inference

Add NVIDIA runtime to `docker-compose.yml`:
```yaml
api:
  runtime: nvidia
  environment:
    - NVIDIA_VISIBLE_DEVICES=all
    - INFERENCE_DEVICE=cuda
```

---

## CI/CD Pipeline

The repository includes a GitHub Actions workflow at `.github/workflows/ci.yml`.

### Pipeline Stages

```
Push to main/PR
       ↓
  1. Checkout code
       ↓
  2. Setup Python 3.11
       ↓
  3. Install system deps (ffmpeg, libsm6, etc.)
       ↓
  4. Install PyTorch (CPU)
       ↓
  5. Install project deps
       ↓
  6. Run pytest (187 tests, coverage ≥ 55%)
       ↓
  7. ONNX export verification
       ↓
  8. Benchmark (optional)
       ↓
  9. Docker image build (no push)
```

### Enabling Docker Push

Add secrets to your GitHub repository:
```
Settings → Secrets → Actions:
  DOCKER_USERNAME = your-dockerhub-username
  DOCKER_PASSWORD = your-dockerhub-token
```

Then update `.github/workflows/ci.yml`:
```yaml
- name: Push to Registry
  if: github.ref == 'refs/heads/main'
  uses: docker/build-push-action@v5
  with:
    push: true
    tags: yourorg/deepguard:latest,yourorg/deepguard:${{ github.sha }}
```

---

## Monitoring Setup

### Prometheus

Prometheus is pre-configured to scrape the DeepGuard API at `http://api:8000/metrics`.

Config at `deployment/prometheus/prometheus.yml`:
```yaml
scrape_configs:
  - job_name: 'deepguard-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

Access Prometheus UI: http://localhost:9090

### Grafana

Default credentials: `admin` / `deepguard123` (change in `.env`)

Pre-provisioned dashboards:
- **DeepGuard Overview**: Request rate, latency P50/P95/P99, error rate
- **Inference Metrics**: Detection throughput, face detection count, model latency

Access Grafana: http://localhost:3001

### Key Metrics to Monitor

| Metric | Alert Threshold |
|---|---|
| `http_requests_total` | Rate drop > 50% |
| `http_request_duration_seconds{p99}` | > 2 seconds |
| `http_errors_total` | > 1% error rate |
| Container memory usage | > 80% of limit |
| Disk space (weights, logs, db) | > 85% full |

---

## Backup & Recovery

### Database Backup

```bash
# Copy SQLite database
docker cp deepguard-api:/app/database/deepguard.db ./backups/deepguard-$(date +%Y%m%d).db

# Or use Docker volume backup
docker run --rm -v deepguard-db:/source -v $(pwd)/backups:/backup alpine \
  tar czf /backup/deepguard-db-$(date +%Y%m%d).tar.gz -C /source .
```

### Model Weights Backup

```bash
docker cp deepguard-api:/app/weights/best_model.pt ./backups/
```

### Automated Backup (Cron)

```bash
# Daily backup at 2 AM
echo "0 2 * * * /path/to/backup_script.sh" | crontab -
```

---

## Updating the Application

```bash
# Pull latest code
git pull origin main

# Rebuild and redeploy (zero-downtime with scale)
docker-compose build api
docker-compose up -d --no-deps api

# Or full restart
docker-compose down
docker-compose up -d --build
```

---

## Log Management

Application logs are written to `/app/logs/` inside the container, mapped to the `deepguard-logs` Docker volume.

```bash
# View recent logs
docker-compose logs --tail=100 api

# Export logs
docker cp deepguard-api:/app/logs/ ./exported-logs/
```

For centralized logging, consider sending logs to:
- **ELK Stack** (Elasticsearch + Logstash + Kibana)
- **Grafana Loki** (lightweight log aggregation)
- **CloudWatch** (AWS)
- **Papertrail** / **Datadog** (SaaS)
