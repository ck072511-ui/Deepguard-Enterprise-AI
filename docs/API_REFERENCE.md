# DeepGuard — API Reference

## Base URL
```
http://localhost:8000/api/v1
```

## Interactive Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## Authentication

Most endpoints are currently open (no auth required by default in development). The API supports JWT Bearer token authentication which can be enabled via configuration.

```bash
# Get JWT token
curl -X POST http://localhost:8000/api/v1/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "your-password"}'

# Use token in subsequent requests
curl -H "Authorization: Bearer <your-token>" http://localhost:8000/api/v1/...
```

---

## Endpoints

### 🏥 Health

#### `GET /health`
Check system health and readiness.

**Response `200 OK`:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "model_loaded": true,
  "database": "connected",
  "uptime_seconds": 3600
}
```

**cURL Example:**
```bash
curl http://localhost:8000/api/v1/health
```

---

### 🖼️ Image Detection

#### `POST /detect/image`
Detect deepfake in a single image file.

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | `File` | ✅ | Image file (JPEG, PNG, WebP, BMP) |
| `return_xai` | `bool` | ❌ | Include XAI heatmaps in response (default: `true`) |

**Response `200 OK`:**
```json
{
  "id": "uuid-string",
  "filename": "test.jpg",
  "media_type": "image",
  "status": "completed",
  "label": 1,
  "label_text": "FAKE",
  "confidence": 0.923,
  "faces_count": 1,
  "processing_time_ms": 245.3,
  "created_at": "2026-07-20T08:00:00Z",
  "completed_at": "2026-07-20T08:00:00.245Z",
  "explainability": {
    "real_probability": 0.077,
    "fake_probability": 0.923,
    "gradcam_image": "base64-encoded-jpeg...",
    "attention_image": "base64-encoded-jpeg...",
    "heatmap_image": "base64-encoded-jpeg...",
    "explanation_text": "High probability of synthetic face detected. GAN-generated blending artifacts visible around eye regions.",
    "confidence_level": "HIGH"
  }
}
```

**Response `422 Unprocessable Entity`** — Invalid file type or size exceeded.

**Response `400 Bad Request`** — No faces detected in the image.

**cURL Example:**
```bash
curl -X POST http://localhost:8000/api/v1/detect/image \
  -F "file=@/path/to/image.jpg"
```

**Python Example:**
```python
import requests

with open("test.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/api/v1/detect/image",
        files={"file": ("test.jpg", f, "image/jpeg")}
    )
print(response.json())
```

---

### 🎥 Video Detection

#### `POST /detect/video`
Detect deepfakes in a video file via sampled-frame analysis.

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | `File` | ✅ | Video file (MP4, AVI, MOV, MKV) |
| `sample_rate` | `int` | ❌ | Frame sampling interval (default: 30) |
| `max_frames` | `int` | ❌ | Maximum frames to analyze (default: 100) |

**Response `200 OK`:**
```json
{
  "id": "uuid-string",
  "filename": "video.mp4",
  "media_type": "video",
  "status": "completed",
  "label": 0,
  "label_text": "REAL",
  "confidence": 0.871,
  "faces_count": 47,
  "frames_analyzed": 47,
  "processing_time_ms": 12450.7,
  "created_at": "2026-07-20T08:00:00Z",
  "completed_at": "2026-07-20T08:00:12Z",
  "explainability": {
    "real_probability": 0.871,
    "fake_probability": 0.129,
    "explanation_text": "Natural facial micro-expressions and consistent lens noise patterns detected.",
    "confidence_level": "HIGH"
  }
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:8000/api/v1/detect/video \
  -F "file=@/path/to/video.mp4"
```

---

### 📦 Batch Detection

#### `POST /detect/batch`
Analyze up to 32 images in a single request.

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `files` | `File[]` | ✅ | Up to 32 image files |

**Response `200 OK`:**
```json
{
  "batch_id": "batch-uuid",
  "total": 5,
  "completed": 5,
  "failed": 0,
  "results": [
    {
      "filename": "img1.jpg",
      "label": 1,
      "label_text": "FAKE",
      "confidence": 0.95
    },
    {
      "filename": "img2.jpg",
      "label": 0,
      "label_text": "REAL",
      "confidence": 0.82
    }
  ]
}
```

---

### 📋 Prediction History

#### `GET /history`
Retrieve paginated detection history.

**Query Parameters:**

| Param | Type | Default | Description |
|---|---|---|---|
| `page` | `int` | `1` | Page number |
| `page_size` | `int` | `20` | Results per page (max 100) |
| `media_type` | `string` | — | Filter by `image` or `video` |
| `label` | `int` | — | Filter by `0` (real) or `1` (fake) |
| `status` | `string` | — | Filter by `completed`, `failed` |

**Response `200 OK`:**
```json
{
  "total": 247,
  "page": 1,
  "page_size": 20,
  "items": [
    {
      "id": "uuid",
      "filename": "test.jpg",
      "media_type": "image",
      "label": 1,
      "label_text": "FAKE",
      "confidence": 0.92,
      "status": "completed",
      "created_at": "2026-07-20T08:00:00Z"
    }
  ]
}
```

#### `GET /history/stats`
Get aggregate statistics over all predictions.

**Response `200 OK`:**
```json
{
  "total_predictions": 247,
  "fake_count": 180,
  "real_count": 67,
  "fake_percentage": 72.9,
  "average_confidence": 0.871,
  "average_processing_time_ms": 312.5,
  "total_faces_analyzed": 1204
}
```

#### `DELETE /history/{id}`
Delete a detection record by ID.

**Response `204 No Content`**

---

### 🤖 Model Registry

#### `GET /models`
List all registered model versions.

**Response `200 OK`:**
```json
{
  "models": [
    {
      "id": "uuid",
      "name": "vit_base_deepguard",
      "version": "1.0.0",
      "architecture": "vit_base_patch16_224",
      "is_active": true,
      "accuracy": 0.954,
      "auc_roc": 0.978,
      "created_at": "2026-07-20T08:00:00Z"
    }
  ]
}
```

#### `POST /models`
Register a new model version.

**Request Body:**
```json
{
  "name": "vit_base_deepguard",
  "version": "2.0.0",
  "architecture": "vit_base_patch16_224",
  "weights_path": "weights/best_model_v2.pt",
  "description": "Retrained with additional DFDC data"
}
```

#### `PUT /models/{id}/activate`
Set a model version as the active inference model.

---

### 📤 Upload Management

#### `POST /upload`
Stage a file upload before triggering detection.

**Request:** `multipart/form-data` with field `file`.

**Response `200 OK`:**
```json
{
  "upload_id": "uuid",
  "filename": "video.mp4",
  "size_bytes": 52428800,
  "content_type": "video/mp4",
  "expires_at": "2026-07-20T09:00:00Z"
}
```

---

### 📈 Metrics (Prometheus)

#### `GET /metrics` *(not in Swagger)*
Prometheus metrics scrape endpoint.

**Sample metrics exposed:**
```
http_requests_total{method="POST",endpoint="/detect/image",status="200"} 42
http_request_duration_seconds{endpoint="/detect/image",quantile="0.95"} 0.312
deepguard_inference_duration_seconds{model="vit_base"} 0.245
deepguard_faces_detected_total 1204
```

---

## Error Response Format

All errors follow a consistent JSON envelope:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "File type not supported. Accepted: JPEG, PNG, WebP, BMP",
    "details": {
      "field": "file",
      "received": "application/pdf"
    }
  },
  "request_id": "X-Request-ID-header-value",
  "timestamp": "2026-07-20T08:00:00Z"
}
```

## HTTP Status Codes

| Code | Meaning |
|---|---|
| `200` | Success |
| `204` | No content (delete) |
| `400` | Bad request (no faces detected, etc.) |
| `401` | Unauthorized (invalid/missing token) |
| `413` | Payload too large (file size exceeded) |
| `422` | Validation error (wrong file type, missing field) |
| `429` | Too many requests (rate limit exceeded) |
| `500` | Internal server error |
| `503` | Service unavailable (model not loaded) |

## Rate Limiting

Default: **60 requests per minute** per IP address.

When exceeded, returns `429 Too Many Requests` with header:
```
Retry-After: 15
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1721462460
```
