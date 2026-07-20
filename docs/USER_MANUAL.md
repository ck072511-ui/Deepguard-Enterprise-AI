# DeepGuard — User Manual

Welcome to DeepGuard! This guide walks you through using the DeepGuard deepfake detection system.

---

## Getting Started

### Accessing the Dashboard

After starting the application, open your web browser and navigate to:

```
http://localhost:80          # Via Nginx (full Docker stack)
http://localhost:8000/docs   # Direct API access
```

Or open `frontend/index.html` directly in your browser for the standalone dashboard.

---

## Dashboard Overview

The DeepGuard dashboard consists of **10 modules** accessible from the sidebar navigation:

| Module | Icon | Purpose |
|---|---|---|
| Dashboard | 🏠 | Overview statistics and recent activity |
| Image Detection | 🖼️ | Upload images for deepfake analysis |
| Video Detection | 🎥 | Upload videos for deepfake analysis |
| Real-Time Webcam | 📷 | Live camera deepfake detection |
| Prediction History | 📋 | Browse all past detection results |
| Analytics | 📊 | Charts and usage statistics |
| Model Metrics | 📈 | Model performance metrics |
| Settings | ⚙️ | Configure model and system settings |
| User Profile | 👤 | User account information |
| Dark Mode | 🌙 | Toggle between light and dark themes |

---

## Image Detection

### Uploading an Image

1. Click **"Image Detection"** in the sidebar
2. Drag and drop an image onto the upload zone, or click **"Choose File"**
3. Supported formats: **JPEG, PNG, WebP, BMP**
4. Maximum file size: **10 MB**
5. Click **"Analyze"** to start detection

### Reading the Results

After analysis completes, you will see:

| Field | Description |
|---|---|
| **Verdict** | REAL or FAKE (with colored badge) |
| **Confidence** | 0–100% confidence score |
| **Faces Detected** | Number of faces found in the image |
| **Processing Time** | Time taken to analyze (milliseconds) |

### Explainability Tabs

The results panel includes three visualization tabs:

- **🔥 Heatmap**: Jet colormap overlay showing regions of interest
- **📡 GradCAM**: Gradient-weighted class activation map highlighting discriminative regions
- **👁️ Attention**: Vision Transformer self-attention rollout showing where the model looked

> **Tip**: Red/yellow regions in the heatmap indicate areas the model found most suspicious.

### Understanding Confidence Scores

| Confidence | Interpretation |
|---|---|
| 90–100% | Very high confidence — strong evidence |
| 70–90% | High confidence — likely correct |
| 50–70% | Moderate confidence — uncertain, verify manually |
| < 50% | Low confidence — inconclusive result |

---

## Video Detection

### Uploading a Video

1. Click **"Video Detection"** in the sidebar
2. Upload your video file (MP4, AVI, MOV, MKV)
3. Maximum file size: **500 MB**
4. Click **"Analyze Video"**

### What Happens During Analysis

DeepGuard samples frames from the video at regular intervals (every 30 frames by default), extracts faces from each frame, runs the ViT model on all face crops, and aggregates results across all frames.

**Progress indicator** shows:
- Frames analyzed / total frames
- Current processing speed (frames/sec)
- Estimated time remaining

### Video Results

Results show:
- Overall verdict (majority vote across frames)
- Frame-by-frame confidence chart
- Aggregate statistics (average confidence, min/max per frame)

---

## Real-Time Webcam Detection

1. Click **"Real-Time Webcam"** in the sidebar
2. Click **"Start Camera"** — your browser will request camera permission
3. Grant the permission when prompted
4. The system will continuously analyze your webcam feed
5. Real-time verdict and confidence display updates every second
6. Click **"Stop Camera"** to end the session

> **Note**: Real-time webcam detection runs inference in the browser by batching frames every 1–2 seconds. Results may be slightly delayed compared to the live video feed.

---

## Prediction History

1. Click **"Prediction History"** in the sidebar
2. Browse all past predictions in the table

### Filtering Results

Use the filter controls to narrow results:
- **Media Type**: Image / Video
- **Verdict**: Real / Fake
- **Status**: Completed / Failed
- **Date Range**: Custom date range picker

### Sorting and Pagination

- Click column headers to sort
- Use page controls at the bottom to navigate
- Adjust page size (10 / 20 / 50 / 100 results)

### Exporting Results

Click **"Export CSV"** to download the current filtered results as a spreadsheet.

---

## Analytics

The Analytics page displays:

| Chart | Description |
|---|---|
| **Detection Timeline** | Line chart of predictions over time |
| **Real vs. Fake Ratio** | Donut chart of overall verdicts |
| **Confidence Distribution** | Histogram of confidence scores |
| **Processing Time Trend** | API latency over time |

Hover over any chart element for detailed tooltips.

---

## Model Metrics

The Model Metrics page shows the performance of the currently active model:

| Metric | Description |
|---|---|
| **Accuracy** | Overall correct classification rate |
| **AUC-ROC** | Area under the ROC curve |
| **F1 Score** | Harmonic mean of precision and recall |
| **Precision** | True positive rate among predicted positives |
| **Recall** | True positive rate among actual positives |
| **False Positive Rate** | Legitimate images incorrectly flagged as fake |
| **Inference Speed** | Average milliseconds per image |

---

## Settings

### Model Configuration

- **Active Model**: Switch between registered model versions
- **Inference Mode**: PyTorch (flexible) or ONNX Runtime (faster)
- **Confidence Threshold**: Minimum confidence to report FAKE (default: 0.5)
- **Face Detection Backend**: MediaPipe, MTCNN, or RetinaFace

### API Settings

- **API URL**: Backend server address (useful when running separately)
- **Request Timeout**: Maximum wait time per request

---

## Keyboard Shortcuts

| Shortcut | Action |
|---|---|
| `Ctrl + U` | Open image upload dialog |
| `Ctrl + H` | Go to prediction history |
| `Ctrl + D` | Toggle dark mode |
| `Esc` | Close current dialog |

---

## Troubleshooting

### "No faces detected" Error

This means the face extractor could not find any faces in your image/video.
- Ensure the image contains at least one clearly visible human face
- Check that the face is not too small (minimum ~50×50 pixels)
- Try a different image with a more frontal face angle

### "Service Unavailable" Error

The backend API is not running. Start it with:
```bash
python -m uvicorn backend.main:app --port 8000 --reload
```

### Webcam Not Working

- Ensure you are using a browser that supports `getUserMedia` (Chrome, Firefox, Edge)
- Check that no other application is using the camera
- Verify browser camera permissions in Settings → Privacy → Camera

### Results Seem Wrong (Low Confidence)

- DeepGuard works best with clear, well-lit frontal face images
- Very compressed or low-resolution images may give uncertain results
- Try adjusting the confidence threshold in Settings
- Consider this a tool to assist — always combine with human judgment for important decisions
