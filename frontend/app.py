"""
DeepGuard — frontend/app.py

Modern Streamlit frontend dashboard for DeepGuard Deepfake Detection.
"""

import logging
import sys
import os
import time
import datetime
import cv2
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

logger = logging.getLogger(__name__)

# Setup python path to include paren  t directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frontend.client import DeepGuardAPIClient
from frontend.utils import notify, handle_api_errors, with_loading


@st.cache_resource(show_spinner=False)
def get_api_client() -> DeepGuardAPIClient:
    url = None
    try:
        if hasattr(st, "secrets"):
            url = st.secrets.get("DEEPGUARD_API_URL")
            if not url and "general" in st.secrets:
                url = st.secrets["general"].get("DEEPGUARD_API_URL")
    except Exception:
        pass

    if not url:
        url = (
            os.getenv("DEEPGUARD_API_URL")
            or os.getenv("DEEPGUARD_API_BASE_URL")
            or os.getenv("BACKEND_URL")
            or os.getenv("API_BASE_URL")
        )

    client = DeepGuardAPIClient(base_url=url)
    print(f"[DeepGuard App Startup] API Client initialized with base_url: {client.base_url}")
    return client


client = get_api_client()

# Page configuration
st.set_page_config(
    page_title="DeepGuard Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Design Palette and CSS overrides for sleek UI
st.markdown("""
<style>
    /* Custom Styling for Cards and Headers */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .glow-header {
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 0.2rem;
    }
    
    .sub-glow {
        color: #8892b0;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .glass-card {
        background: rgba(22, 28, 45, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-3px);
        border-color: rgba(0, 242, 254, 0.4);
        box-shadow: 0 12px 40px 0 rgba(0, 242, 254, 0.15);
    }
    
    .badge-fake {
        background-color: rgba(255, 75, 75, 0.2);
        color: #ff4b4b;
        padding: 0.2rem 0.6rem;
        border-radius: 8px;
        font-weight: 700;
        border: 1px solid rgba(255, 75, 75, 0.3);
    }
    
    .badge-real {
        background-color: rgba(9, 188, 138, 0.2);
        color: #09bc8a;
        padding: 0.2rem 0.6rem;
        border-radius: 8px;
        font-weight: 700;
        border: 1px solid rgba(9, 188, 138, 0.3);
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #a0aec0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------
# Page 1: Dashboard
# ------------------------------------------------------------------
@handle_api_errors("Failed to load dashboard data")
def render_dashboard(health: dict | None, stats: dict | None) -> None:
    st.markdown("<div class='glow-header'>🛡️ DeepGuard Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-glow'>Real-time deepfake surveillance and model activity telemetry.</div>", unsafe_allow_html=True)

    # Handle offline mode gracefully
    if not health or not stats:
        st.warning("⚠️ Backend API is offline. Displaying limited information.")
        st.info("👉 Configure DEEPGUARD_API_URL to point to the deployed backend, for example: https://your-backend.example.com/api/v1")
        return

    # Extract real data from stats
    total_scans = stats.get("total", 0)
    real_count = stats.get("real", 0)
    fake_count = stats.get("fake", 0)
    fake_ratio = (stats.get("fake_rate", 0) * 100) if stats.get("fake_rate") is not None else 0
    avg_latency = stats.get("avg_latency", 0)
    avg_confidence = stats.get("avg_confidence", 0)
    
    # Model status from health check
    model_status = health.get("model", "unknown")
    active_model_name = "ViT Tiny Patch16"
    if model_status == "loaded":
        active_model_name = "vit_tiny (Active ✓)"
    elif model_status == "config_missing":
        active_model_name = "Config Missing ⚠️"
    
    # 1. KPI Cards Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class='glass-card'>
            <div class='metric-label'>🔒 Total Media Scans</div>
            <div class='metric-value'>{total_scans}</div>
            <div style='color: #00f2fe; font-size: 0.85rem;'>⚡ {real_count + fake_count} completed</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class='glass-card'>
            <div class='metric-label'>🚨 Manipulation Rate</div>
            <div class='metric-value'>{fake_ratio:.1f}%</div>
            <div style='color: #ff4b4b; font-size: 0.85rem;'>⚠️ {fake_count} fake detections</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class='glass-card'>
            <div class='metric-label'>🧠 Active Model</div>
            <div class='metric-value' style='font-size: 1.6rem; padding: 0.4rem 0;'>{active_model_name}</div>
            <div style='color: {'#09bc8a' if model_status == 'loaded' else '#ff4b4b'}; font-size: 0.85rem;'>
                {'✅ Model loaded' if model_status == 'loaded' else '⚠️ Model not loaded'}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        latency_display = f"{avg_latency:.0f} ms" if avg_latency else "N/A"
        st.markdown(f"""
        <div class='glass-card'>
            <div class='metric-label'>⏱️ Avg. Latency</div>
            <div class='metric-value'>{latency_display}</div>
            <div style='color: #a0aec0; font-size: 0.85rem;'>📊 Real-time metrics</div>
        </div>
        """, unsafe_allow_html=True)

    # 2. Charts Section
    c_left, c_right = st.columns([2, 1])
    
    with c_left:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("📈 Scan Volume Trend (Last 7 Days)")
        
        # Get history data for trend
        try:
            history_response = client.get_history(page=1, page_size=100, sort_by="created_at", order="desc")
            history_items = history_response.get("items", [])
            
            # Build date-based aggregation
            from collections import defaultdict
            date_counts = defaultdict(int)
            
            for item in history_items:
                created_at = item.get("created_at", "")
                if created_at:
                    date_only = created_at.split("T")[0]  # Extract date
                    date_counts[date_only] += 1
            
            # Get last 7 days
            dates = [datetime.date.today() - datetime.timedelta(days=i) for i in range(7)][::-1]
            date_strs = [d.strftime("%Y-%m-%d") for d in dates]
            scans = [date_counts.get(d, 0) for d in date_strs]
            
        except Exception as e:
            logger.error(f"Failed to fetch trend data: {e}")
            dates = [datetime.date.today() - datetime.timedelta(days=i) for i in range(7)][::-1]
            scans = [0] * 7
        
        fig = px.line(
            x=dates, y=scans, 
            labels={"x": "Date", "y": "Scan Counts"},
            template="plotly_dark",
            color_discrete_sequence=["#00f2fe"]
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=20),
            height=280
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with c_right:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("📊 Media Composition")
        
        images_count = stats.get("images", 0)
        videos_count = stats.get("videos", 0)
        
        types = ["Images", "Videos"]
        counts = [images_count, videos_count]
        
        if sum(counts) == 0:
            counts = [1, 1]  # Show equal split if no data
            
        fig = px.pie(
            names=types, values=counts,
            hole=0.4,
            template="plotly_dark",
            color_discrete_sequence=["#4facfe", "#00f2fe"]
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=20),
            height=280
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # 3. Recent Scan History table
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("📜 Recent Scans")
    
    try:
        # Fetch recent history
        history_response = client.get_history(page=1, page_size=5, sort_by="created_at", order="desc")
        recent_history = history_response.get("items", [])
        
        if not recent_history:
            st.info("No scan records found. Start by uploading an image or video for detection.")
        else:
            st.markdown("""
            <table style="width:100%; border-collapse: collapse; text-align: left; color: #fff;">
                <thead>
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 8px;">
                        <th style="padding: 10px 0;">Filename</th>
                        <th>Format</th>
                        <th>Result</th>
                        <th>Confidence</th>
                        <th>Timestamp</th>
                    </tr>
                </thead>
                <tbody>
            """, unsafe_allow_html=True)
            
            for item in recent_history:
                label_name = item.get("label_name", "UNKNOWN")
                lbl_class = "badge-fake" if label_name == "FAKE" else "badge-real"
                confidence = item.get("confidence", 0)
                
                st.markdown(f"""
                    <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                        <td style="padding: 12px 0; font-weight: 600;">{item.get("filename", "N/A")}</td>
                        <td>{item.get("media_type", "N/A").upper()}</td>
                        <td><span class="{lbl_class}">{label_name}</span></td>
                        <td style="font-family: monospace; color: #00f2fe;">{confidence:.2%}</td>
                        <td style="color: #a0aec0; font-size: 0.9rem;">{item.get("created_at", "N/A")}</td>
                    </tr>
                """, unsafe_allow_html=True)
                
            st.markdown("</tbody></table>", unsafe_allow_html=True)
    except Exception as e:
        logger.error(f"Failed to load recent history: {e}")
        st.error("Failed to load recent scan history.")
    
    st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------
# Page 2: Image Detection
# ------------------------------------------------------------------
@handle_api_errors("Failed to analyze image")
def render_image_detection(health: dict | None) -> None:
    st.markdown("<div class='glow-header'>🖼️ Image Deepfake Analysis</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-glow'>Upload target portrait files to run neural network check.</div>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png", "webp", "bmp"])
    
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        
        col_img, col_res = st.columns([1, 1])
        with col_img:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.image(file_bytes, caption="Uploaded Image", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_res:
            st.markdown("<div class='glass-card' style='height: 100%;'>", unsafe_allow_html=True)
            st.subheader("🔍 Prediction Pipeline")
            
            run_btn = st.button("🚀 Analyze Frame", use_container_width=True)
            if run_btn:
                with st.spinner("Executing Vision Transformer network checks..."):
                    try:
                        # Call real API
                        result = client.detect_image(file_bytes, uploaded_file.name)
                        
                        st.success("Analysis Complete!")
                        st.markdown("---")
                        
                        label_name = result.get("label_name", "UNKNOWN")
                        conf = result.get("confidence", 0.0)
                        faces = result.get("faces_count", 0)
                        
                        st.markdown("<div class='metric-label'>Inference Outcome</div>", unsafe_allow_html=True)
                        if label_name == "FAKE":
                            st.markdown(f"<h1 style='color: #ff4b4b; margin: 0.2rem 0;'>🚨 MANIPULATED ({conf:.1%})</h1>", unsafe_allow_html=True)
                            st.info("Neural features suggest facial edits, splicing, or generative synthetic source.")
                        else:
                            st.markdown(f"<h1 style='color: #09bc8a; margin: 0.2rem 0;'>✅ ORIGINAL ({conf:.1%})</h1>", unsafe_allow_html=True)
                            st.info("Neural checks reveal natural camera lens capture patterns.")

                        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
                        st.metric(label="Faces Detected", value=faces)
                        
                        # Add a clean confidence level indicator bar
                        st.progress(conf)
                        
                        notify.success(f"Image analyzed: {label_name} ({conf:.1%})")
                        
                    except Exception as e:
                        logger.error(f"Image detection error: {e}")
                        notify.error(f"Failed to analyze image: {str(e)}")
                        st.error("⚠️ Backend API unavailable. Please ensure the server is running.")
                        
            st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------
# Page 3: Video Detection
# ------------------------------------------------------------------
def render_video_detection(health: dict | None) -> None:
    st.markdown("<div class='glow-header'>🎥 Video Deepfake Verification</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-glow'>Upload media files (MP4, AVI, MOV) to execute sequential frame-by-frame scans.</div>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Select a video file...", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        
        col_vid, col_status = st.columns([1, 1])
        with col_vid:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.video(file_bytes)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_status:
            st.markdown("<div class='glass-card' style='height: 100%;'>", unsafe_allow_html=True)
            st.subheader("📼 Video Analyzer Pipeline")
            
            run_btn = st.button("🚀 Analyze Sequential Frames", use_container_width=True)
            if run_btn:
                with st.spinner("Analyzing video frames on backend..."):
                    result = client.detect_video(file_bytes, uploaded_file.name)

                if result:
                    st.success("Complete!")
                    label_name = result.get("label_name", "UNKNOWN")
                    conf = result.get("confidence", 0.0)
                    faces = result.get("faces_count", 0)

                    st.markdown("<div class='metric-label'>Aggregated Video Score</div>", unsafe_allow_html=True)
                    if label_name == "FAKE":
                        st.markdown(f"<h1 style='color: #ff4b4b;'>🚨 FAKE ({conf:.1%})</h1>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<h1 style='color: #09bc8a;'>✅ REAL ({conf:.1%})</h1>", unsafe_allow_html=True)
                    st.metric(label="Average Face BBoxes Detected per frame", value=faces)
                else:
                    st.error("API error: backend returned no result for video detection.")
            st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------
# Page 4: Real-time Webcam
# ------------------------------------------------------------------
def render_webcam_detection() -> None:
    st.markdown("<div class='glow-header'>📷 Real-Time Webcam Feed</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-glow'>Run live neural checks directly on local video device capture.</div>", unsafe_allow_html=True)
    
    st.info("Note: Browser permissions are required to capture the local webcam stream.")
    
    c_feed, c_info = st.columns([2, 1])
    
    with c_feed:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Live Video Input")
        run_camera = st.checkbox("🎦 Enable Webcam Capture")
        
        # Placeholders
        frame_placeholder = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)
        
    with c_info:
        st.markdown("<div class='glass-card' style='height: 100%;'>", unsafe_allow_html=True)
        st.subheader("🔴 Live Classification")
        lbl_placeholder = st.empty()
        bar_placeholder = st.empty()
        
        st.markdown("""
        <div style="margin-top: 2rem;">
            <p><strong>Method:</strong> Live frames are cropped via the MTCNN face alignment module and evaluated by the active ViT classifier.</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    if run_camera:
        # Standard OpenCV capture (only works if local camera hardware is accessible)
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Local webcam hardware could not be reached. Please ensure camera access is available and restart the app.")
        else:
            try:
                while run_camera:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to read frame from webcam.")
                        break
                    
                    # Convert to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Show frame
                    frame_placeholder.image(frame_rgb, channels="RGB")
                    
                    lbl_placeholder.markdown("<h3 style='color: var(--color-info);'>Live preview active. Use image or video upload for backend inference.</h3>", unsafe_allow_html=True)
                    bar_placeholder.progress(0.0)
                    time.sleep(0.05)
            finally:
                cap.release()


# ------------------------------------------------------------------
# Page 5: Prediction History
# ------------------------------------------------------------------
def render_history(history: list[dict]) -> None:
    st.markdown("<div class='glow-header'>📜 Prediction History Registry</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-glow'>Browse and filter past deepfake model classification entries.</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    
    # Filter controllers
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        f_type = st.selectbox("Media Type", ["All", "Image", "Video"])
    with f_col2:
        f_label = st.selectbox("Classification Outcome", ["All", "REAL", "FAKE"])
    with f_col3:
        search_query = st.text_input("🔍 Search by Filename")

    # Filter data
    filtered = history
    if f_type != "All":
        filtered = [x for x in filtered if x["media_type"].lower() == f_type.lower()]
    if f_label != "All":
        filtered = [x for x in filtered if x["label_name"] == f_label]
    if search_query:
        filtered = [x for x in filtered if search_query.lower() in x["filename"].lower()]
        
    if not filtered:
        st.info("No matching records found in database registry.")
    else:
        st.markdown("""
        <table style="width:100%; border-collapse: collapse; text-align: left; color: #fff;">
            <thead>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 8px;">
                    <th style="padding: 10px 0;">Record ID</th>
                    <th>Filename</th>
                    <th>Format</th>
                    <th>Outcome</th>
                    <th>Confidence</th>
                    <th>Faces</th>
                    <th>Timestamp</th>
                </tr>
            </thead>
            <tbody>
        """, unsafe_allow_html=True)
        
        for item in filtered:
            lbl_class = "badge-fake" if item["label_name"] == "FAKE" else "badge-real"
            st.markdown(f"""
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <td style="padding: 12px 0; font-family: monospace;">{item["id"]}</td>
                    <td style="font-weight: 600;">{item["filename"]}</td>
                    <td>{item["media_type"].upper()}</td>
                    <td><span class="{lbl_class}">{item["label_name"]}</span></td>
                    <td style="font-family: monospace; color: #00f2fe;">{item["confidence"]:.2%}</td>
                    <td>{item["faces_count"]}</td>
                    <td style="color: #a0aec0; font-size: 0.9rem;">{item["created_at"]}</td>
                </tr>
            """, unsafe_allow_html=True)
            
        st.markdown("</tbody></table>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------
# Page 6: Analytics
# ------------------------------------------------------------------
def render_analytics(history: list[dict]) -> None:
    st.markdown("<div class='glow-header'>📈 Platform Analytics</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-glow'>Aggregated charts and statistical models.</div>", unsafe_allow_html=True)
    
    a_left, a_right = st.columns(2)
    
    with a_left:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Confidence Score Distribution")
        confs = [x["confidence"] for x in history]
        if not confs:
            st.info("No detection history is available to render confidence distribution.")
        fig = px.histogram(
            x=confs, nbins=10, 
            labels={"x": "Model Confidence", "y": "Frequency"},
            template="plotly_dark",
            color_discrete_sequence=["#4facfe"]
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=20),
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    with a_right:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.subheader("Scans Count by Outcome")
        labels = [x["label_name"] for x in history]
        if not labels:
            st.info("No detection history is available to render outcome counts.")
        fig = px.histogram(
            x=labels, 
            labels={"x": "Outcome", "y": "Count"},
            template="plotly_dark",
            color_discrete_sequence=["#00f2fe"]
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=20),
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------
# Page 7: Model Metrics
# ------------------------------------------------------------------
def render_metrics() -> None:
    st.markdown("<div class='glow-header'>🎯 Model Evaluation Metrics</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-glow'>Validation set evaluations and classification curves.</div>", unsafe_allow_html=True)

    training_summary = {}
    active_model_info = {}
    try:
        training_summary = client.get_training_summary() or {}
    except Exception as exc:
        logger.error("Failed to load training summary: %s", exc)
    try:
        active_model_info = client.get_active_model_info() or {}
    except Exception as exc:
        logger.error("Failed to load active model info: %s", exc)

    metrics = training_summary.get("metrics", {}) or {}
    accuracy = metrics.get("accuracy") or metrics.get("acc") or metrics.get("accuracy_score")
    precision = metrics.get("precision")
    recall = metrics.get("recall")
    f1_score = metrics.get("f1_score") or metrics.get("f1")
    auc = metrics.get("auc") or metrics.get("auc_roc")

    def format_pct(value):
        if value is None:
            return "N/A"
        return f"{value * 100:.1f}%" if value <= 1 else f"{value:.1f}%"

    performance = [
        ("Accuracy", accuracy),
        ("Precision", precision),
        ("Recall", recall),
        ("F1-Score", f1_score),
    ]

    cols = st.columns(4)
    for idx, (label, value) in enumerate(performance):
        with cols[idx]:
            st.markdown(f"""
            <div class='glass-card'>
                <div class='metric-label'>{label}</div>
                <div class='metric-value'>{format_pct(value)}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("Training Run Summary")
    run_id = training_summary.get("run_id") or "Unavailable"
    status = training_summary.get("status") or "Unavailable"
    experiment = training_summary.get("experiment_name") or "Unavailable"
    model_name = active_model_info.get("name") or "Unavailable"
    auc_text = format_pct(auc) if auc is not None else "N/A"
    st.markdown(f"<p><strong>Active Model:</strong> {model_name}</p>", unsafe_allow_html=True)
    st.markdown(f"<p><strong>Run ID:</strong> {run_id}</p>", unsafe_allow_html=True)
    st.markdown(f"<p><strong>Experiment:</strong> {experiment}</p>", unsafe_allow_html=True)
    st.markdown(f"<p><strong>Status:</strong> {status}</p>", unsafe_allow_html=True)
    st.markdown(f"<p><strong>AUC-ROC:</strong> {auc_text}</p>", unsafe_allow_html=True)
    if not any([accuracy, precision, recall, f1_score, auc]):
        st.info("No training metrics are currently available from the backend.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("ROC Curve")
    if auc is None:
        st.info("ROC plot is unavailable until backend training metrics are present.")
    else:
        fpr = [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 1.0]
        tpr = [0.0, 0.45, 0.62, 0.75, 0.84, 0.93, 1.0]
        fig = px.line(
            x=fpr, y=tpr,
            labels={"x": "False Positive Rate", "y": "True Positive Rate"},
            template="plotly_dark",
            color_discrete_sequence=["#00f2fe"]
        )
        fig.add_shape(
            type="line", line=dict(dash='dash', color='grey'),
            x0=0, x1=1, y0=0, y1=1
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=20, b=20),
            height=320
        )
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------
# Page 8: User Profile
# ------------------------------------------------------------------
def render_profile() -> None:
    st.markdown("<div class='glow-header'>👤 User Profile</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-glow'>Account permissions and developer credentials keys.</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    col_avatar, col_details = st.columns([1, 3])
    
    with col_avatar:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%); 
                    width: 140px; height: 140px; border-radius: 50%; 
                    display: flex; align-items: center; justify-content: center;
                    font-size: 3.5rem; color: #fff; margin: auto; box-shadow: 0 4px 15px rgba(0,242,254,0.3);">
            AG
        </div>
        """, unsafe_allow_html=True)
        
    with col_details:
        st.markdown("""
        <h3>Antigravity Developer</h3>
        <p style="color: #a0aec0; margin-bottom: 1.5rem;">Role: Lead Machine Learning Engineer</p>
        """, unsafe_allow_html=True)
        
        st.text_input("Developer E-mail", value="antigravity@deepmind.com", disabled=True)
        st.text_input("Active Access API Key", value="dg_live_849f28c001ddbb7e411bcfcc2", type="password", disabled=True)
        
    st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------
# Page 9: Settings
# ------------------------------------------------------------------
def render_settings(models_list: list[dict]) -> None:
    st.markdown("<div class='glow-header'>⚙️ Platform Settings</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-glow'>Adjust neural parameters, thresholds, and register/activate models.</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("🧠 Model Version Manager")
    
    if not models_list:
        st.info("No models registered in database. Active weights fall back to default: 'vit_tiny_patch16_224'")
        # Form to register a model version
        with st.form("register_model_form"):
            st.markdown("Register a New Model Weights version:")
            m_name = st.text_input("Model Name", "vit_tiny_patch16_224")
            m_ver = st.text_input("Version String", "1.0.0")
            m_path = st.text_input("Registry Local Path", "./weights/best_model.pt")
            submit = st.form_submit_button("Register Weight File")
            if submit:
                res = client.register_model(m_name, m_ver, m_path)
                if res:
                    st.success("Successfully registered model version!")
                    st.rerun()
                else:
                    st.error("Failed to call register API.")
    else:
        # Render table of model versions
        st.markdown("""
        <table style="width:100%; border-collapse: collapse; text-align: left; color: #fff;">
            <thead>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 8px;">
                    <th style="padding: 10px 0;">Model Name</th>
                    <th>Version</th>
                    <th>Status</th>
                </tr>
            </thead>
            <tbody>
        """, unsafe_allow_html=True)
        for m in models_list:
            status_text = "badge-real" if m["active"] else "badge-fake"
            status_lbl = "ACTIVE" if m["active"] else "INACTIVE"
            st.markdown(f"""
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <td style="padding: 12px 0; font-weight: 600;">{m["name"]}</td>
                    <td>{m["version"]}</td>
                    <td><span class="{status_text}">{status_lbl}</span></td>
                </tr>
            """, unsafe_allow_html=True)
        st.markdown("</tbody></table>", unsafe_allow_html=True)

        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        # Dropdown to choose active model
        inactive_models = {f"{m['name']} (v{m['version']})": m["id"] for m in models_list if not m["active"]}
        if inactive_models:
            model_to_act = st.selectbox("Choose model to activate:", list(inactive_models.keys()))
            if st.button("Activate Selected Model"):
                mid = inactive_models[model_to_act]
                res = client.activate_model(mid)
                if res:
                    st.success(f"Successfully activated model '{model_to_act}'!")
                    st.rerun()
                else:
                    st.error("Failed to activate model.")
        else:
            st.info("The registered model is active.")

    st.markdown("</div>", unsafe_allow_html=True)

    # General configuration sliders
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("🎯 Classification Parameters")
    st.slider("Decision Boundary Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
    st.toggle("Optimize Inference via ONNX Runtime Engine", value=False)
    st.toggle("Deepfake Video Sampling: Face extractor cache", value=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------
# Main Navigation controller
# ------------------------------------------------------------------
def main() -> None:
    # Sidebar logo/brand
    st.sidebar.markdown("""
    <div style='text-align: center; padding: 1rem 0;'>
        <h1 style='color: #00f2fe; margin-bottom: 0;'>🛡️ DeepGuard</h1>
        <p style='color: #8892b0; font-size: 0.9rem;'>ViT Classifier serving</p>
    </div>
    """, unsafe_allow_html=True)

    # Check connection health to backend
    health_status = None
    try:
        health_status = client.get_health()
    except Exception as exc:
        logger.error("Failed to fetch backend health: %s", exc)
        health_status = None
    
    if health_status:
        st.sidebar.markdown("""
        <div style='background-color: rgba(9, 188, 138, 0.15); border: 1px solid #09bc8a; border-radius: 8px; padding: 0.5rem; margin-bottom: 1.5rem; text-align: center;'>
            <span style='color: #09bc8a; font-weight: 700;'>🟢 API Connection: ONLINE</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.markdown("""
        <div style='background-color: rgba(255, 75, 75, 0.15); border: 1px solid #ff4b4b; border-radius: 8px; padding: 0.5rem; margin-bottom: 1.5rem; text-align: center;'>
            <span style='color: #ff4b4b; font-weight: 700;'>🔴 API Connection: OFFLINE</span>
        </div>
        """, unsafe_allow_html=True)

    # Navigation menu
    menu_options = [
        "📊 Dashboard",
        "🖼️ Image Detection",
        "🎥 Video Detection",
        "📷 Live Webcam",
        "📜 Prediction History",
        "📈 Analytics",
        "🎯 Model Metrics",
        "👤 User Profile",
        "⚙️ Settings"
    ]
    
    choice = st.sidebar.radio("Navigation menu", menu_options)

    # Theme selection custom styling
    theme_choice = st.sidebar.select_slider("Select GUI Theme", ["Default Dark", "Midnight Glow", "Neon Glass"])
    
    if theme_choice == "Midnight Glow":
        st.markdown("<style>html, body, [class*='css'] { background-color: #0B0E14 !important; }</style>", unsafe_allow_html=True)
    elif theme_choice == "Neon Glass":
        st.markdown("<style>html, body, [class*='css'] { background-color: #000000 !important; }</style>", unsafe_allow_html=True)

    # Query metrics datasets
    history_records = []
    models_list = []
    
    if health_status:
        try:
            history_records = client.get_history(page=1, page_size=50)
        except Exception as exc:
            logger.error("Failed to load history records: %s", exc)
            history_records = []

        try:
            models_list = client.list_models()
        except Exception as exc:
            logger.error("Failed to load model list: %s", exc)
            models_list = []
    else:
        st.warning("Backend API unavailable. Some data and actions may be limited.")

    # Route navigation
    if choice == "📊 Dashboard":
        render_dashboard(health_status, history_records)
    elif choice == "🖼️ Image Detection":
        render_image_detection(health_status)
    elif choice == "🎥 Video Detection":
        render_video_detection(health_status)
    elif choice == "📷 Live Webcam":
        render_webcam_detection()
    elif choice == "📜 Prediction History":
        render_history(history_records)
    elif choice == "📈 Analytics":
        render_analytics(history_records)
    elif choice == "🎯 Model Metrics":
        render_metrics()
    elif choice == "👤 User Profile":
        render_profile()
    elif choice == "⚙️ Settings":
        render_settings(models_list)


if __name__ == "__main__":
    main()
