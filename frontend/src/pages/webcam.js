/* ============================================================================
   DeepGuard Frontend Page — Real-time Webcam Feed
   ============================================================================ */

export function renderWebcamDetection(apiClient, container) {
    container.innerHTML = `
        <div class="animate-fade-in">
            <div class="detection-view-grid">
                <!-- Webcam Capture Frame Left -->
                <div class="glass-card flex-col" style="display: flex; gap: 20px;">
                    <div class="card-header">
                        <h3>Webcam Feed</h3>
                        <span class="card-header-sub">Live video feed from browser</span>
                    </div>

                    <div style="display: flex; justify-content: center; position: relative;">
                        <div class="webcam-live-box">
                            <video id="live-webcam-video" autoplay playsinline muted></video>
                            <canvas id="live-webcam-canvas" class="webcam-canvas-overlay"></canvas>
                        </div>
                    </div>

                    <div style="display: flex; gap: 12px; justify-content: center; margin-top: 10px;">
                        <button class="btn-primary" id="btn-webcam-toggle">
                            <i data-lucide="video"></i> Start Capture Feed
                        </button>
                    </div>
                </div>

                <!-- Analysis Information Right -->
                <div class="glass-card" style="display: flex; flex-direction: column; justify-content: space-between;">
                    <div>
                        <div class="card-header">
                            <h3>Real-time Diagnostics</h3>
                            <span class="card-header-sub">Vision Transformer analysis metrics</span>
                        </div>

                        <div style="margin-top: 20px; display: flex; flex-direction: column; gap: 20px;">
                            <div style="background-color: rgba(255,255,255,0.02); border: 1px solid var(--border-color); padding: 14px; border-radius: var(--radius-md); font-size: 0.85rem; color: var(--text-secondary);">
                                <p style="font-weight: 700; margin-bottom: 6px; color: var(--text-primary);">Webcam Scanning Engine</p>
                                When the camera is active, frames are captured at regular intervals (1.5 seconds) and sent to the DeepGuard classification server. Bounding boxes are rendered around detected face regions.
                            </div>

                            <div id="webcam-diagnostics-out" style="display: none; flex-direction: column; gap: 12px;" class="animate-fade-in">
                                <span class="kpi-label">Active Inference Result</span>
                                <h3 id="webcam-live-label" style="font-weight: 800; font-size: 1.4rem;">SCANNING...</h3>
                                
                                <div class="progress-bar-wrapper" style="height: 10px; margin: 4px 0;">
                                    <div class="progress-bar-fill" id="webcam-live-progress" style="width: 0%;"></div>
                                </div>
                                <span id="webcam-live-fps" style="font-size: 0.8rem; color: var(--text-muted); font-family: var(--font-mono);">
                                    Device: Client Camera
                                </span>
                            </div>
                        </div>
                    </div>

                    <div id="webcam-status-footer">
                        <span style="font-size: 0.85rem; color: var(--text-muted); display: block; text-align: center;">
                            Browser authorization for local camera device is required.
                        </span>
                    </div>
                </div>
            </div>
        </div>
    `;

    const video = document.getElementById("live-webcam-video");
    const canvas = document.getElementById("live-webcam-canvas");
    const ctx = canvas.getContext("2d");
    const toggleBtn = document.getElementById("btn-webcam-toggle");
    const diagnostics = document.getElementById("webcam-diagnostics-out");
    const liveLabel = document.getElementById("webcam-live-label");
    const liveProgress = document.getElementById("webcam-live-progress");

    let stream = null;
    let isActive = false;
    let scanInterval = null;

    // Toggle button handler
    toggleBtn.addEventListener("click", async () => {
        if (!isActive) {
            await startWebcam();
        } else {
            stopWebcam();
        }
    });

    async function startWebcam() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480, facingMode: 'user' }
            });
            video.srcObject = stream;
            video.play();
            
            // Set canvas resolutions to match video rendering size
            video.onloadedmetadata = () => {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                drawOverlay(); // start tracking box animation loop
            };

            isActive = true;
            toggleBtn.innerHTML = `<i data-lucide="video-off"></i> Terminate Feed`;
            toggleBtn.style.background = "linear-gradient(135deg, var(--color-error), #b91c1c)";
            toggleBtn.style.boxShadow = "0 4px 12px rgba(239, 68, 68, 0.25)";
            diagnostics.style.display = "flex";
            
            // Start periodic face check loop
            scanInterval = setInterval(performLiveFrameScan, 1500);

            // Toast Alert
            if (window.showToast) window.showToast("Webcam capture started successfully.", "success");
        } catch (err) {
            console.error("Camera permissions rejected or device missing: ", err);
            if (window.showToast) window.showToast("Webcam initialization failed.", "error");
            const footer = document.getElementById("webcam-status-footer");
            if (footer) {
                footer.innerHTML = `
                    <span style="font-size: 0.85rem; color: var(--color-error); display: block; text-align: center;">
                        Camera access failed. Please allow webcam permission or connect a supported device.
                    </span>
                `;
            }
            return;
        }
        lucide.createIcons();
    }

    function stopWebcam() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            video.srcObject = null;
        }
        if (scanInterval) {
            clearInterval(scanInterval);
        }
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        isActive = false;
        toggleBtn.innerHTML = `<i data-lucide="video"></i> Start Capture Feed`;
        toggleBtn.style.background = "linear-gradient(135deg, var(--color-primary), #0284c7)";
        toggleBtn.style.boxShadow = "0 4px 12px rgba(56, 189, 248, 0.25)";
        diagnostics.style.display = "none";
        
        lucide.createIcons();
    }

    // Draws live bounding box guidance while webcam is active
    let frameId = null;
    function drawOverlay() {
        if (!isActive) {
            cancelAnimationFrame(frameId);
            return;
        }

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Draw a static face box overlay around the screen center
        const boxWidth = 200;
        const boxHeight = 220;
        const x = (canvas.width - boxWidth) / 2;
        const y = (canvas.height - boxHeight) / 2;
        
        // Glow effect
        ctx.shadowBlur = 10;
        ctx.shadowColor = "#38bdf8";

        // Draw bounding box corners
        ctx.strokeStyle = "#38bdf8";
        ctx.lineWidth = 3;
        
        // Top-Left corner
        ctx.beginPath();
        ctx.moveTo(x, y + 24); ctx.lineTo(x, y); ctx.lineTo(x + 24, y); ctx.stroke();
        
        // Top-Right corner
        ctx.beginPath();
        ctx.moveTo(x + boxWidth - 24, y); ctx.lineTo(x + boxWidth, y); ctx.lineTo(x + boxWidth, y + 24); ctx.stroke();
        
        // Bottom-Left corner
        ctx.beginPath();
        ctx.moveTo(x, y + boxHeight - 24); ctx.lineTo(x, y + boxHeight); ctx.lineTo(x + 24, y + boxHeight); ctx.stroke();
        
        // Bottom-Right corner
        ctx.beginPath();
        ctx.moveTo(x + boxWidth - 24, y + boxHeight); ctx.lineTo(x + boxWidth, y + boxHeight); ctx.lineTo(x + boxWidth, y + boxHeight - 24); ctx.stroke();

        // Draw scanning animation line moving vertically
        ctx.shadowBlur = 5;
        const scanY = y + ((Math.sin(Date.now() / 300) + 1) / 2) * boxHeight;
        ctx.strokeStyle = "rgba(0, 242, 254, 0.6)";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(x + 4, scanY);
        ctx.lineTo(x + boxWidth - 4, scanY);
        ctx.stroke();

        ctx.shadowBlur = 0; // reset
        frameId = requestAnimationFrame(drawOverlay);
    }

    async function performLiveFrameScan() {
        if (!isActive) return;

        const offscreen = document.createElement('canvas');
        offscreen.width = video.videoWidth;
        offscreen.height = video.videoHeight;
        const oCtx = offscreen.getContext('2d');
        oCtx.drawImage(video, 0, 0);

        offscreen.toBlob(async (blob) => {
            if (!blob) return;
            const file = new File([blob], 'live_frame.jpg', { type: 'image/jpeg' });

            try {
                const res = await apiClient.detectMedia(file);
                if (!isActive) return;

                const isFake = res.label_name === 'FAKE';
                liveLabel.textContent = `${res.label_name} (${(res.confidence * 100).toFixed(1)}%)`;
                liveLabel.style.color = isFake ? 'var(--color-error)' : 'var(--color-success)';
                liveProgress.style.width = `${res.confidence * 100}%`;
                liveProgress.style.backgroundColor = isFake ? 'var(--color-error)' : 'var(--color-success)';
            } catch (err) {
                console.error('Webcam frame detect call failed: ', err);
                if (window.showToast) window.showToast('Live frame analysis failed. Please try again later.', 'error');
            }
        }, 'image/jpeg');
    }

    // Clean up loop on page swap
    container.addEventListener("DOMNodeRemovedFromDocument", () => {
        stopWebcam();
    });

    lucide.createIcons();
}
