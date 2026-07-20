/* ============================================================================
   DeepGuard Frontend Page — Image & Video Detection
   ============================================================================ */

export function renderImageDetection(apiClient, container) {
    container.innerHTML = `
        <div class="animate-fade-in">
            <div class="detection-view-grid">
                <!-- Dropzone Left Area -->
                <div class="glass-card flex-col" style="display: flex; gap: 20px;">
                    <div class="card-header">
                        <h3>Target Media Input</h3>
                        <span class="card-header-sub">JPEG, PNG, WebP or BMP</span>
                    </div>

                    <div class="dropzone-container" id="image-dropzone">
                        <input type="file" id="image-file-input" accept="image/*" style="display: none;">
                        <div class="dropzone-icon">
                            <i data-lucide="upload-cloud"></i>
                        </div>
                        <div>
                            <p style="font-weight: 700; margin-bottom: 4px;">Drag and drop portrait file here</p>
                            <p style="color: var(--text-muted); font-size: 0.85rem;">or click to browse local folders (Max 10 MB)</p>
                        </div>
                    </div>

                    <div class="media-preview-container" id="image-preview-box">
                        <p style="color: var(--text-muted);">No image file staged</p>
                    </div>
                </div>

                <!-- Prediction Info Right Area -->
                <div class="glass-card" style="display: flex; flex-direction: column; justify-content: space-between;">
                    <div>
                        <div class="card-header">
                            <h3>Analysis Pipeline</h3>
                            <span class="card-header-sub">Vision Transformer classification state</span>
                        </div>

                        <!-- Stepper indicators -->
                        <div class="pipeline-stepper">
                            <div class="pipeline-step" id="step-upload">
                                <div class="step-indicator">1</div>
                                <div class="step-details">
                                    <h4>Image Registration</h4>
                                    <p>Stage file upload and read bytes structure.</p>
                                </div>
                            </div>
                            <div class="pipeline-step" id="step-faces">
                                <div class="step-indicator">2</div>
                                <div class="step-details">
                                    <h4>MTCNN Face Extraction</h4>
                                    <p>Find region bounding boxes and crop face textures.</p>
                                </div>
                            </div>
                            <div class="pipeline-step" id="step-model">
                                <div class="step-indicator">3</div>
                                <div class="step-details">
                                    <h4>ViT Core Inference</h4>
                                    <p>Analyze patch embeddings and compute anomalies.</p>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div style="margin-top: 32px;" id="image-result-box">
                        <button class="btn-primary" id="btn-run-image" style="width: 100%; justify-content: center;" disabled>
                            <i data-lucide="play"></i> Run Detection Checks
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `;

    const dropzone = document.getElementById("image-dropzone");
    const fileInput = document.getElementById("image-file-input");
    const previewBox = document.getElementById("image-preview-box");
    const runBtn = document.getElementById("btn-run-image");
    let selectedFile = null;

    // Dropzone Events
    dropzone.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", (e) => handleFileSelect(e.target.files[0]));
    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.style.borderColor = "var(--color-primary)";
    });
    dropzone.addEventListener("dragleave", () => {
        dropzone.style.borderColor = "var(--border-color)";
    });
    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.style.borderColor = "var(--border-color)";
        handleFileSelect(e.dataTransfer.files[0]);
    });

    function handleFileSelect(file) {
        if (!file || !file.type.startsWith('image/')) return;
        selectedFile = file;

        // Render preview
        const reader = new FileReader();
        reader.onload = (e) => {
            previewBox.innerHTML = `<img src="${e.target.result}" alt="Staged image">`;
            runBtn.removeAttribute("disabled");
            resetStepper();
        };
        reader.readAsDataURL(file);
    }

    function resetStepper() {
        document.querySelectorAll(".pipeline-step").forEach(step => {
            step.classList.remove("active", "completed");
        });
    }

    runBtn.addEventListener("click", async () => {
        if (!selectedFile) return;
        runBtn.setAttribute("disabled", "true");
        runBtn.innerHTML = `<span class="skeleton" style="width: 100px; height: 16px; border-radius: 4px; display: inline-block;"></span>`;

        const stepUpload = document.getElementById("step-upload");
        const stepFaces = document.getElementById("step-faces");
        const stepModel = document.getElementById("step-model");

        stepUpload.classList.add("active", "completed");
        stepFaces.classList.add("active", "completed");
        stepModel.classList.add("active");
        
        try {
            const res = await apiClient.detectMedia(selectedFile);
            stepModel.classList.add("completed");

            const isFake = res.label_name === "FAKE";
            const outcomeText = isFake ? "🚨 MANIPULATED / FAKE" : "✅ ORIGINAL / REAL";
            const outcomeColor = isFake ? "var(--color-error)" : "var(--color-success)";
            const infoText = isFake ? 
                "Neural artifacts detected. Higher probability of face manipulation, deepfake edit or GAN synthesis." :
                "No generative manipulation signs found. Pixel distribution matches lens capture properties.";

            let explainHtml = "";
            if (res.explainability) {
                const x = res.explainability;
                explainHtml = `
                    <div style="margin-top: 16px; border-top: 1px solid var(--border-color); padding-top: 16px;">
                        <h4 style="font-size: 0.9rem; font-weight: 700; color: var(--text-primary); margin-bottom: 8px;">Explainable AI Diagnostics</h4>
                        
                        <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: var(--text-secondary); margin-bottom: 8px;">
                            <span>Real Probability: <strong>${(x.real_probability * 100).toFixed(1)}%</strong></span>
                            <span>Fake Probability: <strong>${(x.fake_probability * 100).toFixed(1)}%</strong></span>
                        </div>
                        
                        <p style="font-size: 0.85rem; color: var(--text-secondary); line-height: 1.4; background-color: rgba(255,255,255,0.01); border: 1px solid var(--border-color); padding: 10px; border-radius: var(--radius-md); margin-bottom: 16px;">
                            ${x.explanation}
                        </p>

                        <div class="xai-tabs">
                            <div style="display: flex; gap: 8px; border-bottom: 1px solid var(--border-color); padding-bottom: 6px; margin-bottom: 10px;">
                                <button class="theme-btn active" id="tab-heatmap-btn" style="width: auto; height: auto; padding: 4px 10px; font-size: 0.75rem; border-radius: var(--radius-sm);">Overlay Heatmap</button>
                                <button class="theme-btn" id="tab-gradcam-btn" style="width: auto; height: auto; padding: 4px 10px; font-size: 0.75rem; border-radius: var(--radius-sm);">GradCAM</button>
                                <button class="theme-btn" id="tab-attention-btn" style="width: auto; height: auto; padding: 4px 10px; font-size: 0.75rem; border-radius: var(--radius-sm);">Attention Map</button>
                            </div>
                            <div id="xai-tab-content" style="display: flex; justify-content: center; background: #000; border-radius: var(--radius-md); padding: 8px; border: 1px solid var(--border-color);">
                                <img id="xai-display-img" src="${x.heatmap_b64}" style="max-width: 100%; height: auto; max-height: 220px; object-fit: contain; border-radius: var(--radius-sm);">
                            </div>
                        </div>
                    </div>
                `;
            }

            document.getElementById("image-result-box").innerHTML = `
                <div class="animate-fade-in" style="border-top: 1px solid var(--border-color); padding-top: 16px; overflow-y: auto; max-height: 480px; padding-right: 4px;">
                    <span class="kpi-label">Prediction Outcome</span>
                    <h2 style="color: ${outcomeColor}; margin: 6px 0 12px 0; font-size: 1.35rem; font-weight: 800;">
                        ${outcomeText} (${(res.confidence * 100).toFixed(1)}%)
                    </h2>
                    
                    <div class="progress-bar-wrapper" style="margin-bottom: 16px;">
                        <div class="progress-bar-fill" style="width: ${res.confidence * 100}%; background: ${outcomeColor};"></div>
                    </div>
                    
                    ${explainHtml}

                    <div style="margin-top: 20px;">
                        <button class="btn-secondary" style="width: 100%; justify-content: center;" id="btn-reset-image">
                            <i data-lucide="refresh-cw"></i> Analyze Another Image
                        </button>
                    </div>
                </div>
            `;

            // Add XAI tab switch listeners
            if (res.explainability) {
                const x = res.explainability;
                const img = document.getElementById("xai-display-img");
                const hBtn = document.getElementById("tab-heatmap-btn");
                const gBtn = document.getElementById("tab-gradcam-btn");
                const aBtn = document.getElementById("tab-attention-btn");
                
                const activateTab = (activeBtn, inactiveBtns) => {
                    activeBtn.classList.add("active");
                    inactiveBtns.forEach(btn => btn.classList.remove("active"));
                };

                hBtn.addEventListener("click", () => {
                    img.src = x.heatmap_b64;
                    activateTab(hBtn, [gBtn, aBtn]);
                });
                gBtn.addEventListener("click", () => {
                    img.src = x.gradcam_b64;
                    activateTab(gBtn, [hBtn, aBtn]);
                });
                aBtn.addEventListener("click", () => {
                    img.src = x.attention_b64;
                    activateTab(aBtn, [hBtn, gBtn]);
                });
            }

            lucide.createIcons();
            document.getElementById("btn-reset-image").addEventListener("click", () => renderImageDetection(apiClient, container));
        } catch (e) {
            console.error(e);
            stepModel.classList.add("failed");
            runBtn.removeAttribute("disabled");
            runBtn.innerHTML = `<i data-lucide="play"></i> Run Detection Checks`;
            lucide.createIcons();
        }
    });

    lucide.createIcons();
}

export function renderVideoDetection(apiClient, container) {
    container.innerHTML = `
        <div class="animate-fade-in">
            <div class="detection-view-grid">
                <!-- Dropzone Left Area -->
                <div class="glass-card flex-col" style="display: flex; gap: 20px;">
                    <div class="card-header">
                        <h3>Target Video Input</h3>
                        <span class="card-header-sub">MP4, AVI, MOV or MKV</span>
                    </div>

                    <div class="dropzone-container" id="video-dropzone">
                        <input type="file" id="video-file-input" accept="video/*" style="display: none;">
                        <div class="dropzone-icon">
                            <i data-lucide="upload-cloud"></i>
                        </div>
                        <div>
                            <p style="font-weight: 700; margin-bottom: 4px;">Drag and drop video file here</p>
                            <p style="color: var(--text-muted); font-size: 0.85rem;">or click to browse local files (Max 200 MB)</p>
                        </div>
                    </div>

                    <div class="media-preview-container" id="video-preview-box">
                        <p style="color: var(--text-muted);">No staged video file</p>
                    </div>
                </div>

                <!-- Video Processing Status Area -->
                <div class="glass-card" style="display: flex; flex-direction: column; justify-content: space-between;">
                    <div>
                        <div class="card-header">
                            <h3>Sequential Frame Analyzer</h3>
                            <span class="card-header-sub">Video frames processing</span>
                        </div>

                        <div id="video-pipeline-box">
                            <p style="color: var(--text-secondary); font-size: 0.95rem; line-height: 1.5;">
                                The video analyzer reads the stages container, samples frames periodically, performs MTCNN alignments on bounding face coordinates, and computes prediction votes across the temporal dimensions.
                            </p>
                        </div>
                    </div>

                    <div id="video-result-box" style="margin-top: 32px;">
                        <button class="btn-primary" id="btn-run-video" style="width: 100%; justify-content: center;" disabled>
                            <i data-lucide="play"></i> Analyze Frame Sequences
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `;

    const dropzone = document.getElementById("video-dropzone");
    const fileInput = document.getElementById("video-file-input");
    const previewBox = document.getElementById("video-preview-box");
    const runBtn = document.getElementById("btn-run-video");
    let selectedFile = null;

    dropzone.addEventListener("click", () => fileInput.click());
    fileInput.addEventListener("change", (e) => handleFileSelect(e.target.files[0]));
    dropzone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropzone.style.borderColor = "var(--color-primary)";
    });
    dropzone.addEventListener("dragleave", () => {
        dropzone.style.borderColor = "var(--border-color)";
    });
    dropzone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropzone.style.borderColor = "var(--border-color)";
        handleFileSelect(e.dataTransfer.files[0]);
    });

    function handleFileSelect(file) {
        if (!file || !file.type.startsWith('video/')) return;
        selectedFile = file;

        // Render video tag preview
        const fileURL = URL.createObjectURL(file);
        previewBox.innerHTML = `
            <video controls>
                <source src="${fileURL}" type="${file.type}">
                Your browser does not support video play.
            </video>
        `;
        runBtn.removeAttribute("disabled");
    }

    runBtn.addEventListener("click", async () => {
        if (!selectedFile) return;
        runBtn.setAttribute("disabled", "true");

        const pipelineBox = document.getElementById("video-pipeline-box");
        pipelineBox.innerHTML = `
            <div class="animate-fade-in" style="display: flex; flex-direction: column; gap: 12px;">
                <h4 style="font-weight: 700; color: var(--color-primary);">Scanning sequences... <span id="progress-val">0%</span></h4>
                <div class="progress-bar-wrapper">
                    <div class="progress-bar-fill" id="progress-fill" style="width: 0%;"></div>
                </div>
                <p id="pipeline-status-text" style="color: var(--text-muted); font-size: 0.85rem;">Initializing stream reader...</p>
            </div>
        `;

        const progressFill = document.getElementById("progress-fill");
        const progressVal = document.getElementById("progress-val");
        const statusText = document.getElementById("pipeline-status-text");

        const statusUpdates = [
            "Demuxing video container...",
            "Decoding keyframes...",
            "Running MTCNN alignments...",
            "Forwarding patches into Vision Transformer...",
            "Computing majority-vote statistics..."
        ];

        try {
            statusText.textContent = "Uploading media and waiting for backend inference...";
            const res = await apiClient.detectMedia(selectedFile);

            const isFake = res.label_name === "FAKE";
            const outcomeText = isFake ? "🚨 MANIPULATED VIDEO" : "✅ ORIGINAL VIDEO";
            const outcomeColor = isFake ? "var(--color-error)" : "var(--color-success)";

            pipelineBox.innerHTML = `
                <div class="animate-fade-in" style="display: flex; flex-direction: column; gap: 16px;">
                    <span class="kpi-label">Aggregated Score Result</span>
                    <h2 style="color: ${outcomeColor}; font-weight: 800; font-size: 1.5rem; margin-top: 4px;">
                        ${outcomeText} (${(res.confidence * 100).toFixed(1)}%)
                    </h2>
                    
                    <div class="progress-bar-wrapper" style="height: 10px;">
                        <div class="progress-bar-fill" style="width: ${res.confidence * 100}%; background: ${outcomeColor}"></div>
                    </div>
                    
                    <div style="background-color: rgba(255,255,255,0.02); border: 1px solid var(--border-color); padding: 14px; border-radius: var(--radius-md); font-size: 0.85rem; color: var(--text-secondary);">
                        Average faces detected per frame: <strong>${res.faces_count}</strong>. 
                        Prediction is calculated as temporal aggregate predictions.
                    </div>
                </div>
            `;

            document.getElementById("video-result-box").innerHTML = `
                <button class="btn-secondary" style="width: 100%; justify-content: center;" id="btn-reset-video">
                    <i data-lucide="refresh-cw"></i> Scan New Video
                </button>
            `;
            lucide.createIcons();
            document.getElementById("btn-reset-video").addEventListener("click", () => renderVideoDetection(apiClient, container));
        } catch (e) {
            console.error(e);
            pipelineBox.innerHTML = `<p style="color: var(--color-error);">Error executing video checks: ${e.message}</p>`;
            runBtn.removeAttribute("disabled");
        }
    });

    lucide.createIcons();
}
