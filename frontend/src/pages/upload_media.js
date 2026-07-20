/* ============================================================================
   DeepGuard — Upload Media Widget (Dashboard Embedded Section)
   Reuses apiClient.detectMedia() — no backend code duplicated.
   ============================================================================ */

const IMAGE_ACCEPT = '.jpg,.jpeg,.png,.webp';
const VIDEO_ACCEPT = '.mp4,.avi,.mov,.mkv';
const ALLOWED_IMAGE = ['image/jpeg', 'image/png', 'image/webp'];
const ALLOWED_VIDEO = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska', 'video/x-msvideo'];

/**
 * Render the Upload Media section into a container element.
 * @param {DeepGuardAPIClient} apiClient - Existing API client instance
 * @param {HTMLElement}        container - DOM element to render into
 * @param {Function}           onScanComplete - Callback fired after each successful scan
 */
export function renderUploadMedia(apiClient, container, onScanComplete) {
    container.innerHTML = buildDropzoneHTML();
    lucide.createIcons();
    wireEvents(apiClient, container, onScanComplete);
}

/* ── HTML Templates ──────────────────────────────────────────────────────── */

function buildDropzoneHTML() {
    return `
        <div class="upload-dropzone-wrapper" id="um-dropzone">
            <input type="file" id="um-file-input" accept="${IMAGE_ACCEPT},${VIDEO_ACCEPT}" style="display:none;">
            <div class="upload-dropzone-icon">
                <i data-lucide="upload-cloud"></i>
            </div>
            <div class="upload-dropzone-text">
                <p class="upload-dropzone-title">Drag &amp; drop media here</p>
                <p class="upload-dropzone-sub">Images: JPG, JPEG, PNG, WEBP &nbsp;|&nbsp; Videos: MP4, AVI, MOV, MKV</p>
            </div>
            <button class="btn-secondary upload-browse-btn" id="um-browse-btn">
                <i data-lucide="folder-open"></i> Browse Files
            </button>
        </div>
        <div id="um-preview-area"></div>
        <div id="um-result-area"></div>
    `;
}

function buildPreviewHTML(file, objectUrl, isImage) {
    const sizeLabel = file.size < 1_048_576
        ? `${(file.size / 1024).toFixed(1)} KB`
        : `${(file.size / 1_048_576).toFixed(2)} MB`;

    const mediaEl = isImage
        ? `<img src="${objectUrl}" alt="Preview" class="upload-media-preview-img">`
        : `<video src="${objectUrl}" controls class="upload-media-preview-video"></video>`;

    return `
        <div class="upload-media-preview animate-fade-in">
            <div class="upload-preview-media">${mediaEl}</div>
            <div class="upload-preview-info">
                <span class="upload-file-name">
                    <i data-lucide="${isImage ? 'image' : 'film'}"></i>
                    ${escapeHtml(file.name)}
                </span>
                <span class="upload-file-meta">${isImage ? 'Image' : 'Video'} &nbsp;·&nbsp; ${sizeLabel}</span>
            </div>
        </div>
    `;
}

function buildLoadingHTML() {
    return `
        <div class="upload-loading animate-fade-in">
            <div class="upload-loading-spinner"></div>
            <div class="upload-loading-text">
                <p class="upload-loading-title">Analyzing with Vision Transformer…</p>
                <p class="upload-loading-sub">Running MTCNN face extraction → ViT inference → XAI engine</p>
            </div>
        </div>
    `;
}

function buildResultHTML(res, fileName) {
    const isFake = res.label_name === 'FAKE';
    const cardClass = isFake ? 'fake' : 'real';
    const labelIcon = isFake ? '🚨' : '✅';
    const labelText = isFake ? 'AI Generated (Fake)' : 'Real';
    const confPct   = (res.confidence * 100).toFixed(1);
    const latency   = res.inference_time_ms != null
        ? `${res.inference_time_ms.toFixed(0)} ms`
        : 'N/A';
    const modelName  = res.model_name || res.model_version || 'ViT Model';
    const facesCount = res.faces_count ?? '—';

    const xaiHTML = res.explainability?.heatmap_b64
        ? `<div class="upload-result-xai">
               <img src="${res.explainability.heatmap_b64}" alt="XAI Heatmap" class="upload-xai-thumb">
               <span class="upload-xai-label">XAI Heatmap</span>
           </div>`
        : '';

    return `
        <div class="upload-result-card ${cardClass} animate-fade-in">
            <div class="upload-result-header">
                <div class="upload-result-verdict">
                    <span class="upload-result-icon">${labelIcon}</span>
                    <span class="upload-result-label">${labelText}</span>
                </div>
                <span class="upload-result-conf">${confPct}% confidence</span>
            </div>

            <div class="progress-bar-wrapper upload-result-bar">
                <div class="progress-bar-fill" style="width:${confPct}%; background: var(--upload-result-color);"></div>
            </div>

            <div class="upload-result-meta">
                <div class="upload-meta-item">
                    <i data-lucide="cpu"></i>
                    <span>${escapeHtml(modelName)}</span>
                </div>
                <div class="upload-meta-item">
                    <i data-lucide="timer"></i>
                    <span>${latency}</span>
                </div>
                <div class="upload-meta-item">
                    <i data-lucide="scan-face"></i>
                    <span>${facesCount} face${facesCount !== 1 ? 's' : ''}</span>
                </div>
                <div class="upload-meta-item">
                    <i data-lucide="file"></i>
                    <span>${escapeHtml(fileName)}</span>
                </div>
            </div>

            ${xaiHTML}

            <button class="btn-secondary upload-reset-btn" id="um-reset-btn">
                <i data-lucide="refresh-cw"></i> Scan Another File
            </button>
        </div>
    `;
}

function buildErrorHTML(message) {
    return `
        <div class="upload-error-card animate-fade-in">
            <i data-lucide="alert-octagon"></i>
            <div>
                <p class="upload-error-title">Detection Failed</p>
                <p class="upload-error-msg">${escapeHtml(message)}</p>
            </div>
            <button class="btn-secondary upload-reset-btn" id="um-reset-btn">
                <i data-lucide="refresh-cw"></i> Try Again
            </button>
        </div>
    `;
}

/* ── Event Wiring ─────────────────────────────────────────────────────────── */

function wireEvents(apiClient, container, onScanComplete) {
    const dropzone  = container.querySelector('#um-dropzone');
    const fileInput = container.querySelector('#um-file-input');
    const browseBtn = container.querySelector('#um-browse-btn');

    // Browse button
    browseBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        fileInput.click();
    });

    // Click on dropzone
    dropzone.addEventListener('click', () => fileInput.click());

    // File input change
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) handleFile(file, apiClient, container, onScanComplete);
    });

    // Drag-over
    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });

    // Drag-leave
    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
    });

    // Drop
    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        const file = e.dataTransfer.files[0];
        if (file) handleFile(file, apiClient, container, onScanComplete);
    });
}

/* ── Core File Handler ────────────────────────────────────────────────────── */

async function handleFile(file, apiClient, container, onScanComplete) {
    const isImage = ALLOWED_IMAGE.includes(file.type);
    const isVideo = ALLOWED_VIDEO.includes(file.type) ||
                    /\.(mp4|avi|mov|mkv)$/i.test(file.name);

    if (!isImage && !isVideo) {
        window.showToast('Unsupported file type. Please upload an image (JPG/PNG/WEBP) or video (MP4/AVI/MOV/MKV).', 'error');
        return;
    }

    // 1. Show preview
    const objectUrl = URL.createObjectURL(file);
    const previewArea = container.querySelector('#um-preview-area');
    const resultArea  = container.querySelector('#um-result-area');

    previewArea.innerHTML = buildPreviewHTML(file, objectUrl, isImage);
    resultArea.innerHTML  = buildLoadingHTML();
    lucide.createIcons();

    // 2. Run detection
    try {
        const res = await apiClient.detectMedia(file);

        // 3. Show result card
        resultArea.innerHTML = buildResultHTML(res, file.name);
        lucide.createIcons();

        // 4. Show toast
        const isFake = res.label_name === 'FAKE';
        window.showToast(
            isFake
                ? `🚨 Deepfake detected in "${truncate(file.name, 30)}" (${(res.confidence * 100).toFixed(1)}% confidence)`
                : `✅ Media appears real: "${truncate(file.name, 30)}" (${(res.confidence * 100).toFixed(1)}% confidence)`,
            isFake ? 'error' : 'success'
        );

        // 5. Wire reset button
        const resetBtn = container.querySelector('#um-reset-btn');
        if (resetBtn) {
            resetBtn.addEventListener('click', () => {
                URL.revokeObjectURL(objectUrl);
                previewArea.innerHTML = '';
                resultArea.innerHTML  = '';
                renderUploadMedia(apiClient, container, onScanComplete);
            });
        }

        // 6. Notify dashboard to refresh KPIs & table
        if (typeof onScanComplete === 'function') {
            onScanComplete(res);
        }

    } catch (err) {
        console.error('[UploadMedia] Detection error:', err);
        resultArea.innerHTML = buildErrorHTML(err.message || 'Unknown error occurred.');
        lucide.createIcons();

        const resetBtn = container.querySelector('#um-reset-btn');
        if (resetBtn) {
            resetBtn.addEventListener('click', () => {
                URL.revokeObjectURL(objectUrl);
                previewArea.innerHTML = '';
                resultArea.innerHTML  = '';
                renderUploadMedia(apiClient, container, onScanComplete);
            });
        }
    }
}

/* ── Utilities ────────────────────────────────────────────────────────────── */

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

function truncate(str, max) {
    return str.length > max ? str.slice(0, max - 1) + '…' : str;
}
