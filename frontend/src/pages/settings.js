/* ============================================================================
   DeepGuard Frontend Page — Platform Settings
   ============================================================================ */

export async function renderSettings(apiClient, container) {
    const models = await apiClient.listModels();
    
    container.innerHTML = `
        <div class="animate-fade-in" style="display: flex; flex-direction: column; gap: 24px;">
            <!-- Model Version Manager -->
            <div class="glass-card">
                <div class="card-header" style="border-bottom: 1px solid var(--border-color); padding-bottom: 16px; margin-bottom: 20px;">
                    <h3>🧠 Model Version Manager</h3>
                    <span class="card-header-sub">Manage and switch between active neural network weights</span>
                </div>

                <div class="scans-table-container">
                    <table class="table-scans">
                        <thead>
                            <tr>
                                <th>Model Name</th>
                                <th>Version</th>
                                <th>File Weights Location</th>
                                <th>Registry Date</th>
                                <th>Status</th>
                            </tr>
                        </thead>
                        <tbody id="models-list-tbody">
                            <!-- Models inject -->
                        </tbody>
                    </table>
                </div>

                <!-- Model Activation Form -->
                <div style="margin-top: 24px; display: flex; gap: 20px; align-items: flex-end;" id="activation-actions-container">
                    <!-- Dynamic activation actions -->
                </div>
            </div>

            <!-- Model Register Form -->
            <div class="glass-card">
                <div class="card-header" style="border-bottom: 1px solid var(--border-color); padding-bottom: 16px; margin-bottom: 20px;">
                    <h3>📋 Register New Model Version</h3>
                    <span class="card-header-sub">Add a new Vision Transformer weights file to database registry</span>
                </div>

                <form id="form-register-model" class="form-group" style="margin-bottom: 0;">
                    <div class="form-row">
                        <div class="form-group">
                            <label for="reg-model-name">Model Architecture Name</label>
                            <input type="text" id="reg-model-name" class="form-control" placeholder="e.g. vit_tiny_patch16_224" required>
                        </div>
                        <div class="form-group">
                            <label for="reg-model-version">Version String</label>
                            <input type="text" id="reg-model-version" class="form-control" placeholder="e.g. 1.2.0" required>
                        </div>
                    </div>
                    <div class="form-group">
                        <label for="reg-model-path">Weights Path (Local Disk)</label>
                        <input type="text" id="reg-model-path" class="form-control" placeholder="e.g. ./weights/best_vit_model.pt" required>
                    </div>
                    <div style="display: flex; justify-content: flex-end; margin-top: 10px;">
                        <button type="submit" class="btn-primary">
                            <i data-lucide="plus"></i> Register Weights Version
                        </button>
                    </div>
                </form>
            </div>

            <!-- Configuration Parameters -->
            <div class="glass-card">
                <div class="card-header" style="border-bottom: 1px solid var(--border-color); padding-bottom: 16px; margin-bottom: 20px;">
                    <h3>⚙️ Platform Classification Parameters</h3>
                    <span class="card-header-sub">Configure runtime and boundary thresholds</span>
                </div>

                <div style="display: flex; flex-direction: column; gap: 24px;">
                    <div class="form-group">
                        <label>Decision Boundary Threshold</label>
                        <div class="slider-control">
                            <input type="range" min="0.1" max="0.9" step="0.05" value="0.50" class="slider-input" id="boundary-threshold-slider">
                            <span class="slider-val" id="boundary-threshold-val">0.50</span>
                        </div>
                        <span style="font-size: 0.8rem; color: var(--text-muted); display: block; margin-top: 4px;">
                            Scores above this boundary are classified as FAKE. Default is 0.50.
                        </span>
                    </div>

                    <div style="display: flex; justify-content: space-between; align-items: center; border-top: 1px solid var(--border-color); padding-top: 20px;">
                        <div>
                            <h4 style="font-size: 0.95rem; font-weight: 600;">ONNX Runtime Inference Engine</h4>
                            <p style="font-size: 0.8rem; color: var(--text-muted); margin-top: 2px;">Use ONNX execution providers to accelerate feed-forward predictions.</p>
                        </div>
                        <label class="toggle-switch">
                            <input type="checkbox" id="onnx-engine-toggle">
                            <span class="toggle-slider"></span>
                        </label>
                    </div>

                    <div style="display: flex; justify-content: space-between; align-items: center; border-top: 1px solid var(--border-color); padding-top: 20px;">
                        <div>
                            <h4 style="font-size: 0.95rem; font-weight: 600;">Enable Face Cropping Cache</h4>
                            <p style="font-size: 0.8rem; color: var(--text-muted); margin-top: 2px;">Skip alignment checks for duplicate frame files on disk.</p>
                        </div>
                        <label class="toggle-switch">
                            <input type="checkbox" id="face-cache-toggle" checked>
                            <span class="toggle-slider"></span>
                        </label>
                    </div>
                </div>
            </div>
        </div>
    `;

    const tbody = document.getElementById("models-list-tbody");
    const activationContainer = document.getElementById("activation-actions-container");
    const registerForm = document.getElementById("form-register-model");
    
    const slider = document.getElementById("boundary-threshold-slider");
    const sliderVal = document.getElementById("boundary-threshold-val");

    // Load models lists
    async function loadModelsUI() {
        const latestModels = await apiClient.listModels();
        
        if (latestModels.length === 0) {
            tbody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--text-muted); padding: 20px 0;">No registered model models.</td></tr>`;
            activationContainer.innerHTML = '';
        } else {
            tbody.innerHTML = latestModels.map(m => {
                const statusBadge = m.active ? 
                    `<span class="badge badge-real">Active</span>` : 
                    `<span class="badge badge-fake">Inactive</span>`;
                const regDate = new Date(m.created_at).toLocaleDateString();
                return `
                    <tr>
                        <td style="font-weight: 600; color: var(--text-primary);">${m.name}</td>
                        <td style="font-family: var(--font-mono); font-size: 0.85rem;">v${m.version}</td>
                        <td style="font-family: var(--font-mono); font-size: 0.85rem; color: var(--text-muted);">${m.registry_path}</td>
                        <td style="color: var(--text-muted); font-size: 0.85rem;">${regDate}</td>
                        <td>${statusBadge}</td>
                    </tr>
                `;
            }).join('');

            // Render activation select options
            const inactive = latestModels.filter(m => !m.active);
            if (inactive.length > 0) {
                activationContainer.innerHTML = `
                    <div class="form-group" style="margin-bottom: 0; flex-grow: 1;">
                        <label for="activate-model-select">Choose model to activate</label>
                        <select id="activate-model-select" class="form-control">
                            ${inactive.map(m => `<option value="${m.id}">${m.name} (v${m.version})</option>`).join('')}
                        </select>
                    </div>
                    <button class="btn-primary" id="btn-activate-selected">
                        <i data-lucide="check"></i> Activate Model
                    </button>
                `;

                document.getElementById("btn-activate-selected").addEventListener("click", async () => {
                    const mid = document.getElementById("activate-model-select").value;
                    const res = await apiClient.activateModel(mid);
                    if (res) {
                        if (window.showToast) window.showToast("Model successfully activated.", "success");
                        
                        // Update active model name in topbar
                        const topbarModel = document.getElementById("topbar-model-name");
                        if (topbarModel) topbarModel.textContent = res.name;

                        loadModelsUI();
                    }
                });
            } else {
                activationContainer.innerHTML = `<span style="font-size: 0.9rem; color: var(--text-muted); font-style: italic;">All registered model versions are active.</span>`;
            }
        }
        lucide.createIcons();
    }

    // Submit model handler
    registerForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const name = document.getElementById("reg-model-name").value;
        const version = document.getElementById("reg-model-version").value;
        const path = document.getElementById("reg-model-path").value;

        const res = await apiClient.registerModel(name, version, path);
        if (res) {
            if (window.showToast) window.showToast(`Registered '${name}' v${version} successfully.`, "success");
            registerForm.reset();
            loadModelsUI();
        } else {
            if (window.showToast) window.showToast("Failed to register model configuration.", "error");
        }
    });

    // Slider listener
    slider.addEventListener("input", (e) => {
        sliderVal.textContent = parseFloat(e.target.value).toFixed(2);
    });

    await loadModelsUI();
    lucide.createIcons();
}
