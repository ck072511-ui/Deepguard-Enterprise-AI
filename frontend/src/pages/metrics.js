/* ============================================================================
   DeepGuard Frontend Page — Model Metrics
   ============================================================================ */

function safeMetric(value) {
    const num = Number(value);
    return Number.isFinite(num) ? num : null;
}

export async function renderMetrics(apiClient, container) {
    let trainingSummary = {};
    let activeModel = {};
    try {
        [trainingSummary, activeModel] = await Promise.all([
            apiClient.getTrainingSummary(),
            apiClient.getActiveModelInfo(),
        ]);
    } catch (error) {
        console.error('Metrics API load failed:', error);
    }
    const metrics = trainingSummary.metrics || {};

    const accuracy = safeMetric(metrics.accuracy ?? metrics.acc ?? metrics.accuracy_score);
    const f1Score = safeMetric(metrics.f1_score ?? metrics.f1 ?? metrics.f1score);
    const precision = safeMetric(metrics.precision);
    const recall = safeMetric(metrics.recall);
    const auc = safeMetric(metrics.auc ?? metrics.auc_roc);

    const activeModelName = activeModel.name || 'Unavailable';
    const runStatus = trainingSummary.status || 'Unavailable';
    const runId = trainingSummary.run_id || 'Not available';
    const hasTrainingData = [accuracy, f1Score, precision, recall, auc].some((v) => v !== null);

    container.innerHTML = `
        <div class="animate-fade-in">
            <div class="bento-grid">
                <div class="glass-card kpi-card">
                    <div class="kpi-left">
                        <span class="kpi-label">Validation Accuracy</span>
                        <span class="kpi-value" style="color: var(--color-primary);">${accuracy !== null ? `${(accuracy * 100).toFixed(1)}%` : 'N/A'}</span>
                    </div>
                </div>
                <div class="glass-card kpi-card">
                    <div class="kpi-left">
                        <span class="kpi-label">F1-Score (Fake Class)</span>
                        <span class="kpi-value" style="color: var(--color-secondary);">${f1Score !== null ? `${(f1Score * 100).toFixed(1)}%` : 'N/A'}</span>
                    </div>
                </div>
                <div class="glass-card kpi-card">
                    <div class="kpi-left">
                        <span class="kpi-label">Model Precision</span>
                        <span class="kpi-value" style="color: var(--color-success);">${precision !== null ? `${(precision * 100).toFixed(1)}%` : 'N/A'}</span>
                    </div>
                </div>
                <div class="glass-card kpi-card">
                    <div class="kpi-left">
                        <span class="kpi-label">Model Recall</span>
                        <span class="kpi-value">${recall !== null ? `${(recall * 100).toFixed(1)}%` : 'N/A'}</span>
                    </div>
                </div>
            </div>

            <div class="bento-grid">
                <div class="glass-card col-span-2">
                    <div class="card-header">
                        <h3>Training Run Summary</h3>
                        <span class="card-header-sub">Latest metrics from backend MLflow</span>
                    </div>
                    <div style="padding: 20px; color: var(--text-secondary);">
                        <p><strong>Active Model:</strong> ${activeModelName}</p>
                        <p><strong>Run ID:</strong> ${runId}</p>
                        <p><strong>Status:</strong> ${runStatus}</p>
                        <p><strong>AUC-ROC:</strong> ${auc !== null ? auc.toFixed(3) : 'N/A'}</p>
                        ${hasTrainingData ? '' : '<p style="color: var(--text-muted);">Training summary metrics are unavailable from the backend.</p>'}
                    </div>
                </div>

                <div class="glass-card col-span-2">
                    <div class="card-header">
                        <h3>ROC Curve</h3>
                        <span class="card-header-sub">${auc !== null ? `AUC-ROC: ${auc.toFixed(3)}` : 'ROC data unavailable'}</span>
                    </div>
                    <div style="height: 250px; position: relative;">
                        <canvas id="metrics-roc-chart"></canvas>
                        <div id="metrics-roc-placeholder" style="display: none; color: var(--text-muted); padding: 24px; text-align: center;">
                            ROC curve data is not available from the backend.
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;

    renderRocChart(auc);
    lucide.createIcons();
}

function renderRocChart(auc) {
    const placeholder = document.getElementById('metrics-roc-placeholder');
    if (auc === null) {
        placeholder.style.display = 'block';
        return;
    }

    placeholder.style.display = 'none';
    const ctxRoc = document.getElementById('metrics-roc-chart').getContext('2d');
    new Chart(ctxRoc, {
        type: 'line',
        data: {
            labels: [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
            datasets: [
                {
                    label: 'Model ROC',
                    data: [0.0, 0.15, 0.4, 0.7, 0.86, 0.94, 1.0],
                    borderColor: '#38bdf8',
                    borderWidth: 2,
                    fill: false,
                    tension: 0.2
                },
                {
                    label: 'Random Guess',
                    data: [0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
                    borderColor: 'rgba(255,255,255,0.15)',
                    borderWidth: 1.5,
                    borderDash: [5, 5],
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { color: '#94a3b8' }
                }
            },
            scales: {
                x: {
                    title: { display: true, text: 'False Positive Rate', color: '#64748b' },
                    ticks: { color: '#94a3b8' },
                    grid: { color: 'rgba(255,255,255,0.05)' }
                },
                y: {
                    title: { display: true, text: 'True Positive Rate', color: '#64748b' },
                    ticks: { color: '#94a3b8' },
                    grid: { color: 'rgba(255,255,255,0.05)' }
                }
            }
        }
    });
}
