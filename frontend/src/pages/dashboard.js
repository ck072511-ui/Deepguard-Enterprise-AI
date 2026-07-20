/* ============================================================================
   DeepGuard Frontend Page — Dashboard
   ============================================================================ */

import { renderUploadMedia } from './upload_media.js';

export async function renderDashboard(apiClient, container) {
    // 1. Fetch history records, stats and health
    let health = null;
    let history = [];
    let allHistory = [];
    let stats = { total: 0, fake_rate: 0, avg_latency: null };

    try {
        health = await apiClient.checkHealth();
        const [historyRes, allHistoryRes, statsRes] = await Promise.all([
            apiClient.getHistory(5),
            apiClient.getHistory(50),
            apiClient.getStats(),
        ]);
        history = historyRes.items || historyRes || [];
        allHistory = allHistoryRes.items || allHistoryRes || [];
        stats = statsRes || stats;
    } catch (error) {
        console.error('Dashboard API load failed:', error);
    }
    
    // Calculate statistics
    const totalScans = stats.total ?? 0;
    const fakeRate = stats.fake_rate !== null && stats.fake_rate !== undefined ? (stats.fake_rate * 100).toFixed(1) : '0.0';
    const avgLatency = stats.avg_latency !== null && stats.avg_latency !== undefined ? `${stats.avg_latency.toFixed(0)} ms` : 'N/A';
    
    // Render layout HTML
    container.innerHTML = `
        <div class="animate-fade-in">
            <!-- Bento KPI Cards Row -->
            <div class="bento-grid">
                <div class="glass-card kpi-card">
                    <div class="kpi-left">
                        <span class="kpi-label">Total Scans</span>
                        <span class="kpi-value" id="kpi-scans">${totalScans}</span>
                        <span class="kpi-subtext">Active Live Audit log</span>
                    </div>
                    <div class="kpi-icon"><i data-lucide="scan"></i></div>
                </div>

                <div class="glass-card kpi-card">
                    <div class="kpi-left">
                        <span class="kpi-label">Manipulation Rate</span>
                        <span class="kpi-value" id="kpi-ratio">${fakeRate}%</span>
                        <span class="kpi-subtext">System alerts frequency</span>
                    </div>
                    <div class="kpi-icon" style="color: var(--color-error);"><i data-lucide="alert-triangle"></i></div>
                </div>

                <div class="glass-card kpi-card">
                    <div class="kpi-left">
                        <span class="kpi-label">Avg. Latency</span>
                        <span class="kpi-value" id="kpi-latency">${avgLatency}</span>
                        <span class="kpi-subtext">GPU serving speed</span>
                    </div>
                    <div class="kpi-icon" style="color: var(--color-success);"><i data-lucide="cpu"></i></div>
                </div>

                <div class="glass-card kpi-card">
                    <div class="kpi-left">
                        <span class="kpi-label">API Database</span>
                        <span class="kpi-value" style="font-size: 1.4rem; padding: 6px 0;">
                            ${health?.database === 'connected' ? 'Connected' : health ? 'Disconnected' : 'Unavailable'}
                        </span>
                        <span class="kpi-subtext">Active models loaded</span>
                    </div>
                    <div class="kpi-icon" style="color: var(--color-info);"><i data-lucide="database"></i></div>
                </div>
            </div>

            <!-- Upload Media Section -->
            <div class="glass-card upload-media-section">
                <div class="upload-section-header">
                    <i data-lucide="scan-search" style="width:20px;height:20px;color:var(--color-primary);"></i>
                    <h3>Upload Media for Detection</h3>
                    <span class="upload-section-header-sub">Images &amp; Videos · Auto-detects on file select</span>
                </div>
                <div id="dashboard-upload-widget"></div>
            </div>

            <!-- Charts Bento Row -->
            <div class="bento-grid">
                <div class="glass-card col-span-2">
                    <div class="card-header">
                        <h3>Inference Volume Trend</h3>
                        <span class="card-header-sub">Daily scans</span>
                    </div>
                    <div style="height: 240px; position: relative;">
                        <canvas id="trend-chart"></canvas>
                    </div>
                </div>

                <div class="glass-card col-span-2">
                    <div class="card-header">
                        <h3>Media Distribution</h3>
                        <span class="card-header-sub">Ratio of scans</span>
                    </div>
                    <div style="height: 240px; position: relative; display: flex; justify-content: center;">
                        <canvas id="composition-chart" style="max-width: 240px;"></canvas>
                    </div>
                </div>
            </div>

            <!-- Recent Activity Table -->
            <div class="glass-card col-span-4" style="margin-top: 10px;">
                <div class="card-header">
                    <h3>Recent Detections</h3>
                    <span class="card-header-sub">Real-time scan logs</span>
                </div>
                <div class="scans-table-container">
                    <table class="table-scans">
                        <thead>
                            <tr>
                                <th>Filename</th>
                                <th>Media Format</th>
                                <th>Prediction Outcome</th>
                                <th>Confidence</th>
                                <th>Faces</th>
                                <th>Scan Completed</th>
                            </tr>
                        </thead>
                        <tbody id="recent-scans-tbody">
                            <!-- Table rows go here -->
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    `;

    // Render table rows
    const tbody = document.getElementById("recent-scans-tbody");
    if (history.length === 0) {
        tbody.innerHTML = `<tr><td colspan="6" style="text-align: center; color: var(--text-muted);">No scans logged.</td></tr>`;
    } else {
        tbody.innerHTML = history.map(item => {
            const badgeClass = item.label_name === "FAKE" ? "badge-fake" : "badge-real";
            const formattedDate = new Date(item.created_at).toLocaleTimeString() + " - " + new Date(item.created_at).toLocaleDateString();
            return `
                <tr>
                    <td style="font-weight: 600; color: var(--text-primary);">${item.filename}</td>
                    <td style="text-transform: uppercase; font-family: var(--font-mono); font-size: 0.8rem;">
                        ${item.media_type}
                    </td>
                    <td><span class="badge ${badgeClass}">${item.label_name}</span></td>
                    <td style="font-family: var(--font-mono); color: var(--color-primary); font-weight: 700;">
                        ${(item.confidence * 100).toFixed(2)}%
                    </td>
                    <td>${item.faces_count}</td>
                    <td style="color: var(--text-muted);">${formattedDate}</td>
                </tr>
            `;
        }).join('');
    }

    // 2. Mount Upload Media widget with live-refresh callback
    const uploadWidgetEl = document.getElementById('dashboard-upload-widget');
    if (uploadWidgetEl) {
        renderUploadMedia(apiClient, uploadWidgetEl, async (_scanResult) => {
            // Refresh KPI counters and Recent Detections table after each scan
            try {
                const [newStats, newHistory] = await Promise.all([
                    apiClient.getStats(),
                    apiClient.getHistory(5),
                ]);

                const newTotal    = newStats?.total ?? 0;
                const newFakeRate = newStats?.fake_rate != null
                    ? (newStats.fake_rate * 100).toFixed(1)
                    : '0.0';

                // Update KPI values in-place
                const kpiScans = document.getElementById('kpi-scans');
                const kpiRatio = document.getElementById('kpi-ratio');
                if (kpiScans) {
                    kpiScans.textContent = newTotal;
                    kpiScans.classList.add('animate-fade-in');
                    setTimeout(() => kpiScans.classList.remove('animate-fade-in'), 500);
                }
                if (kpiRatio) {
                    kpiRatio.textContent = `${newFakeRate}%`;
                    kpiRatio.classList.add('animate-fade-in');
                    setTimeout(() => kpiRatio.classList.remove('animate-fade-in'), 500);
                }

                // Refresh Recent Detections table
                const items   = newHistory.items || newHistory || [];
                const tbody   = document.getElementById('recent-scans-tbody');
                if (tbody) {
                    if (items.length === 0) {
                        tbody.innerHTML = `<tr><td colspan="6" style="text-align:center;color:var(--text-muted);">No scans logged.</td></tr>`;
                    } else {
                        tbody.innerHTML = items.map(item => {
                            const badgeClass   = item.label_name === 'FAKE' ? 'badge-fake' : 'badge-real';
                            const formattedDate = new Date(item.created_at).toLocaleTimeString() +
                                                  ' - ' + new Date(item.created_at).toLocaleDateString();
                            return `
                                <tr>
                                    <td style="font-weight:600;color:var(--text-primary);">${item.filename}</td>
                                    <td style="text-transform:uppercase;font-family:var(--font-mono);font-size:0.8rem;">${item.media_type}</td>
                                    <td><span class="badge ${badgeClass}">${item.label_name}</span></td>
                                    <td style="font-family:var(--font-mono);color:var(--color-primary);font-weight:700;">${(item.confidence * 100).toFixed(2)}%</td>
                                    <td>${item.faces_count}</td>
                                    <td style="color:var(--text-muted);">${formattedDate}</td>
                                </tr>
                            `;
                        }).join('');
                    }
                }
            } catch (err) {
                console.warn('[Dashboard] Live refresh after scan failed:', err);
            }
        });
    }

    // 3. Load Chart.js instances
    renderCharts(allHistory);
}

function renderCharts(history) {
    const dayLabels = [];
    const countsByDay = {};
    const now = new Date();

    for (let i = 6; i >= 0; i -= 1) {
        const date = new Date(now);
        date.setDate(now.getDate() - i);
        const label = date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
        dayLabels.push(label);
        countsByDay[label] = 0;
    }

    history.forEach((item) => {
        const date = new Date(item.created_at);
        const label = date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
        if (label in countsByDay) {
            countsByDay[label] += 1;
        }
    });

    const counts = dayLabels.map((label) => countsByDay[label] || 0);
    const imagesCount = history.filter(x => x.media_type === 'image').length;
    const videosCount = history.filter(x => x.media_type === 'video').length;

    const ctxTrend = document.getElementById('trend-chart').getContext('2d');
    new Chart(ctxTrend, {
        type: 'line',
        data: {
            labels: dayLabels,
            datasets: [{
                label: 'Scans Volume',
                data: counts,
                borderColor: '#38bdf8',
                backgroundColor: 'rgba(56, 189, 248, 0.12)',
                borderWidth: 2,
                fill: true,
                tension: 0.35
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { grid: { display: false }, ticks: { color: '#94a3b8' } },
                y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8' } }
            }
        }
    });

    const ctxComp = document.getElementById('composition-chart').getContext('2d');
    new Chart(ctxComp, {
        type: 'doughnut',
        data: {
            labels: ['Images', 'Videos'],
            datasets: [{
                data: [imagesCount, videosCount],
                backgroundColor: ['#38bdf8', '#a855f7'],
                borderWidth: 0,
                hoverOffset: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { color: '#94a3b8', font: { family: 'Inter' } }
                }
            },
            cutout: '60%'
        }
    });
}
