/* ============================================================================
   DeepGuard Frontend Page — Platform Analytics
   ============================================================================ */

export async function renderAnalytics(apiClient, container) {
    let history = [];
    try {
        const historyRes = await apiClient.getHistory(100);
        history = historyRes.items || historyRes || [];
    } catch (error) {
        console.error('Analytics API load failed:', error);
    }
    
    container.innerHTML = `
        <div class="animate-fade-in">
            <div class="bento-grid">
                <!-- Card 1: Confidence Distribution -->
                <div class="glass-card col-span-2">
                    <div class="card-header">
                        <h3>Confidence Score Distribution</h3>
                        <span class="card-header-sub">Model predictions frequency</span>
                    </div>
                    <div style="height: 280px; position: relative;">
                        <canvas id="analytics-confidence-chart"></canvas>
                    </div>
                </div>

                <!-- Card 2: Analysis composition monthly timeline -->
                <div class="glass-card col-span-2">
                    <div class="card-header">
                        <h3>Detection Outcome Counts</h3>
                        <span class="card-header-sub">REAL vs FAKE frequency</span>
                    </div>
                    <div style="height: 280px; position: relative; display: flex; justify-content: center;">
                        <canvas id="analytics-outcome-chart" style="max-width: 280px;"></canvas>
                    </div>
                </div>
            </div>

            <div class="glass-card" style="margin-top: 10px;">
                <div class="card-header">
                    <h3>Scanning Activity Overview</h3>
                    <span class="card-header-sub">Monthly platform usage volume</span>
                </div>
                <div style="height: 300px; position: relative;">
                    <canvas id="analytics-activity-chart"></canvas>
                </div>
            </div>
        </div>
    `;

    // Process data for charts
    renderAnalyticsCharts(history);
    lucide.createIcons();
}

function renderAnalyticsCharts(history) {
    const validConfs = history
        .map((x) => Number(x.confidence))
        .filter((value) => Number.isFinite(value));

    const buckets = ['0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0'];
    const bucketCounts = [0, 0, 0, 0, 0];

    validConfs.forEach((confidence) => {
        if (confidence >= 0.5 && confidence < 0.6) bucketCounts[0] += 1;
        else if (confidence >= 0.6 && confidence < 0.7) bucketCounts[1] += 1;
        else if (confidence >= 0.7 && confidence < 0.8) bucketCounts[2] += 1;
        else if (confidence >= 0.8 && confidence < 0.9) bucketCounts[3] += 1;
        else if (confidence >= 0.9 && confidence <= 1.0) bucketCounts[4] += 1;
    });

    const ctxConf = document.getElementById('analytics-confidence-chart').getContext('2d');
    new Chart(ctxConf, {
        type: 'bar',
        data: {
            labels: buckets,
            datasets: [{
                label: 'Inferences',
                data: bucketCounts,
                backgroundColor: 'rgba(56, 189, 248, 0.4)',
                borderColor: '#38bdf8',
                borderWidth: 1,
                borderRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { ticks: { color: '#94a3b8' }, grid: { display: false } },
                y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } }
            }
        }
    });

    const fakeCount = history.filter((x) => x.label_name === 'FAKE').length;
    const realCount = history.filter((x) => x.label_name === 'REAL').length;

    const ctxOut = document.getElementById('analytics-outcome-chart').getContext('2d');
    new Chart(ctxOut, {
        type: 'doughnut',
        data: {
            labels: ['REAL', 'FAKE'],
            datasets: [{
                data: [realCount, fakeCount],
                backgroundColor: ['#10b981', '#ef4444'],
                borderWidth: 0
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

    const monthLabels = [];
    const monthMap = {};
    const now = new Date();

    for (let i = 6; i >= 0; i -= 1) {
        const d = new Date(now);
        d.setMonth(now.getMonth() - i);
        const label = d.toLocaleString(undefined, { month: 'short' });
        const key = `${d.getFullYear()}-${d.getMonth()}`;
        monthLabels.push(label);
        monthMap[key] = { image: 0, video: 0 };
    }

    history.forEach((item) => {
        const d = new Date(item.created_at);
        const key = `${d.getFullYear()}-${d.getMonth()}`;
        if (key in monthMap) {
            if (item.media_type === 'image') monthMap[key].image += 1;
            if (item.media_type === 'video') monthMap[key].video += 1;
        }
    });

    const imageCounts = Object.values(monthMap).map((entry) => entry.image);
    const videoCounts = Object.values(monthMap).map((entry) => entry.video);

    const ctxAct = document.getElementById('analytics-activity-chart').getContext('2d');
    new Chart(ctxAct, {
        type: 'line',
        data: {
            labels: monthLabels,
            datasets: [
                {
                    label: 'Images Scanned',
                    data: imageCounts,
                    borderColor: '#38bdf8',
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    tension: 0.3
                },
                {
                    label: 'Videos Scanned',
                    data: videoCounts,
                    borderColor: '#a855f7',
                    backgroundColor: 'transparent',
                    borderWidth: 2,
                    tension: 0.3
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: { color: '#94a3b8' }
                }
            },
            scales: {
                x: { ticks: { color: '#94a3b8' }, grid: { display: false } },
                y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } }
            }
        }
    });
}
