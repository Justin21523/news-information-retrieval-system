// CNIRS - Evaluation Page JavaScript (Enhanced)

// DOM Elements
const evalQuery = document.getElementById('eval-query');
const evalTopk = document.getElementById('eval-topk');
const runEvalBtn = document.getElementById('run-eval-btn');
const evalLoading = document.getElementById('eval-loading');
const evalResults = document.getElementById('eval-results');
const metricsCards = document.getElementById('metrics-cards');
const comparisonTable = document.getElementById('comparison-table');

// Chart control checkboxes
const overlayPrecisionCheck = document.getElementById('overlay-precision');
const overlayRecallCheck = document.getElementById('overlay-recall');
const overlayF1Check = document.getElementById('overlay-f1');
const showInterpolatedCheck = document.getElementById('show-interpolated');

// Chart instances
let charts = {
    precision: null,
    recall: null,
    f1: null,
    ndcg: null,
    fbeta: null,
    pAtR: null,
    prCurve: null,
    prInterpolated: null
};

// Current evaluation data
let currentEvalData = null;

// Run evaluation button click
runEvalBtn.addEventListener('click', runEvaluation);

// Enter key to run evaluation
evalQuery.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        runEvaluation();
    }
});

// Chart overlay toggle listeners
if (overlayPrecisionCheck) {
    overlayPrecisionCheck.addEventListener('change', () => {
        if (currentEvalData) displayCharts(currentEvalData.results);
    });
}
if (overlayRecallCheck) {
    overlayRecallCheck.addEventListener('change', () => {
        if (currentEvalData) displayCharts(currentEvalData.results);
    });
}
if (overlayF1Check) {
    overlayF1Check.addEventListener('change', () => {
        if (currentEvalData) displayCharts(currentEvalData.results);
    });
}
if (showInterpolatedCheck) {
    showInterpolatedCheck.addEventListener('change', () => {
        if (currentEvalData) displayCharts(currentEvalData.results);
    });
}

/**
 * Run evaluation for selected models
 */
async function runEvaluation() {
    const query = evalQuery.value.trim();

    if (!query) {
        alert('請輸入測試查詢');
        return;
    }

    // Get selected models
    const selectedModels = Array.from(document.querySelectorAll('.model-checkbox:checked'))
        .map(cb => cb.value);

    if (selectedModels.length === 0) {
        alert('請至少選擇一個模型');
        return;
    }

    const topK = parseInt(evalTopk.value);

    // Show loading
    evalLoading.style.display = 'block';
    evalResults.style.display = 'none';
    runEvalBtn.disabled = true;

    try {
        const response = await fetch('/api/evaluate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query,
                models: selectedModels,
                top_k: topK
            })
        });

        const data = await response.json();

        if (data.success) {
            currentEvalData = data;
            displayEvaluation(data);
        } else {
            alert('評估錯誤: ' + data.error);
        }
    } catch (error) {
        console.error('Evaluation error:', error);
        alert('評估失敗，請稍後再試');
    } finally {
        evalLoading.style.display = 'none';
        runEvalBtn.disabled = false;
    }
}

/**
 * Display evaluation results
 */
function displayEvaluation(data) {
    // Show results container
    evalResults.style.display = 'block';

    // Display metrics summary cards
    displayMetricsCards(data.results);

    // Display charts
    displayCharts(data.results);

    // Display comparison table
    displayComparisonTable(data.results);

    // Scroll to results
    evalResults.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

/**
 * Display metrics summary cards
 */
function displayMetricsCards(results) {
    metricsCards.innerHTML = '';

    Object.entries(results).forEach(([model, metrics]) => {
        if (metrics.error) return;

        const card = document.createElement('div');
        card.className = 'metric-card';

        card.innerHTML = `
            <div class="metric-card-header">${model.toUpperCase()}</div>
            <div class="metric-card-value">${(metrics.map * 100).toFixed(1)}%</div>
            <div class="metric-card-model">MAP</div>
            <div style="font-size: 0.8rem; margin-top: 8px; color: var(--text-secondary);">
                MRR: ${(metrics.mrr * 100).toFixed(1)}% |
                R-Prec: ${(metrics.r_precision * 100).toFixed(1)}%
            </div>
        `;

        metricsCards.appendChild(card);
    });
}

/**
 * Display all charts
 */
function displayCharts(results) {
    // Destroy existing charts
    Object.values(charts).forEach(chart => {
        if (chart) chart.destroy();
    });

    const models = Object.keys(results).filter(m => !results[m].error);
    const colors = getModelColors(models);

    // Check overlay options
    const overlayPrecision = overlayPrecisionCheck?.checked ?? true;
    const overlayRecall = overlayRecallCheck?.checked ?? true;
    const overlayF1 = overlayF1Check?.checked ?? true;
    const showInterpolated = showInterpolatedCheck?.checked ?? true;

    // Precision @ K Chart
    if (overlayPrecision) {
        const precisionData = {
            labels: results[models[0]]?.precision_at_k.map(p => `P@${p.k}`) || [],
            datasets: models.map((model, idx) => ({
                label: model.toUpperCase(),
                data: results[model].precision_at_k.map(p => p.value),
                borderColor: colors[idx],
                backgroundColor: colors[idx] + '33',
                tension: 0.3,
                fill: false,
                borderWidth: 2
            }))
        };
        charts.precision = createLineChart('precision-chart', precisionData);
    }

    // Recall @ K Chart
    if (overlayRecall) {
        const recallData = {
            labels: results[models[0]]?.recall_at_k.map(r => `R@${r.k}`) || [],
            datasets: models.map((model, idx) => ({
                label: model.toUpperCase(),
                data: results[model].recall_at_k.map(r => r.value),
                borderColor: colors[idx],
                backgroundColor: colors[idx] + '33',
                tension: 0.3,
                fill: false,
                borderWidth: 2
            }))
        };
        charts.recall = createLineChart('recall-chart', recallData);
    }

    // F1-Score @ K Chart
    if (overlayF1) {
        const f1Data = {
            labels: results[models[0]]?.f1_at_k.map(f => `F1@${f.k}`) || [],
            datasets: models.map((model, idx) => ({
                label: model.toUpperCase(),
                data: results[model].f1_at_k.map(f => f.value),
                borderColor: colors[idx],
                backgroundColor: colors[idx] + '33',
                tension: 0.3,
                fill: false,
                borderWidth: 2
            }))
        };
        charts.f1 = createLineChart('f1-chart', f1Data);
    }

    // nDCG @ K Chart
    const ndcgData = {
        labels: results[models[0]]?.ndcg_at_k.map(n => `nDCG@${n.k}`) || [],
        datasets: models.map((model, idx) => ({
            label: model.toUpperCase(),
            data: results[model].ndcg_at_k.map(n => n.value),
            borderColor: colors[idx],
            backgroundColor: colors[idx] + '33',
            tension: 0.3,
            fill: false,
            borderWidth: 2
        }))
    };
    charts.ndcg = createLineChart('ndcg-chart', ndcgData);

    // F-beta Scores Chart
    const fbetaData = {
        labels: [],
        datasets: []
    };

    models.forEach((model, idx) => {
        const beta05 = results[model].f_beta_scores.filter(f => f.beta === 0.5);
        const beta20 = results[model].f_beta_scores.filter(f => f.beta === 2.0);

        if (beta05.length > 0) {
            fbetaData.labels = beta05.map(f => `F@${f.k}`);
            fbetaData.datasets.push({
                label: `${model.toUpperCase()} (β=0.5)`,
                data: beta05.map(f => f.value),
                borderColor: colors[idx],
                backgroundColor: colors[idx] + '33',
                tension: 0.3,
                fill: false,
                borderDash: [5, 5],
                borderWidth: 2
            });
            fbetaData.datasets.push({
                label: `${model.toUpperCase()} (β=2.0)`,
                data: beta20.map(f => f.value),
                borderColor: colors[idx],
                backgroundColor: colors[idx] + '55',
                tension: 0.3,
                fill: false,
                borderWidth: 2
            });
        }
    });
    charts.fbeta = createLineChart('fbeta-chart', fbetaData);

    // Precision at Recall Chart
    const pAtRData = {
        labels: results[models[0]]?.precision_at_recall.map(p => `R=${(p.recall*100).toFixed(0)}%`) || [],
        datasets: models.map((model, idx) => ({
            label: model.toUpperCase(),
            data: results[model].precision_at_recall.map(p => p.precision),
            backgroundColor: colors[idx],
            borderColor: colors[idx],
            borderWidth: 2
        }))
    };
    charts.pAtR = createBarChart('p-at-r-chart', pAtRData);

    // Raw PR Curve
    const prData = {
        datasets: models.map((model, idx) => ({
            label: model.toUpperCase(),
            data: results[model].pr_curve.map(p => ({ x: p.recall, y: p.precision })),
            borderColor: colors[idx],
            backgroundColor: colors[idx] + '33',
            tension: 0.3,
            fill: false,
            pointRadius: 3,
            borderWidth: 2
        }))
    };
    charts.prCurve = createScatterChart('pr-curve-chart', prData);

    // Interpolated PR Curve
    if (showInterpolated) {
        const prInterpData = {
            labels: results[models[0]]?.interpolated_precision.map(p => (p.recall*100).toFixed(0) + '%') || [],
            datasets: models.map((model, idx) => ({
                label: model.toUpperCase(),
                data: results[model].interpolated_precision.map(p => p.precision),
                borderColor: colors[idx],
                backgroundColor: colors[idx] + '33',
                tension: 0.0,  // Straight lines for interpolated
                fill: false,
                borderWidth: 2,
                pointRadius: 4
            }))
        };
        charts.prInterpolated = createLineChart('pr-interpolated-chart', prInterpData);
    }
}

/**
 * Create line chart
 */
function createLineChart(canvasId, data) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    return new Chart(ctx, {
        type: 'line',
        data: data,
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: true,
                    position: 'bottom'
                },
                title: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1.0,
                    ticks: {
                        callback: function(value) {
                            return (value * 100).toFixed(0) + '%';
                        }
                    }
                }
            }
        }
    });
}

/**
 * Create bar chart
 */
function createBarChart(canvasId, data) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    return new Chart(ctx, {
        type: 'bar',
        data: data,
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: true,
                    position: 'bottom'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1.0,
                    ticks: {
                        callback: function(value) {
                            return (value * 100).toFixed(0) + '%';
                        }
                    }
                }
            }
        }
    });
}

/**
 * Create scatter chart for PR curve
 */
function createScatterChart(canvasId, data) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    return new Chart(ctx, {
        type: 'line',
        data: data,
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: true,
                    position: 'bottom'
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    min: 0,
                    max: 1,
                    title: {
                        display: true,
                        text: 'Recall'
                    },
                    ticks: {
                        callback: function(value) {
                            return (value * 100).toFixed(0) + '%';
                        }
                    }
                },
                y: {
                    min: 0,
                    max: 1,
                    title: {
                        display: true,
                        text: 'Precision'
                    },
                    ticks: {
                        callback: function(value) {
                            return (value * 100).toFixed(0) + '%';
                        }
                    }
                }
            }
        }
    });
}

/**
 * Display comparison table
 */
function displayComparisonTable(results) {
    const models = Object.keys(results).filter(m => !results[m].error);

    if (models.length === 0) {
        comparisonTable.innerHTML = '<p>沒有可比較的結果</p>';
        return;
    }

    // Find best values for highlighting
    const bestMAP = Math.max(...models.map(m => results[m].map));
    const bestMRR = Math.max(...models.map(m => results[m].mrr));
    const bestRPrec = Math.max(...models.map(m => results[m].r_precision));
    const bestBpref = Math.max(...models.map(m => results[m].bpref));

    let tableHTML = `
        <div class="comparison-table">
            <table>
                <thead>
                    <tr>
                        <th>模型</th>
                        <th>MAP</th>
                        <th>MRR</th>
                        <th>R-Precision</th>
                        <th>Bpref</th>
                        <th>P@5</th>
                        <th>P@10</th>
                        <th>R@10</th>
                        <th>F1@10</th>
                        <th>nDCG@10</th>
                    </tr>
                </thead>
                <tbody>
    `;

    models.forEach(model => {
        const metrics = results[model];
        const p5 = metrics.precision_at_k.find(p => p.k === 5)?.value || 0;
        const p10 = metrics.precision_at_k.find(p => p.k === 10)?.value || 0;
        const r10 = metrics.recall_at_k.find(r => r.k === 10)?.value || 0;
        const f1_10 = metrics.f1_at_k.find(f => f.k === 10)?.value || 0;
        const ndcg10 = metrics.ndcg_at_k.find(n => n.k === 10)?.value || 0;

        tableHTML += `
            <tr>
                <td><strong>${model.toUpperCase()}</strong></td>
                <td class="${metrics.map === bestMAP ? 'best-value' : ''}">${(metrics.map * 100).toFixed(2)}%</td>
                <td class="${metrics.mrr === bestMRR ? 'best-value' : ''}">${(metrics.mrr * 100).toFixed(2)}%</td>
                <td class="${metrics.r_precision === bestRPrec ? 'best-value' : ''}">${(metrics.r_precision * 100).toFixed(2)}%</td>
                <td class="${metrics.bpref === bestBpref ? 'best-value' : ''}">${(metrics.bpref * 100).toFixed(2)}%</td>
                <td>${(p5 * 100).toFixed(2)}%</td>
                <td>${(p10 * 100).toFixed(2)}%</td>
                <td>${(r10 * 100).toFixed(2)}%</td>
                <td>${(f1_10 * 100).toFixed(2)}%</td>
                <td>${(ndcg10 * 100).toFixed(2)}%</td>
            </tr>
        `;
    });

    tableHTML += `
                </tbody>
            </table>
        </div>
    `;

    comparisonTable.innerHTML = tableHTML;
}

/**
 * Get color palette for models
 */
function getModelColors(models) {
    const colorPalette = [
        '#3b82f6', // blue
        '#10b981', // green
        '#f59e0b', // orange
        '#ef4444', // red
        '#8b5cf6', // purple
        '#06b6d4', // cyan
    ];

    return models.map((_, idx) => colorPalette[idx % colorPalette.length]);
}
