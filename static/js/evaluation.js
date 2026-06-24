// CNIRS evaluation dashboard.

const evalQuery = document.getElementById('eval-query');
const evalQuerySet = document.getElementById('eval-query-set');
const evalTopk = document.getElementById('eval-topk');
const evalKValues = document.getElementById('eval-k-values');
const runEvalBtn = document.getElementById('run-eval-btn');
const evalLoading = document.getElementById('eval-loading');
const evalResults = document.getElementById('eval-results');
const metricsCards = document.getElementById('metrics-cards');
const comparisonTable = document.getElementById('comparison-table');
const evaluationMeta = document.getElementById('evaluation-meta');
const perQueryBreakdown = document.getElementById('per-query-breakdown');

const overlayPrecisionCheck = document.getElementById('overlay-precision');
const overlayRecallCheck = document.getElementById('overlay-recall');
const overlayF1Check = document.getElementById('overlay-f1');
const showInterpolatedCheck = document.getElementById('show-interpolated');

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

let currentEvalData = null;

document.addEventListener('DOMContentLoaded', () => {
    loadQuerySets().then(initializeEvaluationFromUrl);
});

if (runEvalBtn) {
    runEvalBtn.addEventListener('click', runEvaluation);
}

if (evalQuery) {
    evalQuery.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            runEvaluation();
        }
    });
}

[overlayPrecisionCheck, overlayRecallCheck, overlayF1Check, showInterpolatedCheck]
    .filter(Boolean)
    .forEach((control) => {
        control.addEventListener('change', () => {
            if (currentEvalData) {
                displayCharts(currentEvalData.results || {});
            }
        });
    });

async function loadQuerySets() {
    if (!evalQuerySet) {
        return;
    }
    try {
        const response = await fetch('api/evaluation/query_sets');
        const payload = await response.json();
        const data = normalizeApiPayload(payload);
        const querySets = data.query_sets || [];
        evalQuerySet.innerHTML = querySets.map((querySet) => `
            <option value="${escapeAttr(querySet.id)}" ${querySet.default ? 'selected' : ''}>
                ${escapeHtml(querySet.name)} (${querySet.query_count} queries)
            </option>
        `).join('');
    } catch (error) {
        console.warn('Unable to load evaluation query sets', error);
    }
}

function initializeEvaluationFromUrl() {
    const params = new URLSearchParams(window.location.search);
    const query = params.get('q') || params.get('query');
    const querySet = params.get('query_set') || params.get('set');
    const models = params.get('models');
    const topK = params.get('top_k') || params.get('topk') || params.get('limit');
    const kValues = params.get('k_values') || params.get('k');
    const shouldRun = params.get('run') === '1' || params.get('run') === 'true';

    if (query && evalQuery) {
        evalQuery.value = query;
    }
    if (querySet && evalQuerySet) {
        const option = evalQuerySet.querySelector(`option[value="${querySet}"]`);
        if (option) {
            evalQuerySet.value = querySet;
        }
    }
    if (topK && evalTopk) {
        evalTopk.value = topK;
    }
    if (kValues && evalKValues) {
        evalKValues.value = kValues;
    }
    if (models) {
        const selected = new Set(models.split(',').map((model) => model.trim()).filter(Boolean));
        document.querySelectorAll('.model-checkbox').forEach((checkbox) => {
            checkbox.checked = selected.has(checkbox.value);
        });
    }
    if (shouldRun) {
        window.setTimeout(runEvaluation, 300);
    }
}

async function runEvaluation() {
    const selectedModels = Array.from(document.querySelectorAll('.model-checkbox:checked'))
        .map((checkbox) => checkbox.value);

    if (selectedModels.length === 0) {
        alert('請至少選擇一個模型');
        return;
    }

    const query = evalQuery?.value.trim() || '';
    const querySet = evalQuerySet?.value || undefined;
    const topK = parseInt(evalTopk?.value || '20', 10);
    const kValues = parseKValues(evalKValues?.value, topK);
    const body = {
        query_set: querySet,
        models: selectedModels,
        top_k: topK,
        k_values: kValues
    };
    if (query) {
        body.queries = [{ id: 'custom_1', query }];
    }

    evalLoading.style.display = 'block';
    evalResults.style.display = 'none';
    runEvalBtn.disabled = true;

    try {
        const response = await fetch('api/evaluate/jobs', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-IR-Session': getIrSessionId()
            },
            body: JSON.stringify(body)
        });
        const payload = await response.json();
        if (!payload.ok && !payload.success) {
            const message = payload.error?.message || payload.message || 'Evaluation failed';
            alert(`評估錯誤: ${message}`);
            return;
        }
        const job = normalizeApiPayload(payload);
        const data = job.result || await pollEvaluationJob(job.job_id);
        currentEvalData = data;
        displayEvaluation(data);
    } catch (error) {
        console.error('Evaluation error:', error);
        alert('評估失敗，請稍後再試');
    } finally {
        evalLoading.style.display = 'none';
        runEvalBtn.disabled = false;
    }
}

async function pollEvaluationJob(jobId) {
    if (!jobId) {
        throw new Error('Evaluation job id is missing');
    }
    for (let attempt = 0; attempt < 180; attempt += 1) {
        await sleep(1000);
        const response = await fetch(`api/evaluate/jobs/${encodeURIComponent(jobId)}`, {
            headers: { 'X-IR-Session': getIrSessionId() }
        });
        const payload = await response.json();
        if (!payload.ok && !payload.success) {
            throw new Error(payload.error?.message || 'Evaluation job failed');
        }
        const job = normalizeApiPayload(payload);
        if (job.status === 'completed') {
            return job.result;
        }
        if (job.status === 'failed') {
            throw new Error(job.error?.message || 'Evaluation job failed');
        }
        const statusText = evalLoading.querySelector('p');
        if (statusText) {
            statusText.textContent = `評估計算中... (${job.status})`;
        }
    }
    throw new Error('Evaluation job timed out');
}

function displayEvaluation(data) {
    evalResults.style.display = 'block';
    displayEvaluationMeta(data);
    displayMetricsCards(data.results || {});
    displayCharts(data.results || {});
    displayComparisonTable(data.results || {});
    displayPerQueryBreakdown(data.per_query || {});
    evalResults.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function displayEvaluationMeta(data) {
    if (!evaluationMeta) {
        return;
    }
    const coverage = data.qrels_coverage || {};
    const dataset = data.dataset || {};
    evaluationMeta.innerHTML = `
        <div class="evaluation-notice">
            <strong>Demo evaluation, not full benchmark.</strong>
            <span>${escapeHtml(data.disclaimer || '')}</span>
        </div>
        <div class="evaluation-meta-grid">
            <div><span>Query Set</span><strong>${escapeHtml(data.query_set_info?.name || data.query_set || '-')}</strong></div>
            <div><span>Corpus</span><strong>${formatNumber(dataset.total_documents || 0)} docs</strong></div>
            <div><span>Qrels Coverage</span><strong>${formatPercent(coverage.coverage || 0)}</strong></div>
            <div><span>Resolved Judgments</span><strong>${coverage.resolved || 0}/${coverage.judgments || 0}</strong></div>
            <div><span>Cache</span><strong>${data.cached ? 'hit' : 'miss'}</strong></div>
        </div>
    `;
}

function displayMetricsCards(results) {
    metricsCards.innerHTML = '';
    Object.entries(results).forEach(([model, metrics]) => {
        if (metrics.error || metrics.available === false) {
            return;
        }
        const card = document.createElement('div');
        card.className = 'metric-card';
        card.innerHTML = `
            <div class="metric-card-header">${escapeHtml(model.toUpperCase())}</div>
            <div class="metric-card-value">${formatPercent(metrics.map || 0)}</div>
            <div class="metric-card-model">MAP</div>
            <div class="metric-card-subline">
                MRR ${formatPercent(metrics.mrr || 0)} · nDCG@10 ${formatPercent(metricAt(metrics.ndcg_at_k, 10))}
            </div>
        `;
        metricsCards.appendChild(card);
    });
}

function displayCharts(results) {
    Object.values(charts).forEach((chart) => {
        if (chart) chart.destroy();
    });
    charts = Object.fromEntries(Object.keys(charts).map((key) => [key, null]));

    const models = Object.keys(results).filter((model) => !results[model].error && results[model].available !== false);
    if (!models.length || typeof Chart === 'undefined') {
        return;
    }
    const colors = getModelColors(models);
    const overlayPrecision = overlayPrecisionCheck?.checked ?? true;
    const overlayRecall = overlayRecallCheck?.checked ?? true;
    const overlayF1 = overlayF1Check?.checked ?? true;
    const showInterpolated = showInterpolatedCheck?.checked ?? true;

    if (overlayPrecision) {
        charts.precision = createLineChart('precision-chart', {
            labels: (results[models[0]]?.precision_at_k || []).map((point) => `P@${point.k}`),
            datasets: models.map((model, index) => metricDataset(model, results[model].precision_at_k, colors[index]))
        });
    }

    if (overlayRecall) {
        charts.recall = createLineChart('recall-chart', {
            labels: (results[models[0]]?.recall_at_k || []).map((point) => `R@${point.k}`),
            datasets: models.map((model, index) => metricDataset(model, results[model].recall_at_k, colors[index]))
        });
    }

    if (overlayF1) {
        charts.f1 = createLineChart('f1-chart', {
            labels: (results[models[0]]?.f1_at_k || []).map((point) => `F1@${point.k}`),
            datasets: models.map((model, index) => metricDataset(model, results[model].f1_at_k, colors[index]))
        });
    }

    charts.ndcg = createLineChart('ndcg-chart', {
        labels: (results[models[0]]?.ndcg_at_k || []).map((point) => `nDCG@${point.k}`),
        datasets: models.map((model, index) => metricDataset(model, results[model].ndcg_at_k, colors[index]))
    });

    const fbetaData = { labels: [], datasets: [] };
    models.forEach((model, index) => {
        const beta05 = (results[model].f_beta_scores || []).filter((point) => point.beta === 0.5);
        const beta20 = (results[model].f_beta_scores || []).filter((point) => point.beta === 2.0);
        if (beta05.length) {
            fbetaData.labels = beta05.map((point) => `F@${point.k}`);
            fbetaData.datasets.push(metricDataset(`${model.toUpperCase()} beta=0.5`, beta05, colors[index], [5, 5]));
            fbetaData.datasets.push(metricDataset(`${model.toUpperCase()} beta=2.0`, beta20, colors[index]));
        }
    });
    charts.fbeta = createLineChart('fbeta-chart', fbetaData);

    charts.pAtR = createBarChart('p-at-r-chart', {
        labels: (results[models[0]]?.precision_at_recall || []).map((point) => `R=${Math.round(point.recall * 100)}%`),
        datasets: models.map((model, index) => ({
            label: model.toUpperCase(),
            data: (results[model].precision_at_recall || []).map((point) => point.precision),
            backgroundColor: colors[index],
            borderColor: colors[index],
            borderWidth: 2
        }))
    });

    charts.prCurve = createScatterChart('pr-curve-chart', {
        datasets: models.map((model, index) => ({
            label: model.toUpperCase(),
            data: (results[model].pr_curve || []).map((point) => ({ x: point.recall, y: point.precision })),
            borderColor: colors[index],
            backgroundColor: `${colors[index]}33`,
            tension: 0.3,
            fill: false,
            pointRadius: 3,
            borderWidth: 2
        }))
    });

    if (showInterpolated) {
        charts.prInterpolated = createLineChart('pr-interpolated-chart', {
            labels: (results[models[0]]?.interpolated_precision || []).map((point) => `${Math.round(point.recall * 100)}%`),
            datasets: models.map((model, index) => metricDataset(model, results[model].interpolated_precision, colors[index]))
        });
    }
}

function metricDataset(label, points, color, dash = []) {
    return {
        label: label.toUpperCase(),
        data: (points || []).map((point) => point.value ?? point.precision ?? 0),
        borderColor: color,
        backgroundColor: `${color}33`,
        tension: 0.3,
        fill: false,
        borderDash: dash,
        borderWidth: 2
    };
}

function createLineChart(canvasId, data) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return null;
    return new Chart(canvas.getContext('2d'), {
        type: 'line',
        data,
        options: chartOptions()
    });
}

function createBarChart(canvasId, data) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return null;
    return new Chart(canvas.getContext('2d'), {
        type: 'bar',
        data,
        options: chartOptions()
    });
}

function createScatterChart(canvasId, data) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return null;
    return new Chart(canvas.getContext('2d'), {
        type: 'line',
        data,
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: { legend: { display: true, position: 'bottom' } },
            scales: {
                x: {
                    type: 'linear',
                    min: 0,
                    max: 1,
                    title: { display: true, text: 'Recall' },
                    ticks: { callback: (value) => `${Math.round(value * 100)}%` }
                },
                y: {
                    min: 0,
                    max: 1,
                    title: { display: true, text: 'Precision' },
                    ticks: { callback: (value) => `${Math.round(value * 100)}%` }
                }
            }
        }
    });
}

function chartOptions() {
    return {
        responsive: true,
        maintainAspectRatio: true,
        plugins: { legend: { display: true, position: 'bottom' } },
        scales: {
            y: {
                beginAtZero: true,
                max: 1.0,
                ticks: { callback: (value) => `${Math.round(value * 100)}%` }
            }
        }
    };
}

function displayComparisonTable(results) {
    const models = Object.keys(results).filter((model) => !results[model].error && results[model].available !== false);
    if (!models.length) {
        comparisonTable.innerHTML = '<p>沒有可比較的結果</p>';
        return;
    }

    const best = {
        map: Math.max(...models.map((model) => results[model].map || 0)),
        mrr: Math.max(...models.map((model) => results[model].mrr || 0)),
        r_precision: Math.max(...models.map((model) => results[model].r_precision || 0)),
        bpref: Math.max(...models.map((model) => results[model].bpref || 0))
    };

    comparisonTable.innerHTML = `
        <div class="comparison-table">
            <table>
                <thead>
                    <tr>
                        <th>Model</th>
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
                    ${models.map((model) => {
                        const metrics = results[model];
                        return `
                            <tr>
                                <td><strong>${escapeHtml(model.toUpperCase())}</strong></td>
                                <td class="${bestClass(metrics.map, best.map)}">${formatPercent(metrics.map || 0)}</td>
                                <td class="${bestClass(metrics.mrr, best.mrr)}">${formatPercent(metrics.mrr || 0)}</td>
                                <td class="${bestClass(metrics.r_precision, best.r_precision)}">${formatPercent(metrics.r_precision || 0)}</td>
                                <td class="${bestClass(metrics.bpref, best.bpref)}">${formatPercent(metrics.bpref || 0)}</td>
                                <td>${formatPercent(metricAt(metrics.precision_at_k, 5))}</td>
                                <td>${formatPercent(metricAt(metrics.precision_at_k, 10))}</td>
                                <td>${formatPercent(metricAt(metrics.recall_at_k, 10))}</td>
                                <td>${formatPercent(metricAt(metrics.f1_at_k, 10))}</td>
                                <td>${formatPercent(metricAt(metrics.ndcg_at_k, 10))}</td>
                            </tr>
                        `;
                    }).join('')}
                </tbody>
            </table>
        </div>
    `;
}

function displayPerQueryBreakdown(perQuery) {
    if (!perQueryBreakdown) {
        return;
    }
    const rows = Object.entries(perQuery);
    if (!rows.length) {
        perQueryBreakdown.innerHTML = '<p>No per-query data.</p>';
        return;
    }

    perQueryBreakdown.innerHTML = rows.map(([queryId, queryData]) => `
        <details class="per-query-item">
            <summary>
                <strong>${escapeHtml(queryId)}</strong>
                <span>${escapeHtml(queryData.query || '')}</span>
            </summary>
            <div class="per-query-model-grid">
                ${Object.entries(queryData.models || {}).map(([model, modelData]) => `
                    <div class="per-query-model">
                        <div class="per-query-model-title">${escapeHtml(model.toUpperCase())}</div>
                        <div class="per-query-metrics">
                            <span>AP ${formatPercent(modelData.metrics?.ap || 0)}</span>
                            <span>RR ${formatPercent(modelData.metrics?.rr || 0)}</span>
                            <span>nDCG@10 ${formatPercent(metricAt(modelData.metrics?.ndcg_at_k, 10))}</span>
                        </div>
                        <ol class="judged-results">
                            ${(modelData.top_results || []).slice(0, 5).map((result) => `
                                <li>
                                    <span class="judgment-pill ${result.relevance > 0 ? 'relevant' : 'unjudged'}">
                                        ${result.judged ? `rel ${result.relevance}` : 'unjudged'}
                                    </span>
                                    <span>${escapeHtml(result.title || `Doc ${result.doc_id}`)}</span>
                                </li>
                            `).join('')}
                        </ol>
                    </div>
                `).join('')}
            </div>
        </details>
    `).join('');
}

function parseKValues(value, topK) {
    const values = (value || '5,10,20')
        .split(',')
        .map((item) => parseInt(item.trim(), 10))
        .filter((item) => Number.isFinite(item) && item > 0 && item <= 100);
    values.push(topK);
    return [...new Set(values)].sort((a, b) => a - b);
}

function normalizeApiPayload(payload) {
    return payload.data || payload;
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function getIrSessionId() {
    const key = 'cnirs_session_id';
    let value = localStorage.getItem(key);
    if (!value) {
        value = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
        localStorage.setItem(key, value);
    }
    return value;
}

function metricAt(series, k) {
    const item = (series || []).find((point) => point.k === k);
    if (item) {
        return item.value || 0;
    }
    const fallback = (series || [])[0];
    return fallback?.value || 0;
}

function bestClass(value, best) {
    return (value || 0) === best ? 'best-value' : '';
}

function getModelColors(models) {
    const palette = ['#2563eb', '#16a34a', '#dc2626', '#9333ea', '#ea580c', '#0891b2'];
    return models.map((_, index) => palette[index % palette.length]);
}

function formatPercent(value) {
    return `${((value || 0) * 100).toFixed(1)}%`;
}

function formatNumber(value) {
    return new Intl.NumberFormat('en-US').format(value || 0);
}

function escapeHtml(value) {
    return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

function escapeAttr(value) {
    return escapeHtml(value).replace(/`/g, '&#096;');
}
