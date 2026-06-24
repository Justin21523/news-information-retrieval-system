/**
 * CNIRS - Search Results Analytics and Visualization
 *
 * Provides visual analytics for search results including:
 * - Source distribution (pie chart)
 * - Category distribution (doughnut chart)
 * - Score distribution (bar chart)
 * - Quick model comparison
 */

// Chart instances (for cleanup on re-render)
let sourceChart = null;
let categoryChart = null;
let scoreChart = null;

// Color palette for charts
const CHART_COLORS = [
    '#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6',
    '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1'
];

// DOM Elements
const analyticsPanel = document.getElementById('analytics-panel');
const analyticsContent = document.getElementById('analytics-content');
const toggleAnalyticsBtn = document.getElementById('toggle-analytics');
const runQuickCompareBtn = document.getElementById('run-quick-compare');
const modelCompareResults = document.getElementById('model-compare-results');

// Analytics panel state
let analyticsExpanded = true;

/**
 * Initialize analytics module
 */
function initAnalytics() {
    // Toggle analytics panel
    if (toggleAnalyticsBtn) {
        toggleAnalyticsBtn.addEventListener('click', () => {
            analyticsExpanded = !analyticsExpanded;
            analyticsContent.style.display = analyticsExpanded ? 'block' : 'none';
            toggleAnalyticsBtn.textContent = analyticsExpanded ? '▼' : '▲';
        });
    }

    // Quick compare button
    if (runQuickCompareBtn) {
        runQuickCompareBtn.addEventListener('click', runQuickModelCompare);
    }
}

/**
 * Update analytics visualizations with search results
 * @param {Object} data - Search response data
 */
function updateAnalytics(data) {
    if (!data || !data.results || data.results.length === 0) {
        analyticsPanel.style.display = 'none';
        return;
    }

    // Show analytics panel
    analyticsPanel.style.display = 'block';

    // Build distribution data
    const sourceDistribution = buildDistribution(data.results, 'source');
    const categoryDistribution = buildDistribution(data.results, 'category_name', 'category');
    const scoreDistribution = buildScoreDistribution(data.results);

    // Render charts
    renderPieChart('source-chart', sourceDistribution, '來源');
    renderDoughnutChart('category-chart', categoryDistribution, '類別');
    renderBarChart('score-chart', scoreDistribution);
}

/**
 * Build distribution from results
 * @param {Array} results - Search results
 * @param {string} field - Primary field name
 * @param {string} fallbackField - Fallback field name
 * @returns {Object} Distribution {label: count}
 */
function buildDistribution(results, field, fallbackField = null) {
    const distribution = {};

    results.forEach(result => {
        let value = result[field] ||
                   (result.metadata && result.metadata[field]) ||
                   (fallbackField && (result[fallbackField] || (result.metadata && result.metadata[fallbackField]))) ||
                   '未知';

        if (!value || value === '') {
            value = '未知';
        }

        distribution[value] = (distribution[value] || 0) + 1;
    });

    return distribution;
}

/**
 * Build score distribution for histogram
 * @param {Array} results - Search results
 * @returns {Object} Distribution data for bar chart
 */
function buildScoreDistribution(results) {
    const scores = results.map(r => r.score || 0);
    const min = Math.min(...scores);
    const max = Math.max(...scores);

    // Create 5 buckets
    const bucketCount = 5;
    const bucketSize = (max - min) / bucketCount || 1;
    const buckets = Array(bucketCount).fill(0);
    const labels = [];

    for (let i = 0; i < bucketCount; i++) {
        const low = min + i * bucketSize;
        const high = min + (i + 1) * bucketSize;
        labels.push(`${low.toFixed(2)}-${high.toFixed(2)}`);
    }

    scores.forEach(score => {
        let bucketIndex = Math.floor((score - min) / bucketSize);
        if (bucketIndex >= bucketCount) bucketIndex = bucketCount - 1;
        if (bucketIndex < 0) bucketIndex = 0;
        buckets[bucketIndex]++;
    });

    return { labels, values: buckets };
}

/**
 * Render pie chart
 * @param {string} canvasId - Canvas element ID
 * @param {Object} distribution - Data distribution
 * @param {string} title - Chart title
 */
function renderPieChart(canvasId, distribution, title) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    // Destroy existing chart
    if (sourceChart) {
        sourceChart.destroy();
    }

    const labels = Object.keys(distribution);
    const values = Object.values(distribution);

    sourceChart = new Chart(canvas, {
        type: 'pie',
        data: {
            labels: labels,
            datasets: [{
                data: values,
                backgroundColor: CHART_COLORS.slice(0, labels.length),
                borderWidth: 2,
                borderColor: '#ffffff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        font: { size: 11 },
                        padding: 10
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const percentage = ((context.raw / total) * 100).toFixed(1);
                            return `${context.label}: ${context.raw} (${percentage}%)`;
                        }
                    }
                }
            }
        }
    });
}

/**
 * Render doughnut chart
 * @param {string} canvasId - Canvas element ID
 * @param {Object} distribution - Data distribution
 * @param {string} title - Chart title
 */
function renderDoughnutChart(canvasId, distribution, title) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    // Destroy existing chart
    if (categoryChart) {
        categoryChart.destroy();
    }

    const labels = Object.keys(distribution);
    const values = Object.values(distribution);

    categoryChart = new Chart(canvas, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: values,
                backgroundColor: CHART_COLORS.slice(0, labels.length),
                borderWidth: 2,
                borderColor: '#ffffff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            cutout: '50%',
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        font: { size: 11 },
                        padding: 10
                    }
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                            const percentage = ((context.raw / total) * 100).toFixed(1);
                            return `${context.label}: ${context.raw} (${percentage}%)`;
                        }
                    }
                }
            }
        }
    });
}

/**
 * Render bar chart for score distribution
 * @param {string} canvasId - Canvas element ID
 * @param {Object} data - { labels: [], values: [] }
 */
function renderBarChart(canvasId, data) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    // Destroy existing chart
    if (scoreChart) {
        scoreChart.destroy();
    }

    scoreChart = new Chart(canvas, {
        type: 'bar',
        data: {
            labels: data.labels,
            datasets: [{
                label: '文檔數量',
                data: data.values,
                backgroundColor: '#3b82f6',
                borderColor: '#2563eb',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        stepSize: 1
                    }
                },
                x: {
                    ticks: {
                        font: { size: 10 },
                        maxRotation: 45
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
}

/**
 * Run quick model comparison for current query
 */
async function runQuickModelCompare() {
    const query = document.getElementById('query-input').value.trim();

    if (!query) {
        alert('請先輸入查詢關鍵字');
        return;
    }

    // Disable button and show loading
    runQuickCompareBtn.disabled = true;
    runQuickCompareBtn.innerHTML = '<span class="spinner-small"></span> 比較中...';

    const models = ['bm25', 'tfidf', 'hybrid', 'lm'];

    try {
        const response = await fetch('api/search/compare', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: query,
                models: models,
                top_k: 10
            })
        });
        const rawData = await response.json();
        const data = rawData.ok === true && rawData.data ? rawData.data : rawData;
        displayModelCompareResults(data);

    } catch (error) {
        console.error('Model comparison error:', error);
        modelCompareResults.innerHTML = '<div class="error">比較失敗，請稍後再試</div>';
        modelCompareResults.style.display = 'block';
    } finally {
        runQuickCompareBtn.disabled = false;
        runQuickCompareBtn.innerHTML = '⚡ 執行快速比較';
    }
}

/**
 * Display model comparison results
 * @param {Object} data - Comparison response
 */
function displayModelCompareResults(data) {
    const rows = Object.entries(data.models || {}).map(([model, payload]) => {
        const results = payload.results || [];
        const avgScore = results.length
            ? results.reduce((sum, result) => sum + Number(result.score || 0), 0) / results.length
            : 0;
        return {
            model,
            name: payload.model_info?.name || model.toUpperCase(),
            executionTime: payload.execution_time || 0,
            totalResults: payload.total_results || results.length,
            avgScore,
            topResult: results[0],
            available: payload.available !== false,
        };
    }).sort((a, b) => b.avgScore - a.avgScore);

    const html = `
        <table class="compare-table">
            <thead>
                <tr>
                    <th>模型</th>
                    <th>響應時間</th>
                    <th>平均分數</th>
                    <th>推薦</th>
                </tr>
            </thead>
            <tbody>
                ${rows.map((r, i) => `
                    <tr class="${i === 0 ? 'best-model' : ''}">
                        <td>${window.ExplanationPanel ? window.ExplanationPanel.escapeHtml(r.name) : r.name}</td>
                        <td>${(r.executionTime * 1000).toFixed(2)} ms</td>
                        <td>${r.avgScore.toFixed(4)}</td>
                        <td>${r.available && i === 0 ? 'best score' : (r.available ? '' : 'unavailable')}</td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
        <div class="compare-note">
            根據平均分數排序；完整排名差異請使用模型對比頁。
        </div>
    `;

    modelCompareResults.innerHTML = html;
    modelCompareResults.style.display = 'block';
}

/**
 * Hook into displayResults to update analytics
 * Override the original displayResults function
 */
const originalDisplayResults = window.displayResults || displayResults;

window.displayResults = function(data) {
    // Call original function
    if (typeof originalDisplayResults === 'function') {
        originalDisplayResults(data);
    }

    // Update analytics
    updateAnalytics(data);
};

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', initAnalytics);
