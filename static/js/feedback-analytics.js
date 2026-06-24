// Feedback analytics dashboard.

const feedbackDays = document.getElementById('feedback-days');
const feedbackLimit = document.getElementById('feedback-limit');
const refreshFeedbackButton = document.getElementById('refresh-feedback-btn');
const feedbackLoading = document.getElementById('feedback-loading');
const feedbackDashboard = document.getElementById('feedback-dashboard');

refreshFeedbackButton?.addEventListener('click', loadFeedbackAnalytics);
document.addEventListener('DOMContentLoaded', loadFeedbackAnalytics);

async function loadFeedbackAnalytics() {
    const days = feedbackDays?.value || '30';
    const limit = feedbackLimit?.value || '20';
    feedbackLoading.style.display = 'block';
    feedbackDashboard.innerHTML = '';
    try {
        const [analyticsResponse, featuresResponse] = await Promise.all([
            fetch(`api/feedback/analytics?days=${encodeURIComponent(days)}&limit=${encodeURIComponent(limit)}`),
            fetch(`api/feedback/features?limit=${encodeURIComponent(Math.min(Number(limit) || 20, 50))}`)
        ]);
        const analyticsPayload = await analyticsResponse.json();
        const featuresPayload = await featuresResponse.json();
        if (!analyticsPayload.ok && !analyticsPayload.success) {
            throw new Error(analyticsPayload.error?.message || 'Analytics failed');
        }
        if (!featuresPayload.ok && !featuresPayload.success) {
            throw new Error(featuresPayload.error?.message || 'Feature preview failed');
        }
        renderDashboard(analyticsPayload.data || analyticsPayload, featuresPayload.data || featuresPayload);
    } catch (error) {
        feedbackDashboard.innerHTML = `<div class="error">${escapeHtml(error.message)}</div>`;
    } finally {
        feedbackLoading.style.display = 'none';
    }
}

function renderDashboard(data, features) {
    const summary = data.summary || {};
    feedbackDashboard.innerHTML = `
        ${renderEmptyState(summary)}
        <section class="feedback-summary-grid">
            ${summaryCard('Searches', summary.total_searches)}
            ${summaryCard('Clicks', summary.total_clicks)}
            ${summaryCard('CTR', formatPercent(summary.ctr))}
            ${summaryCard('Labels', summary.total_relevance_labels)}
            ${summaryCard('Zero Results', summary.zero_result_queries)}
        </section>
        <section class="feedback-grid">
            <article class="feedback-card">
                <h2>Model Metrics</h2>
                ${renderModelMetrics(data.model_metrics || [])}
            </article>
            <article class="feedback-card">
                <h2>Zero-Result Queries</h2>
                ${renderZeroResults(data.zero_result_queries || [])}
            </article>
            <article class="feedback-card">
                <h2>Top Queries</h2>
                ${renderTopQueries(data.top_queries || [])}
            </article>
            <article class="feedback-card">
                <h2>Relevance Labels</h2>
                ${renderRelevanceDistribution(data.relevance_distribution || [])}
            </article>
            <article class="feedback-card feedback-card-wide">
                <h2>Recent Feedback</h2>
                ${renderRecentFeedback(data.recent_feedback || [])}
            </article>
            <article class="feedback-card feedback-card-wide">
                <h2>Learning-to-Rank Feature Preview</h2>
                <p class="explain-muted">Feature foundation only. This does not train or claim a production ranking model.</p>
                ${renderFeaturePreview(features.rows || [])}
            </article>
        </section>
    `;
}

function renderEmptyState(summary) {
    if (summary.has_events) return '';
    return `
        <div class="feedback-empty-state">
            No feedback events yet. Run a search, open “Why this result?”, then send click or relevance feedback.
        </div>
    `;
}

function summaryCard(label, value) {
    return `
        <article class="feedback-summary-card">
            <span>${escapeHtml(label)}</span>
            <strong>${escapeHtml(value ?? 0)}</strong>
        </article>
    `;
}

function renderModelMetrics(rows) {
    if (!rows.length) return '<p class="explain-muted">No model metrics yet.</p>';
    return `
        <table class="diagnostic-table">
            <thead><tr><th>Model</th><th>Searches</th><th>Clicks</th><th>CTR</th><th>Zero Rate</th><th>Latency</th></tr></thead>
            <tbody>
                ${rows.map(row => `
                    <tr>
                        <td>${escapeHtml(row.model)}</td>
                        <td>${escapeHtml(row.searches)}</td>
                        <td>${escapeHtml(row.clicks)}</td>
                        <td>${formatPercent(row.ctr)}</td>
                        <td>${formatPercent(row.zero_result_rate)}</td>
                        <td>${formatMs(row.avg_latency)}</td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;
}

function renderZeroResults(rows) {
    if (!rows.length) return '<p class="explain-muted">No zero-result queries in this window.</p>';
    return `
        <div class="feedback-list">
            ${rows.map(row => `
                <div class="feedback-list-item">
                    <strong>${escapeHtml(row.query)}</strong>
                    <span>${escapeHtml(row.model || '-')} · ${escapeHtml(row.count)} times</span>
                </div>
            `).join('')}
        </div>
    `;
}

function renderTopQueries(rows) {
    if (!rows.length) return '<p class="explain-muted">No queries yet.</p>';
    return `
        <div class="feedback-list">
            ${rows.map(row => `
                <div class="feedback-list-item">
                    <strong>${escapeHtml(row.query)}</strong>
                    <span>${escapeHtml(row.count)} searches · ${formatMs(row.avg_latency)}</span>
                </div>
            `).join('')}
        </div>
    `;
}

function renderRelevanceDistribution(rows) {
    if (!rows.length) return '<p class="explain-muted">No labels yet.</p>';
    const maxCount = Math.max(...rows.map(row => Number(row.count || 0)), 1);
    return `
        <div class="relevance-bars">
            ${rows.map(row => `
                <div class="relevance-bar-row">
                    <span>Grade ${escapeHtml(row.grade)}</span>
                    <div class="relevance-bar"><i style="width:${(Number(row.count || 0) / maxCount) * 100}%"></i></div>
                    <strong>${escapeHtml(row.count)}</strong>
                </div>
            `).join('')}
        </div>
    `;
}

function renderRecentFeedback(rows) {
    if (!rows.length) return '<p class="explain-muted">No feedback events yet.</p>';
    return `
        <table class="diagnostic-table">
            <thead><tr><th>Time</th><th>Type</th><th>Query</th><th>Model</th><th>Doc</th><th>Label</th></tr></thead>
            <tbody>
                ${rows.map(row => `
                    <tr>
                        <td>${escapeHtml(row.timestamp)}</td>
                        <td>${escapeHtml(row.event_type)}</td>
                        <td>${escapeHtml(row.query || '-')}</td>
                        <td>${escapeHtml(row.model || '-')}</td>
                        <td>${escapeHtml(row.article_id || row.doc_id || '-')}</td>
                        <td>${escapeHtml(row.relevance_grade ?? '-')}</td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;
}

function renderFeaturePreview(rows) {
    if (!rows.length) return '<p class="explain-muted">No feature rows yet.</p>';
    return `
        <table class="diagnostic-table">
            <thead><tr><th>Query</th><th>Model</th><th>Doc</th><th>Label</th><th>Field Boost</th><th>BM25</th><th>TF-IDF</th><th>LM</th></tr></thead>
            <tbody>
                ${rows.slice(0, 20).map(row => `
                    <tr>
                        <td>${escapeHtml(row.query)}</td>
                        <td>${escapeHtml(row.model)}</td>
                        <td>${escapeHtml(row.article_id || row.doc_id)}</td>
                        <td>${formatScore(row.label)}</td>
                        <td>${formatScore(row.features?.field_boost)}</td>
                        <td>${formatScore(row.features?.bm25_score)}</td>
                        <td>${formatScore(row.features?.tfidf_score)}</td>
                        <td>${formatScore(row.features?.lm_score)}</td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;
}

function formatPercent(value) {
    return `${(Number(value || 0) * 100).toFixed(1)}%`;
}

function formatMs(value) {
    return `${(Number(value || 0) * 1000).toFixed(1)} ms`;
}

function formatScore(value) {
    const number = Number(value || 0);
    return Number.isFinite(number) ? number.toFixed(4) : '0.0000';
}

function escapeHtml(value) {
    return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}
