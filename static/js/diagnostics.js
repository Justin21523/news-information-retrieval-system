// Ranking diagnostics page.

const diagQuery = document.getElementById('diag-query');
const diagDocId = document.getElementById('diag-doc-id');
const diagButton = document.getElementById('run-diagnostics-btn');
const diagLoading = document.getElementById('diagnostics-loading');
const diagResults = document.getElementById('diagnostics-results');

diagButton?.addEventListener('click', runDiagnostics);

window.addEventListener('DOMContentLoaded', initializeDiagnosticsFromUrl);

function initializeDiagnosticsFromUrl() {
    const params = new URLSearchParams(window.location.search);
    const query = params.get('q') || params.get('query');
    const docId = params.get('doc_id') || params.get('doc');
    const models = params.get('models');
    const shouldRun = params.get('run') === '1' || params.get('run') === 'true';

    if (query) {
        diagQuery.value = query;
    }
    if (docId) {
        diagDocId.value = docId;
    }
    if (models) {
        const selected = new Set(models.split(',').map(item => item.trim()).filter(Boolean));
        document.querySelectorAll('.diag-model').forEach(checkbox => {
            checkbox.checked = selected.has(checkbox.value);
        });
    }
    if (shouldRun) {
        window.setTimeout(runDiagnostics, 300);
    }
}

async function runDiagnostics() {
    const query = diagQuery.value.trim();
    const docId = diagDocId.value.trim();
    const models = Array.from(document.querySelectorAll('.diag-model:checked')).map(item => item.value);
    if (!query || !docId || !models.length) {
        alert('Query, document ID, and at least one model are required.');
        return;
    }
    diagLoading.style.display = 'block';
    diagResults.innerHTML = '';
    try {
        const response = await fetch('api/diagnostics/ranking', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-IR-Session': getIrSessionId()
            },
            body: JSON.stringify({ query, doc_id: docId, models })
        });
        const payload = await response.json();
        if (!payload.ok && !payload.success) {
            throw new Error(payload.error?.message || 'Diagnostics failed');
        }
        renderDiagnostics(payload.data || payload);
    } catch (error) {
        diagResults.innerHTML = `<div class="error">${escapeHtml(error.message)}</div>`;
    } finally {
        diagLoading.style.display = 'none';
    }
}

function renderDiagnostics(data) {
    const doc = data.document || {};
    const metadata = doc.metadata || {};
    const modelCards = Object.entries(data.models || {}).map(([model, info]) => `
        <article class="diagnostics-card">
            <h3>${escapeHtml(model.toUpperCase())}</h3>
            <div class="explain-chip-row">
                <span class="explain-chip score-chip"><strong>total</strong>${formatScore(info.total_score || 0)}</span>
                ${info.doc_length ? `<span class="explain-chip"><strong>doc length</strong>${info.doc_length}</span>` : ''}
                ${info.smoothing ? `<span class="explain-chip"><strong>smoothing</strong>${escapeHtml(info.smoothing)}</span>` : ''}
            </div>
            ${renderTerms(info.terms || [])}
            ${info.components ? Object.entries(info.components).map(([name, component]) => `
                <h4>${escapeHtml(name.toUpperCase())}</h4>
                ${renderTerms(component.terms || [])}
            `).join('') : ''}
        </article>
    `).join('');
    diagResults.innerHTML = `
        <section class="diagnostics-document">
            <h2>${escapeHtml(doc.title || `Document ${doc.doc_id}`)}</h2>
            <div class="explain-chip-row">
                <span class="explain-chip"><strong>doc_id</strong>${escapeHtml(doc.doc_id)}</span>
                <span class="explain-chip"><strong>article</strong>${escapeHtml(doc.article_id || '-')}</span>
                <span class="explain-chip"><strong>source</strong>${escapeHtml(metadata.source_label || metadata.source || '-')}</span>
                <span class="explain-chip"><strong>query terms</strong>${escapeHtml((data.query_terms || []).join(', '))}</span>
            </div>
        </section>
        <section class="diagnostics-card">
            <h3>Field-Aware Ranking Signals</h3>
            ${renderCoverage(data.query_coverage || {})}
            ${renderFieldContributions(data.field_contributions || {})}
            ${renderFieldMatrix(data.field_match_matrix || [])}
        </section>
        <div class="diagnostics-card-grid">${modelCards}</div>
    `;
}

function renderCoverage(coverage) {
    return `
        <div class="explain-chip-row">
            <span class="explain-chip score-chip"><strong>coverage</strong>${formatPercent(coverage.coverage_ratio || 0)}</span>
            <span class="explain-chip"><strong>matched</strong>${escapeHtml((coverage.matched_terms || []).join(', ') || '-')}</span>
            <span class="explain-chip"><strong>missing</strong>${escapeHtml((coverage.missing_terms || []).join(', ') || '-')}</span>
        </div>
    `;
}

function renderFieldContributions(fieldContributions) {
    const fields = fieldContributions.fields || {};
    if (!Object.keys(fields).length) return '<p class="explain-muted">No field contributions.</p>';
    return `
        <table class="diagnostic-table">
            <thead><tr><th>Field</th><th>Weight</th><th>Matches</th><th>Boost</th></tr></thead>
            <tbody>
                ${Object.entries(fields).map(([field, info]) => `
                    <tr>
                        <td>${escapeHtml(field)}</td>
                        <td>${formatScore(info.weight)}</td>
                        <td>${escapeHtml((info.matched_terms || []).join(', ') || '-')}</td>
                        <td>${formatScore(info.boost)}</td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;
}

function renderFieldMatrix(rows) {
    if (!rows.length) return '';
    const fields = ['title', 'tags', 'category', 'content'];
    return `
        <h4>Field Match Heatmap</h4>
        <table class="diagnostic-table field-heatmap">
            <thead><tr><th>Term</th>${fields.map(field => `<th>${escapeHtml(field)}</th>`).join('')}</tr></thead>
            <tbody>
                ${rows.map(row => `
                    <tr>
                        <td>${escapeHtml(row.term)}</td>
                        ${fields.map(field => `<td class="${row[field] ? 'heat-on' : 'heat-off'}">${row[field] ? 'match' : '-'}</td>`).join('')}
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;
}

function renderTerms(terms) {
    if (!terms.length) {
        return '<p class="explain-muted">No term contributions.</p>';
    }
    return `
        <table class="diagnostic-table">
            <thead>
                <tr><th>Term</th><th>TF</th><th>DF</th><th>IDF/P(C)</th><th>Weight/Prob</th><th>Contribution</th></tr>
            </thead>
            <tbody>
                ${terms.map(term => `
                    <tr>
                        <td>${escapeHtml(term.term)}</td>
                        <td>${escapeHtml(term.tf ?? 0)}</td>
                        <td>${escapeHtml(term.df ?? '')}</td>
                        <td>${escapeHtml(term.idf ?? term.p_collection ?? '')}</td>
                        <td>${escapeHtml(term.doc_weight ?? term.p_smoothed ?? term.normalized_tf ?? '')}</td>
                        <td>${formatScore(term.score ?? term.log_prob ?? 0)}</td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;
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

function formatScore(value) {
    const number = Number(value || 0);
    return Number.isFinite(number) ? number.toFixed(4) : '0.0000';
}

function formatPercent(value) {
    return `${(Number(value || 0) * 100).toFixed(1)}%`;
}

function escapeHtml(value) {
    return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}
