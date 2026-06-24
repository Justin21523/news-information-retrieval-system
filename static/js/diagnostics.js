// Ranking diagnostics page.

const diagQuery = document.getElementById('diag-query');
const diagDocId = document.getElementById('diag-doc-id');
const diagButton = document.getElementById('run-diagnostics-btn');
const diagLoading = document.getElementById('diagnostics-loading');
const diagResults = document.getElementById('diagnostics-results');

diagButton?.addEventListener('click', runDiagnostics);

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
        <div class="diagnostics-card-grid">${modelCards}</div>
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

function escapeHtml(value) {
    return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}
