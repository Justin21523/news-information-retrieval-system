// CNIRS - Shared IR explanation renderer

(function () {
    function escapeHtml(value) {
        return String(value ?? '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function formatScore(value) {
        const number = Number(value || 0);
        if (!Number.isFinite(number)) return '0.0000';
        return number.toFixed(4);
    }

    function chip(label, value, className = '') {
        if (value === undefined || value === null || value === '') return '';
        return `<span class="explain-chip ${className}"><strong>${escapeHtml(label)}</strong>${escapeHtml(value)}</span>`;
    }

    function chipList(items, className = '') {
        return (items || []).map(item => `<span class="explain-chip ${className}">${escapeHtml(item)}</span>`).join('');
    }

    function scoreChips(scores) {
        const entries = Object.entries(scores || {});
        if (!entries.length) return '<span class="explain-muted">No component scores</span>';
        return entries
            .map(([name, score]) => chip(name, formatScore(score), `score-chip score-${escapeHtml(name)}`))
            .join('');
    }

    function fieldMatchChips(fieldMatches) {
        const entries = Object.entries(fieldMatches || {});
        if (!entries.length) return '<span class="explain-muted">No field matches</span>';
        return entries.map(([field, terms]) => {
            const text = Array.isArray(terms) ? terms.join(', ') : String(terms || '');
            return chip(field, text, 'field-chip');
        }).join('');
    }

    function boostChips(fieldBoost) {
        const entries = Object.entries(fieldBoost || {}).filter(([key]) => key !== 'boost');
        const boost = fieldBoost && fieldBoost.boost !== undefined
            ? chip('boost', formatScore(fieldBoost.boost), 'boost-chip')
            : '';
        const fields = entries.map(([field, terms]) => {
            const text = Array.isArray(terms) ? terms.join(', ') : String(terms || '');
            return chip(field, text, 'boost-chip');
        }).join('');
        return boost || fields ? `${boost}${fields}` : '<span class="explain-muted">No field boost</span>';
    }

    function rankFeatureSummary(features) {
        const rows = [];
        if (features?.snippet_source) {
            rows.push(chip('snippet', features.snippet_source, 'feature-chip'));
        }
        if (features?.index_cache_used !== undefined) {
            rows.push(chip('cache', features.index_cache_used ? 'hit' : 'rebuilt', 'feature-chip'));
        }
        if (features?.bm25 && Object.keys(features.bm25).length) {
            rows.push(chip('bm25 terms', Object.keys(features.bm25.term_details || {}).length, 'feature-chip'));
        }
        if (features?.lm && Object.keys(features.lm).length) {
            rows.push(chip('lm', features.lm.smoothing || 'query likelihood', 'feature-chip'));
        }
        return rows.join('') || '<span class="explain-muted">No ranking features</span>';
    }

    function renderResultPanel(result, options = {}) {
        const explanation = result?.explanation || {};
        const features = explanation.ranking_features || {};
        const matched = explanation.matched_terms || [];
        const expanded = explanation.expanded_terms || [];
        const title = options.title || 'Why this result?';
        const open = options.open ? ' open' : '';
        const query = options.query || '';
        const docId = result?.doc_id ?? '';
        const model = result?.model || '';

        return `
            <details class="explain-panel result-explanation" data-doc-id="${escapeHtml(docId)}" data-query="${escapeHtml(query)}" data-model="${escapeHtml(model)}"${open}>
                <summary>${escapeHtml(title)}</summary>
                <div class="explain-section">
                    <div class="explain-section-title">Matched Terms</div>
                    <div class="explain-chip-row">${matched.length ? chipList(matched, 'term-chip') : '<span class="explain-muted">No direct matched terms</span>'}</div>
                </div>
                ${expanded.length ? `
                <div class="explain-section">
                    <div class="explain-section-title">Expanded Terms</div>
                    <div class="explain-chip-row">${chipList(expanded, 'term-chip')}</div>
                </div>` : ''}
                <div class="explain-section">
                    <div class="explain-section-title">Component Scores</div>
                    <div class="explain-chip-row">${scoreChips(explanation.component_scores)}</div>
                </div>
                <div class="explain-section">
                    <div class="explain-section-title">Field Matches</div>
                    <div class="explain-chip-row">${fieldMatchChips(explanation.field_matches)}</div>
                </div>
                <div class="explain-section">
                    <div class="explain-section-title">Field Boost</div>
                    <div class="explain-chip-row">${boostChips(features.field_boost)}</div>
                </div>
                <div class="explain-section">
                    <div class="explain-section-title">Ranking Features</div>
                    <div class="explain-chip-row">${rankFeatureSummary(features)}</div>
                </div>
                <div class="explain-section">
                    <div class="explain-section-title">Ranking Diagnostics</div>
                    <button class="btn-mini diagnostics-btn" type="button" onclick="window.ExplanationPanel.loadDiagnostics(this)">Load diagnostics</button>
                    <div class="diagnostics-output"></div>
                </div>
                <div class="explain-section">
                    <div class="explain-section-title">Feedback</div>
                    <div class="feedback-controls">
                        <button class="btn-mini" type="button" onclick="window.ExplanationPanel.sendFeedback(this, 'click')">Track click</button>
                        <button class="btn-mini" type="button" onclick="window.ExplanationPanel.sendFeedback(this, 'relevance', 3)">Relevant</button>
                        <button class="btn-mini" type="button" onclick="window.ExplanationPanel.sendFeedback(this, 'relevance', 0)">Not relevant</button>
                    </div>
                    <div class="feedback-status explain-muted"></div>
                </div>
                ${explanation.relation_reason ? renderRelatedReason(result, { embedded: true }) : ''}
            </details>
        `;
    }

    async function loadDiagnostics(button) {
        const panel = button.closest('.explain-panel');
        const output = panel?.querySelector('.diagnostics-output');
        if (!panel || !output) return;
        const query = panel.dataset.query || '';
        const docId = panel.dataset.docId || '';
        output.innerHTML = '<span class="explain-muted">Loading diagnostics...</span>';
        try {
            const response = await fetch('api/diagnostics/ranking', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-IR-Session': getSessionId()
                },
                body: JSON.stringify({
                    query,
                    doc_id: docId,
                    models: ['bm25', 'tfidf', 'lm']
                })
            });
            const payload = await response.json();
            if (!payload.ok && !payload.success) {
                throw new Error(payload.error?.message || 'Diagnostics failed');
            }
            output.innerHTML = renderDiagnostics(payload.data || payload);
        } catch (error) {
            output.innerHTML = `<span class="error">${escapeHtml(error.message)}</span>`;
        }
    }

    function renderDiagnostics(data) {
        const models = Object.entries(data.models || {});
        if (!models.length) return '<span class="explain-muted">No diagnostics available</span>';
        return models.map(([model, info]) => `
            <div class="diagnostic-model">
                <div class="diagnostic-model-title">${escapeHtml(model.toUpperCase())} ${chip('total', formatScore(info.total_score || 0), 'score-chip')}</div>
                ${renderDiagnosticTerms(info.terms || [])}
                ${info.components ? Object.entries(info.components).map(([name, component]) => `
                    <div class="diagnostic-submodel">${escapeHtml(name.toUpperCase())}</div>
                    ${renderDiagnosticTerms(component.terms || [])}
                `).join('') : ''}
            </div>
        `).join('');
    }

    function renderDiagnosticTerms(terms) {
        if (!terms.length) return '<span class="explain-muted">No term contributions</span>';
        return `
            <table class="diagnostic-table">
                <thead><tr><th>Term</th><th>TF</th><th>IDF/P(C)</th><th>Contribution</th></tr></thead>
                <tbody>
                    ${terms.slice(0, 8).map(term => `
                        <tr>
                            <td>${escapeHtml(term.term)}</td>
                            <td>${escapeHtml(term.tf ?? 0)}</td>
                            <td>${escapeHtml(term.idf ?? term.p_collection ?? '')}</td>
                            <td>${formatScore(term.score ?? term.log_prob ?? 0)}</td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        `;
    }

    async function sendFeedback(button, eventType, relevanceGrade = null) {
        const panel = button.closest('.explain-panel');
        const status = panel?.querySelector('.feedback-status');
        if (!panel) return;
        try {
            const response = await fetch('api/feedback', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-IR-Session': getSessionId()
                },
                body: JSON.stringify({
                    event_type: eventType,
                    query: panel.dataset.query || '',
                    model: panel.dataset.model || '',
                    doc_id: panel.dataset.docId || '',
                    relevance_grade: relevanceGrade,
                    metadata: { source: 'explanation_panel' }
                })
            });
            const payload = await response.json();
            if (!payload.ok && !payload.success) {
                throw new Error(payload.error?.message || 'Feedback failed');
            }
            if (status) status.textContent = 'Feedback saved';
        } catch (error) {
            if (status) status.textContent = error.message;
        }
    }

    function getSessionId() {
        const key = 'cnirs_session_id';
        let value = localStorage.getItem(key);
        if (!value) {
            value = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
            localStorage.setItem(key, value);
        }
        return value;
    }

    function renderDocumentPanel(explanation, taxonomy = {}) {
        const signals = explanation?.signals || {};
        const sections = explanation?.sections || {};
        return `
            <div class="explain-panel explain-panel-static">
                <div class="explain-section">
                    <div class="explain-section-title">Available Analysis</div>
                    <div class="explain-chip-row">
                        ${chip('summary', sections.summary ? 'available' : 'empty', 'feature-chip')}
                        ${chip('keywords', signals.keyword_count || 0, 'feature-chip')}
                        ${chip('KWIC', signals.kwic_match_count || 0, 'feature-chip')}
                        ${chip('related', signals.related_count || 0, 'feature-chip')}
                    </div>
                </div>
                <div class="explain-section">
                    <div class="explain-section-title">Metadata Signals</div>
                    <div class="explain-chip-row">
                        ${chip('topic', taxonomy.topic || signals.taxonomy_topic || '', 'field-chip')}
                        ${chip('source', taxonomy.source || signals.source || '', 'field-chip')}
                        ${chip('category', taxonomy.category || signals.category || '', 'field-chip')}
                    </div>
                </div>
            </div>
        `;
    }

    function renderRelatedReason(result, options = {}) {
        const reason = result?.relation_reason || result?.explanation?.relation_reason || {};
        const content = `
            <div class="explain-section">
                <div class="explain-section-title">Relation Reason</div>
                <div class="explain-chip-row">
                    ${chip('method', reason.method || 'hybrid_lexical', 'reason-chip')}
                    ${reason.same_taxonomy_topic ? chip('taxonomy', 'same', 'reason-chip') : ''}
                    ${reason.same_category ? chip('category', 'same', 'reason-chip') : ''}
                    ${reason.same_source ? chip('source', 'same', 'reason-chip') : ''}
                    ${(reason.shared_tags || []).length ? chip('shared tags', reason.shared_tags.join(', '), 'reason-chip') : ''}
                </div>
            </div>
        `;
        if (options.embedded) return content;
        return `<div class="explain-panel explain-panel-static">${content}</div>`;
    }

    function renderModelBadge(modelInfo, fallbackId = '') {
        const id = modelInfo?.id || fallbackId;
        const name = modelInfo?.name || id.toUpperCase();
        const description = modelInfo?.description || '';
        return `
            <div class="model-badge">
                <span class="model-badge-name">${escapeHtml(name)}</span>
                ${description ? `<span class="model-badge-description">${escapeHtml(description)}</span>` : ''}
            </div>
        `;
    }

    window.ExplanationPanel = {
        escapeHtml,
        formatScore,
        renderResultPanel,
        renderDocumentPanel,
        renderRelatedReason,
        renderModelBadge,
        loadDiagnostics,
        sendFeedback,
        scoreChips,
    };
})();
