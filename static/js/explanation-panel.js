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

        return `
            <details class="explain-panel result-explanation"${open}>
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
                ${explanation.relation_reason ? renderRelatedReason(result, { embedded: true }) : ''}
            </details>
        `;
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
        scoreChips,
    };
})();
