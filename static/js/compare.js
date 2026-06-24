// CNIRS - Model Comparison Page JavaScript

const queryInput = document.getElementById('query-input');
const modelCheckboxes = document.querySelectorAll('input[name="models"]');
const topkInput = document.getElementById('topk-input');
const compareBtn = document.getElementById('compare-btn');
const loading = document.getElementById('loading');
const comparisonContainer = document.getElementById('comparison-container');

queryInput.addEventListener('keypress', (event) => {
    if (event.key === 'Enter') {
        performComparison();
    }
});

compareBtn.addEventListener('click', performComparison);

function normalizeApiPayload(payload) {
    if (!payload) return payload;
    if (payload.ok === true && payload.data) {
        return { ...payload.data, success: true, meta: payload.meta || {}, raw: payload };
    }
    return payload;
}

function apiErrorMessage(payload, fallback = 'Unknown error') {
    if (!payload) return fallback;
    if (typeof payload.error === 'string') return payload.error;
    if (payload.error && payload.error.message) return payload.error.message;
    return payload.message || fallback;
}

async function performComparison() {
    const query = queryInput.value.trim();
    if (!query) {
        alert('請輸入查詢關鍵字');
        return;
    }

    const selectedModels = Array.from(modelCheckboxes)
        .filter(checkbox => checkbox.checked)
        .map(checkbox => checkbox.value);
    if (selectedModels.length === 0) {
        alert('請至少選擇一個模型');
        return;
    }

    loading.style.display = 'block';
    comparisonContainer.innerHTML = '';
    compareBtn.disabled = true;

    try {
        const response = await fetch('api/compare', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query,
                models: selectedModels,
                top_k: parseInt(topkInput.value, 10)
            })
        });
        const data = normalizeApiPayload(await response.json());
        if (data.success) {
            displayComparison(data);
        } else {
            alert(`對比錯誤: ${apiErrorMessage(data)}`);
        }
    } catch (error) {
        console.error('Comparison error:', error);
        alert('對比失敗，請稍後再試');
    } finally {
        loading.style.display = 'none';
        compareBtn.disabled = false;
    }
}

function displayComparison(data) {
    const modelEntries = Object.entries(data.models || {});
    const header = document.createElement('div');
    header.className = 'results-header';
    header.innerHTML = `
        <h2>模型對比結果</h2>
        <div class="results-meta">
            <span>Query: <strong>${escapeHtml(data.query)}</strong></span>
            <span>Models: ${modelEntries.length}</span>
            <span>Time: ${formatMs(data.meta?.execution_time || data.raw?.meta?.execution_time || 0)}</span>
        </div>
    `;
    comparisonContainer.appendChild(header);

    comparisonContainer.appendChild(createComparisonSummary(data.comparison || {}));

    const grid = document.createElement('div');
    grid.className = 'comparison-grid';
    modelEntries.forEach(([modelName, modelData]) => {
        grid.appendChild(createModelCard(modelName, modelData, data.query));
    });
    comparisonContainer.appendChild(grid);

    comparisonContainer.appendChild(createPerformanceTable(modelEntries));
}

function createComparisonSummary(comparison) {
    const section = document.createElement('section');
    section.className = 'comparison-summary explain-panel explain-panel-static';
    const overlapRows = Object.entries(comparison.overlap || {});
    const uniqueRows = Object.entries(comparison.unique_docs || {});
    const rankChanges = comparison.rank_changes || [];
    section.innerHTML = `
        <div class="explain-section">
            <div class="explain-section-title">Model Agreement</div>
            <div class="explain-chip-row">
                ${overlapRows.length ? overlapRows.map(([pair, count]) => `<span class="explain-chip"><strong>${escapeHtml(pair)}</strong>${count}</span>`).join('') : '<span class="explain-muted">No overlap data</span>'}
            </div>
        </div>
        <div class="explain-section">
            <div class="explain-section-title">Unique Documents</div>
            <div class="explain-chip-row">
                ${uniqueRows.length ? uniqueRows.map(([model, count]) => `<span class="explain-chip"><strong>${escapeHtml(model)}</strong>${count}</span>`).join('') : '<span class="explain-muted">No unique document data</span>'}
            </div>
        </div>
        ${rankChanges.length ? `
        <div class="explain-section">
            <div class="explain-section-title">Largest Rank Changes</div>
            <table class="rank-change-table">
                <thead><tr><th>Doc</th><th>Ranks</th><th>Span</th></tr></thead>
                <tbody>
                    ${rankChanges.slice(0, 8).map(item => `
                        <tr>
                            <td>${escapeHtml(item.doc_id)}</td>
                            <td>${Object.entries(item.ranks || {}).map(([model, rank]) => `${escapeHtml(model)} #${rank}`).join(' / ')}</td>
                            <td>${item.rank_span}</td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        </div>` : ''}
    `;
    return section;
}

function createModelCard(modelName, modelData, query) {
    const card = document.createElement('section');
    card.className = 'model-results';
    const executionTime = modelData.execution_time ?? modelData.response_time ?? 0;
    const modelInfo = modelData.model_info || { id: modelName, name: modelName.toUpperCase() };
    const results = modelData.results || [];

    card.innerHTML = `
        <div class="model-header">
            ${window.ExplanationPanel ? window.ExplanationPanel.renderModelBadge(modelInfo, modelName) : `<strong>${escapeHtml(modelName.toUpperCase())}</strong>`}
        </div>
        <div class="model-stats">
            <span>${modelData.total_results || results.length} results</span>
            <span>${formatMs(executionTime)}</span>
            ${modelData.available === false ? '<span class="status-unavailable">unavailable</span>' : ''}
        </div>
        ${modelData.error ? `<div class="error">${escapeHtml(modelData.error.message || 'Model unavailable')}</div>` : ''}
    `;

    const list = document.createElement('div');
    list.className = 'model-results-list';
    if (!results.length) {
        list.innerHTML = '<div class="text-center">No results</div>';
    } else {
        results.forEach(result => list.appendChild(createResultItem(result, query)));
    }
    card.appendChild(list);
    return card;
}

function createResultItem(result, query) {
    const item = document.createElement('article');
    item.className = 'model-result-item';
    item.setAttribute('data-doc-id', result.doc_id);
    item.innerHTML = `
        <div class="model-result-title">${result.rank}. ${highlightQuery(result.title || '', query)}</div>
        <div class="model-result-score">Score: ${formatScore(result.score)}</div>
        <div class="model-result-snippet">${result.highlighted_snippet || escapeHtml(result.snippet || '')}</div>
        ${window.ExplanationPanel ? window.ExplanationPanel.renderResultPanel(result, { title: 'Why this rank?', query }) : ''}
    `;
    item.addEventListener('click', (event) => {
        if (event.target.closest('.explain-panel') || event.target.tagName === 'A' || event.target.tagName === 'BUTTON') {
            return;
        }
        if (typeof openDocumentModal === 'function') {
            openDocumentModal(result.doc_id);
        }
    });
    return item;
}

function createPerformanceTable(modelEntries) {
    const section = document.createElement('section');
    section.className = 'about-section mt-20';
    section.innerHTML = `
        <h2>效能比較</h2>
        <table class="compare-table">
            <thead>
                <tr>
                    <th>模型</th>
                    <th>結果數</th>
                    <th>響應時間</th>
                    <th>平均分數</th>
                </tr>
            </thead>
            <tbody>
                ${modelEntries.map(([modelName, modelData]) => {
                    const results = modelData.results || [];
                    const avgScore = results.length
                        ? results.reduce((sum, result) => sum + Number(result.score || 0), 0) / results.length
                        : 0;
                    const executionTime = modelData.execution_time ?? modelData.response_time ?? 0;
                    return `
                        <tr>
                            <td>${escapeHtml(modelData.model_info?.name || modelName.toUpperCase())}</td>
                            <td>${modelData.total_results || results.length}</td>
                            <td>${formatMs(executionTime)}</td>
                            <td>${formatScore(avgScore)}</td>
                        </tr>
                    `;
                }).join('')}
            </tbody>
        </table>
    `;
    return section;
}

function highlightQuery(text, query) {
    const escaped = escapeHtml(text);
    if (!query) return escaped;
    return query.split(/\s+/).filter(Boolean).reduce((current, term) => {
        const regex = new RegExp(`(${escapeRegex(escapeHtml(term))})`, 'gi');
        return current.replace(regex, '<mark>$1</mark>');
    }, escaped);
}

function formatMs(seconds) {
    return `${(Number(seconds || 0) * 1000).toFixed(2)} ms`;
}

function formatScore(value) {
    return window.ExplanationPanel
        ? window.ExplanationPanel.formatScore(value)
        : Number(value || 0).toFixed(4);
}

function escapeHtml(value) {
    return window.ExplanationPanel
        ? window.ExplanationPanel.escapeHtml(value)
        : String(value ?? '').replace(/[&<>"']/g, char => ({
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#39;',
        }[char]));
}

function escapeRegex(value) {
    return String(value).replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}
