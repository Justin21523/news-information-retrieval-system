// CNIRS - Query Expansion Page JavaScript

// DOM Elements
const queryInput = document.getElementById('query-input');
const modelSelect = document.getElementById('model-select');
const topkInput = document.getElementById('topk-input');
const expansionTermsInput = document.getElementById('expansion-terms');
const expandBtn = document.getElementById('expand-btn');
const loading = document.getElementById('loading');
const expansionContainer = document.getElementById('expansion-container');

// Enter key to expand
queryInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        performExpansion();
    }
});

// Expand button click
expandBtn.addEventListener('click', performExpansion);

/**
 * Perform query expansion
 */
async function performExpansion() {
    const query = queryInput.value.trim();

    if (!query) {
        alert('請輸入查詢關鍵字');
        return;
    }

    const model = modelSelect.value;
    const topK = parseInt(topkInput.value);
    const maxExpansionTerms = parseInt(expansionTermsInput.value);

    // Show loading
    loading.style.display = 'block';
    expansionContainer.innerHTML = '';
    expandBtn.disabled = true;

    try {
        const response = await fetch('api/expand_query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query,
                model,
                top_k: topK,
                use_top_results: true
            })
        });

        const payload = await response.json();
        const data = normalizeApiPayload(payload);

        if (data.success) {
            displayExpansion(data, maxExpansionTerms);
        } else {
            alert('擴展錯誤: ' + apiErrorMessage(data));
        }
    } catch (error) {
        console.error('Expansion error:', error);
        alert('查詢擴展失敗，請稍後再試');
    } finally {
        loading.style.display = 'none';
        expandBtn.disabled = false;
    }
}

/**
 * Display expansion results
 */
function displayExpansion(data, maxExpansionTerms) {
    const terms = normalizeExpansionTerms(data).slice(0, maxExpansionTerms);
    // Create header
    const header = document.createElement('div');
    header.className = 'results-header';
    header.innerHTML = `
        <h2>查詢擴展結果</h2>
        <div class="results-meta">
            <span>原始查詢: <strong>${escapeHtml(data.original_query)}</strong></span>
            <span>方法: ${escapeHtml(data.method || 'rocchio_prf')}</span>
            <span>Query drift: ${formatScore(data.query_drift)}</span>
        </div>
    `;
    expansionContainer.appendChild(header);

    // Expansion summary card
    const summaryCard = document.createElement('div');
    summaryCard.className = 'expansion-summary';
    summaryCard.innerHTML = `
        <div class="summary-section">
            <h3>📝 擴展查詢</h3>
            <div class="expanded-query">${highlightExpansion(data.expanded_query || data.original_query, data.original_query)}</div>
        </div>

        <div class="summary-section">
            <h3>✨ 新增擴展詞 (Top ${terms.length})</h3>
            <div class="expansion-terms">
                ${terms.map(term => `
                    <div class="expansion-term">
                        <span class="term-text">${escapeHtml(term.term)}</span>
                        <span class="term-weight">${formatScore(term.weight)}</span>
                    </div>
                `).join('') || '<p class="explain-muted">目前沒有新增擴展詞。</p>'}
            </div>
        </div>
    `;
    expansionContainer.appendChild(summaryCard);

    const actionCard = document.createElement('div');
    actionCard.className = 'performance-comparison';
    actionCard.innerHTML = `
        <h3>下一步查詢</h3>
        <p class="explain-muted">使用擴展後 query 回到主搜尋頁，比較原始 query 與 Rocchio pseudo-relevance feedback 的結果差異。</p>
        <a class="btn btn-secondary" href="./?q=${encodeURIComponent(data.expanded_query || data.original_query)}&model=bm25&run=1">用擴展查詢搜尋</a>
    `;
    expansionContainer.appendChild(actionCard);
}

function normalizeApiPayload(payload) {
    if (!payload) return payload;
    if (payload.ok === true && payload.data) {
        return { ...payload.data, success: true, meta: payload.meta || {}, raw: payload };
    }
    return payload;
}

function apiErrorMessage(payload, fallback = '未知錯誤') {
    if (!payload) return fallback;
    if (typeof payload.error === 'string') return payload.error;
    if (payload.error && payload.error.message) return payload.error.message;
    return payload.message || fallback;
}

function normalizeExpansionTerms(data) {
    const weights = data.term_weights || {};
    return (data.expanded_terms || data.expansion_terms || []).map((term) => {
        if (typeof term === 'string') return { term, weight: Number(weights[term] || 0) };
        return { term: term.term || term.word || '', weight: Number(term.weight || weights[term.term] || 0) };
    }).filter((item) => item.term);
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

/**
 * Create results card for comparison
 */
function createResultsCard(title, results, query, type) {
    const card = document.createElement('div');
    card.className = 'model-results';

    const header = document.createElement('div');
    header.className = 'model-header';
    header.textContent = title;

    const stats = document.createElement('div');
    stats.className = 'model-stats';
    stats.innerHTML = `<span>📄 ${results.length} 筆結果</span>`;

    const resultsList = document.createElement('div');
    resultsList.className = 'model-results-list';

    if (results.length === 0) {
        resultsList.innerHTML = '<div class="text-center">無結果</div>';
    } else {
        results.forEach(result => {
            const item = document.createElement('div');
            item.className = 'model-result-item';
            item.setAttribute('data-doc-id', result.doc_id);
            item.innerHTML = `
                <div class="model-result-title" data-doc-id="${result.doc_id}">
                    ${result.rank}. ${highlightQuery(result.title, query)}
                </div>
                <div class="model-result-score">
                    Score: ${result.score.toFixed(4)}
                </div>
                <div class="result-doc-id">
                    📄 ${result.doc_id}
                </div>
            `;
            resultsList.appendChild(item);
        });

    // Make results clickable
    makeResultsClickable();
    }

    card.appendChild(header);
    card.appendChild(stats);
    card.appendChild(resultsList);

    return card;
}

/**
 * Highlight expansion terms in expanded query
 */
function highlightExpansion(expandedQuery, originalQuery) {
    const originalTerms = originalQuery.split(/\s+/);
    const expandedTerms = expandedQuery.split(/\s+/);

    return expandedTerms.map(term => {
        const isOriginal = originalTerms.some(orig =>
            orig.toLowerCase() === term.toLowerCase()
        );

        if (isOriginal) {
            return `<span class="original-term">${term}</span>`;
        } else {
            return `<span class="new-term">${term}</span>`;
        }
    }).join(' ');
}

/**
 * Highlight query terms in text
 */
function highlightQuery(text, query) {
    if (!query) return text;

    const terms = query.split(/\s+/);
    let highlighted = text;

    terms.forEach(term => {
        const regex = new RegExp(`(${escapeRegex(term)})`, 'gi');
        highlighted = highlighted.replace(regex, '<mark>$1</mark>');
    });

    return highlighted;
}

/**
 * Escape regex special characters
 */
function escapeRegex(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

// Add custom styling for expansion page
const style = document.createElement('style');
style.textContent = `
    mark {
        background-color: #fef08a;
        padding: 2px 4px;
        border-radius: 2px;
        font-weight: 600;
    }

    .param-info {
        display: flex;
        gap: 20px;
        margin: 15px 0;
        flex-wrap: wrap;
    }

    .param-item {
        display: flex;
        flex-direction: column;
        gap: 5px;
    }

    .param-label {
        font-size: 0.9rem;
        color: #6b7280;
    }

    .param-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2563eb;
    }

    .param-description {
        margin-top: 15px;
        padding: 15px;
        background: #f0f9ff;
        border-left: 4px solid #2563eb;
        border-radius: 4px;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    .expansion-summary {
        background: white;
        border-radius: 12px;
        padding: 30px;
        margin: 20px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }

    .summary-section {
        margin-bottom: 30px;
    }

    .summary-section:last-child {
        margin-bottom: 0;
    }

    .summary-section h3 {
        margin-bottom: 15px;
        color: #1e40af;
    }

    .expanded-query {
        font-size: 1.2rem;
        padding: 20px;
        background: #f9fafb;
        border-radius: 8px;
        line-height: 1.8;
        word-spacing: 5px;
    }

    .original-term {
        color: #059669;
        font-weight: 600;
        padding: 2px 6px;
        background: #d1fae5;
        border-radius: 4px;
    }

    .new-term {
        color: #dc2626;
        font-weight: 600;
        padding: 2px 6px;
        background: #fee2e2;
        border-radius: 4px;
    }

    .expansion-terms {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
    }

    .expansion-term {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 10px 16px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 20px;
        font-size: 0.95rem;
    }

    .term-text {
        font-weight: 600;
    }

    .term-weight {
        padding: 2px 8px;
        background: rgba(255,255,255,0.3);
        border-radius: 10px;
        font-size: 0.85rem;
        font-family: monospace;
    }

    .performance-comparison {
        background: white;
        border-radius: 12px;
        padding: 30px;
        margin: 20px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }

    .performance-comparison h3 {
        margin-bottom: 20px;
        color: #1e40af;
    }

    .perf-stats {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 20px;
    }

    .perf-item {
        text-align: center;
        padding: 20px;
        background: #f9fafb;
        border-radius: 8px;
    }

    .perf-label {
        font-size: 0.9rem;
        color: #6b7280;
        margin-bottom: 10px;
    }

    .perf-value {
        font-size: 2rem;
        font-weight: 700;
        color: #111827;
    }

    .perf-highlight {
        color: #2563eb;
    }

    .perf-positive {
        color: #059669;
    }

    .result-doc-id {
        margin-top: 8px;
        font-size: 0.85rem;
        color: #6b7280;
    }
`;
document.head.appendChild(style);
