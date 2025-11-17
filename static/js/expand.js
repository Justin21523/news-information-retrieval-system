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
        const response = await fetch('/api/expand_query', {
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

        const data = await response.json();

        if (data.success) {
            displayExpansion(data, maxExpansionTerms);
        } else {
            alert('擴展錯誤: ' + data.error);
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
    // Create header
    const header = document.createElement('div');
    header.className = 'results-header';
    header.innerHTML = `
        <h2>查詢擴展結果</h2>
        <div class="results-meta">
            <span>原始查詢: <strong>${data.original_query}</strong></span>
            <span>相關文檔數: ${data.num_relevant}</span>
        </div>
    `;
    expansionContainer.appendChild(header);

    // Expansion summary card
    const summaryCard = document.createElement('div');
    summaryCard.className = 'expansion-summary';
    summaryCard.innerHTML = `
        <div class="summary-section">
            <h3>📝 擴展查詢</h3>
            <div class="expanded-query">${highlightExpansion(data.expanded_query, data.original_query)}</div>
        </div>

        <div class="summary-section">
            <h3>✨ 新增擴展詞 (Top ${Math.min(maxExpansionTerms, data.expansion_terms.length)})</h3>
            <div class="expansion-terms">
                ${data.expansion_terms.slice(0, maxExpansionTerms).map(term => `
                    <div class="expansion-term">
                        <span class="term-text">${term.term}</span>
                        <span class="term-weight">${term.weight.toFixed(4)}</span>
                    </div>
                `).join('')}
            </div>
        </div>
    `;
    expansionContainer.appendChild(summaryCard);

    // Performance comparison
    const perfCard = document.createElement('div');
    perfCard.className = 'performance-comparison';

    const origTotal = data.original_results.total;
    const expTotal = data.expanded_results.total;
    const improvement = origTotal > 0 ? ((expTotal - origTotal) / origTotal * 100).toFixed(1) : 'N/A';

    perfCard.innerHTML = `
        <h3>📊 效能比較</h3>
        <div class="perf-stats">
            <div class="perf-item">
                <div class="perf-label">原始結果數</div>
                <div class="perf-value">${origTotal}</div>
            </div>
            <div class="perf-item">
                <div class="perf-label">擴展後結果數</div>
                <div class="perf-value perf-highlight">${expTotal}</div>
            </div>
            <div class="perf-item">
                <div class="perf-label">結果提升</div>
                <div class="perf-value ${improvement !== 'N/A' && parseFloat(improvement) > 0 ? 'perf-positive' : ''}">${improvement !== 'N/A' ? improvement + '%' : 'N/A'}</div>
            </div>
        </div>
    `;
    expansionContainer.appendChild(perfCard);

    // Side-by-side results comparison
    const comparisonGrid = document.createElement('div');
    comparisonGrid.className = 'comparison-grid';

    // Original results
    const originalCard = createResultsCard(
        '原始查詢結果',
        data.original_results.results,
        data.original_query,
        'original'
    );

    // Expanded results
    const expandedCard = createResultsCard(
        '擴展查詢結果',
        data.expanded_results.results,
        data.expanded_query,
        'expanded'
    );

    comparisonGrid.appendChild(originalCard);
    comparisonGrid.appendChild(expandedCard);
    expansionContainer.appendChild(comparisonGrid);
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
