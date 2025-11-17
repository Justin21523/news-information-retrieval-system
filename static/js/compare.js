// CNIRS - Model Comparison Page JavaScript

// DOM Elements
const queryInput = document.getElementById('query-input');
const modelCheckboxes = document.querySelectorAll('input[name="models"]');
const topkInput = document.getElementById('topk-input');
const compareBtn = document.getElementById('compare-btn');
const loading = document.getElementById('loading');
const comparisonContainer = document.getElementById('comparison-container');

// Enter key to search
queryInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        performComparison();
    }
});

// Compare button click
compareBtn.addEventListener('click', performComparison);

/**
 * Perform model comparison
 */
async function performComparison() {
    const query = queryInput.value.trim();

    if (!query) {
        alert('請輸入查詢關鍵字');
        return;
    }

    // Get selected models
    const selectedModels = Array.from(modelCheckboxes)
        .filter(cb => cb.checked)
        .map(cb => cb.value);

    if (selectedModels.length === 0) {
        alert('請至少選擇一個模型');
        return;
    }

    const topK = parseInt(topkInput.value);

    // Show loading
    loading.style.display = 'block';
    comparisonContainer.innerHTML = '';
    compareBtn.disabled = true;

    try {
        const response = await fetch('/api/compare', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query,
                models: selectedModels,
                top_k: topK
            })
        });

        const data = await response.json();

        if (data.success) {
            displayComparison(data);
        } else {
            alert('對比錯誤: ' + data.error);
        }
    } catch (error) {
        console.error('Comparison error:', error);
        alert('對比失敗，請稍後再試');
    } finally {
        loading.style.display = 'none';
        compareBtn.disabled = false;
    }
}

/**
 * Display comparison results
 */
function displayComparison(data) {
    const models = Object.keys(data.models);

    // Create header
    const header = document.createElement('div');
    header.className = 'results-header';
    header.innerHTML = `
        <h2>模型對比結果</h2>
        <div class="results-meta">
            <span>查詢: <strong>${data.query}</strong></span>
            <span>對比模型: ${models.length} 個</span>
        </div>
    `;
    comparisonContainer.appendChild(header);

    // Create comparison grid
    const grid = document.createElement('div');
    grid.className = 'comparison-grid';

    models.forEach(modelName => {
        const modelData = data.models[modelName];
        const modelCard = createModelCard(modelName, modelData, data.query);
        grid.appendChild(modelCard);
    });

    comparisonContainer.appendChild(grid);

    // Create performance comparison table
    const perfTable = createPerformanceTable(models, data.models);
    comparisonContainer.appendChild(perfTable);
}

/**
 * Create model result card
 */
function createModelCard(modelName, modelData, query) {
    const card = document.createElement('div');
    card.className = 'model-results';

    // Header with model name
    const header = document.createElement('div');
    header.className = 'model-header';
    header.textContent = modelName.toUpperCase();

    // Stats row
    const stats = document.createElement('div');
    stats.className = 'model-stats';
    stats.innerHTML = `
        <span>${modelData.total_results} 筆結果</span>
        <span>⏱️ ${(modelData.response_time * 1000).toFixed(2)} ms</span>
    `;

    // Results list
    const resultsList = document.createElement('div');
    resultsList.className = 'model-results-list';

    if (modelData.results.length === 0) {
        resultsList.innerHTML = '<div class="text-center">無結果</div>';
    } else {
        modelData.results.forEach(result => {
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
 * Create performance comparison table
 */
function createPerformanceTable(models, modelsData) {
    const section = document.createElement('div');
    section.className = 'about-section mt-20';
    section.innerHTML = `
        <h2>效能比較</h2>
        <table style="width: 100%; border-collapse: collapse;">
            <thead>
                <tr style="background: #f3f4f6; border-bottom: 2px solid #e5e7eb;">
                    <th style="padding: 12px; text-align: left;">模型</th>
                    <th style="padding: 12px; text-align: right;">結果數</th>
                    <th style="padding: 12px; text-align: right;">響應時間 (ms)</th>
                    <th style="padding: 12px; text-align: right;">平均分數</th>
                </tr>
            </thead>
            <tbody>
                ${models.map(modelName => {
                    const modelData = modelsData[modelName];
                    const avgScore = modelData.results.length > 0
                        ? (modelData.results.reduce((sum, r) => sum + r.score, 0) / modelData.results.length).toFixed(4)
                        : '0.0000';

                    return `
                        <tr style="border-bottom: 1px solid #e5e7eb;">
                            <td style="padding: 12px; font-weight: 600;">${modelName.toUpperCase()}</td>
                            <td style="padding: 12px; text-align: right;">${modelData.total_results}</td>
                            <td style="padding: 12px; text-align: right;">${(modelData.response_time * 1000).toFixed(2)}</td>
                            <td style="padding: 12px; text-align: right;">${avgScore}</td>
                        </tr>
                    `;
                }).join('')}
            </tbody>
        </table>
    `;

    return section;
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

// Add mark styling
const style = document.createElement('style');
style.textContent = `
    mark {
        background-color: #fef08a;
        padding: 2px 4px;
        border-radius: 2px;
        font-weight: 600;
    }
`;
document.head.appendChild(style);
