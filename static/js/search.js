// CNIRS - Search Page JavaScript

// DOM Elements
const queryInput = document.getElementById('query-input');
const modelSelect = document.getElementById('model-select');
const operatorSelect = document.getElementById('operator-select');
const booleanOperatorGroup = document.getElementById('boolean-operator-group');
const topkInput = document.getElementById('topk-input');
const searchBtn = document.getElementById('search-btn');
const loading = document.getElementById('loading');
const statsPanel = document.getElementById('stats-panel');
const resultsHeader = document.getElementById('results-header');
const resultsList = document.getElementById('results-list');
const resultCount = document.getElementById('result-count');
const responseTime = document.getElementById('response-time');
const modelName = document.getElementById('model-name');

// Filter DOM Elements (managed by facet.js, kept for legacy compatibility)
// Note: These may be null if filter elements don't exist in the template

// Export DOM Elements
const exportJsonBtn = document.getElementById('export-json-btn');
const exportCsvBtn = document.getElementById('export-csv-btn');

// Note: Filter state is now managed by facet.js

// Current search state (for export)
let currentSearchResults = null;
let currentSearchQuery = '';
let currentSearchMetadata = {};

// Load system stats on page load
// Note: Filter options are loaded by facet.js
window.addEventListener('DOMContentLoaded', () => {
    loadSystemStats();
});

// Show/hide Boolean operator select
modelSelect.addEventListener('change', () => {
    if (modelSelect.value === 'boolean') {
        booleanOperatorGroup.style.display = 'block';
    } else {
        booleanOperatorGroup.style.display = 'none';
    }
});

// Enter key to search
queryInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        performSearch();
    }
});

// Search button click
searchBtn.addEventListener('click', () => {
    // Call performSearch (may be overridden by facet.js)
    if (typeof window.performSearch === 'function') {
        window.performSearch();
    } else {
        performSearch();
    }
});

// Expose performSearch to window for facet.js override
window.performSearch = performSearch;

/**
 * Load system statistics
 */
async function loadSystemStats() {
    try {
        const response = await fetch('/api/stats');
        const data = await response.json();

        if (data.success) {
            displayStats(data.stats);
        }
    } catch (error) {
        console.error('Failed to load stats:', error);
    }
}

/**
 * Display system statistics
 */
function displayStats(stats) {
    statsPanel.innerHTML = `
        <div class="stat-item">
            <div class="stat-value">${stats.total_documents || 0}</div>
            <div class="stat-label">文檔數量</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">${(stats.total_terms || 0).toLocaleString()}</div>
            <div class="stat-label">詞彙數</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">${stats.models ? stats.models.length : 0}</div>
            <div class="stat-label">檢索模型</div>
        </div>
    `;
}

/**
 * Perform search
 */
async function performSearch() {
    const query = queryInput.value.trim();

    if (!query) {
        alert('請輸入查詢關鍵字');
        return;
    }

    const model = modelSelect.value;
    const topK = parseInt(topkInput.value);
    const operator = operatorSelect.value;

    // Get active filters (with guard for facet.js not loaded)
    const filters = typeof getActiveFilters === 'function' ? getActiveFilters() : null;

    // Show loading
    loading.style.display = 'block';
    resultsHeader.style.display = 'none';
    resultsList.innerHTML = '';
    searchBtn.disabled = true;

    try {
        const requestBody = {
            query,
            model,
            top_k: topK,
            operator
        };

        // Add filters if any
        if (filters) {
            requestBody.filters = filters;
        }

        const response = await fetch('/api/search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(requestBody)
        });

        const data = await response.json();

        if (data.success) {
            displayResults(data);
        } else {
            alert('搜尋錯誤: ' + data.error);
        }
    } catch (error) {
        console.error('Search error:', error);
        alert('搜尋失敗，請稍後再試');
    } finally {
        loading.style.display = 'none';
        searchBtn.disabled = false;
    }
}

/**
 * Display search results
 */
function displayResults(data) {
    // Safety check: ensure data has required properties
    if (!data || !data.results) {
        console.error('displayResults: Invalid data structure', data);
        resultsList.innerHTML = '<div class="no-results">搜尋結果格式錯誤，請重新搜尋</div>';
        return;
    }

    // Store current search results for export
    currentSearchResults = data.results;
    currentSearchQuery = data.query || '';
    currentSearchMetadata = {
        model: data.model,
        total_results: data.total_results || data.filtered_results || data.results.length,
        response_time: data.response_time,
        filters: data.filters || null
    };

    // Show results header
    resultsHeader.style.display = 'block';

    // Update meta info (handle faceted search response structure)
    const totalCount = data.total_results || data.filtered_results || data.results.length;
    resultCount.textContent = `找到 ${totalCount} 筆結果`;
    responseTime.textContent = data.response_time ? `⏱️ ${(data.response_time * 1000).toFixed(2)} ms` : '';
    modelName.textContent = `📊 ${data.model ? data.model.toUpperCase() : 'Unknown'}`;

    // Display results
    if (data.results.length === 0) {
        resultsList.innerHTML = '<div class="no-results">沒有找到相關結果</div>';
        currentSearchResults = null;
        return;
    }

    resultsList.innerHTML = data.results.map(result => {
        // Get snippet/content to display
        let displayText = '';
        if (result.snippet && result.snippet.trim()) {
            displayText = result.snippet;
        } else if (result.metadata && result.metadata.content) {
            displayText = result.metadata.content.substring(0, 200) + '...';
        } else if (result.title && result.title.trim()) {
            displayText = result.title;
        } else {
            displayText = '無內容摘要';
        }

        // Support both doc_id (regular search) and id (faceted search)
        const docId = result.doc_id || result.id;
        const publishedDate = result.pub_date || result.metadata?.published_date || result.metadata?.date || '';
        const category = result.category_name || result.category || result.metadata?.category || '';
        const source = result.source || result.metadata?.source || '';
        const author = result.author || result.metadata?.author || '';

        // Build metadata chips
        let metaChips = `<span class="meta-chip meta-docid">📄 ${docId}</span>`;

        if (source) {
            metaChips += `<span class="meta-chip meta-source">📰 ${source}</span>`;
        }
        if (publishedDate) {
            metaChips += `<span class="meta-chip meta-date">📅 ${publishedDate}</span>`;
        }
        if (category) {
            metaChips += `<span class="meta-chip meta-category">🏷️ ${category}</span>`;
        }
        if (author) {
            metaChips += `<span class="meta-chip meta-author">✍️ ${author}</span>`;
        }

        return `
        <div class="result-item" data-doc-id="${docId}">
            <div class="result-header">
                <div class="result-rank">#${result.rank}</div>
                <div class="result-title">${highlightQuery(result.title, data.query)}</div>
                <div class="result-score">${result.score.toFixed(4)}</div>
            </div>
            <div class="result-snippet">
                ${highlightQuery(displayText, data.query)}
            </div>
            <div class="result-meta" data-doc-id="${docId}">
                ${metaChips}
            </div>
        </div>
        `;
    }).join('');

    // Make results clickable for document details
    makeResultsClickable();

    // Scroll to results
    resultsHeader.scrollIntoView({ behavior: 'smooth', block: 'start' });
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

// ========== Export Functions ==========

/**
 * Export search results to specified format
 */
async function exportResults(format) {
    if (!currentSearchResults || currentSearchResults.length === 0) {
        alert('沒有可匯出的結果');
        return;
    }

    // Disable export buttons during export
    exportJsonBtn.disabled = true;
    exportCsvBtn.disabled = true;

    try {
        const response = await fetch('/api/export', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                results: currentSearchResults,
                format: format,
                query: currentSearchQuery,
                metadata: currentSearchMetadata
            })
        });

        if (!response.ok) {
            throw new Error('Export failed');
        }

        // Get filename from response headers or generate one
        const contentDisposition = response.headers.get('Content-Disposition');
        let filename = `cnirs_results_${Date.now()}.${format}`;
        if (contentDisposition) {
            const filenameMatch = contentDisposition.match(/filename=(.+)/);
            if (filenameMatch) {
                filename = filenameMatch[1];
            }
        }

        // Download the file
        if (format === 'json') {
            const data = await response.json();
            downloadFile(JSON.stringify(data, null, 2), filename, 'application/json');
        } else if (format === 'csv') {
            const data = await response.text();
            downloadFile(data, filename, 'text/csv');
        }

        // Show success message
        showNotification(`成功匯出 ${currentSearchResults.length} 筆結果為 ${format.toUpperCase()} 格式`, 'success');

    } catch (error) {
        console.error('Export error:', error);
        alert('匯出失敗，請稍後再試');
    } finally {
        // Re-enable export buttons
        exportJsonBtn.disabled = false;
        exportCsvBtn.disabled = false;
    }
}

/**
 * Download file to user's computer
 */
function downloadFile(content, filename, mimeType) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
}

/**
 * Show notification message
 */
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: ${type === 'success' ? '#10b981' : '#3b82f6'};
        color: white;
        padding: 15px 20px;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        z-index: 10000;
        animation: slideInRight 0.3s ease;
    `;

    document.body.appendChild(notification);

    // Auto-remove after 3 seconds
    setTimeout(() => {
        notification.style.animation = 'slideOutRight 0.3s ease';
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 300);
    }, 3000);
}

// Add notification animations
const notificationStyle = document.createElement('style');
notificationStyle.textContent = `
    @keyframes slideInRight {
        from {
            transform: translateX(400px);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    @keyframes slideOutRight {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(400px);
            opacity: 0;
        }
    }
`;
document.head.appendChild(notificationStyle);

// Export button event listeners
if (exportJsonBtn) {
    exportJsonBtn.addEventListener('click', () => exportResults('json'));
}

if (exportCsvBtn) {
    exportCsvBtn.addEventListener('click', () => exportResults('csv'));
}
