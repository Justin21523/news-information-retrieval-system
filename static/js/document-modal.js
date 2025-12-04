// CNIRS - Enhanced Document Details Modal
// Shared module for displaying document details with summary and keyword extraction

// Current document data
let currentDocData = null;

/**
 * Initialize document modal functionality
 * Call this function after DOM is loaded
 */
function initDocumentModal() {
    // Create modal HTML if not exists
    if (!document.getElementById('doc-modal')) {
        const modalHTML = `
            <div id="doc-modal" class="modal">
                <div class="modal-content">
                    <div class="modal-header">
                        <h2 id="modal-title">文檔詳情</h2>
                        <button class="modal-close" id="modal-close">&times;</button>
                    </div>
                    <div class="modal-body" id="modal-body">
                        <div id="modal-loading" style="text-align: center; padding: 40px;">
                            <div class="spinner"></div>
                            <p>載入中...</p>
                        </div>
                        <div id="modal-content-area" style="display: none;"></div>
                    </div>
                </div>
            </div>
        `;
        document.body.insertAdjacentHTML('beforeend', modalHTML);

        // Setup event listeners
        const modal = document.getElementById('doc-modal');
        const closeBtn = document.getElementById('modal-close');

        // Close on button click
        closeBtn.addEventListener('click', closeModal);

        // Close on outside click
        window.addEventListener('click', (e) => {
            if (e.target === modal) {
                closeModal();
            }
        });

        // Close on ESC key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && modal.style.display === 'block') {
                closeModal();
            }
        });
    }
}

/**
 * Open document modal and load details
 */
async function openDocumentModal(docId) {
    const modal = document.getElementById('doc-modal');
    const loading = document.getElementById('modal-loading');
    const contentArea = document.getElementById('modal-content-area');

    // Show modal and loading
    modal.style.display = 'block';
    loading.style.display = 'block';
    contentArea.style.display = 'none';

    try {
        // Fetch document details
        const response = await fetch(`/api/document/${docId}?include_similar=true&top_k=5`);
        const data = await response.json();

        if (data.success) {
            currentDocData = data;
            displayDocumentDetails(data);
        } else {
            contentArea.innerHTML = `<div class="error">載入失敗: ${data.error}</div>`;
            contentArea.style.display = 'block';
        }
    } catch (error) {
        console.error('Error loading document:', error);
        contentArea.innerHTML = '<div class="error">載入文檔時發生錯誤</div>';
        contentArea.style.display = 'block';
    } finally {
        loading.style.display = 'none';
    }
}

/**
 * Display document details with enhanced features
 */
function displayDocumentDetails(data) {
    const contentArea = document.getElementById('modal-content-area');
    // Support both nested (data.document) and flat (data) response structures
    const doc = data.document || data;
    const metadata = doc.metadata || {};

    let html = `
        <div class="doc-details">
            <!-- Document Header -->
            <div class="doc-header">
                <h3>${doc.title || '未命名文檔'}</h3>
                <div class="doc-meta-chips">
                    <span class="meta-chip">📄 ${doc.doc_id}</span>
                    <span class="meta-chip">📅 ${metadata.published_date || metadata.date || '未知日期'}</span>
                    <span class="meta-chip">🏷️ ${metadata.category || '未分類'}</span>
                </div>
            </div>

            <!-- Metadata Section -->
            <div class="doc-metadata">
                <h4>📋 完整 Metadata</h4>
                <div class="metadata-grid">
    `;

    // Display all metadata fields
    for (const [key, value] of Object.entries(metadata)) {
        if (key !== 'content') {  // Skip content here, show separately
            html += `
                <div class="metadata-item">
                    <span class="metadata-key">${key}:</span>
                    <span class="metadata-value">${value}</span>
                </div>
            `;
        }
    }

    html += `
                </div>
            </div>

            <!-- Document Content -->
            <div class="doc-content-section">
                <h4>📖 完整內容</h4>
                <div class="doc-content">
                    ${doc.content || metadata.content || '無內容'}
                </div>
            </div>

            <!-- Summary Section -->
            <div class="doc-summary-section">
                <h4>📝 智能摘要</h4>
                <div class="summary-controls">
                    <select id="summary-method">
                        <option value="key_sentence">關鍵句摘要</option>
                        <option value="lead_k">Lead-K 摘要</option>
                    </select>
                    <input type="number" id="summary-k" value="3" min="1" max="10" style="width: 60px;">
                    <button class="btn-generate" onclick="generateSummary('${doc.doc_id}')">
                        🔄 生成摘要
                    </button>
                </div>
                <div id="summary-result" class="summary-result" style="display: none;">
                    <div class="processing-indicator" id="summary-processing" style="display: none;">
                        <div class="spinner-small"></div>
                        <span>正在生成摘要...</span>
                    </div>
                    <div id="summary-content"></div>
                    <div class="processing-time" id="summary-time"></div>
                </div>
            </div>

            <!-- Keywords Section -->
            <div class="doc-keywords-section">
                <h4>🔑 關鍵詞提取</h4>
                <div class="keywords-controls">
                    <select id="keywords-method">
                        <option value="tfidf">TF-IDF</option>
                        <option value="term_frequency">詞頻統計</option>
                        <option value="textrank">TextRank</option>
                    </select>
                    <input type="number" id="keywords-k" value="10" min="5" max="50" step="5" style="width: 60px;">
                    <button class="btn-generate" onclick="extractKeywords('${doc.doc_id}')">
                        🔍 提取關鍵詞
                    </button>
                </div>
                <div id="keywords-result" class="keywords-result" style="display: none;">
                    <div class="processing-indicator" id="keywords-processing" style="display: none;">
                        <div class="spinner-small"></div>
                        <span>正在提取關鍵詞...</span>
                    </div>
                    <div id="keywords-content"></div>
                    <div class="processing-time" id="keywords-time"></div>
                </div>
            </div>
    `;

    // Similar Documents Section
    if (data.similar_documents && data.similar_documents.length > 0) {
        html += `
            <div class="similar-docs-section">
                <h4>🔗 相似文檔</h4>
                <div class="similar-docs-list">
        `;

        data.similar_documents.forEach(sim => {
            html += `
                <div class="similar-doc-item" onclick="openDocumentModal('${sim.doc_id}')">
                    <div class="similar-doc-title">${sim.title}</div>
                    <div class="similar-doc-meta">
                        <span>📄 ${sim.doc_id}</span>
                        <span>相似度: ${(sim.similarity * 100).toFixed(1)}%</span>
                    </div>
                </div>
            `;
        });

        html += `
                </div>
            </div>
        `;
    }

    html += `</div>`;

    contentArea.innerHTML = html;
    contentArea.style.display = 'block';
}

/**
 * Generate summary for current document
 */
async function generateSummary(docId) {
    const method = document.getElementById('summary-method').value;
    const k = parseInt(document.getElementById('summary-k').value);
    const resultDiv = document.getElementById('summary-result');
    const processingDiv = document.getElementById('summary-processing');
    const contentDiv = document.getElementById('summary-content');
    const timeDiv = document.getElementById('summary-time');

    // Show processing indicator
    resultDiv.style.display = 'block';
    processingDiv.style.display = 'flex';
    contentDiv.innerHTML = '';
    timeDiv.textContent = '';

    try {
        const response = await fetch('/api/summarize', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                doc_id: docId,
                method: method,
                k: k
            })
        });

        const data = await response.json();

        if (data.success) {
            contentDiv.innerHTML = `<div class="summary-text">${data.summary}</div>`;
            timeDiv.textContent = `⏱️ 處理時間: ${(data.processing_time * 1000).toFixed(2)} ms`;
        } else {
            contentDiv.innerHTML = `<div class="error">摘要生成失敗: ${data.error}</div>`;
        }
    } catch (error) {
        console.error('Summary generation error:', error);
        contentDiv.innerHTML = '<div class="error">摘要生成時發生錯誤</div>';
    } finally {
        processingDiv.style.display = 'none';
    }
}

/**
 * Extract keywords for current document
 */
async function extractKeywords(docId) {
    const method = document.getElementById('keywords-method').value;
    const k = parseInt(document.getElementById('keywords-k').value);
    const resultDiv = document.getElementById('keywords-result');
    const processingDiv = document.getElementById('keywords-processing');
    const contentDiv = document.getElementById('keywords-content');
    const timeDiv = document.getElementById('keywords-time');

    // Show processing indicator
    resultDiv.style.display = 'block';
    processingDiv.style.display = 'flex';
    contentDiv.innerHTML = '';
    timeDiv.textContent = '';

    try {
        const response = await fetch('/api/extract_keywords', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                doc_id: docId,
                top_k: k,
                method: method
            })
        });

        const data = await response.json();

        if (data.success) {
            const methodNames = {
                'tfidf': 'TF-IDF',
                'term_frequency': '詞頻統計',
                'textrank': 'TextRank'
            };
            let keywordsHTML = `<div class="method-label">使用算法: ${methodNames[data.method] || data.method}</div>`;
            keywordsHTML += '<div class="keywords-cloud">';
            data.keywords.forEach((kw, idx) => {
                const size = 1 + (1 - idx / data.keywords.length) * 0.5; // Size based on rank
                keywordsHTML += `
                    <span class="keyword-tag" style="font-size: ${size}rem;">
                        ${kw.word}
                        <span class="keyword-score">${kw.score.toFixed(2)}</span>
                    </span>
                `;
            });
            keywordsHTML += '</div>';
            contentDiv.innerHTML = keywordsHTML;
            timeDiv.textContent = `⏱️ 處理時間: ${(data.processing_time * 1000).toFixed(2)} ms`;
        } else {
            contentDiv.innerHTML = `<div class="error">關鍵詞提取失敗: ${data.error}</div>`;
        }
    } catch (error) {
        console.error('Keyword extraction error:', error);
        contentDiv.innerHTML = '<div class="error">關鍵詞提取時發生錯誤</div>';
    } finally {
        processingDiv.style.display = 'none';
    }
}

/**
 * Close document modal
 */
function closeModal() {
    const modal = document.getElementById('doc-modal');
    modal.style.display = 'none';
    currentDocData = null;
}

/**
 * Make search results clickable to open modal
 */
function makeResultsClickable() {
    document.querySelectorAll('.result-item').forEach(item => {
        item.style.cursor = 'pointer';
        item.addEventListener('click', (e) => {
            // Don't trigger if clicking on a button or link
            if (e.target.tagName === 'BUTTON' || e.target.tagName === 'A') {
                return;
            }
            const docId = item.getAttribute('data-doc-id');
            if (docId) {
                openDocumentModal(docId);
            }
        });
    });
}

// Initialize on page load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initDocumentModal);
} else {
    initDocumentModal();
}
