// CNIRS - Enhanced Document Details Modal
// Shared module for displaying document details with summary and keyword extraction


function normalizeApiPayload(data) {
    if (!data) return data;
    if (data.ok === true && data.data) {
        return { ...data.data, success: true, meta: data.meta || {}, raw: data };
    }
    return data;
}

function apiErrorMessage(data, fallback = 'Unknown error') {
    if (!data) return fallback;
    if (typeof data.error === 'string') return data.error;
    if (data.error && data.error.message) return data.error.message;
    return data.message || fallback;
}

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
        const query = typeof currentSearchQuery !== 'undefined' ? currentSearchQuery : '';
        const params = new URLSearchParams({
            include_related: 'true',
            include_kwic: query ? 'true' : 'false',
            top_k: '5'
        });
        if (query) {
            params.set('query', query);
        }
        const response = await fetch(`api/document/${encodeURIComponent(docId)}?${params.toString()}`);
        const data = normalizeApiPayload(await response.json());

        if (data.success) {
            currentDocData = data;
            displayDocumentDetails(data);
        } else {
            contentArea.innerHTML = `<div class="error">載入失敗: ${apiErrorMessage(data)}</div>`;
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
    const doc = data.document || data;
    const metadata = doc.metadata || {};
    const summary = data.summary || {};
    const keywords = data.keywords || {};
    const kwic = data.kwic || {};
    const relatedDocuments = data.related_documents || data.similar_documents || [];
    const taxonomy = data.taxonomy || {};
    const explanation = data.explanation || {};
    const keywordItems = keywords.items || keywords.keywords || [];

    let html = `
        <div class="doc-details">
            <div class="doc-header">
                <h3>${escapeHtml(doc.title || '未命名文檔')}</h3>
                <div class="doc-meta-chips">
                    <span class="meta-chip">📄 ${escapeHtml(doc.doc_id)}</span>
                    <span class="meta-chip">📅 ${escapeHtml(metadata.published_date || metadata.date || '未知日期')}</span>
                    <span class="meta-chip">🏷️ ${escapeHtml(taxonomy.label || metadata.category_name || metadata.category || '未分類')}</span>
                    ${metadata.source_label || metadata.source ? `<span class="meta-chip">📰 ${escapeHtml(metadata.source_label || metadata.source)}</span>` : ''}
                </div>
            </div>

            <div class="doc-insight-grid">
                <section class="doc-summary-section">
                    <h4>摘要 Summary</h4>
                    ${renderSummarySection(summary)}
                </section>

                <section class="doc-keywords-section">
                    <h4>關鍵詞 Keywords</h4>
                    ${renderKeywordsSection(keywordItems)}
                </section>
            </div>

            <section class="doc-kwic-section">
                <h4>KWIC</h4>
                ${renderKwicSection(kwic, doc.doc_id)}
            </section>

            <section class="doc-explanation-section">
                <h4>Why this document?</h4>
                ${renderDocumentExplanation(explanation, taxonomy)}
            </section>

            <div class="doc-metadata">
                <h4>Metadata / Facets</h4>
                <div class="metadata-grid">
    `;

    for (const [key, value] of Object.entries(metadata)) {
        if (key !== 'content') {
            html += `
                <div class="metadata-item">
                    <span class="metadata-key">${escapeHtml(key)}:</span>
                    <span class="metadata-value">${escapeHtml(formatMetadataValue(value))}</span>
                </div>
            `;
        }
    }

    html += `
                </div>
            </div>

            <div class="doc-content-section">
                <h4>完整內容 Full Content</h4>
                <div class="doc-content">
                    ${escapeHtml(doc.content || metadata.content || '無內容')}
                </div>
            </div>

            <details class="doc-tools-section">
                <summary>Generate new analysis</summary>
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
                <div class="kwic-controls">
                    <input type="text" id="kwic-query-input" placeholder="輸入 KWIC query">
                    <button class="btn-generate" onclick="generateKwicFromInput('${doc.doc_id}')">
                        生成 KWIC
                    </button>
                </div>
                <div id="kwic-generated-result"></div>
            </details>
    `;

    if (relatedDocuments.length > 0) {
        html += `
            <div class="similar-docs-section">
                <h4>Related News</h4>
                <div class="similar-docs-list">
        `;

        relatedDocuments.forEach(sim => {
            const reason = sim.relation_reason || {};
            html += `
                <div class="similar-doc-item" onclick="openDocumentModal('${escapeAttribute(sim.doc_id)}')">
                    <div class="similar-doc-title">${escapeHtml(sim.title)}</div>
                    <div class="similar-doc-meta">
                        <span>📄 ${escapeHtml(sim.doc_id)}</span>
                        <span>Score: ${Number(sim.score || 0).toFixed(4)}</span>
                        <span>Similarity: ${(Number(sim.similarity || 0) * 100).toFixed(1)}%</span>
                        ${reason.same_category ? '<span>same category</span>' : ''}
                        ${reason.same_taxonomy_topic ? '<span>same taxonomy</span>' : ''}
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

function renderSummarySection(summary) {
    if (!summary || !summary.available) {
        return '<div class="empty-state">No summary available.</div>';
    }
    const sentences = summary.sentences || [];
    const body = sentences.length
        ? `<ol class="summary-list">${sentences.map(sentence => `<li>${escapeHtml(sentence)}</li>`).join('')}</ol>`
        : `<div class="summary-text">${escapeHtml(summary.text || '')}</div>`;
    return `
        <div class="method-label">${escapeHtml(summary.method || 'summary')}</div>
        ${body}
    `;
}

function renderKeywordsSection(items) {
    if (!items || items.length === 0) {
        return '<div class="empty-state">No keywords available.</div>';
    }
    return `
        <div class="keywords-cloud">
            ${items.slice(0, 16).map((item, idx) => {
                const size = 1 + (1 - idx / Math.max(items.length, 1)) * 0.35;
                const term = item.term || item.word || '';
                return `
                    <span class="keyword-tag" style="font-size: ${size.toFixed(2)}rem;">
                        ${escapeHtml(term)}
                        <span class="keyword-score">${Number(item.score || 0).toFixed(2)}</span>
                    </span>
                `;
            }).join('')}
        </div>
    `;
}

function renderKwicSection(kwic, docId) {
    if (!kwic || !kwic.available) {
        return `
            <div class="empty-state">No KWIC query is active.</div>
            <div class="kwic-inline-controls">
                <input type="text" id="kwic-inline-query" placeholder="輸入 KWIC query">
                <button class="btn-generate" onclick="generateKwicFromInlineInput('${escapeAttribute(docId)}')">生成 KWIC</button>
            </div>
            <div id="kwic-inline-result"></div>
        `;
    }
    if (!kwic.matches || kwic.matches.length === 0) {
        return `<div class="empty-state">No KWIC matches for "${escapeHtml(kwic.query || '')}".</div>`;
    }
    return `
        <div class="method-label">Query: ${escapeHtml(kwic.query || '')} | ${kwic.match_count || 0} matches</div>
        <div class="kwic-list">
            ${kwic.matches.map(match => `
                <div class="kwic-item">${match.highlighted_snippet || escapeHtml(match.plain_snippet || '')}</div>
            `).join('')}
        </div>
    `;
}

function renderDocumentExplanation(explanation, taxonomy) {
    const signals = explanation.signals || {};
    const sections = explanation.sections || {};
    return `
        <div class="explanation-grid">
            <span class="meta-chip">summary: ${sections.summary ? 'available' : 'empty'}</span>
            <span class="meta-chip">keywords: ${sections.keywords ? signals.keyword_count || 0 : 0}</span>
            <span class="meta-chip">KWIC: ${signals.kwic_match_count || 0}</span>
            <span class="meta-chip">related: ${signals.related_count || 0}</span>
            ${taxonomy.topic ? `<span class="meta-chip">topic: ${escapeHtml(taxonomy.topic)}</span>` : ''}
            ${taxonomy.source ? `<span class="meta-chip">source: ${escapeHtml(taxonomy.source)}</span>` : ''}
        </div>
    `;
}

function formatMetadataValue(value) {
    if (Array.isArray(value)) return value.join(', ');
    if (value === null || value === undefined) return '';
    return String(value);
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
        const response = await fetch('api/summarize', {
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

        const data = normalizeApiPayload(await response.json());

        if (data.success) {
            contentDiv.innerHTML = `<div class="summary-text">${data.summary}</div>`;
            timeDiv.textContent = `⏱️ 處理時間: ${(data.processing_time * 1000).toFixed(2)} ms`;
        } else {
            contentDiv.innerHTML = `<div class="error">摘要生成失敗: ${apiErrorMessage(data)}</div>`;
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
        const response = await fetch('api/extract_keywords', {
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

        const data = normalizeApiPayload(await response.json());

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
            contentDiv.innerHTML = `<div class="error">關鍵詞提取失敗: ${apiErrorMessage(data)}</div>`;
        }
    } catch (error) {
        console.error('Keyword extraction error:', error);
        contentDiv.innerHTML = '<div class="error">關鍵詞提取時發生錯誤</div>';
    } finally {
        processingDiv.style.display = 'none';
    }
}

async function generateKwicFromInlineInput(docId) {
    const input = document.getElementById('kwic-inline-query');
    const resultDiv = document.getElementById('kwic-inline-result');
    await loadKwic(docId, input?.value || '', resultDiv);
}

async function generateKwicFromInput(docId) {
    const input = document.getElementById('kwic-query-input');
    const resultDiv = document.getElementById('kwic-generated-result');
    await loadKwic(docId, input?.value || '', resultDiv);
}

async function loadKwic(docId, query, resultDiv) {
    if (!resultDiv) return;
    if (!query.trim()) {
        resultDiv.innerHTML = '<div class="empty-state">請先輸入 KWIC query</div>';
        return;
    }
    resultDiv.innerHTML = '<div class="processing-indicator"><div class="spinner-small"></div><span>正在生成 KWIC...</span></div>';
    try {
        const params = new URLSearchParams({
            include_related: 'false',
            include_kwic: 'true',
            query: query.trim()
        });
        const response = await fetch(`api/document/${encodeURIComponent(docId)}?${params.toString()}`);
        const data = normalizeApiPayload(await response.json());
        if (data.success) {
            resultDiv.innerHTML = renderKwicSection(data.kwic || {}, docId);
        } else {
            resultDiv.innerHTML = `<div class="error">KWIC 生成失敗: ${apiErrorMessage(data)}</div>`;
        }
    } catch (error) {
        console.error('KWIC generation error:', error);
        resultDiv.innerHTML = '<div class="error">KWIC 生成時發生錯誤</div>';
    }
}

function escapeHtml(value) {
    return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function escapeAttribute(value) {
    return escapeHtml(value);
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
            if (e.target.tagName === 'BUTTON' || e.target.tagName === 'A' ||
                e.target.tagName === 'SUMMARY' || e.target.closest('.result-explanation')) {
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
