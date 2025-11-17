// CNIRS - PAT-tree Visualization JavaScript

// Load tree statistics on page load
window.addEventListener('DOMContentLoaded', () => {
    loadTreeStatistics();
});

/**
 * Load and display PAT-tree statistics
 */
async function loadTreeStatistics() {
    try {
        // Make initial request to build tree (this will trigger caching)
        const response = await fetch('/api/pat_tree?max_nodes=1');
        const data = await response.json();

        if (data.success && data.statistics) {
            displayStatistics(data.statistics);
        }
    } catch (error) {
        console.error('Failed to load statistics:', error);
    }
}

/**
 * Display tree statistics
 */
function displayStatistics(stats) {
    const statBoxes = document.querySelectorAll('.stat-box');

    if (statBoxes.length >= 6) {
        statBoxes[0].querySelector('.stat-value').textContent = stats.total_terms.toLocaleString();
        statBoxes[1].querySelector('.stat-value').textContent = stats.unique_terms.toLocaleString();
        statBoxes[2].querySelector('.stat-value').textContent = stats.total_nodes.toLocaleString();
        statBoxes[3].querySelector('.stat-value').textContent = stats.max_depth;
        statBoxes[4].querySelector('.stat-value').textContent = stats.compression_ratio.toFixed(2) + 'x';
        statBoxes[5].querySelector('.stat-value').textContent = stats.avg_term_frequency.toFixed(2);
    }
}

/**
 * Visualize PAT-tree structure
 */
async function visualizeTree() {
    const prefixInput = document.getElementById('prefix-input');
    const maxNodesInput = document.getElementById('max-nodes-input');
    const treeDisplay = document.getElementById('tree-display');
    const processing = document.getElementById('tree-processing');
    const breadcrumb = document.getElementById('tree-breadcrumb');
    const breadcrumbPath = document.getElementById('breadcrumb-path');

    const prefix = prefixInput.value.trim();
    const maxNodes = parseInt(maxNodesInput.value);

    // Show processing indicator
    processing.style.display = 'flex';
    treeDisplay.innerHTML = '<p style="text-align: center; color: var(--text-secondary); padding: 40px;">載入中...</p>';

    try {
        // Build URL with query parameters
        let url = `/api/pat_tree?max_nodes=${maxNodes}`;
        if (prefix) {
            url += `&prefix=${encodeURIComponent(prefix)}`;
        }

        const response = await fetch(url);
        const data = await response.json();

        if (data.success) {
            // Update statistics
            displayStatistics(data.statistics);

            // Update breadcrumb
            breadcrumb.style.display = 'flex';
            breadcrumbPath.textContent = prefix ? `ROOT → ${prefix}` : 'ROOT';

            // Render tree
            if (data.tree && data.tree.children && data.tree.children.length > 0) {
                treeDisplay.innerHTML = renderTree(data.tree);
            } else {
                treeDisplay.innerHTML = '<p style="text-align: center; color: var(--text-secondary); padding: 40px;">未找到符合的詞彙</p>';
            }

            // Display processing time
            const timeInfo = document.createElement('div');
            timeInfo.style.cssText = 'margin-top: 16px; padding: 12px; background: rgba(59, 130, 246, 0.1); border-radius: 8px; text-align: center; font-size: 0.9rem; color: var(--text-secondary);';
            timeInfo.textContent = `⏱️ 處理時間: ${(data.processing_time * 1000).toFixed(2)} ms`;
            treeDisplay.appendChild(timeInfo);
        } else {
            treeDisplay.innerHTML = `<p style="text-align: center; color: #ef4444; padding: 40px;">❌ 錯誤: ${data.error}</p>`;
        }
    } catch (error) {
        console.error('Tree visualization error:', error);
        treeDisplay.innerHTML = '<p style="text-align: center; color: #ef4444; padding: 40px;">❌ 載入失敗，請稍後再試</p>';
    } finally {
        processing.style.display = 'none';
    }
}

/**
 * Render tree structure recursively
 */
function renderTree(node, depth = 0) {
    let html = '';

    // Don't render root label
    if (depth > 0 && node.label) {
        const isTerminal = node.terminal;
        const terminalClass = isTerminal ? 'terminal' : '';

        html += `<div class="tree-node ${terminalClass}">`;
        html += `<span class="node-label ${terminalClass}" title="${node.key || node.label}">`;
        html += `${node.label}`;

        if (isTerminal) {
            html += ' ✓';
        }

        html += '</span>';

        // Add statistics
        if (isTerminal || node.frequency > 0) {
            html += '<span class="node-stats">';

            if (node.frequency > 0) {
                html += `<span class="stat-badge frequency" title="詞頻">F: ${node.frequency}</span>`;
            }

            if (node.doc_count > 0) {
                html += `<span class="stat-badge docs" title="文檔數">D: ${node.doc_count}</span>`;
            }

            html += '</span>';
        }

        // Show full key on click
        if (isTerminal && node.key) {
            html += `<span style="margin-left: 12px; font-size: 0.85rem; color: var(--text-secondary); font-family: 'Courier New', monospace;">→ ${node.key}</span>`;
        }
    }

    // Render children
    if (node.children && node.children.length > 0) {
        for (const child of node.children) {
            html += renderTree(child, depth + 1);
        }
    }

    if (depth > 0) {
        html += '</div>';
    }

    return html;
}

/**
 * Extract keywords from PAT-tree
 */
async function extractKeywords() {
    const methodSelect = document.getElementById('kw-method');
    const topkInput = document.getElementById('kw-topk');
    const minFreqInput = document.getElementById('kw-min-freq');
    const minDocFreqInput = document.getElementById('kw-min-doc-freq');
    const processing = document.getElementById('kw-processing');
    const resultDiv = document.getElementById('keywords-result');
    const summaryDiv = document.getElementById('kw-summary');
    const listDiv = document.getElementById('keywords-list');

    // Get parameters
    const method = methodSelect.value;
    const topK = parseInt(topkInput.value);
    const minFreq = parseInt(minFreqInput.value);
    const minDocFreq = parseInt(minDocFreqInput.value);

    // Show processing indicator
    processing.style.display = 'flex';
    resultDiv.style.display = 'none';

    try {
        const response = await fetch('/api/pat_tree_keywords', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                top_k: topK,
                min_freq: minFreq,
                min_doc_freq: minDocFreq,
                method: method
            })
        });

        const data = await response.json();

        if (data.success) {
            // Display summary
            const methodNames = {
                'tfidf': 'TF-IDF',
                'frequency': '詞頻統計',
                'doc_frequency': '文檔頻率',
                'combined': '綜合評分'
            };

            summaryDiv.innerHTML = `
                使用 <strong>${methodNames[data.method]}</strong> 方法，
                從 <strong>${data.total_candidates.toLocaleString()}</strong> 個候選詞彙中提取
                <strong>${data.keywords.length}</strong> 個關鍵詞
                （處理時間: ${(data.processing_time * 1000).toFixed(2)} ms）
            `;

            // Display keywords
            listDiv.innerHTML = '';

            if (data.keywords.length === 0) {
                listDiv.innerHTML = '<p style="text-align: center; color: var(--text-secondary); padding: 20px;">未找到符合條件的關鍵詞</p>';
            } else {
                data.keywords.forEach(kw => {
                    const item = document.createElement('div');
                    item.className = 'keyword-item';

                    item.innerHTML = `
                        <div class="keyword-rank">#${kw.rank}</div>
                        <div class="keyword-term">${kw.term}</div>
                        <div class="keyword-score">
                            Score: <strong>${kw.score.toFixed(4)}</strong><br>
                            <small style="color: var(--text-secondary);">
                                TF: ${kw.tf.toFixed(4)} × IDF: ${kw.idf.toFixed(2)}
                            </small>
                        </div>
                        <div class="keyword-freq" title="出現次數">
                            ${kw.frequency.toLocaleString()} 次
                        </div>
                        <div class="keyword-docs" title="出現文檔數">
                            ${kw.doc_count.toLocaleString()} 篇
                        </div>
                    `;

                    listDiv.appendChild(item);
                });
            }

            resultDiv.style.display = 'block';

            // Scroll to results
            resultDiv.scrollIntoView({ behavior: 'smooth', block: 'start' });
        } else {
            alert('關鍵詞提取失敗: ' + data.error);
        }
    } catch (error) {
        console.error('Keyword extraction error:', error);
        alert('關鍵詞提取時發生錯誤');
    } finally {
        processing.style.display = 'none';
    }
}

// Allow Enter key to trigger visualization
document.addEventListener('DOMContentLoaded', () => {
    const prefixInput = document.getElementById('prefix-input');
    if (prefixInput) {
        prefixInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                visualizeTree();
            }
        });
    }
});
