// CNIRS analysis graph page.

const graphQuery = document.getElementById('graph-query');
const graphModels = document.getElementById('graph-models');
const graphTopK = document.getElementById('graph-topk');
const graphRunButton = document.getElementById('graph-run-btn');
const graphLoading = document.getElementById('analysis-graph-loading');
const graphSvg = document.getElementById('analysis-graph-svg');
const graphPanel = document.getElementById('analysis-graph-panel');
const graphFlow = document.getElementById('analysis-graph-flow');

graphRunButton?.addEventListener('click', loadAnalysisGraph);
graphQuery?.addEventListener('keypress', (event) => {
    if (event.key === 'Enter') loadAnalysisGraph();
});
document.addEventListener('DOMContentLoaded', initializeGraphFromUrl);

function initializeGraphFromUrl() {
    const params = new URLSearchParams(window.location.search);
    if (params.get('q') || params.get('query')) graphQuery.value = params.get('q') || params.get('query');
    if (params.get('models')) graphModels.value = params.get('models');
    if (params.get('top_k')) graphTopK.value = params.get('top_k');
    loadAnalysisGraph();
}

async function loadAnalysisGraph() {
    const query = graphQuery.value.trim() || '台灣 經濟';
    const models = graphModels.value.trim() || 'bm25,tfidf,hybrid,lm';
    const topK = graphTopK.value || 6;
    graphLoading.style.display = 'block';
    try {
        const params = new URLSearchParams({ query, models, top_k: topK });
        const response = await fetch(`api/analysis/graph?${params.toString()}`);
        const payload = await response.json();
        if (!payload.ok && !payload.success) {
            throw new Error(payload.error?.message || 'Graph failed');
        }
        renderAnalysisGraph(payload.data || payload);
    } catch (error) {
        graphPanel.innerHTML = `<div class="error">${escapeHtml(error.message)}</div>`;
    } finally {
        graphLoading.style.display = 'none';
    }
}

function renderAnalysisGraph(data) {
    renderFlow(data.layers || []);
    const nodes = positionNodes(data.nodes || []);
    const nodeById = Object.fromEntries(nodes.map((node) => [node.id, node]));
    graphSvg.innerHTML = '';

    const edgeLayer = svgEl('g', { class: 'graph-edge-layer' });
    (data.edges || []).forEach((edge) => {
        const source = nodeById[edge.source];
        const target = nodeById[edge.target];
        if (!source || !target) return;
        edgeLayer.appendChild(svgEl('line', {
            x1: source.x,
            y1: source.y,
            x2: target.x,
            y2: target.y,
            class: 'graph-edge'
        }));
        const label = svgEl('text', {
            x: (source.x + target.x) / 2,
            y: (source.y + target.y) / 2 - 6,
            class: 'graph-edge-label'
        });
        label.textContent = edge.label || '';
        edgeLayer.appendChild(label);
    });
    graphSvg.appendChild(edgeLayer);

    const nodeLayer = svgEl('g', { class: 'graph-node-layer' });
    nodes.forEach((node) => {
        const group = svgEl('g', {
            class: `graph-node graph-node-${node.type}`,
            transform: `translate(${node.x}, ${node.y})`,
            tabindex: '0'
        });
        group.appendChild(svgEl('circle', { r: nodeRadius(node), class: 'graph-node-circle' }));
        const label = svgEl('text', { class: 'graph-node-label', y: nodeRadius(node) + 18 });
        label.textContent = truncate(node.label, node.type === 'document' ? 18 : 16);
        group.appendChild(label);
        group.addEventListener('mouseenter', () => renderNodePanel(node, false));
        group.addEventListener('focus', () => renderNodePanel(node, false));
        group.addEventListener('click', () => {
            renderNodePanel(node, true);
            if (node.type === 'document' && typeof openDocumentModal === 'function') {
                openDocumentModal(node.doc_id);
            }
        });
        nodeLayer.appendChild(group);
    });
    graphSvg.appendChild(nodeLayer);

    renderNodePanel(nodes[0] || {}, true);
}

function renderFlow(layers) {
    const labels = {
        query: '查詢',
        processing: '文字處理',
        index: '索引',
        ranking: '排序',
        model: '模型',
        document: '結果文件',
        metadata: 'Metadata',
        feedback: '回饋'
    };
    graphFlow.innerHTML = layers.map((layer, index) => `
        <span class="analysis-flow-step">${escapeHtml(labels[layer] || layer)}</span>
        ${index < layers.length - 1 ? '<span class="analysis-flow-arrow">→</span>' : ''}
    `).join('');
}

function positionNodes(nodes) {
    const lanes = {
        query: 90,
        processing: 220,
        index: 350,
        ranking: 500,
        model: 650,
        document: 860,
        metadata: 1040,
        feedback: 1120
    };
    const groups = {};
    nodes.forEach((node) => {
        const layer = node.layer || 'document';
        groups[layer] = groups[layer] || [];
        groups[layer].push(node);
    });
    return nodes.map((node) => {
        const group = groups[node.layer] || [node];
        const index = group.indexOf(node);
        const step = Math.min(92, 600 / Math.max(group.length, 1));
        return {
            ...node,
            x: lanes[node.layer] || 900,
            y: 80 + index * step + (node.layer === 'metadata' ? 20 : 0)
        };
    });
}

function renderNodePanel(node, pinned) {
    const metrics = node.metrics || {};
    graphPanel.innerHTML = `
        <div class="panel-pin">${pinned ? '已固定節點' : '節點預覽'}</div>
        <h2>${escapeHtml(node.label || '節點')}</h2>
        <div class="explain-chip-row">
            <span class="explain-chip"><strong>type</strong>${escapeHtml(node.type || '-')}</span>
            <span class="explain-chip"><strong>layer</strong>${escapeHtml(node.layer || '-')}</span>
            ${node.doc_id ? `<span class="explain-chip"><strong>doc_id</strong>${escapeHtml(node.doc_id)}</span>` : ''}
        </div>
        <p>${escapeHtml(node.preview || '')}</p>
        ${Object.keys(metrics).length ? `
            <table class="diagnostic-table">
                <tbody>
                    ${Object.entries(metrics).map(([key, value]) => `
                        <tr><th>${escapeHtml(key)}</th><td>${escapeHtml(formatMetric(value))}</td></tr>
                    `).join('')}
                </tbody>
            </table>
        ` : ''}
        ${node.explanation ? renderExplanation(node.explanation) : ''}
    `;
}

function renderExplanation(explanation) {
    const components = explanation.component_scores || {};
    return `
        <h3>Ranking Explanation</h3>
        <div class="explain-chip-row">
            ${(explanation.matched_terms || []).map((term) => `<span class="explain-chip">${escapeHtml(term)}</span>`).join('') || '<span class="explain-muted">無 matched terms</span>'}
        </div>
        ${Object.keys(components).length ? `
            <table class="diagnostic-table">
                <tbody>
                    ${Object.entries(components).map(([key, value]) => `
                        <tr><th>${escapeHtml(key)}</th><td>${escapeHtml(formatMetric(value))}</td></tr>
                    `).join('')}
                </tbody>
            </table>
        ` : ''}
    `;
}

function nodeRadius(node) {
    return node.type === 'document' ? 24 : node.type === 'model' ? 30 : 26;
}

function svgEl(tag, attrs) {
    const element = document.createElementNS('http://www.w3.org/2000/svg', tag);
    Object.entries(attrs || {}).forEach(([key, value]) => element.setAttribute(key, value));
    return element;
}

function truncate(value, length) {
    const text = String(value || '');
    return text.length > length ? `${text.slice(0, length)}...` : text;
}

function formatMetric(value) {
    if (typeof value === 'number') return Number.isFinite(value) ? value.toFixed(4) : '0';
    if (typeof value === 'boolean') return value ? 'true' : 'false';
    if (value && typeof value === 'object') return JSON.stringify(value).slice(0, 120);
    return value ?? '-';
}

function escapeHtml(value) {
    return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}
