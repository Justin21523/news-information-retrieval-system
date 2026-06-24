// Corpus Dashboard JavaScript

const loading = document.getElementById('corpus-loading');
const errorBox = document.getElementById('corpus-error');
const content = document.getElementById('corpus-content');

window.addEventListener('DOMContentLoaded', loadCorpusAudit);
window.addEventListener('DOMContentLoaded', () => {
    const button = document.getElementById('run-topic-explorer');
    if (button) button.addEventListener('click', runTopicExplorer);
    initializeCorpusFromUrl();
});

function initializeCorpusFromUrl() {
    const params = new URLSearchParams(window.location.search);
    const topicQuery = params.get('topic_query') || params.get('q');
    const runTopic = params.get('run_topic') === '1' || params.get('run_topic') === 'true';
    if (topicQuery) {
        document.getElementById('topic-query').value = topicQuery;
    }
    if (params.get('topic_clusters')) {
        document.getElementById('topic-clusters').value = params.get('topic_clusters');
    }
    if (params.get('topic_sample_size')) {
        document.getElementById('topic-sample-size').value = params.get('topic_sample_size');
    }
    if (runTopic) {
        window.setTimeout(runTopicExplorer, 900);
    }
}

async function loadCorpusAudit() {
    try {
        const response = await fetch('api/corpus/audit');
        const payload = await response.json();
        if (!payload.ok) {
            throw new Error(payload.error?.message || 'Corpus audit failed');
        }
        renderCorpusAudit(payload.data, payload.meta || {});
        content.style.display = 'block';
    } catch (error) {
        errorBox.textContent = `Failed to load corpus audit: ${error.message}`;
        errorBox.style.display = 'block';
    } finally {
        loading.style.display = 'none';
    }
}

function renderCorpusAudit(data, meta) {
    renderSummary(data.summary, data.index, meta);
    renderReadiness(data.readiness);
    renderBarList('source-distribution', data.distributions.source);
    renderBarList('taxonomy-distribution', data.distributions.taxonomy_topic);
    renderBarList('year-distribution', data.distributions.published_year);
    renderIndexCache(data.index);
    renderCompleteness(data.metadata_completeness.fields);
    renderFacetQuality(data.facet_quality);
    renderFlow(data.flow);
    renderExternalSources(data.external_sources);
}

function renderSummary(summary, index, meta) {
    const sizeMb = summary.dataset_size ? summary.dataset_size / 1024 / 1024 : 0;
    document.getElementById('corpus-summary').innerHTML = `
        <div class="stat-item">
            <div class="stat-value">${formatNumber(summary.total_documents)}</div>
            <div class="stat-label">Searchable documents</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">${formatNumber(index.vocabulary_size || 0)}</div>
            <div class="stat-label">Vocabulary</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">${sizeMb.toFixed(1)} MB</div>
            <div class="stat-label">Dataset size</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">${index.cache_used ? 'Hit' : 'Built'}</div>
            <div class="stat-label">Index cache</div>
        </div>
        <div class="stat-item">
            <div class="stat-value">${((meta.execution_time || 0) * 1000).toFixed(1)} ms</div>
            <div class="stat-label">Audit time</div>
        </div>
    `;
}

function renderReadiness(readiness) {
    const status = document.getElementById('readiness-status');
    status.textContent = readiness.overall_status.replaceAll('_', ' ');
    status.className = `status-pill ${readiness.overall_status}`;

    document.getElementById('readiness-checks').innerHTML = readiness.checks.map(check => `
        <div class="readiness-item ${check.passed ? 'passed' : 'warning'}">
            <span class="readiness-mark">${check.passed ? 'OK' : 'GAP'}</span>
            <div>
                <strong>${escapeHtml(check.label)}</strong>
                <p>${escapeHtml(check.detail)}</p>
            </div>
        </div>
    `).join('') + `
        <div class="known-gaps">
            ${readiness.known_gaps.map(gap => `<span>${escapeHtml(gap)}</span>`).join('')}
        </div>
    `;
}

function renderBarList(elementId, rows) {
    const max = Math.max(...(rows || []).map(row => row.count), 1);
    document.getElementById(elementId).innerHTML = (rows || []).map(row => `
        <div class="bar-row">
            <div class="bar-row-label">
                <span>${escapeHtml(row.label || row.value)}</span>
                <strong>${formatNumber(row.count)}</strong>
            </div>
            <div class="bar-track">
                <div class="bar-fill" style="width:${Math.max((row.count / max) * 100, 2)}%"></div>
            </div>
        </div>
    `).join('');
}

function renderIndexCache(index) {
    const files = index.cache_files || {};
    const manifest = index.manifest || {};
    document.getElementById('index-cache').innerHTML = `
        <div><strong>Cache used</strong><span>${index.cache_used ? 'Yes' : 'No, built during startup'}</span></div>
        <div><strong>Tokenizer</strong><span>${escapeHtml(manifest.tokenizer_engine || 'unknown')}</span></div>
        <div><strong>Avg doc length</strong><span>${Number(index.avg_doc_length || 0).toFixed(1)} tokens</span></div>
        <div><strong>Manifest</strong><span>${files.manifest?.exists ? 'Present' : 'Missing'}</span></div>
        <div><strong>Lexical cache</strong><span>${files.cache?.exists ? formatBytes(files.cache.size) : 'Missing'}</span></div>
    `;
}

function renderCompleteness(fields) {
    document.getElementById('metadata-completeness').innerHTML = (fields || []).map(field => `
        <div class="completeness-item ${field.status}">
            <div class="completeness-top">
                <strong>${escapeHtml(field.field)}</strong>
                <span>${(field.coverage * 100).toFixed(1)}%</span>
            </div>
            <div class="bar-track">
                <div class="bar-fill" style="width:${field.coverage * 100}%"></div>
            </div>
            <small>${formatNumber(field.present)} present / ${formatNumber(field.missing)} missing</small>
        </div>
    `).join('');
}

function renderFacetQuality(quality) {
    const fields = (quality?.fields || []).filter(field => [
        'source',
        'taxonomy_topic',
        'taxonomy_path',
        'published_year',
        'published_month',
        'tags',
        'source_name',
        'author'
    ].includes(field.field));
    document.getElementById('facet-quality').innerHTML = fields.map(field => `
        <div class="completeness-item ${field.status}">
            <div class="completeness-top">
                <strong>${escapeHtml(field.field)}</strong>
                <span>${(field.coverage * 100).toFixed(1)}%</span>
            </div>
            <div class="bar-track">
                <div class="bar-fill" style="width:${field.coverage * 100}%"></div>
            </div>
            <small>${formatNumber(field.usable_documents)} usable / ${formatNumber(field.missing_or_hidden_documents)} hidden or missing</small>
        </div>
    `).join('');
    document.getElementById('facet-quality-notes').innerHTML = (quality?.notes || [])
        .map(note => `<span>${escapeHtml(note)}</span>`)
        .join('');
}

function renderFlow(steps) {
    document.getElementById('system-flow').innerHTML = (steps || []).map((step, index) => `
        <div class="flow-step">
            <div class="flow-index">${index + 1}</div>
            <div>
                <strong>${escapeHtml(step.label)}</strong>
                <p>${escapeHtml(step.detail)}</p>
            </div>
        </div>
    `).join('');
}

function renderExternalSources(rows) {
    document.getElementById('external-sources').innerHTML = (rows || []).map(row => `
        <div class="external-source">
            <strong>${escapeHtml(row.project)}</strong>
            <code>${escapeHtml(row.path)}</code>
            <p>${escapeHtml(row.role)}</p>
            <small>${escapeHtml(row.import_policy)}</small>
        </div>
    `).join('');
}

async function runTopicExplorer() {
    const button = document.getElementById('run-topic-explorer');
    const status = document.getElementById('topic-status');
    const results = document.getElementById('topic-results');
    const query = document.getElementById('topic-query').value.trim();
    const method = document.getElementById('topic-method').value;
    const nClusters = Number(document.getElementById('topic-clusters').value || 5);
    const sampleSize = Number(document.getElementById('topic-sample-size').value || 120);

    status.textContent = 'Analyzing topic clusters...';
    results.innerHTML = '';
    button.disabled = true;
    try {
        const response = await fetch('api/topics', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query,
                method,
                n_clusters: nClusters,
                sample_size: sampleSize,
                model: 'bm25'
            })
        });
        const payload = await response.json();
        if (!payload.ok) {
            throw new Error(payload.error?.message || 'Topic analysis failed');
        }
        renderTopics(payload.data);
        status.textContent = `${payload.data.method} clustered ${payload.data.sample_size} documents into ${payload.data.n_clusters} topics.`;
    } catch (error) {
        status.textContent = `Topic analysis failed: ${error.message}`;
    } finally {
        button.disabled = false;
    }
}

function renderTopics(data) {
    const topics = data.topics || [];
    const results = document.getElementById('topic-results');
    if (!topics.length) {
        results.innerHTML = `<div class="no-results">${escapeHtml(data.message || 'No topics available')}</div>`;
        return;
    }
    results.innerHTML = topics.map(topic => `
        <article class="topic-card">
            <div class="topic-card-header">
                <h3>${escapeHtml(topic.label)}</h3>
                <span>${formatNumber(topic.size)} docs</span>
            </div>
            <div class="topic-keywords">
                ${(topic.keywords || []).slice(0, 8).map(keyword => `<span>${escapeHtml(keyword.term)} ${Number(keyword.weight || 0).toFixed(3)}</span>`).join('')}
            </div>
            <div class="topic-documents">
                ${(topic.representative_documents || []).map(doc => `
                    <button type="button" class="topic-doc" onclick="window.location.href='./?q=${encodeURIComponent(doc.title || '')}&model=hybrid&run=1'">
                        <strong>${escapeHtml(doc.title || `Document ${doc.doc_id}`)}</strong>
                        <small>${escapeHtml(doc.source_label || doc.source || '')} ${escapeHtml(doc.published_date || '')}</small>
                    </button>
                `).join('')}
            </div>
        </article>
    `).join('');
}

function formatNumber(value) {
    return Number(value || 0).toLocaleString();
}

function formatBytes(value) {
    const bytes = Number(value || 0);
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function escapeHtml(value) {
    const div = document.createElement('div');
    div.textContent = String(value ?? '');
    return div.innerHTML;
}
