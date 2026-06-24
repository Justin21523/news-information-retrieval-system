/**
 * Faceted Search JavaScript Module (Enhanced Version)
 *
 * Features:
 * - Preload facets on page load (before search)
 * - Left sidebar layout with mobile drawer
 * - Dynamic facet filtering
 * - Integrated date range filters
 */


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

// Global state
const FacetState = {
    currentQuery: '',
    currentModel: 'bm25',
    facets: {},
    activeFilters: {},
    searchResults: [],
    corpusDistribution: {},
    isInitialized: false
};

const FACET_FIELD_ORDER = [
    'source',
    'taxonomy_topic',
    'taxonomy_path',
    'published_year',
    'published_month',
    'tags',
    'source_name',
    'content_type',
    'category',
    'category_name',
    'author'
];

// DOM Elements cache
let DOM = {};

/**
 * Initialize faceted search functionality
 */
function initFacetedSearch() {
    if (FacetState.isInitialized) return;
    console.log('Initializing faceted search...');

    // Cache DOM elements
    DOM = {
        sidebar: document.getElementById('filter-sidebar'),
        overlay: document.getElementById('sidebar-overlay'),
        mobileToggle: document.getElementById('mobile-filter-toggle'),
        closeBtn: document.getElementById('close-sidebar'),
        clearBtn: document.getElementById('clear-filters-btn'),
        facetGroups: document.getElementById('facet-groups'),
        activeFilters: document.getElementById('active-filters'),
        queryInput: document.getElementById('query-input'),
        modelSelect: document.getElementById('model-select'),
        topkInput: document.getElementById('topk-input')
    };

    // Setup mobile drawer events
    setupMobileDrawer();

    // Setup clear filters button
    if (DOM.clearBtn) {
        DOM.clearBtn.addEventListener('click', clearAllFilters);
    }

    // Override the performSearch function from search.js
    if (typeof window.performSearch !== 'undefined') {
        window.originalPerformSearch = window.performSearch;
    }
    window.performSearch = performFacetedSearch;

    // Preload all facets on page load
    preloadAllFacets();

    FacetState.isInitialized = true;
}

/**
 * Setup mobile drawer toggle and close functionality
 */
function setupMobileDrawer() {
    // Open drawer
    if (DOM.mobileToggle) {
        DOM.mobileToggle.addEventListener('click', () => {
            openSidebar();
        });
    }

    // Close drawer
    if (DOM.closeBtn) {
        DOM.closeBtn.addEventListener('click', () => {
            closeSidebar();
        });
    }

    // Close on overlay click
    if (DOM.overlay) {
        DOM.overlay.addEventListener('click', () => {
            closeSidebar();
        });
    }

    // Close on ESC key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && DOM.sidebar?.classList.contains('open')) {
            closeSidebar();
        }
    });
}

/**
 * Open the sidebar drawer (mobile)
 */
function openSidebar() {
    if (DOM.sidebar) {
        DOM.sidebar.classList.add('open');
    }
    if (DOM.overlay) {
        DOM.overlay.classList.add('active');
    }
    document.body.style.overflow = 'hidden';
}

/**
 * Close the sidebar drawer (mobile)
 */
function closeSidebar() {
    if (DOM.sidebar) {
        DOM.sidebar.classList.remove('open');
    }
    if (DOM.overlay) {
        DOM.overlay.classList.remove('active');
    }
    document.body.style.overflow = '';
}

/**
 * Preload all available facets (without search)
 */
async function preloadAllFacets() {
    console.log('Preloading all facets...');

    try {
        const response = await fetch('api/all_facets');

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const data = normalizeApiPayload(await response.json());

        if (!data.success) {
            throw new Error(apiErrorMessage(data));
        }

        FacetState.facets = data.facets;
        FacetState.corpusDistribution = data.corpus_distribution || {};
        applyPendingUrlFilters();
        renderCorpusDistribution(FacetState.corpusDistribution);
        renderFacets(data.facets, false); // false = not from search
        updateActiveFiltersDisplay();

        console.log(`Loaded facets for ${data.total_documents} documents`);

    } catch (error) {
        console.error('Failed to preload facets:', error);
        showFacetError('無法載入篩選選項');
    }
}

/**
 * Show facet error message
 */
function showFacetError(message) {
    if (DOM.facetGroups) {
        DOM.facetGroups.innerHTML = `
            <div class="facet-error">
                <span class="error-icon">⚠️</span>
                <span>${message}</span>
            </div>
        `;
    }
}

/**
 * Perform search and update facets
 */
async function performFacetedSearch() {
    const query = DOM.queryInput?.value?.trim();
    const model = DOM.modelSelect?.value || 'bm25';
    const topK = parseInt(DOM.topkInput?.value) || 20;

    if (!query) {
        if (Object.keys(FacetState.activeFilters).length > 0) {
            await performFacetBrowse(topK);
            return;
        }
        alert('請輸入搜尋關鍵字，或直接點選左側 facet 瀏覽文章');
        return;
    }

    FacetState.currentQuery = query;
    FacetState.currentModel = model;

    showLoading(true);

    try {
        // Step 1: Load facets based on search results (to show counts)
        await loadSearchFacets(query, model, 100);

        // Step 2: Perform actual search with filters
        await performFilteredSearch(query, model, topK);

    } catch (error) {
        console.error('Search error:', error);
        alert('搜尋失敗: ' + error.message);
    } finally {
        showLoading(false);
    }
}

/**
 * Load facets based on search results
 */
async function loadSearchFacets(query, model, topK) {
    const response = await fetch('api/facets', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            query: query,
            model: model,
            top_k: topK,
            filters: FacetState.activeFilters
        })
    });

    if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
    }

    const data = normalizeApiPayload(await response.json());

    if (!data.success) {
        throw new Error(apiErrorMessage(data));
    }

    FacetState.facets = data.facets;
    FacetState.corpusDistribution = data.corpus_distribution || FacetState.corpusDistribution;
    renderCorpusDistribution(FacetState.corpusDistribution);
    renderFacets(data.facets, true); // true = from search
}


/**
 * Render corpus-level source/topic/content-type distribution.
 */
function renderCorpusDistribution(distribution) {
    if (!DOM.facetGroups || !distribution || Object.keys(distribution).length === 0) return;

    let panel = document.getElementById('corpus-distribution');
    if (!panel) {
        panel = document.createElement('div');
        panel.id = 'corpus-distribution';
        panel.className = 'corpus-distribution';
        DOM.facetGroups.parentNode.insertBefore(panel, DOM.facetGroups);
    }

    const sourceItems = (distribution.source || []).slice(0, 8)
        .map(item => '<span class="distribution-chip"><span>' + (item.label || item.value) + '</span><strong>' + item.count + '</strong></span>')
        .join('');
    const topicItems = (distribution.taxonomy_topic || []).slice(0, 8)
        .map(item => '<span class="distribution-chip"><span>' + (item.label || item.value) + '</span><strong>' + item.count + '</strong></span>')
        .join('');

    panel.innerHTML =
        '<div class="corpus-distribution-title">語料覆蓋範圍</div>' +
        '<div class="distribution-section"><div class="distribution-label">來源</div>' + sourceItems + '</div>' +
        '<div class="distribution-section"><div class="distribution-label">主題</div>' + topicItems + '</div>';
}

/**
 * Render facets in the sidebar
 */
function renderFacets(facets, fromSearch = false) {
    if (!DOM.facetGroups) return;

    // Clear existing facets
    DOM.facetGroups.innerHTML = '';

    // Check if facets is empty
    if (!facets || Object.keys(facets).length === 0) {
        DOM.facetGroups.innerHTML = `
            <div class="facet-empty">
                <span>沒有可用的篩選選項</span>
            </div>
        `;
        return;
    }

    const orderedFacets = Object.entries(facets).sort(([left], [right]) => (
        facetOrder(left) - facetOrder(right) || left.localeCompare(right)
    ));
    for (const [fieldName, facet] of orderedFacets) {
        if (!facet.values || facet.values.length === 0) continue;

        const facetGroup = createFacetGroup(fieldName, facet);
        DOM.facetGroups.appendChild(facetGroup);
    }
}

function facetOrder(fieldName) {
    const index = FACET_FIELD_ORDER.indexOf(fieldName);
    return index === -1 ? 999 : index;
}

function applyPendingUrlFilters() {
    const pending = window.pendingUrlFilters || {};
    Object.entries(pending).forEach(([fieldName, rawValues]) => {
        const values = Array.isArray(rawValues) ? rawValues : [rawValues];
        const available = new Set((FacetState.facets[fieldName]?.values || []).map((item) => String(item.value)));
        const validValues = values.map(String).filter((value) => available.size === 0 || available.has(value));
        if (validValues.length) {
            FacetState.activeFilters[fieldName] = [...new Set([...(FacetState.activeFilters[fieldName] || []), ...validValues])];
        }
    });
    window.pendingUrlFilters = {};
}

/**
 * Create a facet group element
 */
function createFacetGroup(fieldName, facet) {
    const group = document.createElement('div');
    group.className = 'facet-group';
    group.dataset.field = fieldName;
    if (facet.quality?.collapsed_by_default) {
        group.classList.add('collapsed');
    }

    const header = document.createElement('div');
    header.className = 'facet-header';
    header.innerHTML = `
        <span class="facet-title">${getFieldIcon(fieldName)} ${escapeHtml(getFacetDisplayName(fieldName, facet))}</span>
        <span class="facet-summary">${formatFacetCoverage(facet)}</span>
        <span class="facet-toggle">${facet.quality?.collapsed_by_default ? '▶' : '▼'}</span>
    `;
    header.onclick = () => toggleFacetGroup(group);
    group.appendChild(header);

    const qualityNote = createFacetQualityNote(facet);
    if (qualityNote) {
        group.appendChild(qualityNote);
    }

    const valuesContainer = document.createElement('div');
    valuesContainer.className = 'facet-values';

    // Render facet values (top 8 by default)
    const displayCount = 8;
    const values = facet.values.slice(0, displayCount);

    values.forEach(fv => {
        const label = createFacetLabel(fieldName, fv);
        valuesContainer.appendChild(label);
    });

    group.appendChild(valuesContainer);

    // Show more button if there are more values
    if (facet.values.length > displayCount) {
        const showMoreBtn = document.createElement('button');
        showMoreBtn.className = 'show-more-btn';
        showMoreBtn.textContent = `+ 更多 (${facet.values.length - displayCount})`;
        showMoreBtn.onclick = (e) => {
            e.stopPropagation();
            expandFacetGroup(group, facet, displayCount);
        };
        group.appendChild(showMoreBtn);
    }

    return group;
}

/**
 * Create a facet checkbox label
 */
function createFacetLabel(fieldName, fv) {
    const label = document.createElement('label');
    label.className = 'facet-label';
    label.title = '點選即可套用篩選並更新結果';

    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.name = fieldName;
    checkbox.value = fv.value;
    checkbox.dataset.count = fv.count;
    checkbox.checked = FacetState.activeFilters[fieldName]?.includes(fv.value) || false;
    checkbox.addEventListener('change', () => handleFacetChange(fieldName, fv.value, checkbox.checked, fv));

    const text = document.createElement('span');
    text.className = 'facet-text';
    text.innerHTML = `<span class="facet-value-name">${escapeHtml(facetValueLabel(fv))}</span><span class="facet-count">${formatNumber(fv.count)}</span>`;

    label.appendChild(checkbox);
    label.appendChild(text);

    return label;
}

function createFacetQualityNote(facet) {
    if (!facet.quality) return null;
    const note = document.createElement('div');
    note.className = 'facet-quality-note';
    const coverage = Number(facet.coverage || facet.quality.coverage || 0);
    const hidden = Number(facet.quality.missing_or_hidden_documents || 0);
    const parts = [`${Math.round(coverage * 100)}% 可用`];
    if (hidden > 0) {
        parts.push(`${formatNumber(hidden)} 筆隱藏或缺漏`);
    }
    if (facet.quality.is_low_information) {
        parts.push('低變異度');
    }
    note.textContent = parts.join(' · ');
    return note;
}

function formatFacetCoverage(facet) {
    const coverage = Number(facet.coverage || facet.quality?.coverage || 0);
    if (!Number.isFinite(coverage) || coverage <= 0) return '';
    return `${Math.round(coverage * 100)}%`;
}

/**
 * Toggle facet group collapse
 */
function toggleFacetGroup(group) {
    group.classList.toggle('collapsed');
    const toggle = group.querySelector('.facet-toggle');
    if (toggle) {
        toggle.textContent = group.classList.contains('collapsed') ? '▶' : '▼';
    }
}

/**
 * Expand facet group to show all values
 */
function expandFacetGroup(group, facet, currentCount) {
    const valuesContainer = group.querySelector('.facet-values');
    const fieldName = group.dataset.field;

    // Clear and re-render with all values
    valuesContainer.innerHTML = '';

    facet.values.forEach(fv => {
        const label = createFacetLabel(fieldName, fv);
        valuesContainer.appendChild(label);
    });

    // Replace show more with show less
    const showMoreBtn = group.querySelector('.show-more-btn');
    if (showMoreBtn) {
        showMoreBtn.textContent = '- 收起';
        showMoreBtn.onclick = (e) => {
            e.stopPropagation();
            collapseFacetGroup(group, facet, currentCount);
        };
    }
}

/**
 * Collapse facet group to show fewer values
 */
function collapseFacetGroup(group, facet, displayCount) {
    const valuesContainer = group.querySelector('.facet-values');
    const fieldName = group.dataset.field;

    // Clear and re-render with limited values
    valuesContainer.innerHTML = '';

    facet.values.slice(0, displayCount).forEach(fv => {
        const label = createFacetLabel(fieldName, fv);
        valuesContainer.appendChild(label);
    });

    // Update button text
    const showMoreBtn = group.querySelector('.show-more-btn');
    if (showMoreBtn) {
        showMoreBtn.textContent = `+ 更多 (${facet.values.length - displayCount})`;
        showMoreBtn.onclick = (e) => {
            e.stopPropagation();
            expandFacetGroup(group, facet, displayCount);
        };
    }
}

/**
 * Handle facet checkbox change
 */
function handleFacetChange(fieldName, value, checked, facetValue = null) {
    if (!FacetState.activeFilters[fieldName]) {
        FacetState.activeFilters[fieldName] = [];
    }

    if (checked) {
        if (!FacetState.activeFilters[fieldName].includes(value)) {
            FacetState.activeFilters[fieldName].push(value);
        }
    } else {
        FacetState.activeFilters[fieldName] = FacetState.activeFilters[fieldName].filter(v => v !== value);
        if (FacetState.activeFilters[fieldName].length === 0) {
            delete FacetState.activeFilters[fieldName];
        }
    }

    // Update active filters display
    updateActiveFiltersDisplay();

    runSearchForFacetChange(fieldName, value, facetValue);
}

function runSearchForFacetChange(fieldName, value, facetValue = null) {
    const topK = parseInt(DOM.topkInput?.value) || 20;
    const query = currentOrFacetQuery();
    if (!query) {
        performFacetBrowse(topK).catch((error) => {
            console.error('Facet browse error:', error);
            alert('篩選瀏覽失敗: ' + error.message);
        });
        return;
    }
    FacetState.currentQuery = query;
    FacetState.currentModel = DOM.modelSelect?.value || FacetState.currentModel || 'bm25';
    loadSearchFacets(FacetState.currentQuery, FacetState.currentModel, Math.max(topK, 100))
        .then(() => performFilteredSearch(FacetState.currentQuery, FacetState.currentModel, topK))
        .catch((error) => {
            console.error('Facet click search error:', error);
            alert('篩選查詢失敗: ' + error.message);
        });
}

function currentOrFacetQuery() {
    const typedQuery = DOM.queryInput?.value?.trim();
    if (typedQuery) return typedQuery;
    if (FacetState.currentQuery) return FacetState.currentQuery;
    return '';
}

/**
 * Perform filtered search
 */
async function performFilteredSearch(query, model, topK) {
    showLoading(true);

    try {
        const response = await fetch('api/search/faceted', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: query,
                model: model,
                top_k: topK,
                filters: FacetState.activeFilters
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const data = normalizeApiPayload(await response.json());

        if (!data.success) {
            throw new Error(apiErrorMessage(data));
        }

        FacetState.searchResults = data.results;
        displaySearchResults(data);

    } catch (error) {
        console.error('Filtered search error:', error);
        alert('搜尋失敗: ' + error.message);
    } finally {
        showLoading(false);
    }
}

async function performFacetBrowse(topK) {
    showLoading(true);

    try {
        const response = await fetch('api/search/browse', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                filters: FacetState.activeFilters,
                top_k: topK,
                sort: 'date_desc'
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const data = normalizeApiPayload(await response.json());
        if (!data.success) {
            throw new Error(apiErrorMessage(data));
        }

        FacetState.searchResults = data.results || [];
        FacetState.facets = data.facets || FacetState.facets;
        renderFacets(FacetState.facets, true);
        updateActiveFiltersDisplay();
        displaySearchResults({
            ...data,
            query: '',
            browse_mode: true,
            model: 'facet_browse',
            query_analysis: {
                warnings: ['目前顯示 facet metadata 篩選結果，未使用關鍵字排序。']
            },
            model_info: {
                id: 'facet_browse',
                name: 'Facet Browse',
                description: '直接依 metadata facets 瀏覽符合條件的新聞。'
            }
        });
    } catch (error) {
        console.error('Facet browse error:', error);
        alert('篩選瀏覽失敗: ' + error.message);
    } finally {
        showLoading(false);
    }
}

/**
 * Display search results using the global displayResults function
 */
function displaySearchResults(data) {
    // Trigger existing results display function
    if (typeof displayResults === 'function') {
        displayResults(data);
    } else {
        console.log('Search results:', data.results);
    }
}

/**
 * Update active filters display
 */
function updateActiveFiltersDisplay() {
    if (!DOM.activeFilters) return;

    const hasFilters = Object.keys(FacetState.activeFilters).length > 0;

    if (!hasFilters) {
        DOM.activeFilters.innerHTML = '';
        return;
    }

    let html = `
        <div class="active-filters-label">已套用篩選:</div>
        <div class="active-filter-tags">
    `;

    for (const [fieldName, values] of Object.entries(FacetState.activeFilters)) {
        const facet = FacetState.facets[fieldName];
        const displayName = getFacetDisplayName(fieldName, facet);

        values.forEach(value => {
            const facetValue = facet?.values?.find(item => item.value === value);
            const label = facetValueLabel(facetValue || { value });
            html += `
                <span class="filter-tag">
                    ${displayName}: ${label}
                    <span class="filter-tag-remove" onclick="FacetSearch.removeFilter('${fieldName}', '${value}')">&times;</span>
                </span>
            `;
        });
    }

    html += '</div>';
    DOM.activeFilters.innerHTML = html;
}

/**
 * Remove a specific filter
 */
function removeFilter(fieldName, value) {
    // Update state
    if (FacetState.activeFilters[fieldName]) {
        FacetState.activeFilters[fieldName] = FacetState.activeFilters[fieldName].filter(v => v !== value);
        if (FacetState.activeFilters[fieldName].length === 0) {
            delete FacetState.activeFilters[fieldName];
        }
    }

    // Update checkbox
    const checkbox = document.querySelector(`input[name="${fieldName}"][value="${value}"]`);
    if (checkbox) {
        checkbox.checked = false;
    }

    // Update display
    updateActiveFiltersDisplay();

    if (FacetState.currentQuery) runSearchForFacetChange(fieldName, value);
}

/**
 * Clear all filters
 */
function clearAllFilters() {
    FacetState.activeFilters = {};

    // Uncheck all checkboxes
    document.querySelectorAll('.facet-values input[type="checkbox"]').forEach(cb => {
        cb.checked = false;
    });

    // Update display
    updateActiveFiltersDisplay();

    if (FacetState.currentQuery) runSearchForFacetChange('', '');
}

/**
 * Get icon for field
 */
function getFieldIcon(fieldName) {
    const icons = {
        'source': '📰',
        'category': '🏷️',
        'category_name': '📂',
        'content_type': '🧾',
        'taxonomy_topic': '🗂️',
        'taxonomy_path': '🧭',
        'published_year': '📅',
        'published_month': '🗓️',
        'pub_date': '📅',
        'author': '✍️',
        'tags': '#️⃣'
    };
    return icons[fieldName] || '🔹';
}

function getFacetDisplayName(fieldName, facet = null) {
    const labels = {
        source: '新聞來源',
        source_name: '來源名稱',
        category: '原始分類',
        category_name: '分類名稱',
        content_type: '內容類型',
        taxonomy_topic: '主題分類',
        taxonomy_path: '分類路徑',
        published_year: '年份',
        published_month: '月份',
        pub_date: '發布日期',
        author: '作者',
        tags: '標籤'
    };
    return labels[fieldName] || facet?.display_name || fieldName;
}

function facetValueLabel(fv) {
    if (!fv) return '';
    const value = String(fv.label || fv.value || '');
    const labels = {
        article: '新聞文章',
        news: '新聞',
        business: '財經',
        politics: '政治',
        society: '社會',
        technology: '科技',
        international: '國際',
        sports: '體育',
        entertainment: '娛樂',
        lifestyle: '生活',
        health: '健康',
        education: '教育',
        opinion: '評論',
        general: '綜合'
    };
    return labels[value] || labels[value.toLowerCase()] || value;
}

function formatNumber(value) {
    return Number(value || 0).toLocaleString();
}

function escapeHtml(value) {
    const div = document.createElement('div');
    div.textContent = String(value ?? '');
    return div.innerHTML;
}

/**
 * Show/hide loading indicator
 */
function showLoading(show) {
    const loading = document.getElementById('loading');
    if (loading) {
        loading.style.display = show ? 'flex' : 'none';
    }
}

/**
 * Get active filters for external use
 */
function getActiveFilters() {
    return { ...FacetState.activeFilters };
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initFacetedSearch);
} else {
    initFacetedSearch();
}

// Export for external use
window.FacetSearch = {
    init: initFacetedSearch,
    clearFilters: clearAllFilters,
    removeFilter: removeFilter,
    getFilters: getActiveFilters,
    openSidebar: openSidebar,
    closeSidebar: closeSidebar,
    state: FacetState
};

// Also expose getActiveFilters globally for search.js
window.getActiveFilters = getActiveFilters;
