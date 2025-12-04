/**
 * Faceted Search JavaScript Module (Enhanced Version)
 *
 * Features:
 * - Preload facets on page load (before search)
 * - Left sidebar layout with mobile drawer
 * - Dynamic facet filtering
 * - Integrated date range filters
 */

// Global state
const FacetState = {
    currentQuery: '',
    currentModel: 'bm25',
    facets: {},
    activeFilters: {},
    searchResults: [],
    isInitialized: false
};

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
        const response = await fetch('/api/all_facets');

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        const data = await response.json();

        if (!data.success) {
            throw new Error(data.error || 'Unknown error');
        }

        FacetState.facets = data.facets;
        renderFacets(data.facets, false); // false = not from search

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
        alert('請輸入搜尋關鍵字');
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
    const response = await fetch('/api/facets', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            query: query,
            model: model,
            top_k: topK
        })
    });

    if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
    }

    const data = await response.json();

    if (!data.success) {
        throw new Error(data.error || 'Unknown error');
    }

    FacetState.facets = data.facets;
    renderFacets(data.facets, true); // true = from search
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

    // Render each facet group
    for (const [fieldName, facet] of Object.entries(facets)) {
        // Skip empty facets
        if (!facet.values || facet.values.length === 0) continue;

        const facetGroup = createFacetGroup(fieldName, facet);
        DOM.facetGroups.appendChild(facetGroup);
    }
}

/**
 * Create a facet group element
 */
function createFacetGroup(fieldName, facet) {
    const group = document.createElement('div');
    group.className = 'facet-group';
    group.dataset.field = fieldName;

    // Header with collapse toggle
    const header = document.createElement('div');
    header.className = 'facet-header';
    header.innerHTML = `
        <span class="facet-title">${getFieldIcon(fieldName)} ${facet.display_name}</span>
        <span class="facet-toggle">▼</span>
    `;
    header.onclick = () => toggleFacetGroup(group);
    group.appendChild(header);

    // Values container
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

    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.name = fieldName;
    checkbox.value = fv.value;
    checkbox.dataset.count = fv.count;
    checkbox.checked = FacetState.activeFilters[fieldName]?.includes(fv.value) || false;
    checkbox.addEventListener('change', () => handleFacetChange(fieldName, fv.value, checkbox.checked));

    const text = document.createElement('span');
    text.className = 'facet-text';
    text.innerHTML = `<span class="facet-value-name">${fv.label || fv.value}</span><span class="facet-count">${fv.count}</span>`;

    label.appendChild(checkbox);
    label.appendChild(text);

    return label;
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
function handleFacetChange(fieldName, value, checked) {
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

    // Re-run search with new filters (only if a search has been performed)
    if (FacetState.currentQuery) {
        const topK = parseInt(DOM.topkInput?.value) || 20;
        performFilteredSearch(FacetState.currentQuery, FacetState.currentModel, topK);
    }
}

/**
 * Perform filtered search
 */
async function performFilteredSearch(query, model, topK) {
    showLoading(true);

    try {
        const response = await fetch('/api/search/faceted', {
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

        const data = await response.json();

        if (!data.success) {
            throw new Error(data.error || 'Unknown error');
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
        const displayName = facet?.display_name || fieldName;

        values.forEach(value => {
            html += `
                <span class="filter-tag">
                    ${displayName}: ${value}
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

    // Re-run search if a search has been performed
    if (FacetState.currentQuery) {
        const topK = parseInt(DOM.topkInput?.value) || 20;
        performFilteredSearch(FacetState.currentQuery, FacetState.currentModel, topK);
    }
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

    // Re-run search if a search has been performed
    if (FacetState.currentQuery) {
        const topK = parseInt(DOM.topkInput?.value) || 20;
        performFilteredSearch(FacetState.currentQuery, FacetState.currentModel, topK);
    }
}

/**
 * Get icon for field
 */
function getFieldIcon(fieldName) {
    const icons = {
        'source': '📰',
        'category': '🏷️',
        'category_name': '📂',
        'pub_date': '📅',
        'author': '✍️'
    };
    return icons[fieldName] || '🔹';
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
