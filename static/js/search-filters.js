// CNIRS - Search Filters Module

// ========== Filter Event Listeners ==========

// Toggle filter panel
if (filterToggleBtn) {
    filterToggleBtn.addEventListener('click', () => {
        if (filterPanel.style.display === 'none') {
            filterPanel.style.display = 'block';
            filterToggleBtn.textContent = '🔼 收起篩選';
        } else {
            filterPanel.style.display = 'none';
            filterToggleBtn.textContent = '🔽 進階篩選';
        }
    });
}

// Apply filters button
if (applyFiltersBtn) {
    applyFiltersBtn.addEventListener('click', () => {
        performSearch();
    });
}

// Clear filters button
if (clearFiltersBtn) {
    clearFiltersBtn.addEventListener('click', () => {
        clearAllFilters();
    });
}

// ========== Filter Functions ==========

/**
 * Load available filter options from API
 */
async function loadFilterOptions() {
    try {
        const response = await fetch('/api/filters');
        const data = await response.json();

        if (data.success) {
            availableCategories = data.categories;
            renderCategoryFilters(data.categories);

            // Set date range limits
            if (data.date_range.min && dateFrom && dateTo) {
                dateFrom.min = data.date_range.min;
                dateTo.min = data.date_range.min;
                dateFrom.max = data.date_range.max;
                dateTo.max = data.date_range.max;
            }
        }
    } catch (error) {
        console.error('Failed to load filter options:', error);
    }
}

/**
 * Render category checkboxes
 */
function renderCategoryFilters(categories) {
    if (!categoryFilters) return;

    categoryFilters.innerHTML = '';

    categories.forEach(category => {
        const item = document.createElement('label');
        item.className = 'checkbox-item';
        item.innerHTML = `
            <input type="checkbox" value="${category}" class="category-checkbox">
            <span>${category.toUpperCase()}</span>
        `;

        // Add change listener
        const checkbox = item.querySelector('input');
        checkbox.addEventListener('change', (e) => {
            if (e.target.checked) {
                selectedCategories.add(category);
                item.classList.add('checked');
            } else {
                selectedCategories.delete(category);
                item.classList.remove('checked');
            }
        });

        categoryFilters.appendChild(item);
    });
}

/**
 * Get currently active filters
 */
function getActiveFilters() {
    const filters = {};

    // Category filters
    if (selectedCategories && selectedCategories.size > 0) {
        filters.categories = Array.from(selectedCategories);
    }

    // Date range filters
    if (dateFrom && dateFrom.value) {
        filters.date_from = dateFrom.value;
    }
    if (dateTo && dateTo.value) {
        filters.date_to = dateTo.value;
    }

    return Object.keys(filters).length > 0 ? filters : null;
}

/**
 * Clear all active filters
 */
function clearAllFilters() {
    // Clear category selections
    if (selectedCategories) {
        selectedCategories.clear();
    }

    document.querySelectorAll('.category-checkbox').forEach(cb => {
        cb.checked = false;
    });

    document.querySelectorAll('.checkbox-item').forEach(item => {
        item.classList.remove('checked');
    });

    // Clear date inputs
    if (dateFrom) dateFrom.value = '';
    if (dateTo) dateTo.value = '';

    // Re-run search if there's a query
    if (queryInput && queryInput.value.trim()) {
        performSearch();
    }
}
