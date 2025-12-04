"""
Faceted Search Module

This module implements faceted search (分面搜尋) functionality for
filtering and refining search results through multiple dimensions.

Features:
    - Multi-field filtering (source, category, date, etc.)
    - Range filters (date range, numeric ranges)
    - Multi-select filters (checkbox-style)
    - Dynamic facet counts
    - Filter combination logic

Author: Information Retrieval System
"""

from .facet_engine import FacetEngine, FacetResult, FacetValue
from .facet_filter import (
    FacetFilter,
    FilterCondition,
    RangeFilter,
    FilterOperator,
    create_term_filter,
    create_date_range_filter,
    create_numeric_range_filter
)

__all__ = [
    'FacetEngine',
    'FacetResult',
    'FacetValue',
    'FacetFilter',
    'FilterCondition',
    'RangeFilter',
    'FilterOperator',
    'create_term_filter',
    'create_date_range_filter',
    'create_numeric_range_filter'
]
