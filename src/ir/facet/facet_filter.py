"""
Facet Filter Module for building and combining filter conditions.

This module provides classes for constructing complex filter conditions
that can be applied to search results, supporting:
- Single value filters
- Multi-select filters (OR logic)
- Range filters (date, numeric)
- Filter combination (AND logic across different fields)

Author: Information Retrieval System
"""

from typing import List, Set, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class FilterOperator(Enum):
    """Filter operators for different condition types."""
    EQUALS = "equals"              # Exact match
    IN = "in"                      # Value in set (multi-select)
    RANGE = "range"                # Value within range
    GREATER_THAN = "gt"            # Greater than
    LESS_THAN = "lt"               # Less than
    GREATER_EQUAL = "gte"          # Greater than or equal
    LESS_EQUAL = "lte"             # Less than or equal
    CONTAINS = "contains"          # String contains
    STARTS_WITH = "starts_with"    # String starts with


@dataclass
class FilterCondition:
    """
    Represents a single filter condition.

    Attributes:
        field: Field name to filter on
        operator: Filter operator (from FilterOperator enum)
        value: Filter value (can be single value, list, or tuple for ranges)
        label: Human-readable label for UI display

    Example:
        >>> # Single value filter
        >>> fc1 = FilterCondition("source", FilterOperator.EQUALS, "CNA",
        ...                       label="中央社")
        >>>
        >>> # Multi-select filter
        >>> fc2 = FilterCondition("category", FilterOperator.IN,
        ...                       ["politics", "finance"],
        ...                       label="政治 + 財經")
        >>>
        >>> # Range filter
        >>> fc3 = FilterCondition("pub_date", FilterOperator.RANGE,
        ...                       ("2024-11-01", "2024-11-30"),
        ...                       label="2024年11月")
    """
    field: str
    operator: FilterOperator
    value: Union[str, int, float, List, Tuple, Set]
    label: Optional[str] = None

    def matches(self, doc_value: Any) -> bool:
        """
        Check if a document value matches this filter condition.

        Args:
            doc_value: The value from the document

        Returns:
            True if the document value matches the condition

        Complexity:
            Time: O(1) for most operators, O(n) for IN with n values
            Space: O(1)
        """
        # A missing field cannot satisfy any concrete filter condition.
        if doc_value is None:
            return False

        # Normalize the document value for robust comparisons across types.
        #
        # Many IR pipelines store metadata as strings (JSON), but application code
        # may also pass numbers or datetime objects. Casting to string makes the
        # default behavior predictable, especially for:
        # - categorical facets (source/category/author)
        # - ISO date strings ("YYYY-MM-DD") where lexicographic order matches time
        doc_value_str = str(doc_value)

        if self.operator == FilterOperator.EQUALS:
            # Exact match after string normalization.
            return doc_value_str == str(self.value)

        elif self.operator == FilterOperator.IN:
            # Handle multi-select
            filter_values = self.value if isinstance(self.value, (list, set)) else [self.value]

            # If doc has multiple values (list), check if any match
            if isinstance(doc_value, list):
                # Typical use-case: multi-valued fields such as "tags".
                #
                # Performance note:
                # The list comprehensions allocate repeatedly. For large lists,
                # precomputing a normalized set would be faster, but we keep the
                # implementation straightforward for readability.
                return any(str(v) in [str(fv) for fv in filter_values] for v in doc_value)
            else:
                return doc_value_str in [str(v) for v in filter_values]

        elif self.operator == FilterOperator.RANGE:
            # Range filters assume the caller passes a (min, max) tuple.
            #
            # For date facets, comparing ISO strings is correct:
            #   "2024-11-01" <= "2024-11-15" <= "2024-11-30"
            #
            # For numeric ranges, prefer GREATER/LESS operators (float conversion)
            # or ensure values are comparable after normalization.
            if not isinstance(self.value, tuple) or len(self.value) != 2:
                return False
            min_val, max_val = self.value
            return str(min_val) <= doc_value_str <= str(max_val)

        elif self.operator == FilterOperator.GREATER_THAN:
            # Prefer numeric comparison when possible; fall back to string order.
            try:
                return float(doc_value) > float(self.value)
            except (ValueError, TypeError):
                return doc_value_str > str(self.value)

        elif self.operator == FilterOperator.LESS_THAN:
            # Prefer numeric comparison when possible; fall back to string order.
            try:
                return float(doc_value) < float(self.value)
            except (ValueError, TypeError):
                return doc_value_str < str(self.value)

        elif self.operator == FilterOperator.GREATER_EQUAL:
            # Prefer numeric comparison when possible; fall back to string order.
            try:
                return float(doc_value) >= float(self.value)
            except (ValueError, TypeError):
                return doc_value_str >= str(self.value)

        elif self.operator == FilterOperator.LESS_EQUAL:
            # Prefer numeric comparison when possible; fall back to string order.
            try:
                return float(doc_value) <= float(self.value)
            except (ValueError, TypeError):
                return doc_value_str <= str(self.value)

        elif self.operator == FilterOperator.CONTAINS:
            # Substring match for free-text facets.
            return str(self.value) in doc_value_str

        elif self.operator == FilterOperator.STARTS_WITH:
            # Prefix match, useful for auto-complete or hierarchical categories.
            return doc_value_str.startswith(str(self.value))

        return False

    def to_dict(self) -> dict:
        """
        Convert filter condition to dictionary (for JSON serialization).

        Returns:
            Dictionary representation of the filter condition

        Example:
            >>> fc = FilterCondition("source", FilterOperator.IN, ["CNA", "UDN"])
            >>> fc.to_dict()
            {
                'field': 'source',
                'operator': 'in',
                'value': ['CNA', 'UDN'],
                'label': None
            }
        """
        return {
            'field': self.field,
            'operator': self.operator.value,
            'value': self.value,
            'label': self.label
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'FilterCondition':
        """
        Create FilterCondition from dictionary.

        Args:
            data: Dictionary with filter condition data

        Returns:
            FilterCondition instance
        """
        return cls(
            field=data['field'],
            operator=FilterOperator(data['operator']),
            value=data['value'],
            label=data.get('label')
        )


@dataclass
class RangeFilter(FilterCondition):
    """
    Specialized filter for range conditions (dates, numbers).

    This is a convenience class that automatically sets operator to RANGE.

    Attributes:
        field: Field name
        min_value: Minimum value (inclusive)
        max_value: Maximum value (inclusive)
        label: Display label

    Example:
        >>> # Date range filter
        >>> date_filter = RangeFilter(
        ...     field="pub_date",
        ...     min_value="2024-11-01",
        ...     max_value="2024-11-30",
        ...     label="2024年11月"
        ... )
        >>>
        >>> # Numeric range filter
        >>> score_filter = RangeFilter(
        ...     field="relevance_score",
        ...     min_value=0.5,
        ...     max_value=1.0,
        ...     label="高相關度"
        ... )
    """
    def __init__(self,
                 field: str,
                 min_value: Any,
                 max_value: Any,
                 label: Optional[str] = None):
        """Initialize range filter."""
        super().__init__(
            field=field,
            operator=FilterOperator.RANGE,
            value=(min_value, max_value),
            label=label
        )
        self.min_value = min_value
        self.max_value = max_value


class FacetFilter:
    """
    Manages multiple filter conditions with AND/OR logic.

    This class combines multiple FilterCondition objects and applies
    them to documents with configurable logic:
    - AND across different fields (default)
    - OR within the same field (for multi-select)

    Complexity:
        - add_condition(): O(1)
        - filter(): O(N × C) where N is docs, C is conditions

    Example:
        >>> filter_mgr = FacetFilter()
        >>>
        >>> # Add source filter (multi-select: CNA OR UDN)
        >>> filter_mgr.add_condition(
        ...     FilterCondition("source", FilterOperator.IN, ["CNA", "UDN"])
        ... )
        >>>
        >>> # Add category filter
        >>> filter_mgr.add_condition(
        ...     FilterCondition("category", FilterOperator.EQUALS, "politics")
        ... )
        >>>
        >>> # Add date range filter
        >>> filter_mgr.add_condition(
        ...     RangeFilter("pub_date", "2024-11-01", "2024-11-30")
        ... )
        >>>
        >>> # Apply filters
        >>> filtered_docs = filter_mgr.filter(documents)
    """

    def __init__(self):
        """Initialize facet filter manager."""
        # Store conditions in insertion order.
        #
        # We also maintain a secondary index grouped by field:
        #   field -> [conditions...]
        #
        # This is useful for:
        # - UI rendering (show active filters per facet)
        # - future optimization where we evaluate "OR within a field" directly
        self.conditions: List[FilterCondition] = []
        self._conditions_by_field: dict = {}

    def add_condition(self, condition: FilterCondition) -> None:
        """
        Add a filter condition.

        Args:
            condition: FilterCondition to add

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        self.conditions.append(condition)

        # Group by field for efficient lookup and UI display.
        field = condition.field
        if field not in self._conditions_by_field:
            self._conditions_by_field[field] = []
        self._conditions_by_field[field].append(condition)

    def remove_condition(self, field: str, operator: Optional[FilterOperator] = None) -> None:
        """
        Remove filter condition(s) for a field.

        Args:
            field: Field name to remove conditions for
            operator: Optional operator to match (removes all if None)

        Complexity:
            Time: O(C) where C is total conditions
            Space: O(1)
        """
        # Remove from the main list (order-preserving filter).
        self.conditions = [
            c for c in self.conditions
            if not (c.field == field and (operator is None or c.operator == operator))
        ]

        # Keep the secondary index consistent with the filtered list.
        self._rebuild_field_index()

    def clear(self) -> None:
        """
        Clear all filter conditions.

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        self.conditions = []
        self._conditions_by_field = {}

    def filter(self, documents: List[dict]) -> List[dict]:
        """
        Apply all filter conditions to documents.

        Filter logic:
        - AND across different fields
        - OR within the same field (handled by IN operator)

        Args:
            documents: List of documents to filter

        Returns:
            Filtered list of documents

        Complexity:
            Time: O(N × C) where N is docs, C is conditions
            Space: O(M) where M is matching documents

        Example:
            >>> docs = [
            ...     {"source": "CNA", "category": "politics", "pub_date": "2024-11-15"},
            ...     {"source": "UDN", "category": "finance", "pub_date": "2024-11-16"},
            ...     {"source": "CNA", "category": "politics", "pub_date": "2024-10-20"},
            ... ]
            >>> filter_mgr = FacetFilter()
            >>> filter_mgr.add_condition(
            ...     FilterCondition("source", FilterOperator.IN, ["CNA", "UDN"])
            ... )
            >>> filter_mgr.add_condition(
            ...     RangeFilter("pub_date", "2024-11-01", "2024-11-30")
            ... )
            >>> filtered = filter_mgr.filter(docs)
            >>> len(filtered)
            2  # First two documents match
        """
        if not self.conditions:
            # Fast path: no active filters means no-op.
            return documents

        filtered = []

        for doc in documents:
            # Current semantics:
            # - AND across all conditions
            # - OR within a field should be expressed via a single IN condition
            #
            # Example:
            #   source in {CNA, UDN} AND category == finance
            matches_all = True

            for condition in self.conditions:
                doc_value = doc.get(condition.field)

                if not condition.matches(doc_value):
                    # Short-circuit: one failed condition is enough to reject the doc.
                    matches_all = False
                    break

            if matches_all:
                filtered.append(doc)

        return filtered

    def get_active_filters(self) -> dict:
        """
        Get active filters grouped by field.

        Returns:
            Dictionary mapping field -> list of conditions

        Example:
            >>> filter_mgr.get_active_filters()
            {
                'source': [FilterCondition(...)],
                'pub_date': [RangeFilter(...)]
            }
        """
        return dict(self._conditions_by_field)

    def has_filter(self, field: str) -> bool:
        """
        Check if a field has any active filters.

        Args:
            field: Field name to check

        Returns:
            True if field has active filters
        """
        return field in self._conditions_by_field and len(self._conditions_by_field[field]) > 0

    def get_filter_count(self) -> int:
        """Get total number of active filter conditions."""
        return len(self.conditions)

    def to_dict(self) -> dict:
        """
        Convert all filters to dictionary (for JSON serialization).

        Returns:
            Dictionary with all filter conditions
        """
        return {
            'conditions': [c.to_dict() for c in self.conditions],
            'count': len(self.conditions)
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'FacetFilter':
        """
        Create FacetFilter from dictionary.

        Args:
            data: Dictionary with filter data

        Returns:
            FacetFilter instance
        """
        filter_mgr = cls()
        for condition_data in data.get('conditions', []):
            condition = FilterCondition.from_dict(condition_data)
            filter_mgr.add_condition(condition)
        return filter_mgr

    def _rebuild_field_index(self) -> None:
        """Rebuild the field-based condition index."""
        self._conditions_by_field = {}
        for condition in self.conditions:
            field = condition.field
            if field not in self._conditions_by_field:
                self._conditions_by_field[field] = []
            self._conditions_by_field[field].append(condition)


# Convenience functions for creating common filter types

def create_term_filter(field: str, values: Union[str, List[str]], label: Optional[str] = None) -> FilterCondition:
    """
    Create a term filter (single or multi-select).

    Args:
        field: Field name
        values: Single value or list of values
        label: Display label

    Returns:
        FilterCondition with appropriate operator

    Example:
        >>> # Single select
        >>> f1 = create_term_filter("category", "politics", label="政治")
        >>>
        >>> # Multi-select
        >>> f2 = create_term_filter("source", ["CNA", "UDN"], label="中央社 + 聯合")
    """
    # Map user input to the correct operator:
    # - a list of >1 values => IN (multi-select OR)
    # - a single value => EQUALS
    if isinstance(values, list) and len(values) > 1:
        return FilterCondition(field, FilterOperator.IN, values, label)
    else:
        value = values[0] if isinstance(values, list) else values
        return FilterCondition(field, FilterOperator.EQUALS, value, label)


def create_date_range_filter(field: str,
                            start_date: str,
                            end_date: str,
                            label: Optional[str] = None) -> RangeFilter:
    """
    Create a date range filter.

    Args:
        field: Field name
        start_date: Start date (inclusive)
        end_date: End date (inclusive)
        label: Display label

    Returns:
        RangeFilter for date range

    Example:
        >>> f = create_date_range_filter(
        ...     "pub_date",
        ...     "2024-11-01",
        ...     "2024-11-30",
        ...     label="2024年11月"
        ... )
    """
    return RangeFilter(field, start_date, end_date, label)


def create_numeric_range_filter(field: str,
                               min_value: float,
                               max_value: float,
                               label: Optional[str] = None) -> RangeFilter:
    """
    Create a numeric range filter.

    Args:
        field: Field name
        min_value: Minimum value (inclusive)
        max_value: Maximum value (inclusive)
        label: Display label

    Returns:
        RangeFilter for numeric range

    Example:
        >>> f = create_numeric_range_filter(
        ...     "score",
        ...     0.7,
        ...     1.0,
        ...     label="高分 (0.7+)"
        ... )
    """
    return RangeFilter(field, min_value, max_value, label)
