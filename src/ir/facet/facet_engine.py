"""
Facet Engine for computing facet values and counts from search results.

This module provides the core faceted search engine that:
1. Computes available facet values from search results
2. Calculates document counts for each facet value
3. Supports filtering by multiple facets simultaneously
4. Updates facet counts dynamically based on selected filters

Author: Information Retrieval System
"""

from typing import List, Dict, Set, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import logging


@dataclass
class FacetValue:
    """
    Represents a single facet value with its document count.

    Attributes:
        value: The facet value (e.g., "CNA", "2024-11", "politics")
        count: Number of documents matching this facet value
        label: Human-readable label (optional, defaults to value)

    Example:
        >>> fv = FacetValue(value="CNA", count=150, label="中央社")
        >>> print(f"{fv.label}: {fv.count} documents")
        中央社: 150 documents
    """
    value: str
    count: int
    label: Optional[str] = None

    def __post_init__(self):
        if self.label is None:
            self.label = self.value


@dataclass
class FacetResult:
    """
    Represents facet computation results for a single field.

    Attributes:
        field_name: Name of the facet field (e.g., "source", "category")
        display_name: Human-readable display name
        facet_type: Type of facet ("term", "date_range", "numeric_range")
        values: List of FacetValue objects sorted by count (descending)
        total_docs: Total number of documents in the result set

    Example:
        >>> result = FacetResult(
        ...     field_name="source",
        ...     display_name="新聞來源",
        ...     facet_type="term",
        ...     values=[FacetValue("CNA", 150), FacetValue("UDN", 120)],
        ...     total_docs=270
        ... )
    """
    field_name: str
    display_name: str
    facet_type: str  # "term", "date_range", "numeric_range"
    values: List[FacetValue] = field(default_factory=list)
    total_docs: int = 0

    def get_top_k(self, k: int = 10) -> List[FacetValue]:
        """
        Get top-k facet values by count.

        Args:
            k: Number of top values to return

        Returns:
            List of top-k FacetValue objects

        Complexity:
            Time: O(1) since values are pre-sorted
            Space: O(k)
        """
        return self.values[:k]


class FacetEngine:
    """
    Facet computation engine for search result filtering.

    This engine computes facet distributions from search results and
    supports dynamic facet count updates based on filter selections.

    Supported facet types:
        - Term facets: Discrete values (source, category, author, etc.)
        - Date range facets: Date bucketing (by month, year)
        - Numeric range facets: Numeric bucketing

    Complexity:
        - build_facets(): O(N × F) where N is docs, F is facet fields
        - apply_filters(): O(N × C) where C is filter conditions

    Example:
        >>> engine = FacetEngine()
        >>> engine.configure_facet("source", "新聞來源", "term")
        >>> engine.configure_facet("category", "分類", "term")
        >>> engine.configure_facet("pub_date", "發布日期", "date_range",
        ...                        date_format="%Y-%m")
        >>>
        >>> # Build facets from search results
        >>> docs = [{"source": "CNA", "category": "politics", ...}, ...]
        >>> facets = engine.build_facets(docs)
        >>>
        >>> # Get top sources
        >>> source_facet = facets["source"]
        >>> print(source_facet.get_top_k(5))
    """

    def __init__(self):
        """Initialize the facet engine."""
        self.logger = logging.getLogger(__name__)

        # Facet configuration: field_name -> config
        self.facet_configs: Dict[str, Dict[str, Any]] = {}

        # Cache for facet results
        self._facet_cache: Dict[str, FacetResult] = {}

    def configure_facet(self,
                       field_name: str,
                       display_name: str,
                       facet_type: str = "term",
                       **kwargs) -> None:
        """
        Configure a facet field.

        Args:
            field_name: Field name in document metadata
            display_name: Human-readable display name
            facet_type: Type of facet ("term", "date_range", "numeric_range")
            **kwargs: Additional configuration options:
                - date_format: Format string for date bucketing (e.g., "%Y-%m")
                - range_buckets: List of (min, max, label) tuples for ranges
                - max_values: Maximum number of facet values to return

        Example:
            >>> engine.configure_facet("source", "新聞來源", "term", max_values=20)
            >>> engine.configure_facet("pub_date", "發布月份", "date_range",
            ...                        date_format="%Y-%m")
        """
        self.facet_configs[field_name] = {
            'display_name': display_name,
            'facet_type': facet_type,
            **kwargs
        }
        self.logger.debug(f"Configured facet: {field_name} ({facet_type})")

    def build_facets(self,
                    documents: List[Dict[str, Any]],
                    field_name: Optional[str] = None) -> Dict[str, FacetResult]:
        """
        Build facet distributions from documents.

        Args:
            documents: List of document dictionaries with metadata
            field_name: Optional field name to build facets for (builds all if None)

        Returns:
            Dictionary mapping field_name -> FacetResult

        Complexity:
            Time: O(N × F) where N is document count, F is facet fields
            Space: O(U) where U is unique facet values across all fields

        Example:
            >>> docs = [
            ...     {"source": "CNA", "category": "politics", "pub_date": "2024-11-20"},
            ...     {"source": "UDN", "category": "finance", "pub_date": "2024-11-20"},
            ...     {"source": "CNA", "category": "politics", "pub_date": "2024-11-19"},
            ... ]
            >>> facets = engine.build_facets(docs)
            >>> facets["source"].values
            [FacetValue("CNA", 2), FacetValue("UDN", 1)]
        """
        if not documents:
            self.logger.warning("No documents provided for facet building")
            return {}

        # Determine which fields to process
        fields_to_process = [field_name] if field_name else self.facet_configs.keys()

        results = {}
        total_docs = len(documents)

        for field in fields_to_process:
            if field not in self.facet_configs:
                self.logger.warning(f"Field '{field}' not configured, skipping")
                continue

            config = self.facet_configs[field]
            facet_type = config['facet_type']

            if facet_type == "term":
                result = self._build_term_facet(field, documents, config)
            elif facet_type == "date_range":
                result = self._build_date_range_facet(field, documents, config)
            elif facet_type == "numeric_range":
                result = self._build_numeric_range_facet(field, documents, config)
            else:
                self.logger.error(f"Unknown facet type: {facet_type}")
                continue

            result.total_docs = total_docs
            results[field] = result

        return results

    def _build_term_facet(self,
                         field_name: str,
                         documents: List[Dict[str, Any]],
                         config: Dict[str, Any]) -> FacetResult:
        """
        Build a term facet (discrete values).

        Args:
            field_name: Field to build facet for
            documents: List of documents
            config: Facet configuration

        Returns:
            FacetResult with term distribution

        Complexity:
            Time: O(N) where N is document count
            Space: O(U) where U is unique values
        """
        # Count occurrences of each value
        value_counts: Dict[str, int] = defaultdict(int)

        for doc in documents:
            value = doc.get(field_name)
            if value is not None:
                # Handle both single values and lists
                if isinstance(value, list):
                    for v in value:
                        value_counts[str(v)] += 1
                else:
                    value_counts[str(value)] += 1

        # Sort by count (descending), then by value (ascending)
        sorted_values = sorted(
            value_counts.items(),
            key=lambda x: (-x[1], x[0])
        )

        # Apply max_values limit if configured
        max_values = config.get('max_values', None)
        if max_values:
            sorted_values = sorted_values[:max_values]

        # Create FacetValue objects
        facet_values = [
            FacetValue(value=value, count=count)
            for value, count in sorted_values
        ]

        return FacetResult(
            field_name=field_name,
            display_name=config['display_name'],
            facet_type="term",
            values=facet_values
        )

    def _build_date_range_facet(self,
                               field_name: str,
                               documents: List[Dict[str, Any]],
                               config: Dict[str, Any]) -> FacetResult:
        """
        Build a date range facet with bucketing.

        Args:
            field_name: Field to build facet for
            documents: List of documents
            config: Facet configuration with 'date_format' key

        Returns:
            FacetResult with date range distribution

        Complexity:
            Time: O(N) where N is document count
            Space: O(B) where B is number of buckets

        Example:
            Date format "%Y-%m" groups by month:
            "2024-11-20" -> "2024-11"
            "2024-11-15" -> "2024-11"

        Note:
            Documents with missing or unparseable dates are grouped
            into an "unknown" bucket for transparency.
        """
        date_format = config.get('date_format', '%Y-%m')
        bucket_counts: Dict[str, int] = defaultdict(int)
        unknown_count = 0

        # Extended date format support for various sources
        date_formats = [
            '%Y-%m-%d',              # ISO: 2024-11-20
            '%Y-%m-%dT%H:%M:%S',     # ISO datetime: 2024-11-20T10:30:00
            '%Y-%m-%dT%H:%M:%S%z',   # ISO with timezone: 2024-11-20T10:30:00+08:00
            '%Y-%m-%dT%H:%M:%SZ',    # ISO UTC: 2024-11-20T10:30:00Z
            '%Y/%m/%d',              # Slash format: 2024/11/20
            '%Y/%m/%d %H:%M:%S',     # Slash with time
            '%Y年%m月%d日',           # Chinese format: 2024年11月20日
            '%d-%m-%Y',              # European: 20-11-2024
            '%m/%d/%Y',              # US format: 11/20/2024
        ]

        for doc in documents:
            date_value = doc.get(field_name)

            # Handle missing or empty dates
            if date_value is None or date_value == '':
                unknown_count += 1
                continue

            parsed = False
            try:
                # Parse date string and format according to bucket
                if isinstance(date_value, str):
                    date_value = date_value.strip()

                    # Handle ISO format with timezone offset (remove +HH:MM suffix)
                    if '+' in date_value and 'T' in date_value:
                        date_value = date_value.split('+')[0]
                    elif date_value.endswith('Z'):
                        date_value = date_value[:-1]

                    # Try to parse with multiple formats
                    for fmt in date_formats:
                        try:
                            dt = datetime.strptime(date_value, fmt)
                            bucket = dt.strftime(date_format)
                            bucket_counts[bucket] += 1
                            parsed = True
                            break
                        except ValueError:
                            continue

                elif isinstance(date_value, datetime):
                    bucket = date_value.strftime(date_format)
                    bucket_counts[bucket] += 1
                    parsed = True

            except Exception as e:
                self.logger.warning(f"Failed to parse date '{date_value}': {e}")

            # Count as unknown if parsing failed
            if not parsed:
                unknown_count += 1

        # Sort by bucket (descending - most recent first)
        sorted_buckets = sorted(
            bucket_counts.items(),
            key=lambda x: x[0],
            reverse=True
        )

        facet_values = [
            FacetValue(value=bucket, count=count)
            for bucket, count in sorted_buckets
        ]

        # Add unknown bucket at the end if there are documents with missing dates
        if unknown_count > 0:
            facet_values.append(
                FacetValue(value="unknown", count=unknown_count, label="日期未知")
            )

        return FacetResult(
            field_name=field_name,
            display_name=config['display_name'],
            facet_type="date_range",
            values=facet_values
        )

    def _build_numeric_range_facet(self,
                                  field_name: str,
                                  documents: List[Dict[str, Any]],
                                  config: Dict[str, Any]) -> FacetResult:
        """
        Build a numeric range facet with bucketing.

        Args:
            field_name: Field to build facet for
            documents: List of documents
            config: Facet configuration with 'range_buckets' key

        Returns:
            FacetResult with numeric range distribution

        Complexity:
            Time: O(N × B) where N is docs, B is buckets
            Space: O(B)

        Example:
            range_buckets = [
                (0, 10, "0-10"),
                (10, 100, "10-100"),
                (100, float('inf'), "100+")
            ]
        """
        range_buckets = config.get('range_buckets', [])
        if not range_buckets:
            self.logger.warning(f"No range_buckets configured for {field_name}")
            return FacetResult(
                field_name=field_name,
                display_name=config['display_name'],
                facet_type="numeric_range",
                values=[]
            )

        bucket_counts: Dict[str, int] = defaultdict(int)

        for doc in documents:
            value = doc.get(field_name)
            if value is None:
                continue

            try:
                numeric_value = float(value)

                # Find matching bucket
                for min_val, max_val, label in range_buckets:
                    if min_val <= numeric_value < max_val:
                        bucket_counts[label] += 1
                        break
            except (ValueError, TypeError) as e:
                self.logger.warning(f"Failed to parse numeric value '{value}': {e}")
                continue

        # Preserve bucket order
        facet_values = [
            FacetValue(value=label, count=bucket_counts.get(label, 0))
            for _, _, label in range_buckets
        ]

        # Remove empty buckets
        facet_values = [fv for fv in facet_values if fv.count > 0]

        return FacetResult(
            field_name=field_name,
            display_name=config['display_name'],
            facet_type="numeric_range",
            values=facet_values
        )

    def filter_documents(self,
                        documents: List[Dict[str, Any]],
                        filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Filter documents based on facet filter conditions.

        Args:
            documents: List of documents to filter
            filters: Dictionary of field_name -> filter_value(s)
                    For term facets: single value or list of values
                    For range facets: (min, max) tuple

        Returns:
            Filtered list of documents

        Complexity:
            Time: O(N × F) where N is docs, F is filters
            Space: O(M) where M is matching documents

        Example:
            >>> filters = {
            ...     "source": ["CNA", "UDN"],  # Multi-select
            ...     "category": "politics",     # Single select
            ...     "pub_date": ("2024-11-01", "2024-11-30")  # Range
            ... }
            >>> filtered = engine.filter_documents(docs, filters)
        """
        if not filters:
            return documents

        filtered = []

        for doc in documents:
            match = True

            for field_name, filter_value in filters.items():
                doc_value = doc.get(field_name)

                if doc_value is None:
                    match = False
                    break

                # Handle different filter types
                if isinstance(filter_value, (list, set)):
                    # Multi-select: doc value must be in filter values
                    if isinstance(doc_value, list):
                        # Doc has multiple values, check if any match
                        if not any(v in filter_value for v in doc_value):
                            match = False
                            break
                    else:
                        if str(doc_value) not in [str(v) for v in filter_value]:
                            match = False
                            break

                elif isinstance(filter_value, tuple) and len(filter_value) == 2:
                    # Range filter: doc value must be within range
                    min_val, max_val = filter_value
                    try:
                        if not (min_val <= str(doc_value) <= max_val):
                            match = False
                            break
                    except TypeError:
                        match = False
                        break

                else:
                    # Single value: exact match
                    if str(doc_value) != str(filter_value):
                        match = False
                        break

            if match:
                filtered.append(doc)

        return filtered
