"""
Field-based Indexer for Metadata Search

This module implements field-specific indexing for structured document metadata,
enabling library-style search queries like:
    - title:台灣 AND category:政治
    - author:記者 OR source:中央社
    - date:[2025-11-01 TO 2025-11-13]
    - tags:(AI OR 人工智慧)

Key Features:
    - Independent inverted index for each metadata field
    - Support for multi-value fields (tags)
    - Date range queries
    - Field-specific tokenization strategies
    - Efficient field-based retrieval

Supported Fields:
    - title: Document title (tokenized)
    - content: Full text content (tokenized)
    - author: Author name (exact + tokenized)
    - category: Category code (exact match)
    - category_name: Category name in Chinese (tokenized)
    - tags: List of tags (multi-value, tokenized)
    - published_date: Publication date (range queries)
    - source: Source media (exact match)
    - url: Document URL (exact match)

Author: Information Retrieval System
License: Educational Use
"""

import re
import logging
from typing import Dict, List, Set, Tuple, Optional, Callable, Any
from collections import defaultdict
from datetime import datetime


class FieldIndexer:
    """
    Field-based indexer for structured metadata search.

    Maintains separate inverted indexes for each metadata field,
    enabling precise field-specific queries in library-style IR systems.

    Implementation Overview:
        - For each supported field, build an independent index:
            field -> term -> set(doc_id)
        - For date fields, store a normalized YYYY-MM-DD string per document:
            field -> doc_id -> "YYYY-MM-DD"
        - Query-time operations are set-based (AND/OR/NOT at a higher layer):
            - Single-term lookup: O(1) average
            - Multi-term: union/intersection across sets

    Attributes:
        field_indexes: Dict mapping field_name -> InvertedIndex
        tokenizer: Function for tokenizing text fields
        doc_count: Total number of indexed documents
        supported_fields: Set of fields that can be indexed
    """

    # Define which fields should be tokenized vs exact match
    TEXT_FIELDS = {'title', 'content', 'category_name', 'author'}
    EXACT_FIELDS = {'category', 'source', 'url'}
    MULTI_VALUE_FIELDS = {'tags'}
    DATE_FIELDS = {'published_date'}

    def __init__(self, tokenizer: Optional[Callable[[str], List[str]]] = None):
        """
        Initialize FieldIndexer.

        Args:
            tokenizer: Custom tokenization function for text fields.
                      If None, uses default simple tokenizer.

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        self.logger = logging.getLogger(__name__)
        self.tokenizer = tokenizer or self._default_tokenizer

        # Field-specific inverted indexes
        # Structure: {field_name: {term: set(doc_ids)}}
        self.field_indexes: Dict[str, Dict[str, Set[int]]] = {}

        # Date field for range queries
        # Structure: {field_name: {doc_id: date_value}}
        self.date_indexes: Dict[str, Dict[int, str]] = {}

        # Document count
        self.doc_count: int = 0

        # Supported fields
        self.supported_fields = (
            self.TEXT_FIELDS |
            self.EXACT_FIELDS |
            self.MULTI_VALUE_FIELDS |
            self.DATE_FIELDS
        )

        self.logger.info("FieldIndexer initialized")

    def _default_tokenizer(self, text: str) -> List[str]:
        """
        Default tokenizer for text fields.

        Args:
            text: Input text

        Returns:
            List of tokens

        Complexity:
            Time: O(n) where n = text length
        """
        if not text:
            return []
        # Simple regex-based tokenization
        # For Chinese text, this will split by character
        # In practice, should use CKIP tokenizer
        tokens = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z0-9]+', text.lower())
        return tokens

    def build(self, documents: List[Dict[str, Any]]) -> None:
        """
        Build field indexes from document collection.

        Args:
            documents: List of document dictionaries with metadata fields
                       If a document provides an integer `doc_id`, that id is
                       used as the index key (useful when aligning metadata
                       indexes with an external content index).

        Complexity:
            Time: O(N * F * T) where:
                  N = number of documents
                  F = number of fields
                  T = average tokens per field
            Space: O(N * F * T)

        Example:
            >>> docs = [
            ...     {'title': '台灣新聞', 'category': 'politics', 'tags': ['台灣', '政治']},
            ...     {'title': '經濟報導', 'category': 'economy', 'tags': ['經濟']}
            ... ]
            >>> indexer = FieldIndexer()
            >>> indexer.build(docs)
        """
        self.logger.info(f"Building field indexes for {len(documents)} documents...")

        # Determine the doc_id to use for each input document.
        #
        # By default, doc_ids are assigned by enumeration order (0..N-1).
        # If the caller provides explicit doc_id values, we index using them
        # so field-based search can share the same doc_id space as other
        # retrieval models (e.g., an inverted index for content ranking).
        doc_ids: List[int] = []
        for i, doc in enumerate(documents):
            explicit_doc_id = doc.get('doc_id')
            doc_id = explicit_doc_id if isinstance(explicit_doc_id, int) else i
            doc_ids.append(doc_id)

        # doc_count is used as the "universe size" by NOT queries at a higher
        # layer (QueryExecutor). If doc_ids are contiguous 0..N-1, this equals N.
        self.doc_count = max(doc_ids) + 1 if doc_ids else 0

        # Initialize indexes for all supported fields
        for field in self.supported_fields:
            if field in self.DATE_FIELDS:
                self.date_indexes[field] = {}
            else:
                self.field_indexes[field] = defaultdict(set)

        # Index each document
        for count, (doc_id, doc) in enumerate(zip(doc_ids, documents), start=1):
            self._index_document(doc_id, doc)

            if count % 100 == 0:
                self.logger.info(f"Indexed {count}/{len(documents)} documents")

        # Convert defaultdicts to regular dicts
        for field in self.field_indexes:
            self.field_indexes[field] = dict(self.field_indexes[field])

        # Log statistics
        stats = self._get_index_stats()
        self.logger.info(
            f"Field indexes built: {stats['total_fields']} fields, "
            f"{stats['total_terms']} total terms, "
            f"{stats['total_postings']} total postings"
        )

    def _index_document(self, doc_id: int, doc: Dict[str, Any]) -> None:
        """
        Index a single document across all fields.

        Args:
            doc_id: Document ID
            doc: Document dictionary with metadata fields

        Complexity:
            Time: O(F * T) where F = fields, T = avg tokens
        """
        for field, value in doc.items():
            if field not in self.supported_fields:
                continue

            if value is None or value == '':
                continue

            # Handle different field types
            if field in self.DATE_FIELDS:
                self._index_date_field(doc_id, field, value)
            elif field in self.EXACT_FIELDS:
                self._index_exact_field(doc_id, field, value)
            elif field in self.MULTI_VALUE_FIELDS:
                self._index_multi_value_field(doc_id, field, value)
            elif field in self.TEXT_FIELDS:
                self._index_text_field(doc_id, field, value)

    def _index_text_field(self, doc_id: int, field: str, text: str) -> None:
        """Index a text field with tokenization."""
        tokens = self.tokenizer(text)
        for token in tokens:
            if token:  # Skip empty tokens
                self.field_indexes[field][token].add(doc_id)

    def _index_exact_field(self, doc_id: int, field: str, value: str) -> None:
        """Index an exact match field (no tokenization)."""
        if value:
            # Store lowercase for case-insensitive matching
            self.field_indexes[field][value.lower()].add(doc_id)

    def _index_multi_value_field(self, doc_id: int, field: str, values: List[str]) -> None:
        """Index a multi-value field (like tags)."""
        if not isinstance(values, list):
            values = [values]

        for value in values:
            if not value:
                continue
            # Tokenize each value
            tokens = self.tokenizer(str(value))
            for token in tokens:
                if token:
                    self.field_indexes[field][token].add(doc_id)

    def _index_date_field(self, doc_id: int, field: str, date_value: str) -> None:
        """Index a date field for range queries."""
        # Store date string in YYYY-MM-DD format so lexicographic ordering
        # matches chronological ordering (e.g., "2025-11-09" < "2025-11-10").
        try:
            # Normalize date format
            if isinstance(date_value, datetime):
                date_str = date_value.strftime('%Y-%m-%d')
            else:
                # Assume string in YYYY-MM-DD format
                date_str = str(date_value)[:10]
            self.date_indexes[field][doc_id] = date_str
        except Exception as e:
            self.logger.warning(f"Invalid date format for doc {doc_id}: {date_value}")

    def search_field(self, field: str, term: str) -> Set[int]:
        """
        Search for documents matching a term in a specific field.

        Args:
            field: Field name to search
            term: Search term

        Returns:
            Set of document IDs matching the term

        Complexity:
            Time: O(1) average for exact match, O(T) for tokenized
            Space: O(k) where k = number of matching documents

        Examples:
            >>> indexer.search_field('category', 'politics')
            {0, 5, 12, 18}

            >>> indexer.search_field('title', '台灣')
            {0, 1, 3, 7, 11}
        """
        if field not in self.field_indexes:
            self.logger.warning(f"Field '{field}' not indexed")
            return set()

        # Normalize term. For exact fields, we use lowercase to match the
        # case-insensitive storage performed at index time.
        if field in self.EXACT_FIELDS:
            term = term.lower()
        else:
            # For text fields, might need tokenization
            # But for single term search, use as-is
            term = term.lower()

        return self.field_indexes[field].get(term, set()).copy()

    def search_date_range(self, field: str, start_date: str, end_date: str) -> Set[int]:
        """
        Search for documents within a date range.

        Args:
            field: Date field name
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            Set of document IDs within the date range

        Complexity:
            Time: O(N) where N = number of documents
            Space: O(k) where k = matching documents

        Examples:
            >>> indexer.search_date_range('published_date', '2025-11-01', '2025-11-13')
            {0, 1, 2, 5, 8, 11, 15}
        """
        if field not in self.date_indexes:
            self.logger.warning(f"Date field '{field}' not indexed")
            return set()

        matching_docs = set()
        for doc_id, date_value in self.date_indexes[field].items():
            if start_date <= date_value <= end_date:
                matching_docs.add(doc_id)

        return matching_docs

    def search_multi_terms(self, field: str, terms: List[str],
                          operator: str = 'OR') -> Set[int]:
        """
        Search for multiple terms in a field.

        Args:
            field: Field name
            terms: List of search terms
            operator: 'OR' or 'AND'

        Returns:
            Set of matching document IDs

        Complexity:
            Time: O(T * k) where T = number of terms, k = avg postings
            Space: O(N) worst case

        Examples:
            >>> indexer.search_multi_terms('tags', ['AI', '人工智慧'], 'OR')
            {2, 5, 8, 12, 15, 18}

            >>> indexer.search_multi_terms('tags', ['台灣', '政治'], 'AND')
            {0, 3, 7}
        """
        if not terms:
            return set()

        result_sets = [self.search_field(field, term) for term in terms]

        if operator == 'AND':
            # Intersection of all sets
            result = result_sets[0] if result_sets else set()
            for s in result_sets[1:]:
                result &= s
            return result
        else:  # OR
            # Union of all sets
            result = set()
            for s in result_sets:
                result |= s
            return result

    def get_field_vocabulary(self, field: str) -> Set[str]:
        """
        Get all unique terms in a field.

        Args:
            field: Field name

        Returns:
            Set of unique terms

        Complexity:
            Time: O(1) to O(V) where V = vocabulary size
            Space: O(V)
        """
        if field in self.field_indexes:
            return set(self.field_indexes[field].keys())
        return set()

    def get_document_field_value(self, doc_id: int, field: str) -> Optional[Any]:
        """
        Get the original field value for a document (for date fields).

        Args:
            doc_id: Document ID
            field: Field name

        Returns:
            Field value or None

        Complexity:
            Time: O(1)
        """
        if field in self.date_indexes:
            return self.date_indexes[field].get(doc_id)
        return None

    def _get_index_stats(self) -> Dict[str, Any]:
        """Get indexing statistics."""
        stats = {
            'total_fields': len(self.field_indexes) + len(self.date_indexes),
            'total_terms': 0,
            'total_postings': 0,
            'field_stats': {}
        }

        for field, index in self.field_indexes.items():
            field_terms = len(index)
            field_postings = sum(len(docs) for docs in index.values())
            stats['total_terms'] += field_terms
            stats['total_postings'] += field_postings
            stats['field_stats'][field] = {
                'terms': field_terms,
                'postings': field_postings
            }

        for field in self.date_indexes:
            stats['field_stats'][field] = {
                'documents': len(self.date_indexes[field])
            }

        return stats

    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive index statistics.

        Returns:
            Dictionary with statistics for each field

        Examples:
            >>> stats = indexer.get_stats()
            >>> print(stats)
            {
                'doc_count': 121,
                'total_fields': 9,
                'total_terms': 15234,
                'field_stats': {
                    'title': {'terms': 2341, 'postings': 3456},
                    ...
                }
            }
        """
        stats = self._get_index_stats()
        stats['doc_count'] = self.doc_count
        return stats


def demo():
    """Demonstration of FieldIndexer."""
    print("=" * 60)
    print("FieldIndexer Demo")
    print("=" * 60)

    # Sample documents
    documents = [
        {
            'title': '台灣經濟發展報導',
            'category': 'economy',
            'category_name': '財經',
            'author': '記者張三',
            'tags': ['台灣', '經濟', '發展'],
            'published_date': '2025-11-10',
            'source': '中央社'
        },
        {
            'title': '美國科技產業分析',
            'category': 'technology',
            'category_name': '科技',
            'author': '記者李四',
            'tags': ['美國', '科技', 'AI'],
            'published_date': '2025-11-11',
            'source': '中央社'
        },
        {
            'title': '台灣政治新聞',
            'category': 'politics',
            'category_name': '政治',
            'author': '記者王五',
            'tags': ['台灣', '政治'],
            'published_date': '2025-11-12',
            'source': '中央社'
        }
    ]

    # Build index
    indexer = FieldIndexer()
    indexer.build(documents)

    print("\n1. Field Index Statistics:")
    stats = indexer.get_stats()
    for field, field_stats in stats['field_stats'].items():
        print(f"   {field}: {field_stats}")

    print("\n2. Search by category:")
    results = indexer.search_field('category', 'economy')
    print(f"   category:economy -> {results}")

    print("\n3. Search by title term:")
    results = indexer.search_field('title', '台灣')
    print(f"   title:台灣 -> {results}")

    print("\n4. Search by tags (OR):")
    results = indexer.search_multi_terms('tags', ['台灣', '美國'], 'OR')
    print(f"   tags:(台灣 OR 美國) -> {results}")

    print("\n5. Search by tags (AND):")
    results = indexer.search_multi_terms('tags', ['台灣', '政治'], 'AND')
    print(f"   tags:(台灣 AND 政治) -> {results}")

    print("\n6. Date range search:")
    results = indexer.search_date_range('published_date', '2025-11-10', '2025-11-11')
    print(f"   date:[2025-11-10 TO 2025-11-11] -> {results}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    demo()
