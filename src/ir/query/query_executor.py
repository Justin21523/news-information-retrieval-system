"""
Query Executor for Advanced Metadata Search

This module executes parsed query trees against field indexes,
supporting complex boolean queries with field-specific searches.

Key Features:
    - Execute QueryNode trees from QueryParser
    - Field-specific search via FieldIndexer
    - Boolean operations (AND, OR, NOT)
    - Range queries (date ranges, numeric ranges)
    - Efficient set operations for result combination

Author: Information Retrieval System
Date: 2025-11-17
"""

import logging
from typing import Set, List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from .query_parser import QueryNode, Operator
from ..index.field_indexer import FieldIndexer


@dataclass
class SearchResult:
    """
    Search result with document ID and metadata.

    Attributes:
        doc_id: Document ID
        score: Relevance score (default 1.0 for boolean queries)
        matched_fields: Fields that matched the query
        snippet: Text snippet showing match context
    """
    doc_id: int
    score: float = 1.0
    matched_fields: List[str] = None
    snippet: str = None

    def __post_init__(self):
        if self.matched_fields is None:
            self.matched_fields = []


class QueryExecutor:
    """
    Execute parsed queries against field indexes.

    This class takes QueryNode trees from QueryParser and executes them
    against FieldIndexer to retrieve matching documents.

    Complexity:
        Time: O(N) for simple queries, O(N log N) for complex boolean queries
              where N = number of matching documents
        Space: O(N) for result sets
    """

    def __init__(self, field_indexer: FieldIndexer, documents: List[Dict[str, Any]] = None):
        """
        Initialize QueryExecutor.

        Args:
            field_indexer: FieldIndexer instance for searching
            documents: Optional list of document dictionaries for snippet generation

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        self.logger = logging.getLogger(__name__)
        self.field_indexer = field_indexer
        self.documents = documents or []

    def execute(self, query_node: QueryNode, top_k: int = None) -> List[SearchResult]:
        """
        Execute a query and return matching documents.

        Args:
            query_node: Root QueryNode from parser
            top_k: Maximum number of results to return (None = all)

        Returns:
            List of SearchResult objects sorted by relevance

        Complexity:
            Time: O(N log N) where N = number of matches (for sorting)
            Space: O(N)

        Example:
            >>> from query_parser import parse_query
            >>> node = parse_query("title:台灣 AND category:政治")
            >>> executor = QueryExecutor(field_indexer)
            >>> results = executor.execute(node, top_k=20)
            >>> len(results)
            15
        """
        self.logger.debug(f"Executing query: {query_node}")

        # Execute query to get document IDs
        doc_ids = self._execute_node(query_node)

        # Convert to SearchResult objects
        results = [
            SearchResult(
                doc_id=doc_id,
                score=1.0,  # Boolean queries have uniform score
                matched_fields=self._get_matched_fields(doc_id, query_node)
            )
            for doc_id in doc_ids
        ]

        # Sort by doc_id for now (can be enhanced with ranking)
        results.sort(key=lambda r: r.doc_id)

        # Apply top_k limit
        if top_k:
            results = results[:top_k]

        self.logger.info(f"Query returned {len(results)} results")
        return results

    def _execute_node(self, node: QueryNode) -> Set[int]:
        """
        Recursively execute a query node.

        Args:
            node: QueryNode to execute

        Returns:
            Set of matching document IDs

        Complexity:
            Time: O(N) for simple queries, O(N * M) for complex queries
                  where N = matches, M = number of operations
        """
        if node.operator == Operator.FIELD:
            # Simple field query
            return self._execute_field_query(node)

        elif node.operator == Operator.RANGE:
            # Range query
            return self._execute_range_query(node)

        elif node.operator == Operator.AND:
            # AND: Intersection of children
            if not node.children:
                return set()

            result = self._execute_node(node.children[0])
            for child in node.children[1:]:
                result &= self._execute_node(child)

            return result

        elif node.operator == Operator.OR:
            # OR: Union of children
            if not node.children:
                return set()

            result = set()
            for child in node.children:
                result |= self._execute_node(child)

            return result

        elif node.operator == Operator.NOT:
            # NOT: All documents minus child results
            if not node.children:
                return set()

            all_docs = set(range(self.field_indexer.doc_count))
            not_docs = self._execute_node(node.children[0])

            return all_docs - not_docs

        else:
            self.logger.warning(f"Unknown operator: {node.operator}")
            return set()

    def _execute_field_query(self, node: QueryNode) -> Set[int]:
        """
        Execute a simple field query.

        Args:
            node: QueryNode with FIELD operator

        Returns:
            Set of matching document IDs

        Complexity:
            Time: O(P) where P = postings list size
        """
        field = node.field
        value = node.value

        if not field or not value:
            return set()

        # Use FieldIndexer to search
        result = self.field_indexer.search_field(field, value)

        self.logger.debug(f"Field query {field}:{value} returned {len(result)} docs")
        return result

    def _execute_range_query(self, node: QueryNode) -> Set[int]:
        """
        Execute a range query (e.g., date ranges).

        Args:
            node: QueryNode with RANGE operator

        Returns:
            Set of matching document IDs

        Complexity:
            Time: O(N) where N = documents in index
        """
        field = node.field
        start = node.range_start
        end = node.range_end

        if not field or not start or not end:
            return set()

        # Use FieldIndexer's date range search
        if field in self.field_indexer.date_indexes:
            result = self.field_indexer.search_date_range(field, start, end)
            self.logger.debug(f"Range query {field}:[{start} TO {end}] returned {len(result)} docs")
            return result

        self.logger.warning(f"Range query not supported for field: {field}")
        return set()

    def _get_matched_fields(self, doc_id: int, query_node: QueryNode) -> List[str]:
        """
        Get list of fields that matched for a document.

        Args:
            doc_id: Document ID
            query_node: Query node to analyze

        Returns:
            List of field names that matched

        Complexity:
            Time: O(F) where F = number of fields in query
        """
        matched = set()

        def collect_fields(node):
            """Recursively collect matched fields."""
            if node.operator == Operator.FIELD:
                # Check if this field matches the document
                if doc_id in self.field_indexer.search_field(node.field, node.value):
                    matched.add(node.field)

            elif node.operator == Operator.RANGE:
                # Check if document is in range
                if doc_id in self._execute_range_query(node):
                    matched.add(node.field)

            elif node.children:
                for child in node.children:
                    collect_fields(child)

        collect_fields(query_node)
        return list(matched)

    def execute_structured_query(
        self,
        conditions: List[Dict[str, str]],
        logic: str = "AND",
        top_k: int = None
    ) -> List[SearchResult]:
        """
        Execute a structured query from JSON format.

        This is a convenience method for API endpoints that receive
        queries as structured JSON rather than query strings.

        Args:
            conditions: List of condition dictionaries with:
                - field: Field name
                - operator: Operator (contains, equals, between, etc.)
                - value: Query value or [start, end] for ranges
            logic: How to combine conditions (AND/OR)
            top_k: Maximum number of results

        Returns:
            List of SearchResult objects

        Complexity:
            Time: O(N log N) where N = number of matches
            Space: O(N)

        Example:
            >>> conditions = [
            ...     {"field": "title", "operator": "contains", "value": "台灣"},
            ...     {"field": "category", "operator": "equals", "value": "aipl"}
            ... ]
            >>> results = executor.execute_structured_query(conditions, logic="AND")
        """
        self.logger.debug(f"Executing structured query: {len(conditions)} conditions, logic={logic}")

        if not conditions:
            return []

        # Convert structured conditions to query nodes
        nodes = []
        for cond in conditions:
            node = self._condition_to_node(cond)
            if node:
                nodes.append(node)

        if not nodes:
            return []

        # Combine with specified logic
        if len(nodes) == 1:
            root_node = nodes[0]
        else:
            operator = Operator.AND if logic.upper() == "AND" else Operator.OR
            root_node = QueryNode(operator=operator, children=nodes)

        # Execute the combined query
        return self.execute(root_node, top_k=top_k)

    def _condition_to_node(self, condition: Dict[str, str]) -> Optional[QueryNode]:
        """
        Convert a condition dictionary to a QueryNode.

        Args:
            condition: Condition dictionary with field, operator, value

        Returns:
            QueryNode or None if invalid

        Supported operators:
            - contains: Field contains value (tokenized search)
            - equals: Field exactly equals value
            - starts_with: Field starts with value
            - between: Field value between [start, end] (for dates)
        """
        field = condition.get('field')
        operator = condition.get('operator', 'contains')
        value = condition.get('value')

        if not field or value is None:
            return None

        # Handle range queries (between operator)
        if operator == 'between':
            if isinstance(value, list) and len(value) == 2:
                return QueryNode(
                    operator=Operator.RANGE,
                    field=field,
                    range_start=value[0],
                    range_end=value[1]
                )
            return None

        # Handle regular field queries
        # For now, treat all as simple field queries
        # FieldIndexer will handle tokenization internally
        return QueryNode(
            operator=Operator.FIELD,
            field=field,
            value=str(value)
        )


# Convenience function for quick execution
def execute_query(
    query_str: str,
    field_indexer: FieldIndexer,
    documents: List[Dict[str, Any]] = None,
    top_k: int = None
) -> List[SearchResult]:
    """
    Convenience function to parse and execute a query string.

    Args:
        query_str: Query string to parse and execute
        field_indexer: FieldIndexer instance
        documents: Optional document list
        top_k: Maximum results

    Returns:
        List of SearchResult objects

    Example:
        >>> results = execute_query(
        ...     "title:台灣 AND category:政治",
        ...     field_indexer,
        ...     top_k=20
        ... )
    """
    from .query_parser import parse_query

    query_node = parse_query(query_str)
    executor = QueryExecutor(field_indexer, documents)
    return executor.execute(query_node, top_k=top_k)
