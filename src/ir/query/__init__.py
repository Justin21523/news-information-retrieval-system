"""
Query Module for Advanced Search

This module provides query parsing and building capabilities for
advanced metadata-based search queries.

Key Components:
    - QueryParser: Parse library-style query syntax
    - QueryExecutor: Execute structured queries against field indexes
    - QueryNode: Query tree node representation
    - Operator: Query operators (AND, OR, NOT, FIELD, RANGE)

Author: Information Retrieval System
Date: 2025-11-17
"""

from .query_parser import QueryParser, QueryNode, Operator, parse_query
from .query_executor import QueryExecutor, SearchResult, execute_query

__all__ = [
    'QueryParser',
    'QueryNode',
    'Operator',
    'parse_query',
    'QueryExecutor',
    'SearchResult',
    'execute_query'
]
