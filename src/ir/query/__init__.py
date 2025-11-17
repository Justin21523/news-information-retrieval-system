"""
Query Module for Advanced Search

This module provides query parsing and building capabilities for
advanced metadata-based search queries.

Key Components:
    - QueryParser: Parse library-style query syntax
    - QueryBuilder: Programmatic query construction
    - QueryExecutor: Execute structured queries against field indexes

Author: Information Retrieval System
Date: 2025-11-17
"""

from .query_parser import QueryParser, QueryNode, Operator

__all__ = ['QueryParser', 'QueryNode', 'Operator']
