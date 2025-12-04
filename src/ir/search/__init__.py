"""
Search Module for Information Retrieval System

This module provides unified search interfaces integrating all retrieval methods.

Classes:
    - UnifiedSearchEngine: Main search engine class
    - QueryMode: Query processing modes
    - RankingModel: Ranking models
    - UnifiedSearchResult: Search result data class

Author: Information Retrieval System
"""

from .unified_search import (
    UnifiedSearchEngine,
    QueryMode,
    RankingModel,
    UnifiedSearchResult
)

__all__ = [
    'UnifiedSearchEngine',
    'QueryMode',
    'RankingModel',
    'UnifiedSearchResult'
]
