"""
Pattern Mining Module

This module provides algorithms for pattern mining and term extraction
from Chinese text using statistical methods.

Available Classes:
    - PATTree: PAT-tree (Patricia Tree) for pattern mining
    - Pattern: Extracted pattern with statistics

Key Features:
    - Suffix tree construction
    - Frequent pattern mining
    - Mutual Information (MI) calculation
    - Multi-word term extraction

Author: Information Retrieval System
License: Educational Use
"""

from .pat_tree import PATTree, Pattern, PATNode

__all__ = [
    'PATTree',
    'Pattern',
    'PATNode',
]

__version__ = '0.1.0'
