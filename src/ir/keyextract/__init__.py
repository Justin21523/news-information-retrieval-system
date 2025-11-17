"""
Keyword Extraction Module

This module provides various keyword extraction algorithms for
Traditional Chinese and English text.

Available Methods:
    - TextRank: Graph-based ranking (2004 → 2025 improvements)
    - YAKE: Statistical features (2018)
    - RAKE: Rapid automatic extraction (2010)
    - KeyBERT: BERT embeddings + MMR (2020)

Reference:
    Mihalcea & Tarau (2004). "TextRank: Bringing Order into Text"
    Chen et al. (2025). "An Improved Chinese Keyword Extraction Algorithm
        Based on Complex Networks"
    Campos et al. (2020). "YAKE! Keyword Extraction from Single Documents"
    Rose et al. (2010). "Automatic Keyword Extraction from Individual Documents"

Author: Information Retrieval System
License: Educational Use
"""

from .textrank import TextRankExtractor
from .yake_extractor import YAKEExtractor
from .rake_extractor import RAKEExtractor
from .keybert_extractor import KeyBERTExtractor

__all__ = [
    'TextRankExtractor',
    'YAKEExtractor',
    'RAKEExtractor',
    'KeyBERTExtractor',
]

__version__ = '0.9.0'
