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


def _unavailable_class(name, package):
    """Create a placeholder for unavailable optional extractors."""

    class UnavailableExtractor:
        def __init__(self, *args, **kwargs):
            raise ImportError(f"{name} requires optional dependency: {package}")

    UnavailableExtractor.__name__ = name
    return UnavailableExtractor


try:
    from .yake_extractor import YAKEExtractor
except Exception:
    YAKEExtractor = _unavailable_class("YAKEExtractor", "yake")

try:
    from .rake_extractor import RAKEExtractor
except Exception:
    RAKEExtractor = _unavailable_class("RAKEExtractor", "rake-nltk")

try:
    from .keybert_extractor import KeyBERTExtractor
except Exception:
    KeyBERTExtractor = _unavailable_class("KeyBERTExtractor", "keybert")

__all__ = [
    'TextRankExtractor',
    'YAKEExtractor',
    'RAKEExtractor',
    'KeyBERTExtractor',
]

__version__ = '0.9.0'
