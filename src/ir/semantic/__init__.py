"""
Semantic Retrieval Module

This module provides dense retrieval using neural embeddings (BERT, Sentence-BERT).
"""

from .bert_retrieval import BERTRetrieval, SemanticResult

__all__ = [
    'BERTRetrieval',
    'SemanticResult',
]
