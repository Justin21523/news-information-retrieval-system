"""
Language Model Module

This module provides statistical language modeling including N-gram models
with smoothing and collocation extraction.
"""

from .ngram import NGramModel
from .collocation import CollocationExtractor, CollocationScore

__all__ = [
    'NGramModel',
    'CollocationExtractor',
    'CollocationScore',
]
