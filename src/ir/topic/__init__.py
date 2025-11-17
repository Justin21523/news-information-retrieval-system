"""
Topic Modeling Module

This module provides topic modeling algorithms for Traditional Chinese documents,
including state-of-the-art neural and probabilistic methods.

Available Methods:
    - BERTopic: BERT embeddings + UMAP + HDBSCAN (2020-2025 SOTA)
    - LDA: Latent Dirichlet Allocation (2003 classic, high interpretability)

Key Features:
    - Multilingual support (Traditional Chinese optimized)
    - Dynamic topic modeling
    - Topic visualization
    - Coherence evaluation
    - Model persistence

Reference:
    Grootendorst, M. (2022). "BERTopic: Neural topic modeling with a class-based
        TF-IDF procedure". arXiv preprint arXiv:2203.05794.
    Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). "Latent Dirichlet Allocation".
        Journal of Machine Learning Research, 3, 993-1022.

Author: Information Retrieval System
License: Educational Use
"""

from .bertopic_model import BERTopicModel
from .lda_model import LDAModel

__all__ = [
    'BERTopicModel',
    'LDAModel',
]

__version__ = '0.1.0'
