"""
Recommendation System Module

This module provides content-based, collaborative filtering, and hybrid
recommendation algorithms for document recommendation.

Available Classes:
    - ContentBasedRecommender: Content similarity recommendations
    - CollaborativeFilteringRecommender: User/Item CF and Matrix Factorization
    - HybridRecommender: Combines multiple strategies

Examples:
    >>> from ir.recommendation import ContentBasedRecommender, CollaborativeFilteringRecommender, HybridRecommender
    >>>
    >>> # Content-based
    >>> content_rec = ContentBasedRecommender(documents)
    >>> content_rec.build_tfidf_vectors(vsm)
    >>> recs = content_rec.recommend_similar(doc_id=5, top_k=10)
    >>>
    >>> # Collaborative Filtering
    >>> cf_rec = CollaborativeFilteringRecommender(n_users=100, n_items=1000)
    >>> cf_rec.load_interactions(interactions)
    >>> cf_rec.compute_item_similarity()
    >>> recs = cf_rec.recommend_item_based(user_id=0, top_k=10)
    >>>
    >>> # Hybrid
    >>> hybrid = HybridRecommender(content_rec, cf_rec, fusion_method='weighted')
    >>> recs = hybrid.recommend(user_id=0, top_k=10)

Author: Information Retrieval System
License: Educational Use
"""

from .content_based import ContentBasedRecommender, RecommendationResult
from .collaborative_filtering import (
    CollaborativeFilteringRecommender,
    CFRecommendation
)
from .hybrid_recommender import HybridRecommender, HybridRecommendation

__all__ = [
    'ContentBasedRecommender',
    'RecommendationResult',
    'CollaborativeFilteringRecommender',
    'CFRecommendation',
    'HybridRecommender',
    'HybridRecommendation',
]
