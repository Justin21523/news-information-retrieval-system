"""
Hybrid Recommendation System

This module implements a hybrid recommender that combines multiple
recommendation strategies for improved accuracy and coverage.

Hybrid Strategies:
    1. Weighted: Linear combination of scores
    2. Switching: Choose strategy based on context
    3. Cascade: Refine results using multiple strategies
    4. Feature Combination: Merge all features into one model
    5. Meta-level: Use output of one as input to another

Key Features:
    - Content-based + Collaborative Filtering fusion
    - Cold start handling (new users/items)
    - Diversity-aware recommendations
    - Explanation generation
    - A/B testing support

Complexity:
    - Time: O(max(content_time, cf_time))
    - Space: O(N + recommendations)

References:
    Burke, R. (2002). "Hybrid recommender systems: Survey and experiments".
        User modeling and user-adapted interaction, 12(4), 331-370.

Author: Information Retrieval System
License: Educational Use
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
import logging
from pathlib import Path

# Import recommendation modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ir.recommendation.content_based import ContentBasedRecommender, RecommendationResult
from ir.recommendation.collaborative_filtering import (
    CollaborativeFilteringRecommender,
    CFRecommendation
)


@dataclass
class HybridRecommendation:
    """
    Hybrid recommendation result with multiple signals.

    Attributes:
        doc_id: Document ID
        score: Final hybrid score
        content_score: Content-based score
        cf_score: Collaborative filtering score
        popularity_score: Popularity score
        method: Fusion method used
        explanation: Human-readable explanation
    """
    doc_id: int
    score: float
    content_score: float
    cf_score: float
    popularity_score: float
    method: str
    explanation: str

    def __repr__(self):
        return f"HybridRec(doc={self.doc_id}, score={self.score:.3f}, method={self.method})"


class HybridRecommender:
    """
    Hybrid recommendation system.

    Combines content-based and collaborative filtering approaches
    with multiple fusion strategies.

    Attributes:
        content_recommender: ContentBasedRecommender instance
        cf_recommender: CollaborativeFilteringRecommender instance
        fusion_method: How to combine recommendations
        weights: Weights for different signals
    """

    def __init__(self,
                 content_recommender: Optional[ContentBasedRecommender] = None,
                 cf_recommender: Optional[CollaborativeFilteringRecommender] = None,
                 fusion_method: str = 'weighted',
                 content_weight: float = 0.5,
                 cf_weight: float = 0.4,
                 popularity_weight: float = 0.1):
        """
        Initialize hybrid recommender.

        Args:
            content_recommender: Content-based recommender
            cf_recommender: Collaborative filtering recommender
            fusion_method: 'weighted', 'cascade', 'switching'
            content_weight: Weight for content-based scores
            cf_weight: Weight for CF scores
            popularity_weight: Weight for popularity scores
        """
        self.logger = logging.getLogger(__name__)
        self.content_recommender = content_recommender
        self.cf_recommender = cf_recommender
        self.fusion_method = fusion_method

        # Normalize weights
        total = content_weight + cf_weight + popularity_weight
        self.content_weight = content_weight / total
        self.cf_weight = cf_weight / total
        self.popularity_weight = popularity_weight / total

        # Popularity statistics (lazy computation)
        self.item_popularity = None

        self.logger.info(
            f"HybridRecommender initialized: method={fusion_method}, "
            f"weights=[content={self.content_weight:.2f}, "
            f"cf={self.cf_weight:.2f}, popularity={self.popularity_weight:.2f}]"
        )

    # ========================================================================
    # Popularity Computation
    # ========================================================================

    def compute_popularity(self):
        """
        Compute item popularity from CF interaction data.

        Popularity = number of interactions / total users
        """
        if self.cf_recommender is None:
            self.logger.warning("No CF recommender available for popularity")
            return

        n_items = self.cf_recommender.n_items
        self.item_popularity = np.zeros(n_items)

        for i in range(n_items):
            profile = self.cf_recommender.get_item_profile(i)
            self.item_popularity[i] = profile['popularity']

        self.logger.info(
            f"Computed popularity for {n_items} items, "
            f"mean={self.item_popularity.mean():.4f}"
        )

    # ========================================================================
    # Weighted Fusion
    # ========================================================================

    def recommend_weighted(self,
                          user_id: Optional[int] = None,
                          doc_id: Optional[int] = None,
                          top_k: int = 10,
                          reading_history: Optional[List[int]] = None) -> List[HybridRecommendation]:
        """
        Weighted hybrid recommendation.

        Combines scores from all available methods using linear weights:
            score = w_c × content + w_cf × cf + w_p × popularity

        Args:
            user_id: User ID (for CF)
            doc_id: Document ID (for content-based similar items)
            top_k: Number of recommendations
            reading_history: User's reading history (for content-based personalization)

        Returns:
            List of HybridRecommendation objects

        Examples:
            >>> hybrid = HybridRecommender(content_rec, cf_rec)
            >>> # For existing user (has CF data)
            >>> recs = hybrid.recommend_weighted(user_id=5, top_k=10)
            >>> # For new user (cold start, use content only)
            >>> recs = hybrid.recommend_weighted(reading_history=[0,1,5], top_k=10)
        """
        candidate_scores = defaultdict(lambda: {'content': 0.0, 'cf': 0.0, 'popularity': 0.0})

        # 1. Content-based scores
        if self.content_recommender is not None:
            if doc_id is not None:
                # Similar items to a given document
                content_recs = self.content_recommender.recommend_similar(
                    doc_id=doc_id,
                    top_k=top_k * 3,  # Get more candidates
                    use_embeddings=False
                )
            elif reading_history:
                # Personalized based on reading history
                content_recs = self.content_recommender.recommend_personalized(
                    reading_history=reading_history,
                    top_k=top_k * 3
                )
            else:
                content_recs = []

            for rec in content_recs:
                candidate_scores[rec.doc_id]['content'] = rec.score

        # 2. Collaborative filtering scores
        if self.cf_recommender is not None and user_id is not None:
            # Try multiple CF methods and take the best
            cf_recs = []

            # Item-based CF (usually better for documents)
            try:
                if self.cf_recommender.item_similarity is not None:
                    cf_recs = self.cf_recommender.recommend_item_based(
                        user_id=user_id,
                        top_k=top_k * 3
                    )
            except:
                pass

            # Matrix factorization (if trained)
            if not cf_recs:
                try:
                    if self.cf_recommender.user_factors is not None:
                        cf_recs = self.cf_recommender.recommend_matrix_factorization(
                            user_id=user_id,
                            top_k=top_k * 3
                        )
                except:
                    pass

            for rec in cf_recs:
                candidate_scores[rec.doc_id]['cf'] = rec.score

        # 3. Popularity scores
        if self.item_popularity is not None:
            for doc_id in candidate_scores.keys():
                if doc_id < len(self.item_popularity):
                    candidate_scores[doc_id]['popularity'] = self.item_popularity[doc_id]

        # 4. Combine scores
        final_scores = []
        for doc_id, scores in candidate_scores.items():
            # Normalize scores to [0, 1]
            content_score = self._normalize_score(scores['content'])
            cf_score = self._normalize_score(scores['cf'])
            pop_score = scores['popularity']  # Already in [0, 1]

            # Weighted combination
            final_score = (
                self.content_weight * content_score +
                self.cf_weight * cf_score +
                self.popularity_weight * pop_score
            )

            # Generate explanation
            explanation = self._generate_explanation(
                content_score, cf_score, pop_score
            )

            final_scores.append(HybridRecommendation(
                doc_id=doc_id,
                score=final_score,
                content_score=content_score,
                cf_score=cf_score,
                popularity_score=pop_score,
                method='weighted',
                explanation=explanation
            ))

        # Sort by final score
        final_scores.sort(key=lambda x: x.score, reverse=True)

        self.logger.info(
            f"Weighted fusion: {len(final_scores[:top_k])} recommendations"
        )

        return final_scores[:top_k]

    # ========================================================================
    # Cascade Fusion
    # ========================================================================

    def recommend_cascade(self,
                         user_id: Optional[int] = None,
                         doc_id: Optional[int] = None,
                         top_k: int = 10,
                         reading_history: Optional[List[int]] = None) -> List[HybridRecommendation]:
        """
        Cascade hybrid recommendation.

        Uses multiple methods in sequence:
            1. Content-based generates candidates (high recall)
            2. CF re-ranks candidates (high precision)
            3. Popularity as tiebreaker

        Args:
            user_id: User ID
            doc_id: Document ID
            top_k: Number of recommendations
            reading_history: Reading history

        Returns:
            List of HybridRecommendation objects
        """
        # Stage 1: Content-based candidates
        if doc_id is not None:
            candidates = self.content_recommender.recommend_similar(
                doc_id=doc_id,
                top_k=top_k * 5,  # Large candidate pool
                use_embeddings=False
            )
        elif reading_history:
            candidates = self.content_recommender.recommend_personalized(
                reading_history=reading_history,
                top_k=top_k * 5
            )
        else:
            return []

        candidate_ids = [rec.doc_id for rec in candidates]
        candidate_content_scores = {rec.doc_id: rec.score for rec in candidates}

        # Stage 2: Re-rank with CF
        cf_scores = {}
        if self.cf_recommender is not None and user_id is not None:
            # Compute CF scores for candidates only
            for doc_id in candidate_ids:
                # Get item-based CF score if available
                if self.cf_recommender.item_similarity is not None:
                    # Find similar items to user's history
                    user_items = self.cf_recommender.interaction_matrix[user_id].nonzero()[1]
                    if len(user_items) > 0:
                        # Average similarity to user's items
                        sims = [
                            self.cf_recommender.item_similarity[doc_id, user_item]
                            for user_item in user_items
                            if doc_id < len(self.cf_recommender.item_similarity)
                        ]
                        cf_scores[doc_id] = np.mean(sims) if sims else 0.0

        # Stage 3: Combine and sort
        results = []
        for doc_id in candidate_ids:
            content_score = candidate_content_scores[doc_id]
            cf_score = cf_scores.get(doc_id, 0.0)
            pop_score = (
                self.item_popularity[doc_id]
                if self.item_popularity is not None and doc_id < len(self.item_popularity)
                else 0.0
            )

            # Cascade: prioritize CF if available, then content, then popularity
            if cf_score > 0:
                final_score = 0.7 * cf_score + 0.2 * content_score + 0.1 * pop_score
            else:
                final_score = 0.8 * content_score + 0.2 * pop_score

            results.append(HybridRecommendation(
                doc_id=doc_id,
                score=final_score,
                content_score=content_score,
                cf_score=cf_score,
                popularity_score=pop_score,
                method='cascade',
                explanation=f"Cascade: stage1=content({content_score:.2f}), stage2=cf({cf_score:.2f})"
            ))

        results.sort(key=lambda x: x.score, reverse=True)

        self.logger.info(
            f"Cascade fusion: {len(results[:top_k])} recommendations"
        )

        return results[:top_k]

    # ========================================================================
    # Switching Fusion
    # ========================================================================

    def recommend_switching(self,
                           user_id: Optional[int] = None,
                           doc_id: Optional[int] = None,
                           top_k: int = 10,
                           reading_history: Optional[List[int]] = None) -> List[HybridRecommendation]:
        """
        Switching hybrid recommendation.

        Chooses the best strategy based on context:
            - New user (cold start): Content-based
            - User with <5 interactions: Content-based + Popularity
            - User with >=5 interactions: Collaborative Filtering

        Args:
            user_id: User ID
            doc_id: Document ID
            top_k: Number of recommendations
            reading_history: Reading history

        Returns:
            List of HybridRecommendation objects
        """
        chosen_method = None

        # Determine user type
        if user_id is not None and self.cf_recommender is not None:
            user_profile = self.cf_recommender.get_user_profile(user_id)
            n_interactions = user_profile['n_interactions']

            if n_interactions >= 5:
                # Experienced user: use CF
                chosen_method = 'cf'
            elif n_interactions > 0:
                # Some data: use content + popularity
                chosen_method = 'content_popularity'
            else:
                # New user: content only
                chosen_method = 'content'
        else:
            # No CF data: use content
            chosen_method = 'content'

        self.logger.info(f"Switching to method: {chosen_method}")

        # Execute chosen method
        if chosen_method == 'cf':
            # Pure CF
            cf_recs = self.cf_recommender.recommend_item_based(
                user_id=user_id,
                top_k=top_k
            )
            results = [
                HybridRecommendation(
                    doc_id=rec.doc_id,
                    score=rec.score,
                    content_score=0.0,
                    cf_score=rec.score,
                    popularity_score=0.0,
                    method='switching(cf)',
                    explanation="Experienced user: using collaborative filtering"
                )
                for rec in cf_recs
            ]

        elif chosen_method == 'content_popularity':
            # Content + Popularity
            if doc_id is not None:
                content_recs = self.content_recommender.recommend_similar(
                    doc_id=doc_id,
                    top_k=top_k * 2
                )
            elif reading_history:
                content_recs = self.content_recommender.recommend_personalized(
                    reading_history=reading_history,
                    top_k=top_k * 2
                )
            else:
                content_recs = []

            results = []
            for rec in content_recs:
                pop_score = (
                    self.item_popularity[rec.doc_id]
                    if self.item_popularity is not None
                    else 0.0
                )
                final_score = 0.7 * rec.score + 0.3 * pop_score

                results.append(HybridRecommendation(
                    doc_id=rec.doc_id,
                    score=final_score,
                    content_score=rec.score,
                    cf_score=0.0,
                    popularity_score=pop_score,
                    method='switching(content+pop)',
                    explanation="New user: using content + popularity"
                ))

            results.sort(key=lambda x: x.score, reverse=True)

        else:  # 'content'
            # Pure content
            if doc_id is not None:
                content_recs = self.content_recommender.recommend_similar(
                    doc_id=doc_id,
                    top_k=top_k
                )
            elif reading_history:
                content_recs = self.content_recommender.recommend_personalized(
                    reading_history=reading_history,
                    top_k=top_k
                )
            else:
                content_recs = []

            results = [
                HybridRecommendation(
                    doc_id=rec.doc_id,
                    score=rec.score,
                    content_score=rec.score,
                    cf_score=0.0,
                    popularity_score=0.0,
                    method='switching(content)',
                    explanation="Cold start: using content-based only"
                )
                for rec in content_recs
            ]

        self.logger.info(
            f"Switching fusion ({chosen_method}): {len(results[:top_k])} recommendations"
        )

        return results[:top_k]

    # ========================================================================
    # Unified Interface
    # ========================================================================

    def recommend(self,
                  user_id: Optional[int] = None,
                  doc_id: Optional[int] = None,
                  top_k: int = 10,
                  reading_history: Optional[List[int]] = None,
                  method: Optional[str] = None) -> List[HybridRecommendation]:
        """
        Unified recommendation interface.

        Args:
            user_id: User ID
            doc_id: Document ID
            top_k: Number of recommendations
            reading_history: Reading history
            method: Override fusion method (None = use default)

        Returns:
            List of HybridRecommendation objects

        Examples:
            >>> hybrid = HybridRecommender(content_rec, cf_rec, fusion_method='weighted')
            >>> # Existing user
            >>> recs = hybrid.recommend(user_id=5, top_k=10)
            >>> # New user with history
            >>> recs = hybrid.recommend(reading_history=[0, 5, 12], top_k=10)
            >>> # Similar items
            >>> recs = hybrid.recommend(doc_id=10, top_k=10)
        """
        fusion = method or self.fusion_method

        if fusion == 'weighted':
            return self.recommend_weighted(user_id, doc_id, top_k, reading_history)
        elif fusion == 'cascade':
            return self.recommend_cascade(user_id, doc_id, top_k, reading_history)
        elif fusion == 'switching':
            return self.recommend_switching(user_id, doc_id, top_k, reading_history)
        else:
            raise ValueError(f"Unknown fusion method: {fusion}")

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def _normalize_score(self, score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Normalize score to [0, 1] range."""
        if max_val == min_val:
            return 0.0
        return (score - min_val) / (max_val - min_val)

    def _generate_explanation(self,
                             content_score: float,
                             cf_score: float,
                             pop_score: float) -> str:
        """Generate human-readable explanation."""
        signals = []

        if content_score > 0.5:
            signals.append(f"content similarity ({content_score:.2f})")
        if cf_score > 0.5:
            signals.append(f"users like you enjoyed this ({cf_score:.2f})")
        if pop_score > 0.7:
            signals.append(f"popular item ({pop_score:.2f})")

        if not signals:
            return "Recommended based on mixed signals"

        return "Recommended due to: " + ", ".join(signals)


def demo():
    """Demonstration of hybrid recommendation."""
    print("=" * 70)
    print("Hybrid Recommendation System Demo")
    print("=" * 70)

    print("\n[Note] This is a framework demonstration.")
    print("In production, initialize with actual recommender instances:")
    print("  content_rec = ContentBasedRecommender(documents)")
    print("  cf_rec = CollaborativeFilteringRecommender(n_users, n_items)")
    print("  hybrid = HybridRecommender(content_rec, cf_rec)")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    demo()
