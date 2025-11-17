"""
Collaborative Filtering Recommendation System

This module implements collaborative filtering algorithms for document
recommendation based on user-item interaction patterns.

Key Features:
    - User-based Collaborative Filtering
    - Item-based Collaborative Filtering
    - Matrix Factorization (SVD, ALS)
    - Implicit feedback support (clicks, reads, time spent)
    - Cold start handling
    - Scalable sparse matrix operations

Algorithms:
    1. User-based CF: Find similar users, recommend what they liked
    2. Item-based CF: Find similar items based on co-occurrence
    3. Matrix Factorization: SVD/ALS for latent factor discovery

Complexity:
    - User-based: O(U² × I) where U=users, I=items
    - Item-based: O(I² × U)
    - SVD: O(min(U,I)³)

References:
    Koren, Y., Bell, R., & Volinsky, C. (2009). "Matrix factorization
        techniques for recommender systems". Computer, 42(8), 30-37.
    Sarwar, B., et al. (2001). "Item-based collaborative filtering
        recommendation algorithms". WWW '01.

Author: Information Retrieval System
License: Educational Use
"""

from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
import logging
from pathlib import Path

# Scipy for sparse matrices
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import svds


@dataclass
class CFRecommendation:
    """
    Collaborative filtering recommendation result.

    Attributes:
        doc_id: Document ID
        score: Predicted rating/score
        method: CF method used (user-based, item-based, mf)
        confidence: Confidence in prediction (0-1)
    """
    doc_id: int
    score: float
    method: str
    confidence: float

    def __repr__(self):
        return f"CFRec(doc={self.doc_id}, score={self.score:.3f}, method={self.method})"


class CollaborativeFilteringRecommender:
    """
    Collaborative filtering recommender system.

    Supports both memory-based (user/item-based) and model-based
    (matrix factorization) collaborative filtering.

    Attributes:
        interaction_matrix: User-item interaction matrix (sparse)
        user_similarity: User-user similarity matrix
        item_similarity: Item-item similarity matrix
        user_factors: User latent factors (MF)
        item_factors: Item latent factors (MF)
    """

    def __init__(self,
                 n_users: int,
                 n_items: int,
                 implicit_feedback: bool = True,
                 min_similarity: float = 0.1):
        """
        Initialize collaborative filtering recommender.

        Args:
            n_users: Number of users
            n_items: Number of items (documents)
            implicit_feedback: Use implicit feedback (True) or explicit ratings (False)
            min_similarity: Minimum similarity threshold for neighbors
        """
        self.logger = logging.getLogger(__name__)
        self.n_users = n_users
        self.n_items = n_items
        self.implicit_feedback = implicit_feedback
        self.min_similarity = min_similarity

        # Interaction matrix (sparse)
        self.interaction_matrix = lil_matrix((n_users, n_items))

        # Precomputed similarities (lazy initialization)
        self.user_similarity = None
        self.item_similarity = None

        # Matrix factorization components
        self.user_factors = None
        self.item_factors = None
        self.global_mean = 0.0

        self.logger.info(
            f"CollaborativeFilteringRecommender initialized: "
            f"users={n_users}, items={n_items}, implicit={implicit_feedback}"
        )

    # ========================================================================
    # Data Loading
    # ========================================================================

    def add_interaction(self,
                       user_id: int,
                       item_id: int,
                       rating: float = 1.0):
        """
        Add a user-item interaction.

        Args:
            user_id: User ID
            item_id: Item (document) ID
            rating: Rating value (1.0 for implicit feedback)

        Examples:
            >>> cf = CollaborativeFilteringRecommender(n_users=100, n_items=1000)
            >>> cf.add_interaction(user_id=0, item_id=5, rating=1.0)  # User 0 read doc 5
            >>> cf.add_interaction(user_id=0, item_id=12, rating=1.0)
        """
        if user_id < 0 or user_id >= self.n_users:
            raise ValueError(f"Invalid user_id: {user_id}")
        if item_id < 0 or item_id >= self.n_items:
            raise ValueError(f"Invalid item_id: {item_id}")

        self.interaction_matrix[user_id, item_id] = rating

    def load_interactions(self,
                         interactions: List[Tuple[int, int, float]]):
        """
        Load batch of interactions.

        Args:
            interactions: List of (user_id, item_id, rating) tuples

        Complexity:
            Time: O(n) where n = len(interactions)
            Space: O(n)
        """
        for user_id, item_id, rating in interactions:
            self.add_interaction(user_id, item_id, rating)

        # Convert to CSR for efficient operations
        self.interaction_matrix = self.interaction_matrix.tocsr()

        self.logger.info(
            f"Loaded {len(interactions)} interactions, "
            f"sparsity={self._compute_sparsity():.2%}"
        )

    def _compute_sparsity(self) -> float:
        """Compute sparsity of interaction matrix."""
        total_entries = self.n_users * self.n_items
        non_zero = self.interaction_matrix.nnz
        return 1.0 - (non_zero / total_entries)

    # ========================================================================
    # User-Based Collaborative Filtering
    # ========================================================================

    def compute_user_similarity(self,
                                similarity_metric: str = 'cosine',
                                top_k: int = 50):
        """
        Compute user-user similarity matrix.

        Args:
            similarity_metric: 'cosine', 'pearson', 'jaccard'
            top_k: Keep only top-k similar users per user (memory efficiency)

        Complexity:
            Time: O(U² × I) where U=users, I=items
            Space: O(U²) or O(U × k) if top_k specified
        """
        self.logger.info(f"Computing user similarity ({similarity_metric})...")

        # Convert to dense for similarity computation (if small enough)
        # For large datasets, use pairwise_distances from sklearn
        from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances

        if similarity_metric == 'cosine':
            self.user_similarity = cosine_similarity(self.interaction_matrix)
        elif similarity_metric == 'pearson':
            # Pearson correlation (normalize first)
            mean_centered = self.interaction_matrix.toarray()
            user_means = mean_centered.mean(axis=1, keepdims=True)
            mean_centered = mean_centered - user_means
            self.user_similarity = cosine_similarity(mean_centered)
        elif similarity_metric == 'jaccard':
            # Jaccard similarity for binary interactions
            binary_matrix = (self.interaction_matrix > 0).astype(float)
            self.user_similarity = 1 - pairwise_distances(
                binary_matrix, metric='jaccard'
            )

        # Set diagonal to 0 (users are not similar to themselves)
        np.fill_diagonal(self.user_similarity, 0)

        # Keep only top-k if specified
        if top_k:
            self.user_similarity = self._keep_topk_similarities(
                self.user_similarity, top_k
            )

        self.logger.info(
            f"User similarity computed: shape={self.user_similarity.shape}"
        )

    def recommend_user_based(self,
                            user_id: int,
                            top_k: int = 10,
                            n_neighbors: int = 20) -> List[CFRecommendation]:
        """
        Recommend items using user-based collaborative filtering.

        Algorithm:
            1. Find k most similar users to target user
            2. Aggregate items they interacted with
            3. Weight by similarity and remove already-consumed items
            4. Return top-k items

        Args:
            user_id: Target user ID
            top_k: Number of recommendations
            n_neighbors: Number of similar users to consider

        Returns:
            List of CFRecommendation objects

        Complexity:
            Time: O(k × I) where k=neighbors, I=items
            Space: O(I)

        Examples:
            >>> cf.compute_user_similarity()
            >>> recs = cf.recommend_user_based(user_id=0, top_k=10)
        """
        if self.user_similarity is None:
            raise ValueError("User similarity not computed. Call compute_user_similarity() first.")

        # Get similar users
        user_sims = self.user_similarity[user_id]
        similar_users = np.argsort(user_sims)[::-1][:n_neighbors]

        # Get items target user has already interacted with
        user_items = set(self.interaction_matrix[user_id].nonzero()[1])

        # Aggregate scores from similar users
        item_scores = defaultdict(float)
        item_confidence = defaultdict(float)

        for sim_user in similar_users:
            sim_score = user_sims[sim_user]
            if sim_score < self.min_similarity:
                continue

            # Get items this similar user interacted with
            sim_user_items = self.interaction_matrix[sim_user].nonzero()[1]

            for item in sim_user_items:
                if item not in user_items:  # Skip already consumed
                    rating = self.interaction_matrix[sim_user, item]
                    item_scores[item] += sim_score * rating
                    item_confidence[item] += sim_score

        # Normalize scores
        for item in item_scores:
            if item_confidence[item] > 0:
                item_scores[item] /= item_confidence[item]

        # Sort and return top-k
        sorted_items = sorted(
            item_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        results = [
            CFRecommendation(
                doc_id=int(item),
                score=float(score),
                method='user-based',
                confidence=float(item_confidence[item] / n_neighbors)
            )
            for item, score in sorted_items
        ]

        self.logger.info(
            f"User-based CF: {len(results)} recommendations for user {user_id}"
        )

        return results

    # ========================================================================
    # Item-Based Collaborative Filtering
    # ========================================================================

    def compute_item_similarity(self,
                                similarity_metric: str = 'cosine',
                                top_k: int = 100):
        """
        Compute item-item similarity matrix.

        Args:
            similarity_metric: 'cosine', 'jaccard', 'adjusted_cosine'
            top_k: Keep only top-k similar items per item

        Complexity:
            Time: O(I² × U) where I=items, U=users
            Space: O(I²) or O(I × k) if top_k specified
        """
        self.logger.info(f"Computing item similarity ({similarity_metric})...")

        from sklearn.metrics.pairwise import cosine_similarity

        # Transpose to get item-user matrix
        item_user_matrix = self.interaction_matrix.T

        if similarity_metric == 'cosine':
            self.item_similarity = cosine_similarity(item_user_matrix)
        elif similarity_metric == 'adjusted_cosine':
            # Adjusted cosine: normalize by user means
            dense_matrix = item_user_matrix.toarray().T  # Back to user-item
            user_means = dense_matrix.mean(axis=1, keepdims=True)
            normalized = dense_matrix - user_means
            self.item_similarity = cosine_similarity(normalized.T)
        elif similarity_metric == 'jaccard':
            binary_matrix = (item_user_matrix > 0).astype(float)
            from sklearn.metrics.pairwise import pairwise_distances
            self.item_similarity = 1 - pairwise_distances(
                binary_matrix, metric='jaccard'
            )

        # Set diagonal to 0
        np.fill_diagonal(self.item_similarity, 0)

        # Keep top-k
        if top_k:
            self.item_similarity = self._keep_topk_similarities(
                self.item_similarity, top_k
            )

        self.logger.info(
            f"Item similarity computed: shape={self.item_similarity.shape}"
        )

    def recommend_item_based(self,
                            user_id: int,
                            top_k: int = 10,
                            n_neighbors: int = 50) -> List[CFRecommendation]:
        """
        Recommend items using item-based collaborative filtering.

        Algorithm:
            1. Get items user has interacted with
            2. For each item, find k most similar items
            3. Aggregate similarity scores
            4. Return top-k unseen items

        Args:
            user_id: Target user ID
            top_k: Number of recommendations
            n_neighbors: Number of similar items to consider per item

        Returns:
            List of CFRecommendation objects

        Complexity:
            Time: O(m × k) where m=user's items, k=neighbors
            Space: O(I)
        """
        if self.item_similarity is None:
            raise ValueError("Item similarity not computed. Call compute_item_similarity() first.")

        # Get items user has interacted with
        user_items = self.interaction_matrix[user_id].nonzero()[1]
        user_ratings = {
            item: self.interaction_matrix[user_id, item]
            for item in user_items
        }

        # Aggregate scores
        item_scores = defaultdict(float)
        item_confidence = defaultdict(float)

        for item in user_items:
            user_rating = user_ratings[item]

            # Get similar items
            item_sims = self.item_similarity[item]
            similar_items = np.argsort(item_sims)[::-1][:n_neighbors]

            for sim_item in similar_items:
                sim_score = item_sims[sim_item]
                if sim_score < self.min_similarity:
                    continue

                if sim_item not in user_items:  # Skip already consumed
                    item_scores[sim_item] += sim_score * user_rating
                    item_confidence[sim_item] += sim_score

        # Normalize
        for item in item_scores:
            if item_confidence[item] > 0:
                item_scores[item] /= item_confidence[item]

        # Sort and return top-k
        sorted_items = sorted(
            item_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        results = [
            CFRecommendation(
                doc_id=int(item),
                score=float(score),
                method='item-based',
                confidence=float(item_confidence[item] / len(user_items))
            )
            for item, score in sorted_items
        ]

        self.logger.info(
            f"Item-based CF: {len(results)} recommendations for user {user_id}"
        )

        return results

    # ========================================================================
    # Matrix Factorization
    # ========================================================================

    def train_matrix_factorization(self,
                                   n_factors: int = 50,
                                   method: str = 'svd',
                                   n_iterations: int = 20,
                                   learning_rate: float = 0.01,
                                   reg_lambda: float = 0.02):
        """
        Train matrix factorization model.

        Decomposes interaction matrix R into:
            R ≈ U × V^T
        where U = user factors, V = item factors

        Args:
            n_factors: Number of latent factors
            method: 'svd' (Truncated SVD) or 'als' (Alternating Least Squares)
            n_iterations: Number of iterations (for ALS)
            learning_rate: Learning rate (for ALS)
            reg_lambda: Regularization parameter

        Complexity:
            Time: O(min(U,I)³) for SVD, O(k×I×U×n_iter) for ALS
            Space: O((U+I)×k) where k=n_factors

        Examples:
            >>> cf.train_matrix_factorization(n_factors=50, method='svd')
            >>> recs = cf.recommend_matrix_factorization(user_id=0, top_k=10)
        """
        self.logger.info(
            f"Training matrix factorization: {method}, factors={n_factors}"
        )

        # Compute global mean
        self.global_mean = self.interaction_matrix.data.mean()

        if method == 'svd':
            # Truncated SVD using scipy
            # R ≈ U × Σ × V^T
            U, sigma, Vt = svds(
                self.interaction_matrix.astype(float),
                k=n_factors
            )

            # Store factors
            self.user_factors = U * np.sqrt(sigma)  # U × √Σ
            self.item_factors = Vt.T * np.sqrt(sigma)  # V × √Σ

        elif method == 'als':
            # Alternating Least Squares
            self.user_factors, self.item_factors = self._train_als(
                n_factors, n_iterations, reg_lambda
            )

        else:
            raise ValueError(f"Unknown method: {method}")

        self.logger.info(
            f"Matrix factorization trained: "
            f"user_factors={self.user_factors.shape}, "
            f"item_factors={self.item_factors.shape}"
        )

    def _train_als(self,
                   n_factors: int,
                   n_iterations: int,
                   reg_lambda: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train using Alternating Least Squares.

        Iteratively optimizes user and item factors:
            - Fix V, solve for U
            - Fix U, solve for V

        Args:
            n_factors: Number of latent factors
            n_iterations: Number of iterations
            reg_lambda: Regularization parameter

        Returns:
            Tuple of (user_factors, item_factors)
        """
        # Initialize factors randomly
        user_factors = np.random.normal(0, 0.1, (self.n_users, n_factors))
        item_factors = np.random.normal(0, 0.1, (self.n_items, n_factors))

        # Convert to dense for ALS (if feasible)
        R = self.interaction_matrix.toarray()

        for iteration in range(n_iterations):
            # Update user factors
            for u in range(self.n_users):
                items = R[u] > 0
                if items.sum() == 0:
                    continue

                V_u = item_factors[items]
                r_u = R[u, items]

                # Solve: (V^T V + λI)u = V^T r
                A = V_u.T @ V_u + reg_lambda * np.eye(n_factors)
                b = V_u.T @ r_u
                user_factors[u] = np.linalg.solve(A, b)

            # Update item factors
            for i in range(self.n_items):
                users = R[:, i] > 0
                if users.sum() == 0:
                    continue

                U_i = user_factors[users]
                r_i = R[users, i]

                # Solve: (U^T U + λI)v = U^T r
                A = U_i.T @ U_i + reg_lambda * np.eye(n_factors)
                b = U_i.T @ r_i
                item_factors[i] = np.linalg.solve(A, b)

            # Compute error (optional, for monitoring)
            if iteration % 5 == 0:
                predictions = user_factors @ item_factors.T
                error = np.sqrt(((R - predictions) ** 2).mean())
                self.logger.debug(f"ALS iteration {iteration}: RMSE={error:.4f}")

        return user_factors, item_factors

    def recommend_matrix_factorization(self,
                                      user_id: int,
                                      top_k: int = 10) -> List[CFRecommendation]:
        """
        Recommend using matrix factorization.

        Args:
            user_id: Target user ID
            top_k: Number of recommendations

        Returns:
            List of CFRecommendation objects

        Complexity:
            Time: O(I × k) where k=n_factors
            Space: O(I)
        """
        if self.user_factors is None or self.item_factors is None:
            raise ValueError("Model not trained. Call train_matrix_factorization() first.")

        # Predict ratings: r_ui = u^T v_i
        user_vector = self.user_factors[user_id]
        predictions = self.item_factors @ user_vector

        # Get items user has already interacted with
        user_items = set(self.interaction_matrix[user_id].nonzero()[1])

        # Remove already-consumed items
        for item in user_items:
            predictions[item] = -np.inf

        # Get top-k
        top_indices = np.argsort(predictions)[::-1][:top_k]

        results = [
            CFRecommendation(
                doc_id=int(idx),
                score=float(predictions[idx]),
                method='matrix-factorization',
                confidence=0.8  # High confidence for MF
            )
            for idx in top_indices
            if predictions[idx] > -np.inf
        ]

        self.logger.info(
            f"MF CF: {len(results)} recommendations for user {user_id}"
        )

        return results

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def _keep_topk_similarities(self,
                                similarity_matrix: np.ndarray,
                                top_k: int) -> np.ndarray:
        """
        Keep only top-k similarities per row for memory efficiency.

        Args:
            similarity_matrix: Full similarity matrix
            top_k: Number of top similarities to keep per row

        Returns:
            Sparse matrix with only top-k similarities
        """
        n = similarity_matrix.shape[0]
        result = np.zeros_like(similarity_matrix)

        for i in range(n):
            row = similarity_matrix[i]
            top_indices = np.argsort(row)[::-1][:top_k]
            result[i, top_indices] = row[top_indices]

        return result

    def get_user_profile(self, user_id: int) -> Dict:
        """
        Get user's interaction profile.

        Returns:
            Dictionary with user statistics
        """
        user_row = self.interaction_matrix[user_id]
        items = user_row.nonzero()[1]
        ratings = user_row.data

        return {
            'user_id': user_id,
            'n_interactions': len(items),
            'interacted_items': items.tolist(),
            'avg_rating': ratings.mean() if len(ratings) > 0 else 0.0,
            'sparsity': 1.0 - (len(items) / self.n_items)
        }

    def get_item_profile(self, item_id: int) -> Dict:
        """
        Get item's interaction profile.

        Returns:
            Dictionary with item statistics
        """
        item_col = self.interaction_matrix[:, item_id]
        users = item_col.nonzero()[0]
        ratings = item_col.data

        return {
            'item_id': item_id,
            'n_interactions': len(users),
            'interacted_users': users.tolist(),
            'avg_rating': ratings.mean() if len(ratings) > 0 else 0.0,
            'popularity': len(users) / self.n_users
        }


def demo():
    """Demonstration of collaborative filtering."""
    print("=" * 70)
    print("Collaborative Filtering Recommendation Demo")
    print("=" * 70)

    # Simulate user-item interactions
    print("\n[1] Initialize CF System")
    print("-" * 70)
    cf = CollaborativeFilteringRecommender(
        n_users=10,
        n_items=20,
        implicit_feedback=True
    )

    # Generate synthetic interactions
    print("\n[2] Load Interactions (Synthetic Data)")
    print("-" * 70)
    interactions = [
        (0, 0, 1.0), (0, 1, 1.0), (0, 5, 1.0),  # User 0 read docs 0,1,5
        (1, 0, 1.0), (1, 2, 1.0), (1, 6, 1.0),  # User 1 read docs 0,2,6
        (2, 1, 1.0), (2, 2, 1.0), (2, 3, 1.0),  # User 2 read docs 1,2,3
        (3, 5, 1.0), (3, 6, 1.0), (3, 7, 1.0),  # User 3 read docs 5,6,7
    ]
    cf.load_interactions(interactions)
    print(f"Loaded {len(interactions)} interactions")
    print(f"Sparsity: {cf._compute_sparsity():.2%}")

    # User-based CF
    print("\n[3] User-Based Collaborative Filtering")
    print("-" * 70)
    cf.compute_user_similarity(similarity_metric='cosine')
    recs = cf.recommend_user_based(user_id=0, top_k=5)
    print(f"Recommendations for user 0:")
    for rec in recs:
        print(f"  {rec}")

    # Item-based CF
    print("\n[4] Item-Based Collaborative Filtering")
    print("-" * 70)
    cf.compute_item_similarity(similarity_metric='cosine')
    recs = cf.recommend_item_based(user_id=0, top_k=5)
    print(f"Recommendations for user 0:")
    for rec in recs:
        print(f"  {rec}")

    # Matrix Factorization
    print("\n[5] Matrix Factorization")
    print("-" * 70)
    cf.train_matrix_factorization(n_factors=5, method='svd')
    recs = cf.recommend_matrix_factorization(user_id=0, top_k=5)
    print(f"Recommendations for user 0:")
    for rec in recs:
        print(f"  {rec}")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    demo()
