"""
Content-Based Recommendation System

This module implements content-based filtering for document recommendation
using various similarity metrics and feature extraction techniques.

Key Features:
    - Multiple similarity metrics (Cosine, Jaccard, Euclidean)
    - TF-IDF based similarity
    - BERT embedding similarity
    - Hybrid similarity combinations
    - Personalization based on reading history
    - Diversity-aware recommendations

Complexity:
    - Similarity computation: O(N × d) where N = docs, d = dimensions
    - Top-k retrieval: O(N log k)
    - Space: O(N × d) for feature matrix

Author: Information Retrieval System
License: Educational Use
"""

from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from collections import Counter
import numpy as np
from pathlib import Path
import logging

# Import IR modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@dataclass
class RecommendationResult:
    """
    Recommendation result container.

    Attributes:
        doc_id: Document ID
        score: Similarity/relevance score
        reason: Explanation for recommendation
        features: Key features that contributed to recommendation
    """
    doc_id: int
    score: float
    reason: str
    features: Dict[str, float]

    def __repr__(self):
        return f"Rec(doc={self.doc_id}, score={self.score:.4f}, reason='{self.reason}')"


class ContentBasedRecommender:
    """
    Content-based recommendation system.

    Recommends documents based on content similarity to:
    - A single seed document (similar items)
    - User reading history (personalized)
    - User preferences/profile

    Attributes:
        logger: Logger instance
        documents: Document corpus
        doc_vectors: Document feature vectors (TF-IDF or embeddings)
        similarity_metric: Similarity function to use
    """

    def __init__(self,
                 documents: List[Dict],
                 similarity_metric: str = 'cosine',
                 diversity_lambda: float = 0.3):
        """
        Initialize content-based recommender.

        Args:
            documents: List of document dictionaries
            similarity_metric: 'cosine', 'jaccard', 'euclidean', 'dot'
            diversity_lambda: Balance between relevance and diversity (0-1)
        """
        self.logger = logging.getLogger(__name__)
        self.documents = documents
        self.similarity_metric = similarity_metric
        self.diversity_lambda = diversity_lambda

        # Document feature vectors (lazy initialization)
        self.doc_vectors = None
        self.doc_embeddings = None  # BERT embeddings if available

        # Document metadata for filtering
        self.doc_metadata = self._extract_metadata()

        self.logger.info(
            f"ContentBasedRecommender initialized: "
            f"{len(documents)} docs, metric={similarity_metric}"
        )

    # ========================================================================
    # Feature Extraction
    # ========================================================================

    def _extract_metadata(self) -> Dict[int, Dict]:
        """
        Extract metadata from documents for filtering.

        Returns:
            Dictionary mapping doc_id -> metadata
        """
        metadata = {}
        for i, doc in enumerate(self.documents):
            metadata[i] = {
                'title': doc.get('title', ''),
                'category': doc.get('category', ''),
                'tags': doc.get('tags', []),
                'date': doc.get('published_date', ''),
                'author': doc.get('author', ''),
                'keywords': doc.get('keywords', [])
            }
        return metadata

    def build_tfidf_vectors(self, vsm_model):
        """
        Build TF-IDF feature vectors using existing VSM model.

        Args:
            vsm_model: VectorSpaceModel instance with fitted index

        Complexity:
            Time: O(N × V) where N=docs, V=vocabulary
            Space: O(N × V)
        """
        from scipy.sparse import vstack

        self.doc_vectors = vsm_model.doc_vectors
        self.logger.info(f"Built TF-IDF vectors: shape={self.doc_vectors.shape}")

    def build_bert_embeddings(self, bert_retrieval):
        """
        Build BERT embedding vectors using existing BERT model.

        Args:
            bert_retrieval: BERTRetrieval instance with encoded documents

        Complexity:
            Time: O(1) (reuse existing embeddings)
            Space: O(N × 768)
        """
        self.doc_embeddings = bert_retrieval.doc_embeddings
        self.logger.info(
            f"Built BERT embeddings: shape={self.doc_embeddings.shape}"
        )

    # ========================================================================
    # Similarity Computation
    # ========================================================================

    def compute_similarity(self,
                          doc_id: int,
                          candidate_ids: Optional[List[int]] = None,
                          use_embeddings: bool = False) -> np.ndarray:
        """
        Compute similarity between a document and candidates.

        Args:
            doc_id: Source document ID
            candidate_ids: List of candidate doc IDs (None = all docs)
            use_embeddings: Use BERT embeddings instead of TF-IDF

        Returns:
            Array of similarity scores

        Complexity:
            Time: O(N × d) where d = feature dimensions
            Space: O(N)
        """
        # Choose feature vectors
        if use_embeddings and self.doc_embeddings is not None:
            vectors = self.doc_embeddings
        elif self.doc_vectors is not None:
            vectors = self.doc_vectors
        else:
            raise ValueError("No feature vectors available. Call build_*_vectors() first.")

        # Get source vector
        source_vec = vectors[doc_id]

        # Get candidate vectors
        if candidate_ids is None:
            candidate_vecs = vectors
        else:
            candidate_vecs = vectors[candidate_ids]

        # Compute similarity based on metric
        if self.similarity_metric == 'cosine':
            scores = self._cosine_similarity(source_vec, candidate_vecs)
        elif self.similarity_metric == 'dot':
            scores = self._dot_product(source_vec, candidate_vecs)
        elif self.similarity_metric == 'euclidean':
            scores = self._euclidean_distance(source_vec, candidate_vecs)
        else:
            raise ValueError(f"Unknown similarity metric: {self.similarity_metric}")

        return scores

    def _cosine_similarity(self, vec1, vec2_matrix):
        """Compute cosine similarity."""
        from scipy.spatial.distance import cdist
        from scipy.sparse import issparse

        if issparse(vec1):
            vec1 = vec1.toarray().flatten()
        if issparse(vec2_matrix):
            vec2_matrix = vec2_matrix.toarray()

        # Reshape vec1 for cdist
        vec1 = vec1.reshape(1, -1)

        # Compute cosine similarity (1 - cosine distance)
        distances = cdist(vec1, vec2_matrix, metric='cosine')
        similarities = 1 - distances.flatten()

        return similarities

    def _dot_product(self, vec1, vec2_matrix):
        """Compute dot product similarity."""
        if hasattr(vec2_matrix, 'dot'):
            # Sparse matrix dot product
            return vec2_matrix.dot(vec1.T).toarray().flatten()
        else:
            # Dense matrix multiplication
            return np.dot(vec2_matrix, vec1)

    def _euclidean_distance(self, vec1, vec2_matrix):
        """Compute Euclidean distance (converted to similarity)."""
        from scipy.spatial.distance import cdist
        from scipy.sparse import issparse

        if issparse(vec1):
            vec1 = vec1.toarray().flatten()
        if issparse(vec2_matrix):
            vec2_matrix = vec2_matrix.toarray()

        vec1 = vec1.reshape(1, -1)
        distances = cdist(vec1, vec2_matrix, metric='euclidean').flatten()

        # Convert distance to similarity (1 / (1 + distance))
        similarities = 1.0 / (1.0 + distances)

        return similarities

    # ========================================================================
    # Recommendation Methods
    # ========================================================================

    def recommend_similar(self,
                         doc_id: int,
                         top_k: int = 10,
                         exclude_self: bool = True,
                         use_embeddings: bool = False,
                         apply_diversity: bool = False) -> List[RecommendationResult]:
        """
        Recommend documents similar to a given document.

        Args:
            doc_id: Source document ID
            top_k: Number of recommendations
            exclude_self: Exclude source document from results
            use_embeddings: Use BERT embeddings
            apply_diversity: Apply diversity re-ranking

        Returns:
            List of RecommendationResult objects

        Examples:
            >>> recommender = ContentBasedRecommender(documents)
            >>> recommender.build_tfidf_vectors(vsm)
            >>> recs = recommender.recommend_similar(doc_id=5, top_k=10)
            >>> for rec in recs:
            ...     print(f"{rec.doc_id}: {rec.score:.4f}")
        """
        if doc_id < 0 or doc_id >= len(self.documents):
            raise ValueError(f"Invalid doc_id: {doc_id}")

        # Compute similarities
        similarities = self.compute_similarity(
            doc_id,
            use_embeddings=use_embeddings
        )

        # Exclude self if requested
        if exclude_self:
            similarities[doc_id] = -1.0

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k * 2]  # Get 2x for diversity

        # Apply diversity if requested
        if apply_diversity:
            top_indices = self._apply_diversity_reranking(
                doc_id, top_indices, similarities, top_k
            )
        else:
            top_indices = top_indices[:top_k]

        # Build recommendation results
        results = []
        source_meta = self.doc_metadata[doc_id]

        for idx in top_indices:
            if similarities[idx] < 0:
                continue

            # Determine reason for recommendation
            reason = self._explain_recommendation(doc_id, idx)

            # Extract key features
            features = self._extract_key_features(doc_id, idx)

            results.append(RecommendationResult(
                doc_id=int(idx),
                score=float(similarities[idx]),
                reason=reason,
                features=features
            ))

        self.logger.info(
            f"Recommended {len(results)} similar docs for doc_id={doc_id}"
        )

        return results[:top_k]

    def recommend_personalized(self,
                               reading_history: List[int],
                               top_k: int = 10,
                               use_embeddings: bool = False,
                               apply_diversity: bool = True) -> List[RecommendationResult]:
        """
        Recommend documents based on user reading history.

        Algorithm:
            1. Compute average feature vector of read documents
            2. Find documents similar to this "user profile"
            3. Exclude already-read documents
            4. Apply diversity re-ranking

        Args:
            reading_history: List of doc IDs user has read
            top_k: Number of recommendations
            use_embeddings: Use BERT embeddings
            apply_diversity: Apply diversity re-ranking

        Returns:
            List of RecommendationResult objects

        Examples:
            >>> history = [0, 5, 12, 23]  # Documents user has read
            >>> recs = recommender.recommend_personalized(history, top_k=10)
        """
        if not reading_history:
            raise ValueError("Reading history cannot be empty")

        # Choose feature vectors
        if use_embeddings and self.doc_embeddings is not None:
            vectors = self.doc_embeddings
        else:
            vectors = self.doc_vectors

        # Compute user profile (average of read documents)
        user_profile = np.mean([vectors[i] for i in reading_history], axis=0)

        # Compute similarity to all documents
        if self.similarity_metric == 'cosine':
            similarities = self._cosine_similarity(user_profile, vectors)
        elif self.similarity_metric == 'dot':
            similarities = self._dot_product(user_profile, vectors)
        else:
            similarities = self._euclidean_distance(user_profile, vectors)

        # Exclude already-read documents
        for doc_id in reading_history:
            similarities[doc_id] = -1.0

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k * 2]

        # Apply diversity if requested
        if apply_diversity:
            # Use first read document as reference for diversity
            top_indices = self._apply_diversity_reranking(
                reading_history[0], top_indices, similarities, top_k
            )
        else:
            top_indices = top_indices[:top_k]

        # Build recommendation results
        results = []
        for idx in top_indices:
            if similarities[idx] < 0:
                continue

            reason = f"Based on your reading history ({len(reading_history)} docs)"
            features = {'personalization_score': float(similarities[idx])}

            results.append(RecommendationResult(
                doc_id=int(idx),
                score=float(similarities[idx]),
                reason=reason,
                features=features
            ))

        self.logger.info(
            f"Personalized recommendations: {len(results)} docs for "
            f"{len(reading_history)} history items"
        )

        return results[:top_k]

    # ========================================================================
    # Diversity Re-ranking
    # ========================================================================

    def _apply_diversity_reranking(self,
                                   source_doc_id: int,
                                   candidate_indices: np.ndarray,
                                   relevance_scores: np.ndarray,
                                   top_k: int) -> np.ndarray:
        """
        Apply Maximal Marginal Relevance (MMR) for diversity.

        MMR balances relevance and diversity:
            MMR = λ × relevance(d, q) - (1-λ) × max_similarity(d, S)

        Args:
            source_doc_id: Source document
            candidate_indices: Candidate document indices
            relevance_scores: Relevance scores to source
            top_k: Number of diverse docs to select

        Returns:
            Array of selected indices (diverse + relevant)

        Complexity:
            Time: O(k² × d) where k=top_k, d=dimensions
            Space: O(k)
        """
        selected = []
        remaining = set(candidate_indices.tolist())

        # Choose vectors
        if self.doc_embeddings is not None:
            vectors = self.doc_embeddings
        else:
            vectors = self.doc_vectors

        while len(selected) < top_k and remaining:
            if not selected:
                # First selection: most relevant
                best_idx = candidate_indices[0]
            else:
                # Subsequent selections: balance relevance and diversity
                best_score = -np.inf
                best_idx = None

                for idx in remaining:
                    # Relevance to source
                    relevance = relevance_scores[idx]

                    # Diversity (max similarity to already selected)
                    max_sim = max(
                        self._cosine_similarity(
                            vectors[idx], vectors[np.array([s])]
                        )[0]
                        for s in selected
                    )

                    # MMR score
                    mmr_score = (
                        self.diversity_lambda * relevance -
                        (1 - self.diversity_lambda) * max_sim
                    )

                    if mmr_score > best_score:
                        best_score = mmr_score
                        best_idx = idx

            selected.append(best_idx)
            remaining.discard(best_idx)

        return np.array(selected)

    # ========================================================================
    # Explanation
    # ========================================================================

    def _explain_recommendation(self, source_id: int, rec_id: int) -> str:
        """
        Generate explanation for why document was recommended.

        Args:
            source_id: Source document ID
            rec_id: Recommended document ID

        Returns:
            Human-readable explanation string
        """
        source_meta = self.doc_metadata[source_id]
        rec_meta = self.doc_metadata[rec_id]

        reasons = []

        # Same category
        if source_meta['category'] == rec_meta['category']:
            reasons.append(f"same category ({source_meta['category']})")

        # Common tags
        common_tags = set(source_meta['tags']) & set(rec_meta['tags'])
        if common_tags:
            reasons.append(f"{len(common_tags)} shared tags")

        # Same author
        if source_meta['author'] and source_meta['author'] == rec_meta['author']:
            reasons.append(f"same author ({source_meta['author']})")

        # Content similarity (always present)
        reasons.append("high content similarity")

        return "Similar due to: " + ", ".join(reasons)

    def _extract_key_features(self, source_id: int, rec_id: int) -> Dict[str, float]:
        """
        Extract key features that contributed to recommendation.

        Returns:
            Dictionary of feature names and their contribution scores
        """
        source_meta = self.doc_metadata[source_id]
        rec_meta = self.doc_metadata[rec_id]

        features = {}

        # Category match
        features['category_match'] = float(source_meta['category'] == rec_meta['category'])

        # Tag overlap
        if source_meta['tags'] and rec_meta['tags']:
            overlap = len(set(source_meta['tags']) & set(rec_meta['tags']))
            features['tag_overlap'] = overlap / len(set(source_meta['tags']) | set(rec_meta['tags']))
        else:
            features['tag_overlap'] = 0.0

        # Temporal proximity (if dates available)
        # features['temporal_proximity'] = ...

        return features

    # ========================================================================
    # Filtering
    # ========================================================================

    def recommend_with_filters(self,
                               doc_id: int,
                               top_k: int = 10,
                               category: Optional[str] = None,
                               date_range: Optional[Tuple[str, str]] = None,
                               exclude_ids: Optional[Set[int]] = None,
                               use_embeddings: bool = False) -> List[RecommendationResult]:
        """
        Recommend documents with metadata filters.

        Args:
            doc_id: Source document ID
            top_k: Number of recommendations
            category: Filter by category
            date_range: Filter by date range (start, end)
            exclude_ids: Set of doc IDs to exclude
            use_embeddings: Use BERT embeddings

        Returns:
            List of RecommendationResult objects
        """
        # Get all candidates
        candidate_ids = []
        for i in range(len(self.documents)):
            # Apply filters
            if category and self.doc_metadata[i]['category'] != category:
                continue

            if date_range:
                doc_date = self.doc_metadata[i]['date']
                if not (date_range[0] <= doc_date <= date_range[1]):
                    continue

            if exclude_ids and i in exclude_ids:
                continue

            candidate_ids.append(i)

        # Compute similarities only for candidates
        similarities = self.compute_similarity(
            doc_id,
            candidate_ids=candidate_ids,
            use_embeddings=use_embeddings
        )

        # Get top-k
        top_indices_in_candidates = np.argsort(similarities)[::-1][:top_k]
        top_indices = [candidate_ids[i] for i in top_indices_in_candidates]

        # Build results
        results = []
        for idx in top_indices:
            reason = self._explain_recommendation(doc_id, idx)
            features = self._extract_key_features(doc_id, idx)

            results.append(RecommendationResult(
                doc_id=int(idx),
                score=float(similarities[top_indices_in_candidates[top_indices.index(idx)]]),
                reason=reason,
                features=features
            ))

        return results


def demo():
    """Demonstration of content-based recommendation."""
    print("=" * 70)
    print("Content-Based Recommendation Demo")
    print("=" * 70)

    # Sample documents
    documents = [
        {'id': 0, 'title': '機器學習入門', 'content': '機器學習是人工智慧的核心技術...', 'category': '科技', 'tags': ['AI', 'ML']},
        {'id': 1, 'title': '深度學習原理', 'content': '深度學習使用神經網路...', 'category': '科技', 'tags': ['AI', 'DL']},
        {'id': 2, 'title': '台灣經濟發展', 'content': '台灣經濟近年來...', 'category': '財經', 'tags': ['經濟', '台灣']},
        {'id': 3, 'title': '資訊檢索系統', 'content': '資訊檢索使用倒排索引...', 'category': '科技', 'tags': ['IR', 'Search']},
        {'id': 4, 'title': '自然語言處理', 'content': 'NLP讓電腦理解人類語言...', 'category': '科技', 'tags': ['AI', 'NLP']},
    ]

    print("\n[1] Initialize Recommender")
    print("-" * 70)
    recommender = ContentBasedRecommender(
        documents=documents,
        similarity_metric='cosine',
        diversity_lambda=0.3
    )
    print(f"Loaded {len(documents)} documents")

    print("\n[2] Recommend Similar Documents (requires feature vectors)")
    print("-" * 70)
    print("Note: In production, call recommender.build_tfidf_vectors(vsm)")
    print("      or recommender.build_bert_embeddings(bert_retrieval)")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    demo()
