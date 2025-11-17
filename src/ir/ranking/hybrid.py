"""
Hybrid Ranking and Score Fusion

This module implements hybrid ranking strategies that combine multiple retrieval
signals (sparse and dense) for improved search quality.

Hybrid retrieval combines:
    - Sparse retrieval: BM25, TF-IDF, Language Models (exact term matching)
    - Dense retrieval: BERT embeddings (semantic matching)
    - Other signals: PageRank, recency, popularity

Key Concepts:
    - Score Fusion: Combining scores from multiple rankers
    - Rank Fusion: Combining rankings (not scores)
    - Linear Combination: Weighted sum of scores
    - Reciprocal Rank Fusion (RRF): Rank-based combination

Fusion Strategies:
    1. Linear Combination: score = α*BM25 + β*BERT + γ*LM
    2. Reciprocal Rank Fusion: score = Σ 1/(k + rank_i)
    3. CombSUM: Unnormalized sum
    4. CombMNZ: Sum multiplied by number of matches
    5. Learning to Rank: ML-based combination

Formulas:
    - Linear: score = Σ wi * score_i
    - RRF: score = Σ 1/(k + rank_i) where k=60 typically
    - CombMNZ: score = (Σ score_i) * |{rankers matching doc}|
    - Min-Max Normalization: (x - min) / (max - min)
    - Z-score: (x - mean) / std

Key Features:
    - Multiple fusion strategies
    - Score normalization (min-max, z-score)
    - Automatic weight tuning
    - Support for any number of rankers

Reference:
    Croft (2000). "Combining Approaches to Information Retrieval"
    Montague & Aslam (2002). "Relevance Score Normalization for Metasearch"
    Cormack et al. (2009). "Reciprocal Rank Fusion outperforms Condorcet and
        individual Rank Learning Methods"

Author: Information Retrieval System
License: Educational Use
"""

import math
import logging
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from collections import defaultdict
import numpy as np


@dataclass
class HybridResult:
    """
    Result of hybrid ranking.

    Attributes:
        query: Original query string
        doc_ids: List of document IDs ranked by hybrid score
        scores: Corresponding hybrid scores
        component_scores: Individual scores from each ranker
        num_results: Total number of results
        fusion_method: Fusion strategy used
        weights: Weights used for fusion
    """
    query: str
    doc_ids: List[int]
    scores: List[float]
    component_scores: Dict[str, List[float]]
    num_results: int
    fusion_method: str
    weights: Dict[str, float]


class HybridRanker:
    """
    Hybrid ranker combining multiple retrieval models.

    Combines scores from sparse (BM25, LM) and dense (BERT) retrievers
    using various fusion strategies.

    Attributes:
        rankers: Dictionary of ranker name -> ranker instance
        fusion_method: Fusion strategy ('linear', 'rrf', 'combsum', 'combmnz')
        weights: Fusion weights for each ranker
        normalization: Score normalization method ('minmax', 'zscore', 'none')

    Complexity:
        - Fusion: O(R * N) where R=num rankers, N=result size
        - Normalization: O(N) per ranker
    """

    def __init__(self,
                 rankers: Dict[str, any],
                 fusion_method: str = 'linear',
                 weights: Optional[Dict[str, float]] = None,
                 normalization: str = 'minmax'):
        """
        Initialize hybrid ranker.

        Args:
            rankers: Dictionary of ranker_name -> ranker_instance
            fusion_method: Fusion strategy ('linear', 'rrf', 'combsum', 'combmnz')
            weights: Fusion weights (default: equal weights)
            normalization: Score normalization ('minmax', 'zscore', 'none')

        Complexity:
            Time: O(1)
        """
        self.logger = logging.getLogger(__name__)

        self.rankers = rankers
        self.fusion_method = fusion_method
        self.normalization = normalization

        # Set default weights (equal weights)
        if weights is None:
            num_rankers = len(rankers)
            self.weights = {name: 1.0 / num_rankers for name in rankers.keys()}
        else:
            # Normalize weights to sum to 1
            total_weight = sum(weights.values())
            self.weights = {k: v / total_weight for k, v in weights.items()}

        self.logger.info(
            f"HybridRanker initialized: {list(rankers.keys())}, "
            f"fusion={fusion_method}, normalization={normalization}"
        )

    def search(self, query: str, topk: int = 10,
               ranker_topk: int = 100) -> HybridResult:
        """
        Search using hybrid ranking.

        Args:
            query: Query string
            topk: Number of final results
            ranker_topk: Number of results to retrieve from each ranker

        Returns:
            HybridResult with fused rankings

        Complexity:
            Time: O(R * (search_cost + N)) where R=rankers, N=ranker_topk
            Space: O(R * N)

        Examples:
            >>> hybrid = HybridRanker({'bm25': bm25, 'bert': bert})
            >>> result = hybrid.search("information retrieval", topk=10)
            >>> result.doc_ids
            [5, 12, 3, 18, ...]
        """
        # Retrieve results from each ranker
        ranker_results = {}

        for ranker_name, ranker in self.rankers.items():
            try:
                # Call search method (assumes standard interface)
                result = ranker.search(query, topk=ranker_topk)

                # Extract doc_ids and scores
                ranker_results[ranker_name] = {
                    'doc_ids': result.doc_ids,
                    'scores': result.scores
                }

                self.logger.debug(
                    f"Ranker '{ranker_name}' returned {len(result.doc_ids)} results"
                )
            except Exception as e:
                self.logger.error(f"Error in ranker '{ranker_name}': {e}")
                ranker_results[ranker_name] = {
                    'doc_ids': [],
                    'scores': []
                }

        # Fuse results
        if self.fusion_method == 'linear':
            fused_scores = self._linear_fusion(ranker_results)
        elif self.fusion_method == 'rrf':
            fused_scores = self._reciprocal_rank_fusion(ranker_results)
        elif self.fusion_method == 'combsum':
            fused_scores = self._combsum_fusion(ranker_results)
        elif self.fusion_method == 'combmnz':
            fused_scores = self._combmnz_fusion(ranker_results)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")

        # Sort by fused score
        sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

        # Get top-k
        topk_docs = sorted_docs[:topk]

        doc_ids = [doc_id for doc_id, score in topk_docs]
        scores = [score for doc_id, score in topk_docs]

        # Extract component scores for top-k docs
        component_scores = {}
        for ranker_name, result in ranker_results.items():
            doc_to_score = dict(zip(result['doc_ids'], result['scores']))
            component_scores[ranker_name] = [
                doc_to_score.get(doc_id, 0.0) for doc_id in doc_ids
            ]

        return HybridResult(
            query=query,
            doc_ids=doc_ids,
            scores=scores,
            component_scores=component_scores,
            num_results=len(doc_ids),
            fusion_method=self.fusion_method,
            weights=self.weights
        )

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Normalize scores using specified method.

        Args:
            scores: List of scores

        Returns:
            Normalized scores

        Complexity:
            Time: O(n) where n = len(scores)
        """
        if not scores:
            return []

        if self.normalization == 'minmax':
            # Min-max normalization: (x - min) / (max - min)
            min_score = min(scores)
            max_score = max(scores)

            if max_score == min_score:
                return [1.0] * len(scores)

            return [(s - min_score) / (max_score - min_score) for s in scores]

        elif self.normalization == 'zscore':
            # Z-score normalization: (x - mean) / std
            mean = np.mean(scores)
            std = np.std(scores)

            if std == 0:
                return [0.0] * len(scores)

            return [(s - mean) / std for s in scores]

        else:  # 'none'
            return scores

    def _linear_fusion(self, ranker_results: Dict[str, Dict]) -> Dict[int, float]:
        """
        Linear combination fusion: score = Σ wi * score_i

        Args:
            ranker_results: Results from each ranker

        Returns:
            Dictionary of doc_id -> fused_score

        Complexity:
            Time: O(R * N) where R=rankers, N=avg results per ranker
        """
        fused_scores = defaultdict(float)

        for ranker_name, result in ranker_results.items():
            doc_ids = result['doc_ids']
            scores = result['scores']

            # Normalize scores
            normalized_scores = self._normalize_scores(scores)

            # Weight for this ranker
            weight = self.weights.get(ranker_name, 0.0)

            # Add weighted scores
            for doc_id, score in zip(doc_ids, normalized_scores):
                fused_scores[doc_id] += weight * score

        return dict(fused_scores)

    def _reciprocal_rank_fusion(self, ranker_results: Dict[str, Dict],
                                k: int = 60) -> Dict[int, float]:
        """
        Reciprocal Rank Fusion (RRF).

        RRF(d) = Σ 1 / (k + rank_i(d))

        where rank_i(d) is the rank of document d in ranker i.

        Args:
            ranker_results: Results from each ranker
            k: Constant (typically 60)

        Returns:
            Dictionary of doc_id -> RRF_score

        Complexity:
            Time: O(R * N)

        Reference:
            Cormack et al. (2009). "Reciprocal Rank Fusion"
        """
        fused_scores = defaultdict(float)

        for ranker_name, result in ranker_results.items():
            doc_ids = result['doc_ids']

            # Weight for this ranker
            weight = self.weights.get(ranker_name, 1.0)

            # Calculate RRF score
            for rank, doc_id in enumerate(doc_ids, start=1):
                rrf_score = 1.0 / (k + rank)
                fused_scores[doc_id] += weight * rrf_score

        return dict(fused_scores)

    def _combsum_fusion(self, ranker_results: Dict[str, Dict]) -> Dict[int, float]:
        """
        CombSUM fusion: unnormalized sum of scores.

        score = Σ score_i

        Args:
            ranker_results: Results from each ranker

        Returns:
            Dictionary of doc_id -> fused_score

        Complexity:
            Time: O(R * N)
        """
        fused_scores = defaultdict(float)

        for ranker_name, result in ranker_results.items():
            doc_ids = result['doc_ids']
            scores = result['scores']

            # Weight for this ranker
            weight = self.weights.get(ranker_name, 1.0)

            for doc_id, score in zip(doc_ids, scores):
                fused_scores[doc_id] += weight * score

        return dict(fused_scores)

    def _combmnz_fusion(self, ranker_results: Dict[str, Dict]) -> Dict[int, float]:
        """
        CombMNZ fusion: sum multiplied by number of non-zero matches.

        score = (Σ score_i) * |{rankers matching doc}|

        Favors documents matched by multiple rankers.

        Args:
            ranker_results: Results from each ranker

        Returns:
            Dictionary of doc_id -> fused_score

        Complexity:
            Time: O(R * N)
        """
        # Count number of rankers matching each document
        doc_match_count = defaultdict(int)
        doc_sum_score = defaultdict(float)

        for ranker_name, result in ranker_results.items():
            doc_ids = result['doc_ids']
            scores = result['scores']

            # Weight for this ranker
            weight = self.weights.get(ranker_name, 1.0)

            for doc_id, score in zip(doc_ids, scores):
                if score > 0:
                    doc_match_count[doc_id] += 1
                doc_sum_score[doc_id] += weight * score

        # CombMNZ: multiply sum by match count
        fused_scores = {
            doc_id: doc_sum_score[doc_id] * doc_match_count[doc_id]
            for doc_id in doc_sum_score.keys()
        }

        return fused_scores

    def explain_score(self, query: str, doc_id: int,
                     ranker_topk: int = 100) -> Dict:
        """
        Explain hybrid score breakdown for a document.

        Args:
            query: Query string
            doc_id: Document ID
            ranker_topk: Number of results from each ranker

        Returns:
            Dictionary with score explanation

        Examples:
            >>> explain = hybrid.explain_score("IR", 5)
            >>> print(explain)
            {
                'doc_id': 5,
                'total_score': 0.85,
                'fusion_method': 'linear',
                'component_scores': {
                    'bm25': {'score': 12.3, 'normalized': 0.8, 'weighted': 0.4},
                    'bert': {'score': 0.92, 'normalized': 0.9, 'weighted': 0.45}
                }
            }
        """
        # Get results from each ranker
        component_scores = {}

        for ranker_name, ranker in self.rankers.items():
            try:
                result = ranker.search(query, topk=ranker_topk)

                # Find document in results
                if doc_id in result.doc_ids:
                    idx = result.doc_ids.index(doc_id)
                    raw_score = result.scores[idx]
                    rank = idx + 1
                else:
                    raw_score = 0.0
                    rank = None

                # Normalize score
                normalized_scores = self._normalize_scores(result.scores)
                normalized_score = normalized_scores[idx] if rank else 0.0

                # Weighted score
                weight = self.weights.get(ranker_name, 0.0)
                weighted_score = weight * normalized_score

                component_scores[ranker_name] = {
                    'raw_score': raw_score,
                    'rank': rank,
                    'normalized_score': round(normalized_score, 4),
                    'weight': weight,
                    'weighted_score': round(weighted_score, 4)
                }

            except Exception as e:
                self.logger.error(f"Error explaining ranker '{ranker_name}': {e}")
                component_scores[ranker_name] = {'error': str(e)}

        # Calculate total score
        total_score = sum(
            comp['weighted_score']
            for comp in component_scores.values()
            if 'weighted_score' in comp
        )

        return {
            'doc_id': doc_id,
            'total_score': round(total_score, 4),
            'fusion_method': self.fusion_method,
            'normalization': self.normalization,
            'component_scores': component_scores
        }

    def get_stats(self) -> Dict:
        """Get hybrid ranker statistics."""
        return {
            'num_rankers': len(self.rankers),
            'ranker_names': list(self.rankers.keys()),
            'fusion_method': self.fusion_method,
            'normalization': self.normalization,
            'weights': self.weights
        }


def demo():
    """Demonstration of hybrid ranking."""
    print("=" * 60)
    print("Hybrid Ranking Demo")
    print("=" * 60)

    # Mock ranker results (simulating BM25 and BERT)
    class MockRanker:
        def __init__(self, name, results):
            self.name = name
            self.results = results

        def search(self, query, topk=10):
            from collections import namedtuple
            Result = namedtuple('Result', ['doc_ids', 'scores'])
            return Result(
                doc_ids=self.results['doc_ids'][:topk],
                scores=self.results['scores'][:topk]
            )

    # BM25 results (favors exact term matching)
    bm25_results = {
        'doc_ids': [3, 1, 5, 7, 2],
        'scores': [15.2, 12.8, 10.5, 8.3, 6.1]
    }

    # BERT results (favors semantic similarity)
    bert_results = {
        'doc_ids': [5, 3, 9, 1, 4],
        'scores': [0.92, 0.88, 0.85, 0.82, 0.78]
    }

    # Create rankers
    rankers = {
        'bm25': MockRanker('BM25', bm25_results),
        'bert': MockRanker('BERT', bert_results)
    }

    # Test different fusion methods
    fusion_methods = ['linear', 'rrf', 'combsum', 'combmnz']

    for method in fusion_methods:
        print(f"\n{'-' * 60}")
        print(f"Fusion Method: {method.upper()}")
        print("-" * 60)

        hybrid = HybridRanker(
            rankers=rankers,
            fusion_method=method,
            weights={'bm25': 0.5, 'bert': 0.5},
            normalization='minmax'
        )

        result = hybrid.search("information retrieval", topk=5)

        print(f"Query: {result.query}")
        print(f"Top-{result.num_results} Results:")

        for i, (doc_id, score) in enumerate(zip(result.doc_ids, result.scores), 1):
            bm25_score = result.component_scores['bm25'][i - 1]
            bert_score = result.component_scores['bert'][i - 1]

            print(f"  {i}. Doc {doc_id}: {score:.4f}")
            print(f"     BM25: {bm25_score:.4f}, BERT: {bert_score:.4f}")

    # Explain score
    print(f"\n{'=' * 60}")
    print("Score Explanation")
    print("=" * 60)

    hybrid = HybridRanker(
        rankers=rankers,
        fusion_method='linear',
        weights={'bm25': 0.6, 'bert': 0.4}
    )

    doc_id = 3
    explanation = hybrid.explain_score("information retrieval", doc_id)

    print(f"\nDocument {doc_id}")
    print(f"Total Hybrid Score: {explanation['total_score']}")
    print(f"Fusion Method: {explanation['fusion_method']}")
    print("\nComponent Scores:")

    for ranker_name, comp in explanation['component_scores'].items():
        print(f"  {ranker_name.upper()}:")
        print(f"    Raw Score: {comp.get('raw_score', 'N/A')}")
        print(f"    Rank: {comp.get('rank', 'N/A')}")
        print(f"    Normalized: {comp.get('normalized_score', 'N/A')}")
        print(f"    Weight: {comp.get('weight', 'N/A')}")
        print(f"    Weighted Score: {comp.get('weighted_score', 'N/A')}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    demo()
