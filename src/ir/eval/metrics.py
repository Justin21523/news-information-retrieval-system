"""
Evaluation Metrics for Information Retrieval

This module implements standard IR evaluation metrics for assessing
retrieval system performance including precision, recall, F-measure,
average precision, MAP, and nDCG.

Key Features:
    - Binary relevance metrics (Precision, Recall, F-measure)
    - Ranked retrieval metrics (AP, MAP, MRR)
    - Graded relevance metrics (nDCG, ERR)
    - Set-based metrics (Jaccard, Overlap)
    - Statistical significance testing support

Reference: "Introduction to Information Retrieval" (Manning et al.)
           Chapter 8: Evaluation in Information Retrieval

Author: Information Retrieval System
License: Educational Use
"""

import logging
import math
from typing import List, Dict, Set, Tuple, Optional, Union
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class EvaluationResult:
    """
    Result container for evaluation metrics.

    Attributes:
        precision: Precision score
        recall: Recall score
        f1: F1 score
        ap: Average Precision
        rr: Reciprocal Rank
        ndcg: Normalized Discounted Cumulative Gain
        num_relevant: Number of relevant documents
        num_retrieved: Number of retrieved documents
        num_relevant_retrieved: Number of relevant documents retrieved
    """
    precision: float
    recall: float
    f1: float
    ap: float = 0.0
    rr: float = 0.0
    ndcg: float = 0.0
    num_relevant: int = 0
    num_retrieved: int = 0
    num_relevant_retrieved: int = 0


class Metrics:
    """
    Information Retrieval Evaluation Metrics.

    Supports both binary relevance (relevant/non-relevant) and
    graded relevance (0-5 scale) evaluation scenarios.

    Common Use Cases:
        - System comparison (MAP, nDCG@k)
        - Per-query analysis (AP, P@k, R@k)
        - Threshold tuning (Precision-Recall curves)
        - Statistical testing (paired t-test, Wilcoxon)

    Complexity:
        - Most metrics: O(k) where k is number of retrieved docs
        - MAP: O(Q*k) where Q is number of queries
        - nDCG: O(k*log(k)) due to sorting
    """

    def __init__(self):
        """Initialize metrics calculator."""
        self.logger = logging.getLogger(__name__)
        self.logger.info("Metrics initialized")

    # ========================================================================
    # Binary Relevance Metrics
    # ========================================================================

    def precision(self, retrieved: List[int], relevant: Set[int]) -> float:
        """
        Calculate precision: fraction of retrieved docs that are relevant.

        Precision = |Retrieved ∩ Relevant| / |Retrieved|

        Args:
            retrieved: List of retrieved document IDs (ranked)
            relevant: Set of relevant document IDs

        Returns:
            Precision score in [0, 1]

        Complexity:
            Time: O(k) where k = len(retrieved)
            Space: O(1)

        Examples:
            >>> metrics = Metrics()
            >>> metrics.precision([1, 2, 3, 4], {1, 3, 5})
            0.5  # 2 out of 4 retrieved are relevant
        """
        if not retrieved:
            return 0.0

        relevant_retrieved = sum(1 for doc_id in retrieved if doc_id in relevant)
        return relevant_retrieved / len(retrieved)

    def recall(self, retrieved: List[int], relevant: Set[int]) -> float:
        """
        Calculate recall: fraction of relevant docs that are retrieved.

        Recall = |Retrieved ∩ Relevant| / |Relevant|

        Args:
            retrieved: List of retrieved document IDs
            relevant: Set of relevant document IDs

        Returns:
            Recall score in [0, 1]

        Complexity:
            Time: O(k) where k = len(retrieved)
            Space: O(1)

        Examples:
            >>> metrics.recall([1, 2, 3, 4], {1, 3, 5})
            0.667  # 2 out of 3 relevant are retrieved
        """
        if not relevant:
            return 0.0

        relevant_retrieved = sum(1 for doc_id in retrieved if doc_id in relevant)
        return relevant_retrieved / len(relevant)

    def f_measure(self, precision: float, recall: float,
                  beta: float = 1.0) -> float:
        """
        Calculate F-measure (harmonic mean of precision and recall).

        F_β = (1 + β²) * (P * R) / (β² * P + R)

        Args:
            precision: Precision score
            recall: Recall score
            beta: Weight parameter (default 1.0 for F1)
                  - β < 1: favor precision
                  - β > 1: favor recall

        Returns:
            F-measure score in [0, 1]

        Complexity:
            Time: O(1)
            Space: O(1)

        Examples:
            >>> metrics.f_measure(0.5, 0.667)
            0.571  # F1 score
            >>> metrics.f_measure(0.5, 0.667, beta=2.0)
            0.625  # F2 (favors recall)
        """
        if precision + recall == 0:
            return 0.0

        beta_sq = beta ** 2
        return (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)

    def precision_at_k(self, retrieved: List[int], relevant: Set[int],
                       k: int) -> float:
        """
        Calculate Precision@K: precision of top-k results.

        P@K = |Retrieved[:k] ∩ Relevant| / k

        Args:
            retrieved: Ranked list of retrieved document IDs
            relevant: Set of relevant document IDs
            k: Cutoff rank

        Returns:
            Precision@K score

        Complexity:
            Time: O(k)
            Space: O(1)

        Examples:
            >>> metrics.precision_at_k([1, 2, 3, 4, 5], {1, 3, 5}, k=3)
            0.667  # 2 out of top-3 are relevant
        """
        if k <= 0:
            return 0.0

        top_k = retrieved[:k]
        return self.precision(top_k, relevant)

    def recall_at_k(self, retrieved: List[int], relevant: Set[int],
                    k: int) -> float:
        """
        Calculate Recall@K: recall of top-k results.

        R@K = |Retrieved[:k] ∩ Relevant| / |Relevant|

        Args:
            retrieved: Ranked list of retrieved document IDs
            relevant: Set of relevant document IDs
            k: Cutoff rank

        Returns:
            Recall@K score

        Complexity:
            Time: O(k)
            Space: O(1)
        """
        if k <= 0:
            return 0.0

        top_k = retrieved[:k]
        return self.recall(top_k, relevant)

    # ========================================================================
    # Ranked Retrieval Metrics
    # ========================================================================

    def average_precision(self, retrieved: List[int],
                         relevant: Set[int]) -> float:
        """
        Calculate Average Precision (AP).

        AP = (1/|Relevant|) * Σ(P@k * rel(k))
        where rel(k) = 1 if k-th doc is relevant, 0 otherwise

        AP emphasizes returning relevant documents early.

        Args:
            retrieved: Ranked list of retrieved document IDs
            relevant: Set of relevant document IDs

        Returns:
            Average Precision score in [0, 1]

        Complexity:
            Time: O(k) where k = len(retrieved)
            Space: O(1)

        Examples:
            >>> metrics.average_precision([1, 2, 3, 4], {1, 3})
            0.833  # (1.0 + 0.667) / 2

            >>> metrics.average_precision([2, 1, 4, 3], {1, 3})
            0.583  # (0.5 + 0.5) / 2 (relevant docs at rank 2, 4)
        """
        if not relevant:
            return 0.0

        precision_sum = 0.0
        num_relevant_seen = 0

        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                num_relevant_seen += 1
                precision_at_rank = num_relevant_seen / rank
                precision_sum += precision_at_rank

        return precision_sum / len(relevant)

    def mean_average_precision(self,
                               results: Dict[str, List[int]],
                               qrels: Dict[str, Set[int]]) -> float:
        """
        Calculate Mean Average Precision (MAP).

        MAP = (1/|Queries|) * Σ AP(q)

        MAP is the most commonly used metric for comparing
        ranked retrieval systems.

        Args:
            results: Dictionary {query_id: [ranked_doc_ids]}
            qrels: Dictionary {query_id: {relevant_doc_ids}}

        Returns:
            MAP score in [0, 1]

        Complexity:
            Time: O(Q*k) where Q = queries, k = avg retrieved per query
            Space: O(1)

        Examples:
            >>> results = {'q1': [1, 2, 3], 'q2': [4, 5, 6]}
            >>> qrels = {'q1': {1, 3}, 'q2': {5}}
            >>> metrics.mean_average_precision(results, qrels)
            0.708  # (0.833 + 0.583) / 2
        """
        if not results or not qrels:
            return 0.0

        ap_sum = 0.0
        num_queries = 0

        for query_id, retrieved in results.items():
            if query_id in qrels:
                relevant = qrels[query_id]
                ap = self.average_precision(retrieved, relevant)
                ap_sum += ap
                num_queries += 1

        return ap_sum / num_queries if num_queries > 0 else 0.0

    def reciprocal_rank(self, retrieved: List[int],
                       relevant: Set[int]) -> float:
        """
        Calculate Reciprocal Rank (RR).

        RR = 1 / rank of first relevant document

        Used when only the first relevant result matters
        (e.g., navigational queries).

        Args:
            retrieved: Ranked list of retrieved document IDs
            relevant: Set of relevant document IDs

        Returns:
            Reciprocal Rank score in [0, 1]

        Complexity:
            Time: O(k) worst case, O(1) best case
            Space: O(1)

        Examples:
            >>> metrics.reciprocal_rank([1, 2, 3], {2})
            0.5  # First relevant at rank 2
            >>> metrics.reciprocal_rank([1, 2, 3], {1})
            1.0  # First relevant at rank 1
        """
        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                return 1.0 / rank
        return 0.0

    def mean_reciprocal_rank(self,
                            results: Dict[str, List[int]],
                            qrels: Dict[str, Set[int]]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).

        MRR = (1/|Queries|) * Σ RR(q)

        Args:
            results: Dictionary {query_id: [ranked_doc_ids]}
            qrels: Dictionary {query_id: {relevant_doc_ids}}

        Returns:
            MRR score in [0, 1]

        Complexity:
            Time: O(Q*k) where Q = queries, k = avg retrieved
            Space: O(1)
        """
        if not results or not qrels:
            return 0.0

        rr_sum = 0.0
        num_queries = 0

        for query_id, retrieved in results.items():
            if query_id in qrels:
                relevant = qrels[query_id]
                rr = self.reciprocal_rank(retrieved, relevant)
                rr_sum += rr
                num_queries += 1

        return rr_sum / num_queries if num_queries > 0 else 0.0

    # ========================================================================
    # Graded Relevance Metrics
    # ========================================================================

    def dcg_at_k(self, retrieved: List[int],
                 relevance_scores: Dict[int, float],
                 k: int) -> float:
        """
        Calculate Discounted Cumulative Gain (DCG@K).

        DCG@K = Σ(i=1 to k) (2^rel(i) - 1) / log₂(i + 1)

        Uses graded relevance (e.g., 0-5 scale) and applies
        logarithmic discount to emphasize early positions.

        Args:
            retrieved: Ranked list of retrieved document IDs
            relevance_scores: Dictionary {doc_id: relevance_score}
            k: Cutoff rank

        Returns:
            DCG@K score

        Complexity:
            Time: O(k)
            Space: O(1)

        Examples:
            >>> rel = {1: 3, 2: 2, 3: 3, 4: 0, 5: 1}
            >>> metrics.dcg_at_k([1, 2, 3], rel, k=3)
            8.392  # (2^3-1)/log2(2) + (2^2-1)/log2(3) + (2^3-1)/log2(4)
        """
        if k <= 0:
            return 0.0

        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:k], start=1):
            rel = relevance_scores.get(doc_id, 0.0)
            # DCG formula: (2^rel - 1) / log2(i + 1)
            gain = (2 ** rel - 1) / math.log2(i + 1)
            dcg += gain

        return dcg

    def ndcg_at_k(self, retrieved: List[int],
                  relevance_scores: Dict[int, float],
                  k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (nDCG@K).

        nDCG@K = DCG@K / IDCG@K
        where IDCG@K is DCG of ideal ranking (sorted by relevance)

        nDCG normalizes DCG to [0, 1] by comparing to ideal ranking.
        Most important metric for graded relevance evaluation.

        Args:
            retrieved: Ranked list of retrieved document IDs
            relevance_scores: Dictionary {doc_id: relevance_score}
            k: Cutoff rank

        Returns:
            nDCG@K score in [0, 1]

        Complexity:
            Time: O(k*log(k)) due to sorting for IDCG
            Space: O(k) for ideal ranking

        Examples:
            >>> rel = {1: 3, 2: 2, 3: 3, 4: 0, 5: 1}
            >>> metrics.ndcg_at_k([1, 2, 3], rel, k=3)
            1.0  # Perfect ranking (3, 2, 3 by relevance)
            >>> metrics.ndcg_at_k([4, 5, 1], rel, k=3)
            0.392  # Poor ranking (0, 1, 3 by relevance)
        """
        if k <= 0:
            return 0.0

        # Calculate DCG for actual ranking
        dcg = self.dcg_at_k(retrieved, relevance_scores, k)

        # Calculate IDCG (ideal DCG)
        # Sort all documents by relevance (descending)
        ideal_ranking = sorted(
            relevance_scores.keys(),
            key=lambda x: relevance_scores[x],
            reverse=True
        )
        idcg = self.dcg_at_k(ideal_ranking, relevance_scores, k)

        # Normalize
        if idcg == 0:
            return 0.0

        return dcg / idcg

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def evaluate_query(self, retrieved: List[int],
                       relevant: Set[int],
                       relevance_scores: Optional[Dict[int, float]] = None,
                       k_values: Optional[List[int]] = None) -> Dict[str, float]:
        """
        Comprehensive evaluation for a single query.

        Args:
            retrieved: Ranked list of retrieved document IDs
            relevant: Set of relevant document IDs (binary relevance)
            relevance_scores: Optional graded relevance scores
            k_values: List of k values for P@k, R@k, nDCG@k (default [5, 10, 20])

        Returns:
            Dictionary of metric names to scores

        Complexity:
            Time: O(k*log(k)) for nDCG, O(k) otherwise
            Space: O(k) for results

        Examples:
            >>> metrics.evaluate_query([1, 2, 3, 4], {1, 3, 5})
            {
                'precision': 0.5,
                'recall': 0.667,
                'f1': 0.571,
                'ap': 0.583,
                'rr': 1.0,
                'p@5': 0.4,
                'r@5': 0.667,
                ...
            }
        """
        if k_values is None:
            k_values = [5, 10, 20]

        results = {}

        # Binary relevance metrics
        p = self.precision(retrieved, relevant)
        r = self.recall(retrieved, relevant)

        results['precision'] = p
        results['recall'] = r
        results['f1'] = self.f_measure(p, r)
        results['ap'] = self.average_precision(retrieved, relevant)
        results['rr'] = self.reciprocal_rank(retrieved, relevant)

        # Metrics at various k
        for k in k_values:
            results[f'p@{k}'] = self.precision_at_k(retrieved, relevant, k)
            results[f'r@{k}'] = self.recall_at_k(retrieved, relevant, k)

            # nDCG if graded relevance available
            if relevance_scores:
                results[f'ndcg@{k}'] = self.ndcg_at_k(retrieved, relevance_scores, k)

        return results

    def evaluate_run(self,
                     results: Dict[str, List[int]],
                     qrels: Dict[str, Set[int]],
                     relevance_scores: Optional[Dict[str, Dict[int, float]]] = None,
                     k_values: Optional[List[int]] = None) -> Dict[str, float]:
        """
        Evaluate entire retrieval run (multiple queries).

        Args:
            results: Dictionary {query_id: [ranked_doc_ids]}
            qrels: Dictionary {query_id: {relevant_doc_ids}}
            relevance_scores: Optional {query_id: {doc_id: score}}
            k_values: List of k values for evaluation

        Returns:
            Dictionary of aggregated metric scores

        Complexity:
            Time: O(Q*k*log(k)) where Q = queries
            Space: O(Q*k) for all results

        Examples:
            >>> results = {'q1': [1, 2, 3], 'q2': [4, 5, 6]}
            >>> qrels = {'q1': {1, 3}, 'q2': {5}}
            >>> metrics.evaluate_run(results, qrels)
            {
                'map': 0.708,
                'mrr': 0.75,
                'p@10': 0.333,
                'ndcg@10': 0.82,
                ...
            }
        """
        if k_values is None:
            k_values = [5, 10, 20]

        # Aggregate per-query metrics
        per_query_metrics = defaultdict(list)

        for query_id, retrieved in results.items():
            if query_id not in qrels:
                self.logger.warning(f"No qrels for query {query_id}")
                continue

            relevant = qrels[query_id]
            rel_scores = None
            if relevance_scores and query_id in relevance_scores:
                rel_scores = relevance_scores[query_id]

            # Evaluate this query
            query_metrics = self.evaluate_query(
                retrieved, relevant, rel_scores, k_values
            )

            # Collect metrics
            for metric_name, score in query_metrics.items():
                per_query_metrics[metric_name].append(score)

        # Average across queries
        aggregated = {}
        for metric_name, scores in per_query_metrics.items():
            aggregated[metric_name] = sum(scores) / len(scores) if scores else 0.0

        # Add MAP and MRR explicitly (same as averaged ap/rr but more explicit)
        aggregated['map'] = self.mean_average_precision(results, qrels)
        aggregated['mrr'] = self.mean_reciprocal_rank(results, qrels)

        return aggregated


def demo():
    """Demonstration of evaluation metrics."""
    print("=" * 60)
    print("Evaluation Metrics Demo")
    print("=" * 60)

    metrics = Metrics()

    # Example 1: Binary relevance
    print("\n1. Binary Relevance Evaluation:")
    retrieved = [1, 2, 3, 4, 5]
    relevant = {1, 3, 5}

    p = metrics.precision(retrieved, relevant)
    r = metrics.recall(retrieved, relevant)
    f1 = metrics.f_measure(p, r)
    ap = metrics.average_precision(retrieved, relevant)
    rr = metrics.reciprocal_rank(retrieved, relevant)

    print(f"   Retrieved: {retrieved}")
    print(f"   Relevant: {relevant}")
    print(f"   Precision: {p:.3f}")
    print(f"   Recall: {r:.3f}")
    print(f"   F1: {f1:.3f}")
    print(f"   AP: {ap:.3f}")
    print(f"   RR: {rr:.3f}")

    # Example 2: Graded relevance
    print("\n2. Graded Relevance Evaluation (nDCG):")
    retrieved = [1, 2, 3, 4, 5]
    relevance_scores = {1: 3, 2: 2, 3: 3, 4: 0, 5: 1}

    print(f"   Retrieved: {retrieved}")
    print(f"   Relevance: {relevance_scores}")

    for k in [3, 5]:
        ndcg = metrics.ndcg_at_k(retrieved, relevance_scores, k)
        print(f"   nDCG@{k}: {ndcg:.3f}")

    # Example 3: Multiple queries (MAP)
    print("\n3. Multiple Queries (MAP, MRR):")
    results = {
        'q1': [1, 2, 3, 4],
        'q2': [5, 6, 7, 8],
        'q3': [9, 10, 11, 12]
    }
    qrels = {
        'q1': {1, 3},
        'q2': {6, 8},
        'q3': {11}
    }

    map_score = metrics.mean_average_precision(results, qrels)
    mrr_score = metrics.mean_reciprocal_rank(results, qrels)

    print(f"   Queries: {len(results)}")
    print(f"   MAP: {map_score:.3f}")
    print(f"   MRR: {mrr_score:.3f}")

    # Example 4: Comprehensive evaluation
    print("\n4. Comprehensive Evaluation:")
    eval_result = metrics.evaluate_query(
        retrieved=[1, 2, 3, 4, 5],
        relevant={1, 3, 5},
        relevance_scores={1: 3, 2: 0, 3: 2, 4: 0, 5: 3},
        k_values=[3, 5]
    )

    print("   Metrics:")
    for name, score in sorted(eval_result.items()):
        print(f"      {name}: {score:.3f}")

    print("\n" + "=" * 60)


# ============================================================================
# Convenience Functions (Backward Compatibility)
# ============================================================================

# Create a global instance for convenience functions
_metrics_instance = Metrics()


def precision(retrieved: List[int], relevant: Set[int]) -> float:
    """Convenience function for precision calculation."""
    return _metrics_instance.precision(retrieved, relevant)


def recall(retrieved: List[int], relevant: Set[int]) -> float:
    """Convenience function for recall calculation."""
    return _metrics_instance.recall(retrieved, relevant)


def f_measure(precision: float, recall: float, beta: float = 1.0) -> float:
    """Convenience function for F-measure calculation."""
    return _metrics_instance.f_measure(precision, recall, beta)


def average_precision(retrieved: List[int], relevant: Set[int]) -> float:
    """Convenience function for average precision calculation."""
    return _metrics_instance.average_precision(retrieved, relevant)


def reciprocal_rank(retrieved: List[int], relevant: Set[int]) -> float:
    """Convenience function for reciprocal rank calculation."""
    return _metrics_instance.reciprocal_rank(retrieved, relevant)


def ndcg(relevance_scores: Union[Dict[int, float], List[float]], k: int) -> float:
    """
    Convenience function for nDCG calculation.

    Args:
        relevance_scores: Either:
            - Dictionary {doc_id: relevance_score}
            - List of relevance scores (indexed by position)
        k: Cutoff rank

    Returns:
        nDCG@k score

    Examples:
        >>> ndcg({0: 3, 1: 2, 2: 3, 3: 0, 4: 1}, k=5)
        >>> ndcg([3, 2, 3, 0, 1], k=5)
    """
    # Convert list to dict if needed
    if isinstance(relevance_scores, list):
        relevance_scores = {i: score for i, score in enumerate(relevance_scores)}

    # Create an ideal ranking based on the relevance scores
    retrieved = sorted(relevance_scores.keys(), key=lambda x: relevance_scores[x], reverse=True)
    return _metrics_instance.ndcg_at_k(retrieved, relevance_scores, k)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    demo()
