"""
Keyword Extraction Evaluator

This module provides evaluation metrics for keyword extraction algorithms,
supporting both supervised (with ground truth) and unsupervised evaluation.

Key Metrics:
    - Precision@K: Ratio of correct keywords in top-k
    - Recall@K: Ratio of found keywords from ground truth
    - F1@K: Harmonic mean of Precision and Recall
    - MAP (Mean Average Precision): Quality of ranking
    - MRR (Mean Reciprocal Rank): Position of first relevant keyword
    - nDCG@K: Normalized Discounted Cumulative Gain

Unsupervised Metrics:
    - Diversity: Lexical diversity of extracted keywords
    - Coverage: Text coverage by keywords
    - Coherence: Semantic coherence of keyword set

References:
    Manning et al. (2008). "Introduction to Information Retrieval"
    Hasan & Ng (2014). "Automatic Keyphrase Extraction: A Survey"

Author: Information Retrieval System
License: Educational Use
"""

from typing import List, Set, Dict, Optional, Tuple
import logging
from dataclasses import dataclass
from collections import defaultdict
import math

# Import numpy for numerical computations
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logging.warning("NumPy not available. Some metrics may be unavailable.")


@dataclass
class EvaluationResult:
    """
    Evaluation result for keyword extraction.

    Attributes:
        precision_at_k: Precision at different k values
        recall_at_k: Recall at different k values
        f1_at_k: F1 score at different k values
        map_score: Mean Average Precision
        mrr: Mean Reciprocal Rank
        ndcg_at_k: Normalized Discounted Cumulative Gain at k
        diversity: Diversity score (unsupervised)
        coverage: Coverage score (unsupervised)
    """
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    f1_at_k: Dict[int, float]
    map_score: float
    mrr: float
    ndcg_at_k: Dict[int, float]
    diversity: Optional[float] = None
    coverage: Optional[float] = None

    def __repr__(self):
        return (
            f"EvaluationResult(\n"
            f"  P@5={self.precision_at_k.get(5, 0.0):.4f}, "
            f"  R@5={self.recall_at_k.get(5, 0.0):.4f}, "
            f"  F1@5={self.f1_at_k.get(5, 0.0):.4f}\n"
            f"  MAP={self.map_score:.4f}, "
            f"  MRR={self.mrr:.4f}, "
            f"  nDCG@5={self.ndcg_at_k.get(5, 0.0):.4f}\n"
            f")"
        )


class KeywordEvaluator:
    """
    Evaluator for keyword extraction algorithms.

    Supports both supervised evaluation (with ground truth keywords)
    and unsupervised quality metrics.

    Attributes:
        k_values: List of k values for P@K, R@K, F1@K, nDCG@K
        strict_match: Use exact string matching (vs. fuzzy matching)
        case_sensitive: Case-sensitive keyword comparison

    Examples:
        >>> evaluator = KeywordEvaluator(k_values=[5, 10, 15])
        >>> extracted = ['機器學習', '深度學習', '神經網路', '人工智慧', '資料科學']
        >>> ground_truth = ['機器學習', '人工智慧', '深度學習', '資料分析']
        >>> result = evaluator.evaluate(extracted, ground_truth)
        >>> print(f"P@5: {result.precision_at_k[5]:.4f}")
        P@5: 0.6000
    """

    def __init__(self,
                 k_values: List[int] = [1, 3, 5, 10, 15],
                 strict_match: bool = True,
                 case_sensitive: bool = False):
        """
        Initialize keyword evaluator.

        Args:
            k_values: List of k values for evaluation metrics
            strict_match: Use exact string matching (True) or fuzzy (False)
            case_sensitive: Perform case-sensitive comparison

        """
        self.logger = logging.getLogger(__name__)
        self.k_values = sorted(k_values)
        self.strict_match = strict_match
        self.case_sensitive = case_sensitive

        self.logger.info(
            f"KeywordEvaluator initialized: k_values={k_values}, "
            f"strict_match={strict_match}, case_sensitive={case_sensitive}"
        )

    # ========================================================================
    # Supervised Evaluation (with Ground Truth)
    # ========================================================================

    def evaluate(self,
                 extracted: List[str],
                 ground_truth: List[str],
                 text: Optional[str] = None) -> EvaluationResult:
        """
        Evaluate extracted keywords against ground truth.

        Args:
            extracted: List of extracted keywords (in ranked order)
            ground_truth: List of ground truth keywords
            text: Original text (for unsupervised metrics)

        Returns:
            EvaluationResult with all metrics

        Complexity:
            Time: O(k×g) where k=extracted, g=ground_truth
            Space: O(k)

        Examples:
            >>> evaluator = KeywordEvaluator()
            >>> extracted = ['keyword1', 'keyword2', 'keyword3']
            >>> ground_truth = ['keyword1', 'keyword3', 'keyword4']
            >>> result = evaluator.evaluate(extracted, ground_truth)
        """
        # Normalize keywords
        extracted_norm = self._normalize_keywords(extracted)
        ground_truth_norm = self._normalize_keywords(ground_truth)
        ground_truth_set = set(ground_truth_norm)

        # Calculate Precision@K, Recall@K, F1@K
        precision_at_k = {}
        recall_at_k = {}
        f1_at_k = {}

        for k in self.k_values:
            if k > len(extracted_norm):
                continue

            p_k = self._precision_at_k(extracted_norm, ground_truth_set, k)
            r_k = self._recall_at_k(extracted_norm, ground_truth_set, k)
            f1_k = self._f1_score(p_k, r_k)

            precision_at_k[k] = p_k
            recall_at_k[k] = r_k
            f1_at_k[k] = f1_k

        # Calculate MAP
        map_score = self._mean_average_precision(extracted_norm, ground_truth_set)

        # Calculate MRR
        mrr = self._mean_reciprocal_rank(extracted_norm, ground_truth_set)

        # Calculate nDCG@K
        ndcg_at_k = {}
        for k in self.k_values:
            if k > len(extracted_norm):
                continue
            ndcg_at_k[k] = self._ndcg_at_k(extracted_norm, ground_truth_set, k)

        # Calculate unsupervised metrics if text provided
        diversity = None
        coverage = None
        if text:
            diversity = self._diversity_score(extracted)
            coverage = self._coverage_score(extracted, text)

        return EvaluationResult(
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            f1_at_k=f1_at_k,
            map_score=map_score,
            mrr=mrr,
            ndcg_at_k=ndcg_at_k,
            diversity=diversity,
            coverage=coverage
        )

    def _precision_at_k(self,
                        extracted: List[str],
                        ground_truth: Set[str],
                        k: int) -> float:
        """
        Calculate Precision@K.

        Precision@K = (# relevant in top-k) / k

        Args:
            extracted: Extracted keywords (normalized)
            ground_truth: Ground truth keywords set
            k: Cutoff position

        Returns:
            Precision@K score

        Complexity:
            Time: O(k)
        """
        if k == 0 or not extracted:
            return 0.0

        top_k = extracted[:k]
        relevant_count = sum(1 for kw in top_k if kw in ground_truth)

        return relevant_count / k

    def _recall_at_k(self,
                     extracted: List[str],
                     ground_truth: Set[str],
                     k: int) -> float:
        """
        Calculate Recall@K.

        Recall@K = (# relevant in top-k) / (# ground truth)

        Args:
            extracted: Extracted keywords (normalized)
            ground_truth: Ground truth keywords set
            k: Cutoff position

        Returns:
            Recall@K score

        Complexity:
            Time: O(k)
        """
        if len(ground_truth) == 0 or not extracted:
            return 0.0

        top_k = extracted[:k]
        relevant_count = sum(1 for kw in top_k if kw in ground_truth)

        return relevant_count / len(ground_truth)

    def _f1_score(self, precision: float, recall: float) -> float:
        """
        Calculate F1 score from precision and recall.

        F1 = 2 × (P × R) / (P + R)

        Args:
            precision: Precision score
            recall: Recall score

        Returns:
            F1 score
        """
        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)

    def _mean_average_precision(self,
                                extracted: List[str],
                                ground_truth: Set[str]) -> float:
        """
        Calculate Mean Average Precision (MAP).

        MAP = (1/|R|) × Σ_{k=1}^{n} P(k) × rel(k)

        where:
            - R = number of relevant keywords
            - P(k) = precision at position k
            - rel(k) = 1 if k-th keyword is relevant, 0 otherwise

        Args:
            extracted: Extracted keywords (normalized)
            ground_truth: Ground truth keywords set

        Returns:
            MAP score

        Complexity:
            Time: O(n) where n = len(extracted)
        """
        if len(ground_truth) == 0 or not extracted:
            return 0.0

        relevant_count = 0
        precision_sum = 0.0

        for i, keyword in enumerate(extracted, 1):
            if keyword in ground_truth:
                relevant_count += 1
                precision_at_i = relevant_count / i
                precision_sum += precision_at_i

        if relevant_count == 0:
            return 0.0

        return precision_sum / len(ground_truth)

    def _mean_reciprocal_rank(self,
                             extracted: List[str],
                             ground_truth: Set[str]) -> float:
        """
        Calculate Mean Reciprocal Rank (MRR).

        MRR = 1 / rank_of_first_relevant

        Args:
            extracted: Extracted keywords (normalized)
            ground_truth: Ground truth keywords set

        Returns:
            MRR score

        Complexity:
            Time: O(n) in worst case
        """
        for i, keyword in enumerate(extracted, 1):
            if keyword in ground_truth:
                return 1.0 / i

        return 0.0  # No relevant keyword found

    def _ndcg_at_k(self,
                  extracted: List[str],
                  ground_truth: Set[str],
                  k: int) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain at K.

        DCG@K = Σ_{i=1}^{k} rel_i / log2(i+1)
        nDCG@K = DCG@K / IDCG@K

        where rel_i = 1 if relevant, 0 otherwise

        Args:
            extracted: Extracted keywords (normalized)
            ground_truth: Ground truth keywords set
            k: Cutoff position

        Returns:
            nDCG@K score

        Complexity:
            Time: O(k)
        """
        if k == 0 or not extracted:
            return 0.0

        # Calculate DCG@K
        dcg = 0.0
        for i, keyword in enumerate(extracted[:k], 1):
            relevance = 1.0 if keyword in ground_truth else 0.0
            dcg += relevance / math.log2(i + 1)

        # Calculate IDCG@K (ideal DCG)
        # Ideal ranking: all relevant keywords first
        ideal_k = min(k, len(ground_truth))
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_k + 1))

        if idcg == 0:
            return 0.0

        return dcg / idcg

    # ========================================================================
    # Unsupervised Evaluation (without Ground Truth)
    # ========================================================================

    def _diversity_score(self, keywords: List[str]) -> float:
        """
        Calculate diversity score of keywords.

        Measures lexical diversity using unique tokens ratio.

        Diversity = (# unique tokens in keywords) / (# total tokens)

        Args:
            keywords: List of keywords

        Returns:
            Diversity score (0.0 to 1.0)

        Complexity:
            Time: O(n×m) where n=keywords, m=avg keyword length
        """
        if not keywords:
            return 0.0

        # Tokenize all keywords into words
        all_tokens = []
        for kw in keywords:
            # Simple whitespace tokenization
            tokens = kw.split()
            all_tokens.extend(tokens)

        if not all_tokens:
            return 0.0

        unique_tokens = set(all_tokens)
        return len(unique_tokens) / len(all_tokens)

    def _coverage_score(self, keywords: List[str], text: str) -> float:
        """
        Calculate coverage score of keywords.

        Measures how much of the text is covered by keywords.

        Coverage = (# text tokens in keywords) / (# total text tokens)

        Args:
            keywords: List of keywords
            text: Original text

        Returns:
            Coverage score (0.0 to 1.0)

        Complexity:
            Time: O(n + m) where n=text length, m=keywords length
        """
        if not keywords or not text:
            return 0.0

        # Tokenize text and keywords
        text_tokens = text.split()
        keyword_tokens = set()
        for kw in keywords:
            keyword_tokens.update(kw.split())

        if not text_tokens:
            return 0.0

        # Count how many text tokens are in keywords
        covered = sum(1 for token in text_tokens if token in keyword_tokens)

        return covered / len(text_tokens)

    # ========================================================================
    # Batch Evaluation
    # ========================================================================

    def evaluate_batch(self,
                      extracted_list: List[List[str]],
                      ground_truth_list: List[List[str]],
                      texts: Optional[List[str]] = None) -> List[EvaluationResult]:
        """
        Evaluate multiple documents.

        Args:
            extracted_list: List of extracted keyword lists
            ground_truth_list: List of ground truth keyword lists
            texts: Optional list of original texts

        Returns:
            List of EvaluationResult objects

        Examples:
            >>> evaluator = KeywordEvaluator()
            >>> extracted_batch = [['kw1', 'kw2'], ['kw3', 'kw4']]
            >>> ground_truth_batch = [['kw1', 'kw3'], ['kw3', 'kw5']]
            >>> results = evaluator.evaluate_batch(extracted_batch, ground_truth_batch)
        """
        if len(extracted_list) != len(ground_truth_list):
            raise ValueError("Number of extracted and ground truth lists must match")

        if texts and len(texts) != len(extracted_list):
            raise ValueError("Number of texts must match extracted lists")

        results = []
        for i in range(len(extracted_list)):
            text = texts[i] if texts else None
            result = self.evaluate(extracted_list[i], ground_truth_list[i], text)
            results.append(result)

        self.logger.info(f"Evaluated {len(results)} documents")

        return results

    def aggregate_results(self, results: List[EvaluationResult]) -> EvaluationResult:
        """
        Aggregate results from multiple documents.

        Computes micro-averaged metrics across all documents.

        Args:
            results: List of EvaluationResult objects

        Returns:
            Aggregated EvaluationResult

        Examples:
            >>> evaluator = KeywordEvaluator()
            >>> results = [result1, result2, result3]
            >>> avg_result = evaluator.aggregate_results(results)
        """
        if not results:
            return EvaluationResult({}, {}, {}, 0.0, 0.0, {})

        # Aggregate P@K, R@K, F1@K
        precision_at_k = defaultdict(list)
        recall_at_k = defaultdict(list)
        f1_at_k = defaultdict(list)
        ndcg_at_k = defaultdict(list)

        for result in results:
            for k, p in result.precision_at_k.items():
                precision_at_k[k].append(p)
            for k, r in result.recall_at_k.items():
                recall_at_k[k].append(r)
            for k, f1 in result.f1_at_k.items():
                f1_at_k[k].append(f1)
            for k, ndcg in result.ndcg_at_k.items():
                ndcg_at_k[k].append(ndcg)

        # Compute averages
        avg_precision = {k: sum(vals) / len(vals) for k, vals in precision_at_k.items()}
        avg_recall = {k: sum(vals) / len(vals) for k, vals in recall_at_k.items()}
        avg_f1 = {k: sum(vals) / len(vals) for k, vals in f1_at_k.items()}
        avg_ndcg = {k: sum(vals) / len(vals) for k, vals in ndcg_at_k.items()}

        # Aggregate MAP and MRR
        avg_map = sum(r.map_score for r in results) / len(results)
        avg_mrr = sum(r.mrr for r in results) / len(results)

        # Aggregate diversity and coverage if available
        diversities = [r.diversity for r in results if r.diversity is not None]
        coverages = [r.coverage for r in results if r.coverage is not None]

        avg_diversity = sum(diversities) / len(diversities) if diversities else None
        avg_coverage = sum(coverages) / len(coverages) if coverages else None

        return EvaluationResult(
            precision_at_k=avg_precision,
            recall_at_k=avg_recall,
            f1_at_k=avg_f1,
            map_score=avg_map,
            mrr=avg_mrr,
            ndcg_at_k=avg_ndcg,
            diversity=avg_diversity,
            coverage=avg_coverage
        )

    # ========================================================================
    # Utilities
    # ========================================================================

    def _normalize_keywords(self, keywords: List[str]) -> List[str]:
        """
        Normalize keywords for comparison.

        Args:
            keywords: List of keywords

        Returns:
            Normalized keyword list
        """
        if self.case_sensitive:
            return keywords
        else:
            return [kw.lower() for kw in keywords]


def demo():
    """Demonstration of keyword extraction evaluation."""
    print("=" * 70)
    print("Keyword Extraction Evaluator Demo")
    print("=" * 70)

    # Sample data
    extracted = [
        '機器學習',  # Relevant
        '深度學習',  # Relevant
        '神經網路',  # Relevant
        '人工智慧',  # Relevant
        '資料科學',  # Not relevant
        '大數據',    # Not relevant
        '雲端運算',  # Not relevant
    ]

    ground_truth = [
        '機器學習',
        '人工智慧',
        '深度學習',
        '神經網路',
        '自然語言處理',  # Not extracted
    ]

    text = "機器學習是人工智慧的重要分支。深度學習使用神經網路來建立複雜的模型。"

    # Initialize evaluator
    print("\n[1] Initialize Evaluator")
    print("-" * 70)
    evaluator = KeywordEvaluator(k_values=[1, 3, 5, 7, 10])

    # Evaluate
    print("\n[2] Evaluate Extracted Keywords")
    print("-" * 70)
    print(f"Extracted ({len(extracted)}): {extracted[:5]}...")
    print(f"Ground truth ({len(ground_truth)}): {ground_truth}")

    result = evaluator.evaluate(extracted, ground_truth, text)

    # Print results
    print("\n[3] Evaluation Results")
    print("-" * 70)
    print("Precision@K:")
    for k in [1, 3, 5]:
        if k in result.precision_at_k:
            print(f"  P@{k:2d} = {result.precision_at_k[k]:.4f}")

    print("\nRecall@K:")
    for k in [1, 3, 5]:
        if k in result.recall_at_k:
            print(f"  R@{k:2d} = {result.recall_at_k[k]:.4f}")

    print("\nF1@K:")
    for k in [1, 3, 5]:
        if k in result.f1_at_k:
            print(f"  F1@{k:2d} = {result.f1_at_k[k]:.4f}")

    print(f"\nMAP   = {result.map_score:.4f}")
    print(f"MRR   = {result.mrr:.4f}")

    print("\nnDCG@K:")
    for k in [1, 3, 5]:
        if k in result.ndcg_at_k:
            print(f"  nDCG@{k:2d} = {result.ndcg_at_k[k]:.4f}")

    if result.diversity is not None:
        print(f"\nDiversity = {result.diversity:.4f}")
    if result.coverage is not None:
        print(f"Coverage  = {result.coverage:.4f}")

    # Batch evaluation
    print("\n[4] Batch Evaluation (3 Documents)")
    print("-" * 70)
    extracted_batch = [
        ['keyword1', 'keyword2', 'keyword3'],
        ['word1', 'word2', 'word3', 'word4'],
        ['term1', 'term2'],
    ]
    ground_truth_batch = [
        ['keyword1', 'keyword3', 'keyword4'],
        ['word1', 'word3', 'word5'],
        ['term1', 'term3', 'term4', 'term5'],
    ]

    batch_results = evaluator.evaluate_batch(extracted_batch, ground_truth_batch)
    avg_result = evaluator.aggregate_results(batch_results)

    print(f"Average P@3  = {avg_result.precision_at_k.get(3, 0.0):.4f}")
    print(f"Average R@3  = {avg_result.recall_at_k.get(3, 0.0):.4f}")
    print(f"Average F1@3 = {avg_result.f1_at_k.get(3, 0.0):.4f}")
    print(f"Average MAP  = {avg_result.map_score:.4f}")
    print(f"Average MRR  = {avg_result.mrr:.4f}")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    demo()
