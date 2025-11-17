#!/usr/bin/env python3
"""
Evaluate Retrieval Models for CNIRS Project

This script runs end-to-end evaluation of all retrieval models:
- Boolean Retrieval
- TF-IDF (Vector Space Model)
- BM25 (Probabilistic Ranking)
- BERT (Semantic Search)

It computes standard IR metrics:
- Mean Average Precision (MAP)
- Normalized Discounted Cumulative Gain (nDCG)
- Precision@K
- Recall@K
- F1@K

Usage:
    python scripts/evaluate_retrieval.py --index-dir data/indexes --queries data/evaluation/test_queries.txt --qrels data/evaluation/qrels.txt --output data/results/evaluation_results.json

Author: Information Retrieval System
License: Educational Use
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
from dataclasses import dataclass, asdict
import math

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.unified_retrieval import UnifiedRetrieval, SearchResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class QueryMetrics:
    """
    Evaluation metrics for a single query.

    Attributes:
        query_id: Query identifier
        average_precision: Average Precision (AP)
        ndcg: Normalized Discounted Cumulative Gain
        precision_at_k: Precision@K values
        recall_at_k: Recall@K values
        f1_at_k: F1@K values
        num_relevant: Number of relevant documents
        num_retrieved: Number of retrieved documents
    """
    query_id: str
    average_precision: float
    ndcg: float
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    f1_at_k: Dict[int, float]
    num_relevant: int
    num_retrieved: int


@dataclass
class ModelEvaluation:
    """
    Overall evaluation metrics for a retrieval model.

    Attributes:
        model_name: Name of the model
        mean_average_precision: MAP across all queries
        mean_ndcg: Mean nDCG across all queries
        mean_precision_at_k: Mean P@K for each K
        mean_recall_at_k: Mean R@K for each K
        mean_f1_at_k: Mean F1@K for each K
        query_metrics: Per-query metrics
        total_queries: Number of queries evaluated
        avg_response_time: Average query response time (seconds)
    """
    model_name: str
    mean_average_precision: float
    mean_ndcg: float
    mean_precision_at_k: Dict[int, float]
    mean_recall_at_k: Dict[int, float]
    mean_f1_at_k: Dict[int, float]
    query_metrics: List[QueryMetrics]
    total_queries: int
    avg_response_time: float


class RetrievalEvaluator:
    """
    Evaluator for retrieval models.

    Computes standard IR metrics including MAP, nDCG, P@K, R@K, F1@K.

    Attributes:
        retriever: UnifiedRetrieval instance
        queries: Dictionary of query_id -> query_text
        qrels: Dictionary of query_id -> {doc_id -> relevance}
        k_values: List of K values for P@K, R@K, F1@K
    """

    def __init__(self, index_dir: Path, k_values: List[int] = None):
        """
        Initialize evaluator.

        Args:
            index_dir: Path to indexes directory
            k_values: List of K values for metrics (default: [5, 10, 20])
        """
        logger.info("Initializing RetrievalEvaluator")

        # Initialize retrieval system
        self.retriever = UnifiedRetrieval()
        self.retriever.load_indexes(index_dir)

        # Metrics configuration
        self.k_values = k_values or [5, 10, 20]

        # Data containers
        self.queries = {}
        self.qrels = defaultdict(dict)

        logger.info(f"RetrievalEvaluator initialized with K={self.k_values}")

    def load_queries(self, queries_file: Path):
        """
        Load test queries from file.

        Format: query_id\\tquery_text

        Args:
            queries_file: Path to queries file
        """
        logger.info(f"Loading queries from {queries_file}")

        with open(queries_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split('\t')
                if len(parts) >= 2:
                    query_id = parts[0]
                    query_text = parts[1]
                    self.queries[query_id] = query_text

        logger.info(f"Loaded {len(self.queries)} queries")

    def load_qrels(self, qrels_file: Path):
        """
        Load relevance judgments (QRELS) from file.

        Format: query_id iteration doc_id relevance (TREC standard)
        Or: query_id\\tdoc_id\\trelevance (simplified)

        Args:
            qrels_file: Path to QRELS file
        """
        logger.info(f"Loading QRELS from {qrels_file}")

        with open(qrels_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split()  # Split by whitespace
                if len(parts) >= 4:
                    # TREC format: query_id iteration doc_id relevance
                    query_id = parts[0]
                    doc_id = parts[2]
                    relevance = int(parts[3])
                    self.qrels[query_id][doc_id] = relevance
                elif len(parts) >= 3:
                    # Simplified format: query_id doc_id relevance
                    query_id = parts[0]
                    doc_id = parts[1]
                    relevance = int(parts[2])
                    self.qrels[query_id][doc_id] = relevance

        total_judgments = sum(len(docs) for docs in self.qrels.values())
        logger.info(f"Loaded QRELS for {len(self.qrels)} queries ({total_judgments} judgments)")

    def compute_average_precision(self, results: List[SearchResult],
                                  relevant_docs: Dict[str, int]) -> float:
        """
        Compute Average Precision (AP).

        AP = (1/R) * sum(P(k) * rel(k)) where R is total relevant docs

        Args:
            results: List of search results
            relevant_docs: Dictionary of doc_id -> relevance

        Returns:
            Average Precision value

        Complexity:
            Time: O(n) where n is number of results
        """
        if not relevant_docs:
            return 0.0

        num_relevant = sum(1 for rel in relevant_docs.values() if rel > 0)
        if num_relevant == 0:
            return 0.0

        precision_sum = 0.0
        num_relevant_seen = 0

        for rank, result in enumerate(results, 1):
            if result.doc_id in relevant_docs and relevant_docs[result.doc_id] > 0:
                num_relevant_seen += 1
                precision_at_rank = num_relevant_seen / rank
                precision_sum += precision_at_rank

        return precision_sum / num_relevant

    def compute_dcg(self, results: List[SearchResult],
                    relevant_docs: Dict[str, int], k: int = None) -> float:
        """
        Compute Discounted Cumulative Gain (DCG).

        DCG = sum(rel_i / log2(i+1)) for i in 1..k

        Args:
            results: List of search results
            relevant_docs: Dictionary of doc_id -> relevance
            k: Cutoff rank (None for all results)

        Returns:
            DCG value
        """
        if k is None:
            k = len(results)

        dcg = 0.0
        for rank, result in enumerate(results[:k], 1):
            relevance = relevant_docs.get(result.doc_id, 0)
            dcg += relevance / math.log2(rank + 1)

        return dcg

    def compute_ndcg(self, results: List[SearchResult],
                     relevant_docs: Dict[str, int], k: int = None) -> float:
        """
        Compute Normalized Discounted Cumulative Gain (nDCG).

        nDCG = DCG / IDCG where IDCG is ideal DCG

        Args:
            results: List of search results
            relevant_docs: Dictionary of doc_id -> relevance
            k: Cutoff rank (None for all results)

        Returns:
            nDCG value (0.0 to 1.0)
        """
        dcg = self.compute_dcg(results, relevant_docs, k)

        # Compute ideal DCG (IDCG)
        ideal_ranking = sorted(relevant_docs.values(), reverse=True)
        if k is not None:
            ideal_ranking = ideal_ranking[:k]

        idcg = sum(rel / math.log2(rank + 2) for rank, rel in enumerate(ideal_ranking))

        if idcg == 0.0:
            return 0.0

        return dcg / idcg

    def compute_precision_at_k(self, results: List[SearchResult],
                               relevant_docs: Dict[str, int], k: int) -> float:
        """
        Compute Precision@K.

        P@K = (# relevant in top K) / K

        Args:
            results: List of search results
            relevant_docs: Dictionary of doc_id -> relevance
            k: Cutoff rank

        Returns:
            Precision@K value
        """
        if k == 0:
            return 0.0

        top_k = results[:k]
        num_relevant = sum(1 for r in top_k
                          if r.doc_id in relevant_docs and relevant_docs[r.doc_id] > 0)

        return num_relevant / k

    def compute_recall_at_k(self, results: List[SearchResult],
                           relevant_docs: Dict[str, int], k: int) -> float:
        """
        Compute Recall@K.

        R@K = (# relevant in top K) / (# total relevant)

        Args:
            results: List of search results
            relevant_docs: Dictionary of doc_id -> relevance
            k: Cutoff rank

        Returns:
            Recall@K value
        """
        num_total_relevant = sum(1 for rel in relevant_docs.values() if rel > 0)
        if num_total_relevant == 0:
            return 0.0

        top_k = results[:k]
        num_relevant = sum(1 for r in top_k
                          if r.doc_id in relevant_docs and relevant_docs[r.doc_id] > 0)

        return num_relevant / num_total_relevant

    def compute_f1_at_k(self, precision: float, recall: float) -> float:
        """
        Compute F1@K.

        F1@K = 2 * P@K * R@K / (P@K + R@K)

        Args:
            precision: Precision@K value
            recall: Recall@K value

        Returns:
            F1@K value
        """
        if precision + recall == 0:
            return 0.0

        return 2 * precision * recall / (precision + recall)

    def evaluate_query(self, query_id: str, query_text: str,
                      model: str, top_k: int = 100) -> QueryMetrics:
        """
        Evaluate a single query.

        Args:
            query_id: Query identifier
            query_text: Query text
            model: Retrieval model name
            top_k: Number of results to retrieve

        Returns:
            QueryMetrics for this query
        """
        # Get relevant documents
        relevant_docs = self.qrels.get(query_id, {})

        if not relevant_docs:
            logger.warning(f"No QRELS for query {query_id}")

        # Search
        results = self.retriever.search(query_text, model=model, top_k=top_k)

        # Compute metrics
        ap = self.compute_average_precision(results, relevant_docs)
        ndcg = self.compute_ndcg(results, relevant_docs)

        # Compute P@K, R@K, F1@K
        precision_at_k = {}
        recall_at_k = {}
        f1_at_k = {}

        for k in self.k_values:
            p_k = self.compute_precision_at_k(results, relevant_docs, k)
            r_k = self.compute_recall_at_k(results, relevant_docs, k)
            f1_k = self.compute_f1_at_k(p_k, r_k)

            precision_at_k[k] = p_k
            recall_at_k[k] = r_k
            f1_at_k[k] = f1_k

        return QueryMetrics(
            query_id=query_id,
            average_precision=ap,
            ndcg=ndcg,
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            f1_at_k=f1_at_k,
            num_relevant=sum(1 for rel in relevant_docs.values() if rel > 0),
            num_retrieved=len(results)
        )

    def evaluate_model(self, model: str, top_k: int = 100) -> ModelEvaluation:
        """
        Evaluate a retrieval model on all queries.

        Args:
            model: Model name ('boolean', 'tfidf', 'bm25', 'bert')
            top_k: Number of results to retrieve per query

        Returns:
            ModelEvaluation with aggregated metrics
        """
        logger.info(f"Evaluating model: {model.upper()}")

        query_metrics_list = []
        total_time = 0.0

        for query_id, query_text in self.queries.items():
            logger.info(f"  Query {query_id}: {query_text}")

            # Measure response time
            start_time = time.time()
            query_metrics = self.evaluate_query(query_id, query_text, model, top_k)
            elapsed_time = time.time() - start_time
            total_time += elapsed_time

            query_metrics_list.append(query_metrics)

            logger.info(f"    AP={query_metrics.average_precision:.4f}, "
                       f"nDCG={query_metrics.ndcg:.4f}, "
                       f"P@10={query_metrics.precision_at_k.get(10, 0):.4f}")

        # Compute mean metrics
        num_queries = len(query_metrics_list)

        mean_ap = sum(qm.average_precision for qm in query_metrics_list) / num_queries
        mean_ndcg = sum(qm.ndcg for qm in query_metrics_list) / num_queries

        mean_precision_at_k = {}
        mean_recall_at_k = {}
        mean_f1_at_k = {}

        for k in self.k_values:
            mean_precision_at_k[k] = sum(qm.precision_at_k[k] for qm in query_metrics_list) / num_queries
            mean_recall_at_k[k] = sum(qm.recall_at_k[k] for qm in query_metrics_list) / num_queries
            mean_f1_at_k[k] = sum(qm.f1_at_k[k] for qm in query_metrics_list) / num_queries

        avg_response_time = total_time / num_queries

        logger.info(f"Model {model.upper()}: MAP={mean_ap:.4f}, nDCG={mean_ndcg:.4f}, "
                   f"P@10={mean_precision_at_k.get(10, 0):.4f}")

        return ModelEvaluation(
            model_name=model,
            mean_average_precision=mean_ap,
            mean_ndcg=mean_ndcg,
            mean_precision_at_k=mean_precision_at_k,
            mean_recall_at_k=mean_recall_at_k,
            mean_f1_at_k=mean_f1_at_k,
            query_metrics=query_metrics_list,
            total_queries=num_queries,
            avg_response_time=avg_response_time
        )

    def evaluate_all_models(self, models: List[str], top_k: int = 100) -> Dict[str, ModelEvaluation]:
        """
        Evaluate all specified models.

        Args:
            models: List of model names
            top_k: Number of results per query

        Returns:
            Dictionary of model_name -> ModelEvaluation
        """
        results = {}

        for model in models:
            results[model] = self.evaluate_model(model, top_k)

        return results


def format_results_table(evaluations: Dict[str, ModelEvaluation]) -> str:
    """
    Format evaluation results as a text table.

    Args:
        evaluations: Dictionary of model evaluations

    Returns:
        Formatted table string
    """
    lines = []
    lines.append("=" * 100)
    lines.append("RETRIEVAL MODEL EVALUATION RESULTS")
    lines.append("=" * 100)
    lines.append("")

    # Header
    header = f"{'Model':<10} {'MAP':>8} {'nDCG':>8} {'P@5':>8} {'P@10':>8} {'P@20':>8} {'R@10':>8} {'F1@10':>8} {'Time(s)':>10}"
    lines.append(header)
    lines.append("-" * 100)

    # Rows
    for model_name, eval_result in evaluations.items():
        row = f"{model_name.upper():<10} "
        row += f"{eval_result.mean_average_precision:>8.4f} "
        row += f"{eval_result.mean_ndcg:>8.4f} "
        row += f"{eval_result.mean_precision_at_k.get(5, 0):>8.4f} "
        row += f"{eval_result.mean_precision_at_k.get(10, 0):>8.4f} "
        row += f"{eval_result.mean_precision_at_k.get(20, 0):>8.4f} "
        row += f"{eval_result.mean_recall_at_k.get(10, 0):>8.4f} "
        row += f"{eval_result.mean_f1_at_k.get(10, 0):>8.4f} "
        row += f"{eval_result.avg_response_time:>10.4f}"
        lines.append(row)

    lines.append("=" * 100)

    return "\n".join(lines)


def save_results(evaluations: Dict[str, ModelEvaluation], output_file: Path):
    """
    Save evaluation results to JSON file.

    Args:
        evaluations: Dictionary of model evaluations
        output_file: Output JSON file path
    """
    logger.info(f"Saving results to {output_file}")

    # Convert to serializable format
    results_dict = {}
    for model_name, eval_result in evaluations.items():
        results_dict[model_name] = {
            'model_name': eval_result.model_name,
            'mean_average_precision': eval_result.mean_average_precision,
            'mean_ndcg': eval_result.mean_ndcg,
            'mean_precision_at_k': eval_result.mean_precision_at_k,
            'mean_recall_at_k': eval_result.mean_recall_at_k,
            'mean_f1_at_k': eval_result.mean_f1_at_k,
            'total_queries': eval_result.total_queries,
            'avg_response_time': eval_result.avg_response_time,
            'query_metrics': [asdict(qm) for qm in eval_result.query_metrics]
        }

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=2)

    logger.info(f"Results saved to {output_file}")


def main():
    """Main evaluation workflow."""
    parser = argparse.ArgumentParser(
        description='Evaluate retrieval models on test queries',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate all models
  python scripts/evaluate_retrieval.py

  # Evaluate specific models
  python scripts/evaluate_retrieval.py --models tfidf bm25

  # Custom paths
  python scripts/evaluate_retrieval.py --index-dir data/indexes --queries data/evaluation/test_queries.txt --qrels data/evaluation/qrels.txt
        """
    )

    parser.add_argument('--index-dir', type=str, default='data/indexes',
                       help='Path to indexes directory (default: data/indexes)')
    parser.add_argument('--queries', type=str, default='data/evaluation/test_queries.txt',
                       help='Path to test queries file (default: data/evaluation/test_queries.txt)')
    parser.add_argument('--qrels', type=str, default='data/evaluation/qrels.txt',
                       help='Path to QRELS file (default: data/evaluation/qrels.txt)')
    parser.add_argument('--models', type=str, nargs='+',
                       default=['boolean', 'tfidf', 'bm25', 'bert'],
                       choices=['boolean', 'tfidf', 'bm25', 'bert'],
                       help='Models to evaluate (default: all)')
    parser.add_argument('--top-k', type=int, default=100,
                       help='Number of results to retrieve per query (default: 100)')
    parser.add_argument('--k-values', type=int, nargs='+', default=[5, 10, 20],
                       help='K values for P@K, R@K, F1@K metrics (default: 5 10 20)')
    parser.add_argument('--output', type=str, default='data/results/evaluation_results.json',
                       help='Output JSON file (default: data/results/evaluation_results.json)')
    parser.add_argument('--report', type=str, default='data/results/evaluation_report.txt',
                       help='Output text report (default: data/results/evaluation_report.txt)')

    args = parser.parse_args()

    # Convert paths
    index_dir = Path(args.index_dir)
    queries_file = Path(args.queries)
    qrels_file = Path(args.qrels)
    output_file = Path(args.output)
    report_file = Path(args.report)

    # Check inputs
    if not index_dir.exists():
        logger.error(f"Index directory not found: {index_dir}")
        return 1

    if not queries_file.exists():
        logger.error(f"Queries file not found: {queries_file}")
        return 1

    if not qrels_file.exists():
        logger.error(f"QRELS file not found: {qrels_file}")
        return 1

    # Initialize evaluator
    evaluator = RetrievalEvaluator(index_dir, k_values=args.k_values)

    # Load data
    evaluator.load_queries(queries_file)
    evaluator.load_qrels(qrels_file)

    # Evaluate models
    logger.info(f"Evaluating models: {', '.join(args.models)}")
    evaluations = evaluator.evaluate_all_models(args.models, top_k=args.top_k)

    # Display results
    print("\n" + format_results_table(evaluations))

    # Save results
    save_results(evaluations, output_file)

    # Save text report
    report_file.parent.mkdir(parents=True, exist_ok=True)
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(format_results_table(evaluations))
    logger.info(f"Text report saved to {report_file}")

    logger.info("Evaluation complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
