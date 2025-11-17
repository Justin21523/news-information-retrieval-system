#!/usr/bin/env python3
"""Evaluation CLI Tool

Command-line tool for evaluating IR system results using standard metrics.

Features:
    - Load results and qrels from various formats (JSON, TREC)
    - Calculate multiple evaluation metrics
    - Support binary and graded relevance
    - Per-query and aggregated evaluation
    - Export results to multiple formats (JSON, CSV, TXT)

Usage:
    # Basic evaluation
    python scripts/eval_run.py --results run.json --qrels qrels.txt

    # Specify metrics
    python scripts/eval_run.py --results run.json --qrels qrels.txt --metrics MAP,MRR,P@10,nDCG@10

    # Per-query breakdown
    python scripts/eval_run.py --results run.json --qrels qrels.txt --per-query

    # Export results
    python scripts/eval_run.py --results run.json --qrels qrels.txt --output eval_results.json

Author: Information Retrieval System
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Set, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ir.eval.metrics import Metrics


def load_results_json(filepath: str) -> Dict[str, List[int]]:
    """
    Load results from JSON file.

    Expected format:
    {
        "q1": [doc1, doc2, doc3, ...],
        "q2": [doc4, doc5, ...],
        ...
    }

    Args:
        filepath: Path to JSON file

    Returns:
        Dictionary {query_id: [ranked_doc_ids]}
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Convert to proper format if needed
    results = {}
    for query_id, doc_list in data.items():
        # Ensure doc_ids are integers
        if doc_list and isinstance(doc_list[0], dict):
            # Format: [{"doc_id": 1, "score": 0.9}, ...]
            results[query_id] = [item['doc_id'] for item in doc_list]
        else:
            # Format: [1, 2, 3, ...]
            results[query_id] = [int(doc_id) if not isinstance(doc_id, int) else doc_id
                                for doc_id in doc_list]

    return results


def load_results_trec(filepath: str) -> Dict[str, List[int]]:
    """
    Load results from TREC format file.

    TREC format:
    query_id Q0 doc_id rank score run_id

    Example:
    q1 Q0 doc1 1 0.95 run1
    q1 Q0 doc2 2 0.87 run1

    Args:
        filepath: Path to TREC format file

    Returns:
        Dictionary {query_id: [ranked_doc_ids]}
    """
    results = {}

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue

            query_id = parts[0]
            doc_id = parts[2]
            rank = int(parts[3])

            if query_id not in results:
                results[query_id] = []

            results[query_id].append((rank, doc_id))

    # Sort by rank and extract doc_ids
    for query_id in results:
        results[query_id].sort(key=lambda x: x[0])
        results[query_id] = [doc_id for _, doc_id in results[query_id]]

    return results


def load_qrels_json(filepath: str) -> Dict[str, Set[int]]:
    """
    Load qrels from JSON file.

    Expected format (binary relevance):
    {
        "q1": [doc1, doc2, ...],
        "q2": [doc4, doc5, ...],
        ...
    }

    Or (graded relevance):
    {
        "q1": {"doc1": 3, "doc2": 2, ...},
        "q2": {"doc4": 1, ...},
        ...
    }

    Args:
        filepath: Path to JSON file

    Returns:
        Dictionary {query_id: {relevant_doc_ids}}
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    qrels = {}
    for query_id, rel_data in data.items():
        if isinstance(rel_data, list):
            # Binary: list of doc_ids
            qrels[query_id] = set(int(doc_id) if not isinstance(doc_id, int) else doc_id
                                 for doc_id in rel_data)
        elif isinstance(rel_data, dict):
            # Graded: extract docs with relevance > 0
            qrels[query_id] = set(
                int(doc_id) if not isinstance(doc_id, int) else int(doc_id)
                for doc_id, score in rel_data.items() if score > 0
            )
        else:
            qrels[query_id] = set()

    return qrels


def load_qrels_trec(filepath: str) -> Dict[str, Set[int]]:
    """
    Load qrels from TREC format file.

    TREC qrels format:
    query_id 0 doc_id relevance

    Example:
    q1 0 doc1 1
    q1 0 doc2 0
    q1 0 doc3 1

    Args:
        filepath: Path to TREC format file

    Returns:
        Dictionary {query_id: {relevant_doc_ids}}
    """
    qrels = {}

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue

            query_id = parts[0]
            doc_id = parts[2]
            relevance = int(parts[3])

            if relevance > 0:
                if query_id not in qrels:
                    qrels[query_id] = set()
                qrels[query_id].add(doc_id)

    return qrels


def load_relevance_scores_json(filepath: str) -> Dict[str, Dict[int, float]]:
    """
    Load graded relevance scores from JSON.

    Expected format:
    {
        "q1": {"doc1": 3, "doc2": 2, ...},
        "q2": {"doc4": 1, ...},
        ...
    }

    Args:
        filepath: Path to JSON file

    Returns:
        Dictionary {query_id: {doc_id: relevance_score}}
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    relevance_scores = {}
    for query_id, scores in data.items():
        relevance_scores[query_id] = {
            int(doc_id) if not isinstance(doc_id, int) else int(doc_id): float(score)
            for doc_id, score in scores.items()
        }

    return relevance_scores


def save_results_json(results: Dict[str, float], filepath: str) -> None:
    """Save evaluation results to JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filepath}")


def save_results_csv(results: Dict[str, float], filepath: str) -> None:
    """Save evaluation results to CSV file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("metric,score\n")
        for metric, score in sorted(results.items()):
            f.write(f"{metric},{score:.6f}\n")
    print(f"Results saved to {filepath}")


def save_results_txt(results: Dict[str, float], filepath: str) -> None:
    """Save evaluation results to human-readable text file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("Evaluation Results\n")
        f.write("=" * 60 + "\n\n")

        for metric, score in sorted(results.items()):
            f.write(f"{metric:20s}: {score:.6f}\n")

        f.write("\n" + "=" * 60 + "\n")

    print(f"Results saved to {filepath}")


def print_results(results: Dict[str, float], title: str = "Evaluation Results") -> None:
    """Print evaluation results to console."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    for metric, score in sorted(results.items()):
        print(f"{metric:20s}: {score:.6f}")

    print("=" * 60)


def evaluate(args):
    """Main evaluation function."""
    # Load data
    print(f"Loading results from {args.results}...")

    if args.results.endswith('.json'):
        results = load_results_json(args.results)
    else:
        results = load_results_trec(args.results)

    print(f"  Loaded {len(results)} queries")

    print(f"Loading qrels from {args.qrels}...")

    if args.qrels.endswith('.json'):
        qrels = load_qrels_json(args.qrels)
    else:
        qrels = load_qrels_trec(args.qrels)

    print(f"  Loaded qrels for {len(qrels)} queries")

    # Load relevance scores if provided
    relevance_scores = None
    if args.relevance:
        print(f"Loading relevance scores from {args.relevance}...")
        relevance_scores = load_relevance_scores_json(args.relevance)
        print(f"  Loaded relevance scores for {len(relevance_scores)} queries")

    # Parse k values
    k_values = [int(k) for k in args.k_values.split(',')]

    # Initialize metrics
    metrics = Metrics()

    # Evaluate
    print("\nEvaluating...")
    eval_results = metrics.evaluate_run(
        results, qrels, relevance_scores, k_values
    )

    # Per-query evaluation if requested
    if args.per_query:
        print("\n" + "=" * 60)
        print("Per-Query Evaluation")
        print("=" * 60)

        for query_id in sorted(results.keys()):
            if query_id not in qrels:
                continue

            retrieved = results[query_id]
            relevant = qrels[query_id]
            rel_scores = None
            if relevance_scores and query_id in relevance_scores:
                rel_scores = relevance_scores[query_id]

            query_results = metrics.evaluate_query(
                retrieved, relevant, rel_scores, k_values
            )

            print(f"\nQuery: {query_id}")
            print(f"  Retrieved: {len(retrieved)} documents")
            print(f"  Relevant: {len(relevant)} documents")
            for metric, score in sorted(query_results.items()):
                print(f"  {metric}: {score:.4f}")

    # Print aggregated results
    print_results(eval_results, "Aggregated Evaluation Results")

    # Save results if output specified
    if args.output:
        output_path = Path(args.output)
        ext = output_path.suffix.lower()

        if ext == '.json':
            save_results_json(eval_results, args.output)
        elif ext == '.csv':
            save_results_csv(eval_results, args.output)
        elif ext == '.txt':
            save_results_txt(eval_results, args.output)
        else:
            print(f"Warning: Unknown output format '{ext}', saving as JSON")
            save_results_json(eval_results, args.output)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate IR system results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python scripts/eval_run.py --results run.json --qrels qrels.json

  # Specify k values
  python scripts/eval_run.py --results run.json --qrels qrels.json --k-values 5,10,20,100

  # Per-query breakdown
  python scripts/eval_run.py --results run.json --qrels qrels.json --per-query

  # With graded relevance (nDCG)
  python scripts/eval_run.py --results run.json --qrels qrels.json --relevance grades.json

  # Export results
  python scripts/eval_run.py --results run.json --qrels qrels.json --output eval_results.csv

Supported formats:
  - JSON: results and qrels as dictionaries
  - TREC: standard TREC format (results and qrels)
        """
    )

    parser.add_argument('--results', type=str, required=True,
                       help='Results file (JSON or TREC format)')
    parser.add_argument('--qrels', type=str, required=True,
                       help='Qrels file (JSON or TREC format)')
    parser.add_argument('--relevance', type=str,
                       help='Graded relevance scores (JSON format)')
    parser.add_argument('--k-values', type=str, default='5,10,20',
                       help='Comma-separated k values for P@k, R@k, nDCG@k (default: 5,10,20)')
    parser.add_argument('--per-query', action='store_true',
                       help='Show per-query evaluation breakdown')
    parser.add_argument('--output', type=str,
                       help='Output file for results (JSON, CSV, or TXT)')

    args = parser.parse_args()

    try:
        evaluate(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
