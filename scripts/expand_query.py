#!/usr/bin/env python3
"""Query Expansion CLI Tool

Command-line tool for expanding queries using Rocchio algorithm
and relevance feedback.

Features:
    - Pseudo-relevance feedback (automatic expansion)
    - Explicit relevance feedback (user-provided relevant docs)
    - Parameter tuning (α, β, γ)
    - Integration with VSM retrieval
    - Comparison of original vs expanded results

Usage:
    # Pseudo-relevance feedback (top-k documents)
    python scripts/expand_query.py --query "information retrieval" \
        --mode pseudo --index vsm_index.json --topk 10

    # Explicit relevance feedback
    python scripts/expand_query.py --query "information retrieval" \
        --mode explicit --index vsm_index.json --relevant relevant_docs.txt

    # Custom parameters
    python scripts/expand_query.py --query "search engine" \
        --mode pseudo --index vsm_index.json \
        --alpha 1.0 --beta 0.8 --gamma 0.2 \
        --max-terms 15

Author: Information Retrieval System
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Set

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ir.retrieval.vsm import VectorSpaceModel
from src.ir.ranking.rocchio import RocchioExpander
from src.ir.index.inverted_index import InvertedIndex


def load_relevant_docs(filepath: str) -> Set[int]:
    """
    Load relevant document IDs from file.

    Format: one doc_id per line (or JSON list)

    Args:
        filepath: Path to relevant docs file

    Returns:
        Set of relevant document IDs
    """
    relevant = set()

    try:
        # Try JSON first
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                relevant = set(int(doc_id) for doc_id in data)
            else:
                raise ValueError("JSON must be a list of doc IDs")
    except json.JSONDecodeError:
        # Fall back to text file (one ID per line)
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and line.isdigit():
                    relevant.add(int(line))

    return relevant


def expand_pseudo_feedback(args, vsm: VectorSpaceModel, expander: RocchioExpander):
    """Expand query using pseudo-relevance feedback."""
    print(f"Query: \"{args.query}\"")
    print(f"Mode: Pseudo-relevance feedback (top-{args.topk} documents)")
    print("=" * 60)

    # Step 1: Initial retrieval
    print("\n1. Initial Retrieval:")
    initial_result = vsm.search(args.query, topk=args.topk)

    print(f"   Retrieved {initial_result.num_results} documents")
    if args.verbose:
        for i, doc_id in enumerate(initial_result.doc_ids[:5], 1):
            score = initial_result.scores[doc_id]
            print(f"   {i}. Doc {doc_id}: {score:.4f}")

    # Step 2: Get document vectors for feedback
    print("\n2. Extracting Document Vectors:")
    top_doc_vectors = []
    for doc_id in initial_result.doc_ids[:args.topk]:
        doc_vec = vsm.get_document_vector(doc_id)
        if doc_vec:
            top_doc_vectors.append(doc_vec)

    print(f"   Extracted {len(top_doc_vectors)} document vectors")

    # Step 3: Build query vector
    print("\n3. Building Query Vector:")
    query_tokens = vsm.inverted_index.tokenizer(args.query)
    original_terms = set(query_tokens)

    from collections import defaultdict
    query_tf = defaultdict(int)
    for token in query_tokens:
        query_tf[token] += 1

    query_vector = vsm.term_weighting.vectorize(
        dict(query_tf),
        tf_scheme=vsm.query_tf_scheme,
        idf_scheme=vsm.query_idf_scheme,
        normalize=vsm.query_norm_scheme
    )

    print(f"   Original terms: {list(original_terms)}")
    print(f"   Query vector size: {len(query_vector)}")

    # Step 4: Expand query
    print("\n4. Query Expansion (Rocchio):")
    expanded = expander.expand_with_pseudo_feedback(
        query_vector,
        top_doc_vectors,
        num_relevant=args.num_relevant,
        num_nonrelevant=args.num_nonrelevant,
        original_terms=original_terms
    )

    print(f"   Relevant docs: {expanded.num_relevant}")
    print(f"   Non-relevant docs: {expanded.num_nonrelevant}")
    print(f"   Original terms: {len(expanded.original_terms)}")
    print(f"   Expanded terms: {len(expanded.expanded_terms)}")
    print(f"   Total terms: {len(expanded.all_terms)}")

    # Step 5: Show top expansion terms
    print("\n5. Top Expansion Terms:")
    top_terms = expander.get_top_expansion_terms(expanded, k=10)

    for i, (term, weight) in enumerate(top_terms, 1):
        print(f"   {i}. {term}: {weight:.4f}")

    # Step 6: Retrieve with expanded query
    if not args.no_rerank:
        print("\n6. Re-ranking with Expanded Query:")

        # Build expanded query string
        expanded_query_str = " ".join(expanded.all_terms)

        # Search with expanded query
        expanded_result = vsm.search(expanded_query_str, topk=args.topk)

        print(f"   Retrieved {expanded_result.num_results} documents")

        # Compare results
        print("\n7. Comparison:")
        initial_docs = set(initial_result.doc_ids[:10])
        expanded_docs = set(expanded_result.doc_ids[:10])

        new_docs = expanded_docs - initial_docs
        same_docs = initial_docs & expanded_docs

        print(f"   Same in top-10: {len(same_docs)}")
        print(f"   New in top-10: {len(new_docs)}")

        if new_docs and args.verbose:
            print(f"   New documents: {sorted(new_docs)[:5]}")


def expand_explicit_feedback(args, vsm: VectorSpaceModel, expander: RocchioExpander):
    """Expand query using explicit relevance feedback."""
    print(f"Query: \"{args.query}\"")
    print(f"Mode: Explicit relevance feedback")
    print("=" * 60)

    # Load relevant document IDs
    print("\n1. Loading Relevant Documents:")
    relevant_ids = load_relevant_docs(args.relevant)
    print(f"   Loaded {len(relevant_ids)} relevant document IDs")

    if args.verbose:
        print(f"   IDs: {sorted(relevant_ids)[:10]}")

    # Get document vectors
    print("\n2. Extracting Document Vectors:")
    relevant_vectors = []
    for doc_id in relevant_ids:
        doc_vec = vsm.get_document_vector(doc_id)
        if doc_vec:
            relevant_vectors.append(doc_vec)

    print(f"   Extracted {len(relevant_vectors)} vectors")

    # Build query vector
    print("\n3. Building Query Vector:")
    query_tokens = vsm.inverted_index.tokenizer(args.query)
    original_terms = set(query_tokens)

    from collections import defaultdict
    query_tf = defaultdict(int)
    for token in query_tokens:
        query_tf[token] += 1

    query_vector = vsm.term_weighting.vectorize(
        dict(query_tf),
        tf_scheme=vsm.query_tf_scheme,
        idf_scheme=vsm.query_idf_scheme,
        normalize=vsm.query_norm_scheme
    )

    print(f"   Original terms: {list(original_terms)}")

    # Optionally get non-relevant docs
    nonrelevant_vectors = None
    if args.nonrelevant:
        print("\n4. Loading Non-Relevant Documents:")
        nonrelevant_ids = load_relevant_docs(args.nonrelevant)
        print(f"   Loaded {len(nonrelevant_ids)} non-relevant IDs")

        nonrelevant_vectors = []
        for doc_id in nonrelevant_ids:
            doc_vec = vsm.get_document_vector(doc_id)
            if doc_vec:
                nonrelevant_vectors.append(doc_vec)

        print(f"   Extracted {len(nonrelevant_vectors)} vectors")

    # Expand query
    step_num = 5 if nonrelevant_vectors else 4
    print(f"\n{step_num}. Query Expansion (Rocchio):")
    expanded = expander.expand_query(
        query_vector,
        relevant_vectors,
        nonrelevant_vectors,
        original_terms
    )

    print(f"   Relevant docs: {expanded.num_relevant}")
    print(f"   Non-relevant docs: {expanded.num_nonrelevant}")
    print(f"   Expanded terms: {len(expanded.expanded_terms)}")

    # Show top expansion terms
    step_num += 1
    print(f"\n{step_num}. Top Expansion Terms:")
    top_terms = expander.get_top_expansion_terms(expanded, k=10)

    for i, (term, weight) in enumerate(top_terms, 1):
        print(f"   {i}. {term}: {weight:.4f}")

    # Retrieve with expanded query
    if not args.no_rerank:
        step_num += 1
        print(f"\n{step_num}. Retrieval with Expanded Query:")

        expanded_query_str = " ".join(expanded.all_terms)
        result = vsm.search(expanded_query_str, topk=args.topk)

        print(f"   Retrieved {result.num_results} documents")

        if args.verbose:
            for i, doc_id in enumerate(result.doc_ids[:10], 1):
                score = result.scores[doc_id]
                print(f"   {i}. Doc {doc_id}: {score:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Query Expansion using Rocchio Algorithm',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Pseudo-relevance feedback
  python scripts/expand_query.py --query "information retrieval" \
      --mode pseudo --index vsm_index.json --topk 10

  # Explicit relevance feedback
  python scripts/expand_query.py --query "vector space model" \
      --mode explicit --index vsm_index.json --relevant rel_docs.txt

  # Custom parameters
  python scripts/expand_query.py --query "search" --mode pseudo \
      --index vsm_index.json --alpha 1.0 --beta 0.8 --gamma 0.2

  # With non-relevant docs
  python scripts/expand_query.py --query "IR" --mode explicit \
      --index vsm_index.json --relevant rel.txt --nonrelevant nonrel.txt
        """
    )

    parser.add_argument('--query', type=str, required=True,
                       help='Query string to expand')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['pseudo', 'explicit'],
                       help='Expansion mode: pseudo or explicit feedback')
    parser.add_argument('--index', type=str, required=True,
                       help='Path to VSM index file')

    # Pseudo-feedback options
    parser.add_argument('--topk', type=int, default=20,
                       help='Number of top documents to retrieve (default: 20)')
    parser.add_argument('--num-relevant', type=int, default=10,
                       help='Number of top docs to treat as relevant (default: 10)')
    parser.add_argument('--num-nonrelevant', type=int, default=0,
                       help='Number of docs after relevant to treat as non-relevant')

    # Explicit feedback options
    parser.add_argument('--relevant', type=str,
                       help='File with relevant document IDs (for explicit mode)')
    parser.add_argument('--nonrelevant', type=str,
                       help='File with non-relevant document IDs (optional)')

    # Rocchio parameters
    parser.add_argument('--alpha', type=float, default=1.0,
                       help='Weight for original query (default: 1.0)')
    parser.add_argument('--beta', type=float, default=0.75,
                       help='Weight for relevant docs (default: 0.75)')
    parser.add_argument('--gamma', type=float, default=0.15,
                       help='Weight for non-relevant docs (default: 0.15)')

    # Expansion parameters
    parser.add_argument('--max-terms', type=int, default=10,
                       help='Maximum new terms to add (default: 10)')
    parser.add_argument('--min-weight', type=float, default=0.1,
                       help='Minimum weight for expansion terms (default: 0.1)')

    # Output options
    parser.add_argument('--no-rerank', action='store_true',
                       help='Skip re-ranking with expanded query')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed output')

    args = parser.parse_args()

    # Validate arguments
    if args.mode == 'explicit' and not args.relevant:
        parser.error("--relevant is required for explicit feedback mode")

    try:
        # Load VSM index
        print(f"Loading VSM index from {args.index}...")
        inv_index = InvertedIndex()
        inv_index.load(args.index.replace('.json', '_inv.json'))

        vsm = VectorSpaceModel(inv_index)
        vsm.term_weighting.build_from_index(inv_index)
        vsm._compute_document_vectors()

        print(f"  Loaded: {vsm.inverted_index.doc_count} documents, "
              f"{len(vsm.inverted_index.vocabulary)} terms")

        # Initialize Rocchio expander
        expander = RocchioExpander(
            alpha=args.alpha,
            beta=args.beta,
            gamma=args.gamma,
            max_expansion_terms=args.max_terms,
            min_term_weight=args.min_weight
        )

        print()

        # Execute expansion
        if args.mode == 'pseudo':
            expand_pseudo_feedback(args, vsm, expander)
        else:  # explicit
            expand_explicit_feedback(args, vsm, expander)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
