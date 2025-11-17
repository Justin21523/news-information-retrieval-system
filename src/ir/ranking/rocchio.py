"""
Rocchio Algorithm for Query Expansion

This module implements the Rocchio algorithm for relevance feedback
and query expansion. It adjusts query vectors based on relevant and
non-relevant documents to improve retrieval effectiveness.

Key Features:
    - Classic Rocchio algorithm
    - Pseudo-relevance feedback (blind feedback)
    - Explicit relevance feedback
    - Term selection and filtering
    - Query vector modification

Formula:
    Q_new = α × Q_orig + β × (1/|D_r|) × Σ D_r - γ × (1/|D_nr|) × Σ D_nr

    where:
    - Q_orig: original query vector
    - D_r: set of relevant documents
    - D_nr: set of non-relevant documents
    - α, β, γ: tuning parameters (typically α=1.0, β=0.75, γ=0.15)

Reference: "Introduction to Information Retrieval" (Manning et al.)
           Chapter 9: Relevance Feedback and Query Expansion

Author: Information Retrieval System
License: Educational Use
"""

import logging
import heapq
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

import sys
from pathlib import Path

_parent_dir = Path(__file__).parent.parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))


@dataclass
class ExpandedQuery:
    """
    Result of query expansion.

    Attributes:
        original_terms: Original query terms
        expanded_terms: New terms added by expansion
        all_terms: Combined term list
        term_weights: Weight for each term
        num_relevant: Number of relevant docs used
        num_nonrelevant: Number of non-relevant docs used
    """
    original_terms: List[str]
    expanded_terms: List[str]
    all_terms: List[str]
    term_weights: Dict[str, float]
    num_relevant: int = 0
    num_nonrelevant: int = 0


class RocchioExpander:
    """
    Rocchio Algorithm for Query Expansion.

    Modifies query vectors based on relevance feedback to improve
    retrieval performance. Supports both explicit feedback (user-provided)
    and pseudo-relevance feedback (automatic from top-k results).

    Parameters:
        alpha: Weight for original query (default 1.0)
        beta: Weight for relevant documents (default 0.75)
        gamma: Weight for non-relevant documents (default 0.15)
        max_expansion_terms: Maximum new terms to add (default 10)
        min_term_weight: Minimum weight threshold for new terms (default 0.1)

    Complexity:
        - Expansion: O(|D_r| × V + |D_nr| × V) where V is vocabulary size
        - With top-k selection: O(V × log(k))
    """

    def __init__(self,
                 alpha: float = 1.0,
                 beta: float = 0.75,
                 gamma: float = 0.15,
                 max_expansion_terms: int = 10,
                 min_term_weight: float = 0.1):
        """
        Initialize Rocchio expander.

        Args:
            alpha: Weight for original query (emphasize user intent)
            beta: Weight for relevant docs (learn from positive examples)
            gamma: Weight for non-relevant docs (avoid negative examples)
            max_expansion_terms: Maximum new terms to add
            min_term_weight: Minimum weight for expansion terms
        """
        self.logger = logging.getLogger(__name__)

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.max_expansion_terms = max_expansion_terms
        self.min_term_weight = min_term_weight

        self.logger.info(
            f"RocchioExpander initialized: α={alpha}, β={beta}, γ={gamma}"
        )

    def expand_query(self,
                     query_vector: Dict[str, float],
                     relevant_vectors: List[Dict[str, float]],
                     nonrelevant_vectors: Optional[List[Dict[str, float]]] = None,
                     original_terms: Optional[Set[str]] = None) -> ExpandedQuery:
        """
        Expand query using Rocchio algorithm.

        Q_new = α × Q + β × (1/|D_r|) × Σ D_r - γ × (1/|D_nr|) × Σ D_nr

        Args:
            query_vector: Original query vector {term: weight}
            relevant_vectors: List of relevant document vectors
            nonrelevant_vectors: Optional list of non-relevant document vectors
            original_terms: Optional set of original query terms (for tracking)

        Returns:
            ExpandedQuery with new terms and weights

        Complexity:
            Time: O(V × (|D_r| + |D_nr|)) where V is vocabulary size
            Space: O(V) for expanded vector

        Examples:
            >>> expander = RocchioExpander(alpha=1.0, beta=0.75, gamma=0.15)
            >>> query_vec = {"information": 0.8, "retrieval": 0.6}
            >>> rel_vecs = [
            ...     {"information": 0.5, "retrieval": 0.7, "system": 0.3},
            ...     {"information": 0.6, "database": 0.4}
            ... ]
            >>> expanded = expander.expand_query(query_vec, rel_vecs)
            >>> "system" in expanded.expanded_terms
            True
        """
        if not relevant_vectors:
            self.logger.warning("No relevant documents for expansion")
            return ExpandedQuery(
                original_terms=list(query_vector.keys()),
                expanded_terms=[],
                all_terms=list(query_vector.keys()),
                term_weights=query_vector.copy(),
                num_relevant=0,
                num_nonrelevant=0
            )

        # Track original terms
        if original_terms is None:
            original_terms = set(query_vector.keys())

        # Initialize new query vector
        new_query = defaultdict(float)

        # Step 1: α × Q (original query)
        for term, weight in query_vector.items():
            new_query[term] += self.alpha * weight

        # Step 2: β × (1/|D_r|) × Σ D_r (relevant documents)
        num_relevant = len(relevant_vectors)
        for doc_vec in relevant_vectors:
            for term, weight in doc_vec.items():
                new_query[term] += (self.beta / num_relevant) * weight

        # Step 3: γ × (1/|D_nr|) × Σ D_nr (non-relevant documents)
        num_nonrelevant = 0
        if nonrelevant_vectors:
            num_nonrelevant = len(nonrelevant_vectors)
            for doc_vec in nonrelevant_vectors:
                for term, weight in doc_vec.items():
                    new_query[term] -= (self.gamma / num_nonrelevant) * weight

        # Filter negative weights
        new_query = {term: max(0.0, weight)
                    for term, weight in new_query.items()}

        # Select expansion terms (terms not in original query)
        expansion_candidates = []
        for term, weight in new_query.items():
            if term not in original_terms and weight >= self.min_term_weight:
                expansion_candidates.append((term, weight))

        # Sort by weight and select top-k
        expansion_candidates.sort(key=lambda x: x[1], reverse=True)
        expansion_terms = [term for term, _ in
                          expansion_candidates[:self.max_expansion_terms]]

        # Build final term list and weights
        all_terms = list(original_terms) + expansion_terms
        term_weights = {term: new_query[term] for term in all_terms}

        result = ExpandedQuery(
            original_terms=list(original_terms),
            expanded_terms=expansion_terms,
            all_terms=all_terms,
            term_weights=term_weights,
            num_relevant=num_relevant,
            num_nonrelevant=num_nonrelevant
        )

        self.logger.debug(
            f"Expanded query: {len(original_terms)} → {len(all_terms)} terms "
            f"(+{len(expansion_terms)} new)"
        )

        return result

    def expand_with_pseudo_feedback(self,
                                    query_vector: Dict[str, float],
                                    top_documents: List[Dict[str, float]],
                                    num_relevant: int = 10,
                                    num_nonrelevant: int = 0,
                                    original_terms: Optional[Set[str]] = None) -> ExpandedQuery:
        """
        Expand query using pseudo-relevance feedback (blind feedback).

        Assumes top-k retrieved documents are relevant without user feedback.
        Optionally can treat lower-ranked documents as non-relevant.

        Args:
            query_vector: Original query vector
            top_documents: List of top retrieved document vectors (ranked)
            num_relevant: Number of top docs to treat as relevant (default 10)
            num_nonrelevant: Number of docs after relevant to treat as non-relevant
            original_terms: Original query terms

        Returns:
            ExpandedQuery with new terms

        Complexity:
            Time: O(k × V) where k is num_relevant + num_nonrelevant

        Examples:
            >>> expander = RocchioExpander()
            >>> query_vec = {"search": 0.8}
            >>> top_docs = [
            ...     {"search": 0.7, "engine": 0.5, "web": 0.3},
            ...     {"search": 0.6, "query": 0.4},
            ... ]
            >>> expanded = expander.expand_with_pseudo_feedback(
            ...     query_vec, top_docs, num_relevant=2
            ... )
            >>> len(expanded.expanded_terms) > 0
            True
        """
        if not top_documents:
            self.logger.warning("No documents for pseudo-feedback")
            return ExpandedQuery(
                original_terms=list(query_vector.keys()),
                expanded_terms=[],
                all_terms=list(query_vector.keys()),
                term_weights=query_vector.copy()
            )

        # Split into relevant and non-relevant sets
        relevant_docs = top_documents[:num_relevant]
        nonrelevant_docs = None

        if num_nonrelevant > 0:
            start_idx = num_relevant
            end_idx = num_relevant + num_nonrelevant
            nonrelevant_docs = top_documents[start_idx:end_idx]

        # Apply standard Rocchio
        return self.expand_query(
            query_vector,
            relevant_docs,
            nonrelevant_docs,
            original_terms
        )

    def reweight_query(self,
                      original_query: Dict[str, float],
                      expanded_query: ExpandedQuery,
                      normalize: bool = True) -> Dict[str, float]:
        """
        Create reweighted query vector combining original and expanded terms.

        Args:
            original_query: Original query vector
            expanded_query: Result from expansion
            normalize: Whether to normalize final weights

        Returns:
            Reweighted query vector

        Complexity:
            Time: O(|query|)
            Space: O(|query|)
        """
        # Use weights from expanded query
        reweighted = expanded_query.term_weights.copy()

        # Normalize if requested
        if normalize:
            total = sum(reweighted.values())
            if total > 0:
                reweighted = {term: weight / total
                            for term, weight in reweighted.items()}

        return reweighted

    def get_top_expansion_terms(self,
                               expanded_query: ExpandedQuery,
                               k: int = 5) -> List[Tuple[str, float]]:
        """
        Get top-k expansion terms by weight.

        Args:
            expanded_query: Result from expansion
            k: Number of top terms to return

        Returns:
            List of (term, weight) tuples sorted by weight descending

        Complexity:
            Time: O(n × log(k)) using heap
        """
        expansion_terms = [
            (term, expanded_query.term_weights[term])
            for term in expanded_query.expanded_terms
        ]

        # Use heap for top-k
        if len(expansion_terms) <= k:
            expansion_terms.sort(key=lambda x: x[1], reverse=True)
            return expansion_terms
        else:
            return heapq.nlargest(k, expansion_terms, key=lambda x: x[1])

    def set_parameters(self, alpha: float, beta: float, gamma: float) -> None:
        """
        Update Rocchio parameters.

        Args:
            alpha: Weight for original query
            beta: Weight for relevant documents
            gamma: Weight for non-relevant documents
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.logger.info(
            f"Parameters updated: α={alpha}, β={beta}, γ={gamma}"
        )


def demo():
    """Demonstration of Rocchio query expansion."""
    print("=" * 60)
    print("Rocchio Query Expansion Demo")
    print("=" * 60)

    # Example 1: Basic Rocchio expansion
    print("\n1. Basic Rocchio Expansion:")
    print("   Original query: 'information retrieval'")

    query_vec = {
        "information": 0.8,
        "retrieval": 0.6
    }

    relevant_docs = [
        {"information": 0.5, "retrieval": 0.7, "system": 0.3, "database": 0.2},
        {"information": 0.6, "search": 0.4, "engine": 0.3},
        {"retrieval": 0.5, "document": 0.4, "index": 0.3}
    ]

    expander = RocchioExpander(alpha=1.0, beta=0.75, gamma=0.15)
    expanded = expander.expand_query(query_vec, relevant_docs)

    print(f"   Original terms: {expanded.original_terms}")
    print(f"   Expanded terms: {expanded.expanded_terms[:5]}")
    print(f"   Total terms: {len(expanded.all_terms)}")

    # Example 2: Pseudo-relevance feedback
    print("\n2. Pseudo-Relevance Feedback:")
    print("   Using top-10 retrieved documents")

    top_docs = relevant_docs  # Simulate top results
    expanded_prf = expander.expand_with_pseudo_feedback(
        query_vec, top_docs, num_relevant=3
    )

    print(f"   Relevant docs used: {expanded_prf.num_relevant}")
    print(f"   New terms added: {len(expanded_prf.expanded_terms)}")

    # Example 3: Top expansion terms
    print("\n3. Top Expansion Terms:")
    top_terms = expander.get_top_expansion_terms(expanded, k=5)

    for i, (term, weight) in enumerate(top_terms, 1):
        print(f"   {i}. {term}: {weight:.4f}")

    # Example 4: Parameter tuning
    print("\n4. Parameter Tuning:")
    configs = [
        (1.0, 0.75, 0.15, "Standard"),
        (1.0, 1.0, 0.0, "Positive feedback only"),
        (1.0, 0.5, 0.5, "Balanced pos/neg"),
    ]

    for alpha, beta, gamma, desc in configs:
        expander.set_parameters(alpha, beta, gamma)
        result = expander.expand_query(query_vec, relevant_docs)
        print(f"   {desc} (α={alpha}, β={beta}, γ={gamma}):")
        print(f"      Expansion terms: {len(result.expanded_terms)}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    demo()
