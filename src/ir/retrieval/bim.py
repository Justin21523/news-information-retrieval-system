"""
Binary Independence Model (BIM)

This module implements the Binary Independence Model, a classical probabilistic
retrieval model based on the Probability Ranking Principle (PRP).

BIM assumes that:
1. Terms are independent given relevance (naive Bayes assumption)
2. Documents are represented as binary term vectors (term present or absent)
3. Documents should be ranked by P(R|D) / P(NR|D) - odds of relevance

The model derives the optimal ranking function under these assumptions,
leading to the Robertson-Sparck Jones (RSJ) weighting scheme.

Key Concepts:
    - Probability Ranking Principle: Rank by P(R|D,Q) (probability of relevance)
    - Binary Representation: xi ∈ {0, 1} for each term
    - Independence Assumption: Terms are independent given relevance
    - RSJ Weighting: Optimal term weights under BIM assumptions

Formulas:
    - Retrieval Status Value (RSV): RSV(D,Q) = Σ wi * xi
    - RSJ Weight: wi = log((pi * (1-qi)) / ((1-pi) * qi))
    - Where:
        - pi = P(xi=1|R) = probability term i appears in relevant docs
        - qi = P(xi=1|NR) = probability term i appears in non-relevant docs

Without Relevance Feedback:
    - Assume pi = 0.5 (constant assumption)
    - Estimate qi from collection: qi ≈ dfi / N
    - Leads to IDF-like weighting: wi = log((N - dfi) / dfi)

With Relevance Feedback:
    - Estimate pi from relevant documents: pi = (ri + 0.5) / (R + 1)
    - Estimate qi from non-relevant docs: qi = (dfi - ri + 0.5) / (N - R + 1)
    - Where ri = # relevant docs containing term i, R = total relevant docs

Key Features:
    - Pure probabilistic ranking without heuristics
    - Support for relevance feedback
    - BM25 approximation mode (for comparison)
    - Theoretical foundation for modern ranking functions

Reference:
    Robertson & Sparck Jones (1976). "Relevance Weighting of Search Terms"
    Robertson (1977). "The Probability Ranking Principle in IR"
    Fuhr (1992). "Probabilistic Models in Information Retrieval"

Author: Information Retrieval System
License: Educational Use
"""

import math
import logging
from typing import List, Dict, Set, Tuple, Optional, Callable
from dataclasses import dataclass
from collections import defaultdict, Counter


@dataclass
class BIMResult:
    """
    Result of BIM search.

    Attributes:
        query: Original query string
        doc_ids: List of document IDs ranked by RSV
        scores: Corresponding RSV scores
        num_results: Total number of results
        parameters: BIM parameters used
    """
    query: str
    doc_ids: List[int]
    scores: List[float]
    num_results: int
    parameters: Dict[str, any]


class BinaryIndependenceModel:
    """
    Binary Independence Model for probabilistic retrieval.

    Implements the classic BIM with Robertson-Sparck Jones weighting,
    supporting both with and without relevance feedback.

    Attributes:
        tokenizer: Tokenization function
        inverted_index: Inverted index (term -> set of doc_ids)
        doc_count: Total number of documents
        term_weights: Term weights (wi) for ranking
        use_idf: Use IDF approximation when no relevance feedback

    Complexity:
        - Indexing: O(N * M) where N=docs, M=avg terms per doc
        - Query (no feedback): O(T * D) where T=query terms, D=docs per term
        - Query (with feedback): O(T * D + R) where R=relevant docs
    """

    def __init__(self,
                 tokenizer: Optional[Callable[[str], List[str]]] = None,
                 use_idf: bool = True):
        """
        Initialize Binary Independence Model.

        Args:
            tokenizer: Custom tokenization function
            use_idf: Use IDF approximation for initial retrieval (default: True)

        Complexity:
            Time: O(1)
        """
        self.logger = logging.getLogger(__name__)

        # Tokenizer
        self.tokenizer = tokenizer or self._default_tokenizer

        # Index structures
        self.inverted_index: Dict[str, Set[int]] = defaultdict(set)
        self.doc_count: int = 0
        self.doc_terms: Dict[int, Set[str]] = {}  # doc_id -> set of terms

        # Term statistics
        self.doc_freq: Dict[str, int] = Counter()  # term -> document frequency
        self.vocab: Set[str] = set()

        # BIM weights
        self.term_weights: Dict[str, float] = {}  # term -> wi
        self.use_idf = use_idf

        # Relevance feedback data
        self.relevant_docs: Set[int] = set()
        self.non_relevant_docs: Set[int] = set()

        self.logger.info(f"BinaryIndependenceModel initialized (use_idf={use_idf})")

    def _default_tokenizer(self, text: str) -> List[str]:
        """Default tokenizer (simple split)."""
        import re
        return re.findall(r'\b\w+\b', text.lower())

    def build_index(self, documents: List[str]) -> None:
        """
        Build inverted index from documents.

        Args:
            documents: List of document texts

        Complexity:
            Time: O(N * M) where N=docs, M=avg terms per doc
            Space: O(V * D_avg) where V=vocab, D_avg=avg docs per term
        """
        self.logger.info(f"Building BIM index for {len(documents)} documents...")

        self.doc_count = len(documents)

        # Build inverted index
        for doc_id, doc_text in enumerate(documents):
            tokens = self.tokenizer(doc_text)
            unique_terms = set(tokens)

            # Store document terms
            self.doc_terms[doc_id] = unique_terms

            # Update inverted index
            for term in unique_terms:
                self.inverted_index[term].add(doc_id)
                self.vocab.add(term)

        # Calculate document frequencies
        for term, doc_ids in self.inverted_index.items():
            self.doc_freq[term] = len(doc_ids)

        # Compute initial weights (IDF-based)
        if self.use_idf:
            self._compute_idf_weights()

        vocab_size = len(self.vocab)
        self.logger.info(
            f"BIM index built: {self.doc_count} docs, {vocab_size} terms"
        )

    def _compute_idf_weights(self) -> None:
        """
        Compute IDF-based term weights (without relevance feedback).

        When no relevance information is available, we use:
        - pi = 0.5 (constant assumption)
        - qi ≈ dfi / N

        This leads to: wi ≈ log((N - dfi + 0.5) / (dfi + 0.5))

        Complexity:
            Time: O(V) where V = vocabulary size
        """
        for term in self.vocab:
            df = self.doc_freq[term]

            # IDF-based weight (similar to BM25 IDF)
            weight = math.log((self.doc_count - df + 0.5) / (df + 0.5))
            self.term_weights[term] = max(weight, 0.0)  # Ensure non-negative

    def _compute_rsj_weights(self) -> None:
        """
        Compute Robertson-Sparck Jones weights with relevance feedback.

        RSJ formula:
        wi = log((pi * (1-qi)) / ((1-pi) * qi))

        Where:
        - pi = (ri + 0.5) / (R + 1)
        - qi = (dfi - ri + 0.5) / (N - R + 1)
        - ri = # relevant docs containing term i
        - R = total relevant docs
        - N = total docs

        Complexity:
            Time: O(V) where V = vocabulary size
        """
        R = len(self.relevant_docs)
        N = self.doc_count

        if R == 0:
            # No relevance feedback - fall back to IDF
            self._compute_idf_weights()
            return

        for term in self.vocab:
            df = self.doc_freq[term]

            # Count relevant docs containing this term
            ri = sum(1 for doc_id in self.relevant_docs
                    if term in self.doc_terms.get(doc_id, set()))

            # RSJ formula with smoothing
            pi = (ri + 0.5) / (R + 1)
            qi = (df - ri + 0.5) / (N - R + 1)

            # Compute weight
            if qi > 0 and pi < 1:
                weight = math.log((pi * (1 - qi)) / ((1 - pi) * qi))
                self.term_weights[term] = weight
            else:
                # Edge case - use IDF
                self.term_weights[term] = math.log((N - df + 0.5) / (df + 0.5))

    def set_relevance_feedback(self, relevant_docs: List[int],
                               non_relevant_docs: Optional[List[int]] = None) -> None:
        """
        Update model with relevance feedback.

        Args:
            relevant_docs: List of relevant document IDs
            non_relevant_docs: Optional list of non-relevant document IDs

        Complexity:
            Time: O(V + R) where V=vocab, R=relevant docs
        """
        self.relevant_docs = set(relevant_docs)

        if non_relevant_docs:
            self.non_relevant_docs = set(non_relevant_docs)
        else:
            # Assume all other docs are non-relevant
            all_docs = set(range(self.doc_count))
            self.non_relevant_docs = all_docs - self.relevant_docs

        # Recompute weights with feedback
        self._compute_rsj_weights()

        self.logger.info(
            f"Relevance feedback applied: {len(self.relevant_docs)} relevant, "
            f"{len(self.non_relevant_docs)} non-relevant"
        )

    def compute_rsv(self, query_terms: List[str], doc_id: int) -> float:
        """
        Calculate Retrieval Status Value (RSV) for a document.

        RSV(D,Q) = Σ wi * xi

        where xi = 1 if term i is in document D, 0 otherwise.

        Args:
            query_terms: List of query terms
            doc_id: Document ID

        Returns:
            RSV score

        Complexity:
            Time: O(T) where T = number of query terms

        Examples:
            >>> bim.compute_rsv(["information", "retrieval"], 5)
            12.45
        """
        if doc_id not in self.doc_terms:
            return 0.0

        doc_terms = self.doc_terms[doc_id]
        rsv = 0.0

        for term in query_terms:
            if term in doc_terms:
                # Binary presence: xi = 1
                weight = self.term_weights.get(term, 0.0)
                rsv += weight

        return rsv

    def search(self, query: str, topk: int = 10,
               use_relevance_feedback: bool = False) -> BIMResult:
        """
        Search documents using Binary Independence Model.

        Args:
            query: Query string
            topk: Number of top results to return
            use_relevance_feedback: Whether relevance feedback has been applied

        Returns:
            BIMResult with ranked documents

        Complexity:
            Time: O(T * D + N log K) where T=query terms, D=docs per term,
                  N=candidate docs, K=topk
            Space: O(N) for candidate documents

        Examples:
            >>> bim = BinaryIndependenceModel()
            >>> bim.build_index(documents)
            >>> result = bim.search("information retrieval", topk=10)
            >>> result.doc_ids
            [5, 12, 3, 18, ...]
        """
        # Tokenize query
        query_terms = self.tokenizer(query)

        if not query_terms:
            return BIMResult(
                query=query,
                doc_ids=[],
                scores=[],
                num_results=0,
                parameters={
                    'use_idf': self.use_idf,
                    'has_feedback': len(self.relevant_docs) > 0
                }
            )

        # Get candidate documents (union of all docs containing query terms)
        candidate_docs = set()
        for term in query_terms:
            if term in self.inverted_index:
                candidate_docs.update(self.inverted_index[term])

        # Score all candidate documents
        doc_scores: List[Tuple[int, float]] = []
        for doc_id in candidate_docs:
            rsv = self.compute_rsv(query_terms, doc_id)
            if rsv > 0:
                doc_scores.append((doc_id, rsv))

        # Sort by RSV (descending)
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        # Get top-k
        top_docs = doc_scores[:topk]

        doc_ids = [doc_id for doc_id, score in top_docs]
        scores = [score for doc_id, score in top_docs]

        return BIMResult(
            query=query,
            doc_ids=doc_ids,
            scores=scores,
            num_results=len(doc_ids),
            parameters={
                'use_idf': self.use_idf,
                'has_feedback': len(self.relevant_docs) > 0,
                'num_relevant': len(self.relevant_docs)
            }
        )

    def explain_score(self, query: str, doc_id: int) -> Dict:
        """
        Explain BIM score breakdown for a document.

        Args:
            query: Query string
            doc_id: Document ID

        Returns:
            Dictionary with score explanation

        Examples:
            >>> explain = bim.explain_score("information retrieval", 5)
            >>> print(explain)
            {
                'doc_id': 5,
                'total_rsv': 12.45,
                'has_feedback': False,
                'term_contributions': {
                    'information': {'present': True, 'weight': 6.2, 'contribution': 6.2},
                    'retrieval': {'present': True, 'weight': 6.25, 'contribution': 6.25}
                }
            }
        """
        query_terms = self.tokenizer(query)

        if doc_id not in self.doc_terms:
            return {'error': 'Document not found'}

        doc_terms = self.doc_terms[doc_id]
        term_contributions = {}
        total_rsv = 0.0

        for term in query_terms:
            present = term in doc_terms
            weight = self.term_weights.get(term, 0.0)
            contribution = weight if present else 0.0

            total_rsv += contribution

            term_contributions[term] = {
                'present': present,
                'weight': round(weight, 4),
                'contribution': round(contribution, 4),
                'df': self.doc_freq.get(term, 0)
            }

            # Add RSJ-specific info if feedback exists
            if len(self.relevant_docs) > 0:
                ri = sum(1 for doc in self.relevant_docs
                        if term in self.doc_terms.get(doc, set()))
                term_contributions[term]['ri'] = ri
                term_contributions[term]['R'] = len(self.relevant_docs)

        return {
            'doc_id': doc_id,
            'total_rsv': round(total_rsv, 4),
            'has_feedback': len(self.relevant_docs) > 0,
            'num_relevant_docs': len(self.relevant_docs),
            'parameters': {
                'use_idf': self.use_idf
            },
            'term_contributions': term_contributions
        }

    def get_stats(self) -> Dict:
        """Get BIM statistics."""
        return {
            'doc_count': self.doc_count,
            'vocabulary_size': len(self.vocab),
            'use_idf': self.use_idf,
            'has_relevance_feedback': len(self.relevant_docs) > 0,
            'num_relevant_docs': len(self.relevant_docs),
            'num_non_relevant_docs': len(self.non_relevant_docs)
        }


def demo():
    """Demonstration of Binary Independence Model."""
    print("=" * 60)
    print("Binary Independence Model (BIM) Demo")
    print("=" * 60)

    # Sample documents
    documents = [
        "information retrieval is the process of obtaining information",
        "retrieval models include boolean and vector space models",
        "BM25 is a probabilistic retrieval function",
        "the BM25 ranking function is widely used in search engines",
        "information extraction is related to information retrieval",
        "search engines use various ranking algorithms including BM25",
        "probabilistic models are based on probability theory",
        "the binary independence model assumes term independence"
    ]

    # Build BIM index
    print("\n1. Initial Retrieval (No Relevance Feedback)")
    print("-" * 60)

    bim = BinaryIndependenceModel(use_idf=True)
    bim.build_index(documents)

    stats = bim.get_stats()
    print(f"Dataset: {stats['doc_count']} documents")
    print(f"Vocabulary: {stats['vocabulary_size']} terms")

    # Test query
    query = "information retrieval"
    print(f"\nQuery: '{query}'")

    result = bim.search(query, topk=5)
    print(f"Results: {result.num_results}")
    for i, (doc_id, score) in enumerate(zip(result.doc_ids, result.scores), 1):
        print(f"  {i}. Doc {doc_id}: RSV={score:.4f}")
        print(f"     {documents[doc_id][:60]}...")

    # Explain score
    print(f"\n{'-' * 60}")
    print("Score Explanation (No Feedback)")
    print("-" * 60)

    doc_id = result.doc_ids[0]
    explanation = bim.explain_score(query, doc_id)

    print(f"\nDocument {doc_id}: {documents[doc_id]}")
    print(f"Total RSV: {explanation['total_rsv']}")
    print("\nTerm Contributions:")
    for term, contrib in explanation['term_contributions'].items():
        print(f"  '{term}':")
        print(f"    Present: {contrib['present']}")
        print(f"    Weight: {contrib['weight']:.4f}")
        print(f"    DF: {contrib['df']}")
        print(f"    Contribution: {contrib['contribution']:.4f}")

    # Simulate relevance feedback
    print(f"\n{'=' * 60}")
    print("2. Retrieval with Relevance Feedback")
    print("=" * 60)

    # Assume docs 0, 4 are relevant
    relevant_docs = [0, 4]
    bim.set_relevance_feedback(relevant_docs)

    print(f"\nRelevance Feedback: {len(relevant_docs)} relevant docs")
    print(f"Query: '{query}'")

    result_fb = bim.search(query, topk=5, use_relevance_feedback=True)
    print(f"Results: {result_fb.num_results}")
    for i, (doc_id, score) in enumerate(zip(result_fb.doc_ids, result_fb.scores), 1):
        print(f"  {i}. Doc {doc_id}: RSV={score:.4f}")
        print(f"     {documents[doc_id][:60]}...")

    # Explain score with feedback
    print(f"\n{'-' * 60}")
    print("Score Explanation (With Feedback)")
    print("-" * 60)

    doc_id_fb = result_fb.doc_ids[0]
    explanation_fb = bim.explain_score(query, doc_id_fb)

    print(f"\nDocument {doc_id_fb}: {documents[doc_id_fb]}")
    print(f"Total RSV: {explanation_fb['total_rsv']}")
    print(f"Relevant Docs: {explanation_fb['num_relevant_docs']}")
    print("\nTerm Contributions:")
    for term, contrib in explanation_fb['term_contributions'].items():
        print(f"  '{term}':")
        print(f"    Present: {contrib['present']}")
        print(f"    Weight: {contrib['weight']:.4f}")
        print(f"    DF: {contrib['df']}")
        if 'ri' in contrib:
            print(f"    ri (relevant docs with term): {contrib['ri']}")
            print(f"    R (total relevant docs): {contrib['R']}")
        print(f"    Contribution: {contrib['contribution']:.4f}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    demo()
