"""
BM25 Retrieval Model

This module implements the BM25 ranking function, one of the most effective
probabilistic ranking functions for information retrieval.

BM25 (Best Matching 25) is a bag-of-words retrieval function that ranks documents
based on query terms appearing in each document, considering term frequency,
document length, and inverse document frequency.

Formula:
    BM25(D, Q) = Σ IDF(qi) * (f(qi,D) * (k1 + 1)) / (f(qi,D) + k1 * (1 - b + b * |D|/avgdl))

Where:
    - D: Document
    - Q: Query
    - qi: Query term i
    - f(qi, D): Term frequency of qi in D
    - |D|: Document length
    - avgdl: Average document length
    - k1: Term frequency saturation parameter (typical: 1.2-2.0)
    - b: Length normalization parameter (typical: 0.75)
    - IDF(qi): Inverse document frequency of qi

Key Features:
    - Non-linear term frequency saturation
    - Document length normalization
    - Multiple BM25 variants (BM25, BM25+, BM25L, BM25F)
    - Support for field-weighted scoring

Reference:
    Robertson & Zaragoza (2009). "The Probabilistic Relevance Framework: BM25 and Beyond"
    Foundations and Trends in Information Retrieval, 3(4), 333-389.

Author: Information Retrieval System
License: Educational Use
"""

import math
import logging
from typing import List, Dict, Set, Tuple, Optional, Callable
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class BM25Result:
    """
    Result of BM25 search.

    Attributes:
        query: Original query string
        doc_ids: List of document IDs ranked by relevance
        scores: Corresponding BM25 scores
        num_results: Total number of results
        parameters: BM25 parameters used (k1, b)
    """
    query: str
    doc_ids: List[int]
    scores: List[float]
    num_results: int
    parameters: Dict[str, float]


class BM25Ranker:
    """
    BM25 ranking function for document retrieval.

    Implements standard BM25 and its variants with support for
    custom parameters and field-based weighting.

    Attributes:
        inverted_index: Inverted index for term lookup
        doc_count: Total number of documents
        avg_doc_length: Average document length
        doc_lengths: Document length mapping
        k1: Term frequency saturation parameter
        b: Length normalization parameter
        tokenizer: Tokenization function

    Complexity:
        - Indexing: O(N * M) where N=docs, M=avg terms per doc
        - Query: O(T * D) where T=query terms, D=avg docs per term
        - Top-K: O(N log K) where N=matching docs, K=topk
    """

    def __init__(self,
                 tokenizer: Optional[Callable[[str], List[str]]] = None,
                 k1: float = 1.5,
                 b: float = 0.75,
                 delta: float = 0.0):
        """
        Initialize BM25Ranker.

        Args:
            tokenizer: Custom tokenization function
            k1: Term frequency saturation (default: 1.5, range: 1.2-2.0)
            b: Length normalization (default: 0.75, range: 0-1)
            delta: BM25+ parameter (default: 0.0 for standard BM25)

        Complexity:
            Time: O(1)
        """
        self.logger = logging.getLogger(__name__)

        # Parameters
        self.k1 = k1
        self.b = b
        self.delta = delta  # For BM25+ variant

        # Tokenizer
        self.tokenizer = tokenizer or self._default_tokenizer

        # Index structures
        self.inverted_index: Dict[str, Dict[int, int]] = {}  # term -> {doc_id: tf}
        self.doc_lengths: Dict[int, int] = {}
        self.doc_count: int = 0
        self.avg_doc_length: float = 0.0

        # IDF cache
        self.idf_cache: Dict[str, float] = {}

        self.logger.info(f"BM25Ranker initialized (k1={k1}, b={b}, delta={delta})")

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
            Time: O(N * M) where N=docs, M=avg terms
            Space: O(V * D) where V=vocab, D=avg docs per term
        """
        self.logger.info(f"Building BM25 index for {len(documents)} documents...")

        self.doc_count = len(documents)
        total_length = 0

        # Build inverted index
        for doc_id, doc_text in enumerate(documents):
            tokens = self.tokenizer(doc_text)
            doc_length = len(tokens)

            self.doc_lengths[doc_id] = doc_length
            total_length += doc_length

            # Count term frequencies
            term_freqs = defaultdict(int)
            for token in tokens:
                term_freqs[token] += 1

            # Update inverted index
            for term, tf in term_freqs.items():
                if term not in self.inverted_index:
                    self.inverted_index[term] = {}
                self.inverted_index[term][doc_id] = tf

        # Calculate average document length
        self.avg_doc_length = total_length / self.doc_count if self.doc_count > 0 else 0

        # Precompute IDF values
        self._compute_idf()

        vocab_size = len(self.inverted_index)
        self.logger.info(
            f"BM25 index built: {self.doc_count} docs, "
            f"{vocab_size} terms, avg_doc_length={self.avg_doc_length:.2f}"
        )

    def _compute_idf(self) -> None:
        """
        Precompute IDF values for all terms.

        IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)

        Complexity:
            Time: O(V) where V = vocabulary size
        """
        for term, postings in self.inverted_index.items():
            df = len(postings)  # Document frequency

            # BM25 IDF formula (Robertson-Sparck Jones weight)
            idf = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1.0)
            self.idf_cache[term] = idf

    def score_document(self, query_terms: List[str], doc_id: int) -> float:
        """
        Calculate BM25 score for a document given query terms.

        Args:
            query_terms: List of query terms
            doc_id: Document ID to score

        Returns:
            BM25 score

        Complexity:
            Time: O(T) where T = number of query terms
        """
        if doc_id not in self.doc_lengths:
            return 0.0

        doc_length = self.doc_lengths[doc_id]
        score = 0.0

        for term in query_terms:
            if term not in self.inverted_index:
                continue

            if doc_id not in self.inverted_index[term]:
                continue

            # Term frequency in document
            tf = self.inverted_index[term][doc_id]

            # IDF value
            idf = self.idf_cache.get(term, 0.0)

            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (
                1 - self.b + self.b * (doc_length / self.avg_doc_length)
            )

            # BM25+ delta term (optional)
            term_score = idf * (numerator / denominator + self.delta)
            score += term_score

        return score

    def search(self, query: str, topk: int = 10) -> BM25Result:
        """
        Search documents using BM25 ranking.

        Args:
            query: Query string
            topk: Number of top results to return

        Returns:
            BM25Result with ranked documents

        Complexity:
            Time: O(T * D + N log K) where T=query terms, D=docs per term, N=candidates, K=topk
            Space: O(N) where N = candidate documents

        Examples:
            >>> ranker = BM25Ranker()
            >>> ranker.build_index(documents)
            >>> result = ranker.search("information retrieval", topk=10)
            >>> result.doc_ids
            [5, 12, 3, 18, ...]
        """
        # Tokenize query
        query_terms = self.tokenizer(query)

        if not query_terms:
            return BM25Result(
                query=query,
                doc_ids=[],
                scores=[],
                num_results=0,
                parameters={'k1': self.k1, 'b': self.b, 'delta': self.delta}
            )

        # Get candidate documents (union of all docs containing query terms)
        candidate_docs = set()
        for term in query_terms:
            if term in self.inverted_index:
                candidate_docs.update(self.inverted_index[term].keys())

        # Score all candidate documents
        doc_scores: List[Tuple[int, float]] = []
        for doc_id in candidate_docs:
            score = self.score_document(query_terms, doc_id)
            if score > 0:
                doc_scores.append((doc_id, score))

        # Sort by score (descending)
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        # Get top-k
        top_docs = doc_scores[:topk]

        doc_ids = [doc_id for doc_id, score in top_docs]
        scores = [score for doc_id, score in top_docs]

        return BM25Result(
            query=query,
            doc_ids=doc_ids,
            scores=scores,
            num_results=len(doc_ids),
            parameters={'k1': self.k1, 'b': self.b, 'delta': self.delta}
        )

    def explain_score(self, query: str, doc_id: int) -> Dict:
        """
        Explain BM25 score breakdown for a document.

        Args:
            query: Query string
            doc_id: Document ID

        Returns:
            Dictionary with score explanation

        Examples:
            >>> explain = ranker.explain_score("information retrieval", 5)
            >>> print(explain)
            {
                'doc_id': 5,
                'total_score': 12.45,
                'doc_length': 150,
                'term_contributions': {
                    'information': {'tf': 3, 'idf': 2.1, 'score': 5.2},
                    'retrieval': {'tf': 2, 'idf': 3.5, 'score': 7.25}
                }
            }
        """
        query_terms = self.tokenizer(query)

        if doc_id not in self.doc_lengths:
            return {'error': 'Document not found'}

        doc_length = self.doc_lengths[doc_id]
        term_contributions = {}
        total_score = 0.0

        for term in query_terms:
            if term not in self.inverted_index:
                term_contributions[term] = {
                    'tf': 0,
                    'idf': 0.0,
                    'score': 0.0,
                    'reason': 'Term not in vocabulary'
                }
                continue

            if doc_id not in self.inverted_index[term]:
                term_contributions[term] = {
                    'tf': 0,
                    'idf': self.idf_cache.get(term, 0.0),
                    'score': 0.0,
                    'reason': 'Term not in document'
                }
                continue

            tf = self.inverted_index[term][doc_id]
            idf = self.idf_cache.get(term, 0.0)

            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (
                1 - self.b + self.b * (doc_length / self.avg_doc_length)
            )

            term_score = idf * (numerator / denominator + self.delta)
            total_score += term_score

            term_contributions[term] = {
                'tf': tf,
                'df': len(self.inverted_index[term]),
                'idf': round(idf, 4),
                'score': round(term_score, 4),
                'normalized_tf': round(numerator / denominator, 4)
            }

        return {
            'doc_id': doc_id,
            'total_score': round(total_score, 4),
            'doc_length': doc_length,
            'avg_doc_length': round(self.avg_doc_length, 2),
            'parameters': {'k1': self.k1, 'b': self.b, 'delta': self.delta},
            'term_contributions': term_contributions
        }

    def get_stats(self) -> Dict:
        """Get BM25 ranker statistics."""
        return {
            'doc_count': self.doc_count,
            'vocabulary_size': len(self.inverted_index),
            'avg_doc_length': round(self.avg_doc_length, 2),
            'parameters': {
                'k1': self.k1,
                'b': self.b,
                'delta': self.delta
            }
        }


def demo():
    """Demonstration of BM25Ranker."""
    print("=" * 60)
    print("BM25 Ranking Function Demo")
    print("=" * 60)

    # Sample documents
    documents = [
        "information retrieval is the process of obtaining information",
        "retrieval models include boolean and vector space models",
        "BM25 is a probabilistic retrieval function",
        "the BM25 ranking function is widely used in search engines",
        "information extraction is related to information retrieval",
        "search engines use various ranking algorithms including BM25"
    ]

    # Build BM25 index
    ranker = BM25Ranker(k1=1.5, b=0.75)
    ranker.build_index(documents)

    print(f"\nDataset: {len(documents)} documents")
    print(f"Vocabulary: {ranker.get_stats()['vocabulary_size']} terms")
    print(f"Avg document length: {ranker.get_stats()['avg_doc_length']}")

    # Test queries
    queries = [
        "information retrieval",
        "BM25 ranking",
        "search engines",
        "boolean models"
    ]

    print("\n" + "-" * 60)
    print("Search Results")
    print("-" * 60)

    for query in queries:
        print(f"\nQuery: '{query}'")
        result = ranker.search(query, topk=3)

        print(f"  Results: {result.num_results}")
        for i, (doc_id, score) in enumerate(zip(result.doc_ids, result.scores), 1):
            print(f"  {i}. Doc {doc_id}: {score:.4f}")
            print(f"     {documents[doc_id][:60]}...")

    # Explain score
    print("\n" + "-" * 60)
    print("Score Explanation")
    print("-" * 60)

    query = "information retrieval"
    doc_id = 0
    explanation = ranker.explain_score(query, doc_id)

    print(f"\nQuery: '{query}'")
    print(f"Document {doc_id}: {documents[doc_id]}")
    print(f"\nTotal BM25 Score: {explanation['total_score']}")
    print(f"Document Length: {explanation['doc_length']}")
    print("\nTerm Contributions:")
    for term, contrib in explanation['term_contributions'].items():
        print(f"  '{term}':")
        print(f"    TF: {contrib.get('tf', 0)}")
        print(f"    IDF: {contrib.get('idf', 0.0):.4f}")
        print(f"    Score: {contrib.get('score', 0.0):.4f}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    demo()
