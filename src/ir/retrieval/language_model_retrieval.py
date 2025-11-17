"""
Language Model Retrieval

This module implements query likelihood retrieval using statistical language models.
Documents are ranked by P(Q|D) - the probability of generating the query from
the document's language model.

Language Model (LM) Retrieval is a probabilistic approach that estimates how likely
a document would generate the query terms. Unlike BM25, it models documents as
generative probabilistic models.

Key Concepts:
    - Query Likelihood: Rank documents by P(Q|D) = P(q1|D) * P(q2|D) * ... * P(qn|D)
    - Document Language Model: Probabilistic model of term generation in document
    - Smoothing: Essential for handling zero probabilities (unseen terms)
    - Collection Model: Background model P(w|C) for smoothing

Formulas:
    - Unigram Query Likelihood: P(Q|D) = ∏ P(qi|D)
    - Jelinek-Mercer: P(w|D) = λ * P_ML(w|D) + (1-λ) * P(w|C)
    - Dirichlet: P(w|D) = (tf(w,D) + μ * P(w|C)) / (|D| + μ)
    - KL-Divergence: KL(D||Q) = Σ P(w|Q) * log(P(w|Q) / P(w|D))

Key Features:
    - Multiple smoothing methods (Jelinek-Mercer, Dirichlet, Absolute Discounting)
    - Query likelihood scoring for ranking
    - KL-divergence alternative ranking
    - Support for Chinese and English text

Reference:
    Zhai & Lafferty (2004). "A Study of Smoothing Methods for Language Models Applied to IR"
    Ponte & Croft (1998). "A Language Modeling Approach to Information Retrieval"

Author: Information Retrieval System
License: Educational Use
"""

import math
import logging
from typing import List, Dict, Set, Tuple, Optional, Callable
from dataclasses import dataclass
from collections import defaultdict, Counter


@dataclass
class LMRetrievalResult:
    """
    Result of language model retrieval.

    Attributes:
        query: Original query string
        doc_ids: List of document IDs ranked by likelihood
        scores: Corresponding log-likelihood scores
        num_results: Total number of results
        parameters: LM parameters used (smoothing method, λ, μ)
    """
    query: str
    doc_ids: List[int]
    scores: List[float]
    num_results: int
    parameters: Dict[str, any]


class LanguageModelRetrieval:
    """
    Language Model Retrieval using query likelihood.

    Ranks documents by P(Q|D) - the probability that the document's
    language model would generate the query.

    Attributes:
        smoothing: Smoothing method ('jm', 'dirichlet', 'absolute')
        lambda_param: λ for Jelinek-Mercer smoothing
        mu_param: μ for Dirichlet smoothing
        delta_param: δ for Absolute Discounting
        tokenizer: Tokenization function
        doc_models: Document language models
        collection_model: Collection-wide term probabilities
        doc_lengths: Document lengths

    Complexity:
        - Indexing: O(N * M) where N=docs, M=avg terms per doc
        - Query: O(T * D) where T=query terms, D=candidate docs
        - Smoothing: O(1) per term with precomputed collection model
    """

    def __init__(self,
                 tokenizer: Optional[Callable[[str], List[str]]] = None,
                 smoothing: str = 'dirichlet',
                 lambda_param: float = 0.7,
                 mu_param: float = 2000.0,
                 delta_param: float = 0.7):
        """
        Initialize Language Model Retrieval.

        Args:
            tokenizer: Custom tokenization function
            smoothing: Smoothing method ('jm', 'dirichlet', 'absolute')
            lambda_param: λ for Jelinek-Mercer (default: 0.7, range: 0-1)
            mu_param: μ for Dirichlet prior (default: 2000, range: 500-5000)
            delta_param: δ for Absolute Discounting (default: 0.7)

        Complexity:
            Time: O(1)
        """
        self.logger = logging.getLogger(__name__)

        # Parameters
        self.smoothing = smoothing
        self.lambda_param = lambda_param
        self.mu_param = mu_param
        self.delta_param = delta_param

        # Tokenizer
        self.tokenizer = tokenizer or self._default_tokenizer

        # Document models
        self.doc_models: Dict[int, Dict[str, int]] = {}  # doc_id -> {term: tf}
        self.doc_lengths: Dict[int, int] = {}  # doc_id -> length
        self.doc_count: int = 0

        # Collection model
        self.collection_model: Dict[str, float] = {}  # term -> P(term|C)
        self.collection_size: int = 0  # Total terms in collection
        self.vocab: Set[str] = set()

        self.logger.info(
            f"LanguageModelRetrieval initialized "
            f"(smoothing={smoothing}, λ={lambda_param}, μ={mu_param}, δ={delta_param})"
        )

    def _default_tokenizer(self, text: str) -> List[str]:
        """Default tokenizer (simple split)."""
        import re
        return re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z0-9]+', text.lower())

    def build_index(self, documents: List[str]) -> None:
        """
        Build language models for all documents.

        Args:
            documents: List of document texts

        Complexity:
            Time: O(N * M) where N=docs, M=avg terms per doc
            Space: O(V * D) where V=vocab, D=avg unique terms per doc
        """
        self.logger.info(f"Building LM index for {len(documents)} documents...")

        self.doc_count = len(documents)
        collection_term_counts = Counter()

        # Build document models
        for doc_id, doc_text in enumerate(documents):
            tokens = self.tokenizer(doc_text)
            doc_length = len(tokens)

            # Count term frequencies
            term_freqs = Counter(tokens)

            # Store document model
            self.doc_models[doc_id] = dict(term_freqs)
            self.doc_lengths[doc_id] = doc_length

            # Update collection statistics
            collection_term_counts.update(tokens)
            self.vocab.update(tokens)

        # Build collection model
        self.collection_size = sum(collection_term_counts.values())
        for term, count in collection_term_counts.items():
            self.collection_model[term] = count / self.collection_size

        vocab_size = len(self.vocab)
        avg_doc_length = sum(self.doc_lengths.values()) / self.doc_count

        self.logger.info(
            f"LM index built: {self.doc_count} docs, {vocab_size} vocab, "
            f"avg_doc_length={avg_doc_length:.2f}, collection_size={self.collection_size}"
        )

    def term_probability(self, term: str, doc_id: int) -> float:
        """
        Calculate smoothed probability P(term|doc).

        Args:
            term: Query term
            doc_id: Document ID

        Returns:
            Smoothed probability P(term|doc)

        Complexity:
            Time: O(1) average

        Examples:
            >>> lm.term_probability("retrieval", 5)
            0.0234
        """
        if doc_id not in self.doc_models:
            return 0.0

        # Term frequency in document
        tf = self.doc_models[doc_id].get(term, 0)
        doc_length = self.doc_lengths[doc_id]

        # Collection probability (for smoothing)
        p_collection = self.collection_model.get(term, 1.0 / len(self.vocab))

        # Apply smoothing
        if self.smoothing == 'jm':
            return self._jelinek_mercer_smoothing(tf, doc_length, p_collection)
        elif self.smoothing == 'dirichlet':
            return self._dirichlet_smoothing(tf, doc_length, p_collection)
        elif self.smoothing == 'absolute':
            return self._absolute_discounting_smoothing(tf, doc_length, p_collection)
        else:
            # Maximum likelihood (no smoothing - NOT RECOMMENDED)
            return tf / doc_length if doc_length > 0 else 0.0

    def _jelinek_mercer_smoothing(self, tf: int, doc_length: int, p_collection: float) -> float:
        """
        Jelinek-Mercer (linear interpolation) smoothing.

        P(w|D) = λ * P_ML(w|D) + (1-λ) * P(w|C)

        Args:
            tf: Term frequency in document
            doc_length: Document length
            p_collection: Collection probability P(w|C)

        Returns:
            Smoothed probability
        """
        p_ml = tf / doc_length if doc_length > 0 else 0.0
        return self.lambda_param * p_ml + (1 - self.lambda_param) * p_collection

    def _dirichlet_smoothing(self, tf: int, doc_length: int, p_collection: float) -> float:
        """
        Dirichlet prior smoothing (Bayesian smoothing).

        P(w|D) = (tf + μ * P(w|C)) / (|D| + μ)

        where μ is the Dirichlet prior parameter.

        Args:
            tf: Term frequency in document
            doc_length: Document length
            p_collection: Collection probability P(w|C)

        Returns:
            Smoothed probability
        """
        numerator = tf + self.mu_param * p_collection
        denominator = doc_length + self.mu_param
        return numerator / denominator if denominator > 0 else 0.0

    def _absolute_discounting_smoothing(self, tf: int, doc_length: int, p_collection: float) -> float:
        """
        Absolute discounting smoothing.

        P(w|D) = max(tf - δ, 0) / |D| + α * P(w|C)

        where α is a normalization factor.

        Args:
            tf: Term frequency in document
            doc_length: Document length
            p_collection: Collection probability P(w|C)

        Returns:
            Smoothed probability
        """
        if doc_length == 0:
            return 0.0

        # Discount term frequency
        discounted_tf = max(tf - self.delta_param, 0)

        # Calculate normalization factor
        # α = δ * |{w : tf(w,D) > 0}| / |D|
        unique_terms = len(self.doc_models.get(doc_id, {}))
        alpha = self.delta_param * unique_terms / doc_length

        return (discounted_tf / doc_length) + (alpha * p_collection)

    def query_likelihood(self, query_terms: List[str], doc_id: int) -> float:
        """
        Calculate query likelihood P(Q|D).

        P(Q|D) = P(q1|D) * P(q2|D) * ... * P(qn|D)

        Uses log probabilities to avoid underflow:
        log P(Q|D) = Σ log P(qi|D)

        Args:
            query_terms: List of query terms
            doc_id: Document ID

        Returns:
            Log query likelihood

        Complexity:
            Time: O(T) where T = number of query terms

        Examples:
            >>> lm.query_likelihood(["information", "retrieval"], 5)
            -12.456
        """
        log_likelihood = 0.0

        for term in query_terms:
            prob = self.term_probability(term, doc_id)

            if prob > 0:
                log_likelihood += math.log(prob)
            else:
                # Should not happen with proper smoothing
                log_likelihood += math.log(1e-10)  # Small epsilon

        return log_likelihood

    def search(self, query: str, topk: int = 10) -> LMRetrievalResult:
        """
        Search documents using language model retrieval.

        Ranks documents by query likelihood P(Q|D).

        Args:
            query: Query string
            topk: Number of top results to return

        Returns:
            LMRetrievalResult with ranked documents

        Complexity:
            Time: O(N * T) where N=docs, T=query terms
            Space: O(N) for scoring all documents

        Examples:
            >>> lm = LanguageModelRetrieval()
            >>> lm.build_index(documents)
            >>> result = lm.search("information retrieval", topk=10)
            >>> result.doc_ids
            [5, 12, 3, 18, ...]
        """
        # Tokenize query
        query_terms = self.tokenizer(query)

        if not query_terms:
            return LMRetrievalResult(
                query=query,
                doc_ids=[],
                scores=[],
                num_results=0,
                parameters={
                    'smoothing': self.smoothing,
                    'lambda': self.lambda_param,
                    'mu': self.mu_param,
                    'delta': self.delta_param
                }
            )

        # Score all documents
        doc_scores: List[Tuple[int, float]] = []
        for doc_id in range(self.doc_count):
            score = self.query_likelihood(query_terms, doc_id)
            doc_scores.append((doc_id, score))

        # Sort by score (descending - higher log likelihood is better)
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        # Get top-k
        top_docs = doc_scores[:topk]

        doc_ids = [doc_id for doc_id, score in top_docs]
        scores = [score for doc_id, score in top_docs]

        return LMRetrievalResult(
            query=query,
            doc_ids=doc_ids,
            scores=scores,
            num_results=len(doc_ids),
            parameters={
                'smoothing': self.smoothing,
                'lambda': self.lambda_param,
                'mu': self.mu_param,
                'delta': self.delta_param
            }
        )

    def kl_divergence_score(self, query_terms: List[str], doc_id: int) -> float:
        """
        Calculate KL-divergence score KL(Q||D).

        KL(Q||D) = Σ P(w|Q) * log(P(w|Q) / P(w|D))

        Lower KL-divergence means more similar (closer distribution).
        For ranking, we use negative KL-divergence (higher is better).

        Args:
            query_terms: List of query terms
            doc_id: Document ID

        Returns:
            Negative KL-divergence (for ranking)

        Complexity:
            Time: O(T) where T = unique query terms

        Examples:
            >>> lm.kl_divergence_score(["information", "retrieval"], 5)
            -0.234
        """
        # Build query model (simple maximum likelihood)
        query_counts = Counter(query_terms)
        query_length = len(query_terms)

        kl_div = 0.0

        for term, count in query_counts.items():
            p_query = count / query_length
            p_doc = self.term_probability(term, doc_id)

            if p_doc > 0:
                kl_div += p_query * math.log(p_query / p_doc)
            else:
                # Should not happen with smoothing
                kl_div += p_query * math.log(p_query / 1e-10)

        # Return negative KL-divergence (higher is better for ranking)
        return -kl_div

    def explain_score(self, query: str, doc_id: int) -> Dict:
        """
        Explain language model score breakdown for a document.

        Args:
            query: Query string
            doc_id: Document ID

        Returns:
            Dictionary with score explanation

        Examples:
            >>> explain = lm.explain_score("information retrieval", 5)
            >>> print(explain)
            {
                'doc_id': 5,
                'total_log_likelihood': -12.456,
                'doc_length': 150,
                'smoothing': 'dirichlet',
                'term_contributions': {
                    'information': {'tf': 3, 'p_ml': 0.02, 'p_smoothed': 0.018, 'log_prob': -4.02},
                    'retrieval': {'tf': 2, 'p_ml': 0.013, 'p_smoothed': 0.012, 'log_prob': -4.42}
                }
            }
        """
        query_terms = self.tokenizer(query)

        if doc_id not in self.doc_models:
            return {'error': 'Document not found'}

        doc_length = self.doc_lengths[doc_id]
        term_contributions = {}
        total_log_likelihood = 0.0

        for term in query_terms:
            tf = self.doc_models[doc_id].get(term, 0)
            p_ml = tf / doc_length if doc_length > 0 else 0.0
            p_smoothed = self.term_probability(term, doc_id)
            log_prob = math.log(p_smoothed) if p_smoothed > 0 else math.log(1e-10)

            total_log_likelihood += log_prob

            term_contributions[term] = {
                'tf': tf,
                'p_ml': round(p_ml, 6),
                'p_collection': round(self.collection_model.get(term, 0.0), 6),
                'p_smoothed': round(p_smoothed, 6),
                'log_prob': round(log_prob, 4)
            }

        return {
            'doc_id': doc_id,
            'total_log_likelihood': round(total_log_likelihood, 4),
            'doc_length': doc_length,
            'smoothing': self.smoothing,
            'parameters': {
                'lambda': self.lambda_param,
                'mu': self.mu_param,
                'delta': self.delta_param
            },
            'term_contributions': term_contributions
        }

    def get_stats(self) -> Dict:
        """Get language model retrieval statistics."""
        return {
            'doc_count': self.doc_count,
            'vocabulary_size': len(self.vocab),
            'collection_size': self.collection_size,
            'avg_doc_length': round(sum(self.doc_lengths.values()) / self.doc_count, 2) if self.doc_count > 0 else 0,
            'smoothing': self.smoothing,
            'parameters': {
                'lambda': self.lambda_param,
                'mu': self.mu_param,
                'delta': self.delta_param
            }
        }


def demo():
    """Demonstration of Language Model Retrieval."""
    print("=" * 60)
    print("Language Model Retrieval Demo")
    print("=" * 60)

    # Sample documents
    documents = [
        "information retrieval is the process of obtaining information",
        "retrieval models include boolean and vector space models",
        "language models estimate the probability of word sequences",
        "statistical language modeling is used in information retrieval",
        "query likelihood is a language model approach to retrieval",
        "smoothing is essential for language model retrieval"
    ]

    # Test different smoothing methods
    smoothing_methods = [
        ('dirichlet', {'mu_param': 2000}),
        ('jm', {'lambda_param': 0.7}),
    ]

    for smoothing, params in smoothing_methods:
        print(f"\n{'=' * 60}")
        print(f"Smoothing: {smoothing.upper()}")
        print("=" * 60)

        # Build LM index
        lm = LanguageModelRetrieval(smoothing=smoothing, **params)
        lm.build_index(documents)

        stats = lm.get_stats()
        print(f"\nDataset: {stats['doc_count']} documents")
        print(f"Vocabulary: {stats['vocabulary_size']} terms")
        print(f"Avg document length: {stats['avg_doc_length']}")
        print(f"Parameters: {stats['parameters']}")

        # Test queries
        queries = [
            "information retrieval",
            "language model",
            "smoothing probability"
        ]

        print(f"\n{'-' * 60}")
        print("Search Results")
        print("-" * 60)

        for query in queries:
            print(f"\nQuery: '{query}'")
            result = lm.search(query, topk=3)

            print(f"  Results: {result.num_results}")
            for i, (doc_id, score) in enumerate(zip(result.doc_ids, result.scores), 1):
                print(f"  {i}. Doc {doc_id}: {score:.4f}")
                print(f"     {documents[doc_id][:60]}...")

        # Explain score
        print(f"\n{'-' * 60}")
        print("Score Explanation")
        print("-" * 60)

        query = "language model retrieval"
        doc_id = 4
        explanation = lm.explain_score(query, doc_id)

        print(f"\nQuery: '{query}'")
        print(f"Document {doc_id}: {documents[doc_id]}")
        print(f"\nTotal Log Likelihood: {explanation['total_log_likelihood']}")
        print(f"Document Length: {explanation['doc_length']}")
        print(f"Smoothing: {explanation['smoothing']}")
        print("\nTerm Contributions:")
        for term, contrib in explanation['term_contributions'].items():
            print(f"  '{term}':")
            print(f"    TF: {contrib['tf']}")
            print(f"    P_ML(w|D): {contrib['p_ml']:.6f}")
            print(f"    P(w|C): {contrib['p_collection']:.6f}")
            print(f"    P_smoothed(w|D): {contrib['p_smoothed']:.6f}")
            print(f"    Log P: {contrib['log_prob']:.4f}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    demo()
