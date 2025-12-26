"""
Vector Space Model (VSM) Retrieval System

This module implements the vector space model for information retrieval.
Documents and queries are represented as weighted term vectors, and
similarity is computed using cosine similarity.

How VSM Works (high level):
    - Index time:
        1) Tokenize documents and build an inverted index (term -> postings)
        2) Compute global statistics (df/idf) from the index
        3) Compute a weighted (TF-IDF) vector for each document
    - Query time:
        1) Tokenize the query and build a weighted query vector
        2) Collect candidate documents that contain at least one query term
        3) Compute cosine similarity(query, doc) for each candidate
        4) Return top-k documents by similarity score

Key Features:
    - TF-IDF document representation
    - Cosine similarity ranking
    - Top-k retrieval with heap optimization
    - Multiple weighting schemes
    - Query processing

Author: Information Retrieval System
License: Educational Use
"""

import logging
import heapq
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

import sys
from pathlib import Path

_parent_dir = Path(__file__).parent.parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

from src.ir.index.inverted_index import InvertedIndex
from src.ir.index.term_weighting import TermWeighting


@dataclass
class VSMResult:
    """
    Result of a VSM query.

    Attributes:
        doc_ids: List of document IDs ranked by relevance
        scores: Similarity scores for each document
        query: Original query string
        num_results: Number of results
    """
    doc_ids: List[int]
    scores: Dict[int, float]
    query: str
    num_results: int


class VectorSpaceModel:
    """
    Vector Space Model for document retrieval.

    Uses TF-IDF weighting and cosine similarity to rank documents
    by relevance to a query.

    Weighting Schemes:
        - Documents: ltc (log TF, standard IDF, cosine norm)
        - Queries: lnc (log TF, no IDF, cosine norm) or ltc

    Complexity:
        - Indexing: O(T) where T is total tokens
        - Query: O(V + k*log(k)) where V is vocabulary, k is top-k
        - Top-k retrieval uses min-heap for efficiency

    Attributes:
        inverted_index: Inverted index for document collection
        term_weighting: Term weighting calculator
        doc_vectors: Pre-computed document vectors
    """

    def __init__(self, inverted_index: Optional[InvertedIndex] = None):
        """
        Initialize Vector Space Model.

        Args:
            inverted_index: Optional pre-built inverted index
        """
        self.logger = logging.getLogger(__name__)

        self.inverted_index = inverted_index or InvertedIndex()
        self.term_weighting = TermWeighting()
        self.doc_vectors: Dict[int, Dict[str, float]] = {}

        # Weighting schemes
        # We use SMART notation: [tf][idf][norm]
        # - Documents commonly use ltc (log tf, idf, cosine norm)
        # - Queries often use lnc (log tf, no idf, cosine norm) to avoid
        #   overweighting rare terms in short queries (a classic heuristic).
        self.doc_tf_scheme = 'l'     # log TF for documents
        self.doc_idf_scheme = 't'    # standard IDF
        self.doc_norm_scheme = 'c'   # cosine normalization

        self.query_tf_scheme = 'l'   # log TF for queries
        self.query_idf_scheme = 'n'  # no IDF for queries (lnc scheme)
        self.query_norm_scheme = 'c' # cosine normalization

        self.logger.info("VectorSpaceModel initialized")

    def build_index(self, documents: List[str],
                   metadata: Optional[List[dict]] = None) -> None:
        """
        Build index and compute document vectors.

        Args:
            documents: List of document texts
            metadata: Optional metadata for each document

        Complexity:
            Time: O(T + D*V) where T is total tokens,
                  D is documents, V is vocab size
            Space: O(D*V) for document vectors
        """
        self.logger.info(f"Building VSM index for {len(documents)} documents...")

        # 1) Build inverted index (term -> postings, plus doc lengths/metadata)
        self.inverted_index.build(documents, metadata)

        # 2) Build global weighting statistics (df/idf) from the index
        self.term_weighting.build_from_index(self.inverted_index)

        # 3) Pre-compute document TF-IDF vectors to make queries faster
        self._compute_document_vectors()

        self.logger.info(
            f"VSM index built: {len(self.doc_vectors)} document vectors"
        )

    def _compute_document_vectors(self) -> None:
        """
        Pre-compute weighted vectors for all documents.

        This improves query time by avoiding repeated computation.

        Complexity:
            Time: O(D*V_avg) where D is documents, V_avg is avg vocab per doc
        """
        self.doc_vectors = {}

        for doc_id in range(self.inverted_index.doc_count):
            # Build a sparse term-frequency map for this document, then apply
            # a weighting scheme (default: ltc) to produce a sparse TF-IDF vector.
            doc_tf = self._get_document_tf(doc_id)

            # Vectorize with ltc scheme (typical for documents)
            weighted = self.term_weighting.vectorize(
                doc_tf,
                tf_scheme=self.doc_tf_scheme,
                idf_scheme=self.doc_idf_scheme,
                normalize=self.doc_norm_scheme
            )

            self.doc_vectors[doc_id] = weighted

        self.logger.debug(f"Computed {len(self.doc_vectors)} document vectors")

    def _get_document_tf(self, doc_id: int) -> Dict[str, int]:
        """
        Get term frequency vector for a document.

        Args:
            doc_id: Document ID

        Returns:
            Dictionary {term: count}

        Complexity:
            Time: O(V) where V is vocabulary size
        """
        # NOTE: This implementation scans the full vocabulary and queries the
        # inverted index for each term. This is simple but can be expensive for
        # large vocabularies. A more efficient alternative is to iterate the
        # postings lists (term -> docs) and accumulate tf for this doc.
        doc_tf = {}

        for term in self.inverted_index.vocabulary:
            tf = self.inverted_index.term_frequency(term, doc_id)
            if tf > 0:
                doc_tf[term] = tf

        return doc_tf

    def search(self, query: str, topk: int = 10,
              use_idf_for_query: bool = False) -> VSMResult:
        """
        Search for documents matching query.

        Args:
            query: Query string
            topk: Number of top results to return
            use_idf_for_query: Whether to use IDF for query (ltc vs lnc)

        Returns:
            VSMResult with ranked documents

        Complexity:
            Time: O(|query| + C + k*log(k))
                  where C is candidate docs, k is topk
            Space: O(C) for scores

        Examples:
            >>> vsm = VectorSpaceModel()
            >>> vsm.build_index(["info retrieval", "vector space"])
            >>> result = vsm.search("information retrieval", topk=5)
            >>> result.doc_ids
            [0, 1]
        """
        self.logger.debug(f"Searching: {query}")

        # 1) Tokenize query and build a raw TF vector.
        query_tokens = self.inverted_index.tokenizer(query)

        if not query_tokens:
            return VSMResult(
                doc_ids=[],
                scores={},
                query=query,
                num_results=0
            )

        # Build query term frequency vector (bag-of-words).
        query_tf = defaultdict(int)
        for token in query_tokens:
            query_tf[token] += 1

        # 2) Vectorize query with a SMART-style scheme.
        # Default behavior is "lnc" for queries (log TF, no IDF, cosine norm).
        query_idf_scheme = 't' if use_idf_for_query else 'n'
        query_vector = self.term_weighting.vectorize(
            dict(query_tf),
            tf_scheme=self.query_tf_scheme,
            idf_scheme=query_idf_scheme,
            normalize=self.query_norm_scheme
        )

        # 3) Candidate generation: only score documents that share at least one
        # query term. This avoids scoring the entire collection.
        candidate_docs = set()
        for term in query_vector:
            candidate_docs.update(self.inverted_index.get_doc_ids(term))

        if not candidate_docs:
            return VSMResult(
                doc_ids=[],
                scores={},
                query=query,
                num_results=0
            )

        # 4) Score candidates by cosine similarity of sparse vectors.
        scores = {}
        for doc_id in candidate_docs:
            doc_vector = self.doc_vectors.get(doc_id, {})
            similarity = self.term_weighting.cosine_similarity(
                query_vector, doc_vector
            )
            if similarity > 0:
                scores[doc_id] = similarity

        # 5) Select the top-k results. `heapq.nlargest` avoids sorting the full
        # score list when the candidate set is large and k is small.
        if len(scores) <= topk:
            # All results fit
            ranked_docs = sorted(scores.items(),
                               key=lambda x: x[1],
                               reverse=True)
        else:
            # Use heap for top-k
            ranked_docs = heapq.nlargest(topk, scores.items(),
                                        key=lambda x: x[1])

        # Extract doc IDs and scores
        doc_ids = [doc_id for doc_id, _ in ranked_docs]
        score_dict = {doc_id: score for doc_id, score in ranked_docs}

        result = VSMResult(
            doc_ids=doc_ids,
            scores=score_dict,
            query=query,
            num_results=len(doc_ids)
        )

        self.logger.debug(f"Found {result.num_results} results")
        return result

    def set_weighting_scheme(self, doc_scheme: str = 'ltc',
                            query_scheme: str = 'lnc') -> None:
        """
        Set weighting schemes for documents and queries.

        Args:
            doc_scheme: Document scheme (e.g., 'ltc', 'lnc', 'atc')
            query_scheme: Query scheme (e.g., 'lnc', 'ltc')

        Scheme format: [tf][idf][norm]
            - tf: n (natural), l (log), a (augmented), b (boolean)
            - idf: n (none), t (standard), p (probabilistic)
            - norm: n (none), c (cosine)

        Examples:
            >>> vsm.set_weighting_scheme('ltc', 'lnc')  # Standard
            >>> vsm.set_weighting_scheme('atc', 'atc')  # Augmented
        """
        if len(doc_scheme) != 3 or len(query_scheme) != 3:
            raise ValueError("Scheme must be 3 characters: [tf][idf][norm]")

        # Parse document scheme
        self.doc_tf_scheme = doc_scheme[0]
        self.doc_idf_scheme = doc_scheme[1]
        self.doc_norm_scheme = doc_scheme[2]

        # Parse query scheme
        self.query_tf_scheme = query_scheme[0]
        self.query_idf_scheme = query_scheme[1]
        self.query_norm_scheme = query_scheme[2]

        # Changing weighting schemes invalidates cached document vectors, so we
        # recompute them if an index is already present.
        if self.inverted_index.doc_count > 0:
            self._compute_document_vectors()

        self.logger.info(
            f"Weighting schemes set: docs={doc_scheme}, queries={query_scheme}"
        )

    def get_document_vector(self, doc_id: int) -> Dict[str, float]:
        """
        Get weighted vector for a document.

        Args:
            doc_id: Document ID

        Returns:
            Weighted term vector
        """
        return self.doc_vectors.get(doc_id, {})

    def similarity(self, doc_id1: int, doc_id2: int) -> float:
        """
        Calculate similarity between two documents.

        Args:
            doc_id1: First document ID
            doc_id2: Second document ID

        Returns:
            Cosine similarity score

        Complexity:
            Time: O(min(V1, V2)) where V1, V2 are vocab sizes
        """
        vec1 = self.doc_vectors.get(doc_id1, {})
        vec2 = self.doc_vectors.get(doc_id2, {})

        return self.term_weighting.cosine_similarity(vec1, vec2)

    def get_similar_documents(self, doc_id: int, topk: int = 10) -> List[Tuple[int, float]]:
        """
        Find documents similar to a given document.

        Args:
            doc_id: Reference document ID
            topk: Number of similar documents to return

        Returns:
            List of (doc_id, similarity) tuples

        Complexity:
            Time: O(D*V + k*log(k)) where D is documents, V is vocab
        """
        if doc_id not in self.doc_vectors:
            return []

        ref_vector = self.doc_vectors[doc_id]

        # Calculate similarities
        similarities = []
        for other_id, other_vector in self.doc_vectors.items():
            if other_id == doc_id:
                continue

            sim = self.term_weighting.cosine_similarity(ref_vector, other_vector)
            if sim > 0:
                similarities.append((other_id, sim))

        # Sort and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:topk]

    def get_stats(self) -> dict:
        """Get VSM statistics."""
        stats = self.inverted_index.get_stats()
        stats.update({
            'num_doc_vectors': len(self.doc_vectors),
            'doc_scheme': f"{self.doc_tf_scheme}{self.doc_idf_scheme}{self.doc_norm_scheme}",
            'query_scheme': f"{self.query_tf_scheme}{self.query_idf_scheme}{self.query_norm_scheme}"
        })
        return stats


def demo():
    """Demonstration of Vector Space Model."""
    print("=" * 60)
    print("Vector Space Model Demo")
    print("=" * 60)

    # Sample documents
    documents = [
        "information retrieval systems are important",
        "vector space model for information retrieval",
        "boolean retrieval model",
        "tf idf weighting scheme",
        "cosine similarity for ranking documents"
    ]

    # Build VSM
    vsm = VectorSpaceModel()
    vsm.build_index(documents)

    print(f"\n1. Index Statistics:")
    stats = vsm.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    # Search queries
    print(f"\n2. Search Results:")
    queries = [
        "information retrieval",
        "vector space model",
        "ranking documents"
    ]

    for query in queries:
        result = vsm.search(query, topk=3)
        print(f"\n   Query: '{query}'")
        print(f"   Results: {result.num_results} documents")
        for i, doc_id in enumerate(result.doc_ids, 1):
            score = result.scores[doc_id]
            print(f"      {i}. Doc {doc_id}: {score:.4f}")

    # Document similarity
    print(f"\n3. Document Similarity:")
    print(f"   Doc 0 vs Doc 1: {vsm.similarity(0, 1):.4f}")
    print(f"   Doc 0 vs Doc 2: {vsm.similarity(0, 2):.4f}")

    # Similar documents
    print(f"\n4. Similar to Doc 0:")
    similar = vsm.get_similar_documents(0, topk=3)
    for doc_id, sim in similar:
        print(f"      Doc {doc_id}: {sim:.4f}")

    print("\n" + "=" * 60)


# Alias for backward compatibility
VSMRetrieval = VectorSpaceModel


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    demo()
