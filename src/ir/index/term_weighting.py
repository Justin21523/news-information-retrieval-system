"""
Term Weighting for Vector Space Model

This module implements various term weighting schemes including TF-IDF,
which are essential for the vector space model. It supports multiple
weighting variants (ltc, lnc, etc.) commonly used in IR systems.

Key Features:
    - TF (Term Frequency) calculation
    - IDF (Inverse Document Frequency) calculation
    - TF-IDF weighting with multiple schemes
    - Document normalization
    - Vector representation

Author: Information Retrieval System
License: Educational Use
"""

import math
import logging
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np


class TermWeighting:
    """
    Term Weighting Calculator for Vector Space Model.

    Supports multiple TF-IDF weighting schemes:
        - TF schemes: natural (n), logarithmic (l), augmented (a), boolean (b)
        - IDF schemes: none (n), standard (t), probabilistic (p)
        - Normalization: none (n), cosine (c)

    Common schemes:
        - ltc: log TF, standard IDF, cosine normalization (for documents)
        - lnc: log TF, no IDF, cosine normalization (for queries)
        - atc: augmented TF, standard IDF, cosine normalization

    Attributes:
        doc_count: Total number of documents
        df: Document frequency for each term
        idf: IDF values for each term
    """

    def __init__(self):
        """Initialize term weighting calculator."""
        self.logger = logging.getLogger(__name__)

        self.doc_count: int = 0
        self.df: Dict[str, int] = {}  # Document frequency
        self.idf: Dict[str, float] = {}  # IDF values
        self.avg_doc_length: float = 0.0

        self.logger.info("TermWeighting initialized")

    def build_from_index(self, inverted_index) -> None:
        """
        Build term statistics from inverted index.

        Args:
            inverted_index: InvertedIndex instance

        Complexity:
            Time: O(V) where V is vocabulary size
        """
        self.doc_count = inverted_index.doc_count

        # Calculate DF for each term
        for term in inverted_index.vocabulary:
            self.df[term] = inverted_index.document_frequency(term)

        # Calculate IDF
        self._calculate_idf()

        # Calculate average document length
        if inverted_index.doc_lengths:
            self.avg_doc_length = (
                sum(inverted_index.doc_lengths.values()) / self.doc_count
            )

        self.logger.info(
            f"Built term statistics: {len(self.df)} terms, "
            f"{self.doc_count} documents"
        )

    def _calculate_idf(self) -> None:
        """
        Calculate IDF values for all terms.

        IDF(t) = log(N / df(t))
        where N is total documents, df(t) is document frequency of term t

        Complexity:
            Time: O(V) where V is vocabulary size
        """
        for term, df_value in self.df.items():
            # Standard IDF formula
            self.idf[term] = math.log10(self.doc_count / df_value)

    def tf(self, term: str, doc_vector: Dict[str, int], scheme: str = 'n') -> float:
        """
        Calculate term frequency with various schemes.

        Args:
            term: Term to calculate TF for
            doc_vector: Document term frequency vector {term: count}
            scheme: TF scheme
                - 'n': natural (raw count)
                - 'l': logarithmic (1 + log(count))
                - 'a': augmented (0.5 + 0.5 * count / max_count)
                - 'b': boolean (1 if present, 0 otherwise)

        Returns:
            TF value

        Complexity:
            Time: O(1) for n, l, b; O(|doc_vector|) for a

        Examples:
            >>> tw = TermWeighting()
            >>> doc = {"hello": 3, "world": 5}
            >>> tw.tf("hello", doc, scheme='n')
            3
            >>> tw.tf("hello", doc, scheme='l')
            1.477  # 1 + log10(3)
        """
        if term not in doc_vector:
            return 0.0

        count = doc_vector[term]

        if scheme == 'n':
            # Natural: raw count
            return float(count)

        elif scheme == 'l':
            # Logarithmic: 1 + log(count)
            return 1.0 + math.log10(count) if count > 0 else 0.0

        elif scheme == 'a':
            # Augmented: 0.5 + 0.5 * (count / max_count)
            max_count = max(doc_vector.values()) if doc_vector else 1
            return 0.5 + 0.5 * (count / max_count)

        elif scheme == 'b':
            # Boolean: 1 if present, 0 otherwise
            return 1.0 if count > 0 else 0.0

        else:
            raise ValueError(f"Unknown TF scheme: {scheme}")

    def idf_value(self, term: str, scheme: str = 't') -> float:
        """
        Get IDF value for a term with various schemes.

        Args:
            term: Term to get IDF for
            scheme: IDF scheme
                - 'n': none (always 1.0)
                - 't': standard (log(N/df))
                - 'p': probabilistic (log((N-df)/df))

        Returns:
            IDF value

        Complexity:
            Time: O(1)
        """
        if scheme == 'n':
            # No IDF
            return 1.0

        elif scheme == 't':
            # Standard IDF
            return self.idf.get(term, 0.0)

        elif scheme == 'p':
            # Probabilistic IDF: log((N - df) / df)
            if term not in self.df:
                return 0.0
            df_value = self.df[term]
            if df_value >= self.doc_count:
                return 0.0
            return math.log10((self.doc_count - df_value) / df_value)

        else:
            raise ValueError(f"Unknown IDF scheme: {scheme}")

    def tf_idf(self, term: str, doc_vector: Dict[str, int],
               tf_scheme: str = 'l', idf_scheme: str = 't') -> float:
        """
        Calculate TF-IDF weight for a term in a document.

        Args:
            term: Term
            doc_vector: Document term frequency vector
            tf_scheme: TF scheme (n, l, a, b)
            idf_scheme: IDF scheme (n, t, p)

        Returns:
            TF-IDF weight

        Complexity:
            Time: O(1) for most schemes, O(|doc_vector|) for augmented TF

        Examples:
            >>> tw.tf_idf("hello", {"hello": 3, "world": 5}, 'l', 't')
            2.15  # (1 + log(3)) * log(N/df)
        """
        tf_value = self.tf(term, doc_vector, tf_scheme)
        idf_value = self.idf_value(term, idf_scheme)

        return tf_value * idf_value

    def vectorize(self, doc_vector: Dict[str, int],
                  tf_scheme: str = 'l', idf_scheme: str = 't',
                  normalize: str = 'c') -> Dict[str, float]:
        """
        Convert document to weighted vector.

        Args:
            doc_vector: Raw term frequency vector
            tf_scheme: TF scheme
            idf_scheme: IDF scheme
            normalize: Normalization scheme
                - 'n': none
                - 'c': cosine (L2 normalization)

        Returns:
            Weighted vector {term: weight}

        Complexity:
            Time: O(|doc_vector|)
            Space: O(|doc_vector|)

        Examples:
            >>> doc = {"hello": 3, "world": 5}
            >>> tw.vectorize(doc, 'l', 't', 'c')
            {"hello": 0.6, "world": 0.8}  # normalized
        """
        weighted = {}

        # The returned vector is intentionally sparse:
        # we only include terms that appear in `doc_vector`.
        #
        # SMART notation reminder:
        #   tf_scheme controls local term frequency scaling
        #   idf_scheme controls global document frequency scaling
        #   normalize controls final vector length normalization
        #
        # This method is used for both documents and queries.
        # Calculate TF-IDF for each term
        for term, count in doc_vector.items():
            weight = self.tf_idf(term, doc_vector, tf_scheme, idf_scheme)
            if weight > 0:
                weighted[term] = weight

        # Normalize if requested
        if normalize == 'c':
            # Cosine normalization (L2 norm)
            norm = math.sqrt(sum(w**2 for w in weighted.values()))
            if norm > 0:
                weighted = {term: w / norm for term, w in weighted.items()}

        elif normalize != 'n':
            raise ValueError(f"Unknown normalization scheme: {normalize}")

        return weighted

    def cosine_similarity(self, vec1: Dict[str, float],
                         vec2: Dict[str, float]) -> float:
        """
        Calculate cosine similarity between two vectors.

        Cosine similarity = (vec1 · vec2) / (||vec1|| * ||vec2||)

        Args:
            vec1: First vector {term: weight}
            vec2: Second vector {term: weight}

        Returns:
            Similarity score in [0, 1]

        Complexity:
            Time: O(min(|vec1|, |vec2|))
            Space: O(1)

        Examples:
            >>> v1 = {"hello": 0.6, "world": 0.8}
            >>> v2 = {"hello": 0.8, "world": 0.6}
            >>> tw.cosine_similarity(v1, v2)
            0.96
        """
        # Iterate over the smaller vector to reduce hash lookups.
        if len(vec1) > len(vec2):
            vec1, vec2 = vec2, vec1

        # Calculate dot product
        dot_product = 0.0
        for term in vec1:
            if term in vec2:
                dot_product += vec1[term] * vec2[term]

        if dot_product == 0:
            return 0.0

        # Calculate norms
        norm1 = math.sqrt(sum(w**2 for w in vec1.values()))
        norm2 = math.sqrt(sum(w**2 for w in vec2.values()))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def euclidean_distance(self, vec1: Dict[str, float],
                          vec2: Dict[str, float]) -> float:
        """
        Calculate Euclidean distance between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Distance value

        Complexity:
            Time: O(|vec1| + |vec2|)
        """
        # Get all terms
        all_terms = set(vec1.keys()) | set(vec2.keys())

        # Calculate squared differences
        sum_sq_diff = 0.0
        for term in all_terms:
            v1 = vec1.get(term, 0.0)
            v2 = vec2.get(term, 0.0)
            sum_sq_diff += (v1 - v2) ** 2

        return math.sqrt(sum_sq_diff)

    def get_top_terms(self, doc_vector: Dict[str, int],
                     topk: int = 10,
                     tf_scheme: str = 'l',
                     idf_scheme: str = 't') -> List[Tuple[str, float]]:
        """
        Get top-k terms by TF-IDF weight.

        Args:
            doc_vector: Document term frequency vector
            topk: Number of top terms to return
            tf_scheme: TF scheme
            idf_scheme: IDF scheme

        Returns:
            List of (term, weight) tuples sorted by weight descending

        Complexity:
            Time: O(|doc_vector| * log(topk))
        """
        # Calculate weights
        weights = []
        for term, count in doc_vector.items():
            weight = self.tf_idf(term, doc_vector, tf_scheme, idf_scheme)
            weights.append((term, weight))

        # Sort and return top-k
        weights.sort(key=lambda x: x[1], reverse=True)
        return weights[:topk]

    def bm25_score(self, term: str, doc_vector: Dict[str, int],
                   doc_length: int, k1: float = 1.5, b: float = 0.75) -> float:
        """
        Calculate BM25 score for a term (bonus implementation).

        BM25 = IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (|d| / avgdl)))

        Args:
            term: Term
            doc_vector: Document term frequencies
            doc_length: Document length
            k1: BM25 parameter (term frequency saturation)
            b: BM25 parameter (length normalization)

        Returns:
            BM25 score

        Complexity:
            Time: O(1)
        """
        if term not in doc_vector:
            return 0.0

        tf = doc_vector[term]
        idf = self.idf_value(term, scheme='t')

        # BM25 formula
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (
            1 - b + b * (doc_length / self.avg_doc_length)
        )

        return idf * (numerator / denominator)

    def get_stats(self) -> dict:
        """Get weighting statistics."""
        return {
            'doc_count': self.doc_count,
            'vocabulary_size': len(self.df),
            'avg_doc_length': self.avg_doc_length,
            'max_idf': max(self.idf.values()) if self.idf else 0.0,
            'min_idf': min(self.idf.values()) if self.idf else 0.0
        }


def demo():
    """Demonstration of term weighting."""
    print("=" * 60)
    print("Term Weighting Demo")
    print("=" * 60)

    # Sample document collection (simplified)
    documents = [
        {"information": 2, "retrieval": 3, "system": 1},
        {"retrieval": 1, "model": 2, "boolean": 1},
        {"vector": 2, "space": 2, "model": 1}
    ]

    # Initialize
    tw = TermWeighting()
    tw.doc_count = len(documents)

    # Calculate DF and IDF
    for doc in documents:
        for term in doc:
            tw.df[term] = tw.df.get(term, 0) + 1

    tw._calculate_idf()

    print("\n1. Document Frequency (DF):")
    for term, df in sorted(tw.df.items()):
        print(f"   {term}: {df}")

    print("\n2. Inverse Document Frequency (IDF):")
    for term, idf in sorted(tw.idf.items()):
        print(f"   {term}: {idf:.3f}")

    print("\n3. TF-IDF Weighting (ltc scheme):")
    for i, doc in enumerate(documents):
        weighted = tw.vectorize(doc, tf_scheme='l', idf_scheme='t', normalize='c')
        print(f"\n   Document {i}:")
        for term, weight in sorted(weighted.items(), key=lambda x: x[1], reverse=True):
            print(f"      {term}: {weight:.3f}")

    print("\n4. Cosine Similarity:")
    v1 = tw.vectorize(documents[0], 'l', 't', 'c')
    v2 = tw.vectorize(documents[1], 'l', 't', 'c')
    sim = tw.cosine_similarity(v1, v2)
    print(f"   Doc 0 vs Doc 1: {sim:.3f}")

    print("\n" + "=" * 60)


# Alias for backward compatibility
TFIDFWeighting = TermWeighting


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    demo()
