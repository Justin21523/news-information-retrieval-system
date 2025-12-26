"""
Inverted Index for Information Retrieval

This module implements an inverted index data structure for efficient
document retrieval. The inverted index maps terms to the documents
containing them, enabling fast lookup and Boolean query processing.

Implementation Overview:
    - Indexing:
        1) Tokenize each document into a sequence of normalized terms
        2) Accumulate per-document term frequencies
        3) Store postings lists sorted by doc_id for each term
    - Query-time building blocks:
        - get_postings(term): retrieve a postings list
        - intersect(p1, p2): merge-based AND on sorted postings
        - union(p1, p2): merge-based OR on sorted postings

Key Features:
    - Build inverted index from document collection
    - Support term frequency tracking
    - Document-level indexing
    - Memory-efficient posting lists
    - Flexible tokenization

Author: Information Retrieval System
License: Educational Use
"""

import re
import logging
from bisect import bisect_left
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional, Callable
from pathlib import Path
import json


class InvertedIndex:
    """
    Inverted Index for document retrieval.

    The inverted index maintains a mapping from terms to posting lists,
    where each posting contains a document ID and term frequency.

    Data Structure:
        index: {
            term1: [(doc_id, term_freq), (doc_id, term_freq), ...],
            term2: [(doc_id, term_freq), ...],
            ...
        }

    Invariant:
        Each postings list is sorted by doc_id. This enables linear-time merge
        operations (AND/OR) and also allows binary search for term_frequency().

    Complexity:
        - Build: O(T) where T is total number of tokens
        - Lookup: O(1) average for term
        - Merge postings: O(n + m) where n, m are posting list sizes

    Attributes:
        index (dict): The inverted index mapping terms to posting lists
        doc_count (int): Total number of indexed documents
        doc_lengths (dict): Document lengths (token count)
        vocabulary (set): Set of all unique terms
    """

    def __init__(self, tokenizer: Optional[Callable[[str], List[str]]] = None):
        """
        Initialize InvertedIndex.

        Args:
            tokenizer: Custom tokenization function. If None, uses default.
                      Function signature: tokenizer(text: str) -> List[str]
        """
        self.logger = logging.getLogger(__name__)

        # Core data structures
        self.index: Dict[str, List[Tuple[int, int]]] = {}
        self.doc_count: int = 0
        self.doc_lengths: Dict[int, int] = {}
        self.doc_metadata: Dict[int, dict] = {}  # Store document metadata

        # Tokenizer
        self.tokenizer = tokenizer or self._default_tokenizer

        self.logger.info("InvertedIndex initialized")

    def _default_tokenizer(self, text: str) -> List[str]:
        """
        Default tokenization: lowercase and split on non-alphanumeric.

        Args:
            text: Input text

        Returns:
            List of tokens

        Examples:
            >>> tokenizer("Hello, World!")
            ['hello', 'world']
        """
        # Lowercase and split on non-alphanumeric characters
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens

    def build(self, documents: List[str], metadata: Optional[List[dict]] = None) -> None:
        """
        Build inverted index from a list of documents.

        Args:
            documents: List of document texts
            metadata: Optional list of metadata dicts for each document

        Complexity:
            Time: O(T) where T is total number of tokens across all documents
            Space: O(V + P) where V is vocabulary size, P is total postings

        Examples:
            >>> docs = ["hello world", "world peace", "hello peace"]
            >>> index = InvertedIndex()
            >>> index.build(docs)
            >>> "hello" in index.vocabulary
            True
        """
        self.logger.info(f"Building inverted index for {len(documents)} documents...")

        # Reset index
        self.index = {}
        self.doc_count = len(documents)
        self.doc_lengths = {}
        self.doc_metadata = {}

        # Temporary structure for building: {term: {doc_id: freq}}
        temp_index = defaultdict(lambda: defaultdict(int))

        # Process each document
        for doc_id, text in enumerate(documents):
            # Store metadata if provided
            if metadata and doc_id < len(metadata):
                self.doc_metadata[doc_id] = metadata[doc_id]

            # Tokenize
            tokens = self.tokenizer(text)

            # Store document length
            self.doc_lengths[doc_id] = len(tokens)

            # Count term frequencies
            for term in tokens:
                temp_index[term][doc_id] += 1

        # Convert to final structure: postings lists sorted by doc_id.
        #
        # Sorting is crucial: it makes boolean merges (AND/OR) linear-time
        # and enables binary search for (term, doc_id) lookups.
        for term, doc_freqs in temp_index.items():
            # Sort by doc_id for efficient merging
            postings = sorted(doc_freqs.items(), key=lambda x: x[0])
            self.index[term] = postings

        vocab_size = len(self.index)
        total_postings = sum(len(postings) for postings in self.index.values())

        self.logger.info(
            f"Index built: {self.doc_count} docs, "
            f"{vocab_size} unique terms, "
            f"{total_postings} postings"
        )

    def add_document(self, text: str, metadata: Optional[dict] = None) -> int:
        """
        Add a single document to the existing index.

        Args:
            text: Document text
            metadata: Optional document metadata

        Returns:
            Document ID assigned to this document

        Complexity:
            Time: O(n) where n is number of tokens in document
        """
        doc_id = self.doc_count
        self.doc_count += 1

        # Store metadata
        if metadata:
            self.doc_metadata[doc_id] = metadata

        # Tokenize
        tokens = self.tokenizer(text)
        self.doc_lengths[doc_id] = len(tokens)

        # Count term frequencies
        term_freqs = defaultdict(int)
        for term in tokens:
            term_freqs[term] += 1

        # Update index
        for term, freq in term_freqs.items():
            if term not in self.index:
                self.index[term] = []
            # doc_id is monotonic increasing, so appending preserves the sorted
            # postings invariant (doc_id order) without re-sorting.
            self.index[term].append((doc_id, freq))

        self.logger.debug(f"Added document {doc_id} with {len(tokens)} tokens")
        return doc_id

    def add_document_from_tokens(self, tokens: List[str], metadata: Optional[dict] = None) -> int:
        """
        Add a single document to the index using pre-tokenized tokens.

        This method is optimized for batch processing where documents have already
        been tokenized externally (e.g., via CKIP batch tokenization).

        Args:
            tokens: Pre-tokenized list of terms
            metadata: Optional document metadata

        Returns:
            Document ID assigned to this document

        Complexity:
            Time: O(n) where n is number of tokens
            Space: O(u) where u is unique terms in document

        Example:
            >>> tokens = ["資訊", "檢索", "系統"]
            >>> doc_id = index.add_document_from_tokens(tokens, metadata={'title': 'IR System'})
            >>> print(f"Document indexed with ID: {doc_id}")
        """
        doc_id = self.doc_count
        self.doc_count += 1

        # Store metadata
        if metadata:
            self.doc_metadata[doc_id] = metadata

        # Store document length (no tokenization needed)
        self.doc_lengths[doc_id] = len(tokens)

        # Count term frequencies
        term_freqs = defaultdict(int)
        for term in tokens:
            term_freqs[term] += 1

        # Update index
        for term, freq in term_freqs.items():
            if term not in self.index:
                self.index[term] = []
            # Same monotonic doc_id assumption as add_document().
            self.index[term].append((doc_id, freq))

        self.logger.debug(f"Added pre-tokenized document {doc_id} with {len(tokens)} tokens")
        return doc_id

    def get_postings(self, term: str) -> List[Tuple[int, int]]:
        """
        Get posting list for a term.

        Args:
            term: Query term

        Returns:
            List of (doc_id, term_freq) tuples, sorted by doc_id

        Complexity:
            Time: O(1) average
            Space: O(k) where k is number of documents containing term
        """
        return self.index.get(term, [])

    def get_doc_ids(self, term: str) -> Set[int]:
        """
        Get set of document IDs containing a term.

        Args:
            term: Query term

        Returns:
            Set of document IDs

        Complexity:
            Time: O(k) where k is posting list length
        """
        postings = self.get_postings(term)
        return {doc_id for doc_id, _ in postings}

    def term_frequency(self, term: str, doc_id: int) -> int:
        """
        Get term frequency in a specific document.

        Args:
            term: Query term
            doc_id: Document ID

        Returns:
            Term frequency (0 if term not in document)

        Complexity:
            Time: O(log k) where k is posting list length (binary search)
        """
        postings = self.get_postings(term)
        if not postings:
            return 0

        # Postings are sorted by doc_id, so we can binary search.
        idx = bisect_left(postings, (doc_id, 0))
        if idx < len(postings) and postings[idx][0] == doc_id:
            return postings[idx][1]
        return 0

    def document_frequency(self, term: str) -> int:
        """
        Get document frequency (number of documents containing term).

        Args:
            term: Query term

        Returns:
            Number of documents containing the term

        Complexity:
            Time: O(1)
        """
        return len(self.get_postings(term))

    def intersect(self, postings1: List[Tuple[int, int]],
                  postings2: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Intersect two posting lists (AND operation).

        Uses merge algorithm for efficient intersection of sorted lists.

        Args:
            postings1: First posting list (sorted by doc_id)
            postings2: Second posting list (sorted by doc_id)

        Returns:
            Intersected posting list

        Complexity:
            Time: O(n + m) where n, m are list sizes
            Space: O(min(n, m)) for result

        Examples:
            >>> p1 = [(1, 2), (3, 1), (5, 3)]
            >>> p2 = [(1, 1), (2, 2), (3, 1)]
            >>> index.intersect(p1, p2)
            [(1, 2), (3, 1)]
        """
        result = []
        i, j = 0, 0

        # Merge-based intersection: advance the pointer on the smaller doc_id.
        while i < len(postings1) and j < len(postings2):
            doc1, freq1 = postings1[i]
            doc2, freq2 = postings2[j]

            if doc1 == doc2:
                # Document in both lists
                result.append((doc1, freq1))  # Keep freq from first list
                i += 1
                j += 1
            elif doc1 < doc2:
                i += 1
            else:
                j += 1

        return result

    def union(self, postings1: List[Tuple[int, int]],
              postings2: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Union two posting lists (OR operation).

        Args:
            postings1: First posting list
            postings2: Second posting list

        Returns:
            Union of posting lists

        Complexity:
            Time: O(n + m)
            Space: O(n + m)

        Examples:
            >>> p1 = [(1, 2), (3, 1)]
            >>> p2 = [(2, 1), (3, 2)]
            >>> index.union(p1, p2)
            [(1, 2), (2, 1), (3, 1)]
        """
        result = []
        i, j = 0, 0

        # Merge-based union: emit the smaller doc_id each step.
        while i < len(postings1) and j < len(postings2):
            doc1, freq1 = postings1[i]
            doc2, freq2 = postings2[j]

            if doc1 == doc2:
                result.append((doc1, freq1))  # Keep freq from first
                i += 1
                j += 1
            elif doc1 < doc2:
                result.append((doc1, freq1))
                i += 1
            else:
                result.append((doc2, freq2))
                j += 1

        # Add remaining
        result.extend(postings1[i:])
        result.extend(postings2[j:])

        return result

    def negate(self, postings: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Negate a posting list (NOT operation).

        Returns all documents NOT in the posting list.

        Args:
            postings: Posting list to negate

        Returns:
            Negated posting list

        Complexity:
            Time: O(D + k) where D is total documents, k is posting list size
            Space: O(D - k)
        """
        # NOT is computed against the "universe" of documents [0..doc_count-1].
        # This is convenient for small corpora but can be expensive for large D.
        # Get set of doc IDs in postings
        doc_ids_in_postings = {doc_id for doc_id, _ in postings}

        # All documents not in postings
        result = []
        for doc_id in range(self.doc_count):
            if doc_id not in doc_ids_in_postings:
                result.append((doc_id, 0))  # freq = 0 for negated terms

        return result

    @property
    def vocabulary(self) -> Set[str]:
        """Get the vocabulary (all unique terms)."""
        return set(self.index.keys())

    def get_stats(self) -> dict:
        """
        Get index statistics.

        Returns:
            Dictionary with statistics
        """
        total_postings = sum(len(postings) for postings in self.index.values())
        avg_doc_length = (sum(self.doc_lengths.values()) / self.doc_count
                         if self.doc_count > 0 else 0)

        return {
            'doc_count': self.doc_count,
            'vocabulary_size': len(self.vocabulary),
            'total_postings': total_postings,
            'avg_posting_length': total_postings / len(self.index) if self.index else 0,
            'avg_doc_length': avg_doc_length
        }

    def save(self, filepath: str) -> None:
        """
        Save index to file.

        Args:
            filepath: Output file path (JSON format)
        """
        data = {
            'index': {term: postings for term, postings in self.index.items()},
            'doc_count': self.doc_count,
            'doc_lengths': self.doc_lengths,
            'doc_metadata': self.doc_metadata
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        self.logger.info(f"Index saved to {filepath}")

    def load(self, filepath: str) -> None:
        """
        Load index from file.

        Args:
            filepath: Input file path (JSON format)
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Convert loaded data back to proper types
        self.index = {
            term: [(int(doc_id), int(freq)) for doc_id, freq in postings]
            for term, postings in data['index'].items()
        }
        self.doc_count = data['doc_count']
        self.doc_lengths = {int(k): v for k, v in data['doc_lengths'].items()}
        self.doc_metadata = {int(k): v for k, v in data.get('doc_metadata', {}).items()}

        self.logger.info(f"Index loaded from {filepath}")

    def __repr__(self) -> str:
        """String representation."""
        return (f"InvertedIndex(docs={self.doc_count}, "
                f"vocab={len(self.vocabulary)}, "
                f"postings={sum(len(p) for p in self.index.values())})")


def demo():
    """Demonstration of InvertedIndex functionality."""
    print("=" * 60)
    print("Inverted Index Demo")
    print("=" * 60)

    # Sample documents
    documents = [
        "information retrieval is the process of obtaining information",
        "retrieval models include boolean and vector space models",
        "boolean retrieval uses AND OR NOT operators",
        "vector space model represents documents as vectors",
        "information extraction is related to information retrieval"
    ]

    # Build index
    index = InvertedIndex()
    index.build(documents)

    print(f"\n1. Index Statistics:")
    stats = index.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value:.2f}" if isinstance(value, float) else f"   {key}: {value}")

    # Query examples
    print(f"\n2. Term Lookups:")
    terms = ["information", "retrieval", "boolean", "vector"]
    for term in terms:
        df = index.document_frequency(term)
        doc_ids = index.get_doc_ids(term)
        print(f"   '{term}': DF={df}, Docs={sorted(doc_ids)}")

    # Boolean operations
    print(f"\n3. Boolean Operations:")

    # AND: information AND retrieval
    p1 = index.get_postings("information")
    p2 = index.get_postings("retrieval")
    result_and = index.intersect(p1, p2)
    print(f"   'information' AND 'retrieval': {len(result_and)} docs")
    print(f"      Doc IDs: {[doc_id for doc_id, _ in result_and]}")

    # OR: boolean OR vector
    p3 = index.get_postings("boolean")
    p4 = index.get_postings("vector")
    result_or = index.union(p3, p4)
    print(f"   'boolean' OR 'vector': {len(result_or)} docs")
    print(f"      Doc IDs: {[doc_id for doc_id, _ in result_or]}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    demo()
