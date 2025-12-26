"""
Positional Index for Information Retrieval

This module implements a positional inverted index that stores not only
which documents contain each term, but also the positions where terms
appear within documents. This enables phrase queries and proximity searches.

Implementation Overview:
    - Indexing:
        1) Tokenize each document into a sequence of terms
        2) For each term occurrence, append its token position to a postings list
    - Phrase queries:
        - Restrict to documents that contain all phrase terms
        - Verify that positions align as a consecutive sequence
    - Proximity queries:
        - Check whether any occurrence positions are within a distance window

Key Features:
    - Position-aware indexing
    - Phrase query support
    - Proximity search capability
    - Window-based matching

Author: Information Retrieval System
License: Educational Use
"""

import re
import logging
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional, Callable
import json


class PositionalIndex:
    """
    Positional Inverted Index for phrase and proximity queries.

    Extends the inverted index by storing term positions within documents,
    enabling phrase queries like "information retrieval" and proximity
    queries like "information NEAR/3 retrieval".

    Data Structure:
        index: {
            term1: {
                doc_id1: [pos1, pos2, ...],
                doc_id2: [pos1, pos2, ...],
                ...
            },
            term2: {...},
            ...
        }

    Complexity:
        - Build: O(T) where T is total tokens
        - Phrase query: O(k * min(p1, p2)) where k is docs with both terms,
                        p1, p2 are position list lengths
        - Proximity query: O(k * p1 * p2)

    Attributes:
        index (dict): Positional index mapping
        doc_count (int): Total documents
        doc_lengths (dict): Document lengths
        vocabulary (set): All unique terms
    """

    def __init__(self, tokenizer: Optional[Callable[[str], List[str]]] = None):
        """
        Initialize PositionalIndex.

        Args:
            tokenizer: Custom tokenization function
        """
        self.logger = logging.getLogger(__name__)

        # Core data structures
        self.index: Dict[str, Dict[int, List[int]]] = {}
        self.doc_count: int = 0
        self.doc_lengths: Dict[int, int] = {}
        self.doc_metadata: Dict[int, dict] = {}

        # Tokenizer
        self.tokenizer = tokenizer or self._default_tokenizer

        self.logger.info("PositionalIndex initialized")

    def _default_tokenizer(self, text: str) -> List[str]:
        """Default tokenization."""
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens

    def build(self, documents: List[str], metadata: Optional[List[dict]] = None) -> None:
        """
        Build positional index from documents.

        Args:
            documents: List of document texts
            metadata: Optional metadata for each document

        Complexity:
            Time: O(T) where T is total tokens
            Space: O(V * D * P) where V=vocab, D=avg docs per term, P=avg positions

        Examples:
            >>> docs = ["hello world", "world peace"]
            >>> index = PositionalIndex()
            >>> index.build(docs)
            >>> index.get_positions("world", 0)
            [1]
        """
        self.logger.info(f"Building positional index for {len(documents)} documents...")

        # Reset
        self.index = {}
        self.doc_count = len(documents)
        self.doc_lengths = {}
        self.doc_metadata = {}

        # Temporary structure
        temp_index = defaultdict(lambda: defaultdict(list))

        # Process each document
        for doc_id, text in enumerate(documents):
            # Store metadata
            if metadata and doc_id < len(metadata):
                self.doc_metadata[doc_id] = metadata[doc_id]

            # Tokenize
            tokens = self.tokenizer(text)
            self.doc_lengths[doc_id] = len(tokens)

            # Record positions
            for position, term in enumerate(tokens):
                temp_index[term][doc_id].append(position)

        # Convert to final structure
        self.index = {term: dict(doc_positions)
                     for term, doc_positions in temp_index.items()}

        vocab_size = len(self.index)
        total_postings = sum(len(docs) for docs in self.index.values())
        total_positions = sum(
            len(positions)
            for docs in self.index.values()
            for positions in docs.values()
        )

        self.logger.info(
            f"Positional index built: {self.doc_count} docs, "
            f"{vocab_size} terms, {total_postings} postings, "
            f"{total_positions} positions"
        )

    def add_document(self, text: str, metadata: Optional[dict] = None) -> int:
        """
        Add a document to the index.

        Args:
            text: Document text
            metadata: Optional metadata

        Returns:
            Assigned document ID
        """
        doc_id = self.doc_count
        self.doc_count += 1

        if metadata:
            self.doc_metadata[doc_id] = metadata

        # Tokenize
        tokens = self.tokenizer(text)
        self.doc_lengths[doc_id] = len(tokens)

        # Record positions
        for position, term in enumerate(tokens):
            if term not in self.index:
                self.index[term] = {}
            if doc_id not in self.index[term]:
                self.index[term][doc_id] = []
            self.index[term][doc_id].append(position)

        return doc_id

    def get_positions(self, term: str, doc_id: int) -> List[int]:
        """
        Get positions of term in a document.

        Args:
            term: Query term
            doc_id: Document ID

        Returns:
            List of positions (empty if term not in doc)

        Complexity:
            Time: O(1) average
        """
        if term in self.index and doc_id in self.index[term]:
            return self.index[term][doc_id]
        return []

    def get_doc_ids(self, term: str) -> Set[int]:
        """
        Get document IDs containing a term.

        Args:
            term: Query term

        Returns:
            Set of document IDs
        """
        if term in self.index:
            return set(self.index[term].keys())
        return set()

    def phrase_query(self, phrase: str) -> List[int]:
        """
        Execute a phrase query.

        Finds documents containing the exact phrase with terms in sequence.

        Args:
            phrase: Phrase to search (e.g., "information retrieval")

        Returns:
            List of document IDs containing the phrase

        Complexity:
            Time: O(k * min(p1, p2, ...)) where k is candidate docs,
                  p1, p2, ... are position list lengths for each term
            Space: O(k)

        Examples:
            >>> index.phrase_query("information retrieval")
            [0, 4]
        """
        # Tokenize phrase
        terms = self.tokenizer(phrase)

        if not terms:
            return []

        if len(terms) == 1:
            # Single term - just return docs containing it
            return sorted(self.get_doc_ids(terms[0]))

        # Candidate generation: only consider documents that contain *all* terms.
        # This is a standard optimization to avoid scanning every document.
        candidate_docs = self.get_doc_ids(terms[0])
        for term in terms[1:]:
            candidate_docs &= self.get_doc_ids(term)

        if not candidate_docs:
            return []

        # Check for phrase in each candidate document
        result = []
        for doc_id in candidate_docs:
            if self._has_phrase(doc_id, terms):
                result.append(doc_id)

        return sorted(result)

    def phrase_search(self, phrase: str) -> List[int]:
        """
        Alias for phrase_query() for backward compatibility.

        Args:
            phrase: Phrase to search

        Returns:
            List of document IDs containing the phrase
        """
        return self.phrase_query(phrase)

    def _has_phrase(self, doc_id: int, terms: List[str]) -> bool:
        """
        Check if document contains terms in sequence.

        Args:
            doc_id: Document ID
            terms: List of terms in order

        Returns:
            True if phrase found
        """
        # Naive phrase verification:
        # For each position where the first term occurs, check whether the next
        # terms occur at +1, +2, ... offsets. This is correct but can be slow
        # when the first term appears many times.
        #
        # More efficient variants merge two position lists at a time using
        # pointer arithmetic, achieving O(p1 + p2) per term pair.
        #
        # Positions are naturally sorted because we append positions in
        # increasing order during indexing.
        # Get positions of first term
        positions = self.get_positions(terms[0], doc_id)

        # For each starting position
        for start_pos in positions:
            # Check if subsequent terms follow
            found = True
            for i, term in enumerate(terms[1:], start=1):
                expected_pos = start_pos + i
                term_positions = self.get_positions(term, doc_id)

                if expected_pos not in term_positions:
                    found = False
                    break

            if found:
                return True

        return False

    def proximity_query(self, term1: str, term2: str, max_distance: int) -> List[int]:
        """
        Execute a proximity query.

        Finds documents where term1 and term2 appear within max_distance
        words of each other.

        Args:
            term1: First term
            term2: Second term
            max_distance: Maximum distance between terms (in tokens)

        Returns:
            List of document IDs

        Complexity:
            Time: O(k * p1 * p2) where k is candidate docs,
                  p1, p2 are position list lengths

        Examples:
            >>> index.proximity_query("information", "retrieval", 3)
            [0, 4]  # "information retrieval" or "information processing retrieval"
        """
        # Get candidate documents
        docs1 = self.get_doc_ids(term1)
        docs2 = self.get_doc_ids(term2)
        candidate_docs = docs1 & docs2

        if not candidate_docs:
            return []

        # Check proximity in each document
        result = []
        for doc_id in candidate_docs:
            positions1 = self.get_positions(term1, doc_id)
            positions2 = self.get_positions(term2, doc_id)

            # Check if any pair is within distance
            if self._within_distance(positions1, positions2, max_distance):
                result.append(doc_id)

        return sorted(result)

    def _within_distance(self, positions1: List[int], positions2: List[int],
                        max_distance: int) -> bool:
        """
        Check if any position pair is within distance.

        Args:
            positions1: Positions of first term
            positions2: Positions of second term
            max_distance: Maximum allowed distance

        Returns:
            True if any pair within distance
        """
        # Two-pointer scan over sorted position lists.
        #
        # This is O(p1 + p2) rather than O(p1 * p2) and relies on the invariant
        # that positions are sorted (they are appended in order at build time).
        i = 0
        j = 0
        while i < len(positions1) and j < len(positions2):
            p1 = positions1[i]
            p2 = positions2[j]
            if abs(p1 - p2) <= max_distance:
                return True

            # Move the pointer that points to the smaller position to reduce
            # the absolute distance in subsequent steps.
            if p1 < p2:
                i += 1
            else:
                j += 1

        return False

    def window_query(self, terms: List[str], window_size: int) -> List[int]:
        """
        Find documents where all terms appear within a window.

        Args:
            terms: List of terms to find
            window_size: Window size (in tokens)

        Returns:
            List of document IDs

        Examples:
            >>> index.window_query(["information", "retrieval", "system"], 5)
            [0]
        """
        if not terms:
            return []

        # Get candidate documents (all terms present)
        candidate_docs = self.get_doc_ids(terms[0])
        for term in terms[1:]:
            candidate_docs &= self.get_doc_ids(term)

        if not candidate_docs:
            return []

        # Check window in each document
        result = []
        for doc_id in candidate_docs:
            if self._has_window(doc_id, terms, window_size):
                result.append(doc_id)

        return sorted(result)

    def _has_window(self, doc_id: int, terms: List[str], window_size: int) -> bool:
        """
        Check if document has all terms within a window.

        Args:
            doc_id: Document ID
            terms: Terms to find
            window_size: Window size

        Returns:
            True if window found
        """
        # Get all positions for all terms
        all_positions = []
        for term in terms:
            positions = self.get_positions(term, doc_id)
            all_positions.extend([(pos, term) for pos in positions])

        # Sort by position
        all_positions.sort()

        # Sliding window to check if all terms appear
        from collections import Counter

        for i in range(len(all_positions)):
            window_start = all_positions[i][0]
            window_terms = set()

            # Collect terms in window
            for j in range(i, len(all_positions)):
                pos, term = all_positions[j]
                if pos - window_start < window_size:
                    window_terms.add(term)
                else:
                    break

            # Check if all terms in window
            if len(window_terms) == len(terms):
                return True

        return False

    @property
    def vocabulary(self) -> Set[str]:
        """Get vocabulary."""
        return set(self.index.keys())

    def get_stats(self) -> dict:
        """Get index statistics."""
        total_docs_with_term = sum(len(docs) for docs in self.index.values())
        total_positions = sum(
            len(positions)
            for docs in self.index.values()
            for positions in docs.values()
        )

        return {
            'doc_count': self.doc_count,
            'vocabulary_size': len(self.vocabulary),
            'total_postings': total_docs_with_term,
            'total_positions': total_positions,
            'avg_positions_per_posting': (
                total_positions / total_docs_with_term if total_docs_with_term > 0 else 0
            )
        }

    def save(self, filepath: str) -> None:
        """Save index to file."""
        data = {
            'index': {
                term: {str(doc_id): positions for doc_id, positions in docs.items()}
                for term, docs in self.index.items()
            },
            'doc_count': self.doc_count,
            'doc_lengths': self.doc_lengths,
            'doc_metadata': self.doc_metadata
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        self.logger.info(f"Positional index saved to {filepath}")

    def load(self, filepath: str) -> None:
        """Load index from file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.index = {
            term: {int(doc_id): positions for doc_id, positions in docs.items()}
            for term, docs in data['index'].items()
        }
        self.doc_count = data['doc_count']
        self.doc_lengths = {int(k): v for k, v in data['doc_lengths'].items()}
        self.doc_metadata = {int(k): v for k, v in data.get('doc_metadata', {}).items()}

        self.logger.info(f"Positional index loaded from {filepath}")

    def __repr__(self) -> str:
        """String representation."""
        return (f"PositionalIndex(docs={self.doc_count}, "
                f"vocab={len(self.vocabulary)})")


def demo():
    """Demonstration of PositionalIndex functionality."""
    print("=" * 60)
    print("Positional Index Demo")
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
    index = PositionalIndex()
    index.build(documents)

    print(f"\n1. Index Statistics:")
    stats = index.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value:.2f}" if isinstance(value, float) else f"   {key}: {value}")

    # Phrase queries
    print(f"\n2. Phrase Queries:")
    phrases = [
        "information retrieval",
        "vector space",
        "boolean retrieval",
        "space model"
    ]
    for phrase in phrases:
        result = index.phrase_query(phrase)
        print(f"   \"{phrase}\": {result}")

    # Proximity queries
    print(f"\n3. Proximity Queries:")
    queries = [
        ("information", "retrieval", 1),
        ("information", "process", 3),
        ("boolean", "vector", 5)
    ]
    for term1, term2, dist in queries:
        result = index.proximity_query(term1, term2, dist)
        print(f"   '{term1}' NEAR/{dist} '{term2}': {result}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    demo()
