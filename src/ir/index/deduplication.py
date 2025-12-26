"""
Document Deduplication Mechanisms

This module implements deduplication algorithms for detecting duplicate
and near-duplicate documents in a large corpus. It supports both exact
matching (MD5) and fuzzy matching (SimHash).

Key Features:
    - MD5 hashing for exact duplicate detection
    - SimHash for near-duplicate detection
    - Configurable similarity threshold
    - Memory-efficient hash storage
    - Fast lookup (O(1) for exact, O(n) for fuzzy)

Author: Information Retrieval System
License: Educational Use
"""

import hashlib
import logging
from typing import Set, Dict, List, Optional, Tuple
from collections import defaultdict


class DuplicationDetector:
    """
    Detector for duplicate and near-duplicate documents.

    Uses MD5 for exact duplicates and SimHash for near-duplicates.

    Complexity:
        - add: O(1) for MD5, O(n*k) for SimHash where n is existing hashes
        - is_duplicate: O(1) for exact, O(n) for fuzzy

    Attributes:
        exact_hashes: Set of MD5 hashes for exact matching
        fuzzy_hashes: Dict of SimHash -> doc_id for fuzzy matching
        fuzzy_threshold: Hamming distance threshold for near-duplicates
    """

    def __init__(self, fuzzy_threshold: int = 3):
        """
        Initialize DuplicationDetector.

        Args:
            fuzzy_threshold: Maximum Hamming distance for near-duplicates
                           (typically 3-5 bits for 64-bit SimHash)
        """
        self.logger = logging.getLogger(__name__)

        # Exact deduplication
        self.exact_hashes: Set[str] = set()

        # Fuzzy deduplication
        self.fuzzy_hashes: Dict[int, str] = {}  # simhash -> doc_id
        self.fuzzy_threshold = fuzzy_threshold

        self.exact_duplicates_found = 0
        self.fuzzy_duplicates_found = 0

        self.logger.info(
            f"DuplicationDetector initialized (threshold={fuzzy_threshold})"
        )

    def md5_hash(self, text: str) -> str:
        """
        Calculate MD5 hash of text.

        Args:
            text: Input text

        Returns:
            MD5 hash string (hexadecimal)

        Complexity:
            Time: O(n) where n is text length

        Examples:
            >>> detector = DuplicationDetector()
            >>> detector.md5_hash("hello world")
            '5eb63bbbe01eeed093cb22bb8f5acdc3'
        """
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def simhash(self, text: str, hash_bits: int = 64) -> int:
        """
        Calculate SimHash of text.

        SimHash is a locality-sensitive hash for finding near-duplicates.
        Documents with similar content will have similar hash values.

        Algorithm:
            1. Tokenize text and calculate hash for each token
            2. For each bit position, add token hash weight if bit=1, subtract if bit=0
            3. Final hash: bit=1 if weighted sum > 0, else bit=0

        Args:
            text: Input text
            hash_bits: Number of bits in hash (default 64)

        Returns:
            SimHash value as integer

        Complexity:
            Time: O(n * k) where n is tokens, k is hash_bits
            Space: O(k) for bit vector

        Examples:
            >>> detector.simhash("hello world")
            12345678901234567890
        """
        # Initialize bit vector
        v = [0] * hash_bits

        # Tokenize (simple whitespace split)
        tokens = text.lower().split()

        if not tokens:
            return 0

        for token in tokens:
            # Hash token to integer
            token_hash = int(hashlib.md5(token.encode('utf-8')).hexdigest(), 16)

            # Update bit vector
            for i in range(hash_bits):
                bit_mask = 1 << i
                if token_hash & bit_mask:
                    v[i] += 1  # Bit is 1
                else:
                    v[i] -= 1  # Bit is 0

        # Generate final hash
        fingerprint = 0
        for i in range(hash_bits):
            if v[i] > 0:
                fingerprint |= (1 << i)

        return fingerprint

    def hamming_distance(self, hash1: int, hash2: int) -> int:
        """
        Calculate Hamming distance between two hashes.

        Hamming distance is the number of differing bits.

        Args:
            hash1: First hash
            hash2: Second hash

        Returns:
            Number of differing bits

        Complexity:
            Time: O(k) where k is number of bits

        Examples:
            >>> detector.hamming_distance(0b1010, 0b1100)
            2  # Two bits differ
        """
        # XOR gives 1 where bits differ
        xor = hash1 ^ hash2

        # Count number of 1s (differing bits)
        distance = 0
        while xor:
            distance += 1
            xor &= xor - 1  # Remove rightmost 1

        return distance

    def add_exact(self, content_hash: str) -> bool:
        """
        Add hash to exact deduplication set.

        Args:
            content_hash: MD5 hash string

        Returns:
            True if added (new), False if duplicate

        Complexity:
            Time: O(1)
        """
        if content_hash in self.exact_hashes:
            self.exact_duplicates_found += 1
            return False

        self.exact_hashes.add(content_hash)
        return True

    def add_fuzzy(self, simhash: int, doc_id: str) -> Optional[str]:
        """
        Add SimHash to fuzzy deduplication index.

        Args:
            simhash: SimHash value
            doc_id: Document identifier

        Returns:
            None if new document, or ID of similar document if near-duplicate

        Complexity:
            Time: O(n) where n is number of existing hashes
        """
        # Check against existing hashes
        for existing_hash, existing_id in self.fuzzy_hashes.items():
            distance = self.hamming_distance(simhash, existing_hash)
            if distance <= self.fuzzy_threshold:
                # Near-duplicate found
                self.fuzzy_duplicates_found += 1
                return existing_id

        # No near-duplicate found, add to index
        self.fuzzy_hashes[simhash] = doc_id
        return None

    def is_exact_duplicate(self, text: str) -> bool:
        """
        Check if text is an exact duplicate.

        Args:
            text: Document text

        Returns:
            True if duplicate exists

        Complexity:
            Time: O(n) for hashing + O(1) for lookup
        """
        content_hash = self.md5_hash(text)
        return content_hash in self.exact_hashes

    def is_fuzzy_duplicate(self, text: str) -> Optional[str]:
        """
        Check if text is a near-duplicate.

        Args:
            text: Document text

        Returns:
            None if unique, or doc_id of similar document if near-duplicate

        Complexity:
            Time: O(n*k + m) where n is tokens, k is hash_bits, m is existing docs
        """
        doc_simhash = self.simhash(text)

        for existing_hash, doc_id in self.fuzzy_hashes.items():
            distance = self.hamming_distance(doc_simhash, existing_hash)
            if distance <= self.fuzzy_threshold:
                return doc_id

        return None

    def is_duplicate(self, text: str,
                    check_exact: bool = True,
                    check_fuzzy: bool = True) -> Tuple[bool, Optional[str]]:
        """
        Check if text is a duplicate (exact or fuzzy).

        Args:
            text: Document text
            check_exact: Whether to check exact duplicates
            check_fuzzy: Whether to check fuzzy duplicates

        Returns:
            (is_duplicate, duplicate_id)
            - is_duplicate: True if any duplicate found
            - duplicate_id: None for exact duplicate, doc_id for fuzzy duplicate

        Complexity:
            Time: O(n*k + m) where n is tokens, k is hash_bits, m is existing docs

        Examples:
            >>> detector = DuplicationDetector()
            >>> detector.add_exact(detector.md5_hash("hello world"))
            >>> is_dup, dup_id = detector.is_duplicate("hello world")
            >>> is_dup
            True
        """
        # Check exact duplicate
        if check_exact and self.is_exact_duplicate(text):
            return True, None

        # Check fuzzy duplicate
        if check_fuzzy:
            similar_id = self.is_fuzzy_duplicate(text)
            if similar_id:
                return True, similar_id

        return False, None

    def add_document(self, text: str, doc_id: str,
                    use_exact: bool = True,
                    use_fuzzy: bool = True) -> Tuple[bool, Optional[str]]:
        """
        Add document and check for duplicates.

        Args:
            text: Document text
            doc_id: Document identifier
            use_exact: Whether to use exact matching
            use_fuzzy: Whether to use fuzzy matching

        Returns:
            (is_unique, duplicate_id)
            - is_unique: True if document is unique (not duplicate)
            - duplicate_id: None if unique or exact dup, doc_id if fuzzy dup

        Examples:
            >>> detector = DuplicationDetector()
            >>> is_unique, dup_id = detector.add_document("hello world", "doc1")
            >>> is_unique
            True
            >>> is_unique, dup_id = detector.add_document("hello world", "doc2")
            >>> is_unique
            False
        """
        # Check exact duplicate
        if use_exact:
            content_hash = self.md5_hash(text)
            if not self.add_exact(content_hash):
                # Exact duplicate
                return False, None

        # Check fuzzy duplicate
        if use_fuzzy:
            doc_simhash = self.simhash(text)
            similar_id = self.add_fuzzy(doc_simhash, doc_id)
            if similar_id:
                # Fuzzy duplicate
                return False, similar_id

        return True, None

    def get_stats(self) -> dict:
        """Get deduplication statistics."""
        return {
            'exact_hashes': len(self.exact_hashes),
            'fuzzy_hashes': len(self.fuzzy_hashes),
            'fuzzy_threshold': self.fuzzy_threshold,
            'exact_duplicates_found': self.exact_duplicates_found,
            'fuzzy_duplicates_found': self.fuzzy_duplicates_found,
            'total_duplicates': self.exact_duplicates_found + self.fuzzy_duplicates_found
        }

    def clear(self):
        """Clear all hashes."""
        self.exact_hashes.clear()
        self.fuzzy_hashes.clear()
        self.exact_duplicates_found = 0
        self.fuzzy_duplicates_found = 0
        self.logger.info("Deduplication index cleared")

    def save(self, filepath: str):
        """
        Save deduplication index to file.

        Args:
            filepath: Output file path
        """
        import pickle

        data = {
            'exact_hashes': self.exact_hashes,
            'fuzzy_hashes': self.fuzzy_hashes,
            'fuzzy_threshold': self.fuzzy_threshold,
            'exact_duplicates_found': self.exact_duplicates_found,
            'fuzzy_duplicates_found': self.fuzzy_duplicates_found
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        self.logger.info(f"Deduplication index saved to {filepath}")

    def load(self, filepath: str):
        """
        Load deduplication index from file.

        Args:
            filepath: Input file path
        """
        import pickle

        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.exact_hashes = data['exact_hashes']
        self.fuzzy_hashes = data['fuzzy_hashes']
        self.fuzzy_threshold = data['fuzzy_threshold']
        self.exact_duplicates_found = data['exact_duplicates_found']
        self.fuzzy_duplicates_found = data['fuzzy_duplicates_found']

        self.logger.info(f"Deduplication index loaded from {filepath}")


def demo():
    """Demonstration of deduplication."""
    print("=" * 70)
    print("Document Deduplication Demo")
    print("=" * 70)

    detector = DuplicationDetector(fuzzy_threshold=3)

    # Test documents
    docs = [
        ("doc1", "台灣選舉開始投票 民眾踴躍參與"),
        ("doc2", "台灣選舉開始投票 民眾踴躍參與"),  # Exact duplicate
        ("doc3", "台灣選舉開始投票 民眾熱情參與"),  # Near-duplicate (similar)
        ("doc4", "美國總統大選即將舉行"),         # Different content
        ("doc5", "美國總統大選即將舉行投票"),     # Near-duplicate to doc4
    ]

    print("\n1. Adding Documents:")
    print("-" * 70)

    for doc_id, text in docs:
        is_unique, dup_id = detector.add_document(text, doc_id)

        status = "✓ Added" if is_unique else "✗ Duplicate"
        dup_info = f" (similar to {dup_id})" if dup_id else ""

        print(f"   {doc_id}: {status}{dup_info}")
        print(f"      Text: {text}")

    print("\n2. Deduplication Statistics:")
    print("-" * 70)
    stats = detector.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    print("\n3. SimHash Examples:")
    print("-" * 70)
    text1 = "台灣選舉開始投票"
    text2 = "台灣選舉開始投票"
    text3 = "台灣選舉即將投票"

    hash1 = detector.simhash(text1)
    hash2 = detector.simhash(text2)
    hash3 = detector.simhash(text3)

    print(f"   Text 1: {text1}")
    print(f"      SimHash: {bin(hash1)[:20]}...")
    print(f"   Text 2: {text2}")
    print(f"      SimHash: {bin(hash2)[:20]}...")
    print(f"      Hamming distance: {detector.hamming_distance(hash1, hash2)}")
    print(f"   Text 3: {text3}")
    print(f"      SimHash: {bin(hash3)[:20]}...")
    print(f"      Hamming distance to Text 1: {detector.hamming_distance(hash1, hash3)}")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    demo()
