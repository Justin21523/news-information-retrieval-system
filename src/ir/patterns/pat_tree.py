"""
PAT-tree (Patricia Tree) for Pattern Mining and Term Extraction

This module implements a PAT-tree data structure optimized for Chinese text
pattern mining and statistical term extraction using Mutual Information (MI).

Key Features:
    - Suffix tree construction from token sequences
    - Frequent pattern mining with minimum support
    - Mutual Information (MI) score calculation
    - Multi-word term extraction
    - Pattern frequency statistics

Algorithm Overview:
    1. Tokenize text into word/character sequences
    2. Build suffix tree from all suffixes
    3. Extract patterns meeting minimum support threshold
    4. Compute MI scores for pattern significance
    5. Rank and return top-k terms

Complexity:
    - Construction: O(n²) where n = text length (naive implementation)
    - Pattern search: O(m + k) where m = pattern length, k = occurrences
    - MI calculation: O(p) where p = number of patterns

References:
    Morrison, D. R. (1968). "PATRICIA - Practical Algorithm to Retrieve
        Information Coded in Alphanumeric"
    Church, K. W., & Hanks, P. (1990). "Word Association Norms, Mutual
        Information, and Lexicography"
    Kit, C., & Wilks, Y. (1999). "Unsupervised learning of word boundary
        with description length gain"

Author: Information Retrieval System
License: Educational Use
"""

from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict, Counter
from dataclasses import dataclass
import logging
import math


@dataclass
class Pattern:
    """
    Extracted pattern with statistics.

    Attributes:
        tokens: List of tokens in pattern
        frequency: Number of occurrences
        mi_score: Mutual Information score
        positions: List of starting positions
    """
    tokens: Tuple[str, ...]
    frequency: int
    mi_score: float
    positions: List[int]

    @property
    def text(self) -> str:
        """Get pattern as text string."""
        return ''.join(self.tokens)  # No spaces for Chinese

    def __repr__(self):
        return f"Pattern('{self.text}', freq={self.frequency}, MI={self.mi_score:.3f})"


class PATNode:
    """
    PAT-tree node.

    Attributes:
        children: Dictionary of child nodes (token -> node)
        frequency: Number of times this path appears
        is_end: Whether this node marks end of a pattern
        positions: List of starting positions in original text
    """

    def __init__(self):
        self.children: Dict[str, 'PATNode'] = {}
        self.frequency: int = 0
        self.is_end: bool = False
        self.positions: List[int] = []

    def __repr__(self):
        return f"PATNode(freq={self.frequency}, children={len(self.children)})"


class PATTree:
    """
    PAT-tree for pattern mining and term extraction.

    Implements a suffix tree structure for efficient pattern mining
    and statistical term extraction from Chinese text.

    Attributes:
        root: Root node of the tree
        min_pattern_length: Minimum pattern length (in tokens)
        max_pattern_length: Maximum pattern length (in tokens)
        min_frequency: Minimum frequency threshold
        total_tokens: Total number of tokens in corpus
        token_freq: Token frequency counter
        logger: Logger instance

    Examples:
        >>> tree = PATTree(min_pattern_length=2, min_frequency=2)
        >>> tokens = ['機器', '學習', '是', '人工', '智慧', '的', '分支']
        >>> tree.insert_sequence(tokens)
        >>> patterns = tree.extract_patterns(top_k=5)
        >>> for p in patterns:
        ...     print(f"{p.text}: freq={p.frequency}, MI={p.mi_score:.3f}")
    """

    def __init__(self,
                 min_pattern_length: int = 2,
                 max_pattern_length: int = 5,
                 min_frequency: int = 2):
        """
        Initialize PAT-tree.

        Args:
            min_pattern_length: Minimum pattern length in tokens (>=2)
            max_pattern_length: Maximum pattern length in tokens (<=10 recommended)
            min_frequency: Minimum frequency to consider a pattern
        """
        self.root = PATNode()
        self.min_pattern_length = max(2, min_pattern_length)
        self.max_pattern_length = max_pattern_length
        self.min_frequency = min_frequency
        self.total_tokens = 0
        self.token_freq = Counter()
        self.logger = logging.getLogger(__name__)

        self.logger.info(
            f"PATTree initialized: min_len={min_pattern_length}, "
            f"max_len={max_pattern_length}, min_freq={min_frequency}"
        )

    # ========================================================================
    # Tree Construction
    # ========================================================================

    def insert_sequence(self, tokens: List[str]) -> None:
        """
        Insert a token sequence into the PAT-tree.

        Creates suffix tree by inserting all suffixes of the sequence.

        Args:
            tokens: List of tokens (words or characters)

        Complexity:
            Time: O(n²) where n = len(tokens)
            Space: O(n²) worst case
        """
        n = len(tokens)
        self.total_tokens += n

        # Update token frequencies
        self.token_freq.update(tokens)

        # Insert all suffixes
        for start_pos in range(n):
            self._insert_suffix(tokens, start_pos)

        self.logger.debug(f"Inserted sequence of {n} tokens")

    def _insert_suffix(self, tokens: List[str], start_pos: int) -> None:
        """
        Insert a single suffix starting at start_pos.

        Args:
            tokens: Full token sequence
            start_pos: Starting position of this suffix
        """
        node = self.root
        suffix_len = len(tokens) - start_pos

        # Insert up to max_pattern_length tokens
        for i in range(min(suffix_len, self.max_pattern_length)):
            token = tokens[start_pos + i]

            # Create child node if doesn't exist
            if token not in node.children:
                node.children[token] = PATNode()

            node = node.children[token]
            node.frequency += 1
            node.positions.append(start_pos)

            # Mark as potential pattern end if length is sufficient
            if i + 1 >= self.min_pattern_length:
                node.is_end = True

    def insert_text(self, text: str, tokenizer) -> None:
        """
        Insert text after tokenization.

        Args:
            text: Input text
            tokenizer: ChineseTokenizer instance

        Complexity:
            Time: O(n²) where n = text length after tokenization
        """
        tokens = tokenizer.tokenize(text)
        self.insert_sequence(tokens)

    # ========================================================================
    # Pattern Extraction
    # ========================================================================

    def extract_patterns(self,
                        top_k: Optional[int] = None,
                        use_mi_score: bool = True) -> List[Pattern]:
        """
        Extract patterns from PAT-tree.

        Args:
            top_k: Number of top patterns to return (None = all)
            use_mi_score: Whether to compute and use MI scores for ranking

        Returns:
            List of Pattern objects sorted by score (MI or frequency)

        Complexity:
            Time: O(p log p) where p = number of patterns
            Space: O(p)
        """
        patterns = []

        # Traverse tree and collect patterns
        self._collect_patterns(self.root, [], patterns)

        # Filter by frequency
        patterns = [p for p in patterns if p.frequency >= self.min_frequency]

        if not patterns:
            self.logger.warning("No patterns found meeting minimum frequency")
            return []

        # Compute MI scores if requested
        if use_mi_score:
            for pattern in patterns:
                pattern.mi_score = self._calculate_mi(pattern)

            # Sort by MI score (descending)
            patterns.sort(key=lambda p: p.mi_score, reverse=True)
        else:
            # Sort by frequency (descending)
            patterns.sort(key=lambda p: p.frequency, reverse=True)

        # Return top-k
        result = patterns[:top_k] if top_k else patterns

        self.logger.info(
            f"Extracted {len(result)} patterns from {len(patterns)} candidates"
        )

        return result

    def _collect_patterns(self,
                         node: PATNode,
                         current_path: List[str],
                         patterns: List[Pattern]) -> None:
        """
        Recursively collect all patterns from tree.

        Args:
            node: Current node
            current_path: Current token path
            patterns: List to append patterns to
        """
        # If this node marks end of valid pattern
        if node.is_end and len(current_path) >= self.min_pattern_length:
            pattern = Pattern(
                tokens=tuple(current_path),
                frequency=node.frequency,
                mi_score=0.0,  # Will be calculated later if needed
                positions=node.positions.copy()
            )
            patterns.append(pattern)

        # Recurse to children (up to max length)
        if len(current_path) < self.max_pattern_length:
            for token, child in node.children.items():
                self._collect_patterns(child, current_path + [token], patterns)

    # ========================================================================
    # Mutual Information Calculation
    # ========================================================================

    def _calculate_mi(self, pattern: Pattern) -> float:
        """
        Calculate Mutual Information score for a pattern.

        MI measures how much information the joint occurrence of tokens
        provides compared to their independent occurrences.

        Formula (for bigram):
            MI(x,y) = log2( P(x,y) / (P(x) * P(y)) )
                    = log2( (f_xy * N) / (f_x * f_y) )

        For n-grams:
            MI(w1,...,wn) = log2( P(w1,...,wn) / (P(w1) * ... * P(wn)) )

        Args:
            pattern: Pattern to calculate MI for

        Returns:
            MI score (higher = stronger association)

        Complexity:
            Time: O(n) where n = pattern length
            Space: O(1)
        """
        if self.total_tokens == 0:
            return 0.0

        n = len(pattern.tokens)
        if n < 2:
            return 0.0

        # Joint probability: P(w1,...,wn)
        pattern_freq = pattern.frequency
        p_joint = pattern_freq / self.total_tokens

        # Independent probabilities: P(w1) * ... * P(wn)
        p_independent = 1.0
        for token in pattern.tokens:
            token_freq = self.token_freq[token]
            if token_freq == 0:
                return 0.0  # Shouldn't happen, but handle gracefully
            p_independent *= (token_freq / self.total_tokens)

        # MI score
        if p_independent == 0:
            return 0.0

        mi_score = math.log2(p_joint / p_independent)

        return mi_score

    def calculate_pmi(self, pattern: Pattern) -> float:
        """
        Calculate Pointwise Mutual Information (PMI).

        Alias for _calculate_mi for external use.

        Args:
            pattern: Pattern to calculate PMI for

        Returns:
            PMI score
        """
        return self._calculate_mi(pattern)

    # ========================================================================
    # Search and Query
    # ========================================================================

    def search(self, query_tokens: List[str]) -> Optional[PATNode]:
        """
        Search for a specific token sequence.

        Args:
            query_tokens: Token sequence to search for

        Returns:
            PATNode if found, None otherwise

        Complexity:
            Time: O(m) where m = len(query_tokens)
            Space: O(1)
        """
        node = self.root

        for token in query_tokens:
            if token not in node.children:
                return None
            node = node.children[token]

        return node

    def get_frequency(self, tokens: List[str]) -> int:
        """
        Get frequency of a token sequence.

        Args:
            tokens: Token sequence

        Returns:
            Frequency count (0 if not found)
        """
        node = self.search(tokens)
        return node.frequency if node else 0

    # ========================================================================
    # Statistics
    # ========================================================================

    def get_statistics(self) -> Dict:
        """
        Get tree statistics.

        Returns:
            Dictionary with statistics
        """
        def count_nodes(node):
            count = 1  # Current node
            for child in node.children.values():
                count += count_nodes(child)
            return count

        total_nodes = count_nodes(self.root)
        unique_tokens = len(self.token_freq)

        return {
            'total_tokens': self.total_tokens,
            'unique_tokens': unique_tokens,
            'total_nodes': total_nodes,
            'min_pattern_length': self.min_pattern_length,
            'max_pattern_length': self.max_pattern_length,
            'min_frequency': self.min_frequency
        }

    def print_tree(self, node: Optional[PATNode] = None,
                   prefix: str = "", tokens: List[str] = None) -> None:
        """
        Print tree structure (for debugging).

        Args:
            node: Node to start from (None = root)
            prefix: String prefix for indentation
            tokens: Current token path
        """
        if node is None:
            node = self.root
            tokens = []
            print("PAT-Tree Structure:")
            print("=" * 50)

        if tokens:
            pattern_str = ''.join(tokens)
            end_marker = " [END]" if node.is_end else ""
            print(f"{prefix}{pattern_str} (freq={node.frequency}){end_marker}")

        for token, child in node.children.items():
            self.print_tree(
                child,
                prefix + "  ",
                tokens + [token]
            )

    def __repr__(self):
        stats = self.get_statistics()
        return (
            f"PATTree(tokens={stats['total_tokens']}, "
            f"nodes={stats['total_nodes']}, "
            f"unique={stats['unique_tokens']})"
        )


def demo():
    """Demonstration of PAT-tree usage."""
    print("=" * 70)
    print("PAT-tree Pattern Mining Demo")
    print("=" * 70)

    # Sample Chinese text
    texts = [
        "機器學習是人工智慧的重要分支",
        "深度學習是機器學習的子領域",
        "機器學習和深度學習都是人工智慧技術",
        "自然語言處理也是人工智慧的重要應用"
    ]

    # Initialize tree
    print("\n[1] Initialize PAT-tree")
    print("-" * 70)
    tree = PATTree(
        min_pattern_length=2,
        max_pattern_length=4,
        min_frequency=2
    )
    print(tree)

    # Insert texts
    print("\n[2] Insert Token Sequences")
    print("-" * 70)

    # Simple character-based tokenization for demo
    for text in texts:
        tokens = list(text)  # Character-level for demo
        tree.insert_sequence(tokens)
        print(f"Inserted: {text[:30]}...")

    # Show statistics
    print("\n[3] Tree Statistics")
    print("-" * 70)
    stats = tree.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Extract patterns
    print("\n[4] Extract Top Patterns (by MI score)")
    print("-" * 70)
    patterns = tree.extract_patterns(top_k=10, use_mi_score=True)

    for i, pattern in enumerate(patterns, 1):
        print(f"{i:2d}. {pattern.text:15s}  "
              f"freq={pattern.frequency:3d}  MI={pattern.mi_score:6.3f}")

    # Extract by frequency
    print("\n[5] Extract Top Patterns (by frequency)")
    print("-" * 70)
    patterns_freq = tree.extract_patterns(top_k=10, use_mi_score=False)

    for i, pattern in enumerate(patterns_freq, 1):
        print(f"{i:2d}. {pattern.text:15s}  freq={pattern.frequency:3d}")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    demo()
