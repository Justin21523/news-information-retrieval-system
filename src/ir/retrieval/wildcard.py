"""
Wildcard Query Processing

This module implements wildcard query expansion for search queries.
Supports * (multiple chars) and ? (single char) wildcards.

Key Features:
    - Wildcard pattern matching against vocabulary
    - Efficient term expansion using regex
    - Support for prefix, suffix, and infix wildcards
    - Integration with Boolean query engine

Query Examples:
    - info* → information, inform, info, ...
    - *tion → information, education, nation, ...
    - te?t → test, text, tent, ...
    - *form* → information, transform, format, ...

Author: Information Retrieval System
License: Educational Use
"""

import re
import logging
from typing import List, Set, Pattern


class WildcardExpander:
    """
    Wildcard query expander for information retrieval.

    Expands wildcard patterns to matching terms from vocabulary.
    Uses regex pattern matching for efficient expansion.

    Attributes:
        logger: Logger instance
        max_expansions: Maximum number of terms to expand to (prevent explosion)

    Complexity:
        - Pattern matching: O(V) where V = vocabulary size
        - With max_expansions: O(min(V, max_expansions))
    """

    def __init__(self, max_expansions: int = 50):
        """
        Initialize WildcardExpander.

        Args:
            max_expansions: Maximum number of expanded terms (default: 50)

        Complexity:
            Time: O(1)
        """
        self.logger = logging.getLogger(__name__)
        self.max_expansions = max_expansions

        self.logger.info(f"WildcardExpander initialized (max_expansions={max_expansions})")

    def has_wildcard(self, term: str) -> bool:
        """
        Check if term contains wildcard characters.

        Args:
            term: Query term

        Returns:
            True if term contains * or ?

        Complexity:
            Time: O(1)

        Examples:
            >>> has_wildcard("info*")
            True

            >>> has_wildcard("information")
            False
        """
        return '*' in term or '?' in term

    def expand(self, pattern: str, vocabulary: Set[str]) -> List[str]:
        """
        Expand wildcard pattern to matching terms.

        Args:
            pattern: Wildcard pattern (e.g., "info*", "te?t")
            vocabulary: Set of terms to match against

        Returns:
            List of matching terms (up to max_expansions)

        Complexity:
            Time: O(V * P) where V = vocabulary size, P = pattern matching
            Space: O(k) where k = matching terms

        Examples:
            >>> vocabulary = {"information", "inform", "info", "data"}
            >>> expand("info*", vocabulary)
            ['information', 'inform', 'info']

            >>> expand("*form*", vocabulary)
            ['information', 'inform']
        """
        if not self.has_wildcard(pattern):
            # No wildcard - return as-is if in vocabulary
            return [pattern] if pattern in vocabulary else []

        # Convert wildcard pattern to regex
        regex_pattern = self._wildcard_to_regex(pattern)
        compiled_pattern = re.compile(regex_pattern, re.IGNORECASE)

        # Find matching terms
        matching_terms = []
        for term in vocabulary:
            if compiled_pattern.fullmatch(term):
                matching_terms.append(term)

                # Limit expansions
                if len(matching_terms) >= self.max_expansions:
                    self.logger.warning(
                        f"Wildcard expansion limit reached ({self.max_expansions}). "
                        f"Pattern: '{pattern}' matched {len(matching_terms)}+ terms."
                    )
                    break

        self.logger.debug(f"Expanded '{pattern}' to {len(matching_terms)} terms")
        return sorted(matching_terms)

    def _wildcard_to_regex(self, pattern: str) -> str:
        """
        Convert wildcard pattern to regex pattern.

        Args:
            pattern: Wildcard pattern

        Returns:
            Regex pattern string

        Complexity:
            Time: O(n) where n = pattern length

        Examples:
            >>> _wildcard_to_regex("info*")
            'info.*'

            >>> _wildcard_to_regex("te?t")
            'te.t'

            >>> _wildcard_to_regex("*form*")
            '.*form.*'
        """
        # Escape special regex characters except * and ?
        # Characters to escape: . ^ $ + { } [ ] \ | ( )
        escaped = re.escape(pattern)

        # Replace escaped wildcards with regex equivalents
        # re.escape converts * to \* and ? to \?
        regex = escaped.replace(r'\*', '.*')  # * -> match any chars
        regex = regex.replace(r'\?', '.')     # ? -> match single char

        return regex

    def expand_multiple(self, patterns: List[str], vocabulary: Set[str]) -> List[str]:
        """
        Expand multiple wildcard patterns.

        Args:
            patterns: List of wildcard patterns
            vocabulary: Set of terms to match against

        Returns:
            Combined list of unique matching terms

        Complexity:
            Time: O(P * V) where P = number of patterns, V = vocabulary

        Examples:
            >>> vocabulary = {"information", "data", "inform", "database"}
            >>> expand_multiple(["info*", "data*"], vocabulary)
            ['data', 'database', 'inform', 'information']
        """
        all_matches = set()

        for pattern in patterns:
            matches = self.expand(pattern, vocabulary)
            all_matches.update(matches)

        return sorted(all_matches)

    def get_stats(self, pattern: str, vocabulary: Set[str]) -> dict:
        """
        Get expansion statistics without full expansion.

        Args:
            pattern: Wildcard pattern
            vocabulary: Set of terms

        Returns:
            Dictionary with expansion statistics

        Examples:
            >>> get_stats("info*", vocabulary)
            {'pattern': 'info*', 'matches': 3, 'is_prefix': True, ...}
        """
        matches = self.expand(pattern, vocabulary)

        stats = {
            'pattern': pattern,
            'matches': len(matches),
            'is_prefix': pattern.endswith('*') and '*' not in pattern[:-1],
            'is_suffix': pattern.startswith('*') and '*' not in pattern[1:],
            'is_infix': pattern.startswith('*') and pattern.endswith('*'),
            'sample_terms': matches[:5]  # Show first 5
        }

        return stats


def demo():
    """Demonstration of WildcardExpander."""
    print("=" * 60)
    print("Wildcard Query Expander Demo")
    print("=" * 60)

    # Sample vocabulary
    vocabulary = {
        'information', 'inform', 'info', 'informative',
        'data', 'database', 'datasheet', 'metadata',
        'retrieval', 'retrieve', 'retrieved',
        'search', 'research', 'searching',
        'test', 'text', 'context', 'testing'
    }

    expander = WildcardExpander(max_expansions=10)

    print(f"\nVocabulary: {sorted(vocabulary)}")
    print(f"\nTotal terms: {len(vocabulary)}")

    # Test patterns
    patterns = [
        "info*",      # Prefix wildcard
        "*tion",      # Suffix wildcard
        "te?t",       # Single char wildcard
        "*data*",     # Infix wildcard
        "search*",    # Prefix
        "*ing",       # Suffix
    ]

    print("\n" + "-" * 60)
    print("Wildcard Expansion Tests")
    print("-" * 60)

    for pattern in patterns:
        matches = expander.expand(pattern, vocabulary)
        stats = expander.get_stats(pattern, vocabulary)

        print(f"\nPattern: '{pattern}'")
        print(f"  Type: ", end="")
        if stats['is_prefix']:
            print("Prefix")
        elif stats['is_suffix']:
            print("Suffix")
        elif stats['is_infix']:
            print("Infix")
        else:
            print("Mixed")

        print(f"  Matches: {len(matches)} terms")
        if matches:
            print(f"  Results: {matches}")
        else:
            print(f"  Results: (no matches)")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    demo()
