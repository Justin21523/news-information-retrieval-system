"""
Fuzzy Query Processing

This module implements fuzzy query matching using edit distance algorithms.
Supports finding similar terms within specified edit distance threshold.

Key Features:
    - Levenshtein (edit) distance calculation
    - Fuzzy term matching against vocabulary
    - Configurable distance threshold
    - Integration with Boolean query engine

Query Examples:
    - test~1 → test, tests, text, rest, ...
    - 台灣~2 → 台灣, 台北, 中國, ...
    - information~3 → information, informative, transform, ...

Author: Information Retrieval System
License: Educational Use
"""

import logging
from typing import List, Set, Tuple


class FuzzyMatcher:
    """
    Fuzzy query matcher using Levenshtein distance.

    Finds terms within specified edit distance of query term.
    Uses dynamic programming for efficient distance calculation.

    Attributes:
        logger: Logger instance
        max_distance: Default maximum edit distance
        max_expansions: Maximum number of fuzzy matches to return

    Complexity:
        - Edit distance: O(m * n) where m, n are string lengths
        - Fuzzy matching: O(V * m * n) where V = vocabulary size
    """

    def __init__(self, max_distance: int = 2, max_expansions: int = 50):
        """
        Initialize FuzzyMatcher.

        Args:
            max_distance: Default maximum edit distance (default: 2)
            max_expansions: Maximum number of fuzzy matches (default: 50)

        Complexity:
            Time: O(1)
        """
        self.logger = logging.getLogger(__name__)
        self.max_distance = max_distance
        self.max_expansions = max_expansions

        self.logger.info(
            f"FuzzyMatcher initialized "
            f"(max_distance={max_distance}, max_expansions={max_expansions})"
        )

    def edit_distance(self, str1: str, str2: str) -> int:
        """
        Calculate Levenshtein (edit) distance between two strings.

        Edit operations: insertion, deletion, substitution.

        Args:
            str1: First string
            str2: Second string

        Returns:
            Minimum number of edits to transform str1 to str2

        Complexity:
            Time: O(m * n) where m, n are string lengths
            Space: O(m * n) for DP table

        Examples:
            >>> edit_distance("test", "text")
            1  # substitute 's' with 'x'

            >>> edit_distance("test", "tests")
            1  # insert 's'

            >>> edit_distance("information", "transform")
            7
        """
        m, n = len(str1), len(str2)

        # Create DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i  # Delete all chars from str1
        for j in range(n + 1):
            dp[0][j] = j  # Insert all chars from str2

        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if str1[i-1] == str2[j-1]:
                    # Characters match - no edit needed
                    dp[i][j] = dp[i-1][j-1]
                else:
                    # Take minimum of: insert, delete, substitute
                    dp[i][j] = 1 + min(
                        dp[i][j-1],    # Insert
                        dp[i-1][j],    # Delete
                        dp[i-1][j-1]   # Substitute
                    )

        return dp[m][n]

    def fuzzy_match(self, query: str, vocabulary: Set[str],
                    max_distance: int = None) -> List[Tuple[str, int]]:
        """
        Find terms within edit distance of query term.

        Args:
            query: Query term
            vocabulary: Set of terms to match against
            max_distance: Maximum edit distance (uses default if None)

        Returns:
            List of (term, distance) tuples, sorted by distance

        Complexity:
            Time: O(V * m * n) where V = vocabulary size
            Space: O(k) where k = matching terms

        Examples:
            >>> vocabulary = {"test", "text", "tests", "rest", "best"}
            >>> fuzzy_match("test", vocabulary, max_distance=1)
            [('test', 0), ('tests', 1), ('text', 1), ('rest', 1), ('best', 1)]
        """
        if max_distance is None:
            max_distance = self.max_distance

        query_lower = query.lower()
        matches = []

        # Calculate distance for each term
        for term in vocabulary:
            distance = self.edit_distance(query_lower, term.lower())

            if distance <= max_distance:
                matches.append((term, distance))

                # Limit expansions
                if len(matches) >= self.max_expansions:
                    self.logger.warning(
                        f"Fuzzy expansion limit reached ({self.max_expansions}). "
                        f"Query: '{query}' with max_distance={max_distance}"
                    )
                    break

        # Sort by distance (closer matches first), then alphabetically
        matches.sort(key=lambda x: (x[1], x[0]))

        self.logger.debug(
            f"Fuzzy matched '{query}' to {len(matches)} terms "
            f"(max_distance={max_distance})"
        )

        return matches

    def expand(self, query: str, vocabulary: Set[str],
               max_distance: int = None) -> List[str]:
        """
        Expand query term to fuzzy matches.

        Args:
            query: Query term
            vocabulary: Set of terms to match against
            max_distance: Maximum edit distance (uses default if None)

        Returns:
            List of matching terms (without distances)

        Complexity:
            Time: O(V * m * n)

        Examples:
            >>> vocabulary = {"test", "text", "tests", "rest"}
            >>> expand("test", vocabulary, max_distance=1)
            ['test', 'rest', 'tests', 'text']
        """
        matches = self.fuzzy_match(query, vocabulary, max_distance)
        return [term for term, distance in matches]

    def get_distance_groups(self, query: str, vocabulary: Set[str],
                           max_distance: int = None) -> dict:
        """
        Group fuzzy matches by edit distance.

        Args:
            query: Query term
            vocabulary: Set of terms
            max_distance: Maximum edit distance

        Returns:
            Dictionary mapping distance -> list of terms

        Examples:
            >>> get_distance_groups("test", vocabulary, max_distance=2)
            {
                0: ['test'],
                1: ['tests', 'text', 'rest', 'best'],
                2: ['tested', 'fastest', ...]
            }
        """
        matches = self.fuzzy_match(query, vocabulary, max_distance)

        groups = {}
        for term, distance in matches:
            if distance not in groups:
                groups[distance] = []
            groups[distance].append(term)

        return groups

    def get_stats(self, query: str, vocabulary: Set[str],
                  max_distance: int = None) -> dict:
        """
        Get fuzzy matching statistics.

        Args:
            query: Query term
            vocabulary: Set of terms
            max_distance: Maximum edit distance

        Returns:
            Dictionary with statistics

        Examples:
            >>> get_stats("test", vocabulary, max_distance=2)
            {'query': 'test', 'max_distance': 2, 'total_matches': 10, ...}
        """
        if max_distance is None:
            max_distance = self.max_distance

        matches = self.fuzzy_match(query, vocabulary, max_distance)
        groups = self.get_distance_groups(query, vocabulary, max_distance)

        stats = {
            'query': query,
            'max_distance': max_distance,
            'total_matches': len(matches),
            'distance_distribution': {d: len(terms) for d, terms in groups.items()},
            'sample_matches': matches[:5],  # First 5
            'exact_match': query.lower() in vocabulary
        }

        return stats


def demo():
    """Demonstration of FuzzyMatcher."""
    print("=" * 60)
    print("Fuzzy Query Matcher Demo")
    print("=" * 60)

    # Sample vocabulary
    vocabulary = {
        'test', 'text', 'tests', 'tested', 'testing', 'tester',
        'rest', 'best', 'west', 'pest', 'fest', 'nest',
        'taste', 'paste', 'waste', 'haste',
        'data', 'date', 'gate', 'rate', 'late',
        'information', 'inform', 'transformation', 'format'
    }

    matcher = FuzzyMatcher(max_distance=2, max_expansions=20)

    print(f"\nVocabulary: {sorted(vocabulary)}")
    print(f"Total terms: {len(vocabulary)}")

    # Test queries
    queries = [
        ("test", 1),
        ("test", 2),
        ("data", 1),
        ("inform", 2),
    ]

    print("\n" + "-" * 60)
    print("Fuzzy Matching Tests")
    print("-" * 60)

    for query, max_dist in queries:
        print(f"\nQuery: '{query}' (max_distance={max_dist})")

        # Get matches
        matches = matcher.fuzzy_match(query, vocabulary, max_distance=max_dist)

        # Get statistics
        stats = matcher.get_stats(query, vocabulary, max_distance=max_dist)

        print(f"  Total matches: {stats['total_matches']}")
        print(f"  Distance distribution: {stats['distance_distribution']}")

        if matches:
            print(f"  Matches:")
            for term, distance in matches[:10]:  # Show first 10
                print(f"    - {term} (distance={distance})")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    demo()
