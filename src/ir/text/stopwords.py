"""
Stopwords Filter for Chinese Text

This module provides stopwords filtering functionality for Chinese text processing,
with support for Traditional Chinese (繁體中文) stopwords.

Key Features:
    - Load stopwords from file
    - Filter tokens efficiently with set-based lookup
    - Support custom stopwords extension
    - Statistics and analysis tools

Complexity:
    - is_stopword(): O(1) with set-based lookup
    - filter(): O(n) where n = number of tokens
    - Space: O(s) where s = stopwords set size

Reference:
    stopwords-iso: https://github.com/stopwords-iso/stopwords-zh
    Traditional-Chinese-Stopwords: https://github.com/bryanchw/Traditional-Chinese-Stopwords-Library

Author: Information Retrieval System
License: Educational Use
"""

from typing import List, Set, Optional
import logging
from pathlib import Path


class StopwordsFilter:
    """
    Stopwords filter for Chinese text.

    Provides efficient stopword filtering with O(1) lookup time.

    Attributes:
        stopwords: Set of stopwords
        case_sensitive: Whether to use case-sensitive matching

    Examples:
        >>> filter = StopwordsFilter()
        >>> tokens = ['這', '是', '一個', '測試', '的', '句子']
        >>> filtered = filter.filter(tokens)
        >>> print(filtered)
        ['測試', '句子']

        >>> filter.is_stopword('的')
        True
        >>> filter.is_stopword('測試')
        False
    """

    DEFAULT_STOPWORDS_FILE = Path(__file__).parent.parent.parent.parent / \
                            'datasets' / 'stopwords' / 'zh_traditional.txt'

    def __init__(self,
                 stopwords_file: Optional[str] = None,
                 case_sensitive: bool = False,
                 additional_stopwords: Optional[List[str]] = None):
        """
        Initialize stopwords filter.

        Args:
            stopwords_file: Path to stopwords file (one word per line)
                          If None, uses default Traditional Chinese stopwords
            case_sensitive: Whether to perform case-sensitive filtering
            additional_stopwords: Extra stopwords to add

        Raises:
            FileNotFoundError: If stopwords file does not exist

        Complexity:
            Time: O(s) where s = number of stopwords
            Space: O(s)
        """
        self.logger = logging.getLogger(__name__)
        self.case_sensitive = case_sensitive
        self.stopwords: Set[str] = set()

        # Load stopwords from file
        if stopwords_file:
            self._load_from_file(stopwords_file)
        elif self.DEFAULT_STOPWORDS_FILE.exists():
            self._load_from_file(str(self.DEFAULT_STOPWORDS_FILE))
            self.logger.info(f"Loaded default stopwords: {self.DEFAULT_STOPWORDS_FILE}")
        else:
            self.logger.warning(
                f"Default stopwords file not found: {self.DEFAULT_STOPWORDS_FILE}. "
                "Filter will be empty."
            )

        # Add custom stopwords
        if additional_stopwords:
            self.add_stopwords(additional_stopwords)

        self.logger.info(
            f"StopwordsFilter initialized: {len(self.stopwords)} stopwords, "
            f"case_sensitive={case_sensitive}"
        )

    def _load_from_file(self, file_path: str):
        """
        Load stopwords from file.

        File format:
            - One word per line
            - Lines starting with # are comments
            - Empty lines are ignored

        Args:
            file_path: Path to stopwords file

        Raises:
            FileNotFoundError: If file does not exist
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Stopwords file not found: {file_path}")

        count = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()

                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue

                # Normalize case if not case-sensitive
                if not self.case_sensitive:
                    line = line.lower()

                self.stopwords.add(line)
                count += 1

        self.logger.info(f"Loaded {count} stopwords from {file_path}")

    # ========================================================================
    # Core Filtering Methods
    # ========================================================================

    def is_stopword(self, token: str) -> bool:
        """
        Check if a token is a stopword.

        Args:
            token: Token to check

        Returns:
            True if token is a stopword, False otherwise

        Complexity:
            Time: O(1) with set-based lookup
            Space: O(1)

        Examples:
            >>> filter = StopwordsFilter()
            >>> filter.is_stopword('的')
            True
            >>> filter.is_stopword('機器')
            False
        """
        if not self.case_sensitive:
            token = token.lower()

        return token in self.stopwords

    def filter(self, tokens: List[str]) -> List[str]:
        """
        Filter stopwords from token list.

        Args:
            tokens: List of tokens

        Returns:
            Filtered token list (stopwords removed)

        Complexity:
            Time: O(n) where n = number of tokens
            Space: O(k) where k = number of non-stopwords

        Examples:
            >>> filter = StopwordsFilter()
            >>> tokens = ['機器', '學習', '是', '一個', '重要', '的', '領域']
            >>> filtered = filter.filter(tokens)
            >>> print(filtered)
            ['機器', '學習', '重要', '領域']
        """
        if not self.case_sensitive:
            return [token for token in tokens
                   if token.lower() not in self.stopwords]
        else:
            return [token for token in tokens
                   if token not in self.stopwords]

    def filter_with_positions(self, tokens: List[str]) -> List[tuple[str, int]]:
        """
        Filter stopwords and preserve original positions.

        Args:
            tokens: List of tokens

        Returns:
            List of (token, original_position) tuples

        Complexity:
            Time: O(n) where n = number of tokens
            Space: O(k) where k = number of non-stopwords

        Examples:
            >>> filter = StopwordsFilter()
            >>> tokens = ['機器', '學習', '是', '重要', '的']
            >>> filtered = filter.filter_with_positions(tokens)
            >>> print(filtered)
            [('機器', 0), ('學習', 1), ('重要', 3)]
        """
        result = []
        for i, token in enumerate(tokens):
            if not self.is_stopword(token):
                result.append((token, i))
        return result

    # ========================================================================
    # Stopwords Management
    # ========================================================================

    def add_stopword(self, word: str):
        """
        Add a single stopword.

        Args:
            word: Stopword to add

        Examples:
            >>> filter = StopwordsFilter()
            >>> filter.add_stopword('自訂停用詞')
            >>> filter.is_stopword('自訂停用詞')
            True
        """
        if not self.case_sensitive:
            word = word.lower()

        if word not in self.stopwords:
            self.stopwords.add(word)
            self.logger.debug(f"Added stopword: {word}")

    def add_stopwords(self, words: List[str]):
        """
        Add multiple stopwords.

        Args:
            words: List of stopwords to add

        Examples:
            >>> filter = StopwordsFilter()
            >>> filter.add_stopwords(['詞一', '詞二', '詞三'])
            >>> len(filter.stopwords)  # Will include these + defaults
        """
        for word in words:
            self.add_stopword(word)

        self.logger.info(f"Added {len(words)} custom stopwords")

    def remove_stopword(self, word: str):
        """
        Remove a stopword from the set.

        Useful for domain-specific filtering where common stopwords
        are actually meaningful.

        Args:
            word: Stopword to remove

        Examples:
            >>> filter = StopwordsFilter()
            >>> filter.remove_stopword('是')
            >>> filter.is_stopword('是')
            False
        """
        if not self.case_sensitive:
            word = word.lower()

        if word in self.stopwords:
            self.stopwords.remove(word)
            self.logger.debug(f"Removed stopword: {word}")

    def clear(self):
        """Clear all stopwords."""
        self.stopwords.clear()
        self.logger.info("Cleared all stopwords")

    # ========================================================================
    # Statistics and Analysis
    # ========================================================================

    def count_stopwords(self, tokens: List[str]) -> int:
        """
        Count number of stopwords in token list.

        Args:
            tokens: List of tokens

        Returns:
            Number of stopwords

        Complexity:
            Time: O(n) where n = number of tokens
        """
        return sum(1 for token in tokens if self.is_stopword(token))

    def stopword_ratio(self, tokens: List[str]) -> float:
        """
        Calculate ratio of stopwords in token list.

        Args:
            tokens: List of tokens

        Returns:
            Ratio of stopwords (0.0 to 1.0)

        Examples:
            >>> filter = StopwordsFilter()
            >>> tokens = ['這', '是', '測試', '的', '句子']
            >>> ratio = filter.stopword_ratio(tokens)
            >>> print(f"{ratio:.2%}")  # e.g., "60.00%"
        """
        if not tokens:
            return 0.0

        stopword_count = self.count_stopwords(tokens)
        return stopword_count / len(tokens)

    def get_stats(self, tokens: List[str]) -> dict:
        """
        Get comprehensive statistics about stopwords in token list.

        Args:
            tokens: List of tokens

        Returns:
            Dictionary with statistics

        Examples:
            >>> filter = StopwordsFilter()
            >>> tokens = ['這', '是', '一個', '測試', '的', '句子']
            >>> stats = filter.get_stats(tokens)
            >>> print(stats)
            {
                'total_tokens': 6,
                'stopword_count': 4,
                'non_stopword_count': 2,
                'stopword_ratio': 0.667,
                'filtered_tokens': ['測試', '句子']
            }
        """
        total = len(tokens)
        stopword_count = self.count_stopwords(tokens)
        non_stopword_count = total - stopword_count
        filtered = self.filter(tokens)

        return {
            'total_tokens': total,
            'stopword_count': stopword_count,
            'non_stopword_count': non_stopword_count,
            'stopword_ratio': stopword_count / total if total > 0 else 0.0,
            'filtered_tokens': filtered
        }

    @property
    def size(self) -> int:
        """Get number of stopwords in the set."""
        return len(self.stopwords)

    def __len__(self) -> int:
        """Get number of stopwords (for len(filter))."""
        return len(self.stopwords)

    def __contains__(self, word: str) -> bool:
        """Check if word is a stopword (for 'word in filter')."""
        return self.is_stopword(word)


def demo():
    """Demonstration of StopwordsFilter functionality."""
    print("=" * 70)
    print("Stopwords Filter Demo (Traditional Chinese)")
    print("=" * 70)

    # Initialize filter
    stopwords_filter = StopwordsFilter()
    print(f"\n[1] Loaded {len(stopwords_filter)} stopwords")
    print(f"    Sample stopwords: {list(stopwords_filter.stopwords)[:10]}")

    # Example 1: Basic filtering
    print("\n[2] Basic Filtering:")
    print("-" * 70)
    tokens = ['這', '是', '一個', '機器', '學習', '的', '測試', '案例']
    print(f"Original:  {' | '.join(tokens)}")

    filtered = stopwords_filter.filter(tokens)
    print(f"Filtered:  {' | '.join(filtered)}")

    # Example 2: Stopword detection
    print("\n[3] Stopword Detection:")
    print("-" * 70)
    test_words = ['的', '是', '機器', '學習', '在', '一個']
    for word in test_words:
        is_stop = stopwords_filter.is_stopword(word)
        print(f"  '{word}' → {'Stopword' if is_stop else 'Content word'}")

    # Example 3: Statistics
    print("\n[4] Statistics:")
    print("-" * 70)
    sample_text_tokens = [
        '機器', '學習', '是', '人工', '智慧', '的', '重要', '分支',
        '它', '可以', '讓', '電腦', '從', '資料', '中', '學習', '模式'
    ]
    stats = stopwords_filter.get_stats(sample_text_tokens)
    print(f"  Total tokens:        {stats['total_tokens']}")
    print(f"  Stopwords:           {stats['stopword_count']} ({stats['stopword_ratio']:.1%})")
    print(f"  Content words:       {stats['non_stopword_count']}")
    print(f"  After filtering:     {' | '.join(stats['filtered_tokens'])}")

    # Example 4: Custom stopwords
    print("\n[5] Adding Custom Stopwords:")
    print("-" * 70)
    stopwords_filter.add_stopwords(['機器', '電腦'])
    print("  Added custom stopwords: ['機器', '電腦']")

    filtered_custom = stopwords_filter.filter(sample_text_tokens)
    print(f"  Result: {' | '.join(filtered_custom)}")

    # Example 5: Position preservation
    print("\n[6] Position Preservation:")
    print("-" * 70)
    tokens = ['資訊', '檢索', '是', '一個', '重要', '的', '研究', '領域']
    with_positions = stopwords_filter.filter_with_positions(tokens)
    print(f"  Original: {tokens}")
    print(f"  Filtered with positions:")
    for token, pos in with_positions:
        print(f"    {token:10s} (position {pos})")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    demo()
