"""
Dynamic Summarization with KWIC (KeyWord In Context)

This module implements dynamic summarization techniques that generate
contextual snippets around query keywords at runtime.

Key Features:
    - KWIC (KeyWord In Context) generation
    - Flexible windowing strategies
    - Context caching for performance
    - Highlighting and formatting
    - Multi-match handling

Reference: "Introduction to Information Retrieval" (Manning et al.)
           Chapter 23: Web Search Basics - Snippets and Context

Author: Information Retrieval System
License: Educational Use
"""

import logging
import re
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import time

import sys
from pathlib import Path

_parent_dir = Path(__file__).parent.parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))


@dataclass
class KWICMatch:
    """
    KWIC match representation.

    Attributes:
        keyword: Matched keyword
        position: Position in document (character offset)
        left_context: Text before keyword
        right_context: Text after keyword
        doc_id: Document identifier
    """
    keyword: str
    position: int
    left_context: str
    right_context: str
    doc_id: Optional[int] = None

    @property
    def snippet(self) -> str:
        """Get formatted snippet with keyword highlighted."""
        return f"{self.left_context}**{self.keyword}**{self.right_context}"

    @property
    def plain_snippet(self) -> str:
        """Get plain snippet without highlighting."""
        return f"{self.left_context}{self.keyword}{self.right_context}"


@dataclass
class KWICResult:
    """
    KWIC result container.

    Attributes:
        matches: List of KWIC matches
        query: Original query
        num_documents: Number of documents searched
        cache_hit: Whether result was cached
    """
    matches: List[KWICMatch]
    query: str
    num_documents: int = 1
    cache_hit: bool = False

    @property
    def num_matches(self) -> int:
        """Total number of matches."""
        return len(self.matches)

    def get_snippets(self, max_snippets: Optional[int] = None) -> List[str]:
        """
        Get formatted snippets.

        Args:
            max_snippets: Maximum number of snippets to return

        Returns:
            List of formatted snippets
        """
        matches = self.matches[:max_snippets] if max_snippets else self.matches
        return [match.snippet for match in matches]


class KWICGenerator:
    """
    KWIC (KeyWord In Context) Generator.

    Generates contextual snippets around query keywords with:
    - Fixed-width windows
    - Sentence-boundary aware windows
    - Intelligent context extraction
    - Result caching

    Complexity:
        - Window extraction: O(n × m) where n=document length, m=query terms
        - With caching: O(1) for repeated queries
    """

    def __init__(self,
                 window_size: int = 50,
                 window_type: str = 'fixed',
                 case_sensitive: bool = False,
                 enable_cache: bool = True,
                 max_cache_size: int = 1000):
        """
        Initialize KWIC generator.

        Args:
            window_size: Context window size (characters or words)
            window_type: Window type ('fixed', 'sentence', 'adaptive')
            case_sensitive: Case-sensitive matching
            enable_cache: Enable result caching
            max_cache_size: Maximum cache entries
        """
        self.logger = logging.getLogger(__name__)
        self.window_size = window_size
        self.window_type = window_type
        self.case_sensitive = case_sensitive
        self.enable_cache = enable_cache
        self.max_cache_size = max_cache_size

        # Cache: {(query, doc_hash): (result, timestamp)}
        self._cache: Dict[Tuple[str, int], Tuple[KWICResult, float]] = {}

        self.logger.info(
            f"KWICGenerator initialized: window={window_size}, "
            f"type={window_type}, cache={enable_cache}"
        )

    # ========================================================================
    # KWIC Generation
    # ========================================================================

    def generate(self, text: str, query: str,
                max_matches: Optional[int] = None,
                doc_id: Optional[int] = None) -> KWICResult:
        """
        Generate KWIC snippets for query in text.

        Args:
            text: Document text
            query: Query string (space-separated keywords)
            max_matches: Maximum matches to return
            doc_id: Document identifier

        Returns:
            KWICResult with matches

        Complexity:
            Time: O(n × m) where n=text length, m=query terms
            Space: O(k × w) where k=matches, w=window size

        Examples:
            >>> generator = KWICGenerator(window_size=20)
            >>> text = "Machine learning is a subset of artificial intelligence"
            >>> result = generator.generate(text, "machine learning")
            >>> result.num_matches
            2
        """
        # Check cache
        if self.enable_cache:
            cache_key = (query, hash(text))
            if cache_key in self._cache:
                cached_result, _ = self._cache[cache_key]
                self.logger.debug(f"Cache hit for query: {query}")
                cached_result.cache_hit = True
                return cached_result

        # Parse query
        keywords = self._parse_query(query)

        if not keywords:
            return KWICResult(matches=[], query=query, num_documents=1)

        # Find all matches
        matches = []
        for keyword in keywords:
            keyword_matches = self._find_keyword_matches(text, keyword, doc_id)
            matches.extend(keyword_matches)

        # Sort by position
        matches.sort(key=lambda m: m.position)

        # Limit matches
        if max_matches:
            matches = matches[:max_matches]

        result = KWICResult(
            matches=matches,
            query=query,
            num_documents=1,
            cache_hit=False
        )

        # Cache result
        if self.enable_cache:
            self._add_to_cache(cache_key, result)

        self.logger.info(f"Generated {len(matches)} KWIC matches for query: '{query}'")

        return result

    def _parse_query(self, query: str) -> List[str]:
        """
        Parse query into keywords.

        Args:
            query: Query string

        Returns:
            List of keywords
        """
        # Simple tokenization
        keywords = query.split()

        # Apply case sensitivity
        if not self.case_sensitive:
            keywords = [kw.lower() for kw in keywords]

        return keywords

    def _find_keyword_matches(self, text: str, keyword: str,
                             doc_id: Optional[int]) -> List[KWICMatch]:
        """
        Find all matches of keyword in text.

        Args:
            text: Document text
            keyword: Keyword to find
            doc_id: Document identifier

        Returns:
            List of KWICMatch objects
        """
        matches = []

        # Prepare text for matching
        search_text = text if self.case_sensitive else text.lower()
        search_keyword = keyword if self.case_sensitive else keyword.lower()

        # Find all occurrences
        start = 0
        while True:
            pos = search_text.find(search_keyword, start)
            if pos == -1:
                break

            # Extract context
            left_context, right_context = self._extract_context(text, pos, len(keyword))

            match = KWICMatch(
                keyword=text[pos:pos+len(keyword)],  # Original case
                position=pos,
                left_context=left_context,
                right_context=right_context,
                doc_id=doc_id
            )
            matches.append(match)

            start = pos + len(keyword)

        return matches

    def _extract_context(self, text: str, position: int,
                        keyword_length: int) -> Tuple[str, str]:
        """
        Extract left and right context around keyword.

        Args:
            text: Document text
            position: Keyword position
            keyword_length: Length of keyword

        Returns:
            Tuple (left_context, right_context)
        """
        if self.window_type == 'fixed':
            return self._extract_fixed_window(text, position, keyword_length)
        elif self.window_type == 'sentence':
            return self._extract_sentence_window(text, position, keyword_length)
        elif self.window_type == 'adaptive':
            return self._extract_adaptive_window(text, position, keyword_length)
        else:
            raise ValueError(f"Unknown window type: {self.window_type}")

    def _extract_fixed_window(self, text: str, position: int,
                             keyword_length: int) -> Tuple[str, str]:
        """
        Extract fixed-size character window.

        Args:
            text: Document text
            position: Keyword position
            keyword_length: Length of keyword

        Returns:
            Tuple (left_context, right_context)

        Examples:
            >>> generator = KWICGenerator(window_size=10, window_type='fixed')
            >>> left, right = generator._extract_fixed_window("This is a test document", 10, 4)
            >>> len(left) <= 10 and len(right) <= 10
            True
        """
        # Left context
        left_start = max(0, position - self.window_size)
        left_context = text[left_start:position]

        # Trim to word boundary
        if left_start > 0 and ' ' in left_context:
            left_context = left_context[left_context.find(' ') + 1:]

        # Right context
        right_end = min(len(text), position + keyword_length + self.window_size)
        right_context = text[position + keyword_length:right_end]

        # Trim to word boundary
        if right_end < len(text) and ' ' in right_context:
            right_context = right_context[:right_context.rfind(' ')]

        return left_context, right_context

    def _extract_sentence_window(self, text: str, position: int,
                                 keyword_length: int) -> Tuple[str, str]:
        """
        Extract sentence containing keyword.

        Args:
            text: Document text
            position: Keyword position
            keyword_length: Length of keyword

        Returns:
            Tuple (left_context, right_context)
        """
        # Find sentence boundaries
        sentence_terminators = '.!?'

        # Find start of sentence (after previous terminator)
        sent_start = 0
        for i in range(position - 1, -1, -1):
            if text[i] in sentence_terminators:
                sent_start = i + 1
                break

        # Find end of sentence
        sent_end = len(text)
        for i in range(position + keyword_length, len(text)):
            if text[i] in sentence_terminators:
                sent_end = i
                break

        # Extract contexts
        left_context = text[sent_start:position].strip()
        right_context = text[position + keyword_length:sent_end].strip()

        return left_context, right_context

    def _extract_adaptive_window(self, text: str, position: int,
                                keyword_length: int) -> Tuple[str, str]:
        """
        Extract adaptive window based on content.

        Tries to extract complete phrases by looking for natural breakpoints
        (commas, semicolons, etc.) within the window.

        Args:
            text: Document text
            position: Keyword position
            keyword_length: Length of keyword

        Returns:
            Tuple (left_context, right_context)
        """
        # Start with fixed window
        left_context, right_context = self._extract_fixed_window(
            text, position, keyword_length
        )

        # Look for natural breakpoints
        breakpoints = [',', ';', ':', '-', '(', ')']

        # Adjust left context
        for bp in breakpoints:
            if bp in left_context:
                idx = left_context.rfind(bp)
                left_context = left_context[idx + 1:].strip()
                break

        # Adjust right context
        for bp in breakpoints:
            if bp in right_context:
                idx = right_context.find(bp)
                right_context = right_context[:idx].strip()
                break

        return left_context, right_context

    # ========================================================================
    # Multi-Document KWIC
    # ========================================================================

    def generate_multi(self, documents: List[str], query: str,
                      max_matches_per_doc: int = 3) -> KWICResult:
        """
        Generate KWIC snippets across multiple documents.

        Args:
            documents: List of document texts
            query: Query string
            max_matches_per_doc: Max matches per document

        Returns:
            KWICResult with matches from all documents

        Complexity:
            Time: O(d × n × m) where d=documents, n=doc length, m=query terms
            Space: O(d × k × w) where k=matches per doc
        """
        self.logger.info(f"Multi-document KWIC: {len(documents)} documents")

        all_matches = []

        for doc_id, doc_text in enumerate(documents):
            result = self.generate(doc_text, query, max_matches_per_doc, doc_id)
            all_matches.extend(result.matches)

        return KWICResult(
            matches=all_matches,
            query=query,
            num_documents=len(documents),
            cache_hit=False
        )

    # ========================================================================
    # Cache Management
    # ========================================================================

    def _add_to_cache(self, key: Tuple[str, int], result: KWICResult):
        """
        Add result to cache with LRU eviction.

        Args:
            key: Cache key
            result: Result to cache
        """
        # Check size limit
        if len(self._cache) >= self.max_cache_size:
            # Remove oldest entry (simple LRU)
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]
            self.logger.debug("Cache eviction: removed oldest entry")

        self._cache[key] = (result, time.time())

    def clear_cache(self):
        """Clear the cache."""
        self._cache.clear()
        self.logger.info("Cache cleared")

    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        return {
            'size': len(self._cache),
            'max_size': self.max_cache_size,
            'enabled': self.enable_cache
        }

    # ========================================================================
    # Formatting and Display
    # ========================================================================

    def format_results(self, result: KWICResult,
                      max_display: Optional[int] = None,
                      highlight_style: str = 'markdown') -> str:
        """
        Format KWIC results for display.

        Args:
            result: KWICResult to format
            max_display: Maximum matches to display
            highlight_style: Highlighting style ('markdown', 'ansi', 'html')

        Returns:
            Formatted string

        Examples:
            >>> generator = KWICGenerator()
            >>> text = "Machine learning is powerful"
            >>> result = generator.generate(text, "learning")
            >>> output = generator.format_results(result, highlight_style='markdown')
            >>> '**learning**' in output
            True
        """
        matches = result.matches[:max_display] if max_display else result.matches

        lines = []
        lines.append(f"Query: '{result.query}'")
        lines.append(f"Matches: {result.num_matches} in {result.num_documents} document(s)")

        if result.cache_hit:
            lines.append("[Cached result]")

        lines.append("-" * 60)

        for i, match in enumerate(matches):
            # Apply highlighting
            if highlight_style == 'markdown':
                snippet = match.snippet
            elif highlight_style == 'ansi':
                snippet = f"{match.left_context}\033[1;31m{match.keyword}\033[0m{match.right_context}"
            elif highlight_style == 'html':
                snippet = f"{match.left_context}<mark>{match.keyword}</mark>{match.right_context}"
            else:
                snippet = match.plain_snippet

            # Format line
            doc_info = f"[Doc {match.doc_id}] " if match.doc_id is not None else ""
            lines.append(f"{i+1}. {doc_info}{snippet}")

        return '\n'.join(lines)


def demo():
    """Demonstration of dynamic KWIC summarization."""
    print("=" * 60)
    print("Dynamic KWIC Summarization Demo")
    print("=" * 60)

    # Sample document
    text = """
    Machine learning is a subset of artificial intelligence that enables
    computers to learn from data without explicit programming. Deep learning
    is a type of machine learning based on neural networks. Neural networks
    consist of layers of interconnected nodes. Applications of machine learning
    include image recognition, natural language processing, and recommendation
    systems. The field of machine learning has grown rapidly in recent years.
    """

    # Example 1: Fixed window KWIC
    print("\n1. Fixed Window KWIC (window=30):")
    generator = KWICGenerator(window_size=30, window_type='fixed')
    result = generator.generate(text, "machine learning")

    output = generator.format_results(result, max_display=3)
    print(output)

    # Example 2: Sentence window KWIC
    print("\n2. Sentence Window KWIC:")
    generator = KWICGenerator(window_type='sentence')
    result = generator.generate(text, "neural networks")

    output = generator.format_results(result)
    print(output)

    # Example 3: Adaptive window KWIC
    print("\n3. Adaptive Window KWIC:")
    generator = KWICGenerator(window_size=40, window_type='adaptive')
    result = generator.generate(text, "learning")

    output = generator.format_results(result, max_display=2)
    print(output)

    # Example 4: Multi-document KWIC
    print("\n4. Multi-Document KWIC:")
    docs = [
        "Python is a popular programming language. It is easy to learn.",
        "Machine learning often uses Python for implementation.",
        "Python has many libraries for data science and machine learning."
    ]

    generator = KWICGenerator(window_size=25)
    result = generator.generate_multi(docs, "python learning", max_matches_per_doc=2)

    output = generator.format_results(result)
    print(output)

    # Example 5: Cache demonstration
    print("\n5. Cache Demonstration:")
    generator = KWICGenerator(enable_cache=True)

    # First call (cache miss)
    start = time.time()
    result1 = generator.generate(text, "machine")
    time1 = time.time() - start
    print(f"   First call: {result1.num_matches} matches, {time1*1000:.2f}ms, cache_hit={result1.cache_hit}")

    # Second call (cache hit)
    start = time.time()
    result2 = generator.generate(text, "machine")
    time2 = time.time() - start
    print(f"   Second call: {result2.num_matches} matches, {time2*1000:.2f}ms, cache_hit={result2.cache_hit}")

    # Cache stats
    stats = generator.get_cache_stats()
    print(f"   Cache stats: {stats}")

    print("\n" + "=" * 60)


# Alias for backward compatibility
KWIC = KWICGenerator


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    demo()
