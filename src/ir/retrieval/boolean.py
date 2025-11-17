"""
Boolean Retrieval System

This module implements a Boolean query engine that supports AND, OR, NOT
operations and phrase queries. It uses inverted and positional indices for
efficient query processing.

Key Features:
    - Boolean operators: AND, OR, NOT
    - Phrase queries: "exact phrase"
    - Field-based queries: field:value, field:"phrase"
    - Date range queries: date:[start TO end]
    - Nested queries with parentheses (Shunting Yard algorithm)
    - Query optimization (term ordering)
    - Result ranking by term frequency

Query Syntax Examples:
    - title:台灣 AND category:政治
    - author:"記者" OR source:中央社
    - date:[2025-11-01 TO 2025-11-13]
    - (title:AI OR content:人工智慧) AND NOT category:娛樂
    - tags:科技 AND published_date:[2025-11-10 TO 2025-11-13]

Author: Information Retrieval System
License: Educational Use
"""

import re
import logging
from typing import List, Set, Tuple, Optional
from dataclasses import dataclass

import sys
from pathlib import Path

# Add parent directory to path for imports
_parent_dir = Path(__file__).parent.parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

from src.ir.index.inverted_index import InvertedIndex
from src.ir.index.positional_index import PositionalIndex
from src.ir.index.field_indexer import FieldIndexer
from src.ir.retrieval.wildcard import WildcardExpander


@dataclass
class QueryResult:
    """
    Result of a Boolean query.

    Attributes:
        doc_ids: List of matching document IDs
        query: Original query string
        num_results: Number of results
        scores: Optional scores for ranking
    """
    doc_ids: List[int]
    query: str
    num_results: int
    scores: Optional[dict] = None


class BooleanQueryEngine:
    """
    Boolean Query Engine for document retrieval.

    Supports:
        - AND, OR, NOT operators
        - Phrase queries (quoted strings)
        - Field-based queries (field:value)
        - Date range queries (date:[start TO end])
        - Operator precedence with Shunting Yard algorithm
        - Query optimization

    Query Syntax:
        - AND: term1 AND term2 (intersection)
        - OR: term1 OR term2 (union)
        - NOT: NOT term (negation)
        - Phrase: "exact phrase" (sequential terms)
        - Field: field:value (specific field search)
        - Field phrase: field:"phrase"
        - Date range: field:[2025-11-01 TO 2025-11-13]
        - Grouping: (term1 OR term2) AND term3

    Examples:
        >>> engine = BooleanQueryEngine(inverted_index, positional_index, field_indexer)
        >>> result = engine.query("title:台灣 AND category:政治")
        >>> result.doc_ids
        [0, 4, 16]

        >>> result = engine.query("date:[2025-11-01 TO 2025-11-13]")
        >>> result.doc_ids
        [0, 1, 2, 5, 8]

    Complexity:
        - Simple query: O(k) where k is result size
        - Complex query: O(k * log k) with optimization
        - Field query: O(1) average lookup
    """

    def __init__(self, inverted_index: InvertedIndex,
                 positional_index: Optional[PositionalIndex] = None,
                 field_indexer: Optional[FieldIndexer] = None,
                 max_wildcard_expansions: int = 50):
        """
        Initialize Boolean Query Engine.

        Args:
            inverted_index: Inverted index for term lookup
            positional_index: Optional positional index for phrase queries
            field_indexer: Optional field indexer for metadata search
            max_wildcard_expansions: Maximum terms to expand for wildcard queries
        """
        self.logger = logging.getLogger(__name__)
        self.inverted_index = inverted_index
        self.positional_index = positional_index
        self.field_indexer = field_indexer

        # Initialize wildcard expander
        self.wildcard_expander = WildcardExpander(max_expansions=max_wildcard_expansions)

        self.logger.info("BooleanQueryEngine initialized")

    def query(self, query_str: str, optimize: bool = True,
              rank_results: bool = False) -> QueryResult:
        """
        Execute a Boolean query.

        Args:
            query_str: Query string
            optimize: Whether to optimize query execution order
            rank_results: Whether to rank results by relevance

        Returns:
            QueryResult with matching documents

        Examples:
            >>> engine.query("information AND retrieval")
            QueryResult(doc_ids=[0, 4], query='...', num_results=2)
        """
        self.logger.debug(f"Executing query: {query_str}")

        # Parse query
        parsed = self._parse_query(query_str)

        # Execute query
        doc_ids = self._execute_query(parsed, optimize)

        # Rank if requested
        scores = None
        if rank_results and doc_ids:
            scores = self._rank_results(query_str, doc_ids)
            doc_ids = sorted(doc_ids, key=lambda x: scores[x], reverse=True)

        result = QueryResult(
            doc_ids=list(doc_ids),
            query=query_str,
            num_results=len(doc_ids),
            scores=scores
        )

        self.logger.debug(f"Query returned {result.num_results} results")
        return result

    def _parse_query(self, query_str: str) -> dict:
        """
        Parse query string into structured format.

        Handles:
            - Phrase extraction ("quoted phrases")
            - Field queries (field:value, field:"phrase")
            - Date range queries (field:[start TO end])
            - Operator identification (AND, OR, NOT)
            - Parentheses grouping

        Args:
            query_str: Raw query string

        Returns:
            Parsed query structure with tokens, phrases, and field queries

        Examples:
            >>> _parse_query("title:台灣 AND category:政治")
            {'tokens': ['title:台灣', 'AND', 'category:政治'], ...}

            >>> _parse_query('author:"記者" OR source:中央社')
            {'tokens': ['author:__PHRASE_0__', 'OR', 'source:中央社'], ...}
        """
        # Extract phrases first (preserve spaces in phrases)
        # Handle both regular phrases and field phrases
        phrases = []
        phrase_pattern = r'"([^"]+)"'

        def replace_phrase(match):
            phrase = match.group(1)
            placeholder = f"__PHRASE_{len(phrases)}__"
            phrases.append(phrase)
            return placeholder

        query_processed = re.sub(phrase_pattern, replace_phrase, query_str)

        # Tokenize (split on whitespace and operators)
        # Pattern explanation:
        # - \(|\) : parentheses
        # - AND|OR|NOT : boolean operators
        # - NEAR/\d+ : proximity operator (e.g., NEAR/3)
        # - __PHRASE_\d+__ : phrase placeholders
        # - \w+:\[[\w\s-]+\sTO\s[\w\s-]+\] : date range (field:[start TO end])
        # - \w+:__PHRASE_\d+__ : field phrase (field:"phrase")
        # - \w+:[\w\u4e00-\u9fff]+ : field term (field:value, supports Chinese)
        # - [\w\u4e00-\u9fff]+ : regular terms (supports Chinese)
        tokens = re.findall(
            r'\(|\)|AND|OR|NOT|NEAR/\d+|'
            r'__PHRASE_\d+__|'
            r'\w+:\[[\w\s-]+\sTO\s[\w\s-]+\]|'
            r'\w+:__PHRASE_\d+__|'
            r'\w+:[\w\u4e00-\u9fff]+|'
            r'[\w\u4e00-\u9fff]+',
            query_processed,
            re.IGNORECASE
        )

        return {
            'tokens': tokens,
            'phrases': phrases,
            'original': query_str
        }

    def _execute_query(self, parsed: dict, optimize: bool) -> Set[int]:
        """
        Execute parsed query.

        Args:
            parsed: Parsed query structure
            optimize: Whether to optimize

        Returns:
            Set of matching document IDs
        """
        tokens = parsed['tokens']
        phrases = parsed['phrases']

        if not tokens:
            return set()

        # Convert to postfix notation (RPN) for easier evaluation
        postfix = self._to_postfix(tokens)

        # Evaluate postfix expression
        result = self._evaluate_postfix(postfix, phrases, optimize)

        return result

    def _to_postfix(self, tokens: List[str]) -> List[str]:
        """
        Convert infix notation to postfix (Reverse Polish Notation).

        Uses Shunting Yard algorithm.

        Args:
            tokens: List of tokens in infix notation

        Returns:
            List of tokens in postfix notation
        """
        # Operator precedence
        # NEAR/n has same precedence as AND (binary proximity operator)
        precedence = {
            'NOT': 3,
            'AND': 2,
            'OR': 1,
            '(': 0
        }

        output = []
        operator_stack = []

        for token in tokens:
            token_upper = token.upper()

            # Check if it's a NEAR/n operator
            if token_upper.startswith('NEAR/'):
                # NEAR has same precedence as AND
                while (operator_stack and
                       operator_stack[-1] != '(' and
                       precedence.get(operator_stack[-1].upper().split('/')[0], 0) >= precedence['AND']):
                    output.append(operator_stack.pop())
                operator_stack.append(token)

            elif token_upper in ['AND', 'OR', 'NOT']:
                # Pop operators with higher or equal precedence
                while (operator_stack and
                       operator_stack[-1] != '(' and
                       precedence.get(operator_stack[-1].upper().split('/')[0], 0) >= precedence[token_upper]):
                    output.append(operator_stack.pop())
                operator_stack.append(token)

            elif token == '(':
                operator_stack.append(token)

            elif token == ')':
                # Pop until matching '('
                while operator_stack and operator_stack[-1] != '(':
                    output.append(operator_stack.pop())
                if operator_stack:
                    operator_stack.pop()  # Remove '('

            else:
                # Operand (term or phrase placeholder)
                output.append(token)

        # Pop remaining operators
        while operator_stack:
            output.append(operator_stack.pop())

        return output

    def _evaluate_postfix(self, postfix: List[str], phrases: List[str],
                         optimize: bool) -> Set[int]:
        """
        Evaluate postfix expression.

        Args:
            postfix: Postfix token list
            phrases: List of phrases
            optimize: Whether to optimize

        Returns:
            Set of matching doc IDs
        """
        stack = []

        for token in postfix:
            token_upper = token.upper()

            # Check for NEAR/n operator
            if token_upper.startswith('NEAR/'):
                if len(stack) < 2:
                    continue

                # Extract distance
                distance = int(token_upper.split('/')[1])

                # Get operands (these should be term strings, not result sets)
                right_item = stack.pop()
                left_item = stack.pop()

                # Handle NEAR query
                result = self._process_near_query(left_item, right_item, distance)
                stack.append(result)

            elif token_upper == 'AND':
                if len(stack) < 2:
                    continue
                right = stack.pop()
                left = stack.pop()
                result = left & right
                stack.append(result)

            elif token_upper == 'OR':
                if len(stack) < 2:
                    continue
                right = stack.pop()
                left = stack.pop()
                result = left | right
                stack.append(result)

            elif token_upper == 'NOT':
                if len(stack) < 1:
                    continue
                operand = stack.pop()
                # NOT: all documents except those in operand
                all_docs = set(range(self.inverted_index.doc_count))
                result = all_docs - operand
                stack.append(result)

            else:
                # Operand: term, phrase, field query, or field phrase

                # Check for field phrase query (field:__PHRASE_N__)
                if ':__PHRASE_' in token:
                    field_name, phrase_token = token.split(':', 1)
                    phrase_idx = int(re.search(r'\d+', phrase_token).group())
                    phrase = phrases[phrase_idx]

                    # For field phrases, search each term in the field
                    # (Full positional field search would require FieldIndexer extension)
                    if self.field_indexer:
                        phrase_terms = phrase.lower().split()
                        doc_ids = self.field_indexer.search_multi_terms(
                            field_name, phrase_terms, operator='AND'
                        )
                    else:
                        # Fallback to regular phrase search
                        doc_ids = self._process_phrase(phrase)

                # Check for regular phrase query (__PHRASE_N__)
                elif token.startswith('__PHRASE_'):
                    phrase_idx = int(re.search(r'\d+', token).group())
                    phrase = phrases[phrase_idx]
                    doc_ids = self._process_phrase(phrase)

                # Regular term or field query
                else:
                    doc_ids = self._process_term(token)

                stack.append(doc_ids)

        # Final result
        if stack:
            return stack[0]
        return set()

    def _process_term(self, term: str) -> Set[int]:
        """
        Get documents containing a term.

        Handles:
            - Regular terms (term)
            - Wildcard terms (info*, te?t, *form*)
            - Field queries (field:value)
            - Date range queries (field:[start TO end])

        Args:
            term: Query term (may include field prefix, wildcards)

        Returns:
            Set of doc IDs

        Examples:
            >>> _process_term("台灣")  # regular term
            {0, 1, 3, 7}

            >>> _process_term("info*")  # wildcard
            {0, 2, 5, 8}

            >>> _process_term("title:台灣")  # field query
            {0, 3}

            >>> _process_term("published_date:[2025-11-01 TO 2025-11-13]")
            {0, 1, 2, 5, 8}

        Complexity:
            Time: O(1) average for lookup, O(V) for wildcard, O(N) for date range
        """
        # Check if this is a date range query
        range_match = re.match(r'(\w+):\[([\w\s-]+)\sTO\s([\w\s-]+)\]', term)
        if range_match:
            field_name = range_match.group(1)
            start_date = range_match.group(2)
            end_date = range_match.group(3)
            return self._process_date_range(field_name, start_date, end_date)

        # Check if this is a field query
        if ':' in term:
            field_name, field_value = term.split(':', 1)
            return self._process_field_query(field_name, field_value)

        term_lower = term.lower()

        # Check for wildcard pattern
        if self.wildcard_expander.has_wildcard(term_lower):
            return self._process_wildcard(term_lower)

        # Regular term query
        return self.inverted_index.get_doc_ids(term_lower)

    def _process_wildcard(self, pattern: str) -> Set[int]:
        """
        Process wildcard query pattern.

        Args:
            pattern: Wildcard pattern (e.g., "info*", "te?t")

        Returns:
            Set of matching document IDs

        Complexity:
            Time: O(V + E * k) where V = vocabulary size, E = expanded terms, k = avg postings

        Examples:
            >>> _process_wildcard("info*")
            {0, 1, 3, 5}  # Union of all docs containing info, inform, information, etc.
        """
        # Expand wildcard to matching terms
        vocabulary = self.inverted_index.vocabulary
        expanded_terms = self.wildcard_expander.expand(pattern, vocabulary)

        if not expanded_terms:
            self.logger.debug(f"No terms matched wildcard pattern: {pattern}")
            return set()

        self.logger.debug(f"Wildcard '{pattern}' expanded to {len(expanded_terms)} terms")

        # Union of all matching documents (OR operation)
        result = set()
        for term in expanded_terms:
            result |= self.inverted_index.get_doc_ids(term)

        return result

    def _process_field_query(self, field: str, value: str) -> Set[int]:
        """
        Process field-based query.

        Args:
            field: Field name (e.g., 'title', 'author', 'category')
            value: Search value

        Returns:
            Set of matching document IDs

        Complexity:
            Time: O(1) average for exact fields, O(k) for tokenized fields
        """
        if not self.field_indexer:
            self.logger.warning(f"Field indexer not available, falling back to content search")
            # Fallback to regular search in content
            return self.inverted_index.get_doc_ids(value.lower())

        # Field search using FieldIndexer
        value_lower = value.lower()
        return self.field_indexer.search_field(field, value_lower)

    def _process_date_range(self, field: str, start_date: str, end_date: str) -> Set[int]:
        """
        Process date range query.

        Args:
            field: Date field name (e.g., 'published_date')
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            Set of matching document IDs

        Complexity:
            Time: O(N) where N = number of documents
        """
        if not self.field_indexer:
            self.logger.warning(f"Field indexer not available for date range query")
            return set()

        return self.field_indexer.search_date_range(field, start_date, end_date)

    def _process_near_query(self, left_item, right_item, distance: int) -> Set[int]:
        """
        Process NEAR/n proximity query.

        Args:
            left_item: Left operand (term string or result set)
            right_item: Right operand (term string or result set)
            distance: Maximum distance between terms

        Returns:
            Set of matching document IDs

        Complexity:
            Time: O(k * p1 * p2) where k = candidate docs

        Examples:
            >>> _process_near_query("資訊", "檢索", 3)
            {0, 1, 5}  # Documents where "資訊" and "檢索" are within 3 words
        """
        if not self.positional_index:
            self.logger.warning("Positional index not available for NEAR query")
            # Fallback to AND operation
            if isinstance(left_item, set) and isinstance(right_item, set):
                return left_item & right_item
            elif isinstance(left_item, str) and isinstance(right_item, str):
                return self.inverted_index.get_doc_ids(left_item.lower()) & \
                       self.inverted_index.get_doc_ids(right_item.lower())
            else:
                return set()

        # Handle different operand types
        # Case 1: Both are terms (strings)
        if isinstance(left_item, str) and isinstance(right_item, str):
            left_term = left_item.lower()
            right_term = right_item.lower()
            result = self.positional_index.proximity_query(left_term, right_term, distance)
            return set(result)

        # Case 2: Left is result set, right is term (unsupported - return AND)
        elif isinstance(left_item, set) and isinstance(right_item, str):
            self.logger.warning("NEAR with result set on left not fully supported, using AND")
            right_docs = self.inverted_index.get_doc_ids(right_item.lower())
            return left_item & right_docs

        # Case 3: Left is term, right is result set (unsupported - return AND)
        elif isinstance(left_item, str) and isinstance(right_item, set):
            self.logger.warning("NEAR with result set on right not fully supported, using AND")
            left_docs = self.inverted_index.get_doc_ids(left_item.lower())
            return left_docs & right_item

        # Case 4: Both are result sets (unsupported - return AND)
        elif isinstance(left_item, set) and isinstance(right_item, set):
            self.logger.warning("NEAR with result sets not supported, using AND")
            return left_item & right_item

        return set()

    def _process_phrase(self, phrase: str, field_prefix: str = None) -> Set[int]:
        """
        Get documents containing an exact phrase.

        Handles:
            - Regular phrases ("台灣新聞")
            - Field phrases via external handling (field:"phrase")

        Args:
            phrase: Phrase string
            field_prefix: Optional field name for field-restricted phrase search

        Returns:
            Set of doc IDs

        Note:
            Field phrase queries like title:"台灣新聞" are handled by checking
            if the field contains all terms in the phrase. Full positional field
            search would require additional FieldIndexer enhancements.

        Complexity:
            Time: O(P * k) where P = phrase length, k = posting list size
        """
        if not self.positional_index:
            self.logger.warning("Positional index not available for phrase query")
            # Fallback: return docs containing all terms
            terms = phrase.lower().split()
            if not terms:
                return set()

            result = self.inverted_index.get_doc_ids(terms[0])
            for term in terms[1:]:
                result &= self.inverted_index.get_doc_ids(term)
            return result

        # Use positional index for exact phrase matching
        doc_ids = self.positional_index.phrase_query(phrase)
        return set(doc_ids)

    def _rank_results(self, query_str: str, doc_ids: List[int]) -> dict:
        """
        Rank results by simple relevance score.

        Uses term frequency as a basic scoring mechanism.

        Args:
            query_str: Original query
            doc_ids: Document IDs to rank

        Returns:
            Dictionary mapping doc_id to score
        """
        # Extract terms from query (ignore operators and phrases)
        terms = re.findall(r'\b(?!AND|OR|NOT\b)\w+', query_str, re.IGNORECASE)
        terms = [t.lower() for t in terms]

        scores = {}
        for doc_id in doc_ids:
            score = 0.0
            for term in terms:
                # Add term frequency
                tf = self.inverted_index.term_frequency(term, doc_id)
                score += tf

            scores[doc_id] = score

        return scores

    def simple_query(self, terms: List[str], operator: str = 'AND') -> List[int]:
        """
        Execute a simple multi-term query with single operator.

        Args:
            terms: List of query terms
            operator: 'AND', 'OR', or 'NOT'

        Returns:
            List of matching document IDs

        Examples:
            >>> engine.simple_query(["information", "retrieval"], "AND")
            [0, 4]
        """
        if not terms:
            return []

        if operator.upper() == 'AND':
            # Intersection
            result = self.inverted_index.get_doc_ids(terms[0].lower())
            for term in terms[1:]:
                result &= self.inverted_index.get_doc_ids(term.lower())
            return sorted(result)

        elif operator.upper() == 'OR':
            # Union
            result = set()
            for term in terms:
                result |= self.inverted_index.get_doc_ids(term.lower())
            return sorted(result)

        elif operator.upper() == 'NOT':
            # Negation (all docs except those with terms)
            excluded = set()
            for term in terms:
                excluded |= self.inverted_index.get_doc_ids(term.lower())

            all_docs = set(range(self.inverted_index.doc_count))
            result = all_docs - excluded
            return sorted(result)

        return []

    def phrase_query(self, phrase: str) -> List[int]:
        """
        Execute a phrase query.

        Args:
            phrase: Phrase to search

        Returns:
            List of document IDs

        Examples:
            >>> engine.phrase_query("information retrieval")
            [0, 4]
        """
        if not self.positional_index:
            raise ValueError("Positional index required for phrase queries")

        return self.positional_index.phrase_query(phrase)

    def search(self, query_str: str, **kwargs) -> List[int]:
        """
        Alias for query() method for backward compatibility.

        Args:
            query_str: Query string
            **kwargs: Additional arguments passed to query()

        Returns:
            List of matching document IDs

        Examples:
            >>> engine.search("information AND retrieval")
            [0, 4]
        """
        result = self.query(query_str, **kwargs)
        return result.doc_ids


def demo():
    """Demonstration of Boolean Query Engine."""
    print("=" * 60)
    print("Boolean Query Engine Demo")
    print("=" * 60)

    # Sample documents
    documents = [
        "information retrieval is the process of obtaining information",
        "retrieval models include boolean and vector space models",
        "boolean retrieval uses AND OR NOT operators",
        "vector space model represents documents as vectors",
        "information extraction is related to information retrieval"
    ]

    # Build indices
    inv_index = InvertedIndex()
    inv_index.build(documents)

    pos_index = PositionalIndex()
    pos_index.build(documents)

    # Create engine
    engine = BooleanQueryEngine(inv_index, pos_index)

    # Test queries
    print("\n1. Simple Queries:")
    queries = [
        "information",
        "information AND retrieval",
        "boolean OR vector",
        "NOT vector"
    ]

    for q in queries:
        result = engine.query(q)
        print(f"   Query: '{q}'")
        print(f"   Results: {result.doc_ids} ({result.num_results} docs)\n")

    # Phrase queries
    print("2. Phrase Queries:")
    phrase_queries = [
        '"information retrieval"',
        '"vector space"',
        '"boolean retrieval"'
    ]

    for q in phrase_queries:
        result = engine.query(q)
        print(f"   Query: {q}")
        print(f"   Results: {result.doc_ids} ({result.num_results} docs)\n")

    # Complex queries
    print("3. Complex Queries:")
    complex_queries = [
        "information AND retrieval AND NOT extraction",
        '(boolean OR vector) AND model',
        '"vector space" OR "boolean retrieval"'
    ]

    for q in complex_queries:
        result = engine.query(q)
        print(f"   Query: '{q}'")
        print(f"   Results: {result.doc_ids} ({result.num_results} docs)\n")

    print("=" * 60)


# Alias for backward compatibility
BooleanRetrieval = BooleanQueryEngine


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    demo()
