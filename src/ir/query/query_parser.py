"""
Query Parser for Advanced Metadata Search

This module provides a parser for library-style search queries with support for:
    - Field-specific queries: title:台灣, category:政治
    - Boolean operators: AND, OR, NOT
    - Parentheses for grouping: (term1 OR term2) AND term3
    - Range queries: date:[2025-11-01 TO 2025-11-13]
    - Multi-value fields: tags:(AI OR 機器學習)

Grammar (simplified):
    Query := OrExpr
    OrExpr := AndExpr (OR AndExpr)*
    AndExpr := NotExpr (AND NotExpr)*
    NotExpr := NOT NotExpr | Term
    Term := FieldQuery | ( OrExpr )
    FieldQuery := FIELD : VALUE | FIELD : [ VALUE TO VALUE ]

Example Queries:
    - title:台灣
    - title:台灣 AND category:政治
    - (title:台灣 OR title:中國) AND NOT category:sports
    - date:[2025-11-01 TO 2025-11-13]
    - tags:(AI OR 機器學習 OR 人工智慧)

Author: Information Retrieval System
Date: 2025-11-17
"""

import re
import logging
from enum import Enum
from typing import List, Set, Tuple, Optional, Dict, Any
from dataclasses import dataclass


class Operator(Enum):
    """Boolean operators for query nodes."""
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    FIELD = "FIELD"  # Field-specific query
    RANGE = "RANGE"  # Range query


@dataclass
class QueryNode:
    """
    Node in the query parse tree.

    Attributes:
        operator: Type of operation (AND, OR, NOT, FIELD, RANGE)
        field: Field name for field queries (e.g., 'title', 'category')
        value: Query value for field queries
        children: Child nodes for composite queries
        range_start: Start value for range queries
        range_end: End value for range queries
    """
    operator: Operator
    field: Optional[str] = None
    value: Optional[str] = None
    children: Optional[List['QueryNode']] = None
    range_start: Optional[str] = None
    range_end: Optional[str] = None

    def __repr__(self) -> str:
        """String representation for debugging."""
        if self.operator == Operator.FIELD:
            return f"FIELD({self.field}:{self.value})"
        elif self.operator == Operator.RANGE:
            return f"RANGE({self.field}:[{self.range_start} TO {self.range_end}])"
        elif self.operator == Operator.NOT:
            return f"NOT({self.children[0]})"
        else:
            children_repr = " ".join(str(c) for c in self.children)
            return f"{self.operator.value}({children_repr})"


class QueryParser:
    """
    Parser for library-style search queries.

    Parses query strings into Abstract Syntax Trees (AST) represented as QueryNode.
    Supports field queries, boolean operators, grouping, and range queries.

    Parsing strategy:
        - Tokenization via a single combined regex (`TOKEN_RE`)
        - Recursive descent parsing that mirrors the grammar rules
        - Precedence is handled by function layering:
            _parse_or_expr -> _parse_and_expr -> _parse_not_expr -> _parse_term

    Operator precedence (high -> low):
        NOT > AND (explicit or implicit) > OR

    Complexity:
        Time: O(n) where n = query string length
        Space: O(d) where d = parse tree depth
    """

    # Token patterns
    TOKEN_PATTERNS = [
        # Parentheses/brackets delimit grouping and range queries.
        ('LPAREN', r'\('),
        ('RPAREN', r'\)'),
        ('LBRACKET', r'\['),
        ('RBRACKET', r'\]'),
        ('COLON', r':'),
        # Boolean keywords are matched as whole words to avoid partial matches
        # inside values (e.g., "android" should not produce an "AND" token).
        ('AND', r'\bAND\b'),
        ('OR', r'\bOR\b'),
        ('NOT', r'\bNOT\b'),
        ('TO', r'\bTO\b'),
        # Fields are ASCII identifiers (we keep them strict to reduce ambiguity).
        ('FIELD', r'[a-zA-Z_][a-zA-Z0-9_]*'),  # Field names
        # Values accept Chinese characters, alphanumerics, underscore, dash, and dot.
        # This covers category codes, dates, and common metadata values.
        ('VALUE', r'[\u4e00-\u9fff\w\-\.]+'),  # Values (Chinese, alphanumeric, dash, dot)
        ('WHITESPACE', r'\s+'),
    ]

    # Compile regex patterns
    TOKEN_REGEX = '|'.join(f'(?P<{name}>{pattern})' for name, pattern in TOKEN_PATTERNS)
    TOKEN_RE = re.compile(TOKEN_REGEX)

    def __init__(self):
        """
        Initialize QueryParser.

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        self.logger = logging.getLogger(__name__)
        self.tokens: List[Tuple[str, str]] = []
        self.pos: int = 0

    def parse(self, query_str: str) -> QueryNode:
        """
        Parse query string into QueryNode tree.

        Args:
            query_str: Query string to parse

        Returns:
            Root QueryNode of the parse tree

        Raises:
            SyntaxError: If query string has syntax errors

        Complexity:
            Time: O(n) where n = query string length
            Space: O(d) where d = parse tree depth

        Example:
            >>> parser = QueryParser()
            >>> node = parser.parse("title:台灣 AND category:政治")
            >>> node.operator
            <Operator.AND: 'AND'>
            >>> len(node.children)
            2
        """
        self.logger.debug(f"Parsing query: {query_str}")

        # 1) Tokenize input into a stream like:
        #    [('FIELD','title'), ('COLON',':'), ('VALUE','台灣'), ('AND','AND'), ...]
        # 2) Parse the token stream into an AST (QueryNode tree).
        self.tokens = self._tokenize(query_str)
        self.pos = 0

        if not self.tokens:
            raise SyntaxError("Empty query")

        # Parse
        try:
            node = self._parse_or_expr()

            # Check for unconsumed tokens
            if self.pos < len(self.tokens):
                raise SyntaxError(f"Unexpected token: {self.tokens[self.pos]}")

            self.logger.debug(f"Parse tree: {node}")
            return node

        except IndexError:
            raise SyntaxError("Unexpected end of query")

    def _tokenize(self, query_str: str) -> List[Tuple[str, str]]:
        """
        Tokenize query string.

        Args:
            query_str: Query string

        Returns:
            List of (token_type, token_value) tuples

        Complexity:
            Time: O(n) where n = query string length
        """
        # We emit a flat token stream of (type, lexeme) pairs.
        # The recursive-descent parser consumes this stream left-to-right.
        tokens = []
        prev_token_type = None

        for match in self.TOKEN_RE.finditer(query_str):
            token_type = match.lastgroup
            token_value = match.group()

            # Skip whitespace
            if token_type == 'WHITESPACE':
                continue

            # Disambiguation rule:
            # After "FIELD:" the next token is a *value*, even if it looks like a
            # field name. For example, in "category:aipl" the substring "aipl"
            # matches the FIELD regex, but semantically it is a value.
            #
            # Without this rule, the parser would see:
            #   FIELD(category) COLON FIELD(aipl)
            # and fail because a field query expects FIELD COLON VALUE.
            if prev_token_type == 'COLON' and token_type == 'FIELD':
                # Check if it's a keyword (AND, OR, NOT, TO) - keep as is
                if token_value.upper() not in ('AND', 'OR', 'NOT', 'TO'):
                    token_type = 'VALUE'

            tokens.append((token_type, token_value))
            prev_token_type = token_type

        self.logger.debug(f"Tokens: {tokens}")
        return tokens

    def _current_token(self) -> Optional[Tuple[str, str]]:
        """
        Get current token without consuming it.

        Returns:
            (token_type, token_value) or None if at end
        """
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def _consume_token(self, expected_type: Optional[str] = None) -> Tuple[str, str]:
        """
        Consume and return current token.

        Args:
            expected_type: Expected token type (optional)

        Returns:
            (token_type, token_value)

        Raises:
            SyntaxError: If expected token type doesn't match
        """
        if self.pos >= len(self.tokens):
            raise SyntaxError(f"Expected {expected_type}, got end of query")

        token_type, token_value = self.tokens[self.pos]

        if expected_type and token_type != expected_type:
            raise SyntaxError(f"Expected {expected_type}, got {token_type}")

        self.pos += 1
        return token_type, token_value

    def _parse_or_expr(self) -> QueryNode:
        """
        Parse OR expression: AndExpr (OR AndExpr)*

        Returns:
            QueryNode for OR expression
        """
        # OR is the lowest-precedence operator, so we parse a full AND expression
        # first and then fold additional OR clauses left-associatively.
        left = self._parse_and_expr()

        # Check for OR operator
        while self._current_token() and self._current_token()[0] == 'OR':
            self._consume_token('OR')
            right = self._parse_and_expr()

            # Create OR node
            left = QueryNode(
                operator=Operator.OR,
                children=[left, right]
            )

        return left

    def _parse_and_expr(self) -> QueryNode:
        """
        Parse AND expression: NotExpr (AND NotExpr)*

        Returns:
            QueryNode for AND expression
        """
        # AND binds tighter than OR. We parse a NOT expression first and then
        # fold subsequent AND/implicit-AND terms left-associatively.
        left = self._parse_not_expr()

        # AND has higher precedence than OR. We also support "implicit AND":
        # two adjacent terms without an explicit operator are interpreted as
        # conjunction (common in library/search-engine style queries).
        while True:
            current = self._current_token()

            if not current:
                break

            token_type, _ = current

            # Explicit AND
            if token_type == 'AND':
                self._consume_token('AND')
                right = self._parse_not_expr()

                # Build a binary AND node.
                left = QueryNode(
                    operator=Operator.AND,
                    children=[left, right]
                )

            # Implicit AND (two terms without operator)
            elif token_type in ('FIELD', 'LPAREN', 'NOT'):
                # We treat adjacency as AND to support common "library search"
                # style queries like: title:台灣 category:政治
                right = self._parse_not_expr()

                left = QueryNode(
                    operator=Operator.AND,
                    children=[left, right]
                )

            else:
                break

        return left

    def _parse_not_expr(self) -> QueryNode:
        """
        Parse NOT expression: NOT NotExpr | Term

        Returns:
            QueryNode for NOT expression or term
        """
        current = self._current_token()

        # NOT is unary and has the highest precedence. We parse it recursively
        # so that "NOT NOT x" works (right-associative behavior).
        if current and current[0] == 'NOT':
            self._consume_token('NOT')
            child = self._parse_not_expr()

            return QueryNode(
                operator=Operator.NOT,
                children=[child]
            )

        return self._parse_term()

    def _parse_term(self) -> QueryNode:
        """
        Parse term: FieldQuery | ( OrExpr )

        Returns:
            QueryNode for term
        """
        current = self._current_token()

        if not current:
            raise SyntaxError("Expected term, got end of query")

        token_type, _ = current

        # Parenthesized expression: recurse back to the lowest-precedence rule.
        # This is how "(A OR B) AND C" overrides the default precedence.
        if token_type == 'LPAREN':
            self._consume_token('LPAREN')
            # Parse a full sub-expression inside parentheses.
            node = self._parse_or_expr()
            self._consume_token('RPAREN')
            return node

        # Field query
        if token_type == 'FIELD':
            return self._parse_field_query()

        raise SyntaxError(f"Expected FIELD or LPAREN, got {token_type}")

    def _parse_field_query(self) -> QueryNode:
        """
        Parse field query: FIELD : VALUE | FIELD : [ VALUE TO VALUE ]

        Returns:
            QueryNode for field query

        Example:
            title:台灣 -> FIELD(title:台灣)
            date:[2025-11-01 TO 2025-11-13] -> RANGE(date:[...])
        """
        # A field query starts with a FIELD token (e.g., "title"), followed by ":".
        _, field_name = self._consume_token('FIELD')

        # Consume colon
        self._consume_token('COLON')

        current = self._current_token()

        if not current:
            raise SyntaxError(f"Expected value after {field_name}:")

        token_type, _ = current

        # Range query: field:[start TO end]
        if token_type == 'LBRACKET':
            return self._parse_range_query(field_name)

        # Parenthesized multi-value: field:(value1 OR value2)
        # This is syntactic sugar for:
        #   (field:value1 OR field:value2 OR ...)
        if token_type == 'LPAREN':
            self._consume_token('LPAREN')
            # Parse as OR expression with field context
            value_node = self._parse_or_multi_values(field_name)
            self._consume_token('RPAREN')
            return value_node

        # Simple field query: field:value
        _, value = self._consume_token('VALUE')

        return QueryNode(
            operator=Operator.FIELD,
            field=field_name,
            value=value
        )

    def _parse_range_query(self, field_name: str) -> QueryNode:
        """
        Parse range query: [ VALUE TO VALUE ]

        Args:
            field_name: Field name for range query

        Returns:
            QueryNode for range query
        """
        self._consume_token('LBRACKET')

        _, start_value = self._consume_token('VALUE')
        self._consume_token('TO')
        _, end_value = self._consume_token('VALUE')

        self._consume_token('RBRACKET')

        return QueryNode(
            operator=Operator.RANGE,
            field=field_name,
            range_start=start_value,
            range_end=end_value
        )

    def _parse_or_multi_values(self, field_name: str) -> QueryNode:
        """
        Parse multi-value field query: value1 OR value2 OR value3

        Args:
            field_name: Field name for all values

        Returns:
            QueryNode with OR operator for multiple values
        """
        values = []

        # Parse first value
        _, value = self._consume_token('VALUE')
        values.append(value)

        # Parse additional values with OR
        while self._current_token() and self._current_token()[0] == 'OR':
            self._consume_token('OR')
            _, value = self._consume_token('VALUE')
            values.append(value)

        # Create OR node with FIELD children
        if len(values) == 1:
            return QueryNode(
                operator=Operator.FIELD,
                field=field_name,
                value=values[0]
            )

        children = [
            QueryNode(operator=Operator.FIELD, field=field_name, value=v)
            for v in values
        ]

        return QueryNode(
            operator=Operator.OR,
            children=children
        )


# Convenience function for quick parsing
def parse_query(query_str: str) -> QueryNode:
    """
    Convenience function to parse a query string.

    Args:
        query_str: Query string to parse

    Returns:
        Root QueryNode of the parse tree

    Example:
        >>> node = parse_query("title:台灣 AND category:政治")
        >>> node.operator
        <Operator.AND: 'AND'>
    """
    parser = QueryParser()
    return parser.parse(query_str)
