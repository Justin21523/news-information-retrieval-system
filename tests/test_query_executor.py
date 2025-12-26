"""
Unit Tests for QueryExecutor

These tests cover field-based boolean execution over FieldIndexer, including
AND/OR/NOT set semantics and date range queries.

Author: Information Retrieval System
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ir.index.field_indexer import FieldIndexer
from src.ir.query.query_executor import QueryExecutor
from src.ir.query.query_parser import QueryParser


@pytest.fixture
def sample_documents():
    """Return a small, deterministic document set for field query execution."""
    return [
        {
            "title": "台灣 政治",
            "content": "台灣 政治 新聞",
            "category": "politics",
            "source": "中央社",
            "author": "記者張三",
            "published_date": "2025-11-10",
        },
        {
            "title": "台灣 經濟",
            "content": "台灣 經濟 發展",
            "category": "economy",
            "source": "中央社",
            "author": "記者李四",
            "published_date": "2025-11-12",
        },
        {
            "title": "美國 政治",
            "content": "美國 政治 分析",
            "category": "politics",
            "source": "BBC",
            "author": "記者王五",
            "published_date": "2025-11-01",
        },
    ]


@pytest.fixture
def field_indexer(sample_documents):
    """Build and return a FieldIndexer for the sample documents."""
    indexer = FieldIndexer()
    indexer.build(sample_documents)
    return indexer


@pytest.fixture
def executor(field_indexer, sample_documents):
    """Create a QueryExecutor bound to the sample index."""
    return QueryExecutor(field_indexer, documents=sample_documents)


@pytest.fixture
def parser():
    """Create a QueryParser instance."""
    return QueryParser()


@pytest.mark.unit
@pytest.mark.retrieval
class TestQueryExecutorBooleanSemantics:
    """Unit tests for boolean set operations over field queries."""

    def test_and_intersection(self, parser, executor):
        """AND returns intersection of child result sets."""
        node = parser.parse("title:台灣 AND category:politics")
        results = executor.execute(node)

        assert [r.doc_id for r in results] == [0]
        assert set(results[0].matched_fields) == {"title", "category"}

    def test_or_union(self, parser, executor):
        """OR returns union of child result sets."""
        node = parser.parse("category:economy OR category:politics")
        results = executor.execute(node)

        assert [r.doc_id for r in results] == [0, 1, 2]

    def test_not_complement(self, parser, executor):
        """NOT returns the complement relative to the indexed document universe."""
        node = parser.parse("category:politics AND NOT title:美國")
        results = executor.execute(node)

        assert [r.doc_id for r in results] == [0]


@pytest.mark.unit
@pytest.mark.retrieval
class TestQueryExecutorRangeQueries:
    """Unit tests for range query execution."""

    def test_date_range_alias_date_field(self, parser, executor):
        """Support the user-facing `date:[start TO end]` alias for published_date."""
        node = parser.parse("date:[2025-11-09 TO 2025-11-12]")
        results = executor.execute(node)

        assert [r.doc_id for r in results] == [0, 1]
        assert set(results[0].matched_fields) == {"date"}

    def test_date_range_published_date_field(self, parser, executor):
        """Support `published_date:[start TO end]` for structured metadata indexes."""
        node = parser.parse("published_date:[2025-11-09 TO 2025-11-12]")
        results = executor.execute(node)

        assert [r.doc_id for r in results] == [0, 1]
        assert set(results[0].matched_fields) == {"published_date"}


@pytest.mark.unit
@pytest.mark.retrieval
class TestQueryExecutorStructuredQueries:
    """Unit tests for JSON-like structured queries."""

    def test_structured_between_date_range(self, executor):
        """The `between` operator maps to a RANGE query node."""
        conditions = [
            {
                "field": "published_date",
                "operator": "between",
                "value": ["2025-11-09", "2025-11-12"],
            }
        ]
        results = executor.execute_structured_query(conditions, logic="AND")

        assert [r.doc_id for r in results] == [0, 1]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
