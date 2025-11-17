"""
Unit Tests for Boolean Query Engine

Author: Information Retrieval System
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ir.index.inverted_index import InvertedIndex
from src.ir.index.positional_index import PositionalIndex
from src.ir.retrieval.boolean import BooleanQueryEngine


@pytest.fixture
def sample_docs():
    """Sample documents."""
    return [
        "information retrieval systems",
        "boolean retrieval models",
        "vector space models",
        "information extraction",
        "retrieval evaluation"
    ]


@pytest.fixture
def engine(sample_docs):
    """Create Boolean query engine."""
    inv_idx = InvertedIndex()
    inv_idx.build(sample_docs)

    pos_idx = PositionalIndex()
    pos_idx.build(sample_docs)

    return BooleanQueryEngine(inv_idx, pos_idx)


@pytest.mark.unit
@pytest.mark.retrieval
class TestSimpleQueries:
    """Test simple queries."""

    def test_single_term(self, engine):
        """Test single term query."""
        result = engine.query("information")
        assert len(result.doc_ids) == 2
        assert 0 in result.doc_ids
        assert 3 in result.doc_ids

    def test_and_query(self, engine):
        """Test AND query."""
        result = engine.query("information AND retrieval")
        assert len(result.doc_ids) == 1
        assert 0 in result.doc_ids

    def test_or_query(self, engine):
        """Test OR query."""
        result = engine.query("boolean OR vector")
        assert len(result.doc_ids) == 2

    def test_not_query(self, engine):
        """Test NOT query."""
        result = engine.query("NOT information")
        assert 0 not in result.doc_ids
        assert 3 not in result.doc_ids


@pytest.mark.unit
@pytest.mark.retrieval
class TestPhraseQueries:
    """Test phrase queries."""

    def test_phrase_query(self, engine):
        """Test phrase query."""
        result = engine.query('"information retrieval"')
        assert 0 in result.doc_ids

    def test_phrase_not_found(self, engine):
        """Test phrase not in docs."""
        result = engine.query('"boolean extraction"')
        assert len(result.doc_ids) == 0


@pytest.mark.unit
@pytest.mark.retrieval
class TestComplexQueries:
    """Test complex queries."""

    def test_and_or(self, engine):
        """Test AND with OR."""
        result = engine.query("(boolean OR vector) AND models")
        assert len(result.doc_ids) >= 1

    def test_not_with_and(self, engine):
        """Test NOT with AND."""
        result = engine.query("retrieval AND NOT extraction")
        assert 3 not in result.doc_ids


@pytest.mark.unit
@pytest.mark.retrieval
class TestRanking:
    """Test result ranking."""

    def test_ranked_results(self, engine):
        """Test ranked results."""
        result = engine.query("retrieval", rank_results=True)
        assert result.scores is not None
        assert len(result.scores) == len(result.doc_ids)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
