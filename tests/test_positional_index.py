"""
Unit Tests for Positional Index

Author: Information Retrieval System
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ir.index.positional_index import PositionalIndex


@pytest.fixture
def sample_docs():
    """Sample documents."""
    return [
        "information retrieval is important",
        "retrieval models include boolean and vector models",
        "vector space model",
        "information extraction systems"
    ]


@pytest.fixture
def index(sample_docs):
    """Build positional index."""
    idx = PositionalIndex()
    idx.build(sample_docs)
    return idx


@pytest.mark.unit
@pytest.mark.index
class TestPositionalIndexBasic:
    """Basic positional index tests."""

    def test_build_index(self, sample_docs):
        """Test building index."""
        idx = PositionalIndex()
        idx.build(sample_docs)
        assert idx.doc_count == 4
        assert len(idx.vocabulary) > 0

    def test_get_positions(self, index):
        """Test position retrieval."""
        positions = index.get_positions("information", 0)
        assert positions == [0]

        positions = index.get_positions("models", 1)
        assert positions == [1, 6]

    def test_get_doc_ids(self, index):
        """Test document ID retrieval."""
        doc_ids = index.get_doc_ids("information")
        assert doc_ids == {0, 3}


@pytest.mark.unit
@pytest.mark.index
class TestPhraseQuery:
    """Test phrase queries."""

    def test_simple_phrase(self, index):
        """Test simple phrase query."""
        result = index.phrase_query("information retrieval")
        assert 0 in result

    def test_phrase_not_found(self, index):
        """Test phrase not in documents."""
        result = index.phrase_query("boolean retrieval")
        assert len(result) == 0

    def test_single_term_phrase(self, index):
        """Test single-term phrase."""
        result = index.phrase_query("information")
        assert 0 in result
        assert 3 in result


@pytest.mark.unit
@pytest.mark.index
class TestProximityQuery:
    """Test proximity queries."""

    def test_proximity_near(self, index):
        """Test proximity query."""
        result = index.proximity_query("information", "retrieval", 1)
        assert 0 in result

    def test_proximity_too_far(self, index):
        """Test terms too far apart."""
        result = index.proximity_query("information", "systems", 1)
        assert len(result) == 0


@pytest.mark.integration
@pytest.mark.index
class TestSaveLoad:
    """Test save/load."""

    def test_save_load(self, index, tmp_path):
        """Test saving and loading."""
        filepath = tmp_path / "pos_index.json"

        index.save(str(filepath))
        assert filepath.exists()

        new_index = PositionalIndex()
        new_index.load(str(filepath))

        assert new_index.doc_count == index.doc_count
        assert new_index.vocabulary == index.vocabulary


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
