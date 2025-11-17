"""
Unit Tests for Inverted Index

Test suite for inverted index data structure and operations.

Author: Information Retrieval System
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ir.index.inverted_index import InvertedIndex


@pytest.fixture
def sample_docs():
    """Sample documents for testing."""
    return [
        "hello world",
        "world peace",
        "hello peace",
        "information retrieval",
        "retrieval systems"
    ]


@pytest.fixture
def index(sample_docs):
    """Build index from sample docs."""
    idx = InvertedIndex()
    idx.build(sample_docs)
    return idx


@pytest.mark.unit
@pytest.mark.index
class TestInvertedIndexBasic:
    """Basic inverted index tests."""

    def test_build_index(self, sample_docs):
        """Test building index."""
        idx = InvertedIndex()
        idx.build(sample_docs)

        assert idx.doc_count == 5
        assert len(idx.vocabulary) > 0
        assert "hello" in idx.vocabulary
        assert "world" in idx.vocabulary

    def test_get_postings(self, index):
        """Test retrieving posting lists."""
        postings = index.get_postings("hello")
        assert len(postings) == 2
        assert (0, 1) in postings
        assert (2, 1) in postings

    def test_get_doc_ids(self, index):
        """Test getting document IDs."""
        doc_ids = index.get_doc_ids("world")
        assert doc_ids == {0, 1}

    def test_term_frequency(self, index):
        """Test term frequency calculation."""
        tf = index.term_frequency("hello", 0)
        assert tf == 1

        tf_missing = index.term_frequency("hello", 99)
        assert tf_missing == 0

    def test_document_frequency(self, index):
        """Test document frequency."""
        df = index.document_frequency("hello")
        assert df == 2

        df = index.document_frequency("retrieval")
        assert df == 2


@pytest.mark.unit
@pytest.mark.index
class TestBooleanOperations:
    """Test Boolean operations on posting lists."""

    def test_intersect(self, index):
        """Test AND operation."""
        p1 = index.get_postings("hello")
        p2 = index.get_postings("peace")
        result = index.intersect(p1, p2)

        # Only doc 2 has both
        assert len(result) == 1
        assert result[0][0] == 2

    def test_union(self, index):
        """Test OR operation."""
        p1 = index.get_postings("hello")
        p2 = index.get_postings("world")
        result = index.union(p1, p2)

        # Docs 0, 1, 2
        doc_ids = {doc_id for doc_id, _ in result}
        assert doc_ids == {0, 1, 2}

    def test_negate(self, index):
        """Test NOT operation."""
        p1 = index.get_postings("hello")
        result = index.negate(p1)

        # All docs except 0, 2
        doc_ids = {doc_id for doc_id, _ in result}
        assert doc_ids == {1, 3, 4}


@pytest.mark.unit
@pytest.mark.index
class TestIndexStats:
    """Test index statistics."""

    def test_get_stats(self, index):
        """Test statistics calculation."""
        stats = index.get_stats()

        assert stats['doc_count'] == 5
        assert stats['vocabulary_size'] > 0
        assert stats['total_postings'] > 0
        assert stats['avg_doc_length'] > 0


@pytest.mark.integration
@pytest.mark.index
class TestSaveLoad:
    """Test save/load functionality."""

    def test_save_load(self, index, tmp_path):
        """Test saving and loading index."""
        filepath = tmp_path / "test_index.json"

        # Save
        index.save(str(filepath))
        assert filepath.exists()

        # Load
        new_index = InvertedIndex()
        new_index.load(str(filepath))

        assert new_index.doc_count == index.doc_count
        assert new_index.vocabulary == index.vocabulary


@pytest.mark.unit
@pytest.mark.index
class TestEdgeCases:
    """Test edge cases."""

    def test_empty_documents(self):
        """Test with empty document list."""
        idx = InvertedIndex()
        idx.build([])
        assert idx.doc_count == 0
        assert len(idx.vocabulary) == 0

    def test_single_document(self):
        """Test with single document."""
        idx = InvertedIndex()
        idx.build(["hello world"])
        assert idx.doc_count == 1
        assert "hello" in idx.vocabulary

    def test_missing_term(self, index):
        """Test querying missing term."""
        postings = index.get_postings("nonexistent")
        assert len(postings) == 0

        doc_ids = index.get_doc_ids("nonexistent")
        assert len(doc_ids) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
