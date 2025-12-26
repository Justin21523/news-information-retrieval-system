"""Tests for Vector Space Model"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ir.retrieval.vsm import VectorSpaceModel


@pytest.fixture
def sample_docs():
    """Return a small document collection for VSM unit tests."""
    return [
        "information retrieval systems",
        "vector space model",
        "boolean retrieval"
    ]


@pytest.fixture
def vsm(sample_docs):
    """Build and return a VectorSpaceModel instance over the sample documents."""
    vsm = VectorSpaceModel()
    vsm.build_index(sample_docs)
    return vsm


@pytest.mark.unit
class TestVSMBasic:
    """Unit tests for VSM index construction and basic query handling."""
    def test_build_index(self, sample_docs):
        """Build an index and verify document/vector counts are consistent."""
        vsm = VectorSpaceModel()
        vsm.build_index(sample_docs)
        assert vsm.inverted_index.doc_count == 3
        assert len(vsm.doc_vectors) == 3

    def test_search(self, vsm):
        """Search returns at least one result for a matching multi-term query."""
        result = vsm.search("information retrieval")
        assert result.num_results > 0
        assert len(result.doc_ids) > 0

    def test_empty_query(self, vsm):
        """Empty query returns an empty result without raising errors."""
        result = vsm.search("")
        assert result.num_results == 0


@pytest.mark.unit
class TestRanking:
    """Unit tests for ranked retrieval output from the VSM search API."""
    def test_ranked_results(self, vsm):
        """Ensure returned document IDs are sorted by score descending."""
        result = vsm.search("information retrieval", topk=3)
        # Check scores are descending
        scores = [result.scores[doc_id] for doc_id in result.doc_ids]
        assert scores == sorted(scores, reverse=True)

    def test_topk_limit(self, vsm):
        """Respect the top-k cut when returning ranked results."""
        result = vsm.search("information", topk=2)
        assert len(result.doc_ids) <= 2


@pytest.mark.unit
class TestDocumentSimilarity:
    """Unit tests for document-document similarity computations."""
    def test_similarity(self, vsm):
        """Return a bounded similarity score for two different documents."""
        sim = vsm.similarity(0, 1)
        assert 0.0 <= sim <= 1.0

    def test_self_similarity(self, vsm):
        """Return similarity 1.0 when comparing a document with itself."""
        sim = vsm.similarity(0, 0)
        assert sim == pytest.approx(1.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
