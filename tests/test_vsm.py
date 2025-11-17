"""Tests for Vector Space Model"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ir.retrieval.vsm import VectorSpaceModel


@pytest.fixture
def sample_docs():
    return [
        "information retrieval systems",
        "vector space model",
        "boolean retrieval"
    ]


@pytest.fixture
def vsm(sample_docs):
    vsm = VectorSpaceModel()
    vsm.build_index(sample_docs)
    return vsm


@pytest.mark.unit
class TestVSMBasic:
    def test_build_index(self, sample_docs):
        vsm = VectorSpaceModel()
        vsm.build_index(sample_docs)
        assert vsm.inverted_index.doc_count == 3
        assert len(vsm.doc_vectors) == 3

    def test_search(self, vsm):
        result = vsm.search("information retrieval")
        assert result.num_results > 0
        assert len(result.doc_ids) > 0

    def test_empty_query(self, vsm):
        result = vsm.search("")
        assert result.num_results == 0


@pytest.mark.unit
class TestRanking:
    def test_ranked_results(self, vsm):
        result = vsm.search("information retrieval", topk=3)
        # Check scores are descending
        scores = [result.scores[doc_id] for doc_id in result.doc_ids]
        assert scores == sorted(scores, reverse=True)

    def test_topk_limit(self, vsm):
        result = vsm.search("information", topk=2)
        assert len(result.doc_ids) <= 2


@pytest.mark.unit
class TestDocumentSimilarity:
    def test_similarity(self, vsm):
        sim = vsm.similarity(0, 1)
        assert 0.0 <= sim <= 1.0

    def test_self_similarity(self, vsm):
        sim = vsm.similarity(0, 0)
        assert sim == pytest.approx(1.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
