"""Tests for Term Weighting"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ir.index.term_weighting import TermWeighting
from src.ir.index.inverted_index import InvertedIndex


@pytest.fixture
def sample_index():
    docs = ["hello world", "world peace", "hello peace world"]
    idx = InvertedIndex()
    idx.build(docs)
    return idx


@pytest.fixture
def tw(sample_index):
    tw = TermWeighting()
    tw.build_from_index(sample_index)
    return tw


@pytest.mark.unit
class TestTFCalculation:
    def test_natural_tf(self, tw):
        doc = {"hello": 3, "world": 2}
        assert tw.tf("hello", doc, 'n') == 3.0

    def test_log_tf(self, tw):
        doc = {"hello": 10}
        tf = tw.tf("hello", doc, 'l')
        assert tf > 1.0

    def test_boolean_tf(self, tw):
        doc = {"hello": 5}
        assert tw.tf("hello", doc, 'b') == 1.0


@pytest.mark.unit
class TestIDFCalculation:
    def test_standard_idf(self, tw):
        idf = tw.idf_value("hello", 't')
        assert idf > 0

    def test_no_idf(self, tw):
        assert tw.idf_value("hello", 'n') == 1.0


@pytest.mark.unit
class TestVectorization:
    def test_vectorize(self, tw):
        doc = {"hello": 2, "world": 3}
        vec = tw.vectorize(doc, 'l', 't', 'c')
        assert len(vec) > 0
        assert all(0 <= v <= 1 for v in vec.values())


@pytest.mark.unit
class TestCosineSimilarity:
    def test_identical_vectors(self, tw):
        v = {"hello": 0.6, "world": 0.8}
        assert tw.cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self, tw):
        v1 = {"hello": 1.0}
        v2 = {"world": 1.0}
        assert tw.cosine_similarity(v1, v2) == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
