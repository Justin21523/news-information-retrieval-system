"""Tests for Term Weighting"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ir.index.term_weighting import TermWeighting
from src.ir.index.inverted_index import InvertedIndex


@pytest.fixture
def sample_index():
    """Build and return a small inverted index for term-weighting tests."""
    docs = ["hello world", "world peace", "hello peace world"]
    idx = InvertedIndex()
    idx.build(docs)
    return idx


@pytest.fixture
def tw(sample_index):
    """Return a TermWeighting instance built from the sample index."""
    tw = TermWeighting()
    tw.build_from_index(sample_index)
    return tw


@pytest.mark.unit
class TestTFCalculation:
    """Unit tests for TF calculation schemes."""
    def test_natural_tf(self, tw):
        """Use natural TF scheme where TF equals raw term count."""
        doc = {"hello": 3, "world": 2}
        assert tw.tf("hello", doc, 'n') == 3.0

    def test_log_tf(self, tw):
        """Use log TF scheme which grows sublinearly with raw counts."""
        doc = {"hello": 10}
        tf = tw.tf("hello", doc, 'l')
        assert tf > 1.0

    def test_boolean_tf(self, tw):
        """Use boolean TF scheme which caps TF to 1.0 when term exists."""
        doc = {"hello": 5}
        assert tw.tf("hello", doc, 'b') == 1.0


@pytest.mark.unit
class TestIDFCalculation:
    """Unit tests for IDF calculation schemes."""
    def test_standard_idf(self, tw):
        """Compute IDF with the standard scheme (should be positive)."""
        idf = tw.idf_value("hello", 't')
        assert idf > 0

    def test_no_idf(self, tw):
        """Use no-IDF scheme which returns a constant scaling factor (1.0)."""
        assert tw.idf_value("hello", 'n') == 1.0


@pytest.mark.unit
class TestVectorization:
    """Unit tests for document vectorization (TF-IDF + normalization)."""
    def test_vectorize(self, tw):
        """Vectorize a term-count dict into a normalized TF-IDF sparse vector."""
        doc = {"hello": 2, "world": 3}
        vec = tw.vectorize(doc, 'l', 't', 'c')
        assert len(vec) > 0
        assert all(0 <= v <= 1 for v in vec.values())


@pytest.mark.unit
class TestCosineSimilarity:
    """Unit tests for cosine similarity behavior."""
    def test_identical_vectors(self, tw):
        """Return cosine similarity 1.0 for identical vectors."""
        v = {"hello": 0.6, "world": 0.8}
        assert tw.cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self, tw):
        """Return cosine similarity 0.0 when vectors share no common terms."""
        v1 = {"hello": 1.0}
        v2 = {"world": 1.0}
        assert tw.cosine_similarity(v1, v2) == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
