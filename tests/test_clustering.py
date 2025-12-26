"""Tests for Clustering Algorithms"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ir.cluster.doc_cluster import DocumentClusterer, Cluster
from src.ir.cluster.term_cluster import TermClusterer, TermCluster


# Document Clustering Tests
@pytest.fixture
def doc_clusterer():
    """Return a DocumentClusterer instance for unit tests."""
    return DocumentClusterer()


@pytest.fixture
def sample_docs():
    """Return a small, separable set of document vectors for clustering tests."""
    return {
        0: {"term1": 0.8, "term2": 0.6},
        1: {"term1": 0.7, "term2": 0.5},
        2: {"term3": 0.9, "term4": 0.7},
        3: {"term3": 0.8, "term4": 0.6}
    }


@pytest.mark.unit
class TestDocumentClustering:
    """Unit tests for document clustering algorithms and utilities."""
    def test_cosine_similarity(self, doc_clusterer):
        """Return 1.0 cosine similarity for identical sparse vectors."""
        vec1 = {"term1": 1.0}
        vec2 = {"term1": 1.0}
        sim = doc_clusterer.cosine_similarity(vec1, vec2)
        assert sim == pytest.approx(1.0)

    def test_hac_clustering(self, doc_clusterer, sample_docs):
        """Cluster separable document vectors into k clusters with HAC."""
        result = doc_clusterer.hierarchical_clustering(sample_docs, k=2)
        assert result.num_clusters == 2
        assert len(result.clusters) == 2

    def test_kmeans_clustering(self, doc_clusterer, sample_docs):
        """Cluster separable document vectors into k clusters with k-means."""
        result = doc_clusterer.kmeans_clustering(sample_docs, k=2, random_seed=42)
        assert result.num_clusters == 2
        assert len(result.assignments) == 4


# Term Clustering Tests
@pytest.fixture
def term_clusterer():
    """Return a TermClusterer instance for unit tests."""
    return TermClusterer()


@pytest.mark.unit
class TestTermClustering:
    """Unit tests for term clustering algorithms (edit distance, star clustering)."""
    def test_edit_distance(self, term_clusterer):
        """Compute edit distance (Levenshtein) for two short words."""
        dist = term_clusterer.edit_distance("cat", "hat")
        assert dist == 1

    def test_star_clustering(self, term_clusterer):
        """Group similar terms with star clustering using a similarity threshold."""
        terms = ["color", "colour", "paint"]
        clusters = term_clusterer.star_clustering(terms, similarity_threshold=0.7)
        assert len(clusters) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
