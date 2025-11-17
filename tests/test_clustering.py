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
    return DocumentClusterer()


@pytest.fixture
def sample_docs():
    return {
        0: {"term1": 0.8, "term2": 0.6},
        1: {"term1": 0.7, "term2": 0.5},
        2: {"term3": 0.9, "term4": 0.7},
        3: {"term3": 0.8, "term4": 0.6}
    }


@pytest.mark.unit
class TestDocumentClustering:
    def test_cosine_similarity(self, doc_clusterer):
        vec1 = {"term1": 1.0}
        vec2 = {"term1": 1.0}
        sim = doc_clusterer.cosine_similarity(vec1, vec2)
        assert sim == pytest.approx(1.0)

    def test_hac_clustering(self, doc_clusterer, sample_docs):
        result = doc_clusterer.hierarchical_clustering(sample_docs, k=2)
        assert result.num_clusters == 2
        assert len(result.clusters) == 2

    def test_kmeans_clustering(self, doc_clusterer, sample_docs):
        result = doc_clusterer.kmeans_clustering(sample_docs, k=2, random_seed=42)
        assert result.num_clusters == 2
        assert len(result.assignments) == 4


# Term Clustering Tests
@pytest.fixture
def term_clusterer():
    return TermClusterer()


@pytest.mark.unit
class TestTermClustering:
    def test_edit_distance(self, term_clusterer):
        dist = term_clusterer.edit_distance("cat", "hat")
        assert dist == 1

    def test_star_clustering(self, term_clusterer):
        terms = ["color", "colour", "paint"]
        clusters = term_clusterer.star_clustering(terms, similarity_threshold=0.7)
        assert len(clusters) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
