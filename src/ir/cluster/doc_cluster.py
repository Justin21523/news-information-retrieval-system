"""
Document Clustering Algorithms

This module implements clustering algorithms for grouping similar documents.
Supports hierarchical (single-link, complete-link, average-link) and
flat (K-means) clustering methods.

Key Features:
    - Hierarchical Agglomerative Clustering (HAC)
    - K-means clustering
    - Multiple distance/similarity metrics
    - Dendrogram visualization support
    - Cluster quality evaluation

Reference: "Introduction to Information Retrieval" (Manning et al.)
           Chapter 17: Hierarchical Clustering
           Chapter 16: Flat Clustering

Author: Information Retrieval System
License: Educational Use
"""

import logging
import math
import random
from typing import List, Dict, Set, Tuple, Optional, Union
from dataclasses import dataclass
from collections import defaultdict

import sys
from pathlib import Path

_parent_dir = Path(__file__).parent.parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))


@dataclass
class Cluster:
    """
    Cluster representation.

    Attributes:
        cluster_id: Unique cluster identifier
        doc_ids: List of document IDs in this cluster
        centroid: Cluster centroid (for flat clustering)
        size: Number of documents
    """
    cluster_id: int
    doc_ids: List[int]
    centroid: Optional[Dict[str, float]] = None

    @property
    def size(self) -> int:
        return len(self.doc_ids)


@dataclass
class ClusteringResult:
    """
    Result of clustering.

    Attributes:
        clusters: List of clusters
        num_clusters: Number of clusters
        assignments: Document to cluster mapping
        quality_score: Optional quality metric (silhouette, etc.)
    """
    clusters: List[Cluster]
    num_clusters: int
    assignments: Dict[int, int]  # doc_id -> cluster_id
    quality_score: Optional[float] = None


class DocumentClusterer:
    """
    Document Clustering Engine.

    Supports multiple clustering algorithms:
    - Hierarchical: Single-link, Complete-link, Average-link
    - Flat: K-means

    Complexity:
        - HAC: O(n³) naive, O(n² log n) with priority queue
        - K-means: O(k × n × i × d) where k=clusters, i=iterations, d=dimensions
    """

    def __init__(self):
        """Initialize document clusterer."""
        self.logger = logging.getLogger(__name__)
        self._tokenizer = None
        self.logger.info("DocumentClusterer initialized")

    def _default_tokenizer(self, text: str) -> List[str]:
        """Default tokenization for Chinese text."""
        import re
        # Simple tokenization - split on whitespace and punctuation
        tokens = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z0-9]+', text.lower())
        return tokens

    # ========================================================================
    # Distance/Similarity Metrics
    # ========================================================================

    def cosine_similarity(self, vec1: Dict[str, float],
                         vec2: Dict[str, float]) -> float:
        """
        Calculate cosine similarity between two document vectors.

        Args:
            vec1: First document vector
            vec2: Second document vector

        Returns:
            Similarity score in [0, 1]

        Complexity:
            Time: O(min(|v1|, |v2|))
        """
        dot_product = sum(vec1.get(term, 0) * vec2.get(term, 0)
                         for term in set(vec1.keys()) | set(vec2.keys()))

        if dot_product == 0:
            return 0.0

        norm1 = math.sqrt(sum(w**2 for w in vec1.values()))
        norm2 = math.sqrt(sum(w**2 for w in vec2.values()))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def euclidean_distance(self, vec1: Dict[str, float],
                          vec2: Dict[str, float]) -> float:
        """
        Calculate Euclidean distance between two vectors.

        Args:
            vec1: First document vector
            vec2: Second document vector

        Returns:
            Distance value (lower is more similar)

        Complexity:
            Time: O(|v1| + |v2|)
        """
        all_terms = set(vec1.keys()) | set(vec2.keys())
        sum_sq_diff = sum((vec1.get(term, 0) - vec2.get(term, 0))**2
                         for term in all_terms)
        return math.sqrt(sum_sq_diff)

    def jaccard_similarity(self, vec1: Dict[str, float],
                          vec2: Dict[str, float]) -> float:
        """
        Calculate Jaccard similarity based on non-zero terms.

        Args:
            vec1: First document vector
            vec2: Second document vector

        Returns:
            Similarity score in [0, 1]
        """
        terms1 = set(vec1.keys())
        terms2 = set(vec2.keys())

        if not terms1 and not terms2:
            return 1.0

        intersection = len(terms1 & terms2)
        union = len(terms1 | terms2)

        return intersection / union if union > 0 else 0.0

    def cluster(self, documents: List[str], n_clusters: int = 3,
                method: str = 'hierarchical', **kwargs) -> List[Cluster]:
        """
        Simplified clustering interface that works with raw text documents.

        Args:
            documents: List of text documents
            n_clusters: Number of clusters (default 3)
            method: Clustering method ('hierarchical' or 'kmeans')
            **kwargs: Additional arguments for clustering

        Returns:
            List of Cluster objects

        Examples:
            >>> clusterer = DocumentClusterer()
            >>> corpus = ["text 1", "text 2", "text 3"]
            >>> clusters = clusterer.cluster(corpus, n_clusters=2)
            >>> len(clusters)
            2
        """
        from collections import Counter

        # Vectorize documents using simple TF representation
        doc_vectors = {}
        for doc_id, doc_text in enumerate(documents):
            tokens = self._default_tokenizer(doc_text)
            tf = Counter(tokens)
            # Convert to float dict
            doc_vectors[doc_id] = {term: float(count) for term, count in tf.items()}

        # Cluster using the specified method
        if method == 'hierarchical':
            result = self.hierarchical_clustering(
                doc_vectors,
                k=n_clusters,
                linkage=kwargs.get('linkage', 'complete'),
                similarity_metric=kwargs.get('similarity_metric', 'cosine')
            )
        elif method == 'kmeans':
            result = self.kmeans_clustering(
                doc_vectors,
                k=n_clusters,
                max_iterations=kwargs.get('max_iterations', 100)
            )
        else:
            raise ValueError(f"Unknown clustering method: {method}")

        return result.clusters

    # ========================================================================
    # Hierarchical Agglomerative Clustering (HAC)
    # ========================================================================

    def hierarchical_clustering(self,
                                documents: Dict[int, Dict[str, float]],
                                k: int,
                                linkage: str = 'complete',
                                similarity_metric: str = 'cosine') -> ClusteringResult:
        """
        Hierarchical Agglomerative Clustering.

        Builds a hierarchy of clusters bottom-up, merging most similar
        clusters until k clusters remain.

        Args:
            documents: Dictionary {doc_id: vector}
            k: Number of clusters
            linkage: Linkage criterion ('single', 'complete', 'average')
            similarity_metric: Similarity metric ('cosine', 'euclidean', 'jaccard')

        Returns:
            ClusteringResult with k clusters

        Complexity:
            Time: O(n² log n) with priority queue
            Space: O(n²) for similarity matrix

        Examples:
            >>> clusterer = DocumentClusterer()
            >>> docs = {
            ...     0: {"term1": 0.8, "term2": 0.6},
            ...     1: {"term1": 0.7, "term3": 0.5},
            ...     2: {"term4": 0.9}
            ... }
            >>> result = clusterer.hierarchical_clustering(docs, k=2)
            >>> result.num_clusters
            2
        """
        if k >= len(documents):
            # Each document is its own cluster
            clusters = [Cluster(cluster_id=i, doc_ids=[doc_id])
                       for i, doc_id in enumerate(documents.keys())]
            assignments = {doc_id: i for i, doc_id in enumerate(documents.keys())}
            return ClusteringResult(
                clusters=clusters,
                num_clusters=len(clusters),
                assignments=assignments
            )

        self.logger.info(
            f"Starting HAC: {len(documents)} docs → {k} clusters ({linkage} linkage)"
        )

        # Initialize: each document is a cluster
        clusters = {i: {doc_id} for i, doc_id in enumerate(documents.keys())}
        cluster_id_counter = len(documents)

        # Build similarity matrix
        sim_func = self._get_similarity_function(similarity_metric)
        similarities = self._compute_pairwise_similarities(documents, sim_func)

        # Merge until k clusters remain
        while len(clusters) > k:
            # Find most similar pair of clusters
            best_pair = None
            best_sim = -float('inf')

            cluster_ids = list(clusters.keys())
            for i in range(len(cluster_ids)):
                for j in range(i + 1, len(cluster_ids)):
                    c1_id, c2_id = cluster_ids[i], cluster_ids[j]
                    sim = self._cluster_similarity(
                        clusters[c1_id], clusters[c2_id],
                        similarities, linkage
                    )

                    if sim > best_sim:
                        best_sim = sim
                        best_pair = (c1_id, c2_id)

            if best_pair is None:
                break

            # Merge clusters
            c1_id, c2_id = best_pair
            new_cluster = clusters[c1_id] | clusters[c2_id]

            del clusters[c1_id]
            del clusters[c2_id]
            clusters[cluster_id_counter] = new_cluster
            cluster_id_counter += 1

        # Build result
        result_clusters = []
        assignments = {}

        for cluster_id, doc_set in enumerate(clusters.values()):
            doc_list = sorted(doc_set)
            result_clusters.append(
                Cluster(cluster_id=cluster_id, doc_ids=doc_list)
            )
            for doc_id in doc_list:
                assignments[doc_id] = cluster_id

        self.logger.info(f"HAC completed: {len(result_clusters)} clusters")

        return ClusteringResult(
            clusters=result_clusters,
            num_clusters=len(result_clusters),
            assignments=assignments
        )

    def _get_similarity_function(self, metric: str):
        """Get similarity function by name."""
        if metric == 'cosine':
            return self.cosine_similarity
        elif metric == 'euclidean':
            # Convert distance to similarity
            return lambda v1, v2: 1.0 / (1.0 + self.euclidean_distance(v1, v2))
        elif metric == 'jaccard':
            return self.jaccard_similarity
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def _compute_pairwise_similarities(self, documents: Dict[int, Dict[str, float]],
                                      sim_func) -> Dict[Tuple[int, int], float]:
        """Compute pairwise document similarities."""
        similarities = {}
        doc_ids = list(documents.keys())

        for i in range(len(doc_ids)):
            for j in range(i + 1, len(doc_ids)):
                doc1_id, doc2_id = doc_ids[i], doc_ids[j]
                sim = sim_func(documents[doc1_id], documents[doc2_id])
                similarities[(doc1_id, doc2_id)] = sim
                similarities[(doc2_id, doc1_id)] = sim

        return similarities

    def _cluster_similarity(self, cluster1: Set[int], cluster2: Set[int],
                           similarities: Dict[Tuple[int, int], float],
                           linkage: str) -> float:
        """
        Calculate similarity between two clusters.

        Args:
            cluster1: Set of doc IDs in cluster 1
            cluster2: Set of doc IDs in cluster 2
            similarities: Pairwise document similarities
            linkage: Linkage criterion

        Returns:
            Cluster similarity score
        """
        sims = []
        for doc1 in cluster1:
            for doc2 in cluster2:
                if (doc1, doc2) in similarities:
                    sims.append(similarities[(doc1, doc2)])

        if not sims:
            return 0.0

        if linkage == 'single':
            # Single-link: maximum similarity
            return max(sims)
        elif linkage == 'complete':
            # Complete-link: minimum similarity
            return min(sims)
        elif linkage == 'average':
            # Average-link: mean similarity
            return sum(sims) / len(sims)
        else:
            raise ValueError(f"Unknown linkage: {linkage}")

    # ========================================================================
    # K-means Clustering
    # ========================================================================

    def kmeans_clustering(self,
                         documents: Dict[int, Dict[str, float]],
                         k: int,
                         max_iterations: int = 100,
                         tolerance: float = 1e-4,
                         random_seed: Optional[int] = None) -> ClusteringResult:
        """
        K-means clustering.

        Iteratively assigns documents to nearest centroid and updates
        centroids until convergence.

        Args:
            documents: Dictionary {doc_id: vector}
            k: Number of clusters
            max_iterations: Maximum iterations (default 100)
            tolerance: Convergence threshold (default 1e-4)
            random_seed: Random seed for reproducibility

        Returns:
            ClusteringResult with k clusters

        Complexity:
            Time: O(k × n × i × d) where i=iterations, d=dimensions
            Space: O(k × d) for centroids

        Examples:
            >>> clusterer = DocumentClusterer()
            >>> result = clusterer.kmeans_clustering(docs, k=3)
            >>> result.num_clusters
            3
        """
        if k >= len(documents):
            # Each document is its own cluster
            clusters = [Cluster(cluster_id=i, doc_ids=[doc_id])
                       for i, doc_id in enumerate(documents.keys())]
            assignments = {doc_id: i for i, doc_id in enumerate(documents.keys())}
            return ClusteringResult(
                clusters=clusters,
                num_clusters=len(clusters),
                assignments=assignments
            )

        self.logger.info(f"Starting K-means: {len(documents)} docs → {k} clusters")

        if random_seed is not None:
            random.seed(random_seed)

        # Initialize centroids randomly
        doc_ids = list(documents.keys())
        initial_doc_ids = random.sample(doc_ids, k)
        centroids = {i: documents[doc_id].copy()
                    for i, doc_id in enumerate(initial_doc_ids)}

        assignments = {}

        for iteration in range(max_iterations):
            # Assignment step
            new_assignments = {}
            for doc_id, doc_vec in documents.items():
                # Find nearest centroid
                best_cluster = 0
                best_sim = -float('inf')

                for cluster_id, centroid in centroids.items():
                    sim = self.cosine_similarity(doc_vec, centroid)
                    if sim > best_sim:
                        best_sim = sim
                        best_cluster = cluster_id

                new_assignments[doc_id] = best_cluster

            # Check convergence
            if new_assignments == assignments:
                self.logger.info(f"K-means converged at iteration {iteration}")
                break

            assignments = new_assignments

            # Update step: recompute centroids
            cluster_docs = defaultdict(list)
            for doc_id, cluster_id in assignments.items():
                cluster_docs[cluster_id].append(documents[doc_id])

            old_centroids = centroids.copy()
            centroids = {}

            for cluster_id in range(k):
                if cluster_id in cluster_docs and cluster_docs[cluster_id]:
                    centroids[cluster_id] = self._compute_centroid(
                        cluster_docs[cluster_id]
                    )
                else:
                    # Empty cluster: reinitialize
                    centroids[cluster_id] = old_centroids.get(
                        cluster_id, documents[random.choice(doc_ids)].copy()
                    )

            # Check centroid change
            max_change = 0.0
            for cluster_id in range(k):
                if cluster_id in old_centroids:
                    change = self._centroid_distance(
                        centroids[cluster_id], old_centroids[cluster_id]
                    )
                    max_change = max(max_change, change)

            if max_change < tolerance:
                self.logger.info(
                    f"K-means converged at iteration {iteration} (change < {tolerance})"
                )
                break

        # Build result
        cluster_docs = defaultdict(list)
        for doc_id, cluster_id in assignments.items():
            cluster_docs[cluster_id].append(doc_id)

        result_clusters = []
        for cluster_id in range(k):
            doc_list = sorted(cluster_docs.get(cluster_id, []))
            result_clusters.append(
                Cluster(
                    cluster_id=cluster_id,
                    doc_ids=doc_list,
                    centroid=centroids.get(cluster_id)
                )
            )

        self.logger.info(f"K-means completed: {k} clusters")

        return ClusteringResult(
            clusters=result_clusters,
            num_clusters=k,
            assignments=assignments
        )

    def _compute_centroid(self, doc_vectors: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Compute centroid of document vectors.

        Args:
            doc_vectors: List of document vectors

        Returns:
            Centroid vector (mean of all documents)
        """
        if not doc_vectors:
            return {}

        centroid = defaultdict(float)
        for doc_vec in doc_vectors:
            for term, weight in doc_vec.items():
                centroid[term] += weight

        # Average
        num_docs = len(doc_vectors)
        return {term: weight / num_docs for term, weight in centroid.items()}

    def _centroid_distance(self, cent1: Dict[str, float],
                          cent2: Dict[str, float]) -> float:
        """Calculate distance between two centroids."""
        return self.euclidean_distance(cent1, cent2)

    # ========================================================================
    # Cluster Quality Evaluation
    # ========================================================================

    def evaluate_clusters(self, documents: Dict[int, Dict[str, float]],
                         result: ClusteringResult) -> float:
        """
        Evaluate clustering quality using silhouette score.

        Silhouette score measures how similar documents are to their own
        cluster compared to other clusters.

        Args:
            documents: Document vectors
            result: Clustering result

        Returns:
            Silhouette score in [-1, 1] (higher is better)

        Complexity:
            Time: O(n²)
        """
        if result.num_clusters <= 1:
            return 0.0

        silhouette_scores = []

        for doc_id, doc_vec in documents.items():
            cluster_id = result.assignments[doc_id]

            # Compute a(i): mean distance to points in same cluster
            same_cluster_docs = result.clusters[cluster_id].doc_ids
            if len(same_cluster_docs) <= 1:
                silhouette_scores.append(0.0)
                continue

            a_i = sum(1 - self.cosine_similarity(doc_vec, documents[other_id])
                     for other_id in same_cluster_docs if other_id != doc_id)
            a_i /= (len(same_cluster_docs) - 1)

            # Compute b(i): mean distance to points in nearest other cluster
            b_i = float('inf')
            for other_cluster in result.clusters:
                if other_cluster.cluster_id == cluster_id:
                    continue

                if not other_cluster.doc_ids:
                    continue

                mean_dist = sum(1 - self.cosine_similarity(doc_vec, documents[other_id])
                               for other_id in other_cluster.doc_ids)
                mean_dist /= len(other_cluster.doc_ids)

                b_i = min(b_i, mean_dist)

            # Silhouette score for this point
            if b_i == float('inf'):
                s_i = 0.0
            else:
                s_i = (b_i - a_i) / max(a_i, b_i)

            silhouette_scores.append(s_i)

        # Mean silhouette score
        return sum(silhouette_scores) / len(silhouette_scores) if silhouette_scores else 0.0


def demo():
    """Demonstration of document clustering."""
    print("=" * 60)
    print("Document Clustering Demo")
    print("=" * 60)

    # Sample documents (simplified vectors)
    documents = {
        0: {"information": 0.8, "retrieval": 0.7, "system": 0.5},
        1: {"information": 0.7, "retrieval": 0.8, "search": 0.4},
        2: {"database": 0.9, "query": 0.7, "sql": 0.6},
        3: {"database": 0.8, "management": 0.7, "system": 0.5},
        4: {"machine": 0.9, "learning": 0.8, "model": 0.7},
        5: {"deep": 0.8, "learning": 0.9, "neural": 0.7},
    }

    clusterer = DocumentClusterer()

    # Example 1: Hierarchical clustering
    print("\n1. Hierarchical Clustering (Complete-link):")
    hac_result = clusterer.hierarchical_clustering(
        documents, k=3, linkage='complete'
    )

    for cluster in hac_result.clusters:
        print(f"   Cluster {cluster.cluster_id}: docs {cluster.doc_ids}")

    # Example 2: K-means clustering
    print("\n2. K-means Clustering:")
    kmeans_result = clusterer.kmeans_clustering(
        documents, k=3, random_seed=42
    )

    for cluster in kmeans_result.clusters:
        print(f"   Cluster {cluster.cluster_id}: docs {cluster.doc_ids} (size={cluster.size})")

    # Example 3: Cluster quality
    print("\n3. Cluster Quality (Silhouette Score):")
    hac_score = clusterer.evaluate_clusters(documents, hac_result)
    kmeans_score = clusterer.evaluate_clusters(documents, kmeans_result)

    print(f"   HAC silhouette: {hac_score:.3f}")
    print(f"   K-means silhouette: {kmeans_score:.3f}")

    print("\n" + "=" * 60)


# Alias for backward compatibility
HierarchicalDocCluster = DocumentClusterer


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    demo()
