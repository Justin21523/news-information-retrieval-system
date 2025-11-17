"""
Term Clustering Algorithms

This module implements clustering algorithms for grouping similar terms
based on string similarity and co-occurrence patterns.

Key Features:
    - String-based clustering (edit distance)
    - Co-occurrence based clustering
    - Star clustering algorithm
    - Term similarity metrics

Reference: "Introduction to Information Retrieval" (Manning et al.)
           Chapter 3: Dictionaries and tolerant retrieval

Author: Information Retrieval System
License: Educational Use
"""

import logging
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

import sys
from pathlib import Path

_parent_dir = Path(__file__).parent.parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))


@dataclass
class TermCluster:
    """
    Term cluster representation.

    Attributes:
        cluster_id: Unique identifier
        terms: List of terms in cluster
        center: Representative term (star center)
    """
    cluster_id: int
    terms: List[str]
    center: Optional[str] = None

    @property
    def size(self) -> int:
        return len(self.terms)


class TermClusterer:
    """
    Term Clustering Engine.

    Clusters similar terms based on:
    - String similarity (edit distance)
    - Co-occurrence patterns

    Algorithms:
    - Star clustering
    - Edit distance clustering

    Complexity:
        - Edit distance clustering: O(n²) for pairwise distances
        - Star clustering: O(n²) for similarity matrix
    """

    def __init__(self):
        """Initialize term clusterer."""
        self.logger = logging.getLogger(__name__)
        self.logger.info("TermClusterer initialized")

    # ========================================================================
    # String Similarity Metrics
    # ========================================================================

    def edit_distance(self, str1: str, str2: str) -> int:
        """
        Calculate edit distance (Levenshtein distance) between two strings.

        Edit distance is the minimum number of single-character edits
        (insertions, deletions, substitutions) to transform str1 into str2.

        Args:
            str1: First string
            str2: Second string

        Returns:
            Edit distance (integer)

        Complexity:
            Time: O(m × n) where m, n are string lengths
            Space: O(m × n) for DP table

        Examples:
            >>> clusterer = TermClusterer()
            >>> clusterer.edit_distance("kitten", "sitting")
            3
            >>> clusterer.edit_distance("hello", "hello")
            0
        """
        m, n = len(str1), len(str2)

        # Create DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if str1[i-1] == str2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],    # deletion
                        dp[i][j-1],    # insertion
                        dp[i-1][j-1]   # substitution
                    )

        return dp[m][n]

    def normalized_edit_distance(self, str1: str, str2: str) -> float:
        """
        Normalized edit distance in [0, 1].

        Args:
            str1: First string
            str2: Second string

        Returns:
            Normalized distance (0 = identical, 1 = completely different)
        """
        max_len = max(len(str1), len(str2))
        if max_len == 0:
            return 0.0

        return self.edit_distance(str1, str2) / max_len

    def prefix_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate prefix similarity (length of common prefix / avg length).

        Args:
            str1: First string
            str2: Second string

        Returns:
            Prefix similarity in [0, 1]
        """
        min_len = min(len(str1), len(str2))
        common_prefix = 0

        for i in range(min_len):
            if str1[i] == str2[i]:
                common_prefix += 1
            else:
                break

        avg_len = (len(str1) + len(str2)) / 2
        return common_prefix / avg_len if avg_len > 0 else 0.0

    # ========================================================================
    # Star Clustering
    # ========================================================================

    def star_clustering(self,
                       terms: List[str],
                       similarity_threshold: float = 0.7,
                       similarity_metric: str = 'edit_distance') -> List[TermCluster]:
        """
        Star clustering algorithm.

        Greedily selects "star centers" and assigns similar terms to them.
        Each center forms a cluster with all sufficiently similar terms.

        Algorithm:
        1. Sort terms by potential (number of similar terms)
        2. Select highest potential unclustered term as center
        3. Assign all similar terms to this cluster
        4. Remove clustered terms and repeat

        Args:
            terms: List of terms to cluster
            similarity_threshold: Minimum similarity to join cluster
            similarity_metric: Similarity metric ('edit_distance', 'prefix')

        Returns:
            List of term clusters

        Complexity:
            Time: O(n²) for similarity matrix + O(n²) for clustering
            Space: O(n²) for similarity matrix

        Examples:
            >>> clusterer = TermClusterer()
            >>> terms = ["color", "colour", "colored", "paint", "painted"]
            >>> clusters = clusterer.star_clustering(terms, threshold=0.7)
            >>> len(clusters)
            2
        """
        if not terms:
            return []

        self.logger.info(
            f"Starting star clustering: {len(terms)} terms, "
            f"threshold={similarity_threshold}"
        )

        # Compute similarity matrix
        similarities = self._compute_term_similarities(
            terms, similarity_metric
        )

        # Calculate potential for each term
        potentials = {}
        for term in terms:
            potential = sum(1 for other in terms
                          if similarities.get((term, other), 0) >= similarity_threshold)
            potentials[term] = potential

        # Star clustering
        clusters = []
        clustered_terms = set()
        cluster_id = 0

        while len(clustered_terms) < len(terms):
            # Find unclustered term with highest potential
            best_center = None
            best_potential = -1

            for term in terms:
                if term not in clustered_terms:
                    if potentials[term] > best_potential:
                        best_potential = potentials[term]
                        best_center = term

            if best_center is None:
                break

            # Create cluster with this center
            cluster_terms = [best_center]
            clustered_terms.add(best_center)

            # Add similar terms
            for term in terms:
                if term not in clustered_terms:
                    sim = similarities.get((best_center, term), 0)
                    if sim >= similarity_threshold:
                        cluster_terms.append(term)
                        clustered_terms.add(term)

            clusters.append(TermCluster(
                cluster_id=cluster_id,
                terms=cluster_terms,
                center=best_center
            ))
            cluster_id += 1

        self.logger.info(f"Star clustering completed: {len(clusters)} clusters")
        return clusters

    def _compute_term_similarities(self, terms: List[str],
                                  metric: str) -> Dict[Tuple[str, str], float]:
        """
        Compute pairwise term similarities.

        Args:
            terms: List of terms
            metric: Similarity metric

        Returns:
            Dictionary {(term1, term2): similarity}
        """
        similarities = {}

        for i in range(len(terms)):
            for j in range(len(terms)):
                if i == j:
                    similarities[(terms[i], terms[j])] = 1.0
                    continue

                if metric == 'edit_distance':
                    # Convert distance to similarity
                    dist = self.normalized_edit_distance(terms[i], terms[j])
                    sim = 1.0 - dist
                elif metric == 'prefix':
                    sim = self.prefix_similarity(terms[i], terms[j])
                else:
                    raise ValueError(f"Unknown metric: {metric}")

                similarities[(terms[i], terms[j])] = sim

        return similarities

    # ========================================================================
    # Simple Edit Distance Clustering
    # ========================================================================

    def edit_distance_clustering(self,
                                 terms: List[str],
                                 max_distance: int = 2) -> List[TermCluster]:
        """
        Simple edit distance clustering.

        Groups terms with edit distance <= max_distance.
        Uses greedy approach: first unclustered term becomes center.

        Args:
            terms: List of terms to cluster
            max_distance: Maximum edit distance to join cluster

        Returns:
            List of term clusters

        Complexity:
            Time: O(n² × m) where n=terms, m=avg term length
            Space: O(n)

        Examples:
            >>> clusterer = TermClusterer()
            >>> terms = ["cat", "cats", "dog", "dogs"]
            >>> clusters = clusterer.edit_distance_clustering(terms, max_distance=1)
            >>> len(clusters)
            2
        """
        if not terms:
            return []

        self.logger.info(
            f"Starting edit distance clustering: {len(terms)} terms, "
            f"max_distance={max_distance}"
        )

        clusters = []
        clustered_terms = set()
        cluster_id = 0

        for center_term in terms:
            if center_term in clustered_terms:
                continue

            # Create cluster with this term as center
            cluster_terms = [center_term]
            clustered_terms.add(center_term)

            # Add similar terms
            for other_term in terms:
                if other_term not in clustered_terms:
                    dist = self.edit_distance(center_term, other_term)
                    if dist <= max_distance:
                        cluster_terms.append(other_term)
                        clustered_terms.add(other_term)

            clusters.append(TermCluster(
                cluster_id=cluster_id,
                terms=cluster_terms,
                center=center_term
            ))
            cluster_id += 1

        self.logger.info(
            f"Edit distance clustering completed: {len(clusters)} clusters"
        )
        return clusters

    # ========================================================================
    # Co-occurrence Based Clustering
    # ========================================================================

    def cooccurrence_clustering(self,
                               terms: List[str],
                               documents: List[Set[str]],
                               min_cooccurrence: int = 2) -> List[TermCluster]:
        """
        Cluster terms based on co-occurrence in documents.

        Terms that frequently appear together are grouped into clusters.

        Args:
            terms: List of terms to cluster
            documents: List of document term sets
            min_cooccurrence: Minimum co-occurrence count

        Returns:
            List of term clusters

        Complexity:
            Time: O(n² × d) where n=terms, d=documents
            Space: O(n²) for co-occurrence matrix
        """
        if not terms or not documents:
            return []

        self.logger.info(
            f"Starting co-occurrence clustering: {len(terms)} terms, "
            f"{len(documents)} documents"
        )

        # Build co-occurrence matrix
        cooccurrence = defaultdict(int)

        for doc_terms in documents:
            doc_term_list = [t for t in terms if t in doc_terms]
            for i in range(len(doc_term_list)):
                for j in range(i + 1, len(doc_term_list)):
                    t1, t2 = doc_term_list[i], doc_term_list[j]
                    cooccurrence[(t1, t2)] += 1
                    cooccurrence[(t2, t1)] += 1

        # Cluster terms with high co-occurrence
        clusters = []
        clustered_terms = set()
        cluster_id = 0

        for term in terms:
            if term in clustered_terms:
                continue

            # Find terms that co-occur with this term
            cluster_terms = [term]
            clustered_terms.add(term)

            for other_term in terms:
                if other_term not in clustered_terms:
                    count = cooccurrence.get((term, other_term), 0)
                    if count >= min_cooccurrence:
                        cluster_terms.append(other_term)
                        clustered_terms.add(other_term)

            clusters.append(TermCluster(
                cluster_id=cluster_id,
                terms=cluster_terms,
                center=term
            ))
            cluster_id += 1

        self.logger.info(
            f"Co-occurrence clustering completed: {len(clusters)} clusters"
        )
        return clusters


def demo():
    """Demonstration of term clustering."""
    print("=" * 60)
    print("Term Clustering Demo")
    print("=" * 60)

    # Sample terms
    terms = [
        "color", "colour", "colored",
        "search", "searching", "searched",
        "database", "databases",
        "information", "informations"
    ]

    clusterer = TermClusterer()

    # Example 1: Edit distance
    print("\n1. Edit Distance Examples:")
    print(f"   edit_distance('color', 'colour') = {clusterer.edit_distance('color', 'colour')}")
    print(f"   edit_distance('search', 'searching') = {clusterer.edit_distance('search', 'searching')}")
    print(f"   normalized_edit_distance('color', 'colour') = {clusterer.normalized_edit_distance('color', 'colour'):.3f}")

    # Example 2: Star clustering
    print("\n2. Star Clustering:")
    star_clusters = clusterer.star_clustering(terms, similarity_threshold=0.6)

    for cluster in star_clusters:
        print(f"   Cluster {cluster.cluster_id} (center: {cluster.center}):")
        print(f"      Terms: {cluster.terms}")

    # Example 3: Edit distance clustering
    print("\n3. Edit Distance Clustering (max_distance=2):")
    edit_clusters = clusterer.edit_distance_clustering(terms, max_distance=2)

    for cluster in edit_clusters:
        print(f"   Cluster {cluster.cluster_id}: {cluster.terms}")

    # Example 4: Co-occurrence clustering
    print("\n4. Co-occurrence Clustering:")
    documents = [
        {"color", "information", "search"},
        {"colour", "information", "database"},
        {"colored", "search", "database"},
        {"searching", "databases", "information"}
    ]

    cooccur_clusters = clusterer.cooccurrence_clustering(
        terms, documents, min_cooccurrence=1
    )

    for cluster in cooccur_clusters:
        print(f"   Cluster {cluster.cluster_id}: {cluster.terms}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    demo()
