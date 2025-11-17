#!/usr/bin/env python3
"""Document Clustering CLI Tool

Command-line tool for clustering documents using various algorithms.

Usage:
    # Hierarchical clustering
    python scripts/cluster_docs.py --index vsm_index.json --algorithm hac --k 5

    # K-means clustering
    python scripts/cluster_docs.py --index vsm_index.json --algorithm kmeans --k 5

Author: Information Retrieval System
"""

import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ir.cluster.doc_cluster import DocumentClusterer
from src.ir.index.inverted_index import InvertedIndex
from src.ir.index.term_weighting import TermWeighting


def main():
    parser = argparse.ArgumentParser(description='Document Clustering')
    parser.add_argument('--index', type=str, required=True,
                       help='Path to inverted index file')
    parser.add_argument('--algorithm', type=str, required=True,
                       choices=['hac', 'kmeans'],
                       help='Clustering algorithm')
    parser.add_argument('--k', type=int, required=True,
                       help='Number of clusters')
    parser.add_argument('--linkage', type=str, default='complete',
                       choices=['single', 'complete', 'average'],
                       help='Linkage for HAC (default: complete)')
    parser.add_argument('--max-iterations', type=int, default=100,
                       help='Max iterations for K-means (default: 100)')
    parser.add_argument('--seed', type=int,
                       help='Random seed for K-means')
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed output')

    args = parser.parse_args()

    # Load index
    print(f"Loading index from {args.index}...")
    inv_index = InvertedIndex()
    inv_index.load(args.index)

    # Build document vectors
    print("Building document vectors...")
    tw = TermWeighting()
    tw.build_from_index(inv_index)

    documents = {}
    for doc_id in range(inv_index.doc_count):
        doc_tf = {}
        for term in inv_index.vocabulary:
            tf = inv_index.term_frequency(term, doc_id)
            if tf > 0:
                doc_tf[term] = tf

        doc_vec = tw.vectorize(doc_tf, tf_scheme='l', idf_scheme='t', normalize='c')
        documents[doc_id] = doc_vec

    print(f"  {len(documents)} document vectors created")

    # Cluster
    clusterer = DocumentClusterer()

    print(f"\nClustering with {args.algorithm.upper()} (k={args.k})...")

    if args.algorithm == 'hac':
        result = clusterer.hierarchical_clustering(
            documents, k=args.k, linkage=args.linkage
        )
    else:  # kmeans
        result = clusterer.kmeans_clustering(
            documents, k=args.k,
            max_iterations=args.max_iterations,
            random_seed=args.seed
        )

    # Display results
    print(f"\nClustering Results:")
    print(f"  Number of clusters: {result.num_clusters}")

    for cluster in result.clusters:
        print(f"\nCluster {cluster.cluster_id} (size={cluster.size}):")
        print(f"  Documents: {cluster.doc_ids[:20]}")
        if len(cluster.doc_ids) > 20:
            print(f"  ... and {len(cluster.doc_ids) - 20} more")

    # Evaluate quality
    score = clusterer.evaluate_clusters(documents, result)
    print(f"\nSilhouette Score: {score:.4f}")


if __name__ == '__main__':
    main()
