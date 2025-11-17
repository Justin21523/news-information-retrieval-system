#!/usr/bin/env python3
"""
Search Index Builder for CNIRS Project

This script builds multiple search indexes from preprocessed news articles:
1. Inverted Index (term -> documents)
2. Positional Index (term -> documents with positions)
3. TF-IDF Vectors (for VSM ranking)
4. BM25 Index (for BM25 ranking)

All indexes are saved to data/indexes/ for use by retrieval systems.

Usage:
    python scripts/build_indexes.py \\
        --input data/preprocessed/cna_mvp_preprocessed.jsonl \\
        --output data/indexes

Complexity:
    Time: O(N*M) where N=number of documents, M=average document length
    Space: O(V + P) where V=vocabulary size, P=total postings

Author: Information Retrieval System
License: Educational Use
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from collections import Counter
from datetime import datetime
import time
import math

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import IR modules
from src.ir.index.inverted_index import InvertedIndex
from src.ir.index.positional_index import PositionalIndex
from src.ir.index.term_weighting import TFIDFWeighting
from src.ir.retrieval.bm25 import BM25Ranker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IndexBuilder:
    """
    Multi-index builder for news retrieval system.

    Builds and saves multiple index types from preprocessed articles.

    Attributes:
        inverted_index: Standard inverted index
        positional_index: Positional inverted index
        tfidf_weighting: TF-IDF weighting calculator
        bm25_ranker: BM25 ranking function
    """

    def __init__(self):
        """Initialize index builder with all indexing components."""
        logger.info("Initializing IndexBuilder")

        # Initialize indexers
        self.inverted_index = InvertedIndex()
        self.positional_index = PositionalIndex()
        self.tfidf_weighting = None  # Will be initialized after building inverted index
        self.bm25_ranker = None       # Will be initialized after building inverted index

        # Statistics
        self.doc_id_map = {}  # article_id -> numeric doc_id
        self.reverse_doc_map = {}  # numeric doc_id -> article_id
        self.doc_count = 0
        self.total_tokens = 0
        self.vocabulary_size = 0

        logger.info("IndexBuilder initialized")

    def load_articles(self, file_path: Path) -> List[Dict]:
        """
        Load preprocessed articles from JSONL file.

        Args:
            file_path: Path to preprocessed JSONL file

        Returns:
            List of article dictionaries
        """
        logger.info(f"Loading articles from {file_path}")
        articles = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    articles.append(json.loads(line))

        logger.info(f"Loaded {len(articles)} articles")
        return articles

    def build_all_indexes(self, articles: List[Dict]) -> Dict:
        """
        Build all indexes from articles.

        Args:
            articles: List of preprocessed article dictionaries

        Returns:
            Dictionary of index statistics
        """
        logger.info("Building all indexes...")
        start_time = time.time()

        # Step 1: Build inverted index
        logger.info("Step 1/4: Building inverted index...")
        self._build_inverted_index(articles)

        # Step 2: Build positional index
        logger.info("Step 2/4: Building positional index...")
        self._build_positional_index(articles)

        # Step 3: Build TF-IDF index
        logger.info("Step 3/4: Computing TF-IDF weights...")
        self._build_tfidf_index()

        # Step 4: Build BM25 index
        logger.info("Step 4/4: Building BM25 index...")
        self._build_bm25_index()

        build_time = time.time() - start_time

        # Collect statistics
        stats = {
            'doc_count': self.doc_count,
            'vocabulary_size': self.vocabulary_size,
            'total_tokens': self.total_tokens,
            'avg_doc_length': self.total_tokens / self.doc_count if self.doc_count > 0 else 0,
            'build_time': build_time,
            'build_time_per_doc': build_time / self.doc_count if self.doc_count > 0 else 0
        }

        logger.info(f"All indexes built in {build_time:.2f} seconds")
        return stats

    def _build_inverted_index(self, articles: List[Dict]):
        """Build standard inverted index."""
        documents = []
        metadata_list = []

        for idx, article in enumerate(articles):
            # Create numeric doc_id
            article_id = article['article_id']
            self.doc_id_map[article_id] = idx
            self.reverse_doc_map[idx] = article_id

            # Get tokens (use preprocessed tokens)
            tokens = article.get('tokens_content', [])

            # Join tokens into document text for inverted index
            doc_text = ' '.join(tokens)
            documents.append(doc_text)

            # Prepare metadata
            metadata = {
                'article_id': article_id,
                'title': article.get('title', ''),
                'url': article.get('url', ''),
                'published_date': article.get('published_date', ''),
                'category': article.get('category', '')
            }
            metadata_list.append(metadata)

            self.total_tokens += len(tokens)

        # Build index
        self.inverted_index.build(documents, metadata=metadata_list)
        self.doc_count = len(articles)
        self.vocabulary_size = len(self.inverted_index.vocabulary)

        logger.info(f"Inverted index built: {self.doc_count} docs, {self.vocabulary_size} terms")

    def _build_positional_index(self, articles: List[Dict]):
        """Build positional inverted index."""
        # Prepare documents as text (positional index will tokenize)
        documents = []

        for article in articles:
            tokens = article.get('tokens_content', [])
            # Join tokens back to text for positional index
            doc_text = ' '.join(tokens)
            documents.append(doc_text)

        # Build index (it will tokenize internally)
        self.positional_index.build(documents)

        logger.info(f"Positional index built: {len(documents)} docs")

    def _build_tfidf_index(self):
        """Build TF-IDF weighting index."""
        # Initialize TF-IDF weighting
        self.tfidf_weighting = TFIDFWeighting()

        # Build from inverted index
        self.tfidf_weighting.build_from_index(self.inverted_index)

        # Compute document vectors manually
        self.document_vectors = {}
        self.doc_norms = {}

        for doc_id in range(self.doc_count):
            # Get document term frequencies from inverted index
            doc_vector = {}
            for term, postings in self.inverted_index.index.items():
                for posting_doc_id, tf in postings:
                    if posting_doc_id == doc_id:
                        doc_vector[term] = tf
                        break

            # Compute TF-IDF vector
            tfidf_vector = {}
            for term, tf_value in doc_vector.items():
                # Use logarithmic TF and standard IDF
                tf_weight = 1.0 + math.log10(tf_value) if tf_value > 0 else 0.0
                idf_weight = self.tfidf_weighting.idf.get(term, 0.0)
                tfidf_vector[term] = tf_weight * idf_weight

            # Compute L2 norm for cosine normalization
            norm = math.sqrt(sum(w ** 2 for w in tfidf_vector.values()))

            # Normalize
            if norm > 0:
                tfidf_vector = {term: weight / norm for term, weight in tfidf_vector.items()}

            self.document_vectors[doc_id] = tfidf_vector
            self.doc_norms[doc_id] = norm

        logger.info(f"TF-IDF vectors computed for {self.doc_count} documents")

    def _build_bm25_index(self):
        """Build BM25 index."""
        # Initialize BM25 ranker
        self.bm25_ranker = BM25Ranker(
            k1=1.5,  # Standard BM25 parameters
            b=0.75
        )

        # Build index from inverted index
        # BM25 uses the same inverted index structure
        self.bm25_ranker.inverted_index = self.inverted_index.index
        self.bm25_ranker.doc_lengths = self.inverted_index.doc_lengths
        self.bm25_ranker.doc_count = self.inverted_index.doc_count

        # Compute average document length
        total_length = sum(self.inverted_index.doc_lengths.values())
        self.bm25_ranker.avg_doc_length = total_length / self.doc_count if self.doc_count > 0 else 0

        # Precompute IDF values
        self.bm25_ranker.idf = {}
        for term, postings in self.inverted_index.index.items():
            df = len(postings)
            idf = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1.0)
            self.bm25_ranker.idf[term] = idf

        logger.info(f"BM25 index built with {len(self.bm25_ranker.idf)} terms")

    def save_indexes(self, output_dir: Path):
        """
        Save all indexes to disk.

        Args:
            output_dir: Output directory for index files
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving indexes to {output_dir}")

        # Save inverted index
        inverted_path = output_dir / 'inverted_index.pkl'
        with open(inverted_path, 'wb') as f:
            pickle.dump({
                'index': self.inverted_index.index,
                'doc_count': self.inverted_index.doc_count,
                'doc_lengths': self.inverted_index.doc_lengths,
                'doc_metadata': self.inverted_index.doc_metadata,
                'vocabulary': list(self.inverted_index.vocabulary)
            }, f)
        logger.info(f"Saved inverted index: {inverted_path} ({inverted_path.stat().st_size / 1024:.1f} KB)")

        # Save positional index
        positional_path = output_dir / 'positional_index.pkl'
        with open(positional_path, 'wb') as f:
            pickle.dump({
                'index': self.positional_index.index,
                'doc_count': self.positional_index.doc_count
            }, f)
        logger.info(f"Saved positional index: {positional_path} ({positional_path.stat().st_size / 1024:.1f} KB)")

        # Save TF-IDF data
        tfidf_path = output_dir / 'tfidf_vectors.pkl'
        with open(tfidf_path, 'wb') as f:
            pickle.dump({
                'document_vectors': self.document_vectors,
                'idf': self.tfidf_weighting.idf,
                'doc_norms': self.doc_norms
            }, f)
        logger.info(f"Saved TF-IDF vectors: {tfidf_path} ({tfidf_path.stat().st_size / 1024:.1f} KB)")

        # Save BM25 data
        bm25_path = output_dir / 'bm25_index.pkl'
        with open(bm25_path, 'wb') as f:
            pickle.dump({
                'idf': self.bm25_ranker.idf,
                'avg_doc_length': self.bm25_ranker.avg_doc_length,
                'k1': self.bm25_ranker.k1,
                'b': self.bm25_ranker.b
            }, f)
        logger.info(f"Saved BM25 index: {bm25_path} ({bm25_path.stat().st_size / 1024:.1f} KB)")

        # Save document ID mappings
        doc_map_path = output_dir / 'doc_id_map.json'
        with open(doc_map_path, 'w', encoding='utf-8') as f:
            json.dump({
                'article_to_doc': self.doc_id_map,
                'doc_to_article': {str(k): v for k, v in self.reverse_doc_map.items()}
            }, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved document ID map: {doc_map_path}")

        logger.info("All indexes saved successfully")

    def save_statistics(self, stats: Dict, output_path: Path):
        """
        Save index statistics report.

        Args:
            stats: Statistics dictionary
            output_path: Output file path
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("INDEX BUILDING STATISTICS\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("COLLECTION STATISTICS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total documents:        {stats['doc_count']}\n")
            f.write(f"Vocabulary size:        {stats['vocabulary_size']:,}\n")
            f.write(f"Total tokens:           {stats['total_tokens']:,}\n")
            f.write(f"Avg document length:    {stats['avg_doc_length']:.1f} tokens\n\n")

            f.write("INDEX BUILD PERFORMANCE\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total build time:       {stats['build_time']:.2f} seconds\n")
            f.write(f"Time per document:      {stats['build_time_per_doc']:.4f} seconds\n")
            f.write(f"Throughput:             {stats['doc_count']/stats['build_time']:.1f} docs/second\n\n")

            f.write("INDEX TYPES CREATED\n")
            f.write("-" * 80 + "\n")
            f.write("✓ Inverted Index (term -> documents)\n")
            f.write("✓ Positional Index (term -> documents + positions)\n")
            f.write("✓ TF-IDF Vectors (for VSM ranking)\n")
            f.write("✓ BM25 Index (for BM25 ranking)\n")

        logger.info(f"Statistics saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Build search indexes from preprocessed news articles',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/build_indexes.py \\
      --input data/preprocessed/cna_mvp_preprocessed.jsonl \\
      --output data/indexes

  # With custom statistics output
  python scripts/build_indexes.py \\
      --input data/preprocessed/cna_mvp_preprocessed.jsonl \\
      --output data/indexes \\
      --stats data/stats/index_stats.txt
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input JSONL file (preprocessed articles)'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for index files'
    )

    parser.add_argument(
        '--stats',
        type=str,
        default=None,
        help='Output path for statistics report (optional)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate paths
    input_path = Path(args.input)
    output_dir = Path(args.output)

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    # Initialize builder
    builder = IndexBuilder()

    # Load articles
    articles = builder.load_articles(input_path)

    # Build all indexes
    stats = builder.build_all_indexes(articles)

    # Save indexes
    builder.save_indexes(output_dir)

    # Save statistics
    if args.stats:
        stats_path = Path(args.stats)
    else:
        stats_path = output_dir.parent / 'stats' / 'index_build_stats.txt'

    builder.save_statistics(stats, stats_path)

    # Print summary
    print("\n" + "=" * 80)
    print("INDEX BUILDING COMPLETED")
    print("=" * 80)
    print(f"Input:           {input_path}")
    print(f"Output:          {output_dir}")
    print(f"Statistics:      {stats_path}")
    print(f"Documents:       {stats['doc_count']}")
    print(f"Vocabulary:      {stats['vocabulary_size']:,} terms")
    print(f"Build time:      {stats['build_time']:.2f} seconds")
    print(f"Throughput:      {stats['doc_count']/stats['build_time']:.1f} docs/second")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
