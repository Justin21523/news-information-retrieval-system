#!/usr/bin/env python3
"""
Build Indexes from Preprocessed JSONL

This script builds all necessary search indexes from preprocessed JSONL files
that already contain tokens. This is much faster than re-tokenizing.

Builds:
- inverted_index.pkl (倒排索引)
- positional_index.pkl (位置索引)
- tfidf_vectors.pkl (TF-IDF 向量)
- bm25_index.pkl (BM25 索引)
- field_index.pkl (欄位索引)
- doc_id_map.json (文檔 ID 映射)

Usage:
    python scripts/build_indexes_from_preprocessed.py \
        --input data/preprocessed/merged_14days_preprocessed.jsonl \
        --output data/indexes_10k

Author: CNIRS Project
"""

import json
import pickle
import argparse
import logging
import math
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IndexBuilder:
    """
    Build search indexes from preprocessed JSONL with pre-computed tokens.
    """

    def __init__(self):
        # Core indexes
        self.inverted_index = defaultdict(list)  # term -> [(doc_id, tf), ...]
        self.positional_index = defaultdict(list)  # term -> [(doc_id, [positions]), ...]
        self.doc_metadata = {}  # doc_id -> metadata dict

        # ID mappings
        self.article_to_doc = {}  # article_id -> doc_id
        self.doc_to_article = {}  # doc_id -> article_id

        # Statistics for TF-IDF and BM25
        self.doc_count = 0
        self.doc_lengths = {}  # doc_id -> length
        self.total_doc_length = 0
        self.term_doc_freq = defaultdict(int)  # term -> number of docs containing term

        # Field index
        self.field_index = {
            'source': defaultdict(set),
            'category': defaultdict(set),
            'category_name': defaultdict(set),
            'published_date': defaultdict(set),
            'author': defaultdict(set),
        }

    def add_document(self, record: Dict[str, Any]) -> int:
        """
        Add a preprocessed document to the indexes.

        Args:
            record: Preprocessed record with tokens_title, tokens_content

        Returns:
            Assigned doc_id
        """
        doc_id = self.doc_count
        article_id = record.get('article_id', str(doc_id))

        # ID mappings
        self.article_to_doc[article_id] = doc_id
        self.doc_to_article[doc_id] = article_id

        # Get pre-computed tokens
        tokens_title = record.get('tokens_title', [])
        tokens_content = record.get('tokens_content', [])
        all_tokens = tokens_title + tokens_content

        # Store metadata
        self.doc_metadata[doc_id] = {
            'article_id': article_id,
            'title': record.get('title', ''),
            'content': record.get('content', ''),
            'source': record.get('source', ''),
            'category': record.get('category', ''),
            'category_name': record.get('category_name', ''),
            'published_date': record.get('published_date', ''),
            'author': record.get('author', ''),
            'url': record.get('url', ''),
            'length': len(all_tokens)
        }

        # Update statistics
        self.doc_lengths[doc_id] = len(all_tokens)
        self.total_doc_length += len(all_tokens)

        # Build inverted index (term -> doc_id, tf)
        term_freq = defaultdict(int)
        for token in all_tokens:
            term_freq[token] += 1

        for term, tf in term_freq.items():
            self.inverted_index[term].append((doc_id, tf))
            self.term_doc_freq[term] += 1

        # Build positional index
        for pos, token in enumerate(all_tokens):
            # Find existing entry or create new
            found = False
            for entry in self.positional_index[token]:
                if entry[0] == doc_id:
                    entry[1].append(pos)
                    found = True
                    break
            if not found:
                self.positional_index[token].append([doc_id, [pos]])

        # Build field index
        for field in self.field_index.keys():
            value = record.get(field, '')
            if value:
                self.field_index[field][value].add(doc_id)

        self.doc_count += 1
        return doc_id

    def compute_tfidf_vectors(self) -> Dict:
        """
        Compute TF-IDF vectors for all documents.

        Optimized version: iterates through inverted index once (O(P))
        instead of nested loop (O(D×T)).

        Returns:
            TF-IDF data dictionary
        """
        logger.info("Computing TF-IDF vectors...")

        # Calculate IDF for each term
        idf = {}
        for term, df in self.term_doc_freq.items():
            idf[term] = math.log10(self.doc_count / df)

        # Initialize document vectors - O(D)
        document_vectors = {doc_id: {} for doc_id in range(self.doc_count)}

        # Iterate through inverted index once - O(P) where P is total postings
        term_count = 0
        for term, postings in self.inverted_index.items():
            term_idf = idf[term]
            for doc_id, tf in postings:
                # Log TF * IDF
                tf_weight = 1.0 + math.log10(tf) if tf > 0 else 0
                document_vectors[doc_id][term] = tf_weight * term_idf

            term_count += 1
            if term_count % 20000 == 0:
                logger.info(f"  TF-IDF: processed {term_count}/{len(self.inverted_index)} terms")

        # Normalize vectors - O(D)
        logger.info("Normalizing TF-IDF vectors...")
        for doc_id, doc_vector in document_vectors.items():
            norm = math.sqrt(sum(w ** 2 for w in doc_vector.values()))
            if norm > 0:
                document_vectors[doc_id] = {t: w / norm for t, w in doc_vector.items()}

        return {
            'idf': idf,
            'document_vectors': document_vectors,
            'doc_count': self.doc_count
        }

    def compute_bm25_index(self, k1: float = 1.5, b: float = 0.75) -> Dict:
        """
        Compute BM25 index data.

        Args:
            k1: Term frequency saturation parameter
            b: Length normalization parameter

        Returns:
            BM25 index data dictionary
        """
        logger.info("Computing BM25 index...")

        avg_doc_length = self.total_doc_length / self.doc_count if self.doc_count > 0 else 0

        # Calculate IDF for BM25 (Robertson-Spärck Jones formula)
        idf = {}
        for term, df in self.term_doc_freq.items():
            # IDF = log((N - df + 0.5) / (df + 0.5))
            idf[term] = math.log((self.doc_count - df + 0.5) / (df + 0.5))

        return {
            'idf': idf,
            'k1': k1,
            'b': b,
            'avg_doc_length': avg_doc_length,
            'doc_count': self.doc_count
        }

    def save_indexes(self, output_dir: Path):
        """
        Save all indexes to disk.

        Args:
            output_dir: Output directory
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save inverted index
        inverted_data = {
            'index': dict(self.inverted_index),
            'doc_metadata': self.doc_metadata
        }
        with open(output_dir / 'inverted_index.pkl', 'wb') as f:
            pickle.dump(inverted_data, f)
        logger.info(f"Saved inverted_index.pkl ({len(self.inverted_index)} terms)")

        # Save positional index
        positional_data = {
            'index': {k: [tuple(v) if isinstance(v, list) else v for v in vals]
                     for k, vals in self.positional_index.items()}
        }
        with open(output_dir / 'positional_index.pkl', 'wb') as f:
            pickle.dump(positional_data, f)
        logger.info(f"Saved positional_index.pkl")

        # Save TF-IDF vectors
        tfidf_data = self.compute_tfidf_vectors()
        with open(output_dir / 'tfidf_vectors.pkl', 'wb') as f:
            pickle.dump(tfidf_data, f)
        logger.info(f"Saved tfidf_vectors.pkl")

        # Save BM25 index
        bm25_data = self.compute_bm25_index()
        with open(output_dir / 'bm25_index.pkl', 'wb') as f:
            pickle.dump(bm25_data, f)
        logger.info(f"Saved bm25_index.pkl")

        # Save field index
        field_data = {
            field: {k: list(v) for k, v in index.items()}
            for field, index in self.field_index.items()
        }
        with open(output_dir / 'field_index.pkl', 'wb') as f:
            pickle.dump(field_data, f)
        logger.info(f"Saved field_index.pkl")

        # Save doc ID map
        doc_map = {
            'article_to_doc': self.article_to_doc,
            'doc_to_article': {str(k): v for k, v in self.doc_to_article.items()}
        }
        with open(output_dir / 'doc_id_map.json', 'w', encoding='utf-8') as f:
            json.dump(doc_map, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved doc_id_map.json ({self.doc_count} documents)")

    def get_stats(self) -> Dict:
        """Get index statistics."""
        avg_doc_length = self.total_doc_length / self.doc_count if self.doc_count > 0 else 0

        return {
            'doc_count': self.doc_count,
            'vocabulary_size': len(self.inverted_index),
            'total_postings': sum(len(p) for p in self.inverted_index.values()),
            'avg_doc_length': avg_doc_length,
            'sources': len(self.field_index['source']),
            'categories': len(self.field_index['category_name'])
        }


def build_indexes(input_file: Path, output_dir: Path, limit: int = None) -> Dict:
    """
    Build all indexes from preprocessed JSONL.

    Args:
        input_file: Input preprocessed JSONL file
        output_dir: Output directory for indexes
        limit: Optional limit on documents

    Returns:
        Statistics dictionary
    """
    builder = IndexBuilder()

    logger.info(f"Reading from {input_file}...")
    start_time = time.time()

    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f):
            if limit and line_num >= limit:
                break

            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
                builder.add_document(record)

                if (line_num + 1) % 1000 == 0:
                    elapsed = time.time() - start_time
                    rate = (line_num + 1) / elapsed
                    logger.info(f"Processed {line_num + 1} documents ({rate:.1f} docs/sec)")

            except json.JSONDecodeError as e:
                logger.warning(f"JSON error at line {line_num + 1}: {e}")
            except Exception as e:
                logger.warning(f"Error at line {line_num + 1}: {e}")

    elapsed = time.time() - start_time
    logger.info(f"Read {builder.doc_count} documents in {elapsed:.1f} seconds")

    # Save indexes
    logger.info(f"Saving indexes to {output_dir}...")
    builder.save_indexes(output_dir)

    return builder.get_stats()


def main():
    parser = argparse.ArgumentParser(
        description='Build indexes from preprocessed JSONL',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        default='data/preprocessed/merged_14days_preprocessed.jsonl',
        help='Input preprocessed JSONL file'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/indexes_10k',
        help='Output directory for indexes'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of documents to process'
    )

    args = parser.parse_args()

    input_file = Path(args.input)
    output_dir = Path(args.output)

    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return 1

    logger.info("=" * 60)
    logger.info("Build Indexes from Preprocessed Data")
    logger.info("=" * 60)
    logger.info(f"Input: {input_file}")
    logger.info(f"Output: {output_dir}")
    if args.limit:
        logger.info(f"Limit: {args.limit}")
    logger.info("")

    # Build indexes
    stats = build_indexes(input_file, output_dir, limit=args.limit)

    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("INDEX BUILD COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Documents indexed: {stats['doc_count']}")
    logger.info(f"Vocabulary size: {stats['vocabulary_size']:,}")
    logger.info(f"Total postings: {stats['total_postings']:,}")
    logger.info(f"Avg doc length: {stats['avg_doc_length']:.1f} tokens")
    logger.info(f"Sources: {stats['sources']}")
    logger.info(f"Categories: {stats['categories']}")
    logger.info(f"Output: {output_dir}")

    return 0


if __name__ == '__main__':
    exit(main())
