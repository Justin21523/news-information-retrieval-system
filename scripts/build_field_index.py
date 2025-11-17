#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Build Field Index for Advanced Metadata Search

This script builds a field-based inverted index from preprocessed documents,
enabling library-style multi-field queries like:
    - title:台灣 AND category:政治
    - date:[2025-11-01 TO 2025-11-13]
    - tags:(AI OR 機器學習)

The field index supports:
    - Field-specific search (title, content, category, tags, date, author)
    - Complex boolean queries (AND, OR, NOT)
    - Date range queries
    - Multi-value field searches (tags)

Usage:
    python scripts/build_field_index.py
    python scripts/build_field_index.py --input data/preprocessed/custom.jsonl
    python scripts/build_field_index.py --output data/indexes/field_index_custom.pkl

Author: Information Retrieval System
Date: 2025-11-17
"""

import json
import pickle
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ir.index.field_indexer import FieldIndexer
from src.ir.text.chinese_tokenizer import ChineseTokenizer


def setup_logging(verbose: bool = False) -> None:
    """
    Setup logging configuration.

    Args:
        verbose: Enable debug logging if True
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/build_field_index.log')
        ]
    )


def load_documents(input_file: Path) -> List[Dict[str, Any]]:
    """
    Load preprocessed documents from JSONL file.

    Args:
        input_file: Path to preprocessed JSONL file

    Returns:
        List of document dictionaries with metadata fields

    Complexity:
        Time: O(N) where N = number of documents
        Space: O(N)
    """
    logging.info(f"Loading documents from {input_file}...")

    documents = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                doc = json.loads(line)
                documents.append(doc)
            except json.JSONDecodeError as e:
                logging.warning(f"Failed to parse line {line_num}: {e}")
                continue

    logging.info(f"Loaded {len(documents)} documents")
    return documents


def build_field_index(
    documents: List[Dict[str, Any]],
    tokenizer: ChineseTokenizer,
    output_file: Path
) -> FieldIndexer:
    """
    Build field index from document collection.

    Args:
        documents: List of document dictionaries
        tokenizer: Chinese tokenizer for text fields
        output_file: Path to save the field index

    Returns:
        Built FieldIndexer instance

    Complexity:
        Time: O(N * F * T) where:
              N = number of documents
              F = number of fields per document
              T = average tokens per field
        Space: O(N * F * T)
    """
    logging.info("Building field index...")

    # Initialize field indexer with tokenizer
    def tokenize_fn(text: str) -> List[str]:
        """Tokenization wrapper for field indexer."""
        return tokenizer.tokenize(text)

    indexer = FieldIndexer(tokenizer=tokenize_fn)

    # Build indexes
    indexer.build(documents)

    # Save to disk
    logging.info(f"Saving field index to {output_file}...")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'wb') as f:
        pickle.dump(indexer, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Print statistics
    stats = indexer._get_index_stats()
    logging.info("=" * 60)
    logging.info("Field Index Statistics:")
    logging.info("=" * 60)
    logging.info(f"Total documents: {indexer.doc_count}")
    logging.info(f"Total fields: {stats['total_fields']}")
    logging.info(f"Total unique terms: {stats['total_terms']}")
    logging.info(f"Total postings: {stats['total_postings']}")
    logging.info("")
    logging.info("Field-wise statistics:")
    for field, field_stats in stats['fields'].items():
        logging.info(f"  {field:15s}: {field_stats['terms']:5d} terms, "
                    f"{field_stats['postings']:6d} postings")
    logging.info("=" * 60)

    # Calculate index size
    index_size_mb = output_file.stat().st_size / (1024 * 1024)
    logging.info(f"Index file size: {index_size_mb:.2f} MB")

    return indexer


def test_field_index(indexer: FieldIndexer) -> None:
    """
    Run basic tests on the built field index.

    Args:
        indexer: Built FieldIndexer instance
    """
    logging.info("\nRunning field index tests...")

    # Test 1: Search title field
    logging.info("\n[Test 1] Searching title field for '台灣':")
    results = indexer.search_field('title', '台灣')
    logging.info(f"  Found {len(results)} documents")
    if results:
        logging.info(f"  Sample doc IDs: {list(results)[:5]}")

    # Test 2: Search category field
    logging.info("\n[Test 2] Searching category field for 'aipl':")
    results = indexer.search_field('category', 'aipl')
    logging.info(f"  Found {len(results)} documents")

    # Test 3: Date range query
    logging.info("\n[Test 3] Date range query [2025-11-10 TO 2025-11-13]:")
    results = indexer.search_date_range('published_date', '2025-11-10', '2025-11-13')
    logging.info(f"  Found {len(results)} documents")

    # Test 4: Multi-term search
    logging.info("\n[Test 4] Multi-term search in title (台灣 OR 中國):")
    results = indexer.search_multi_terms('title', ['台灣', '中國'], operator='OR')
    logging.info(f"  Found {len(results)} documents")

    # Test 5: Tags field
    logging.info("\n[Test 5] Searching tags field for '政治':")
    results = indexer.search_field('tags', '政治')
    logging.info(f"  Found {len(results)} documents")

    logging.info("\n✓ All tests completed")


def main():
    """
    Main function for building field index.

    Steps:
        1. Parse command-line arguments
        2. Setup logging
        3. Load preprocessed documents
        4. Initialize tokenizer
        5. Build field index
        6. Save to disk
        7. Run tests
    """
    parser = argparse.ArgumentParser(
        description='Build field-based index for advanced metadata search',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Build index from default preprocessed file
    python scripts/build_field_index.py

    # Build index from custom input file
    python scripts/build_field_index.py --input data/preprocessed/custom.jsonl

    # Save index to custom location
    python scripts/build_field_index.py --output data/indexes/field_index_custom.pkl

    # Enable verbose logging
    python scripts/build_field_index.py --verbose

    # Skip tests after building
    python scripts/build_field_index.py --no-test
        """
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        default='data/preprocessed/cna_mvp_preprocessed.jsonl',
        help='Input JSONL file with preprocessed documents (default: cna_mvp_preprocessed.jsonl)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/indexes/field_index.pkl',
        help='Output pickle file for field index (default: data/indexes/field_index.pkl)'
    )

    parser.add_argument(
        '--tokenizer', '-t',
        type=str,
        choices=['jieba', 'ckip'],
        default='jieba',
        help='Tokenizer to use for text fields (default: jieba)'
    )

    parser.add_argument(
        '--no-test',
        action='store_true',
        help='Skip running tests after building index'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose (debug) logging'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Convert paths
    input_file = project_root / args.input
    output_file = project_root / args.output

    # Validate input file
    if not input_file.exists():
        logging.error(f"Input file not found: {input_file}")
        sys.exit(1)

    logging.info("=" * 60)
    logging.info("Field Index Builder")
    logging.info("=" * 60)
    logging.info(f"Input file:  {input_file}")
    logging.info(f"Output file: {output_file}")
    logging.info(f"Tokenizer:   {args.tokenizer}")
    logging.info("=" * 60)

    try:
        # Load documents
        documents = load_documents(input_file)

        if not documents:
            logging.error("No documents loaded. Exiting.")
            sys.exit(1)

        # Initialize tokenizer
        logging.info(f"Initializing {args.tokenizer} tokenizer...")
        tokenizer = ChineseTokenizer(engine=args.tokenizer)

        # Build field index
        indexer = build_field_index(documents, tokenizer, output_file)

        # Run tests
        if not args.no_test:
            test_field_index(indexer)

        logging.info("\n✓ Field index built successfully!")
        logging.info(f"Index saved to: {output_file}")

    except KeyboardInterrupt:
        logging.warning("\nBuild interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error building field index: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
