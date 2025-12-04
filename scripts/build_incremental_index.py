#!/usr/bin/env python
"""
Build Incremental Index - CLI Tool

Command-line interface for building and updating the search index
incrementally from news article JSONL files.

Usage:
    # Build from all files in directory
    python scripts/build_incremental_index.py --data-dir data/raw

    # Build from specific pattern
    python scripts/build_incremental_index.py --data-dir data/raw --pattern "*_14days.jsonl"

    # Limit documents
    python scripts/build_incremental_index.py --data-dir data/raw --limit 10000

    # Resume from checkpoint
    python scripts/build_incremental_index.py --data-dir data/raw --resume

    # Full rebuild
    python scripts/build_incremental_index.py --data-dir data/raw --full-rebuild

Author: Information Retrieval System
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ir.index.incremental_builder import IncrementalIndexBuilder


def setup_logging(verbose: bool = False, log_file: Optional[str] = None):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO

    handlers = [logging.StreamHandler()]

    # Add file handler if log file specified
    if log_file:
        # Create directory if needed
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode='a'))

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def main():
    parser = argparse.ArgumentParser(
        description='Build incremental search index from news articles',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build index from all files
  python %(prog)s --data-dir data/raw

  # Build with specific pattern
  python %(prog)s --data-dir data/raw --pattern "ltn_*.jsonl"

  # Limit to 10000 documents
  python %(prog)s --data-dir data/raw --limit 10000

  # Resume from checkpoint
  python %(prog)s --data-dir data/raw --resume

  # Full rebuild (clear existing)
  python %(prog)s --data-dir data/raw --full-rebuild

  # Disable deduplication
  python %(prog)s --data-dir data/raw --no-dedup
        """
    )

    # Required arguments
    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Directory containing JSONL files'
    )

    # Optional arguments
    parser.add_argument(
        '--index-dir',
        type=str,
        default='data/index',
        help='Directory to store index files (default: data/index)'
    )

    parser.add_argument(
        '--pattern',
        type=str,
        default='*.jsonl',
        help='File pattern to match (default: *.jsonl)'
    )

    parser.add_argument(
        '--limit',
        type=int,
        help='Maximum total documents to index'
    )

    parser.add_argument(
        '--limit-per-file',
        type=int,
        help='Maximum documents per file'
    )

    parser.add_argument(
        '--checkpoint-interval',
        type=int,
        default=1000,
        help='Save checkpoint every N documents (default: 1000)'
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from last checkpoint'
    )

    parser.add_argument(
        '--full-rebuild',
        action='store_true',
        help='Full rebuild (clear existing index)'
    )

    parser.add_argument(
        '--no-dedup',
        action='store_true',
        help='Disable deduplication'
    )

    parser.add_argument(
        '--fuzzy-threshold',
        type=int,
        default=3,
        help='SimHash Hamming distance threshold for fuzzy matching (default: 3)'
    )

    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Verbose output (DEBUG level)'
    )

    args = parser.parse_args()

    # Setup logging with log file in index directory
    log_file = f"{args.index_dir}/build.log"
    setup_logging(args.verbose, log_file=log_file)
    logger = logging.getLogger(__name__)

    # Print header
    print("=" * 80)
    print("Incremental Index Builder")
    print("=" * 80)
    print(f"Data directory: {args.data_dir}")
    print(f"Index directory: {args.index_dir}")
    print(f"Pattern: {args.pattern}")
    print(f"Checkpoint interval: {args.checkpoint_interval}")
    print(f"Deduplication: {'Disabled' if args.no_dedup else f'Enabled (threshold={args.fuzzy_threshold})'}")

    if args.limit:
        print(f"Total limit: {args.limit}")
    if args.limit_per_file:
        print(f"Per-file limit: {args.limit_per_file}")

    print("=" * 80)
    print()

    # Validate data directory
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return 1

    # Check for JSONL files
    jsonl_files = list(data_dir.glob(args.pattern))
    if not jsonl_files:
        logger.error(f"No JSONL files found matching '{args.pattern}' in {data_dir}")
        return 1

    logger.info(f"Found {len(jsonl_files)} JSONL files")

    # Initialize builder
    try:
        builder = IncrementalIndexBuilder(
            index_dir=args.index_dir,
            checkpoint_interval=args.checkpoint_interval,
            use_dedup=not args.no_dedup,
            fuzzy_threshold=args.fuzzy_threshold
        )

        # Handle full rebuild
        if args.full_rebuild:
            logger.warning("Full rebuild requested - clearing existing index")
            index_path = Path(args.index_dir)
            if index_path.exists():
                import shutil
                shutil.rmtree(index_path)
                index_path.mkdir(parents=True)
            logger.info("Index directory cleared")

        # Handle resume
        elif args.resume:
            logger.info("Attempting to resume from checkpoint...")
            if builder.load_checkpoint():
                logger.info("Resumed from checkpoint")
            else:
                logger.info("No checkpoint found, starting fresh")

        # Build index
        logger.info("\nStarting indexing process...")
        start_time = datetime.now()

        stats = builder.build_from_directory(
            directory=args.data_dir,
            pattern=args.pattern,
            limit_per_file=args.limit_per_file,
            total_limit=args.limit
        )

        # Finalize
        logger.info("\nFinalizing index...")
        builder.finalize()

        elapsed = (datetime.now() - start_time).total_seconds()

        # Print final statistics
        print("\n" + "=" * 80)
        print("INDEXING COMPLETE")
        print("=" * 80)
        print(f"\nTime elapsed: {elapsed:.1f}s")
        print(f"\nStatistics:")
        print(f"  Documents processed: {stats['docs_processed']:,}")
        print(f"  Documents indexed: {stats['docs_indexed']:,}")
        print(f"  Duplicates skipped: {stats['docs_duplicates']:,}")
        print(f"  Errors: {stats['docs_errors']:,}")
        print(f"  Duplication rate: {stats['duplication_rate']:.2f}%")
        print(f"\nIndex Statistics:")
        print(f"  Vocabulary size: {stats['vocabulary_size']:,} terms")
        print(f"  Total postings: {stats['total_postings']:,}")
        print(f"  Average document length: {stats['avg_doc_length']:.1f} tokens")

        if not args.no_dedup:
            print(f"\nDeduplication Statistics:")
            print(f"  Exact hashes: {stats['exact_hashes']:,}")
            print(f"  Fuzzy hashes: {stats['fuzzy_hashes']:,}")
            print(f"  Exact duplicates found: {stats['exact_duplicates_found']:,}")
            print(f"  Fuzzy duplicates found: {stats['fuzzy_duplicates_found']:,}")

        print(f"\nIndex saved to: {args.index_dir}")
        print("=" * 80)

        logger.info("Build completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Build failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
