#!/usr/bin/env python
"""
Import JSONL to PostgreSQL - CLI Tool

Command-line interface for importing news articles from JSONL files
into PostgreSQL database.

Usage:
    # Import all files from directory
    python scripts/import_jsonl_to_postgres.py --data-dir data/raw

    # Import specific file
    python scripts/import_jsonl_to_postgres.py --file data/raw/ltn_14days.jsonl

    # Import with limit
    python scripts/import_jsonl_to_postgres.py --data-dir data/raw --limit 10000

    # Custom database configuration
    python scripts/import_jsonl_to_postgres.py --data-dir data/raw \\
        --db-host localhost --db-name ir_news --db-user postgres

Author: Information Retrieval System
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.postgres_manager import PostgresManager
from src.database.jsonl_importer import JSONLImporter


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    parser = argparse.ArgumentParser(
        description='Import news articles from JSONL files into PostgreSQL',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Import all JSONL files from directory
  python %(prog)s --data-dir data/raw

  # Import specific file
  python %(prog)s --file data/raw/ltn_14days.jsonl

  # Import with limits
  python %(prog)s --data-dir data/raw --limit 10000 --batch-size 500

  # Custom database configuration
  python %(prog)s --data-dir data/raw \\
      --db-host localhost --db-port 5432 --db-name ir_news \\
      --db-user postgres --db-password your_password

  # Create schema first (if not exists)
  python %(prog)s --create-schema --db-name ir_news
        """
    )

    # Data source arguments
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        '--data-dir',
        type=str,
        help='Directory containing JSONL files'
    )
    source_group.add_argument(
        '--file',
        type=str,
        help='Single JSONL file to import'
    )

    # Database configuration
    parser.add_argument(
        '--db-host',
        type=str,
        default='localhost',
        help='PostgreSQL host (default: localhost)'
    )

    parser.add_argument(
        '--db-port',
        type=int,
        default=5432,
        help='PostgreSQL port (default: 5432)'
    )

    parser.add_argument(
        '--db-name',
        type=str,
        default='ir_news',
        help='Database name (default: ir_news)'
    )

    parser.add_argument(
        '--db-user',
        type=str,
        default='postgres',
        help='Database user (default: postgres)'
    )

    parser.add_argument(
        '--db-password',
        type=str,
        default='postgres',
        help='Database password (default: postgres)'
    )

    # Import options
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.jsonl',
        help='File pattern to match (default: *.jsonl)'
    )

    parser.add_argument(
        '--limit',
        type=int,
        help='Maximum total documents to import'
    )

    parser.add_argument(
        '--limit-per-file',
        type=int,
        help='Maximum documents per file'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        help='Batch size for bulk insert (default: 1000)'
    )

    # Schema management
    parser.add_argument(
        '--create-schema',
        action='store_true',
        help='Create database schema if not exists'
    )

    parser.add_argument(
        '--drop-schema',
        action='store_true',
        help='Drop existing schema (WARNING: destroys all data)'
    )

    # Other options
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Verbose output (DEBUG level)'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Print header
    print("=" * 80)
    print("JSONL to PostgreSQL Importer")
    print("=" * 80)
    print(f"Database: {args.db_host}:{args.db_port}/{args.db_name}")
    print(f"User: {args.db_user}")
    print(f"Batch size: {args.batch_size}")
    if args.limit:
        print(f"Total limit: {args.limit}")
    if args.limit_per_file:
        print(f"Per-file limit: {args.limit_per_file}")
    print("=" * 80)
    print()

    # Initialize database manager
    try:
        logger.info("Connecting to PostgreSQL...")
        db_manager = PostgresManager(
            host=args.db_host,
            port=args.db_port,
            database=args.db_name,
            user=args.db_user,
            password=args.db_password
        )

        # Handle schema operations
        if args.drop_schema:
            response = input("⚠️  WARNING: This will delete ALL data. Type 'yes' to confirm: ")
            if response.lower() == 'yes':
                logger.warning("Dropping schema...")
                db_manager.drop_schema()
                logger.info("Schema dropped")
            else:
                logger.info("Drop cancelled")
                return 0

        if args.create_schema:
            logger.info("Creating schema...")
            db_manager.create_schema()
            logger.info("Schema created")
            if not args.data_dir and not args.file:
                # Only schema creation, exit
                db_manager.close()
                return 0

        # Initialize importer
        importer = JSONLImporter(
            db_manager=db_manager,
            batch_size=args.batch_size
        )

        start_time = datetime.now()

        # Import data
        if args.file:
            # Import single file
            filepath = Path(args.file)
            if not filepath.exists():
                logger.error(f"File not found: {filepath}")
                return 1

            logger.info(f"Importing from {filepath}...")
            stats = importer.import_file(str(filepath), limit=args.limit)

        elif args.data_dir:
            # Import directory
            directory = Path(args.data_dir)
            if not directory.exists():
                logger.error(f"Directory not found: {directory}")
                return 1

            # Check for files
            files = list(directory.glob(args.pattern))
            if not files:
                logger.error(f"No files found matching '{args.pattern}' in {directory}")
                return 1

            logger.info(f"Found {len(files)} files to import")
            stats = importer.import_directory(
                directory=str(directory),
                pattern=args.pattern,
                limit_per_file=args.limit_per_file,
                total_limit=args.limit
            )

        elapsed = (datetime.now() - start_time).total_seconds()

        # Print final statistics
        print("\n" + "=" * 80)
        print("IMPORT COMPLETE")
        print("=" * 80)
        print(f"Time elapsed: {elapsed:.1f}s")
        print(f"\nStatistics:")
        print(f"  Documents processed: {stats['total_processed']:,}")
        print(f"  Documents inserted: {stats['total_inserted']:,}")
        print(f"  Duplicates skipped: {stats['total_duplicates']:,}")
        print(f"  Errors: {stats['total_errors']:,}")
        print(f"  Duplication rate: {stats['duplication_rate']:.2f}%")

        # Get corpus statistics
        print(f"\nDatabase Statistics:")
        corpus_stats = db_manager.get_corpus_stats()
        print(f"  Total articles in DB: {corpus_stats['total_articles']:,}")
        print(f"  Total sources: {corpus_stats['total_sources']}")
        if corpus_stats.get('source_counts'):
            print(f"  Source breakdown:")
            for source, count in sorted(corpus_stats['source_counts'].items(), key=lambda x: x[1], reverse=True):
                print(f"      {source}: {count:,}")

        print("=" * 80)

        # Close database
        db_manager.close()
        logger.info("Import completed successfully")
        return 0

    except KeyboardInterrupt:
        logger.warning("\nImport interrupted by user")
        return 1

    except Exception as e:
        logger.error(f"Import failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
