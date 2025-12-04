#!/usr/bin/env python
"""
Import All JSONL Files to PostgreSQL

Imports all collected news articles from data/raw directory into PostgreSQL.

Usage:
    python scripts/import_all_to_postgres.py

Author: Information Retrieval System
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.postgres_manager import PostgresManager
from src.database.jsonl_importer import JSONLImporter


def main():
    """Main import function."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    print("=" * 70)
    print("Import All News Articles to PostgreSQL")
    print("=" * 70)

    # Initialize database manager
    print("\n1. Connecting to PostgreSQL...")
    try:
        db_manager = PostgresManager(
            host='localhost',
            port=5432,
            database='ir_news',
            user='postgres',
            password='postgres'
        )
        print("   ✓ Connected successfully")
    except Exception as e:
        print(f"   ✗ Connection failed: {e}")
        return 1

    # Create importer
    print("\n2. Initializing importer...")
    importer = JSONLImporter(db_manager, batch_size=1000)
    print("   ✓ Importer ready (batch_size=1000)")

    # Import all files from data/raw
    print("\n3. Starting bulk import from data/raw...")
    print("-" * 70)

    try:
        stats = importer.import_directory(
            directory='data/raw',
            pattern='*.jsonl'
        )

        print("\n4. Import Complete!")
        print("=" * 70)
        print(f"Final Statistics:")
        print(f"  Total processed: {stats['total_processed']:,}")
        print(f"  Total inserted: {stats['total_inserted']:,}")
        print(f"  Duplicates: {stats['total_duplicates']:,}")
        print(f"  Errors: {stats['total_errors']:,}")
        print(f"  Duplication rate: {stats['duplication_rate']:.2f}%")
        print("=" * 70)

        # Get final database statistics
        print("\n5. Database Statistics:")
        print("-" * 70)
        corpus_stats = db_manager.get_corpus_stats()
        print(f"  Total articles: {corpus_stats['total_articles']:,}")
        print(f"  Total sources: {corpus_stats['total_sources']}")
        print(f"  Total categories: {corpus_stats['total_categories']}")

        if corpus_stats.get('source_counts'):
            print("\n  Articles by source:")
            for source, count in sorted(corpus_stats['source_counts'].items(),
                                       key=lambda x: x[1], reverse=True):
                print(f"    {source}: {count:,}")

        if corpus_stats.get('category_counts'):
            print("\n  Articles by category:")
            for category, count in sorted(corpus_stats['category_counts'].items(),
                                         key=lambda x: x[1], reverse=True)[:10]:
                print(f"    {category}: {count:,}")

        print("\n" + "=" * 70)

    except Exception as e:
        logger.error(f"Import failed: {e}", exc_info=True)
        return 1
    finally:
        db_manager.close()
        print("\nDatabase connection closed.")

    return 0


if __name__ == '__main__':
    sys.exit(main())
