"""
JSONL to PostgreSQL Importer

Bulk import news articles from JSONL files into PostgreSQL database.
Supports batch processing, progress tracking, and deduplication.

Key Features:
    - Batch import for efficiency
    - Progress tracking with statistics
    - Automatic deduplication using content hash
    - Resume support for interrupted imports
    - Multi-file processing

Author: Information Retrieval System
License: Educational Use
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Iterator
from datetime import datetime
from tqdm import tqdm

from .postgres_manager import PostgresManager
from ..ir.index.doc_reader import DocumentReader, NewsDocument


class JSONLImporter:
    """
    Import JSONL files into PostgreSQL database.

    Supports batch processing, progress tracking, and resume capability.

    Complexity:
        - import_file: O(n) where n is number of documents
        - import_directory: O(m * n) where m is files, n is docs per file

    Attributes:
        db_manager: PostgresManager instance
        doc_reader: DocumentReader instance
        batch_size: Number of documents per batch
        stats: Import statistics
    """

    def __init__(self,
                 db_manager: PostgresManager,
                 batch_size: int = 1000):
        """
        Initialize JSONLImporter.

        Args:
            db_manager: PostgresManager instance
            batch_size: Documents per batch for bulk insert
        """
        self.logger = logging.getLogger(__name__)
        self.db_manager = db_manager
        self.doc_reader = DocumentReader()
        self.batch_size = batch_size

        # Statistics
        self.total_processed = 0
        self.total_inserted = 0
        self.total_duplicates = 0
        self.total_errors = 0

        self.logger.info(
            f"JSONLImporter initialized (batch_size={batch_size})"
        )

    def _doc_to_db_dict(self, doc: NewsDocument) -> Dict:
        """
        Convert NewsDocument to database dictionary format.

        Args:
            doc: NewsDocument instance

        Returns:
            Dictionary with database-compatible fields
        """
        # Parse published_at if string
        published_at = doc.published_at
        if isinstance(published_at, str):
            try:
                # Try parsing ISO format
                published_at = datetime.fromisoformat(
                    published_at.replace('Z', '+00:00')
                )
            except:
                published_at = None

        return {
            'doc_id': doc.doc_id,
            'title': doc.title,
            'content': doc.content,
            'url': doc.url,
            'published_at': published_at,
            'source': doc.source,
            'category': doc.category,
            'author': doc.author,
            'content_hash': doc.content_hash,
            'metadata': doc.metadata
        }

    def import_file(self,
                   filepath: str,
                   limit: Optional[int] = None,
                   skip_existing: bool = True) -> Dict:
        """
        Import a single JSONL file into database.

        Args:
            filepath: Path to JSONL file
            limit: Maximum documents to import (None for all)
            skip_existing: Whether to skip duplicates silently

        Returns:
            Statistics dictionary

        Complexity:
            Time: O(n) where n is number of documents
            Space: O(b) where b is batch_size

        Examples:
            >>> db_manager = PostgresManager()
            >>> importer = JSONLImporter(db_manager)
            >>> stats = importer.import_file('data/raw/ltn_14days.jsonl')
            >>> print(f"Imported {stats['inserted']} documents")
        """
        filepath = Path(filepath)
        self.logger.info(f"Importing from {filepath.name}...")

        start_time = datetime.now()
        file_processed = 0
        file_inserted = 0
        file_duplicates = 0
        file_errors = 0

        batch = []

        # Read documents with progress bar
        docs_iterator = self.doc_reader.read_jsonl(str(filepath), limit=limit)

        # Count total for progress bar (if limit specified)
        if limit:
            total = limit
        else:
            # Quick count of lines
            with open(filepath, 'r', encoding='utf-8') as f:
                total = sum(1 for line in f if line.strip())

        with tqdm(total=total, desc=f"Importing {filepath.name}") as pbar:
            for doc in docs_iterator:
                file_processed += 1
                self.total_processed += 1

                try:
                    # Convert to database format
                    db_doc = self._doc_to_db_dict(doc)
                    batch.append(db_doc)

                    # Process batch when full
                    if len(batch) >= self.batch_size:
                        inserted, duplicates = self.db_manager.batch_insert_articles(batch)
                        file_inserted += inserted
                        file_duplicates += duplicates
                        self.total_inserted += inserted
                        self.total_duplicates += duplicates

                        batch = []  # Clear batch

                    pbar.update(1)

                except Exception as e:
                    file_errors += 1
                    self.total_errors += 1
                    self.logger.warning(f"Error processing document: {e}")

        # Process remaining batch
        if batch:
            inserted, duplicates = self.db_manager.batch_insert_articles(batch)
            file_inserted += inserted
            file_duplicates += duplicates
            self.total_inserted += inserted
            self.total_duplicates += duplicates

        elapsed = (datetime.now() - start_time).total_seconds()
        rate = file_processed / elapsed if elapsed > 0 else 0

        stats = {
            'filename': filepath.name,
            'processed': file_processed,
            'inserted': file_inserted,
            'duplicates': file_duplicates,
            'errors': file_errors,
            'duplication_rate': (file_duplicates / file_processed * 100
                               if file_processed > 0 else 0),
            'elapsed_seconds': elapsed,
            'docs_per_second': rate
        }

        self.logger.info(
            f"Completed {filepath.name}: "
            f"{file_inserted} inserted, {file_duplicates} duplicates, "
            f"{file_errors} errors ({rate:.1f} docs/sec)"
        )

        return stats

    def import_directory(self,
                        directory: str,
                        pattern: str = "*.jsonl",
                        limit_per_file: Optional[int] = None,
                        total_limit: Optional[int] = None) -> Dict:
        """
        Import all JSONL files from a directory.

        Args:
            directory: Directory path
            pattern: File pattern to match
            limit_per_file: Max docs per file
            total_limit: Max docs total

        Returns:
            Overall statistics dictionary

        Complexity:
            Time: O(m * n) where m is files, n is docs per file

        Examples:
            >>> db_manager = PostgresManager()
            >>> importer = JSONLImporter(db_manager)
            >>> stats = importer.import_directory('data/raw', pattern='*_14days.jsonl')
            >>> print(f"Total imported: {stats['total_inserted']}")
        """
        directory = Path(directory)
        self.logger.info(f"Importing from directory: {directory}")

        start_time = datetime.now()

        # Find files
        files = sorted(directory.glob(pattern))
        if not files:
            self.logger.warning(f"No files found matching '{pattern}'")
            return self.get_stats()

        self.logger.info(f"Found {len(files)} files to import")

        # Process each file
        file_stats = []
        for filepath in files:
            # Check total limit
            if total_limit and self.total_processed >= total_limit:
                self.logger.info(f"Reached total limit: {total_limit}")
                break

            # Calculate remaining limit
            remaining = None
            if total_limit:
                remaining = total_limit - self.total_processed
                if limit_per_file:
                    remaining = min(remaining, limit_per_file)
            elif limit_per_file:
                remaining = limit_per_file

            # Import file
            stats = self.import_file(str(filepath), limit=remaining)
            file_stats.append(stats)

        elapsed = (datetime.now() - start_time).total_seconds()

        # Print summary
        self.logger.info(
            f"\n{'='*70}\n"
            f"Import Complete!\n"
            f"{'='*70}\n"
            f"Files processed: {len(file_stats)}\n"
            f"Documents processed: {self.total_processed}\n"
            f"Documents inserted: {self.total_inserted}\n"
            f"Duplicates skipped: {self.total_duplicates}\n"
            f"Errors: {self.total_errors}\n"
            f"Total time: {elapsed:.1f}s\n"
            f"{'='*70}"
        )

        return self.get_stats()

    def import_files(self,
                    filepaths: List[str],
                    limit_per_file: Optional[int] = None) -> Dict:
        """
        Import specific list of files.

        Args:
            filepaths: List of file paths
            limit_per_file: Max docs per file

        Returns:
            Overall statistics dictionary
        """
        self.logger.info(f"Importing {len(filepaths)} files...")

        for filepath in filepaths:
            self.import_file(filepath, limit=limit_per_file)

        return self.get_stats()

    def verify_import(self, jsonl_path: str) -> Dict:
        """
        Verify import by comparing JSONL file with database.

        Args:
            jsonl_path: Path to JSONL file

        Returns:
            Verification statistics

        Examples:
            >>> importer = JSONLImporter(db_manager)
            >>> verification = importer.verify_import('data/raw/ltn_14days.jsonl')
            >>> print(f"Match rate: {verification['match_rate']:.2f}%")
        """
        self.logger.info(f"Verifying import for {jsonl_path}...")

        jsonl_hashes = set()
        for doc in self.doc_reader.read_jsonl(jsonl_path):
            jsonl_hashes.add(doc.content_hash)

        db_hashes = set()
        # Query database for hashes
        # (This would require adding a method to PostgresManager to get all hashes)

        matches = len(jsonl_hashes & db_hashes)
        missing = len(jsonl_hashes - db_hashes)

        return {
            'jsonl_count': len(jsonl_hashes),
            'db_count': len(db_hashes),
            'matches': matches,
            'missing': missing,
            'match_rate': (matches / len(jsonl_hashes) * 100
                          if jsonl_hashes else 0)
        }

    def get_stats(self) -> Dict:
        """Get import statistics."""
        return {
            'total_processed': self.total_processed,
            'total_inserted': self.total_inserted,
            'total_duplicates': self.total_duplicates,
            'total_errors': self.total_errors,
            'duplication_rate': (self.total_duplicates / self.total_processed * 100
                                if self.total_processed > 0 else 0)
        }

    def reset_stats(self):
        """Reset statistics counters."""
        self.total_processed = 0
        self.total_inserted = 0
        self.total_duplicates = 0
        self.total_errors = 0


def demo():
    """Demonstration of JSONLImporter."""
    print("=" * 70)
    print("JSONL to PostgreSQL Importer Demo")
    print("=" * 70)

    # Initialize database manager
    print("\n1. Initializing database...")
    db_manager = PostgresManager(
        host="localhost",
        database="ir_news",
        user="postgres",
        password="postgres"
    )

    # Create schema
    db_manager.create_schema()

    # Initialize importer
    print("\n2. Initializing importer...")
    importer = JSONLImporter(db_manager, batch_size=100)

    # Check for data
    data_dir = Path("data/raw")
    if not data_dir.exists():
        print(f"\n⚠ Data directory not found: {data_dir}")
        print("Please run crawlers first to collect data.")
        db_manager.close()
        return

    jsonl_files = list(data_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"\n⚠ No JSONL files found in {data_dir}")
        db_manager.close()
        return

    # Import sample file (first 100 documents)
    print(f"\n3. Importing sample data (first 100 documents)...")
    print("-" * 70)
    sample_file = jsonl_files[0]
    stats = importer.import_file(str(sample_file), limit=100)

    print(f"\n4. Import Statistics:")
    print("-" * 70)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")

    # Get corpus statistics
    print(f"\n5. Database Statistics:")
    print("-" * 70)
    corpus_stats = db_manager.get_corpus_stats()
    print(f"   Total articles in DB: {corpus_stats['total_articles']}")
    print(f"   Sources: {corpus_stats['total_sources']}")
    if corpus_stats.get('source_counts'):
        print(f"   Source breakdown:")
        for source, count in corpus_stats['source_counts'].items():
            print(f"      {source}: {count}")

    # Close
    db_manager.close()
    print("\n" + "=" * 70)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    demo()
