"""
Document Reader for News Corpus

This module provides utilities to read news articles from JSONL files
and PostgreSQL database. It handles document normalization,
metadata extraction, and supports batch reading for large corpora.

Key Features:
    - Read JSONL files from crawler output
    - Read from PostgreSQL database
    - Normalize document structure
    - Extract metadata (title, content, date, source)
    - Batch reading for memory efficiency
    - Progress tracking

Author: Information Retrieval System
License: Educational Use
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Iterator, Optional, Tuple, Any
from datetime import datetime
import hashlib


class NewsDocument:
    """
    News document representation.

    Attributes:
        doc_id: Document identifier (assigned by index)
        title: Article title
        content: Article content
        url: Article URL
        published_at: Publication date
        source: News source (e.g., 'ltn', 'yahoo', 'udn')
        category: Article category (e.g., 'politics', 'finance')
        author: Article author (optional)
        content_hash: MD5 hash for deduplication
        metadata: Additional metadata dict
    """

    def __init__(self, **kwargs):
        """
        Initialize NewsDocument.

        Args:
            **kwargs: Document fields (title, content, url, etc.)
        """
        # `doc_id` is treated as the cross-module primary key.
        # It may be:
        # - None when first loaded from raw sources (JSONL / DB)
        # - assigned later by an indexer (e.g., InvertedIndex.add_document)
        self.doc_id: Optional[int] = kwargs.get('doc_id')
        self.title: str = kwargs.get('title', '')
        self.content: str = kwargs.get('content', '')
        self.url: str = kwargs.get('url', '')
        self.published_at: Optional[str] = kwargs.get('published_at')
        self.source: str = kwargs.get('source', '')
        self.category: Optional[str] = kwargs.get('category')
        self.author: Optional[str] = kwargs.get('author')
        self.metadata: dict = kwargs.get('metadata', {})

        # Precompute a content hash used by the deduplication module.
        # This is a convenience field; it is not a cryptographic guarantee.
        self.content_hash = self._calculate_hash()

    def _calculate_hash(self) -> str:
        """
        Calculate MD5 hash of content for deduplication.

        Uses title + content for hashing.

        Returns:
            MD5 hash string

        Complexity:
            Time: O(n) where n is content length
        """
        # Use a stable, deterministic representation of the document content.
        #
        # We include both title and content so that:
        # - title-only duplicates can be detected
        # - short content with different titles is less likely to collide
        #
        # NOTE:
        # MD5 is used purely as a fast checksum for deduplication, not for security.
        text = f"{self.title}\n{self.content}"
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def get_full_text(self) -> str:
        """
        Get full text (title + content) for indexing.

        Returns:
            Combined text string
        """
        # Newline is a simple separator that preserves a weak boundary between
        # title and body. Most tokenizers treat it as whitespace.
        return f"{self.title}\n{self.content}"

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'doc_id': self.doc_id,
            'title': self.title,
            'content': self.content,
            'url': self.url,
            'published_at': self.published_at,
            'source': self.source,
            'category': self.category,
            'author': self.author,
            'content_hash': self.content_hash,
            'metadata': self.metadata
        }

    def __repr__(self) -> str:
        """String representation."""
        return (f"NewsDocument(source={self.source}, "
                f"title='{self.title[:50]}...', "
                f"date={self.published_at})")


class DocumentReader:
    """
    Document reader for JSONL news files.

    Reads news articles from JSONL files produced by crawlers.
    Supports batch reading and progress tracking.

    Complexity:
        - read_jsonl: O(n) where n is number of lines
        - read_directory: O(m * n) where m is number of files

    Attributes:
        logger: Logging instance
        docs_read: Counter for documents read
    """

    def __init__(self):
        """Initialize DocumentReader."""
        self.logger = logging.getLogger(__name__)
        self.docs_read = 0

    def read_jsonl(self, filepath: str,
                   limit: Optional[int] = None) -> Iterator[NewsDocument]:
        """
        Read documents from a JSONL file.

        Args:
            filepath: Path to JSONL file
            limit: Maximum number of documents to read (None for all)

        Yields:
            NewsDocument instances

        Complexity:
            Time: O(n) where n is number of lines
            Space: O(1) per iteration (generator)

        Examples:
            >>> reader = DocumentReader()
            >>> for doc in reader.read_jsonl('/mnt/c/data/information-retrieval/raw/ltn_14days.jsonl', limit=100):
            ...     print(doc.title)
        """
        filepath = Path(filepath)

        if not filepath.exists():
            self.logger.warning(f"File not found: {filepath}")
            return

        self.logger.info(f"Reading JSONL: {filepath.name}")

        count = 0
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, 1):
                # Check limit
                if limit and count >= limit:
                    break

                # Skip empty lines
                line = line.strip()
                if not line:
                    continue

                try:
                    # JSONL format: one JSON object per line.
                    # We parse each line independently so a single corrupt line
                    # does not block the whole file.
                    data = json.loads(line)

                    # Convert raw dict to a structured object to keep downstream
                    # indexing code simple and uniform.
                    doc = NewsDocument(**data)

                    # Skip documents without content
                    if not doc.content:
                        self.logger.debug(f"Line {line_no}: Empty content, skipping")
                        continue

                    count += 1
                    self.docs_read += 1

                    yield doc

                except json.JSONDecodeError as e:
                    self.logger.warning(
                        f"Line {line_no}: JSON decode error - {e}"
                    )
                    continue
                except Exception as e:
                    # Defensive programming:
                    # Crawlers may emit heterogeneous schemas, missing fields,
                    # or unexpected types; treat them as per-line errors.
                    self.logger.warning(
                        f"Line {line_no}: Error creating document - {e}"
                    )
                    continue

        self.logger.info(f"Read {count} documents from {filepath.name}")

    def read_directory(self,
                      directory: str,
                      pattern: str = "*.jsonl",
                      limit_per_file: Optional[int] = None,
                      total_limit: Optional[int] = None) -> Iterator[NewsDocument]:
        """
        Read documents from all JSONL files in a directory.

        Args:
            directory: Directory path
            pattern: File pattern (e.g., '*.jsonl', 'ltn_*.jsonl')
            limit_per_file: Max docs per file (None for all)
            total_limit: Max docs total (None for all)

        Yields:
            NewsDocument instances

        Complexity:
            Time: O(m * n) where m is files, n is lines per file

        Examples:
            >>> reader = DocumentReader()
            >>> for doc in reader.read_directory('/mnt/c/data/information-retrieval/raw', pattern='*_14days.jsonl'):
            ...     print(doc.source, doc.title)
        """
        directory = Path(directory)

        if not directory.exists():
            self.logger.error(f"Directory not found: {directory}")
            return

        # Find and sort matching files so iteration is deterministic.
        # Determinism is helpful for debugging and for reproducible indexing.
        files = sorted(directory.glob(pattern))

        if not files:
            self.logger.warning(
                f"No files matching '{pattern}' in {directory}"
            )
            return

        self.logger.info(
            f"Reading {len(files)} files from {directory}"
        )

        total_count = 0

        for filepath in files:
            # Check total limit
            if total_limit and total_count >= total_limit:
                break

            # Calculate remaining limit
            # We enforce limits at two levels:
            # - per file (limit_per_file)
            # - total across all files (total_limit)
            remaining = None
            if total_limit:
                remaining = total_limit - total_count
                if limit_per_file:
                    remaining = min(remaining, limit_per_file)
            elif limit_per_file:
                remaining = limit_per_file

            # Read file
            file_count = 0
            for doc in self.read_jsonl(str(filepath), limit=remaining):
                yield doc
                file_count += 1
                total_count += 1

                if total_limit and total_count >= total_limit:
                    break

        self.logger.info(
            f"Read {total_count} documents total from {len(files)} files"
        )

    def read_files(self,
                  filepaths: List[str],
                  limit_per_file: Optional[int] = None) -> Iterator[NewsDocument]:
        """
        Read documents from a list of files.

        Args:
            filepaths: List of file paths
            limit_per_file: Max docs per file

        Yields:
            NewsDocument instances
        """
        for filepath in filepaths:
            for doc in self.read_jsonl(filepath, limit=limit_per_file):
                yield doc

    def get_file_stats(self, filepath: str) -> dict:
        """
        Get statistics for a JSONL file.

        Args:
            filepath: Path to JSONL file

        Returns:
            Dictionary with statistics

        Examples:
            >>> reader = DocumentReader()
            >>> stats = reader.get_file_stats('/mnt/c/data/information-retrieval/raw/ltn_14days.jsonl')
            >>> print(f"Total docs: {stats['total_documents']}")
        """
        filepath = Path(filepath)

        if not filepath.exists():
            return {'error': 'File not found'}

        # We compute basic descriptive stats in a single pass to avoid loading
        # large files into memory.
        total_docs = 0
        sources = set()
        categories = set()
        date_range = []
        errors = 0

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    total_docs += 1

                    if 'source' in data:
                        sources.add(data['source'])
                    if 'category' in data:
                        categories.add(data['category'])
                    if 'published_at' in data:
                        date_range.append(data['published_at'])

                except:
                    # Count parse errors but keep going; stats is best-effort.
                    errors += 1

        # Sort dates to get range
        date_range = sorted(date_range) if date_range else []

        return {
            'filename': filepath.name,
            'total_documents': total_docs,
            'sources': list(sources),
            'categories': list(categories),
            'date_range': {
                'earliest': date_range[0] if date_range else None,
                'latest': date_range[-1] if date_range else None
            },
            'errors': errors,
            'file_size_mb': filepath.stat().st_size / (1024 * 1024)
        }

    def reset_counter(self):
        """Reset document counter."""
        self.docs_read = 0

    def read_from_postgres(self,
                          db_manager: Any,
                          source: Optional[str] = None,
                          category: Optional[str] = None,
                          limit: Optional[int] = None,
                          batch_size: int = 1000) -> Iterator[NewsDocument]:
        """
        Read documents from PostgreSQL database.

        Args:
            db_manager: PostgresManager instance
            source: Filter by source (None for all)
            category: Filter by category (None for all)
            limit: Maximum documents to read (None for all)
            batch_size: Fetch batch size for efficiency

        Yields:
            NewsDocument instances

        Complexity:
            Time: O(n) where n is number of documents
            Space: O(b) where b is batch_size

        Examples:
            >>> from src.database.postgres_manager import PostgresManager
            >>> db_manager = PostgresManager()
            >>> reader = DocumentReader()
            >>> for doc in reader.read_from_postgres(db_manager, source='ltn', limit=100):
            ...     print(doc.title)
        """
        self.logger.info(f"Reading from PostgreSQL (source={source}, limit={limit})")

        # Build query incrementally to keep the API flexible.
        # Parameters are bound via `%s` placeholders to avoid SQL injection
        # and to let the DB driver handle proper escaping.
        query = "SELECT * FROM news_articles WHERE 1=1"
        params = []

        if source:
            query += " AND source = %s"
            params.append(source)

        if category:
            query += " AND category = %s"
            params.append(category)

        query += " ORDER BY published_at DESC"

        if limit:
            query += f" LIMIT {limit}"

        try:
            with db_manager.get_connection() as conn:
                # Use a server-side cursor to stream results in batches.
                # This avoids pulling the entire result set into memory.
                cursor_name = f"doc_cursor_{id(self)}"
                with conn.cursor(name=cursor_name) as cursor:
                    cursor.execute(query, params)

                    count = 0
                    while True:
                        # Fetch in batches to balance network/IO overhead and memory use.
                        rows = cursor.fetchmany(batch_size)
                        if not rows:
                            break

                        for row in rows:
                            # Convert database rows to a dict compatible with NewsDocument.
                            #
                            # Depending on cursor configuration, a row can be:
                            # - dict-like (RealDictCursor)
                            # - tuple-like (default cursor)
                            if isinstance(row, dict):
                                doc_data = row
                            else:
                                # Manual conversion for tuple results
                                doc_data = {
                                    'doc_id': row[1],
                                    'title': row[2],
                                    'content': row[3],
                                    'url': row[4],
                                    'published_at': row[5].isoformat() if row[5] else None,
                                    'source': row[6],
                                    'category': row[7],
                                    'author': row[8],
                                    'content_hash': row[9],
                                    'metadata': row[10]
                                }

                            # Create NewsDocument
                            doc = NewsDocument(**doc_data)

                            count += 1
                            self.docs_read += 1

                            yield doc

                    self.logger.info(f"Read {count} documents from PostgreSQL")

        except Exception as e:
            self.logger.error(f"Error reading from PostgreSQL: {e}")
            raise

    def read_from_postgres_iter(self,
                                db_manager: Any,
                                doc_ids: Optional[List[int]] = None,
                                content_hashes: Optional[List[str]] = None) -> Iterator[NewsDocument]:
        """
        Read specific documents from PostgreSQL by ID or hash.

        Args:
            db_manager: PostgresManager instance
            doc_ids: List of document IDs to fetch
            content_hashes: List of content hashes to fetch

        Yields:
            NewsDocument instances

        Examples:
            >>> reader = DocumentReader()
            >>> doc_ids = [1, 2, 3]
            >>> for doc in reader.read_from_postgres_iter(db_manager, doc_ids=doc_ids):
            ...     print(doc.title)
        """
        if not doc_ids and not content_hashes:
            self.logger.warning("No doc_ids or content_hashes provided")
            return

        try:
            with db_manager.get_connection() as conn:
                with conn.cursor() as cursor:
                    if doc_ids:
                        # Fetch by IDs via PostgreSQL's ANY(array) operator.
                        query = """
                        SELECT * FROM news_articles
                        WHERE doc_id = ANY(%s)
                        ORDER BY doc_id
                        """
                        cursor.execute(query, (doc_ids,))

                    elif content_hashes:
                        # Fetch by content hashes (useful for deduplication pipelines).
                        query = """
                        SELECT * FROM news_articles
                        WHERE content_hash = ANY(%s)
                        """
                        cursor.execute(query, (content_hashes,))

                    for row in cursor:
                        # Convert row to dict
                        doc_data = {
                            'doc_id': row[1],
                            'title': row[2],
                            'content': row[3],
                            'url': row[4],
                            'published_at': row[5].isoformat() if row[5] else None,
                            'source': row[6],
                            'category': row[7],
                            'author': row[8],
                            'content_hash': row[9],
                            'metadata': row[10]
                        }

                        doc = NewsDocument(**doc_data)
                        self.docs_read += 1
                        yield doc

        except Exception as e:
            self.logger.error(f"Error reading documents by ID/hash: {e}")
            raise


def demo():
    """Demonstration of DocumentReader."""
    print("=" * 70)
    print("Document Reader Demo")
    print("=" * 70)

    # Example: Read a few documents
    reader = DocumentReader()
    data_dir = Path("/mnt/c/data/information-retrieval/raw")

    if not data_dir.exists():
        print(f"\n⚠ Data directory not found: {data_dir}")
        print("Please run crawlers first to collect data.")
        return

    # Get file stats
    jsonl_files = list(data_dir.glob("*.jsonl"))
    if not jsonl_files:
        print(f"\n⚠ No JSONL files found in {data_dir}")
        return

    print(f"\n1. File Statistics:")
    print(f"   Found {len(jsonl_files)} JSONL files")

    # Show stats for first file
    if jsonl_files:
        sample_file = jsonl_files[0]
        stats = reader.get_file_stats(str(sample_file))
        print(f"\n   Sample file: {stats['filename']}")
        print(f"   Total documents: {stats['total_documents']}")
        print(f"   Sources: {stats['sources']}")
        print(f"   File size: {stats['file_size_mb']:.2f} MB")
        if stats['date_range']['earliest']:
            print(f"   Date range: {stats['date_range']['earliest']} to "
                  f"{stats['date_range']['latest']}")

    # Read sample documents
    print(f"\n2. Sample Documents (first 3):")
    count = 0
    for doc in reader.read_directory(str(data_dir), limit_per_file=1, total_limit=3):
        count += 1
        print(f"\n   Document {count}:")
        print(f"      Source: {doc.source}")
        print(f"      Title: {doc.title[:60]}...")
        print(f"      Date: {doc.published_at}")
        print(f"      Content length: {len(doc.content)} chars")
        print(f"      Hash: {doc.content_hash[:16]}...")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    demo()
