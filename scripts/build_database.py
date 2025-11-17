#!/usr/bin/env python3
"""
SQLite Database Builder for CNIRS Project

This script creates and populates a SQLite database with preprocessed news articles.
The database provides structured storage for efficient querying and retrieval.

Schema:
    - news: Main table with article content and NLP features
    - Indexes on: published_date, category, article_id

Usage:
    python scripts/build_database.py \\
        --input data/preprocessed/cna_mvp_preprocessed.jsonl \\
        --output data/database/cnirs.db

Complexity:
    Time: O(N) where N=number of articles
    Space: O(N) for database storage

Author: Information Retrieval System
License: Educational Use
"""

import argparse
import json
import logging
import sqlite3
import sys
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatabaseBuilder:
    """
    SQLite database builder for news articles.

    Creates and populates database with preprocessed article data
    including NLP features (tokens, entities, keywords, summary).

    Attributes:
        db_path: Path to SQLite database file
        conn: Database connection
        cursor: Database cursor
    """

    # Database schema
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS news (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        article_id TEXT UNIQUE NOT NULL,
        source TEXT,
        url TEXT,
        title TEXT NOT NULL,
        content TEXT NOT NULL,
        category TEXT,
        category_name TEXT,
        published_date TEXT,
        author TEXT,

        -- NLP features (stored as JSON)
        tokens_title TEXT,      -- JSON array
        tokens_content TEXT,    -- JSON array
        entities TEXT,          -- JSON array of entity objects
        keywords TEXT,          -- JSON array
        summary TEXT,           -- Summarized text
        tags TEXT,              -- JSON array

        -- Metadata
        crawled_at TEXT,
        processed_at TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    CREATE INDEX IF NOT EXISTS idx_article_id ON news(article_id);
    CREATE INDEX IF NOT EXISTS idx_published_date ON news(published_date);
    CREATE INDEX IF NOT EXISTS idx_category ON news(category);
    CREATE INDEX IF NOT EXISTS idx_source ON news(source);

    -- Full-text search support (optional, for future use)
    CREATE VIRTUAL TABLE IF NOT EXISTS news_fts USING fts5(
        title, content, keywords,
        content='news',
        content_rowid='id'
    );
    """

    def __init__(self, db_path: Path):
        """
        Initialize database builder.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self.cursor = None

        logger.info(f"Initializing database: {db_path}")

    def create_database(self):
        """Create database and tables."""
        # Create parent directory
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Connect to database
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()

        logger.info("Creating database schema...")

        # Execute schema
        self.cursor.executescript(self.SCHEMA)
        self.conn.commit()

        logger.info("Database schema created successfully")

    def insert_articles(self, articles: List[Dict]) -> int:
        """
        Insert articles into database.

        Args:
            articles: List of preprocessed article dictionaries

        Returns:
            Number of articles inserted
        """
        logger.info(f"Inserting {len(articles)} articles into database...")

        insert_query = """
        INSERT OR REPLACE INTO news (
            article_id, source, url, title, content,
            category, category_name, published_date, author,
            tokens_title, tokens_content, entities, keywords, summary, tags,
            crawled_at, processed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        inserted = 0
        failed = 0

        for article in articles:
            try:
                # Prepare data
                data = (
                    article.get('article_id', ''),
                    article.get('source', ''),
                    article.get('url', ''),
                    article.get('title', ''),
                    article.get('content', ''),
                    article.get('category', ''),
                    article.get('category_name', ''),
                    article.get('published_date', ''),
                    article.get('author', ''),
                    # NLP features (convert to JSON strings)
                    json.dumps(article.get('tokens_title', []), ensure_ascii=False),
                    json.dumps(article.get('tokens_content', []), ensure_ascii=False),
                    json.dumps(article.get('entities', []), ensure_ascii=False),
                    json.dumps(article.get('keywords', []), ensure_ascii=False),
                    article.get('summary', ''),
                    json.dumps(article.get('tags', []), ensure_ascii=False),
                    article.get('crawled_at', ''),
                    article.get('processed_at', '')
                )

                self.cursor.execute(insert_query, data)
                inserted += 1

                if inserted % 50 == 0:
                    logger.info(f"Inserted {inserted}/{len(articles)} articles...")

            except Exception as e:
                logger.error(f"Failed to insert article {article.get('article_id', 'unknown')}: {e}")
                failed += 1

        # Commit transaction
        self.conn.commit()

        logger.info(f"Insertion completed: {inserted} successful, {failed} failed")
        return inserted

    def populate_fts(self):
        """
        Populate full-text search index.

        This enables fast text search queries using SQLite FTS5.
        """
        logger.info("Populating full-text search index...")

        try:
            self.cursor.execute("""
                INSERT INTO news_fts(rowid, title, content, keywords)
                SELECT id, title, content, keywords FROM news
            """)
            self.conn.commit()
            logger.info("Full-text search index populated successfully")
        except Exception as e:
            logger.warning(f"Failed to populate FTS index: {e}")

    def get_statistics(self) -> Dict:
        """
        Get database statistics.

        Returns:
            Dictionary of statistics
        """
        stats = {}

        # Total articles
        self.cursor.execute("SELECT COUNT(*) FROM news")
        stats['total_articles'] = self.cursor.fetchone()[0]

        # Articles by category
        self.cursor.execute("""
            SELECT category_name, COUNT(*)
            FROM news
            GROUP BY category_name
            ORDER BY COUNT(*) DESC
        """)
        stats['by_category'] = dict(self.cursor.fetchall())

        # Articles by source
        self.cursor.execute("""
            SELECT source, COUNT(*)
            FROM news
            GROUP BY source
            ORDER BY COUNT(*) DESC
        """)
        stats['by_source'] = dict(self.cursor.fetchall())

        # Date range
        self.cursor.execute("""
            SELECT MIN(published_date), MAX(published_date)
            FROM news
        """)
        min_date, max_date = self.cursor.fetchone()
        stats['date_range'] = {'min': min_date, 'max': max_date}

        # Database size
        stats['db_size_mb'] = self.db_path.stat().st_size / (1024 * 1024)

        return stats

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")


def load_articles(file_path: Path) -> List[Dict]:
    """Load articles from JSONL file."""
    articles = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                articles.append(json.loads(line))
    return articles


def generate_report(stats: Dict, output_path: Path):
    """Generate database statistics report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("DATABASE STATISTICS\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("COLLECTION STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total articles:       {stats['total_articles']}\n")
        f.write(f"Database size:        {stats['db_size_mb']:.2f} MB\n")
        f.write(f"Date range:           {stats['date_range']['min']} to {stats['date_range']['max']}\n\n")

        f.write("ARTICLES BY CATEGORY\n")
        f.write("-" * 80 + "\n")
        for category, count in stats['by_category'].items():
            f.write(f"{category:20s} {count:5d}\n")
        f.write("\n")

        f.write("ARTICLES BY SOURCE\n")
        f.write("-" * 80 + "\n")
        for source, count in stats['by_source'].items():
            f.write(f"{source:20s} {count:5d}\n")

    logger.info(f"Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Build SQLite database from preprocessed news articles',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/build_database.py \\
      --input data/preprocessed/cna_mvp_preprocessed.jsonl \\
      --output data/database/cnirs.db

  # With custom statistics output
  python scripts/build_database.py \\
      --input data/preprocessed/cna_mvp_preprocessed.jsonl \\
      --output data/database/cnirs.db \\
      --stats data/stats/database_stats.txt

  # Skip FTS index
  python scripts/build_database.py \\
      --input data/preprocessed/cna_mvp_preprocessed.jsonl \\
      --output data/database/cnirs.db \\
      --no-fts
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
        help='Output SQLite database file (.db)'
    )

    parser.add_argument(
        '--stats',
        type=str,
        default=None,
        help='Output path for statistics report (optional)'
    )

    parser.add_argument(
        '--no-fts',
        action='store_true',
        help='Skip creating full-text search index'
    )

    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing database'
    )

    args = parser.parse_args()

    # Validate paths
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    if output_path.exists() and not args.overwrite:
        logger.error(f"Database already exists: {output_path}")
        logger.error("Use --overwrite to replace it")
        sys.exit(1)

    # Delete existing database if overwrite
    if output_path.exists() and args.overwrite:
        output_path.unlink()
        logger.info(f"Removed existing database: {output_path}")

    # Load articles
    logger.info(f"Loading articles from {input_path}")
    articles = load_articles(input_path)
    logger.info(f"Loaded {len(articles)} articles")

    # Initialize database builder
    builder = DatabaseBuilder(output_path)

    # Create database
    start_time = time.time()
    builder.create_database()

    # Insert articles
    inserted = builder.insert_articles(articles)

    # Populate FTS index
    if not args.no_fts:
        builder.populate_fts()

    # Get statistics
    stats = builder.get_statistics()
    stats['build_time'] = time.time() - start_time

    # Save statistics
    if args.stats:
        stats_path = Path(args.stats)
    else:
        stats_path = output_path.parent.parent / 'stats' / 'database_stats.txt'

    generate_report(stats, stats_path)

    # Close database
    builder.close()

    # Print summary
    print("\n" + "=" * 80)
    print("DATABASE BUILD COMPLETED")
    print("=" * 80)
    print(f"Input:           {input_path}")
    print(f"Output:          {output_path}")
    print(f"Statistics:      {stats_path}")
    print(f"Articles:        {inserted}")
    print(f"Database size:   {stats['db_size_mb']:.2f} MB")
    print(f"Build time:      {stats['build_time']:.2f} seconds")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
