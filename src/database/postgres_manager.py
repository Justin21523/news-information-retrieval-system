"""
PostgreSQL Connection and Schema Manager

Manages database connections, schema creation, and basic CRUD operations
for the news articles corpus.

Key Features:
    - Connection pooling
    - Automatic schema initialization
    - Transaction management
    - Full-text search indexes
    - Content hash deduplication

Author: Information Retrieval System
License: Educational Use
"""

import logging
import psycopg2
from psycopg2 import pool, sql
from psycopg2.extras import RealDictCursor, execute_values
from typing import Dict, List, Optional, Tuple, Any
from contextlib import contextmanager
from datetime import datetime


class PostgresManager:
    """
    PostgreSQL database manager for news articles.

    Handles connection pooling, schema management, and basic operations.

    Complexity:
        - connect: O(1)
        - insert_article: O(1)
        - batch_insert: O(n) where n is number of articles

    Attributes:
        connection_pool: psycopg2 connection pool
        logger: Logging instance
    """

    def __init__(self,
                 host: str = "localhost",
                 port: int = 5432,
                 database: str = "ir_news",
                 user: str = "postgres",
                 password: str = "postgres",
                 min_conn: int = 1,
                 max_conn: int = 10):
        """
        Initialize PostgresManager.

        Args:
            host: Database host
            port: Database port
            database: Database name
            user: Database user
            password: Database password
            min_conn: Minimum connections in pool
            max_conn: Maximum connections in pool
        """
        self.logger = logging.getLogger(__name__)

        self.config = {
            'host': host,
            'port': port,
            'database': database,
            'user': user,
            'password': password
        }

        # Initialize connection pool
        try:
            self.connection_pool = psycopg2.pool.SimpleConnectionPool(
                min_conn,
                max_conn,
                **self.config
            )

            if self.connection_pool:
                self.logger.info(
                    f"PostgreSQL connection pool created "
                    f"(database={database}, max_conn={max_conn})"
                )
            else:
                raise Exception("Connection pool creation failed")

        except psycopg2.Error as e:
            self.logger.error(f"Failed to create connection pool: {e}")
            raise

    @contextmanager
    def get_connection(self):
        """
        Get database connection from pool (context manager).

        Yields:
            psycopg2 connection

        Examples:
            >>> manager = PostgresManager()
            >>> with manager.get_connection() as conn:
            ...     cursor = conn.cursor()
            ...     cursor.execute("SELECT 1")
        """
        conn = None
        try:
            conn = self.connection_pool.getconn()
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            self.logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                self.connection_pool.putconn(conn)

    def create_schema(self):
        """
        Create database schema for news articles.

        Creates:
            - news_articles table
            - Indexes for content_hash, source, published_at
            - Full-text search index on title and content
            - Statistics table for metadata
        """
        self.logger.info("Creating database schema...")

        schema_sql = """
        -- News articles table
        CREATE TABLE IF NOT EXISTS news_articles (
            id SERIAL PRIMARY KEY,
            doc_id INTEGER UNIQUE,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            url TEXT UNIQUE,
            published_at TIMESTAMP,
            source VARCHAR(50) NOT NULL,
            category VARCHAR(50),
            author VARCHAR(200),
            content_hash VARCHAR(32) UNIQUE NOT NULL,
            metadata JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Indexes for efficient queries
        CREATE INDEX IF NOT EXISTS idx_content_hash
            ON news_articles(content_hash);

        CREATE INDEX IF NOT EXISTS idx_source
            ON news_articles(source);

        CREATE INDEX IF NOT EXISTS idx_published_at
            ON news_articles(published_at DESC);

        CREATE INDEX IF NOT EXISTS idx_category
            ON news_articles(category);

        -- Full-text search index (GIN)
        CREATE INDEX IF NOT EXISTS idx_fts_title_content
            ON news_articles
            USING GIN (to_tsvector('english', title || ' ' || content));

        -- Statistics table
        CREATE TABLE IF NOT EXISTS corpus_statistics (
            id SERIAL PRIMARY KEY,
            stat_key VARCHAR(100) UNIQUE NOT NULL,
            stat_value JSONB,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Update timestamp trigger
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = CURRENT_TIMESTAMP;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;

        DROP TRIGGER IF EXISTS update_news_articles_updated_at
            ON news_articles;

        CREATE TRIGGER update_news_articles_updated_at
            BEFORE UPDATE ON news_articles
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at_column();
        """

        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(schema_sql)
                conn.commit()

        self.logger.info("Database schema created successfully")

    def drop_schema(self):
        """Drop all tables (USE WITH CAUTION)."""
        self.logger.warning("Dropping all tables...")

        drop_sql = """
        DROP TABLE IF EXISTS news_articles CASCADE;
        DROP TABLE IF EXISTS corpus_statistics CASCADE;
        DROP FUNCTION IF EXISTS update_updated_at_column CASCADE;
        """

        with self.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(drop_sql)
                conn.commit()

        self.logger.info("Schema dropped")

    def insert_article(self, article: Dict[str, Any]) -> Optional[int]:
        """
        Insert a single news article.

        Args:
            article: Dictionary with article fields

        Returns:
            Article ID if inserted, None if duplicate

        Complexity:
            Time: O(1)

        Examples:
            >>> manager = PostgresManager()
            >>> article = {
            ...     'title': 'Test News',
            ...     'content': 'This is a test',
            ...     'url': 'https://example.com/test',
            ...     'source': 'test',
            ...     'content_hash': 'abc123...'
            ... }
            >>> article_id = manager.insert_article(article)
        """
        insert_sql = """
        INSERT INTO news_articles (
            doc_id, title, content, url, published_at,
            source, category, author, content_hash, metadata
        ) VALUES (
            %(doc_id)s, %(title)s, %(content)s, %(url)s, %(published_at)s,
            %(source)s, %(category)s, %(author)s, %(content_hash)s, %(metadata)s
        )
        ON CONFLICT (content_hash) DO NOTHING
        RETURNING id;
        """

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(insert_sql, article)
                    result = cursor.fetchone()
                    conn.commit()

                    if result:
                        return result[0]
                    else:
                        self.logger.debug(
                            f"Duplicate article skipped: {article.get('title', '')[:50]}"
                        )
                        return None

        except psycopg2.Error as e:
            self.logger.warning(f"Insert failed: {e}")
            return None

    def batch_insert_articles(self, articles: List[Dict[str, Any]]) -> Tuple[int, int]:
        """
        Batch insert multiple articles (more efficient).

        Args:
            articles: List of article dictionaries

        Returns:
            (inserted_count, duplicate_count) tuple

        Complexity:
            Time: O(n) where n is number of articles

        Examples:
            >>> manager = PostgresManager()
            >>> articles = [...]  # List of article dicts
            >>> inserted, duplicates = manager.batch_insert_articles(articles)
        """
        if not articles:
            return 0, 0

        insert_sql = """
        INSERT INTO news_articles (
            doc_id, title, content, url, published_at,
            source, category, author, content_hash, metadata
        ) VALUES %s
        ON CONFLICT (content_hash) DO NOTHING;
        """

        # Prepare values tuple
        values = [
            (
                article.get('doc_id'),
                article.get('title', ''),
                article.get('content', ''),
                article.get('url', ''),
                article.get('published_at'),
                article.get('source', ''),
                article.get('category'),
                article.get('author'),
                article.get('content_hash'),
                psycopg2.extras.Json(article.get('metadata', {}))
            )
            for article in articles
        ]

        try:
            with self.get_connection() as conn:
                with conn.cursor() as cursor:
                    # Get count before insert
                    cursor.execute("SELECT COUNT(*) FROM news_articles")
                    count_before = cursor.fetchone()[0]

                    # Batch insert
                    execute_values(cursor, insert_sql, values)
                    conn.commit()

                    # Get count after insert
                    cursor.execute("SELECT COUNT(*) FROM news_articles")
                    count_after = cursor.fetchone()[0]

                    inserted = count_after - count_before
                    duplicates = len(articles) - inserted

                    self.logger.info(
                        f"Batch insert: {inserted} inserted, {duplicates} duplicates"
                    )

                    return inserted, duplicates

        except psycopg2.Error as e:
            self.logger.error(f"Batch insert failed: {e}")
            return 0, len(articles)

    def get_article_by_hash(self, content_hash: str) -> Optional[Dict]:
        """
        Get article by content hash.

        Args:
            content_hash: MD5 content hash

        Returns:
            Article dictionary or None

        Complexity:
            Time: O(1) with index
        """
        query_sql = """
        SELECT * FROM news_articles
        WHERE content_hash = %s;
        """

        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query_sql, (content_hash,))
                result = cursor.fetchone()
                return dict(result) if result else None

    def get_articles_by_source(self, source: str, limit: int = 100) -> List[Dict]:
        """
        Get articles by source.

        Args:
            source: News source (e.g., 'ltn', 'udn')
            limit: Maximum number of articles

        Returns:
            List of article dictionaries

        Complexity:
            Time: O(n) where n is result size
        """
        query_sql = """
        SELECT * FROM news_articles
        WHERE source = %s
        ORDER BY published_at DESC
        LIMIT %s;
        """

        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query_sql, (source, limit))
                results = cursor.fetchall()
                return [dict(row) for row in results]

    def get_corpus_stats(self) -> Dict[str, Any]:
        """
        Get corpus statistics.

        Returns:
            Dictionary with statistics

        Examples:
            >>> manager = PostgresManager()
            >>> stats = manager.get_corpus_stats()
            >>> print(f"Total articles: {stats['total_articles']}")
        """
        stats_sql = """
        SELECT
            COUNT(*) as total_articles,
            COUNT(DISTINCT source) as total_sources,
            COUNT(DISTINCT category) as total_categories,
            MIN(published_at) as earliest_date,
            MAX(published_at) as latest_date,
            MIN(created_at) as first_indexed,
            MAX(created_at) as last_indexed
        FROM news_articles;
        """

        source_stats_sql = """
        SELECT source, COUNT(*) as count
        FROM news_articles
        GROUP BY source
        ORDER BY count DESC;
        """

        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                # Overall stats
                cursor.execute(stats_sql)
                stats = dict(cursor.fetchone())

                # Source breakdown
                cursor.execute(source_stats_sql)
                source_counts = {row['source']: row['count'] for row in cursor.fetchall()}
                stats['source_counts'] = source_counts

                return stats

    def close(self):
        """Close all connections in pool."""
        if self.connection_pool:
            self.connection_pool.closeall()
            self.logger.info("Connection pool closed")


def demo():
    """Demonstration of PostgresManager."""
    print("=" * 70)
    print("PostgreSQL Manager Demo")
    print("=" * 70)

    # Initialize manager
    print("\n1. Initializing PostgresManager...")
    manager = PostgresManager(
        host="localhost",
        database="ir_news",
        user="postgres",
        password="postgres"
    )

    # Create schema
    print("\n2. Creating schema...")
    manager.create_schema()

    # Insert sample article
    print("\n3. Inserting sample article...")
    sample_article = {
        'title': '測試新聞標題',
        'content': '這是一則測試新聞內容，用於展示PostgreSQL整合功能。',
        'url': 'https://example.com/test/1',
        'source': 'test',
        'category': 'technology',
        'published_at': datetime.now(),
        'content_hash': 'test_hash_123',
        'metadata': {'test': True}
    }

    article_id = manager.insert_article(sample_article)
    if article_id:
        print(f"   Article inserted with ID: {article_id}")
    else:
        print("   Article was duplicate, skipped")

    # Get statistics
    print("\n4. Corpus Statistics:")
    print("-" * 70)
    stats = manager.get_corpus_stats()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for k, v in value.items():
                print(f"      {k}: {v}")
        else:
            print(f"   {key}: {value}")

    # Close
    manager.close()
    print("\n" + "=" * 70)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    demo()
