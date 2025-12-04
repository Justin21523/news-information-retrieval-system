#!/usr/bin/env python
"""
PostgreSQL Database Initialization Script

Creates the IR news database and schema for the Information Retrieval system.

Usage:
    python scripts/init_postgres.py

Author: Information Retrieval System
"""

import sys
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.postgres_manager import PostgresManager


def create_database(host='localhost', port=5432, user='postgres', password='postgres', dbname='ir_news'):
    """
    Create the IR news database if it doesn't exist.

    Args:
        host: Database host
        port: Database port
        user: Database user
        password: Database password
        dbname: Database name to create
    """
    print(f"=" * 70)
    print("PostgreSQL Database Initialization")
    print(f"=" * 70)

    # Connect to default 'postgres' database to create new database
    print(f"\n1. Connecting to PostgreSQL server...")
    try:
        conn = psycopg2.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database='postgres'  # Connect to default database first
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()

        print(f"   ✓ Connected to PostgreSQL {host}:{port}")

    except psycopg2.Error as e:
        print(f"   ✗ Failed to connect: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure PostgreSQL is running: sudo service postgresql status")
        print("  2. Check credentials in pg_hba.conf")
        print("  3. Update password if needed")
        sys.exit(1)

    # Check if database exists
    print(f"\n2. Checking if database '{dbname}' exists...")
    cursor.execute(
        "SELECT 1 FROM pg_database WHERE datname = %s",
        (dbname,)
    )
    exists = cursor.fetchone()

    if exists:
        print(f"   ℹ Database '{dbname}' already exists")
    else:
        print(f"   Creating database '{dbname}'...")
        try:
            cursor.execute(f'CREATE DATABASE {dbname}')
            print(f"   ✓ Database '{dbname}' created successfully")
        except psycopg2.Error as e:
            print(f"   ✗ Failed to create database: {e}")
            cursor.close()
            conn.close()
            sys.exit(1)

    cursor.close()
    conn.close()

    # Initialize schema using PostgresManager
    print(f"\n3. Initializing database schema...")
    try:
        manager = PostgresManager(
            host=host,
            port=port,
            database=dbname,
            user=user,
            password=password
        )

        print(f"   ✓ Connected to database '{dbname}'")

        # Create schema
        manager.create_schema()
        print(f"   ✓ Schema created successfully")

        # Verify tables
        print(f"\n4. Verifying tables...")
        with manager.get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = 'public'
                    ORDER BY table_name;
                """)
                tables = cursor.fetchall()

                if tables:
                    print(f"   Tables created:")
                    for table in tables:
                        print(f"      • {table[0]}")
                else:
                    print(f"   ⚠ No tables found")

        # Get initial statistics
        print(f"\n5. Initial database statistics:")
        stats = manager.get_corpus_stats()
        print(f"   Total articles: {stats['total_articles']}")
        print(f"   Total sources: {stats['total_sources']}")
        print(f"   Total categories: {stats['total_categories']}")

        manager.close()

        print(f"\n{'=' * 70}")
        print("Database initialization complete!")
        print(f"{'=' * 70}")
        print(f"\nNext steps:")
        print(f"  1. Import JSONL data:")
        print(f"     python -m src.database.jsonl_importer")
        print(f"  2. Or run the unified search system:")
        print(f"     python scripts/search_news.py --build --from-db")
        print(f"{'=' * 70}\n")

    except Exception as e:
        print(f"   ✗ Failed to initialize schema: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Initialize PostgreSQL database for IR news system'
    )
    parser.add_argument('--host', default='localhost', help='Database host')
    parser.add_argument('--port', type=int, default=5432, help='Database port')
    parser.add_argument('--user', default='postgres', help='Database user')
    parser.add_argument('--password', default='postgres', help='Database password')
    parser.add_argument('--dbname', default='ir_news', help='Database name')

    args = parser.parse_args()

    create_database(
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        dbname=args.dbname
    )


if __name__ == '__main__':
    main()
