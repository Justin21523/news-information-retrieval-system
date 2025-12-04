#!/usr/bin/env python
"""
Background Tasks Progress Monitor

Monitors progress of:
- Index building
- PostgreSQL import

Usage:
    python scripts/monitor_progress.py

Author: Information Retrieval System
"""

import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime


def check_process(pid: str) -> bool:
    """Check if process is running."""
    try:
        result = subprocess.run(
            ['ps', '-p', pid],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except:
        return False


def tail_file(filepath: str, lines: int = 10) -> list:
    """Get last n lines from file."""
    try:
        result = subprocess.run(
            ['tail', f'-{lines}', filepath],
            capture_output=True,
            text=True
        )
        return result.stdout.strip().split('\n') if result.stdout else []
    except:
        return []


def count_lines(filepath: str) -> int:
    """Count lines in file."""
    try:
        result = subprocess.run(
            ['wc', '-l', filepath],
            capture_output=True,
            text=True
        )
        return int(result.stdout.split()[0]) if result.stdout else 0
    except:
        return 0


def check_index_progress():
    """Check index building progress."""
    log_file = '/tmp/build_50k_index.log'

    print("\n[1] Index Building (50,000 documents)")
    print("-" * 70)

    if not Path(log_file).exists():
        print("   ⚠ Log file not found")
        return

    # Count lines (rough progress indicator)
    line_count = count_lines(log_file)
    print(f"   Log lines: {line_count:,}")

    # Get last lines
    last_lines = tail_file(log_file, 15)

    # Look for progress indicators
    for line in last_lines:
        if 'INFO' in line or '%' in line or 'documents' in line:
            print(f"   {line.strip()}")

    # Check if completed
    for line in last_lines:
        if 'Statistics' in line or 'Complete' in line:
            print("\n   ✓ Index building completed!")
            return

    print("\n   🔄 Index building in progress...")


def check_postgres_progress():
    """Check PostgreSQL import progress."""
    log_file = '/tmp/postgres_import.log'

    print("\n[2] PostgreSQL Import (257,137 articles)")
    print("-" * 70)

    if not Path(log_file).exists():
        print("   ⚠ Log file not found")
        return

    # Count lines
    line_count = count_lines(log_file)
    print(f"   Log lines: {line_count:,}")

    # Get last lines
    last_lines = tail_file(log_file, 20)

    # Look for current file being processed
    current_file = None
    for line in reversed(last_lines):
        if 'Importing' in line and '.jsonl' in line:
            parts = line.split()
            for part in parts:
                if '.jsonl' in part:
                    current_file = part.rstrip(':')
                    break
            break

    if current_file:
        print(f"   Current file: {current_file}")

    # Look for progress bars or stats
    for line in last_lines:
        if '%|' in line or 'inserted' in line or 'duplicates' in line:
            print(f"   {line.strip()}")

    # Check if completed
    for line in last_lines:
        if 'Import Complete' in line or 'Database connection closed' in line:
            print("\n   ✓ PostgreSQL import completed!")
            return

    print("\n   🔄 PostgreSQL import in progress...")


def check_database_stats():
    """Check current database statistics."""
    print("\n[3] Database Status")
    print("-" * 70)

    try:
        import psycopg2

        conn = psycopg2.connect(
            host='localhost',
            database='ir_news',
            user='postgres',
            password='postgres'
        )

        cursor = conn.cursor()

        # Count articles
        cursor.execute("SELECT COUNT(*) FROM news_articles")
        article_count = cursor.fetchone()[0]

        # Count sources
        cursor.execute("SELECT COUNT(DISTINCT source) FROM news_articles")
        source_count = cursor.fetchone()[0]

        # Get source breakdown
        cursor.execute("""
            SELECT source, COUNT(*) as count
            FROM news_articles
            GROUP BY source
            ORDER BY count DESC
            LIMIT 5
        """)
        sources = cursor.fetchall()

        print(f"   Total articles: {article_count:,}")
        print(f"   Total sources: {source_count}")

        if sources:
            print(f"\n   Top sources:")
            for source, count in sources:
                print(f"      {source}: {count:,}")

        cursor.close()
        conn.close()

    except Exception as e:
        print(f"   ⚠ Cannot connect to database: {e}")


def check_index_status():
    """Check index directory status."""
    print("\n[4] Index Status")
    print("-" * 70)

    index_dir = Path('data/index_50k')

    if not index_dir.exists():
        print("   ⚠ Index directory not found")
        return

    # Check for index files
    index_files = list(index_dir.rglob('*.pkl'))
    meta_files = list(index_dir.rglob('*.json'))

    print(f"   Index files: {len(index_files)}")
    print(f"   Metadata files: {len(meta_files)}")

    if index_files:
        # Calculate total size
        total_size = sum(f.stat().st_size for f in index_files)
        print(f"   Total size: {total_size / 1024 / 1024:.2f} MB")
        print("\n   ✓ Index files found!")
    else:
        print("\n   🔄 Index files not yet generated...")


def main():
    """Main monitoring function."""
    print("=" * 70)
    print("Background Tasks Progress Monitor")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    check_index_progress()
    check_postgres_progress()
    check_database_stats()
    check_index_status()

    print("\n" + "=" * 70)
    print("Monitoring complete!")
    print("\nTo continuously monitor:")
    print("  watch -n 10 python scripts/monitor_progress.py")
    print("\nOr view logs directly:")
    print("  tail -f /tmp/build_50k_index.log")
    print("  tail -f /tmp/postgres_import.log")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")
        sys.exit(0)
