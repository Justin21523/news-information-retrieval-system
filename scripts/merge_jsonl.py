#!/usr/bin/env python3
"""
Merge Multiple JSONL Files for CNIRS

This script merges multiple news source JSONL files into a single file,
normalizing field names for consistency.

Usage:
    python scripts/merge_jsonl.py --output data/raw/merged_14days.jsonl

Author: CNIRS Project
"""

import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Source files to merge
SOURCE_FILES = [
    'ltn_14days.jsonl',
    'nextapple_14days.jsonl',
    'setn_14days.jsonl',
    'udn_14days.jsonl',
    'pts_14days.jsonl',
]

# Required fields for the merged output
REQUIRED_FIELDS = [
    'article_id',
    'title',
    'content',
    'source',
    'category',
    'published_date',
]


def normalize_record(record: Dict[str, Any], source_file: str) -> Dict[str, Any]:
    """
    Normalize a record to have consistent field names.

    Args:
        record: Original record from JSONL
        source_file: Name of the source file (for debugging)

    Returns:
        Normalized record with standard field names
    """
    normalized = {}

    # Direct mappings
    normalized['article_id'] = record.get('article_id', '')
    normalized['url'] = record.get('url', '')
    normalized['source'] = record.get('source', '')
    normalized['source_name'] = record.get('source_name', '')
    normalized['title'] = record.get('title', '')
    normalized['content'] = record.get('content', '')
    normalized['author'] = record.get('author', '')
    normalized['tags'] = record.get('tags', [])
    normalized['image_url'] = record.get('image_url', '')
    normalized['crawled_at'] = record.get('crawled_at', '')

    # Normalize date field (publish_date / published_date)
    date_value = record.get('published_date') or record.get('publish_date', '')
    normalized['published_date'] = date_value

    # Normalize category fields
    normalized['category'] = record.get('category', '')
    # Use category_name if available, else category_code, else category
    category_name = record.get('category_name') or record.get('category_code') or record.get('category', '')
    normalized['category_name'] = category_name

    # Optional fields
    normalized['description'] = record.get('description', '')

    return normalized


def validate_record(record: Dict[str, Any]) -> bool:
    """
    Validate that a record has all required fields with non-empty values.

    Args:
        record: Record to validate

    Returns:
        True if valid, False otherwise
    """
    # Must have article_id
    if not record.get('article_id'):
        return False

    # Must have title
    if not record.get('title'):
        return False

    # Must have content (at least 10 characters)
    content = record.get('content', '')
    if not content or len(content) < 10:
        return False

    return True


def merge_jsonl_files(data_dir: Path, output_file: Path) -> Dict[str, int]:
    """
    Merge multiple JSONL files into a single output file.

    Args:
        data_dir: Directory containing source JSONL files
        output_file: Path to output merged JSONL file

    Returns:
        Statistics dictionary with counts per source
    """
    stats = {
        'total': 0,
        'valid': 0,
        'skipped': 0,
        'by_source': {}
    }

    seen_ids = set()  # For deduplication

    with open(output_file, 'w', encoding='utf-8') as outf:
        for source_file in SOURCE_FILES:
            file_path = data_dir / source_file

            if not file_path.exists():
                logger.warning(f"Source file not found: {file_path}")
                continue

            source_count = 0
            source_valid = 0

            logger.info(f"Processing: {source_file}")

            with open(file_path, 'r', encoding='utf-8') as inf:
                for line_num, line in enumerate(inf, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        record = json.loads(line)
                        source_count += 1
                        stats['total'] += 1

                        # Normalize the record
                        normalized = normalize_record(record, source_file)

                        # Deduplicate by article_id
                        article_id = normalized.get('article_id', '')
                        if article_id in seen_ids:
                            stats['skipped'] += 1
                            continue
                        seen_ids.add(article_id)

                        # Validate
                        if not validate_record(normalized):
                            stats['skipped'] += 1
                            continue

                        # Write to output
                        outf.write(json.dumps(normalized, ensure_ascii=False) + '\n')
                        source_valid += 1
                        stats['valid'] += 1

                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON error in {source_file} line {line_num}: {e}")
                        stats['skipped'] += 1

            stats['by_source'][source_file] = {
                'total': source_count,
                'valid': source_valid
            }
            logger.info(f"  {source_file}: {source_valid}/{source_count} valid records")

    return stats


def main():
    parser = argparse.ArgumentParser(description='Merge JSONL files for CNIRS')
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/raw',
        help='Directory containing source JSONL files'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/raw/merged_14days.jsonl',
        help='Output merged JSONL file'
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_file = Path(args.output)

    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return 1

    # Create output directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("CNIRS JSONL Merger")
    logger.info("=" * 60)
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output file: {output_file}")
    logger.info("")

    # Merge files
    start_time = datetime.now()
    stats = merge_jsonl_files(data_dir, output_file)
    elapsed = (datetime.now() - start_time).total_seconds()

    # Print summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("MERGE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total records processed: {stats['total']}")
    logger.info(f"Valid records written: {stats['valid']}")
    logger.info(f"Skipped (invalid/duplicate): {stats['skipped']}")
    logger.info(f"Elapsed time: {elapsed:.2f} seconds")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Output size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")

    return 0


if __name__ == '__main__':
    exit(main())
