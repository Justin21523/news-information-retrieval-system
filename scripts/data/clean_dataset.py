#!/usr/bin/env python
"""
Dataset Cleaning and Deduplication Script

Cleans and deduplicates collected news articles.

Features:
- Remove invalid JSON entries
- Deduplicate by article_id
- Standardize date formats
- Clean text content
- Validate required fields

Usage:
    python scripts/data/clean_dataset.py \
        --input data/raw/cna_mvp_dataset.jsonl \
        --output data/processed/cna_mvp_cleaned.jsonl

Author: CNIRS Development Team
License: Educational Use Only
"""

import json
import argparse
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Clean text content.

    Args:
        text: Raw text string

    Returns:
        str: Cleaned text
    """
    if not text:
        return ""

    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)

    # Remove leading/trailing whitespace
    text = text.strip()

    # Remove special characters that might cause issues
    # Keep Chinese, English, numbers, and common punctuation
    # text = re.sub(r'[^\u4e00-\u9fff\u3000-\u303fa-zA-Z0-9\s.,;:!?()\[\]{}<>\'\"，。；：！？（）「」『』【】《》、]', '', text)

    return text


def standardize_date(date_str: str) -> str:
    """
    Standardize date format to YYYY-MM-DD.

    Args:
        date_str: Date string in various formats

    Returns:
        str: Standardized date (YYYY-MM-DD) or None if invalid
    """
    if not date_str:
        return None

    # Already in YYYY-MM-DD format
    if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
        return date_str

    # Try ISO format (YYYY-MM-DDTHH:MM:SS)
    if 'T' in date_str:
        date_str = date_str.split('T')[0]
        if re.match(r'^\d{4}-\d{2}-\d{2}$', date_str):
            return date_str

    # Try YYYY/MM/DD format
    match = re.match(r'(\d{4})/(\d{1,2})/(\d{1,2})', date_str)
    if match:
        year, month, day = match.groups()
        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"

    logger.warning(f"Could not parse date: {date_str}")
    return None


def validate_article(article: dict, required_fields: list) -> tuple:
    """
    Validate article has required fields and valid data.

    Args:
        article: Article dictionary
        required_fields: List of required field names

    Returns:
        tuple: (is_valid: bool, error_message: str)
    """
    # Check required fields exist
    for field in required_fields:
        if field not in article:
            return False, f"Missing required field: {field}"
        if not article[field]:
            return False, f"Empty required field: {field}"

    # Validate article_id format (should be digits)
    if 'article_id' in article:
        if not re.match(r'^\d+$', str(article['article_id'])):
            return False, f"Invalid article_id format: {article['article_id']}"

    # Validate URL format
    if 'url' in article:
        if not article['url'].startswith('http'):
            return False, f"Invalid URL: {article['url']}"

    # Check content length
    if 'content' in article:
        if len(article['content']) < 50:
            return False, f"Content too short: {len(article['content'])} chars"

    return True, None


def clean_dataset(input_file: str, output_file: str):
    """
    Clean and deduplicate dataset.

    Args:
        input_file: Path to input JSONL file
        output_file: Path to output cleaned JSONL file
    """
    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.exists():
        logger.error(f"Input file not found: {input_file}")
        return

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Required fields
    required_fields = ['source', 'article_id', 'url', 'title', 'content']

    # Statistics
    stats = {
        'total_lines': 0,
        'json_errors': 0,
        'validation_errors': 0,
        'duplicates': 0,
        'cleaned': 0,
    }

    # Track seen article IDs for deduplication
    seen_ids = set()

    # Validation errors
    validation_errors = defaultdict(int)

    logger.info(f"Cleaning dataset: {input_file}")
    logger.info(f"Output: {output_file}")
    logger.info("=" * 70)

    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:

        for line_num, line in enumerate(infile, 1):
            stats['total_lines'] += 1

            # Skip empty lines
            if not line.strip():
                stats['json_errors'] += 1
                logger.debug(f"Line {line_num}: Empty line")
                continue

            # Parse JSON
            try:
                article = json.loads(line.strip())
            except json.JSONDecodeError as e:
                stats['json_errors'] += 1
                logger.warning(f"Line {line_num}: JSON decode error: {e}")
                continue

            # Validate article
            is_valid, error_msg = validate_article(article, required_fields)
            if not is_valid:
                stats['validation_errors'] += 1
                validation_errors[error_msg] += 1
                logger.debug(f"Line {line_num}: {error_msg}")
                continue

            # Check for duplicates
            article_id = article['article_id']
            if article_id in seen_ids:
                stats['duplicates'] += 1
                logger.debug(f"Line {line_num}: Duplicate article_id: {article_id}")
                continue
            seen_ids.add(article_id)

            # Clean article data
            cleaned_article = {
                'source': article['source'],
                'article_id': article['article_id'],
                'url': article['url'],
                'title': clean_text(article['title']),
                'content': clean_text(article['content']),
                'category': article.get('category', ''),
                'category_name': article.get('category_name', ''),
                'published_date': standardize_date(article.get('published_date')),
                'author': clean_text(article.get('author', '')),
                'tags': article.get('tags', []) if article.get('tags') else [],
                'crawled_at': article.get('crawled_at', ''),
            }

            # Write cleaned article
            outfile.write(json.dumps(cleaned_article, ensure_ascii=False) + '\n')
            stats['cleaned'] += 1

    # Print statistics
    logger.info("\n" + "=" * 70)
    logger.info("📊 Cleaning Statistics")
    logger.info("=" * 70)
    logger.info(f"Total lines processed: {stats['total_lines']}")
    logger.info(f"JSON parse errors: {stats['json_errors']}")
    logger.info(f"Validation errors: {stats['validation_errors']}")
    logger.info(f"Duplicate articles: {stats['duplicates']}")
    logger.info(f"Cleaned articles: {stats['cleaned']}")

    if validation_errors:
        logger.info("\n⚠️ Validation Error Breakdown:")
        for error, count in sorted(validation_errors.items(), key=lambda x: -x[1]):
            logger.info(f"  - {error}: {count}")

    logger.info("\n" + "=" * 70)
    logger.info(f"✅ Cleaning complete!")
    logger.info(f"📁 Output saved to: {output_file}")
    logger.info(f"📊 Success rate: {stats['cleaned']/stats['total_lines']*100:.1f}%")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Clean and deduplicate JSONL dataset')
    parser.add_argument('--input', type=str, required=True, help='Input JSONL file')
    parser.add_argument('--output', type=str, required=True, help='Output JSONL file')

    args = parser.parse_args()

    clean_dataset(args.input, args.output)


if __name__ == '__main__':
    main()
