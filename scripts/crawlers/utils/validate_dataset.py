#!/usr/bin/env python
"""
Dataset Validation Script

Validates collected JSONL dataset and generates statistics.

Usage:
    python scripts/crawlers/utils/validate_dataset.py --input data/raw/cna_mvp_dataset.jsonl

Author: CNIRS Development Team
License: Educational Use Only
"""

import json
import argparse
from pathlib import Path
from collections import Counter
from datetime import datetime
import statistics


def validate_dataset(input_file: str):
    """
    Validate dataset and generate statistics.

    Args:
        input_file: Path to JSONL dataset file

    Returns:
        dict: Validation results and statistics
    """
    input_path = Path(input_file)

    if not input_path.exists():
        print(f"Error: File not found: {input_file}")
        return None

    # Statistics
    stats = {
        'total_articles': 0,
        'valid_articles': 0,
        'missing_fields': Counter(),
        'empty_fields': Counter(),
        'categories': Counter(),
        'dates': Counter(),
        'authors': Counter(),
        'content_lengths': [],
        'tag_counts': [],
        'errors': []
    }

    # Required fields
    required_fields = ['source', 'article_id', 'url', 'title', 'content', 'category', 'category_name']

    # Optional fields that should not be None
    optional_fields = ['published_date', 'author', 'tags']

    print(f"Validating dataset: {input_file}")
    print("=" * 70)

    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            stats['total_articles'] += 1

            try:
                article = json.loads(line.strip())

                # Check required fields
                is_valid = True
                for field in required_fields:
                    if field not in article:
                        stats['missing_fields'][field] += 1
                        is_valid = False
                    elif not article[field]:
                        stats['empty_fields'][field] += 1
                        is_valid = False

                # Check optional fields
                for field in optional_fields:
                    if field not in article:
                        stats['missing_fields'][field] += 1
                    elif article[field] is None:
                        stats['empty_fields'][field] += 1

                if is_valid:
                    stats['valid_articles'] += 1

                # Collect statistics
                if 'category' in article:
                    stats['categories'][article['category']] += 1

                if 'published_date' in article and article['published_date']:
                    stats['dates'][article['published_date']] += 1

                if 'author' in article and article['author']:
                    stats['authors'][article['author']] += 1

                if 'content' in article and article['content']:
                    stats['content_lengths'].append(len(article['content']))

                if 'tags' in article and article['tags']:
                    stats['tag_counts'].append(len(article['tags']))

            except json.JSONDecodeError as e:
                stats['errors'].append(f"Line {line_num}: JSON decode error: {e}")
            except Exception as e:
                stats['errors'].append(f"Line {line_num}: {str(e)}")

    # Print statistics
    print(f"\n📊 Dataset Statistics")
    print("=" * 70)
    print(f"Total articles: {stats['total_articles']}")
    print(f"Valid articles: {stats['valid_articles']}")
    print(f"Invalid articles: {stats['total_articles'] - stats['valid_articles']}")

    if stats['errors']:
        print(f"\n❌ Errors ({len(stats['errors'])}):")
        for error in stats['errors'][:10]:  # Show first 10 errors
            print(f"  - {error}")

    if stats['missing_fields']:
        print(f"\n⚠️ Missing Fields:")
        for field, count in stats['missing_fields'].most_common():
            print(f"  - {field}: {count} articles")

    if stats['empty_fields']:
        print(f"\n⚠️ Empty/Null Fields:")
        for field, count in stats['empty_fields'].most_common():
            print(f"  - {field}: {count} articles")

    print(f"\n📁 Category Distribution:")
    for category, count in stats['categories'].most_common():
        print(f"  - {category}: {count} articles")

    if stats['dates']:
        print(f"\n📅 Date Distribution (Top 10):")
        for date, count in stats['dates'].most_common(10):
            print(f"  - {date}: {count} articles")

    if stats['authors']:
        print(f"\n✍️ Author Distribution (Top 10):")
        for author, count in stats['authors'].most_common(10):
            print(f"  - {author}: {count} articles")

    if stats['content_lengths']:
        print(f"\n📝 Content Length Statistics:")
        print(f"  - Min: {min(stats['content_lengths'])} characters")
        print(f"  - Max: {max(stats['content_lengths'])} characters")
        print(f"  - Mean: {statistics.mean(stats['content_lengths']):.1f} characters")
        print(f"  - Median: {statistics.median(stats['content_lengths']):.1f} characters")

    if stats['tag_counts']:
        print(f"\n🏷️ Tags per Article:")
        print(f"  - Min: {min(stats['tag_counts'])} tags")
        print(f"  - Max: {max(stats['tag_counts'])} tags")
        print(f"  - Mean: {statistics.mean(stats['tag_counts']):.1f} tags")
        print(f"  - Median: {statistics.median(stats['tag_counts'])} tags")
        print(f"  - Articles with tags: {len([c for c in stats['tag_counts'] if c > 0])}")
        print(f"  - Articles without tags: {stats['total_articles'] - len([c for c in stats['tag_counts'] if c > 0])}")

    print("\n" + "=" * 70)
    print(f"✅ Validation complete!")
    print(f"📊 Success rate: {stats['valid_articles']/stats['total_articles']*100:.1f}%")

    return stats


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Validate JSONL dataset')
    parser.add_argument('--input', type=str, required=True, help='Input JSONL file')

    args = parser.parse_args()

    validate_dataset(args.input)


if __name__ == '__main__':
    main()
