#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test SETN (三立新聞網) Spider (Deep Refactoring v2.0)

Tests:
1. Google News sitemap mode
2. Sitemap index mode (category sitemaps)
3. Metadata extraction quality

Author: Information Retrieval System
Date: 2025-11-19
Version: 2.0
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_setn_google_news():
    """Test SETN spider with Google News sitemap"""
    print("=" * 70)
    print("Testing SETN Spider (Google News Sitemap)")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    spider_file = project_root / 'scripts' / 'crawlers' / 'setn_spider.py'

    cmd = [
        sys.executable, '-m', 'scrapy', 'runspider',
        str(spider_file),
        '-a', 'mode=sitemap',
        '-a', 'sitemap=google_news',
        '-a', 'max_articles=5',
        '-s', 'LOG_LEVEL=INFO',
        '-s', 'ROBOTSTXT_OBEY=False',
    ]

    print("Command:", ' '.join(cmd[3:]), "\n")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=str(project_root))
        output = result.stdout + result.stderr

        print("=" * 70)
        print("RESULTS:")
        print("=" * 70)

        for line in output.split('\n'):
            if any(kw in line for kw in ['SETN Spider', 'Mode:', 'Found.*URLs', 'Articles successfully', 'Success rate', 'METADATA QUALITY']):
                print(line)

        if 'Articles successfully scraped:' in output:
            import re
            match = re.search(r'Articles successfully scraped: (\d+)', output)
            if match and int(match.group(1)) > 0:
                print("\n✓ TEST PASSED - SETN spider working (Google News)!")
                return True

        print("\n✗ TEST FAILED - No articles scraped")
        return False

    except subprocess.TimeoutExpired:
        print("\n✗ TIMEOUT")
        return False
    except Exception as e:
        print(f"\n✗ ERROR - {str(e)}")
        return False


def test_setn_sitemap_index():
    """Test SETN spider with sitemap index (category sitemaps)"""
    print("\n" + "=" * 70)
    print("Testing SETN Spider (Sitemap Index - Category Sitemaps)")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    spider_file = project_root / 'scripts' / 'crawlers' / 'setn_spider.py'

    cmd = [
        sys.executable, '-m', 'scrapy', 'runspider',
        str(spider_file),
        '-a', 'mode=sitemap',
        '-a', 'sitemap=index',
        '-a', 'category=news',  # Politics category
        '-a', 'max_articles=5',
        '-s', 'LOG_LEVEL=INFO',
        '-s', 'ROBOTSTXT_OBEY=False',
    ]

    print("Command:", ' '.join(cmd[3:]), "\n")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=str(project_root))
        output = result.stdout + result.stderr

        print("=" * 70)
        print("RESULTS:")
        print("=" * 70)

        for line in output.split('\n'):
            if any(kw in line for kw in ['SETN Spider', 'sitemap index', 'Category sitemaps', 'Articles successfully']):
                print(line)

        if 'Articles successfully scraped:' in output:
            import re
            match = re.search(r'Articles successfully scraped: (\d+)', output)
            if match and int(match.group(1)) > 0:
                print("\n✓ TEST PASSED - Sitemap index mode working!")
                return True

        print("\n⚠ TEST INCOMPLETE")
        return False

    except subprocess.TimeoutExpired:
        print("\n✗ TIMEOUT")
        return False
    except Exception as e:
        print(f"\n✗ ERROR - {str(e)}")
        return False


def test_setn_metadata():
    """Test metadata extraction quality"""
    print("\n" + "=" * 70)
    print("Testing SETN Spider (Metadata Quality)")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    spider_file = project_root / 'scripts' / 'crawlers' / 'setn_spider.py'

    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        output_file = f.name

    cmd = [
        sys.executable, '-m', 'scrapy', 'runspider',
        str(spider_file),
        '-a', 'mode=sitemap',
        '-a', 'max_articles=3',
        '-o', output_file,
        '-s', 'LOG_LEVEL=ERROR',
        '-s', 'ROBOTSTXT_OBEY=False',
    ]

    print("Command:", ' '.join(cmd[3:]), "\n")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=str(project_root))

        import json
        articles = []
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        articles.append(json.loads(line))
        except:
            pass

        print("=" * 70)
        print("METADATA QUALITY CHECK:")
        print("=" * 70)

        if not articles:
            print("✗ No articles found")
            return False

        print(f"Scraped {len(articles)} articles\n")

        quality = {'has_title': 0, 'has_content': 0, 'has_date': 0, 'has_tags': 0}

        for article in articles:
            if article.get('title'):
                quality['has_title'] += 1
            if article.get('content') and len(article['content']) > 50:
                quality['has_content'] += 1
            if article.get('published_date'):
                quality['has_date'] += 1
            if article.get('tags') and len(article['tags']) > 0:
                quality['has_tags'] += 1

        total = len(articles)
        print("Metadata Completeness:")
        for field, count in quality.items():
            pct = 100 * count / total
            status = "✓" if pct >= 80 else "⚠" if pct >= 50 else "✗"
            print(f"  {status} {field.replace('has_', '').title()}: {count}/{total} ({pct:.1f}%)")

        Path(output_file).unlink(missing_ok=True)

        core_quality = quality['has_title'] >= total and quality['has_content'] >= total

        if core_quality:
            print("\n✓ TEST PASSED - Metadata extraction working!")
            return True
        else:
            print("\n✗ TEST FAILED")
            return False

    except subprocess.TimeoutExpired:
        print("\n✗ TIMEOUT")
        return False
    except Exception as e:
        print(f"\n✗ ERROR - {str(e)}")
        return False


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("SETN SPIDER COMPREHENSIVE TEST (Deep Refactoring v2.0)")
    print("=" * 70)
    print("Key Features:")
    print("  - Multi-sitemap support (Google News + category sitemaps)")
    print("  - 2-mode crawling (sitemap/list)")
    print("  - 6+ fallback strategies per metadata field")
    print("  - NO Playwright needed (standard Scrapy)")
    print("  - Recent articles only (historical IDs not accessible)")
    print("=" * 70)

    results = {}

    results['google_news'] = test_setn_google_news()
    results['sitemap_index'] = test_setn_sitemap_index()
    results['metadata'] = test_setn_metadata()

    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    for test_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status} - {test_name}")

    print("\n" + "=" * 70)
    print("FEATURES:")
    print("=" * 70)
    print("Sitemap: Google News + Category sitemaps (index)")
    print("Metadata: Title, content, date, author, keywords/tags from sitemap")
    print("Speed: Fast (no Playwright overhead)")
    print("Limitation: Historical IDs not accessible (recent articles only)")
    print("=" * 70)

    sys.exit(0 if all(results.values()) else 1)
