#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comprehensive test for CTI (ChinaTimes) spider (Deep Refactoring v2.0)

Tests CTI crawler with multi-sitemap strategy, list mode with deep pagination,
Playwright-based Cloudflare bypass, and comprehensive metadata extraction.

Tests:
1. Sitemap mode (todaynews) - most recent
2. Sitemap mode (todaynews_d2) - 2-day coverage
3. List mode (politics category)
4. Metadata extraction quality

Author: Information Retrieval System
Date: 2025-11-19
Version: 2.0 (Deep Refactoring)
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_cti_sitemap_todaynews():
    """Test CTI spider with today's news sitemap (recommended mode)"""
    print("=" * 70)
    print("Testing CTI Spider (Sitemap Mode: todaynews - Deep Refactoring v2.0)")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    spider_file = project_root / 'scripts' / 'crawlers' / 'cti_spider.py'

    # Test parameters
    cmd = [
        sys.executable, '-m', 'scrapy', 'runspider',
        str(spider_file),
        '-a', 'mode=sitemap',
        '-a', 'sitemap=todaynews',
        '-a', 'max_articles=5',  # Limit for quick test
        '-s', 'LOG_LEVEL=INFO',
        '-s', 'ROBOTSTXT_OBEY=False',
    ]

    print("Command:", ' '.join(cmd[3:]))
    print()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes for Playwright + Cloudflare
            cwd=str(project_root)
        )

        output = result.stdout + result.stderr

        # Extract key information
        print("=" * 70)
        print("RESULTS:")
        print("=" * 70)

        for line in output.split('\n'):
            if any(keyword in line for keyword in [
                'CTI Spider', 'Mode:', 'Sitemap:', 'Found.*URLs',
                'Articles successfully', 'Success rate',
                'Sitemap URLs discovered', 'METADATA QUALITY', 'Cloudflare'
            ]):
                print(line)

        # Check success
        if 'Articles successfully scraped:' in output:
            import re
            match = re.search(r'Articles successfully scraped: (\d+)', output)
            if match and int(match.group(1)) > 0:
                print("\n✓ TEST PASSED - CTI spider working (sitemap:todaynews)!")
                return True

        print("\n✗ TEST FAILED - No articles scraped")
        print("\nLast 30 lines of output:")
        print("-" * 70)
        for line in output.split('\n')[-30:]:
            if line.strip():
                print(line)
        return False

    except subprocess.TimeoutExpired:
        print("\n✗ TIMEOUT - Exceeded 300 seconds")
        return False
    except Exception as e:
        print(f"\n✗ ERROR - {str(e)}")
        return False


def test_cti_sitemap_todaynews_d2():
    """Test CTI spider with 2-day news sitemap"""
    print("\n" + "=" * 70)
    print("Testing CTI Spider (Sitemap Mode: todaynews_d2)")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    spider_file = project_root / 'scripts' / 'crawlers' / 'cti_spider.py'

    cmd = [
        sys.executable, '-m', 'scrapy', 'runspider',
        str(spider_file),
        '-a', 'mode=sitemap',
        '-a', 'sitemap=todaynews_d2',
        '-a', 'max_articles=5',
        '-s', 'LOG_LEVEL=INFO',
        '-s', 'ROBOTSTXT_OBEY=False',
    ]

    print("Command:", ' '.join(cmd[3:]))
    print()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(project_root)
        )

        output = result.stdout + result.stderr

        # Extract key information
        print("=" * 70)
        print("RESULTS:")
        print("=" * 70)

        for line in output.split('\n'):
            if any(keyword in line for keyword in [
                'CTI Spider', 'Sitemap:', 'Found.*URLs', 'Articles successfully',
                'Sitemap URLs discovered'
            ]):
                print(line)

        if 'Articles successfully scraped:' in output:
            import re
            match = re.search(r'Articles successfully scraped: (\d+)', output)
            if match and int(match.group(1)) > 0:
                print("\n✓ TEST PASSED - 2-day sitemap working!")
                return True

        print("\n⚠ TEST INCOMPLETE - 2-day sitemap needs verification")
        return False

    except subprocess.TimeoutExpired:
        print("\n✗ TIMEOUT - Exceeded 300 seconds")
        return False
    except Exception as e:
        print(f"\n✗ ERROR - {str(e)}")
        return False


def test_cti_list_mode():
    """Test CTI spider in list mode (category-based with deep pagination)"""
    print("\n" + "=" * 70)
    print("Testing CTI Spider (List Mode - Politics Category)")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    spider_file = project_root / 'scripts' / 'crawlers' / 'cti_spider.py'

    # Test parameters
    cmd = [
        sys.executable, '-m', 'scrapy', 'runspider',
        str(spider_file),
        '-a', 'mode=list',
        '-a', 'category=politics',
        '-a', 'days=3',  # 3 days for quick test
        '-a', 'max_articles=5',
        '-s', 'LOG_LEVEL=INFO',
        '-s', 'ROBOTSTXT_OBEY=False',
    ]

    print("Command:", ' '.join(cmd[3:]))
    print()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(project_root)
        )

        output = result.stdout + result.stderr

        # Extract key information
        print("=" * 70)
        print("RESULTS:")
        print("=" * 70)

        for line in output.split('\n'):
            if any(keyword in line for keyword in [
                'CTI Spider', 'Mode:', 'Category', 'Found.*articles',
                'Articles successfully', 'List pages visited'
            ]):
                print(line)

        # Check success
        if 'Articles successfully scraped:' in output and 'Mode: list' in output:
            import re
            match = re.search(r'Articles successfully scraped: (\d+)', output)
            if match and int(match.group(1)) > 0:
                print(f"\n✓ TEST PASSED - List mode working (politics category)!")
                return True

        print("\n⚠ TEST INCOMPLETE - List mode needs verification")
        return False

    except subprocess.TimeoutExpired:
        print("\n✗ TIMEOUT - Exceeded 300 seconds")
        return False
    except Exception as e:
        print(f"\n✗ ERROR - {str(e)}")
        return False


def test_cti_metadata_extraction():
    """Test comprehensive metadata extraction quality"""
    print("\n" + "=" * 70)
    print("Testing CTI Spider (Metadata Extraction Quality)")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    spider_file = project_root / 'scripts' / 'crawlers' / 'cti_spider.py'

    # Test with JSONL output to verify metadata
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        output_file = f.name

    cmd = [
        sys.executable, '-m', 'scrapy', 'runspider',
        str(spider_file),
        '-a', 'mode=sitemap',
        '-a', 'sitemap=todaynews',
        '-a', 'max_articles=3',
        '-o', output_file,
        '-s', 'LOG_LEVEL=ERROR',  # Minimal logging
        '-s', 'ROBOTSTXT_OBEY=False',
    ]

    print("Command:", ' '.join(cmd[3:]))
    print("Checking metadata quality (title, content, date, author, category, tags)...")
    print()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(project_root)
        )

        # Read and analyze output
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

        print(f"Scraped {len(articles)} articles for metadata analysis\n")

        metadata_quality = {
            'has_title': 0,
            'has_content': 0,
            'has_date': 0,
            'has_author': 0,
            'has_category': 0,
            'has_tags': 0,
            'has_image': 0,
            'has_description': 0,
        }

        for article in articles:
            if article.get('title'):
                metadata_quality['has_title'] += 1
            if article.get('content') and len(article['content']) > 100:
                metadata_quality['has_content'] += 1
            if article.get('published_date'):
                metadata_quality['has_date'] += 1
            if article.get('author') and article['author'] != 'CTI News':
                metadata_quality['has_author'] += 1
            if article.get('category') and article['category'] != 'news':
                metadata_quality['has_category'] += 1
            if article.get('tags') and len(article['tags']) > 0:
                metadata_quality['has_tags'] += 1
            if article.get('image_url'):
                metadata_quality['has_image'] += 1
            if article.get('description'):
                metadata_quality['has_description'] += 1

        # Print quality report
        total = len(articles)
        print("Metadata Completeness:")
        for field, count in metadata_quality.items():
            percentage = 100 * count / total
            status = "✓" if percentage >= 80 else "⚠" if percentage >= 50 else "✗"
            print(f"  {status} {field.replace('has_', '').title()}: {count}/{total} ({percentage:.1f}%)")

        # Cleanup
        Path(output_file).unlink(missing_ok=True)

        # Success if core fields are present
        core_quality = (
            metadata_quality['has_title'] >= total and
            metadata_quality['has_content'] >= total and
            metadata_quality['has_date'] >= total * 0.8  # 80% threshold for dates
        )

        if core_quality:
            print("\n✓ TEST PASSED - Comprehensive metadata extraction working!")
            return True
        else:
            print("\n✗ TEST FAILED - Metadata extraction incomplete")
            return False

    except subprocess.TimeoutExpired:
        print("\n✗ TIMEOUT - Exceeded 300 seconds")
        return False
    except Exception as e:
        print(f"\n✗ ERROR - {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("CTI SPIDER COMPREHENSIVE TEST (Deep Refactoring v2.0)")
    print("=" * 70)
    print("Key Features:")
    print("  - Multi-sitemap support (8+ sitemaps from robots.txt)")
    print("  - List mode with deep pagination (500 pages)")
    print("  - Playwright with Cloudflare bypass")
    print("  - 6+ fallback strategies per metadata field")
    print("  - JSON-LD structured data extraction")
    print("  - 2-year historical target (730 days)")
    print("  - Comprehensive statistics tracking")
    print("=" * 70)

    results = {}

    # Test 1: Sitemap mode (todaynews) - recommended
    results['sitemap_todaynews'] = test_cti_sitemap_todaynews()

    # Test 2: Sitemap mode (todaynews_d2)
    results['sitemap_todaynews_d2'] = test_cti_sitemap_todaynews_d2()

    # Test 3: List mode (politics)
    results['list_mode'] = test_cti_list_mode()

    # Test 4: Metadata extraction
    results['metadata_extraction'] = test_cti_metadata_extraction()

    # Summary
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
    print("DEEP REFACTORING IMPROVEMENTS:")
    print("=" * 70)
    print("Multi-sitemap: 8+ sitemaps (todaynews, todaynews_d2, article_all, etc.)")
    print("List mode: Deep pagination up to 500 pages per category")
    print("Cloudflare: Playwright-based bypass with anti-detection")
    print("Metadata: 6+ fallback strategies per field (title, content, date, author, etc.)")
    print("Historical: 2-year target (730 days) via date-based URL filtering")
    print("Categories: All major categories (politics, money, society, world, etc.)")
    print("Statistics: Per-sitemap and per-category detailed tracking")
    print("Quality: Production-ready with comprehensive error handling")
    print("=" * 70)

    sys.exit(0 if all(results.values()) else 1)
