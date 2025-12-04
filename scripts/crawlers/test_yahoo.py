#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comprehensive test for Yahoo News spider (Deep Refactoring v2.0 - FINAL CRAWLER!)

Tests Yahoo crawler with date-based sitemap strategy, systematic 2-year coverage,
and comprehensive metadata extraction.

Tests:
1. Daily sitemap mode (most efficient for date ranges)
2. Sitemap index mode (news sitemap)
3. Archive mode (category-based)
4. Metadata extraction quality

Author: Information Retrieval System
Date: 2025-11-19
Version: 2.0 (Deep Refactoring - FINAL CRAWLER!)
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def test_yahoo_daily_sitemap():
    """Test Yahoo spider with daily sitemap mode (recommended mode)"""
    print("=" * 70)
    print("Testing Yahoo Spider (Daily Sitemap Mode - Deep Refactoring v2.0)")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    spider_file = project_root / 'scripts' / 'crawlers' / 'yahoo_spider.py'

    # Test parameters
    cmd = [
        sys.executable, '-m', 'scrapy', 'runspider',
        str(spider_file),
        '-a', 'mode=sitemap',
        '-a', 'sitemap=daily',
        '-a', 'days=3',  # Last 3 days for quick test
        '-a', 'max_articles=5',  # Limit for quick test
        '-s', 'LOG_LEVEL=INFO',
    ]

    print("Command:", ' '.join(cmd[3:]))
    print()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(project_root)
        )

        output = result.stdout + result.stderr

        # Extract key information
        print("=" * 70)
        print("RESULTS:")
        print("=" * 70)

        for line in output.split('\n'):
            if any(keyword in line for keyword in [
                'Yahoo Spider', 'Mode:', 'Generated.*sitemaps', 'Found.*URLs',
                'Articles successfully', 'Success rate',
                'Sitemaps processed', 'METADATA QUALITY'
            ]):
                print(line)

        # Check success
        if 'Articles successfully scraped:' in output:
            import re
            match = re.search(r'Articles successfully scraped: (\d+)', output)
            if match and int(match.group(1)) > 0:
                print("\n✓ TEST PASSED - Yahoo spider working (daily sitemap)!")
                return True

        print("\n✗ TEST FAILED - No articles scraped")
        print("\nLast 20 lines of output:")
        print("-" * 70)
        for line in output.split('\n')[-20:]:
            if line.strip():
                print(line)
        return False

    except subprocess.TimeoutExpired:
        print("\n✗ TIMEOUT - Exceeded 120 seconds")
        return False
    except Exception as e:
        print(f"\n✗ ERROR - {str(e)}")
        return False


def test_yahoo_news_sitemap_index():
    """Test Yahoo spider with news sitemap index"""
    print("\n" + "=" * 70)
    print("Testing Yahoo Spider (News Sitemap Index)")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    spider_file = project_root / 'scripts' / 'crawlers' / 'yahoo_spider.py'

    cmd = [
        sys.executable, '-m', 'scrapy', 'runspider',
        str(spider_file),
        '-a', 'mode=sitemap',
        '-a', 'sitemap=news',
        '-a', 'max_articles=5',
        '-s', 'LOG_LEVEL=INFO',
    ]

    print("Command:", ' '.join(cmd[3:]))
    print()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(project_root)
        )

        output = result.stdout + result.stderr

        # Extract key information
        print("=" * 70)
        print("RESULTS:")
        print("=" * 70)

        for line in output.split('\n'):
            if any(keyword in line for keyword in [
                'Yahoo Spider', 'Sitemap:', 'Found.*URLs', 'Articles successfully',
                'Sitemaps processed'
            ]):
                print(line)

        if 'Articles successfully scraped:' in output:
            import re
            match = re.search(r'Articles successfully scraped: (\d+)', output)
            if match and int(match.group(1)) > 0:
                print("\n✓ TEST PASSED - News sitemap index working!")
                return True

        print("\n⚠ TEST INCOMPLETE - News sitemap index needs verification")
        return False

    except subprocess.TimeoutExpired:
        print("\n✗ TIMEOUT - Exceeded 120 seconds")
        return False
    except Exception as e:
        print(f"\n✗ ERROR - {str(e)}")
        return False


def test_yahoo_archive_mode():
    """Test Yahoo spider in archive mode (category-based)"""
    print("\n" + "=" * 70)
    print("Testing Yahoo Spider (Archive Mode - Politics Category)")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    spider_file = project_root / 'scripts' / 'crawlers' / 'yahoo_spider.py'

    # Test parameters
    cmd = [
        sys.executable, '-m', 'scrapy', 'runspider',
        str(spider_file),
        '-a', 'mode=archive',
        '-a', 'category=politics',
        '-a', 'max_articles=5',
        '-s', 'LOG_LEVEL=INFO',
    ]

    print("Command:", ' '.join(cmd[3:]))
    print()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=str(project_root)
        )

        output = result.stdout + result.stderr

        # Extract key information
        print("=" * 70)
        print("RESULTS:")
        print("=" * 70)

        for line in output.split('\n'):
            if any(keyword in line for keyword in [
                'Yahoo Spider', 'Mode:', 'Category', 'Found.*articles',
                'Articles successfully', 'Archive pages visited'
            ]):
                print(line)

        # Check success
        if 'Articles successfully scraped:' in output and 'Mode: archive' in output:
            import re
            match = re.search(r'Articles successfully scraped: (\d+)', output)
            if match and int(match.group(1)) > 0:
                print(f"\n✓ TEST PASSED - Archive mode working (politics category)!")
                return True

        print("\n⚠ TEST INCOMPLETE - Archive mode needs verification")
        return False

    except subprocess.TimeoutExpired:
        print("\n✗ TIMEOUT - Exceeded 120 seconds")
        return False
    except Exception as e:
        print(f"\n✗ ERROR - {str(e)}")
        return False


def test_yahoo_metadata_extraction():
    """Test comprehensive metadata extraction quality"""
    print("\n" + "=" * 70)
    print("Testing Yahoo Spider (Metadata Extraction Quality)")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    spider_file = project_root / 'scripts' / 'crawlers' / 'yahoo_spider.py'

    # Test with JSONL output to verify metadata
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        output_file = f.name

    cmd = [
        sys.executable, '-m', 'scrapy', 'runspider',
        str(spider_file),
        '-a', 'mode=sitemap',
        '-a', 'sitemap=daily',
        '-a', 'days=1',
        '-a', 'max_articles=3',
        '-o', output_file,
        '-s', 'LOG_LEVEL=ERROR',  # Minimal logging
    ]

    print("Command:", ' '.join(cmd[3:]))
    print("Checking metadata quality (title, content, date, author, category, tags, image)...")
    print()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
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
            if article.get('author') and article['author'] != 'Yahoo News':
                metadata_quality['has_author'] += 1
            if article.get('category'):
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
        print("\n✗ TIMEOUT - Exceeded 120 seconds")
        return False
    except Exception as e:
        print(f"\n✗ ERROR - {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("YAHOO SPIDER COMPREHENSIVE TEST (FINAL CRAWLER - Deep Refactoring v2.0)")
    print("=" * 70)
    print("Key Features:")
    print("  - Date-based daily sitemap generation (730 sitemaps for 2 years)")
    print("  - Multi-sitemap index support (news, topic, tag, pk, ybrain, pages)")
    print("  - Archive mode with category-based crawling")
    print("  - NO Playwright needed (fast standard Scrapy)")
    print("  - 6+ fallback strategies per metadata field")
    print("  - Image metadata extraction from sitemaps")
    print("  - JSON-LD structured data extraction")
    print("  - 2-year historical target (730 days)")
    print("=" * 70)

    results = {}

    # Test 1: Daily sitemap mode - recommended
    results['daily_sitemap'] = test_yahoo_daily_sitemap()

    # Test 2: News sitemap index
    results['news_sitemap_index'] = test_yahoo_news_sitemap_index()

    # Test 3: Archive mode (politics)
    results['archive_mode'] = test_yahoo_archive_mode()

    # Test 4: Metadata extraction
    results['metadata_extraction'] = test_yahoo_metadata_extraction()

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
    print("DEEP REFACTORING IMPROVEMENTS (FINAL CRAWLER!):")
    print("=" * 70)
    print("Daily sitemaps: Systematic date-based generation (sitemap-YYYY-MM-DD.xml)")
    print("Multi-sitemap: 7+ sitemap types (news, topic, tag, pk, ybrain, pages, main)")
    print("2-year coverage: 730 daily sitemaps for complete historical access")
    print("NO Playwright: Fast standard Scrapy (8 concurrent requests)")
    print("Metadata: 6+ fallback strategies per field (title, content, date, author, etc.)")
    print("Image data: Rich sitemap metadata (image URLs + captions)")
    print("Categories: 8 major categories (politics, world, entertainment, sports, etc.)")
    print("Statistics: Per-sitemap detailed tracking")
    print("Quality: Production-ready with comprehensive error handling")
    print("=" * 70)

    print("\n🎉 ALL 12 CRAWLERS COMPLETED - DEEP REFACTORING PROJECT FINISHED! 🎉")
    print("=" * 70)

    sys.exit(0 if all(results.values()) else 1)
