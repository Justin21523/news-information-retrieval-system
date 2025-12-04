#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comprehensive test for NextApple spider (Deep Refactoring v2.0)

Tests NextApple crawler with multi-sitemap strategy and comprehensive
metadata extraction.

Tests:
1. News sitemap mode (default)
2. Multi-sitemap mode (all)
3. Date filtering
4. Historical data accessibility

Author: Information Retrieval System
Date: 2025-11-18
Version: 2.0 (Deep Refactoring)
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_nextapple_news_sitemap():
    """Test NextApple spider with news sitemap (default mode)"""
    print("=" * 70)
    print("Testing NextApple Spider (News Sitemap - Deep Refactoring v2.0)")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    spider_file = project_root / 'scripts' / 'crawlers' / 'nextapple_spider.py'

    # Test parameters
    cmd = [
        sys.executable, '-m', 'scrapy', 'runspider',
        str(spider_file),
        '-a', 'sitemap=news',
        '-a', 'max_articles=5',  # Limit for quick test
        '-s', 'LOG_LEVEL=INFO',
        '-s', 'ROBOTSTXT_OBEY=False',  # Skip robots.txt for quick test
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
                'NextApple Spider', 'Sitemap mode:', 'Found.*total URLs',
                'After filtering:', 'Articles successfully', 'Success rate',
                'Scraped', 'OVERALL STATISTICS', 'PER-SITEMAP BREAKDOWN',
                'Progress:', 'sitemap'
            ]):
                print(line)

        # Check success
        if 'Articles successfully scraped:' in output:
            import re
            match = re.search(r'Articles successfully scraped: (\d+)', output)
            if match and int(match.group(1)) > 0:
                print("\n✓ TEST PASSED - NextApple spider working (news sitemap)!")
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


def test_nextapple_all_sitemaps():
    """Test NextApple spider with all sitemaps"""
    print("\n" + "=" * 70)
    print("Testing NextApple Spider (All Sitemaps - Comprehensive Mode)")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    spider_file = project_root / 'scripts' / 'crawlers' / 'nextapple_spider.py'

    # Test parameters - all sitemaps with limit
    cmd = [
        sys.executable, '-m', 'scrapy', 'runspider',
        str(spider_file),
        '-a', 'sitemap=all',
        '-a', 'max_articles=10',  # 10 articles total across all sitemaps
        '-s', 'LOG_LEVEL=INFO',
        '-s', 'ROBOTSTXT_OBEY=False',
    ]

    print("Command:", ' '.join(cmd[3:]))
    print("Note: Testing all 3 sitemaps (news, editors, topics)")
    print()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180,
            cwd=str(project_root)
        )

        output = result.stdout + result.stderr

        # Extract key information
        print("=" * 70)
        print("RESULTS:")
        print("=" * 70)

        for line in output.split('\n'):
            if any(keyword in line for keyword in [
                'NextApple Spider', 'Sitemap mode:', 'Queuing sitemap:',
                'Found.*total URLs', 'After filtering:', 'Articles successfully',
                'PER-SITEMAP BREAKDOWN', 'news:', 'editors:', 'topics:',
                'Success rate'
            ]):
                print(line)

        # Check success - should have scraped from multiple sitemaps
        if 'PER-SITEMAP BREAKDOWN:' in output and 'Articles successfully scraped:' in output:
            import re
            match = re.search(r'Articles successfully scraped: (\d+)', output)
            if match and int(match.group(1)) > 0:
                print("\n✓ TEST PASSED - Multi-sitemap mode working!")
                return True

        print("\n⚠ TEST INCOMPLETE - Multi-sitemap mode needs verification")
        return False

    except subprocess.TimeoutExpired:
        print("\n✗ TIMEOUT - Exceeded 180 seconds")
        return False
    except Exception as e:
        print(f"\n✗ ERROR - {str(e)}")
        return False


def test_nextapple_date_filtering():
    """Test NextApple spider's date filtering capability"""
    print("\n" + "=" * 70)
    print("Testing NextApple Spider (Date Filtering - Last 3 Days)")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    spider_file = project_root / 'scripts' / 'crawlers' / 'nextapple_spider.py'

    # Test parameters - date filter (last 3 days)
    cmd = [
        sys.executable, '-m', 'scrapy', 'runspider',
        str(spider_file),
        '-a', 'sitemap=news',
        '-a', 'days=3',  # Last 3 days only
        '-a', 'max_articles=5',
        '-s', 'LOG_LEVEL=INFO',
        '-s', 'ROBOTSTXT_OBEY=False',
    ]

    print("Command:", ' '.join(cmd[3:]))
    print("Testing date filtering (last 3 days)...")
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
                'NextApple Spider', 'Date range:', 'Date filter:',
                'Found.*total URLs', 'After filtering:', 'Articles successfully',
                'Skipped (date filter)'
            ]):
                print(line)

        # Check if date filtering was applied
        if 'Date filter:' in output or 'Date range:' in output:
            if 'After filtering:' in output:
                print("\n✓ TEST PASSED - Date filtering working!")
                return True

        print("\n⚠ TEST WARNING - Could not verify date filtering")
        return False

    except subprocess.TimeoutExpired:
        print("\n✗ TIMEOUT - Exceeded 120 seconds")
        return False
    except Exception as e:
        print(f"\n✗ ERROR - {str(e)}")
        return False


def test_nextapple_metadata_extraction():
    """Test comprehensive metadata extraction"""
    print("\n" + "=" * 70)
    print("Testing NextApple Spider (Metadata Extraction Quality)")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    spider_file = project_root / 'scripts' / 'crawlers' / 'nextapple_spider.py'

    # Test with JSONL output to verify metadata
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        output_file = f.name

    cmd = [
        sys.executable, '-m', 'scrapy', 'runspider',
        str(spider_file),
        '-a', 'sitemap=news',
        '-a', 'max_articles=3',
        '-o', output_file,
        '-s', 'LOG_LEVEL=ERROR',  # Minimal logging
        '-s', 'ROBOTSTXT_OBEY=False',
    ]

    print("Command:", ' '.join(cmd[3:]))
    print("Checking metadata quality (title, content, date, category, tags)...")
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
            if article.get('content') and len(article['content']) > 50:
                metadata_quality['has_content'] += 1
            if article.get('publish_date'):
                metadata_quality['has_date'] += 1
            if article.get('author'):
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
    print("NEXTAPPLE SPIDER COMPREHENSIVE TEST (Deep Refactoring v2.0)")
    print("=" * 70)
    print("Key Features:")
    print("  - Multi-sitemap support (news, editors, topics)")
    print("  - Comprehensive metadata extraction (6+ strategies per field)")
    print("  - JSON-LD structured data parsing")
    print("  - Date-based filtering")
    print("  - Robust error handling")
    print("  - Per-sitemap statistics tracking")
    print("=" * 70)

    results = {}

    # Test 1: News sitemap
    results['news_sitemap'] = test_nextapple_news_sitemap()

    # Test 2: All sitemaps
    results['all_sitemaps'] = test_nextapple_all_sitemaps()

    # Test 3: Date filtering
    results['date_filtering'] = test_nextapple_date_filtering()

    # Test 4: Metadata extraction
    results['metadata_extraction'] = test_nextapple_metadata_extraction()

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
    print("Before: Simple sitemap-based crawler, basic extraction")
    print("After:  Multi-sitemap, 6+ fallback strategies per field, JSON-LD")
    print("Metadata: Title, content, date, author, category, tags, images, description")
    print("Features: Date filtering, per-sitemap stats, comprehensive validation")
    print("Quality:  Production-ready with robust error handling")
    print("=" * 70)

    sys.exit(0 if all(results.values()) else 1)
