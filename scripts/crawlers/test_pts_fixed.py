#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick test for fixed PTS spider

Tests PTS crawler with new multi-mode strategy.
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_pts_dailynews():
    """Test PTS spider in dailynews mode"""
    print("=" * 70)
    print("Testing PTS Spider (Mode: dailynews)")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    spider_file = project_root / 'scripts' / 'crawlers' / 'pts_spider.py'

    # Test parameters
    cmd = [
        sys.executable, '-m', 'scrapy', 'runspider',
        str(spider_file),
        '-a', 'mode=dailynews',
        '-a', 'days=7',
        '-s', 'CLOSESPIDER_ITEMCOUNT=5',  # Stop after 5 articles
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
            timeout=90,
            cwd=str(project_root)
        )

        output = result.stdout + result.stderr

        # Extract key information
        print("=" * 70)
        print("RESULTS:")
        print("=" * 70)

        for line in output.split('\n'):
            if any(keyword in line for keyword in [
                'PTS Spider', 'Mode:', 'Articles', 'Not found',
                'Failed', 'Success', 'Hit rate', 'Scraped', 'article_id',
                'Found.*unique', 'Successfully parsed'
            ]):
                print(line)

        # Check success
        if 'Successfully parsed' in output or 'Articles successfully crawled' in output:
            print("\n✓ TEST PASSED - PTS spider is working (dailynews mode)!")
            return True
        else:
            print("\n✗ TEST FAILED - No articles scraped")
            print("\nLast 20 lines of output:")
            print("-" * 70)
            for line in output.split('\n')[-20:]:
                if line.strip():
                    print(line)
            return False

    except subprocess.TimeoutExpired:
        print("\n✗ TIMEOUT - Exceeded 90 seconds")
        return False
    except Exception as e:
        print(f"\n✗ ERROR - {str(e)}")
        return False

def test_pts_sequential():
    """Test PTS spider in sequential mode (small range)"""
    print("\n" + "=" * 70)
    print("Testing PTS Spider (Mode: sequential - small range)")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    spider_file = project_root / 'scripts' / 'crawlers' / 'pts_spider.py'

    # Test parameters - small ID range for quick test
    cmd = [
        sys.executable, '-m', 'scrapy', 'runspider',
        str(spider_file),
        '-a', 'mode=sequential',
        '-a', 'start_id=781850',  # Recent IDs
        '-a', 'end_id=781900',    # 50 IDs to try
        '-s', 'CLOSESPIDER_ITEMCOUNT=3',
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
                'PTS Spider', 'Mode:', 'Articles', 'Not found',
                'Failed', 'Success', 'Hit rate', 'ID range', '404 count'
            ]):
                print(line)

        # Check success
        if 'Articles successfully crawled' in output and 'Mode: sequential' in output:
            # Extract article count
            import re
            match = re.search(r'Articles successfully crawled: (\d+)', output)
            if match and int(match.group(1)) > 0:
                print(f"\n✓ TEST PASSED - PTS spider working in sequential mode!")
                return True

        print("\n⚠ TEST INCOMPLETE - Sequential mode needs verification")
        return False

    except subprocess.TimeoutExpired:
        print("\n✗ TIMEOUT - Exceeded 120 seconds")
        return False
    except Exception as e:
        print(f"\n✗ ERROR - {str(e)}")
        return False

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("PTS SPIDER COMPREHENSIVE TEST")
    print("=" * 70)

    results = {}

    # Test 1: dailynews mode
    results['dailynews'] = test_pts_dailynews()

    # Test 2: sequential mode
    results['sequential'] = test_pts_sequential()

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    for mode, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status} - {mode} mode")

    print("=" * 70)

    sys.exit(0 if all(results.values()) else 1)
