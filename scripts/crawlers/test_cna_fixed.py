#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick test for fixed CNA spider

Tests CNA crawler with direct URL generation strategy.
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_cna():
    """Test CNA spider with minimal parameters"""
    print("=" * 70)
    print("Testing CNA Spider (Fixed - Direct URL Generation)")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    spider_file = project_root / 'scripts' / 'crawlers' / 'cna_spider.py'

    # Test parameters
    cmd = [
        sys.executable, '-m', 'scrapy', 'runspider',
        str(spider_file),
        '-a', 'start_date=2025-11-18',
        '-a', 'end_date=2025-11-18',
        '-a', 'max_id=50',  # Only try first 50 IDs per category
        '-s', 'CLOSESPIDER_ITEMCOUNT=5',  # Stop after 5 articles
        '-s', 'LOG_LEVEL=INFO',
        '-s', 'ROBOTSTXT_OBEY=False',  # Skip robots.txt for quick test
        '-s', 'DOWNLOAD_DELAY=0.3',
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
                'CNA Spider', 'Generated', 'Articles', 'Not found',
                'Failed', 'Success', 'Hit rate', 'Scraped', 'article_id'
            ]):
                print(line)

        # Check success
        if 'Successfully parsed article' in output or 'Articles successfully crawled' in output:
            print("\n✓ TEST PASSED - CNA spider is working!")
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

if __name__ == '__main__':
    success = test_cna()
    sys.exit(0 if success else 1)
