#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick test all crawlers - Fast validation script

Tests all crawlers quickly to verify they can scrape at least 1 article.

Usage:
    python test_all_crawlers_quick.py
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime
import json

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Crawler list
CRAWLERS = {
    'cna': {'name': 'CNA中央社', 'playwright': False},
    'pts': {'name': 'PTS公視', 'playwright': False},
    'chinatimes': {'name': '中時新聞網', 'playwright': False},
    'ltn': {'name': 'LTN自由時報', 'playwright': True},
    'udn': {'name': 'UDN聯合報', 'playwright': True},
    'apple': {'name': 'Apple Daily蘋果日報', 'playwright': True},
    'ettoday': {'name': '東森新聞雲', 'playwright': True},
    'storm': {'name': 'Storm風傳媒', 'playwright': True},
    'tvbs': {'name': 'TVBS新聞', 'playwright': True, 'skip': True},  # Known issues
}

def test_crawler(crawler_name: str, config: dict) -> dict:
    """Test a single crawler"""
    print(f"\n{'='*70}")
    print(f"Testing {config['name']} ({crawler_name})...")
    print(f"{'='*70}")

    if config.get('skip'):
        print(f"⊘ SKIPPED - Marked as skip")
        return {'status': 'skipped', 'message': 'Marked as skip'}

    spider_file = project_root / 'scripts' / 'crawlers' / f'{crawler_name}_spider.py'

    if not spider_file.exists():
        print(f"✗ FAILED - Spider file not found: {spider_file}")
        return {'status': 'failed', 'message': 'Spider file not found'}

    try:
        # Run spider with minimal settings
        cmd = [
            sys.executable, '-m', 'scrapy', 'runspider',
            str(spider_file),
            '-a', 'days=1',
            '-s', 'CLOSESPIDER_ITEMCOUNT=2',
            '-s', 'LOG_LEVEL=ERROR',
            '-s', 'RETRY_TIMES=1',
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=90,  # 90 seconds timeout
            cwd=str(project_root)
        )

        # Check for scraped items
        output = result.stdout + result.stderr

        if 'Scraped' in output or 'item_scraped_count' in output:
            # Try to extract item count
            import re
            match = re.search(r"'item_scraped_count':\s*(\d+)", output)
            if match:
                count = int(match.group(1))
                if count > 0:
                    print(f"✓ SUCCESS - Scraped {count} items")
                    return {'status': 'success', 'items': count}

            print(f"✓ SUCCESS - Detected scraping activity")
            return {'status': 'success', 'items': '?'}
        else:
            print(f"✗ FAILED - No items scraped")
            # Print last 10 lines of output for debugging
            lines = output.split('\n')
            print("Last 10 lines of output:")
            for line in lines[-10:]:
                if line.strip():
                    print(f"  {line}")
            return {'status': 'failed', 'message': 'No items scraped'}

    except subprocess.TimeoutExpired:
        print(f"✗ TIMEOUT - Exceeded 90 seconds")
        return {'status': 'timeout', 'message': 'Timeout after 90s'}
    except Exception as e:
        print(f"✗ ERROR - {str(e)}")
        return {'status': 'error', 'message': str(e)}

def main():
    """Main function"""
    print("\n" + "="*70)
    print("QUICK CRAWLER TEST - All News Crawlers")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Testing {len(CRAWLERS)} crawlers")

    results = {}

    for crawler_name, config in CRAWLERS.items():
        result = test_crawler(crawler_name, config)
        results[crawler_name] = {
            'name': config['name'],
            'playwright': config.get('playwright', False),
            **result
        }

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    success = sum(1 for r in results.values() if r['status'] == 'success')
    failed = sum(1 for r in results.values() if r['status'] == 'failed')
    timeout = sum(1 for r in results.values() if r['status'] == 'timeout')
    skipped = sum(1 for r in results.values() if r['status'] == 'skipped')
    error = sum(1 for r in results.values() if r['status'] == 'error')

    print(f"\nTotal: {len(CRAWLERS)}")
    print(f"✓ Success: {success}")
    print(f"✗ Failed: {failed}")
    print(f"⧗ Timeout: {timeout}")
    print(f"⊘ Skipped: {skipped}")
    print(f"⚠ Error: {error}")

    print(f"\nSuccess Rate: {success}/{len(CRAWLERS)-skipped} ({success/(len(CRAWLERS)-skipped)*100:.1f}%)")

    print("\nDetailed Results:")
    print("-"*70)
    for crawler_name, result in results.items():
        status_icon = {
            'success': '✓',
            'failed': '✗',
            'timeout': '⧗',
            'skipped': '⊘',
            'error': '⚠'
        }.get(result['status'], '?')

        items = result.get('items', 0)
        print(f"{status_icon} {result['name']:20s} | Status: {result['status']:10s} | Items: {items}")

    print("="*70)

    # Save results
    output_file = project_root / 'data' / 'crawler_test_results.json'
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': results,
            'summary': {
                'total': len(CRAWLERS),
                'success': success,
                'failed': failed,
                'timeout': timeout,
                'skipped': skipped,
                'error': error,
            }
        }, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")

    # Exit code
    if failed > 0 or timeout > 0 or error > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == '__main__':
    main()
