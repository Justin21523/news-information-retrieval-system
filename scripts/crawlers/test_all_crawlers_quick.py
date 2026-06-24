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
from typing import Dict, Any

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Crawler list (aligned with `scripts/crawlers/mass_collect.py`).
CRAWLERS = {
    'chinatimes': {'name': '中時新聞網', 'script': 'chinatimes_spider.py', 'playwright': False},
    'pts': {'name': '公視 PTS', 'script': 'pts_spider.py', 'playwright': False},
    'ltn': {'name': '自由時報 LTN', 'script': 'ltn_spider.py', 'playwright': False},
    'udn': {'name': '聯合報 UDN', 'script': 'udn_spider.py', 'playwright': False},
    'nextapple': {'name': '壹蘋 NextApple(news)', 'script': 'nextapple_spider.py', 'playwright': False},
    'setn': {'name': '三立 SETN', 'script': 'setn_spider.py', 'playwright': False},
    'yahoo': {'name': 'Yahoo 奇摩新聞', 'script': 'yahoo_spider.py', 'playwright': False},
    'cna': {'name': '中央社 CNA(v2)', 'script': 'cna_spider_v2.py', 'playwright': True},
    'ettoday': {'name': '東森新聞雲 ETtoday', 'script': 'ettoday_spider.py', 'playwright': True},
    'apple': {'name': 'Apple Daily / NextApple(.com.tw)', 'script': 'apple_daily_spider.py', 'playwright': True},
    'storm': {'name': '風傳媒 Storm', 'script': 'storm_spider.py', 'playwright': True},
    'tvbs': {'name': 'TVBS 新聞', 'script': 'tvbs_spider.py', 'playwright': True},
    'ftv': {'name': '民視 FTV', 'script': 'ftv_spider.py', 'playwright': True},
    'cti': {'name': '中天 CTI', 'script': 'cti_spider.py', 'playwright': True},
}

def test_crawler(crawler_name: str, config: Dict[str, Any]) -> dict:
    """Test a single crawler."""
    print(f"\n{'='*70}")
    print(f"Testing {config['name']} ({crawler_name})...")
    print(f"{'='*70}")

    if config.get('skip'):
        print(f"⊘ SKIPPED - Marked as skip")
        return {'status': 'skipped', 'message': 'Marked as skip'}

    spider_file = project_root / 'scripts' / 'crawlers' / config['script']

    if not spider_file.exists():
        print(f"✗ FAILED - Spider file not found: {spider_file}")
        return {'status': 'failed', 'message': 'Spider file not found'}

    try:
        # Output file
        output_dir = project_root / 'data' / 'test_output'
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f"quick_{crawler_name}_{timestamp}.jsonl"

        # Run spider with minimal settings (write output and check line count).
        cmd = [sys.executable, '-m', 'scrapy', 'runspider', str(spider_file)]

        # Prefer 1-day window; many spiders support `days`.
        cmd += ['-a', 'days=1']

        # ChinaTimes can queue many requests per list page which slows shutdown when
        # using CLOSESPIDER_ITEMCOUNT. Keep the smoke run bounded.
        if crawler_name == 'chinatimes':
            cmd += ['-a', 'max_pages=1', '-a', 'max_links_per_page=10']

        # FTV defaults to crawling all categories; keep smoke run minimal.
        if crawler_name == 'ftv':
            cmd += ['-a', 'category=politics', '-a', 'max_pages=1', '-a', 'max_articles=20']

        # For CNA v2, pass explicit start/end date to reduce pagination.
        if crawler_name == 'cna':
            today = datetime.now().strftime('%Y-%m-%d')
            cmd += ['-a', f'start_date={today}', '-a', f'end_date={today}']

        cmd += [
            '-O', str(output_file),  # overwrite for deterministic check
            '-s', 'CLOSESPIDER_ITEMCOUNT=1',
            '-s', 'LOG_LEVEL=ERROR',
            '-s', 'RETRY_TIMES=1',
        ]

        if crawler_name == 'chinatimes':
            cmd += [
                '-s', 'CONCURRENT_REQUESTS=1',
                '-s', 'CONCURRENT_REQUESTS_PER_DOMAIN=1',
                '-s', 'DOWNLOAD_DELAY=0',
                '-s', 'DOWNLOAD_TIMEOUT=30',
            ]

        if config.get('playwright'):
            cmd += ['-s', 'TWISTED_REACTOR=twisted.internet.asyncioreactor.AsyncioSelectorReactor']

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180 if config.get('playwright') else 90,
            cwd=str(project_root)
        )

        if output_file.exists():
            line_count = 0
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        line_count += 1

            if line_count > 0 and result.returncode == 0:
                print(f"✓ SUCCESS - Scraped {line_count} items (output: {output_file})")
                return {'status': 'success', 'items': line_count, 'output': str(output_file)}

        output_text = result.stdout + result.stderr
        blocked_markers = (
            'blocked_by_cloudflare',
            'cloudflare block detected',
            'sorry, you have been blocked',
            'attention required! | cloudflare',
        )
        if any(marker in output_text.lower() for marker in blocked_markers):
            print("✗ BLOCKED - Cloudflare/WAF blocked the crawler")
            output_lines = output_text.splitlines()
            print("Last 20 lines of output:")
            for line in output_lines[-20:]:
                if line.strip():
                    print(f"  {line}")
            return {'status': 'blocked', 'message': 'Blocked by Cloudflare/WAF'}

        print("✗ FAILED - No items scraped")
        output = output_text.splitlines()
        print("Last 20 lines of output:")
        for line in output[-20:]:
            if line.strip():
                print(f"  {line}")
        return {'status': 'failed', 'message': 'No items scraped', 'output': str(output_file)}

    except subprocess.TimeoutExpired:
        timeout_s = 180 if config.get('playwright') else 90
        print(f"✗ TIMEOUT - Exceeded {timeout_s} seconds")
        return {'status': 'timeout', 'message': f'Timeout after {timeout_s}s'}
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
    blocked = sum(1 for r in results.values() if r['status'] == 'blocked')
    skipped = sum(1 for r in results.values() if r['status'] == 'skipped')
    error = sum(1 for r in results.values() if r['status'] == 'error')

    print(f"\nTotal: {len(CRAWLERS)}")
    print(f"✓ Success: {success}")
    print(f"✗ Failed: {failed}")
    print(f"⧗ Timeout: {timeout}")
    print(f"⛔ Blocked: {blocked}")
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
            'blocked': '⛔',
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
                'blocked': blocked,
                'skipped': skipped,
                'error': error,
            }
        }, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")

    # Exit code
    if failed > 0 or timeout > 0 or blocked > 0 or error > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == '__main__':
    main()
