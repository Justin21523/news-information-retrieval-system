#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Crawler Integration Test Script

Tests all news crawlers with limited item count to verify functionality.

Usage:
    python test_all_crawlers.py
    python test_all_crawlers.py --crawler tvbs
    python test_all_crawlers.py --quick  # Only 1 item per crawler

Author: Information Retrieval System
Date: 2025-11-18
"""

import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


class CrawlerTester:
    """Test suite for news crawlers."""

    def __init__(self, test_items: int = 3):
        """
        Initialize crawler tester.

        Args:
            test_items: Number of items to scrape per crawler (for testing)
        """
        self.test_items = test_items
        self.test_output_dir = project_root / 'data' / 'test_output'
        self.test_output_dir.mkdir(parents=True, exist_ok=True)

        self.results = {
            'timestamp': datetime.now().isoformat(),
            'test_items': test_items,
            'crawlers_tested': [],
            'crawlers_passed': [],
            'crawlers_failed': [],
        }

    def test_crawler(self, crawler_name: str, spider_module: str, spider_class: str,
                     requires_playwright: bool = False) -> bool:
        """
        Test a single crawler.

        Args:
            crawler_name: Short name (e.g., 'tvbs')
            spider_module: Module name (e.g., 'tvbs_spider')
            spider_class: Spider class name (e.g., 'TVBSNewsSpider')
            requires_playwright: Whether Playwright is required

        Returns:
            bool: True if test passed, False otherwise
        """
        logger.info("=" * 70)
        logger.info(f"Testing Crawler: {crawler_name}")
        logger.info("=" * 70)

        try:
            # Install reactor if Playwright is required
            if requires_playwright:
                from twisted.internet import asyncioreactor
                try:
                    asyncioreactor.install()
                    logger.info("Asyncio reactor installed for Playwright")
                except Exception as e:
                    logger.warning(f"Reactor already installed: {e}")

            # Import spider dynamically
            spider_mod = __import__(spider_module, fromlist=[spider_class])
            spider_cls = getattr(spider_mod, spider_class)

            # Import Scrapy
            from scrapy.crawler import CrawlerProcess
            from scrapy.utils.project import get_project_settings

            # Prepare settings
            settings = get_project_settings()
            settings.update({
                'LOG_LEVEL': 'INFO',
                'ROBOTSTXT_OBEY': True,
                'CONCURRENT_REQUESTS_PER_DOMAIN': 1,
                'DOWNLOAD_DELAY': 2,
                'CLOSESPIDER_ITEMCOUNT': self.test_items,  # Limit items for testing
            })

            # Add Playwright settings if needed
            if requires_playwright:
                settings.update({
                    'DOWNLOAD_HANDLERS': {
                        "http": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
                        "https": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
                    },
                    'PLAYWRIGHT_BROWSER_TYPE': 'chromium',
                    'PLAYWRIGHT_LAUNCH_OPTIONS': {
                        'headless': True,
                        'timeout': 60000,
                    },
                })

            # Output file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = self.test_output_dir / f"test_{crawler_name}_{timestamp}.jsonl"

            settings['FEEDS'] = {
                str(output_file): {
                    'format': 'jsonlines',
                    'encoding': 'utf8',
                    'store_empty': False,
                    'overwrite': True,
                }
            }

            # Create crawler process
            process = CrawlerProcess(settings)

            # Run spider with minimal data (1 day)
            logger.info(f"Starting crawler {crawler_name} (max {self.test_items} items)...")
            process.crawl(spider_cls, days=1)
            process.start()

            # Verify output
            if output_file.exists():
                items = []
                with open(output_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        items.append(json.loads(line))

                logger.info(f"✓ Test PASSED: {len(items)} items scraped")
                logger.info(f"  Output: {output_file}")

                # Verify item structure
                if items:
                    sample = items[0]
                    required_fields = ['article_id', 'title', 'content', 'url', 'source']
                    missing = [f for f in required_fields if f not in sample]
                    if missing:
                        logger.warning(f"  Missing fields in output: {missing}")
                    else:
                        logger.info(f"  ✓ All required fields present")

                self.results['crawlers_passed'].append(crawler_name)
                return True
            else:
                logger.error(f"✗ Test FAILED: No output file generated")
                self.results['crawlers_failed'].append(crawler_name)
                return False

        except Exception as e:
            logger.error(f"✗ Test FAILED: {e}", exc_info=True)
            self.results['crawlers_failed'].append(crawler_name)
            return False

    def test_all(self, crawler_names: List[str] = None):
        """
        Test all or specified crawlers.

        Args:
            crawler_names: List of crawler names to test (None = all)
        """
        # Crawler configurations
        crawlers = {
            'tvbs': {
                'module': 'tvbs_spider',
                'class': 'TVBSNewsSpider',
                'playwright': True,
            },
            'chinatimes': {
                'module': 'chinatimes_spider',
                'class': 'ChinaTimesSpider',
                'playwright': False,
            },
            'ettoday': {
                'module': 'ettoday_spider',
                'class': 'ETtodaySpider',
                'playwright': True,
            },
            'storm': {
                'module': 'storm_spider',
                'class': 'StormMediaSpider',
                'playwright': True,
            },
        }

        # Filter if specific crawlers requested
        if crawler_names:
            crawlers = {k: v for k, v in crawlers.items() if k in crawler_names}

        # Test each crawler
        for name, config in crawlers.items():
            self.results['crawlers_tested'].append(name)
            self.test_crawler(
                crawler_name=name,
                spider_module=config['module'],
                spider_class=config['class'],
                requires_playwright=config['playwright']
            )

        # Print summary
        self.print_summary()

    def print_summary(self):
        """Print test summary."""
        logger.info("")
        logger.info("=" * 70)
        logger.info("Test Summary")
        logger.info("=" * 70)
        logger.info(f"Total tested: {len(self.results['crawlers_tested'])}")
        logger.info(f"Passed: {len(self.results['crawlers_passed'])}")
        logger.info(f"Failed: {len(self.results['crawlers_failed'])}")

        if self.results['crawlers_passed']:
            logger.info(f"\n✓ Passed crawlers:")
            for name in self.results['crawlers_passed']:
                logger.info(f"  - {name}")

        if self.results['crawlers_failed']:
            logger.info(f"\n✗ Failed crawlers:")
            for name in self.results['crawlers_failed']:
                logger.info(f"  - {name}")

        logger.info("=" * 70)

        # Save results
        result_file = self.test_output_dir / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to: {result_file}")


def main():
    """Main function for CLI."""
    parser = argparse.ArgumentParser(
        description='Test news crawler system',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--crawler', '-c',
        type=str,
        help='Specific crawler to test (comma-separated)'
    )

    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Quick test (only 1 item per crawler)'
    )

    parser.add_argument(
        '--items', '-n',
        type=int,
        default=3,
        help='Number of items to scrape per crawler (default: 3)'
    )

    args = parser.parse_args()

    # Determine test items
    test_items = 1 if args.quick else args.items

    # Parse crawler names
    crawler_names = None
    if args.crawler:
        crawler_names = [c.strip() for c in args.crawler.split(',')]

    # Run tests
    logger.info("Starting crawler integration tests...")
    logger.info(f"Test items per crawler: {test_items}")

    tester = CrawlerTester(test_items=test_items)
    tester.test_all(crawler_names=crawler_names)


if __name__ == '__main__':
    main()
