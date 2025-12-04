#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Single Crawler Test Script

Properly tests a single crawler with reactor installation for Playwright.

Usage:
    python test_single_crawler.py tvbs
    python test_single_crawler.py chinatimes
    python test_single_crawler.py ettoday
    python test_single_crawler.py storm

Author: Information Retrieval System
Date: 2025-11-18
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


# Crawler configurations
CRAWLERS = {
    'tvbs': {
        'module': 'tvbs_spider',
        'class': 'TVBSNewsSpider',
        'playwright': True,
        'name': 'TVBS新聞',
    },
    'chinatimes': {
        'module': 'chinatimes_spider',
        'class': 'ChinaTimesSpider',
        'playwright': False,
        'name': '中時新聞網',
    },
    'ettoday': {
        'module': 'ettoday_spider',
        'class': 'ETtodaySpider',
        'playwright': True,
        'name': '東森新聞雲',
    },
    'storm': {
        'module': 'storm_spider',
        'class': 'StormMediaSpider',
        'playwright': True,
        'name': '風傳媒',
    },
}


def test_crawler(crawler_name: str, test_items: int = 3):
    """
    Test a single crawler with proper reactor setup.

    Args:
        crawler_name: Name of crawler to test
        test_items: Number of items to scrape (default: 3)
    """
    if crawler_name not in CRAWLERS:
        logger.error(f"Unknown crawler: {crawler_name}")
        logger.info(f"Available crawlers: {', '.join(CRAWLERS.keys())}")
        sys.exit(1)

    config = CRAWLERS[crawler_name]
    logger.info("=" * 70)
    logger.info(f"Testing: {config['name']} ({crawler_name})")
    logger.info(f"Playwright: {'Yes' if config['playwright'] else 'No'}")
    logger.info(f"Test items: {test_items}")
    logger.info("=" * 70)

    try:
        # Install reactor for Playwright if needed
        if config['playwright']:
            logger.info("Installing asyncio reactor for Playwright...")
            from twisted.internet import asyncioreactor
            asyncioreactor.install()
            logger.info("✓ Asyncio reactor installed")

        # Import spider
        logger.info(f"Importing {config['module']}.{config['class']}...")
        spider_module = __import__(config['module'], fromlist=[config['class']])
        spider_class = getattr(spider_module, config['class'])
        logger.info("✓ Spider imported successfully")

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
            'CLOSESPIDER_ITEMCOUNT': test_items,
        })

        # Add Playwright settings if needed
        if config['playwright']:
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
        output_dir = project_root / 'data' / 'test_output'
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = output_dir / f"test_{crawler_name}_{timestamp}.jsonl"

        settings['FEEDS'] = {
            str(output_file): {
                'format': 'jsonlines',
                'encoding': 'utf8',
                'store_empty': False,
                'overwrite': True,
            }
        }

        # Create crawler process
        logger.info("Creating Scrapy crawler process...")
        process = CrawlerProcess(settings)

        # Run spider
        logger.info(f"Starting crawler (max {test_items} items, 1 day data)...")
        logger.info("")
        process.crawl(spider_class, days=1)
        process.start()

        # Verify output
        logger.info("")
        logger.info("=" * 70)
        if output_file.exists():
            import json
            items = []
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    items.append(json.loads(line))

            logger.info(f"✓ TEST PASSED: {len(items)} items scraped")
            logger.info(f"  Output file: {output_file}")

            if items:
                sample = items[0]
                logger.info(f"  Sample title: {sample.get('title', '')[:60]}...")
                logger.info(f"  Sample content length: {len(sample.get('content', ''))} chars")

                # Verify required fields
                required_fields = ['article_id', 'title', 'content', 'url', 'source']
                missing = [f for f in required_fields if f not in sample]
                if missing:
                    logger.warning(f"  ⚠ Missing fields: {missing}")
                else:
                    logger.info(f"  ✓ All required fields present")

            logger.info("=" * 70)
            return True
        else:
            logger.error(f"✗ TEST FAILED: No output file generated")
            logger.info("=" * 70)
            return False

    except Exception as e:
        logger.error(f"✗ TEST FAILED: {e}", exc_info=True)
        logger.info("=" * 70)
        return False


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python test_single_crawler.py <crawler_name> [test_items]")
        print(f"Available crawlers: {', '.join(CRAWLERS.keys())}")
        sys.exit(1)

    crawler = sys.argv[1]
    items = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    success = test_crawler(crawler, items)
    sys.exit(0 if success else 1)
