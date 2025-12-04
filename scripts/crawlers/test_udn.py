#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for UDN spider with proper asyncio reactor setup.
"""

# CRITICAL: Must install asyncio reactor BEFORE any Scrapy imports
import sys
from twisted.internet import asyncioreactor
asyncioreactor.install()

from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from udn_spider import UDNNewsSpider


def main():
    """Run UDN spider with proper settings."""

    # Configure settings
    settings = {
        'LOG_LEVEL': 'INFO',
        'ROBOTSTXT_OBEY': True,
        'CONCURRENT_REQUESTS_PER_DOMAIN': 1,
        'DOWNLOAD_DELAY': 2,
        'RETRY_TIMES': 3,
        'RETRY_HTTP_CODES': [500, 502, 503, 504, 408, 429],

        # Playwright settings
        'DOWNLOAD_HANDLERS': {
            "http": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
            "https": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
        },
        'PLAYWRIGHT_BROWSER_TYPE': 'chromium',
        'PLAYWRIGHT_LAUNCH_OPTIONS': {
            'headless': True,
            'timeout': 60000,
        },

        # Output settings
        'FEEDS': {
            'data/raw/test_udn_news.jsonl': {
                'format': 'jsonlines',
                'encoding': 'utf8',
                'store_empty': False,
                'overwrite': True,
            }
        },

        # Close spider after N items (for testing)
        'CLOSESPIDER_ITEMCOUNT': 3,
    }

    # Create crawler process
    process = CrawlerProcess(settings)

    # Run spider with test parameters
    process.crawl(
        UDNNewsSpider,
        category='politics',  # Test with politics category
        days=1,               # Only last 1 day
    )

    # Start crawling
    print("=" * 70)
    print("Starting UDN Spider Test")
    print("=" * 70)
    process.start()


if __name__ == '__main__':
    main()
