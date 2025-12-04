#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified Crawler Management Script

This script provides a unified interface to run all news crawlers with
consistent configuration and output handling.

Supported Crawlers:
    - CNA (中央社): Central News Agency
    - LTN (自由時報): Liberty Times Net
    - UDN (聯合報): United Daily News (coming soon)
    - Apple Daily (蘋果日報): Apple Daily (coming soon)
    - PTS (公視): Public Television Service

Features:
    - Unified CLI interface
    - Automatic output organization
    - Parallel crawler execution
    - Progress tracking and logging
    - Error handling and retry logic

Usage:
    # Run single crawler
    python run_crawlers.py --crawler ltn --days 7 --category politics

    # Run multiple crawlers
    python run_crawlers.py --crawler ltn,cna --days 3

    # Run all crawlers
    python run_crawlers.py --all --days 7

    # Custom output directory
    python run_crawlers.py --crawler ltn --output-dir data/raw/custom/

Author: Information Retrieval System
Date: 2025-11-17
"""

import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Scrapy imports
try:
    from scrapy.crawler import CrawlerProcess, CrawlerRunner
    from scrapy.utils.project import get_project_settings
    from twisted.internet import reactor, defer
    SCRAPY_AVAILABLE = True
except ImportError:
    SCRAPY_AVAILABLE = False
    logging.warning("Scrapy not installed. Please install: pip install scrapy scrapy-playwright playwright")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/crawlers.log')
    ]
)
logger = logging.getLogger(__name__)


class CrawlerConfig:
    """Configuration for a single crawler."""

    def __init__(self, name: str, spider_class: str, module_path: str,
                 description: str, requires_playwright: bool = False):
        """
        Initialize crawler configuration.

        Args:
            name: Crawler short name (e.g., 'ltn', 'cna')
            spider_class: Spider class name
            module_path: Path to spider module relative to scripts/crawlers/
            description: Human-readable description
            requires_playwright: Whether this crawler needs Playwright
        """
        self.name = name
        self.spider_class = spider_class
        self.module_path = module_path
        self.description = description
        self.requires_playwright = requires_playwright


# Crawler registry
CRAWLERS: Dict[str, CrawlerConfig] = {
    'cna': CrawlerConfig(
        name='cna',
        spider_class='CNANewsSpider',
        module_path='cna_spider',
        description='中央社 Central News Agency',
        requires_playwright=False
    ),
    'ltn': CrawlerConfig(
        name='ltn',
        spider_class='LTNNewsSpider',
        module_path='ltn_spider',
        description='自由時報 Liberty Times Net',
        requires_playwright=True
    ),
    'pts': CrawlerConfig(
        name='pts',
        spider_class='PTSNewsSpider',
        module_path='pts_spider',
        description='公視 Public Television Service',
        requires_playwright=False
    ),
    'udn': CrawlerConfig(
        name='udn',
        spider_class='UDNNewsSpider',
        module_path='udn_spider',
        description='聯合報 United Daily News',
        requires_playwright=True
    ),
    'apple': CrawlerConfig(
        name='apple',
        spider_class='AppleDailySpider',
        module_path='apple_daily_spider',
        description='蘋果日報 Apple Daily',
        requires_playwright=True
    ),
    'tvbs': CrawlerConfig(
        name='tvbs',
        spider_class='TVBSNewsSpider',
        module_path='tvbs_spider',
        description='TVBS新聞 TVBS News',
        requires_playwright=True
    ),
    'chinatimes': CrawlerConfig(
        name='chinatimes',
        spider_class='ChinaTimesSpider',
        module_path='chinatimes_spider',
        description='中時新聞網 China Times News',
        requires_playwright=False
    ),
    'ettoday': CrawlerConfig(
        name='ettoday',
        spider_class='ETtodaySpider',
        module_path='ettoday_spider',
        description='東森新聞雲 ETtoday News Cloud',
        requires_playwright=True
    ),
    'storm': CrawlerConfig(
        name='storm',
        spider_class='StormMediaSpider',
        module_path='storm_spider',
        description='風傳媒 Storm Media',
        requires_playwright=True
    ),
}


class CrawlerManager:
    """
    Manager for running multiple crawlers.

    Features:
        - Sequential or parallel execution
        - Unified output management
        - Statistics collection
        - Error handling
    """

    def __init__(self, output_dir: Path = None):
        """
        Initialize crawler manager.

        Args:
            output_dir: Directory for output files (default: data/raw/)
        """
        self.output_dir = output_dir or (project_root / 'data' / 'raw')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.stats = {
            'start_time': None,
            'end_time': None,
            'crawlers_run': [],
            'crawlers_failed': [],
            'total_items': 0,
        }

    def run_crawler(self, crawler_name: str, **kwargs) -> bool:
        """
        Run a single crawler.

        Args:
            crawler_name: Name of crawler to run
            **kwargs: Additional arguments for spider

        Returns:
            bool: True if successful, False otherwise
        """
        if crawler_name not in CRAWLERS:
            logger.error(f"Unknown crawler: {crawler_name}")
            logger.info(f"Available crawlers: {', '.join(CRAWLERS.keys())}")
            return False

        config = CRAWLERS[crawler_name]

        logger.info("=" * 70)
        logger.info(f"Starting Crawler: {config.description} ({config.name})")
        logger.info("=" * 70)

        try:
            # Import spider module dynamically
            module_name = f"{config.module_path}"
            spider_module = __import__(module_name, fromlist=[config.spider_class])
            spider_class = getattr(spider_module, config.spider_class)

            # Prepare settings
            settings = self._get_crawler_settings(config)

            # Generate output file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = self.output_dir / f"{config.name}_news_{timestamp}.jsonl"

            # Update settings with output file
            settings['FEEDS'] = {
                str(output_file): {
                    'format': 'jsonlines',
                    'encoding': 'utf8',
                    'store_empty': False,
                    'overwrite': False,
                }
            }

            # Create crawler process
            process = CrawlerProcess(settings)

            # Run spider with provided kwargs
            process.crawl(spider_class, **kwargs)

            # Start crawling
            logger.info(f"Output will be saved to: {output_file}")
            process.start()

            # Check output
            if output_file.exists():
                item_count = sum(1 for _ in open(output_file, 'r', encoding='utf-8'))
                logger.info(f"✓ Crawler completed: {item_count} items scraped")
                self.stats['total_items'] += item_count
                self.stats['crawlers_run'].append(config.name)
                return True
            else:
                logger.warning(f"✗ No output file generated")
                self.stats['crawlers_failed'].append(config.name)
                return False

        except Exception as e:
            logger.error(f"✗ Crawler failed: {e}", exc_info=True)
            self.stats['crawlers_failed'].append(config.name)
            return False

    def run_multiple(self, crawler_names: List[str], **kwargs) -> Dict[str, bool]:
        """
        Run multiple crawlers sequentially.

        Args:
            crawler_names: List of crawler names
            **kwargs: Arguments for spiders

        Returns:
            dict: Mapping of crawler name to success status
        """
        self.stats['start_time'] = datetime.now()

        results = {}

        for crawler_name in crawler_names:
            if crawler_name not in CRAWLERS:
                logger.warning(f"Skipping unknown crawler: {crawler_name}")
                results[crawler_name] = False
                continue

            success = self.run_crawler(crawler_name, **kwargs)
            results[crawler_name] = success

        self.stats['end_time'] = datetime.now()

        # Print summary
        self._print_summary()

        return results

    def _get_crawler_settings(self, config: CrawlerConfig) -> dict:
        """
        Get Scrapy settings for a crawler.

        Args:
            config: Crawler configuration

        Returns:
            dict: Scrapy settings
        """
        settings = {
            'LOG_LEVEL': 'INFO',
            'ROBOTSTXT_OBEY': True,
            'CONCURRENT_REQUESTS_PER_DOMAIN': 1,
            'DOWNLOAD_DELAY': 2,
            'RETRY_TIMES': 3,
            'RETRY_HTTP_CODES': [500, 502, 503, 504, 408, 429],
        }

        # Add Playwright settings if needed
        if config.requires_playwright:
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

        return settings

    def _print_summary(self):
        """Print execution summary."""
        logger.info("")
        logger.info("=" * 70)
        logger.info("Crawler Execution Summary")
        logger.info("=" * 70)

        if self.stats['start_time'] and self.stats['end_time']:
            duration = self.stats['end_time'] - self.stats['start_time']
            logger.info(f"Duration: {duration}")

        logger.info(f"Crawlers run: {len(self.stats['crawlers_run'])}")
        if self.stats['crawlers_run']:
            logger.info(f"  - {', '.join(self.stats['crawlers_run'])}")

        if self.stats['crawlers_failed']:
            logger.info(f"Crawlers failed: {len(self.stats['crawlers_failed'])}")
            logger.info(f"  - {', '.join(self.stats['crawlers_failed'])}")

        logger.info(f"Total items scraped: {self.stats['total_items']}")
        logger.info("=" * 70)


def main():
    """Main function for CLI interface."""

    parser = argparse.ArgumentParser(
        description='Unified News Crawler Management System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run LTN crawler for 7 days
    python run_crawlers.py --crawler ltn --days 7

    # Run multiple crawlers
    python run_crawlers.py --crawler ltn,cna --days 3

    # Run all available crawlers
    python run_crawlers.py --all --days 7

    # Run with custom date range
    python run_crawlers.py --crawler ltn --start-date 2025-11-01 --end-date 2025-11-13

    # Run with category filter (if supported by crawler)
    python run_crawlers.py --crawler ltn --days 3 --category politics

Available Crawlers:
""" + '\n'.join([f"    {name}: {cfg.description}" for name, cfg in CRAWLERS.items()])
    )

    parser.add_argument(
        '--crawler', '-c',
        type=str,
        help='Crawler name(s) to run (comma-separated). Use --list to see all.'
    )

    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Run all available crawlers'
    )

    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List all available crawlers and exit'
    )

    parser.add_argument(
        '--days', '-d',
        type=int,
        default=7,
        help='Number of days to crawl (default: 7)'
    )

    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--end-date',
        type=str,
        help='End date (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--category',
        type=str,
        help='Category filter (if supported by crawler)'
    )

    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        help='Output directory (default: data/raw/)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # List crawlers
    if args.list:
        print("\nAvailable Crawlers:")
        print("=" * 70)
        for name, config in CRAWLERS.items():
            playwright_marker = " [Playwright]" if config.requires_playwright else ""
            print(f"  {name:10s}: {config.description}{playwright_marker}")
        print("=" * 70)
        return

    # Check Scrapy availability
    if not SCRAPY_AVAILABLE:
        logger.error("Scrapy is not installed!")
        logger.error("Install with: pip install scrapy scrapy-playwright playwright")
        logger.error("Then run: playwright install chromium")
        sys.exit(1)

    # Determine which crawlers to run
    if args.all:
        crawler_names = list(CRAWLERS.keys())
    elif args.crawler:
        crawler_names = [c.strip() for c in args.crawler.split(',')]
    else:
        logger.error("Please specify --crawler or --all")
        parser.print_help()
        sys.exit(1)

    # Prepare output directory
    output_dir = Path(args.output_dir) if args.output_dir else None

    # Prepare spider kwargs
    spider_kwargs = {'days': args.days}

    if args.start_date and args.end_date:
        spider_kwargs['start_date'] = args.start_date
        spider_kwargs['end_date'] = args.end_date
        del spider_kwargs['days']

    if args.category:
        spider_kwargs['category'] = args.category

    # Run crawlers
    logger.info("Starting crawler manager...")
    logger.info(f"Crawlers to run: {', '.join(crawler_names)}")
    logger.info(f"Arguments: {spider_kwargs}")

    manager = CrawlerManager(output_dir=output_dir)
    results = manager.run_multiple(crawler_names, **spider_kwargs)

    # Exit with appropriate code
    failed_count = sum(1 for success in results.values() if not success)
    sys.exit(1 if failed_count > 0 else 0)


if __name__ == '__main__':
    main()
