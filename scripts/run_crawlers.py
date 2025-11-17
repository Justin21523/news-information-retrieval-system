#!/usr/bin/env python
"""
Run News Crawlers

Unified script to run all news crawlers (CNA, PTS, TechNews).

Usage:
    # Crawl all sources for a date range
    python scripts/run_crawlers.py --start-date 2024-01-01 --end-date 2024-01-31

    # Crawl specific source
    python scripts/run_crawlers.py --source cna --start-date 2024-01-01 --end-date 2024-01-31

    # Dry run (test mode)
    python scripts/run_crawlers.py --dry-run --start-date 2024-01-01 --end-date 2024-01-05

Author: CNIRS Development Team
License: Educational Use Only
"""

import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run CNIRS news crawlers to collect data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Crawl all sources for January 2024
  python scripts/run_crawlers.py --start-date 2024-01-01 --end-date 2024-01-31

  # Crawl only CNA for specific date range
  python scripts/run_crawlers.py --source cna --start-date 2024-01-01 --end-date 2024-01-10

  # Dry run (small test)
  python scripts/run_crawlers.py --dry-run --start-date 2024-01-01 --end-date 2024-01-03

  # Crawl with custom output directory
  python scripts/run_crawlers.py --output-dir data/test --start-date 2024-01-01 --end-date 2024-01-05
        """
    )

    parser.add_argument(
        '--source',
        type=str,
        choices=['all', 'cna', 'pts', 'technews'],
        default='all',
        help='News source to crawl (default: all)'
    )

    parser.add_argument(
        '--start-date',
        type=str,
        required=True,
        help='Start date in YYYY-MM-DD format'
    )

    parser.add_argument(
        '--end-date',
        type=str,
        required=True,
        help='End date in YYYY-MM-DD format'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/raw',
        help='Output directory for crawled data (default: data/raw)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Dry run mode (only crawl 5 days for testing)'
    )

    parser.add_argument(
        '--concurrent',
        action='store_true',
        help='Run all spiders concurrently (faster but more resource-intensive)'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )

    return parser.parse_args()


def validate_dates(start_date: str, end_date: str):
    """
    Validate date strings.

    Args:
        start_date: Start date string
        end_date: End date string

    Returns:
        tuple: (start_datetime, end_datetime)
    """
    try:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')

        if start_dt > end_dt:
            logger.error("Start date must be before end date")
            sys.exit(1)

        if end_dt > datetime.now():
            logger.warning("End date is in the future, adjusting to today")
            end_dt = datetime.now()

        return start_dt, end_dt

    except ValueError as e:
        logger.error(f"Invalid date format: {e}")
        logger.error("Please use YYYY-MM-DD format (e.g., 2024-01-01)")
        sys.exit(1)


def run_spiders(sources: list, start_date: str, end_date: str, output_dir: str, log_level: str, concurrent: bool = False):
    """
    Run Scrapy spiders.

    Args:
        sources: List of spider names to run
        start_date: Start date string
        end_date: End date string
        output_dir: Output directory path
        log_level: Logging level
        concurrent: Whether to run spiders concurrently
    """
    from scrapy.crawler import CrawlerProcess, CrawlerRunner
    from scrapy.utils.project import get_project_settings
    from scrapy.utils.log import configure_logging
    from twisted.internet import reactor, defer

    # Load settings
    settings = get_project_settings()

    # Override settings
    settings.update({
        'LOG_LEVEL': log_level,
        'DOWNLOAD_DELAY': 2,
        'CONCURRENT_REQUESTS_PER_DOMAIN': 2,
        'ROBOTSTXT_OBEY': True,
        'USER_AGENT': 'CNIRS Academic Research Bot (Educational Use)',
        'FEEDS': {
            f'{output_dir}/%(name)s_{start_date}_to_{end_date}.jsonl': {
                'format': 'jsonlines',
                'encoding': 'utf8',
                'store_empty': False,
                'overwrite': False,
            }
        }
    })

    # Import spiders
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))

    from crawlers.cna_spider import CNANewsSpider
    from crawlers.pts_spider import PTSNewsSpider
    from crawlers.technews_spider import TechNewsSpider

    spider_map = {
        'cna': CNANewsSpider,
        'pts': PTSNewsSpider,
        'technews': TechNewsSpider,
    }

    if concurrent:
        # Concurrent execution using CrawlerRunner
        configure_logging(settings)
        runner = CrawlerRunner(settings)

        @defer.inlineCallbacks
        def crawl():
            for source in sources:
                spider_cls = spider_map[source]
                yield runner.crawl(
                    spider_cls,
                    start_date=start_date,
                    end_date=end_date
                )
            reactor.stop()

        crawl()
        reactor.run()

    else:
        # Sequential execution using CrawlerProcess
        process = CrawlerProcess(settings)

        for source in sources:
            spider_cls = spider_map[source]
            logger.info(f"Starting {source.upper()} spider...")
            process.crawl(
                spider_cls,
                start_date=start_date,
                end_date=end_date
            )

        logger.info("Starting crawl process...")
        process.start()


def main():
    """Main function."""
    args = parse_args()

    # Validate dates
    start_dt, end_dt = validate_dates(args.start_date, args.end_date)

    # Dry run mode
    if args.dry_run:
        logger.info("=== DRY RUN MODE ===")
        # Limit to 5 days
        from datetime import timedelta
        end_dt = min(end_dt, start_dt + timedelta(days=4))
        args.end_date = end_dt.strftime('%Y-%m-%d')
        logger.info(f"Limiting date range to: {args.start_date} to {args.end_date}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine sources to crawl
    if args.source == 'all':
        sources = ['cna', 'pts', 'technews']
    else:
        sources = [args.source]

    logger.info("=" * 70)
    logger.info("CNIRS News Crawler")
    logger.info("=" * 70)
    logger.info(f"Sources: {', '.join([s.upper() for s in sources])}")
    logger.info(f"Date range: {args.start_date} to {args.end_date}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Concurrent mode: {args.concurrent}")
    logger.info(f"Log level: {args.log_level}")
    logger.info("=" * 70)

    # Run spiders
    try:
        run_spiders(
            sources=sources,
            start_date=args.start_date,
            end_date=args.end_date,
            output_dir=args.output_dir,
            log_level=args.log_level,
            concurrent=args.concurrent
        )

        logger.info("=" * 70)
        logger.info("Crawl completed successfully!")
        logger.info(f"Output files saved to: {args.output_dir}")
        logger.info("=" * 70)

    except KeyboardInterrupt:
        logger.warning("Crawl interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Crawl failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
