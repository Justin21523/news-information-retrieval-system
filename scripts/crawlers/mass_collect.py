#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mass News Collection System - Flexible and Scalable

Support various collection scenarios:
- Date ranges: days, months, years, or specific date ranges
- Source selection: all, specific sources, or exclude sources
- Category filtering: all or specific categories
- Execution modes: sequential, parallel (with resource limits)
- Output management: separate files or combined

Author: Information Retrieval System
Date: 2025-11-19
Version: 1.0
"""

import sys
import argparse
import subprocess
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import time
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# News source configurations
NEWS_SOURCES = {
    'cna': {
        'name': '中央社 (Central News Agency)',
        'script': 'cna_spider.py',
        'speed': 'fast',
        'params': {
            'days': '-a days={days}',
        },
        'categories': None,  # No category support
    },
    'pts': {
        'name': '公視 (Public Television Service)',
        'script': 'pts_spider.py',
        'speed': 'fast',
        'params': {
            'mode': '-a mode=sequential',
            'start_id': '-a start_id=690000',    # Earliest available (Apr 2024)
            'end_id': '-a end_id=782000',        # Latest articles as of Nov 2025
            'max_articles': '-a max_articles=50000',  # Limit to prevent excessive crawling
        },
        'categories': None,
        'extra_settings': '-s CONCURRENT_REQUESTS=16 -s CONCURRENT_REQUESTS_PER_DOMAIN=8 -s DOWNLOAD_DELAY=0.5',
    },
    'ltn': {
        'name': '自由時報 (Liberty Times Net)',
        'script': 'ltn_spider.py',
        'speed': 'fast',
        'params': {
            'mode': '-a mode=sequential',
            'start_id': '-a start_id=5235000',  # Approximately 14 days ago (~1,038 IDs/day)
            'end_id': '-a end_id=5250000',      # Latest articles as of 2025-11-18
        },
        'categories': None,
        'extra_settings': '-s CONCURRENT_REQUESTS=16 -s CONCURRENT_REQUESTS_PER_DOMAIN=8 -s DOWNLOAD_DELAY=0.5',
    },
    'udn': {
        'name': '聯合報 (United Daily News)',
        'script': 'udn_spider.py',
        'speed': 'fast',
        'params': {
            'mode': '-a mode=sequential',
            'start_id': '-a start_id=9133000',  # Approximately 14 days ago (~1,071 IDs/day)
            'end_id': '-a end_id=9148000',      # Latest articles as of 2025-11-19
        },
        'categories': None,
        'extra_settings': '-s CONCURRENT_REQUESTS=16 -s CONCURRENT_REQUESTS_PER_DOMAIN=8 -s DOWNLOAD_DELAY=0.5',
    },
    'nextapple': {
        'name': '壹蘋 (Next Apple)',
        'script': 'nextapple_spider.py',
        'speed': 'fast',
        'params': {
            'sitemap': '-a sitemap=all',  # Use all sitemaps (news+editors+topics = 1,345 URLs)
            'days': '-a days={days}',
        },
        'categories': None,
        'extra_settings': '-s CONCURRENT_REQUESTS=12 -s CONCURRENT_REQUESTS_PER_DOMAIN=6 -s DOWNLOAD_DELAY=0.8',
    },
    'setn': {
        'name': '三立 (SET News)',
        'script': 'setn_spider.py',
        'speed': 'fast',
        'params': {
            'mode': '-a mode=sitemap',
            'sitemap': '-a sitemap=index',  # Use sitemap index to get all categories
            'days': '-a days={days}',
        },
        'categories': None,
        'extra_settings': '-s CONCURRENT_REQUESTS=16 -s CONCURRENT_REQUESTS_PER_DOMAIN=8 -s DOWNLOAD_DELAY=0.5',
    },
    'yahoo': {
        'name': '奇摩 (Yahoo News Taiwan)',
        'script': 'yahoo_spider.py',
        'speed': 'fast',
        'params': {
            'mode': '-a mode=sitemap',
            'sitemap': '-a sitemap=daily',
            'days': '-a days={days}',
        },
        'categories': ['politics', 'world', 'entertainment', 'sports', 'finance', 'tech', 'health', 'lifestyle'],
    },
    'storm': {
        'name': '風傳媒 (Storm Media)',
        'script': 'storm_spider.py',
        'speed': 'medium',  # Playwright optimized
        'params': {
            'mode': '-a mode=sitemap',
            'days': '-a days={days}',
        },
        'categories': None,
        'extra_settings': (
            '-s ROBOTSTXT_OBEY=False '
            '-s CONCURRENT_REQUESTS=3 '
            '-s CONCURRENT_REQUESTS_PER_DOMAIN=2 '
            '-s DOWNLOAD_DELAY=2 '
            '-s RETRY_TIMES=5 '
            '-s RETRY_HTTP_CODES=[500,502,503,504,408,429,403] '
            '-s DOWNLOAD_TIMEOUT=60 '
            '-s AUTOTHROTTLE_ENABLED=True '
            '-s AUTOTHROTTLE_START_DELAY=2 '
            '-s AUTOTHROTTLE_MAX_DELAY=10'
        ),
    },
    'tvbs': {
        'name': '新聞 (TVBS News)',
        'script': 'tvbs_spider.py',
        'speed': 'medium',  # Playwright optimized
        'params': {
            'mode': '-a mode=sitemap',
            'sitemap': '-a sitemap=latest',
            'days': '-a days={days}',
        },
        'categories': None,
        'extra_settings': (
            '-s ROBOTSTXT_OBEY=False '
            '-s CONCURRENT_REQUESTS=3 '
            '-s CONCURRENT_REQUESTS_PER_DOMAIN=2 '
            '-s DOWNLOAD_DELAY=2 '
            '-s RETRY_TIMES=5 '
            '-s RETRY_HTTP_CODES=[500,502,503,504,408,429,403] '
            '-s DOWNLOAD_TIMEOUT=60 '
            '-s AUTOTHROTTLE_ENABLED=True '
            '-s AUTOTHROTTLE_START_DELAY=2 '
            '-s AUTOTHROTTLE_MAX_DELAY=10'
        ),
    },
    'ftv': {
        'name': '民視 (Formosa TV News)',
        'script': 'ftv_spider.py',
        'speed': 'medium',  # Playwright optimized
        'params': {
            'mode': '-a mode=list',
            'category': '-a category={category}',
            'days': '-a days={days}',
        },
        'categories': ['politics', 'finance', 'culture', 'international', 'life'],
        'extra_settings': (
            '-s ROBOTSTXT_OBEY=False '
            '-s CONCURRENT_REQUESTS=2 '
            '-s CONCURRENT_REQUESTS_PER_DOMAIN=2 '
            '-s DOWNLOAD_DELAY=3 '
            '-s RETRY_TIMES=5 '
            '-s RETRY_HTTP_CODES=[500,502,503,504,408,429,403] '
            '-s DOWNLOAD_TIMEOUT=90 '
            '-s AUTOTHROTTLE_ENABLED=True '
            '-s AUTOTHROTTLE_START_DELAY=3 '
            '-s AUTOTHROTTLE_MAX_DELAY=15'
        ),
    },
    'cti': {
        'name': '中天 (China Times)',
        'script': 'cti_spider.py',
        'speed': 'slow',  # Playwright + Cloudflare
        'params': {
            'mode': '-a mode=list',  # Use list mode instead of sitemap to avoid Cloudflare
            'category': '-a category={category}',
            'days': '-a days={days}',
        },
        'categories': ['politics', 'money', 'society', 'world', 'entertainment', 'life'],
        'extra_settings': (
            '-s ROBOTSTXT_OBEY=False '
            '-s CONCURRENT_REQUESTS=1 '
            '-s CONCURRENT_REQUESTS_PER_DOMAIN=1 '
            '-s DOWNLOAD_DELAY=5 '
            '-s RETRY_TIMES=8 '
            '-s RETRY_HTTP_CODES=[500,502,503,504,408,429,403,520] '
            '-s DOWNLOAD_TIMEOUT=120 '
            '-s AUTOTHROTTLE_ENABLED=True '
            '-s AUTOTHROTTLE_START_DELAY=5 '
            '-s AUTOTHROTTLE_MAX_DELAY=30 '
            '-s AUTOTHROTTLE_TARGET_CONCURRENCY=0.5'
        ),
    },
}


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Mass News Collection System - Flexible and Scalable',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect last 14 days from all sources
  python mass_collect.py --days 14

  # Collect last 30 days from specific sources
  python mass_collect.py --days 30 --sources cna,pts,ltn

  # Collect specific date range
  python mass_collect.py --start-date 2024-01-01 --end-date 2024-12-31

  # Collect last 3 months (fast sources only)
  python mass_collect.py --months 3 --speed-filter fast

  # Collect 1 year from all sources
  python mass_collect.py --years 1

  # Exclude slow sources
  python mass_collect.py --days 14 --exclude-sources storm,tvbs,ftv,cti

  # Collect specific categories
  python mass_collect.py --days 14 --sources ftv,cti --categories politics,finance

  # Run in background with nice priority
  python mass_collect.py --days 14 --nice 10 --background
        """
    )

    # Date range options (mutually exclusive)
    date_group = parser.add_mutually_exclusive_group(required=True)
    date_group.add_argument('--days', type=int, help='Number of days to collect (e.g., 14)')
    date_group.add_argument('--months', type=int, help='Number of months to collect (e.g., 3)')
    date_group.add_argument('--years', type=int, help='Number of years to collect (e.g., 1)')
    date_group.add_argument('--date-range', nargs=2, metavar=('START', 'END'),
                           help='Specific date range (YYYY-MM-DD YYYY-MM-DD)')

    # Source selection
    parser.add_argument('--sources', type=str,
                       help='Comma-separated list of sources to collect (e.g., cna,pts,ltn). Default: all')
    parser.add_argument('--exclude-sources', type=str,
                       help='Comma-separated list of sources to exclude (e.g., storm,cti)')
    parser.add_argument('--speed-filter', choices=['fast', 'slow', 'very_slow'],
                       help='Only collect from sources with specified speed')

    # Category filtering
    parser.add_argument('--categories', type=str,
                       help='Comma-separated list of categories (e.g., politics,finance)')

    # Output options
    parser.add_argument('--output-dir', type=str, default='data/raw',
                       help='Output directory for collected data (default: data/raw)')
    parser.add_argument('--output-suffix', type=str,
                       help='Suffix for output files (e.g., "2024Q1"). Default: auto-generated')

    # Execution options
    parser.add_argument('--parallel', action='store_true',
                       help='Run all tasks in parallel (faster but uses more resources)')
    parser.add_argument('--nice', type=int, default=0,
                       help='Nice value for CPU priority (0-19, higher = lower priority). Default: 0')
    parser.add_argument('--background', action='store_true',
                       help='Run in background and save output to log file')

    # Logging options
    parser.add_argument('--log-level', choices=['ERROR', 'WARNING', 'INFO', 'DEBUG'],
                       default='INFO', help='Scrapy log level (default: INFO)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Print commands without executing')

    return parser.parse_args()


def calculate_date_range(args) -> tuple:
    """Calculate start and end dates based on arguments"""
    end_date = datetime.now()

    if args.days:
        start_date = end_date - timedelta(days=args.days)
        suffix = f"{args.days}days"
    elif args.months:
        start_date = end_date - timedelta(days=args.months * 30)
        suffix = f"{args.months}months"
    elif args.years:
        start_date = end_date - timedelta(days=args.years * 365)
        suffix = f"{args.years}years"
    elif args.date_range:
        start_date = datetime.strptime(args.date_range[0], '%Y-%m-%d')
        end_date = datetime.strptime(args.date_range[1], '%Y-%m-%d')
        suffix = f"{args.date_range[0]}_to_{args.date_range[1]}"
    else:
        raise ValueError("Date range not specified")

    days = (end_date - start_date).days
    return start_date, end_date, days, suffix


def filter_sources(args) -> List[str]:
    """Filter sources based on arguments"""
    sources = list(NEWS_SOURCES.keys())

    # Apply source selection
    if args.sources:
        selected = [s.strip() for s in args.sources.split(',')]
        sources = [s for s in sources if s in selected]

    # Apply exclusion
    if args.exclude_sources:
        excluded = [s.strip() for s in args.exclude_sources.split(',')]
        sources = [s for s in sources if s not in excluded]

    # Apply speed filter
    if args.speed_filter:
        sources = [s for s in sources if NEWS_SOURCES[s]['speed'] == args.speed_filter]

    return sources


def build_command(source_id: str, days: int, category: Optional[str], args) -> Dict:
    """Build scrapy command for a source"""
    source = NEWS_SOURCES[source_id]
    script_path = PROJECT_ROOT / 'scripts' / 'crawlers' / source['script']

    # Determine output filename
    if args.output_suffix:
        suffix = args.output_suffix
    else:
        _, _, _, suffix = calculate_date_range(args)

    if category:
        output_file = f"{source_id}_{category}_{suffix}.jsonl"
    else:
        output_file = f"{source_id}_{suffix}.jsonl"

    output_path = PROJECT_ROOT / args.output_dir / output_file

    # Build command string (with conda environment activation)
    cmd_str = f'source activate ai_env && python -m scrapy runspider {script_path}'

    # Add source-specific parameters
    for param_key, param_template in source['params'].items():
        if param_key == 'days':
            cmd_str += f' {param_template.format(days=days)}'
        elif param_key == 'category' and category:
            cmd_str += f' {param_template.format(category=category)}'
        elif param_key != 'category':
            cmd_str += f' {param_template}'

    # Add output
    cmd_str += f' -o {output_path}'

    # Add log level
    cmd_str += f' -s LOG_LEVEL={args.log_level}'

    # Add encoding settings to ensure proper Chinese character display
    cmd_str += ' -s FEED_EXPORT_ENCODING=utf-8'
    cmd_str += ' -s ENSURE_ASCII=False'

    # Add extra settings if any
    if 'extra_settings' in source:
        cmd_str += f' {source["extra_settings"]}'

    # Wrap in bash -c
    cmd_parts = ['bash', '-c', cmd_str]

    return {
        'source_id': source_id,
        'source_name': source['name'],
        'category': category,
        'command': cmd_parts,
        'output_file': str(output_path),
        'speed': source['speed'],
    }


def execute_collection(commands: List[Dict], args):
    """Execute collection commands"""
    total = len(commands)
    successful = 0
    failed = 0

    logger.info(f"Starting collection of {total} tasks")
    logger.info(f"Output directory: {PROJECT_ROOT / args.output_dir}")

    if args.parallel:
        logger.info("Execution mode: PARALLEL (all tasks running simultaneously)")
    else:
        logger.info("Execution mode: SEQUENTIAL (one task at a time)")

    start_time = time.time()

    if args.parallel:
        # Parallel execution mode
        processes = []

        for idx, cmd_info in enumerate(commands, 1):
            source_name = cmd_info['source_name']
            category = cmd_info['category']
            category_str = f" [{category}]" if category else ""

            logger.info(f"[{idx}/{total}] Starting {source_name}{category_str}...")

            if args.dry_run:
                logger.info(f"DRY RUN: {' '.join(cmd_info['command'])}")
                continue

            try:
                # Apply nice priority if specified
                cmd = cmd_info['command'].copy()
                if args.nice > 0:
                    cmd = ['nice', '-n', str(args.nice)] + cmd

                # Start process in background
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=str(PROJECT_ROOT)
                )
                processes.append({
                    'process': process,
                    'name': source_name,
                    'category': category_str,
                    'idx': idx
                })
                logger.info(f"  ✓ Started (PID: {process.pid})")

            except Exception as e:
                logger.error(f"✗ {source_name}{category_str} failed to start: {str(e)}")
                failed += 1

        # Wait for all processes to complete
        logger.info(f"\nAll {len(processes)} tasks started. Waiting for completion...")

        for proc_info in processes:
            try:
                stdout, stderr = proc_info['process'].communicate(timeout=14400)  # 4 hour timeout

                if proc_info['process'].returncode == 0:
                    logger.info(f"✓ [{proc_info['idx']}/{total}] {proc_info['name']}{proc_info['category']} completed successfully")
                    successful += 1
                else:
                    logger.error(f"✗ [{proc_info['idx']}/{total}] {proc_info['name']}{proc_info['category']} failed with exit code {proc_info['process'].returncode}")
                    failed += 1

            except subprocess.TimeoutExpired:
                proc_info['process'].kill()
                logger.error(f"✗ [{proc_info['idx']}/{total}] {proc_info['name']}{proc_info['category']} timed out (4 hours)")
                failed += 1
            except Exception as e:
                logger.error(f"✗ [{proc_info['idx']}/{total}] {proc_info['name']}{proc_info['category']} error: {str(e)}")
                failed += 1

    else:
        # Sequential execution mode (original)
        for idx, cmd_info in enumerate(commands, 1):
            source_name = cmd_info['source_name']
            category = cmd_info['category']
            category_str = f" [{category}]" if category else ""

            logger.info(f"[{idx}/{total}] Collecting {source_name}{category_str}...")

            if args.dry_run:
                logger.info(f"DRY RUN: {' '.join(cmd_info['command'])}")
                continue

            try:
                # Apply nice priority if specified
                if args.nice > 0:
                    cmd_info['command'] = ['nice', '-n', str(args.nice)] + cmd_info['command']

                # Execute command
                result = subprocess.run(
                    cmd_info['command'],
                    capture_output=True,
                    text=True,
                    cwd=str(PROJECT_ROOT),
                    timeout=14400  # 4 hour timeout per source
                )

                if result.returncode == 0:
                    logger.info(f"✓ {source_name}{category_str} completed successfully")
                    successful += 1
                else:
                    logger.error(f"✗ {source_name}{category_str} failed with exit code {result.returncode}")
                    failed += 1

            except subprocess.TimeoutExpired:
                logger.error(f"✗ {source_name}{category_str} timed out (4 hours)")
                failed += 1
            except Exception as e:
                logger.error(f"✗ {source_name}{category_str} error: {str(e)}")
                failed += 1

    # Summary
    duration = time.time() - start_time
    hours = int(duration // 3600)
    minutes = int((duration % 3600) // 60)
    seconds = int(duration % 60)

    logger.info("=" * 70)
    logger.info("Collection Summary")
    logger.info("=" * 70)
    logger.info(f"Total tasks: {total}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Duration: {hours}h {minutes}m {seconds}s")
    logger.info("=" * 70)


def main():
    """Main execution function"""
    args = parse_arguments()

    # Calculate date range
    start_date, end_date, days, suffix = calculate_date_range(args)
    logger.info(f"Date range: {start_date.date()} to {end_date.date()} ({days} days)")

    # Filter sources
    sources = filter_sources(args)
    if not sources:
        logger.error("No sources selected after filtering")
        sys.exit(1)

    logger.info(f"Selected sources: {', '.join(sources)}")

    # Build commands
    commands = []
    for source_id in sources:
        source = NEWS_SOURCES[source_id]

        # Check if source supports categories
        if source.get('categories'):
            # If categories argument provided, use selected categories
            if args.categories:
                selected_categories = [c.strip() for c in args.categories.split(',')]
                available_categories = source['categories']
                categories = [c for c in selected_categories if c in available_categories]
            else:
                # AUTO: Use ALL available categories for category-supported sources
                categories = source['categories']

            for category in categories:
                commands.append(build_command(source_id, days, category, args))
        else:
            commands.append(build_command(source_id, days, None, args))

    logger.info(f"Total collection tasks: {len(commands)}")

    # Create output directory
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Execute collection
    execute_collection(commands, args)


if __name__ == '__main__':
    main()
