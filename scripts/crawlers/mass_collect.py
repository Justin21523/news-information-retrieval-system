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
import os
import shutil
import shlex
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

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
        'script': 'cna_spider_v2.py',
        'speed': 'medium',
        'playwright': True,
        'params': {
            'start_date': '-a start_date={start_date}',
            'end_date': '-a end_date={end_date}',
        },
        'categories': None,  # No category support
    },
    'pts': {
        'name': '公視 (Public Television Service)',
        'script': 'pts_spider.py',
        'speed': 'fast',
        'playwright': False,
        'params': {
            'mode': '-a mode=sequential',
            'start_date': '-a start_date={start_date}',
            'end_date': '-a end_date={end_date}',
            'max_articles': '-a max_articles=50000',  # Safety limit
        },
        'categories': None,
        'extra_settings': '-s CONCURRENT_REQUESTS=16 -s CONCURRENT_REQUESTS_PER_DOMAIN=8 -s DOWNLOAD_DELAY=0.5',
    },
    'ltn': {
        'name': '自由時報 (Liberty Times Net)',
        'script': 'ltn_spider.py',
        'speed': 'fast',
        'playwright': False,
        'params': {
            'mode': '-a mode=sequential',
            'start_date': '-a start_date={start_date}',
            'end_date': '-a end_date={end_date}',
        },
        'categories': None,
        'extra_settings': '-s CONCURRENT_REQUESTS=16 -s CONCURRENT_REQUESTS_PER_DOMAIN=8 -s DOWNLOAD_DELAY=0.5',
    },
    'udn': {
        'name': '聯合報 (United Daily News)',
        'script': 'udn_spider.py',
        'speed': 'fast',
        'playwright': False,
        'params': {
            'mode': '-a mode=sequential',
            'start_date': '-a start_date={start_date}',
            'end_date': '-a end_date={end_date}',
        },
        'categories': None,
        'extra_settings': '-s CONCURRENT_REQUESTS=16 -s CONCURRENT_REQUESTS_PER_DOMAIN=8 -s DOWNLOAD_DELAY=0.5',
    },
    'nextapple': {
        'name': '壹蘋 (Next Apple)',
        'script': 'nextapple_spider.py',
        'speed': 'fast',
        'playwright': False,
        'params': {
            'sitemap': '-a sitemap=all',  # Use all sitemaps (news+editors+topics = 1,345 URLs)
            'start_date': '-a start_date={start_date}',
            'end_date': '-a end_date={end_date}',
        },
        'categories': None,
        'extra_settings': '-s CONCURRENT_REQUESTS=12 -s CONCURRENT_REQUESTS_PER_DOMAIN=6 -s DOWNLOAD_DELAY=0.8',
    },
    'setn': {
        'name': '三立 (SET News)',
        'script': 'setn_spider.py',
        'speed': 'fast',
        'playwright': False,
        'params': {
            'mode': '-a mode=sitemap',
            'sitemap': '-a sitemap=index',  # Use sitemap index to get all categories
            'category': '-a category={category}',
            'start_date': '-a start_date={start_date}',
            'end_date': '-a end_date={end_date}',
        },
        'topic_categories': {
            'politics': ['news'],
            'finance': ['finance'],
        },
        'categories': None,
        'extra_settings': '-s CONCURRENT_REQUESTS=16 -s CONCURRENT_REQUESTS_PER_DOMAIN=8 -s DOWNLOAD_DELAY=0.5',
    },
    'yahoo': {
        'name': '奇摩 (Yahoo News Taiwan)',
        'script': 'yahoo_spider.py',
        'speed': 'fast',
        'playwright': False,
        'params': {
            'mode': '-a mode=sitemap',
            'sitemap': '-a sitemap=daily',
            'start_date': '-a start_date={start_date}',
            'end_date': '-a end_date={end_date}',
        },
        # NOTE: The current Yahoo spider does not reliably support category-scoped crawling.
        # Treat it as an all-category source to avoid duplicated work.
        'categories': None,
    },
    'chinatimes': {
        'name': '中時新聞網 (China Times News)',
        'script': 'chinatimes_spider.py',
        'speed': 'fast',
        'playwright': False,
        'params': {
            'start_date': '-a start_date={start_date}',
            'end_date': '-a end_date={end_date}',
        },
        'categories': None,
    },
    'ettoday': {
        'name': '東森新聞雲 (ETtoday)',
        'script': 'ettoday_spider.py',
        'speed': 'medium',
        'playwright': True,
        'params': {
            'category': '-a category={category}',
            'start_date': '-a start_date={start_date}',
            'end_date': '-a end_date={end_date}',
        },
        'topic_categories': {
            'politics': ['politics'],
            'finance': ['finance'],
        },
        'categories': None,
    },
    'apple': {
        'name': '蘋果日報 / NextApple (Apple Daily)',
        'script': 'apple_daily_spider.py',
        'speed': 'medium',
        'playwright': True,
        'params': {
            'category': '-a category={category}',
            'start_date': '-a start_date={start_date}',
            'end_date': '-a end_date={end_date}',
        },
        'topic_categories': {
            'politics': ['politics'],
            'finance': ['economy'],
        },
        'categories': None,
    },
    'storm': {
        'name': '風傳媒 (Storm Media)',
        'script': 'storm_spider.py',
        'speed': 'medium',  # Playwright optimized
        'playwright': True,
        'params': {
            'mode': '-a mode=hybrid',
            'start_date': '-a start_date={start_date}',
            'end_date': '-a end_date={end_date}',
        },
        'categories': None,
        'extra_settings': (
            '-s ROBOTSTXT_OBEY=False '
            '-s CONCURRENT_REQUESTS=3 '
            '-s CONCURRENT_REQUESTS_PER_DOMAIN=2 '
            '-s DOWNLOAD_DELAY=2 '
            '-s RETRY_TIMES=5 '
            '-s RETRY_HTTP_CODES=500,502,503,504,408,429,403 '
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
        'playwright': True,
        'params': {
            'mode': '-a mode=sitemap',
            'sitemap': '-a sitemap=index',
            'category': '-a category={category}',
            'start_date': '-a start_date={start_date}',
            'end_date': '-a end_date={end_date}',
        },
        'topic_categories': {
            'politics': ['politics'],
            'finance': ['money'],
        },
        'categories': None,
        'extra_settings': (
            '-s ROBOTSTXT_OBEY=False '
            '-s CONCURRENT_REQUESTS=3 '
            '-s CONCURRENT_REQUESTS_PER_DOMAIN=2 '
            '-s DOWNLOAD_DELAY=2 '
            '-s RETRY_TIMES=5 '
            '-s RETRY_HTTP_CODES=500,502,503,504,408,429,403 '
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
        'playwright': True,
        'params': {
            'category': '-a category={category}',
            'days': '-a days={days}',
        },
        'categories': ['politics', 'business', 'society', 'entertainment', 'life', 'sports', 'tech', 'world'],
        'topic_categories': {
            'politics': ['politics'],
            'finance': ['business'],
        },
        'extra_settings': (
            '-s ROBOTSTXT_OBEY=False '
            '-s CONCURRENT_REQUESTS=2 '
            '-s CONCURRENT_REQUESTS_PER_DOMAIN=2 '
            '-s DOWNLOAD_DELAY=3 '
            '-s RETRY_TIMES=5 '
            '-s RETRY_HTTP_CODES=500,502,503,504,408,429,403 '
            '-s DOWNLOAD_TIMEOUT=90 '
            '-s AUTOTHROTTLE_ENABLED=True '
            '-s AUTOTHROTTLE_START_DELAY=3 '
            '-s AUTOTHROTTLE_MAX_DELAY=15'
        ),
    },
    'cti': {
        'name': '中天新聞網 (CTI News)',
        'script': 'cti_spider.py',
        'speed': 'fast',  # Sitemap-based, no Playwright
        'playwright': False,
        'params': {
            'sitemap': '-a sitemap=all',
            'start_date': '-a start_date={start_date}',
            'end_date': '-a end_date={end_date}',
        },
        'categories': None,
        'extra_settings': '-s CONCURRENT_REQUESTS=12 -s CONCURRENT_REQUESTS_PER_DOMAIN=6 -s DOWNLOAD_DELAY=0.5',
    },
}

TOPIC_ALIASES = {
    'politics': {'politics', 'politic', '政治', '政治時事', '時事', '政經'},
    'finance': {'finance', 'economy', 'money', 'business', '財經', '經濟', '股票', '股市'},
}


def normalize_topics(topics: Optional[str]) -> List[str]:
    """
    Normalize user-provided topic aliases into canonical topic codes.

    Args:
        topics: Comma-separated topic list (e.g., "politics,finance" or "政治,財經").

    Returns:
        List[str]: Canonical topics in a stable order.

    Complexity:
        Time: O(t * a) where t = token count, a = alias set size
        Space: O(t)
    """
    if not topics:
        return []

    tokens = [t.strip() for t in topics.split(',') if t.strip()]
    normalized: List[str] = []
    for token in tokens:
        token_norm = token.strip().lower()
        matched = None
        for canonical, aliases in TOPIC_ALIASES.items():
            if token_norm in {a.lower() for a in aliases}:
                matched = canonical
                break
        if matched and matched not in normalized:
            normalized.append(matched)
    return normalized


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
  python mass_collect.py --date-range 2024-01-01 2024-12-31

  # Collect last 3 months (fast sources only)
  python mass_collect.py --months 3 --speed-filter fast

  # Collect 1 year from all sources
  python mass_collect.py --years 1

  # Exclude slow sources
  python mass_collect.py --days 14 --exclude-sources storm,tvbs,ftv,cti

  # Collect specific categories
  python mass_collect.py --days 14 --sources ftv,cti --categories politics,finance

  # Collect 1 year for politics + finance topics (category-capable sources only)
  python mass_collect.py --years 1 --topics politics,finance --parallel --max-workers 32 --max-playwright-workers 4

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
    parser.add_argument('--speed-filter', choices=['fast', 'medium', 'slow'],
                       help='Only collect from sources with specified speed')

    # Category / topic filtering
    filter_group = parser.add_mutually_exclusive_group()
    filter_group.add_argument('--categories', type=str,
                              help='Comma-separated list of categories (e.g., politics,finance)')
    filter_group.add_argument('--topics', type=str,
                              help='Comma-separated topics (politics,finance / 政治,財經). '
                                   'Sources without category support will crawl all.')

    # Output options
    parser.add_argument('--output-dir', type=str, default='/mnt/c/data/information-retrieval/raw',
                       help='Output directory for collected data (default: /mnt/c/data/information-retrieval/raw)')
    parser.add_argument('--output-suffix', type=str,
                       help='Suffix for output files (e.g., "2024Q1"). Default: auto-generated')

    # Execution options
    parser.add_argument('--parallel', action='store_true',
                       help='Run all tasks in parallel (faster but uses more resources)')
    parser.add_argument('--max-workers', type=int, default=4,
                       help='Maximum concurrent tasks when using --parallel (default: 4)')
    parser.add_argument('--max-playwright-workers', type=int, default=2,
                       help='Maximum concurrent Playwright tasks when using --parallel (default: 2)')
    parser.add_argument('--nice', type=int, default=0,
                       help='Nice value for CPU priority (0-19, higher = lower priority). Default: 0')
    parser.add_argument('--background', action='store_true',
                       help='Run in background and save output to log file')
    parser.add_argument('--timeout-hours', type=int, default=4,
                       help='Per-task timeout in hours (default: 4, 0 = no timeout)')
    parser.add_argument('--retries', type=int, default=0,
                       help='Retry failed tasks N times (default: 0)')
    parser.add_argument('--retry-backoff-seconds', type=int, default=60,
                       help='Backoff seconds between retries (default: 60)')
    parser.add_argument('--jobdir', type=str, default='data/jobdir',
                       help='Base JOBDIR for Scrapy resume (default: data/jobdir)')
    parser.add_argument('--jobdir-disable-sources', type=str,
                       help='Comma-separated sources to run WITHOUT JOBDIR (default: none)')
    parser.add_argument('--jobdir-reset-sources', type=str,
                       help='Comma-separated sources to RESET JOBDIR before running (default: none)')
    parser.add_argument('--jobdir-repair-on-corruption', action='store_true',
                       help='On disk-queue corruption, backup/reset JOBDIR and retry (requires --retries >= 1)')
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip tasks whose output file already exists and is non-empty')

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


def _safe_slug(text: str) -> str:
    """
    Build a filesystem-safe slug for log/jobdir names.

    Complexity:
        Time: O(n) where n = len(text)
        Space: O(n)
    """
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    return ''.join(ch if ch in allowed else '_' for ch in text)


def _parse_csv_set(value: Optional[str]) -> set[str]:
    """
    Parse a comma-separated list into a set of tokens.

    Complexity:
        Time: O(n) where n = len(value)
        Space: O(k) where k = number of tokens
    """
    if not value:
        return set()
    return {token.strip() for token in value.split(',') if token.strip()}


def _read_file_tail(path: Path, max_bytes: int = 120_000) -> str:
    """
    Read the tail of a text file to avoid loading huge logs into memory.

    Complexity:
        Time: O(max_bytes)
        Space: O(max_bytes)
    """
    try:
        with open(path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            offset = max(0, size - int(max_bytes))
            f.seek(offset, os.SEEK_SET)
            data = f.read()
        return data.decode('utf-8', errors='replace')
    except FileNotFoundError:
        return ""


def _log_indicates_jobdir_corruption(log_file: Path) -> bool:
    """
    Detect common Scrapy/queuelib disk-queue corruption signatures in logs.

    Complexity:
        Time: O(max_bytes)
        Space: O(max_bytes)
    """
    tail = _read_file_tail(log_file)
    if not tail:
        return False

    has_queue_stack = any(marker in tail for marker in ("queuelib/queue.py", "scrapy/squeues.py", "scrapy/pqueues.py"))
    has_corruption = any(
        marker in tail
        for marker in (
            "OSError: [Errno 22]",
            "Invalid argument",
            "unpack requires a buffer",
            "EOFError",
        )
    )
    return has_queue_stack and has_corruption


def _backup_and_reset_jobdir(jobdir: Path, reason: str) -> Optional[Path]:
    """
    Backup an existing JOBDIR directory and recreate a fresh one.

    Complexity:
        Time: O(1) average (rename within filesystem)
        Space: O(1)
    """
    try:
        backup_path = None
        if jobdir.exists():
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = jobdir.with_name(f"{jobdir.name}__{_safe_slug(reason)}_{ts}")
            shutil.move(str(jobdir), str(backup_path))
        jobdir.mkdir(parents=True, exist_ok=True)
        return backup_path
    except Exception as e:
        logger.warning(f"Failed to backup/reset JOBDIR {jobdir}: {e}")
        try:
            jobdir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return None


def build_command(
    source_id: str,
    start_date: datetime,
    end_date: datetime,
    days: int,
    category: Optional[str],
    output_tag: Optional[str],
    suffix: str,
    args,
) -> Dict:
    """Build scrapy command for a source"""
    source = NEWS_SOURCES[source_id]
    script_path = PROJECT_ROOT / 'scripts' / 'crawlers' / source['script']

    # Determine output filename
    final_suffix = args.output_suffix or suffix

    label = output_tag or category
    if label:
        output_file = f"{source_id}_{label}_{final_suffix}.jsonl"
    else:
        output_file = f"{source_id}_{final_suffix}.jsonl"

    output_path = PROJECT_ROOT / args.output_dir / output_file

    if args.skip_existing and output_path.exists() and output_path.stat().st_size > 0:
        return {
            'source_id': source_id,
            'source_name': source['name'],
            'category': category,
            'output_tag': output_tag,
            'command': None,
            'output_file': str(output_path),
            'speed': source['speed'],
            'playwright': bool(source.get('playwright')),
            'skipped': True,
        }

    # Build command args (avoid shell for safety/portability)
    cmd_parts = [sys.executable, '-m', 'scrapy', 'runspider', str(script_path)]

    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    # Add source-specific parameters
    for param_key, param_template in source['params'].items():
        formatted: Optional[str] = None
        if param_key == 'days':
            formatted = param_template.format(days=days)
        elif param_key == 'start_date':
            formatted = param_template.format(start_date=start_date_str)
        elif param_key == 'end_date':
            formatted = param_template.format(end_date=end_date_str)
        elif param_key == 'category':
            if category:
                formatted = param_template.format(category=category)
        else:
            formatted = param_template

        if formatted:
            cmd_parts += shlex.split(formatted)

    # Add output (append mode)
    cmd_parts += ['-o', str(output_path)]

    # Add log level
    cmd_parts += ['-s', f'LOG_LEVEL={args.log_level}']

    # Add encoding settings to ensure proper Chinese character display
    cmd_parts += ['-s', 'FEED_EXPORT_ENCODING=utf-8']
    cmd_parts += ['-s', 'ENSURE_ASCII=False']

    # Add JOBDIR for resume support
    jobdir_path: Optional[Path] = None
    disable_sources = getattr(args, "_jobdir_disable_sources", set())
    reset_sources = getattr(args, "_jobdir_reset_sources", set())
    if args.jobdir and source_id not in disable_sources:
        job_category = output_tag or category or 'all'
        jobdir_path = PROJECT_ROOT / args.jobdir / source_id / _safe_slug(job_category) / _safe_slug(final_suffix)
        jobdir_path.mkdir(parents=True, exist_ok=True)
        cmd_parts += ['-s', f'JOBDIR={jobdir_path}']

    # Add extra settings if any
    if 'extra_settings' in source:
        cmd_parts += shlex.split(source['extra_settings'])

    return {
        'source_id': source_id,
        'source_name': source['name'],
        'category': category,
        'output_tag': output_tag,
        'command': cmd_parts,
        'output_file': str(output_path),
        'speed': source['speed'],
        'playwright': bool(source.get('playwright')),
        'jobdir': str(jobdir_path) if jobdir_path else None,
        'jobdir_reset': bool(jobdir_path and source_id in reset_sources),
        'skipped': False,
    }


def _run_task(cmd_info: Dict, args, timeout_seconds: Optional[int], log_dir: Path,
              playwright_semaphore: threading.Semaphore) -> Dict:
    """
    Run a single collection task.

    Complexity:
        Time: O(1) (excludes external process runtime)
        Space: O(1)
    """
    if cmd_info.get('skipped'):
        return {**cmd_info, 'returncode': 0, 'status': 'skipped', 'log_file': None}

    source_id = cmd_info['source_id']
    category = cmd_info.get('category')
    label = cmd_info.get('output_tag') or category
    category_slug = _safe_slug(label) if label else 'all'
    log_file = log_dir / f"{source_id}_{category_slug}_{_safe_slug(Path(cmd_info['output_file']).stem)}.log"

    jobdir = Path(cmd_info['jobdir']) if cmd_info.get('jobdir') else None
    if jobdir and cmd_info.get('jobdir_reset'):
        backup = _backup_and_reset_jobdir(jobdir, reason="reset")
        if backup:
            logger.info(f"Reset JOBDIR for {source_id}: {jobdir} -> {backup}")

    cmd = cmd_info['command']
    if args.nice > 0:
        cmd = ['nice', '-n', str(args.nice)] + cmd

    # Limit concurrent Playwright tasks to avoid timeouts/OOM.
    semaphore = playwright_semaphore if cmd_info.get('playwright') else None

    max_attempts = max(1, int(args.retries) + 1)
    for attempt in range(1, max_attempts + 1):
        try:
            with (semaphore or _NullContext()):
                mode = 'w' if attempt == 1 else 'a'
                with open(log_file, mode, encoding='utf-8') as lf:
                    if attempt > 1:
                        lf.write("\n\n")
                    lf.write(f"# Task: {cmd_info['source_name']} ({source_id})\n")
                    if cmd_info.get('output_tag'):
                        lf.write(f"# Topic: {cmd_info['output_tag']}\n")
                    lf.write(f"# Category: {category or ''}\n")
                    lf.write(f"# Output: {cmd_info['output_file']}\n")
                    lf.write(f"# Attempt: {attempt}/{max_attempts}\n")
                    if jobdir:
                        lf.write(f"# JOBDIR: {jobdir}\n")
                    lf.write(f"# Command: {' '.join(cmd)}\n\n")
                    lf.flush()

                    result = subprocess.run(
                        cmd,
                        stdout=lf,
                        stderr=subprocess.STDOUT,
                        text=True,
                        cwd=str(PROJECT_ROOT),
                        timeout=timeout_seconds,
                    )

            ok = result.returncode == 0
            output_path = Path(cmd_info['output_file'])
            has_output = output_path.exists() and output_path.stat().st_size > 0

            corruption = bool(
                jobdir
                and args.jobdir_repair_on_corruption
                and _log_indicates_jobdir_corruption(log_file)
            )

            if ok and has_output and not corruption:
                return {**cmd_info, 'returncode': result.returncode, 'status': 'success', 'log_file': str(log_file)}

            if attempt < max_attempts and corruption:
                backup = _backup_and_reset_jobdir(jobdir, reason="corrupt_jobdir")
                with open(log_file, 'a', encoding='utf-8') as lf:
                    lf.write("\n")
                    lf.write("# Detected disk-queue corruption; JOBDIR has been reset for retry.\n")
                    if backup:
                        lf.write(f"# JOBDIR backup: {backup}\n")

            if attempt < max_attempts:
                time.sleep(int(args.retry_backoff_seconds))
                continue

            return {**cmd_info, 'returncode': result.returncode, 'status': 'failed', 'log_file': str(log_file)}

        except subprocess.TimeoutExpired:
            if attempt < max_attempts:
                time.sleep(int(args.retry_backoff_seconds))
                continue
            return {**cmd_info, 'returncode': 124, 'status': 'timeout', 'log_file': str(log_file)}
        except Exception as e:
            # Treat unexpected runner errors as failures.
            with open(log_file, 'a', encoding='utf-8') as lf:
                lf.write(f"\n# Runner error: {e}\n")
            if attempt < max_attempts:
                time.sleep(int(args.retry_backoff_seconds))
                continue
            return {**cmd_info, 'returncode': 1, 'status': 'error', 'log_file': str(log_file)}


class _NullContext:
    """Fallback context manager."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def execute_collection(commands: List[Dict], args):
    """Execute collection commands."""
    total = len(commands)
    successful = 0
    failed = 0
    skipped = 0

    logger.info(f"Starting collection of {total} tasks")
    logger.info(f"Output directory: {PROJECT_ROOT / args.output_dir}")

    if args.parallel:
        logger.info(f"Execution mode: PARALLEL (max_workers={args.max_workers}, max_playwright_workers={args.max_playwright_workers})")
    else:
        logger.info("Execution mode: SEQUENTIAL (one task at a time)")

    start_time = time.time()
    timeout_seconds = None if int(args.timeout_hours) <= 0 else int(args.timeout_hours) * 3600

    log_dir = PROJECT_ROOT / 'logs' / 'mass_collect'
    log_dir.mkdir(parents=True, exist_ok=True)

    playwright_semaphore = threading.Semaphore(max(1, int(args.max_playwright_workers)))

    if args.parallel:
        # Parallel execution mode with worker limits (avoid PIPE deadlocks by logging to files)
        if args.dry_run:
            for idx, cmd_info in enumerate(commands, 1):
                if cmd_info.get('skipped'):
                    logger.info(f"[{idx}/{total}] SKIP existing: {cmd_info['source_name']}")
                    continue
                logger.info(f"[{idx}/{total}] DRY RUN: {' '.join(cmd_info['command'])}")
            return

        with ThreadPoolExecutor(max_workers=max(1, int(args.max_workers))) as pool:
            futures = {
                pool.submit(_run_task, cmd_info, args, timeout_seconds, log_dir, playwright_semaphore): cmd_info
                for cmd_info in commands
            }

            for fut in as_completed(futures):
                result = fut.result()
                status = result.get('status')
                name = result.get('source_name')
                category = result.get('category')
                label = result.get('output_tag') or category
                category_str = f" [{label}]" if label else ""
                if status == 'success':
                    logger.info(f"✓ {name}{category_str} completed (log: {result.get('log_file')})")
                    successful += 1
                elif status == 'skipped':
                    logger.info(f"− {name}{category_str} skipped (existing output)")
                    skipped += 1
                else:
                    logger.error(f"✗ {name}{category_str} {status} (log: {result.get('log_file')})")
                    failed += 1

    else:
        # Sequential execution mode (original)
        for idx, cmd_info in enumerate(commands, 1):
            source_name = cmd_info['source_name']
            category = cmd_info['category']
            label = cmd_info.get('output_tag') or category
            category_str = f" [{label}]" if label else ""

            logger.info(f"[{idx}/{total}] Collecting {source_name}{category_str}...")

            if args.dry_run:
                if cmd_info.get('skipped'):
                    logger.info("SKIP existing output")
                else:
                    logger.info(f"DRY RUN: {' '.join(cmd_info['command'])}")
                continue

            result = _run_task(cmd_info, args, timeout_seconds, log_dir, playwright_semaphore)
            if result.get('status') == 'success':
                logger.info(f"✓ {source_name}{category_str} completed (log: {result.get('log_file')})")
                successful += 1
            elif result.get('status') == 'skipped':
                logger.info(f"− {source_name}{category_str} skipped (existing output)")
                skipped += 1
            else:
                logger.error(f"✗ {source_name}{category_str} {result.get('status')} (log: {result.get('log_file')})")
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
    logger.info(f"Skipped: {skipped}")
    logger.info(f"Duration: {hours}h {minutes}m {seconds}s")
    logger.info("=" * 70)


def main():
    """Main execution function"""
    args = parse_arguments()

    # Calculate date range
    start_date, end_date, days, suffix = calculate_date_range(args)
    logger.info(f"Date range: {start_date.date()} to {end_date.date()} ({days} days)")

    if args.background and not args.dry_run:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = PROJECT_ROOT / 'logs' / 'mass_collect'
        log_dir.mkdir(parents=True, exist_ok=True)

        output_suffix = args.output_suffix or suffix
        log_file = log_dir / f"mass_collect_{_safe_slug(output_suffix)}_{timestamp}.out"

        cmd = [sys.executable, str(Path(__file__).resolve())]
        cmd += [a for a in sys.argv[1:] if a != '--background']

        with open(log_file, 'w', encoding='utf-8') as lf:
            lf.write(f"# Started at: {datetime.now().isoformat()}\n")
            lf.write(f"# Command: {' '.join(cmd)}\n\n")
            lf.flush()
            proc = subprocess.Popen(
                cmd,
                stdout=lf,
                stderr=subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),
                start_new_session=True,
                text=True,
            )

        logger.info(f"Started background collection PID={proc.pid} (log: {log_file})")
        return

    args._jobdir_disable_sources = _parse_csv_set(args.jobdir_disable_sources)
    args._jobdir_reset_sources = _parse_csv_set(args.jobdir_reset_sources)

    # Filter sources
    sources = filter_sources(args)
    if not sources:
        logger.error("No sources selected after filtering")
        sys.exit(1)

    logger.info(f"Selected sources: {', '.join(sources)}")

    # Build commands
    requested_topics = normalize_topics(getattr(args, 'topics', None))
    if getattr(args, 'topics', None) and not requested_topics:
        logger.warning("No valid topics parsed from --topics; falling back to full crawling.")

    commands = []
    for source_id in sources:
        source = NEWS_SOURCES[source_id]

        if requested_topics:
            topic_map = source.get('topic_categories') or {}
            supports_category = 'category' in (source.get('params') or {})
            topic_tasks = []

            if topic_map and supports_category:
                for topic in requested_topics:
                    mapped = topic_map.get(topic, []) or []
                    if isinstance(mapped, str):
                        mapped = [mapped]
                    for spider_category in mapped:
                        if not spider_category:
                            continue
                        output_tag = topic if len(mapped) == 1 else f"{topic}_{spider_category}"
                        topic_tasks.append((spider_category, output_tag))

            if topic_tasks:
                seen = set()
                for spider_category, output_tag in topic_tasks:
                    key = (source_id, spider_category, output_tag)
                    if key in seen:
                        continue
                    seen.add(key)
                    commands.append(
                        build_command(
                            source_id,
                            start_date,
                            end_date,
                            days,
                            spider_category,
                            output_tag,
                            suffix,
                            args,
                        )
                    )
                continue

            commands.append(build_command(source_id, start_date, end_date, days, None, None, suffix, args))
            continue

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
                commands.append(build_command(source_id, start_date, end_date, days, category, None, suffix, args))
        else:
            commands.append(build_command(source_id, start_date, end_date, days, None, None, suffix, args))

    logger.info(f"Total collection tasks: {len(commands)}")

    # Create output directory
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Execute collection
    execute_collection(commands, args)


if __name__ == '__main__':
    main()
