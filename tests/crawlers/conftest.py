#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pytest configuration for crawler tests.

This file contains shared fixtures and configuration for all crawler tests.

Author: Information Retrieval System
Date: 2025-11-18
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# ========== Test Configuration ==========

@pytest.fixture(scope="session")
def test_config():
    """
    Shared test configuration.

    Returns:
        dict: Test configuration parameters
    """
    return {
        'test_days': 1,  # Only test 1 day of data
        'test_items': 2,  # Limit to 2 items per test
        'timeout': 120,  # 120 seconds timeout
        'output_dir': project_root / 'data' / 'test_output',
    }


@pytest.fixture(scope="session")
def date_range():
    """
    Provide test date range (yesterday to today).

    Returns:
        tuple: (start_date, end_date) in YYYY-MM-DD format
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1)

    return (
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )


# ========== Crawler Fixtures ==========

@pytest.fixture(scope="module")
def crawler_registry():
    """
    Registry of all available crawlers.

    Returns:
        dict: Crawler configurations
    """
    return {
        'cna': {
            'name': 'CNA中央社',
            'module': 'scripts.crawlers.cna_spider',
            'class': 'CNANewsSpider',
            'playwright': False,
            'categories': ['politics', 'finance', 'society'],
        },
        'ltn': {
            'name': '自由時報',
            'module': 'scripts.crawlers.ltn_spider',
            'class': 'LTNNewsSpider',
            'playwright': True,
            'categories': ['politics', 'business', 'society'],
        },
        'pts': {
            'name': '公視',
            'module': 'scripts.crawlers.pts_spider',
            'class': 'PTSNewsSpider',
            'playwright': False,
            'categories': ['politics', 'finance', 'society'],
        },
        'udn': {
            'name': '聯合報',
            'module': 'scripts.crawlers.udn_spider',
            'class': 'UDNNewsSpider',
            'playwright': True,
            'categories': ['politics', 'finance', 'society'],
        },
        'apple': {
            'name': '蘋果日報',
            'module': 'scripts.crawlers.apple_daily_spider',
            'class': 'AppleDailySpider',
            'playwright': True,
            'categories': ['politics', 'economy', 'society'],
        },
        'tvbs': {
            'name': 'TVBS新聞',
            'module': 'scripts.crawlers.tvbs_spider',
            'class': 'TVBSNewsSpider',
            'playwright': True,
            'categories': ['politics', 'money', 'local'],
            'skip': True,  # Skip due to timeout issues
        },
        'chinatimes': {
            'name': '中時新聞網',
            'module': 'scripts.crawlers.chinatimes_spider',
            'class': 'ChinaTimesSpider',
            'playwright': False,
            'categories': ['politic', 'money', 'society'],
        },
        'ettoday': {
            'name': '東森新聞雲',
            'module': 'scripts.crawlers.ettoday_spider',
            'class': 'ETtodaySpider',
            'playwright': True,
            'categories': ['politics', 'society', 'finance'],
        },
        'storm': {
            'name': '風傳媒',
            'module': 'scripts.crawlers.storm_spider',
            'class': 'StormMediaSpider',
            'playwright': True,
            'categories': ['politics', 'finance', 'lifestyle'],
        },
    }


@pytest.fixture
def valid_article_schema():
    """
    Expected schema for article output.

    Returns:
        dict: Required fields and their types
    """
    return {
        'article_id': str,
        'title': str,
        'content': str,
        'url': str,
        'source': str,
        'source_name': str,
        'published_date': str,
        'author': str,
        'category': str,
        'crawled_at': str,
    }


# ========== Utility Fixtures ==========

@pytest.fixture
def sample_article():
    """
    Sample article data for testing.

    Returns:
        dict: Sample article
    """
    return {
        'article_id': 'test123456',
        'title': '測試新聞標題',
        'content': '這是測試新聞內容。' * 50,  # Min 100 chars
        'url': 'https://news.example.com/test/123',
        'source': 'TestSource',
        'source_name': '測試來源',
        'published_date': '2025-11-18',
        'author': '測試記者',
        'category': 'politics',
        'category_name': '政治',
        'tags': ['測試', '新聞'],
        'image_url': 'https://example.com/image.jpg',
        'crawled_at': datetime.now().isoformat(),
    }


@pytest.fixture(scope="session")
def reactor_installed():
    """
    Ensure asyncio reactor is installed for Playwright tests.

    This should be called before any Playwright spider tests.
    """
    try:
        from twisted.internet import asyncioreactor
        asyncioreactor.install()
        return True
    except Exception as e:
        # Reactor already installed
        return True


# ========== Markers ==========

def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "playwright: marks tests that require Playwright"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks unit tests"
    )
