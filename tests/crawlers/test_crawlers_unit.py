#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for news crawlers.

Tests crawler initialization, configuration, and utility methods.

Author: Information Retrieval System
Date: 2025-11-18
"""

import pytest
import sys
from pathlib import Path
from datetime import datetime
import importlib

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestCrawlerInitialization:
    """Test crawler initialization and configuration."""

    @pytest.mark.unit
    @pytest.mark.parametrize("crawler_name", [
        'cna', 'ltn', 'pts', 'chinatimes', 'ettoday'
    ])
    def test_crawler_init_default(self, crawler_name, crawler_registry):
        """Test crawler can be initialized with default parameters."""
        config = crawler_registry[crawler_name]

        # Skip if marked
        if config.get('skip'):
            pytest.skip(f"Crawler {crawler_name} marked as skip")

        # Import spider class
        module = importlib.import_module(config['module'])
        spider_class = getattr(module, config['class'])

        # Initialize spider
        spider = spider_class()

        # Basic assertions
        assert spider is not None
        assert hasattr(spider, 'name')
        assert hasattr(spider, 'start_requests')
        assert hasattr(spider, 'parse')

    @pytest.mark.unit
    @pytest.mark.parametrize("crawler_name", ['chinatimes', 'ettoday'])
    def test_crawler_init_with_params(self, crawler_name, crawler_registry):
        """Test crawler initialization with custom parameters."""
        config = crawler_registry[crawler_name]

        # Import spider class
        module = importlib.import_module(config['module'])
        spider_class = getattr(module, config['class'])

        # Initialize with parameters
        spider = spider_class(
            category='politics',
            days=3,
        )

        assert spider.category == 'politics'
        assert hasattr(spider, 'start_date')
        assert hasattr(spider, 'end_date')

    @pytest.mark.unit
    def test_crawler_date_range(self, crawler_registry, date_range):
        """Test crawler date range configuration."""
        config = crawler_registry['chinatimes']

        module = importlib.import_module(config['module'])
        spider_class = getattr(module, config['class'])

        start_date, end_date = date_range

        spider = spider_class(
            start_date=start_date,
            end_date=end_date
        )

        assert spider.start_date is not None
        assert spider.end_date is not None
        assert spider.start_date <= spider.end_date


class TestCrawlerUtilities:
    """Test crawler utility methods."""

    @pytest.mark.unit
    def test_article_id_generation(self, crawler_registry):
        """Test article ID generation is consistent."""
        config = crawler_registry['chinatimes']

        module = importlib.import_module(config['module'])
        spider_class = getattr(module, config['class'])

        spider = spider_class()

        # Same URL should generate same ID
        url1 = "https://example.com/news/123"
        url2 = "https://example.com/news/123"
        url3 = "https://example.com/news/456"

        id1 = spider._generate_article_id(url1)
        id2 = spider._generate_article_id(url2)
        id3 = spider._generate_article_id(url3)

        assert id1 == id2
        assert id1 != id3
        assert len(id1) == 16  # MD5 hash truncated to 16 chars

    @pytest.mark.unit
    def test_text_cleaning(self, crawler_registry):
        """Test text cleaning utility."""
        config = crawler_registry['chinatimes']

        module = importlib.import_module(config['module'])
        spider_class = getattr(module, config['class'])

        spider = spider_class()

        # Test HTML tag removal
        dirty_text = "<p>Test   content</p> <br/> extra  spaces"
        clean_text = spider._clean_text(dirty_text)

        assert '<p>' not in clean_text
        assert '<br/>' not in clean_text
        assert '  ' not in clean_text  # Multiple spaces removed
        assert clean_text == "Test content extra spaces"

    @pytest.mark.unit
    @pytest.mark.parametrize("date_text,expected", [
        ("2025-11-18", "2025-11-18"),
        ("2025/11/18", "2025-11-18"),
        ("2025年11月18日", "2025-11-18"),
    ])
    def test_date_parsing(self, date_text, expected, crawler_registry):
        """Test date parsing from various formats."""
        config = crawler_registry['chinatimes']

        module = importlib.import_module(config['module'])
        spider_class = getattr(module, config['class'])

        spider = spider_class()

        parsed_date = spider._parse_publish_date(date_text)
        assert parsed_date == expected


class TestArticleValidation:
    """Test article data validation."""

    @pytest.mark.unit
    def test_valid_article_structure(self, sample_article, valid_article_schema):
        """Test article has all required fields."""
        for field, field_type in valid_article_schema.items():
            assert field in sample_article, f"Missing field: {field}"
            assert isinstance(sample_article[field], field_type), \
                f"Field {field} has wrong type"

    @pytest.mark.unit
    def test_article_content_length(self, sample_article):
        """Test article content meets minimum length requirement."""
        assert len(sample_article['content']) >= 100, \
            "Article content should be at least 100 characters"

    @pytest.mark.unit
    def test_article_url_format(self, sample_article):
        """Test article URL is valid."""
        url = sample_article['url']
        assert url.startswith('http://') or url.startswith('https://'), \
            "Article URL should start with http:// or https://"

    @pytest.mark.unit
    def test_article_date_format(self, sample_article):
        """Test article date is in correct format."""
        date_str = sample_article['published_date']

        # Should be YYYY-MM-DD format
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
            valid_date = True
        except ValueError:
            valid_date = False

        assert valid_date, "Published date should be in YYYY-MM-DD format"


class TestCrawlerConfiguration:
    """Test crawler configuration settings."""

    @pytest.mark.unit
    @pytest.mark.parametrize("crawler_name", ['chinatimes', 'ettoday'])
    def test_crawler_has_custom_settings(self, crawler_name, crawler_registry):
        """Test crawler has custom settings defined."""
        config = crawler_registry[crawler_name]

        module = importlib.import_module(config['module'])
        spider_class = getattr(module, config['class'])

        assert hasattr(spider_class, 'custom_settings')
        assert isinstance(spider_class.custom_settings, dict)
        assert 'DOWNLOAD_DELAY' in spider_class.custom_settings

    @pytest.mark.unit
    def test_crawler_respects_robots_txt(self, crawler_registry):
        """Test crawlers respect robots.txt."""
        for crawler_name, config in crawler_registry.items():
            if config.get('skip'):
                continue

            module = importlib.import_module(config['module'])
            spider_class = getattr(module, config['class'])

            settings = getattr(spider_class, 'custom_settings', {})
            robotstxt_obey = settings.get('ROBOTSTXT_OBEY', True)

            assert robotstxt_obey is True, \
                f"Crawler {crawler_name} should obey robots.txt"

    @pytest.mark.unit
    @pytest.mark.playwright
    def test_playwright_crawler_settings(self, crawler_registry):
        """Test Playwright crawlers have correct settings."""
        playwright_crawlers = [
            name for name, config in crawler_registry.items()
            if config.get('playwright') and not config.get('skip')
        ]

        for crawler_name in playwright_crawlers:
            config = crawler_registry[crawler_name]

            module = importlib.import_module(config['module'])
            spider_class = getattr(module, config['class'])

            settings = spider_class.custom_settings

            # Check Playwright settings
            assert 'DOWNLOAD_HANDLERS' in settings
            assert 'PLAYWRIGHT_BROWSER_TYPE' in settings

            # Check browser type
            assert settings['PLAYWRIGHT_BROWSER_TYPE'] == 'chromium'


# ========== Run Tests ==========

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
