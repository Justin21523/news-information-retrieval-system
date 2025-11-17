#!/usr/bin/env python
"""
Base Playwright Spider with Anti-Detection

This module provides a base class for Playwright-enabled Scrapy spiders
with comprehensive anti-detection mechanisms including:
- Browser fingerprint randomization
- User-Agent rotation
- Human behavior simulation
- Stealth mode to hide automation markers

Usage:
    Inherit from BasePlaywrightSpider instead of scrapy.Spider

Author: CNIRS Development Team
License: Educational Use Only
"""

import scrapy
import random
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from fake_useragent import UserAgent

logger = logging.getLogger(__name__)


class BasePlaywrightSpider(scrapy.Spider):
    """
    Base spider class with Playwright support and anti-detection features.

    Features:
        - Automatic Playwright integration
        - Browser fingerprint randomization
        - User-Agent rotation
        - Viewport randomization
        - Timezone and locale settings
        - Human-like delays

    Attributes:
        use_playwright (bool): Enable Playwright rendering
        playwright_context_kwargs (dict): Context settings for browser fingerprinting
        playwright_page_kwargs (dict): Page settings
    """

    # Enable Playwright by default
    use_playwright = True

    # User-Agent rotator
    ua = UserAgent()

    # Common screen resolutions for viewport randomization
    VIEWPORTS = [
        {'width': 1920, 'height': 1080},  # Full HD
        {'width': 1366, 'height': 768},   # Common laptop
        {'width': 1536, 'height': 864},   # HD+
        {'width': 1440, 'height': 900},   # WXGA+
        {'width': 1680, 'height': 1050},  # WSXGA+
        {'width': 2560, 'height': 1440},  # 2K
    ]

    def __init__(self, *args, **kwargs):
        """Initialize base spider with anti-detection settings."""
        super().__init__(*args, **kwargs)

        # Generate random fingerprint for this session
        self._init_fingerprint()

        logger.info(f"Initialized {self.name} with fingerprint:")
        logger.info(f"  User-Agent: {self.current_user_agent[:50]}...")
        logger.info(f"  Viewport: {self.current_viewport}")
        logger.info(f"  Timezone: {self.timezone}")

    def _init_fingerprint(self):
        """Initialize randomized browser fingerprint."""
        # Random User-Agent
        self.current_user_agent = self.ua.random

        # Random viewport
        self.current_viewport = random.choice(self.VIEWPORTS)

        # Set timezone (Taiwan)
        self.timezone = 'Asia/Taipei'

        # Set locale (Traditional Chinese)
        self.locale = 'zh-TW'

        # Random device scale factor
        self.device_scale_factor = random.choice([1, 1.5, 2])

        # Configure Playwright context (browser-level settings)
        self.playwright_context_kwargs = {
            'viewport': self.current_viewport,
            'user_agent': self.current_user_agent,
            'locale': self.locale,
            'timezone_id': self.timezone,
            'device_scale_factor': self.device_scale_factor,
            'has_touch': random.choice([True, False]),
            'is_mobile': False,
            'java_script_enabled': True,
            'ignore_https_errors': True,
            'bypass_csp': True,  # Bypass Content Security Policy
        }

        # Configure Playwright page settings
        self.playwright_page_kwargs = {
            'wait_until': 'networkidle',  # Wait until network is idle
        }

    def get_playwright_meta(self, **kwargs) -> Dict[str, Any]:
        """
        Get Playwright metadata for request.

        This method should be called to get meta dict for Playwright requests.
        It includes fingerprint settings and can be extended with custom settings.

        Args:
            **kwargs: Additional Playwright settings to override defaults

        Returns:
            dict: Meta dictionary with Playwright settings

        Example:
            >>> yield scrapy.Request(
            ...     url='https://example.com',
            ...     callback=self.parse,
            ...     meta=self.get_playwright_meta(
            ...         playwright_page_methods=[
            ...             PageMethod('wait_for_selector', 'div.content')
            ...         ]
            ...     )
            ... )
        """
        meta = {
            'playwright': True,
            'playwright_context_kwargs': self.playwright_context_kwargs.copy(),
            'playwright_page_kwargs': self.playwright_page_kwargs.copy(),
        }

        # Merge with custom kwargs
        if kwargs:
            for key, value in kwargs.items():
                if key in meta:
                    if isinstance(meta[key], dict):
                        meta[key].update(value)
                    else:
                        meta[key] = value
                else:
                    meta[key] = value

        return meta

    def human_delay(self, min_seconds: float = 1.0, max_seconds: float = 3.0) -> float:
        """
        Generate human-like random delay.

        Args:
            min_seconds: Minimum delay in seconds
            max_seconds: Maximum delay in seconds

        Returns:
            float: Random delay value
        """
        # Add some randomness with normal distribution
        mean = (min_seconds + max_seconds) / 2
        std = (max_seconds - min_seconds) / 4
        delay = random.gauss(mean, std)

        # Clamp to min/max
        delay = max(min_seconds, min(max_seconds, delay))

        return delay

    def parse_date_from_text(self, date_text: str) -> Optional[str]:
        """
        Parse date from various Chinese date formats.

        Args:
            date_text: Date string in Chinese format

        Returns:
            str: ISO format date (YYYY-MM-DD) or None if parsing fails

        Examples:
            >>> self.parse_date_from_text("2024年1月15日")
            '2024-01-15'
            >>> self.parse_date_from_text("2024/01/15")
            '2024-01-15'
        """
        import re

        # Try different date patterns
        patterns = [
            # 2024年1月15日
            r'(\d{4})年(\d{1,2})月(\d{1,2})日',
            # 2024/01/15 or 2024-01-15
            r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})',
            # 民國年份: 113年1月15日
            r'(\d{2,3})年(\d{1,2})月(\d{1,2})日',
        ]

        for pattern in patterns:
            match = re.search(pattern, date_text)
            if match:
                year, month, day = match.groups()

                # Convert ROC year to AD if needed (ROC year < 200)
                year = int(year)
                if year < 200:
                    year += 1911  # ROC to AD conversion

                try:
                    date_obj = datetime(year, int(month), int(day))
                    return date_obj.strftime('%Y-%m-%d')
                except ValueError:
                    continue

        # Try parsing with dateutil if available
        try:
            from dateutil import parser
            date_obj = parser.parse(date_text, fuzzy=True)
            return date_obj.strftime('%Y-%m-%d')
        except (ImportError, ValueError, TypeError):
            pass

        logger.warning(f"Failed to parse date: {date_text}")
        return None

    def extract_text(self, selector, css: str = None, xpath: str = None,
                    default: str = '') -> str:
        """
        Safely extract and clean text from selector.

        Args:
            selector: Scrapy selector object
            css: CSS selector string
            xpath: XPath selector string
            default: Default value if extraction fails

        Returns:
            str: Extracted and cleaned text
        """
        try:
            if css:
                text = selector.css(css).get()
            elif xpath:
                text = selector.xpath(xpath).get()
            else:
                text = selector.get()

            if text:
                # Clean whitespace
                text = ' '.join(text.split())
                return text.strip()
            return default
        except Exception as e:
            logger.warning(f"Failed to extract text: {e}")
            return default

    def closed(self, reason):
        """Log spider closure statistics."""
        logger.info("=" * 70)
        logger.info(f"Spider {self.name} closed: {reason}")
        logger.info("=" * 70)


class PlaywrightPageMethods:
    """
    Common Playwright page methods for different scenarios.

    Usage:
        from scrapy_playwright.page import PageMethod

        meta = self.get_playwright_meta(
            playwright_page_methods=[
                PageMethod('wait_for_selector', 'div.content'),
                PageMethod('evaluate', '() => window.scrollTo(0, document.body.scrollHeight)'),
            ]
        )
    """

    @staticmethod
    def wait_for_selector(selector: str, timeout: int = 30000):
        """Wait for element to appear."""
        from scrapy_playwright.page import PageMethod
        return PageMethod('wait_for_selector', selector, timeout=timeout)

    @staticmethod
    def scroll_to_bottom():
        """Scroll to page bottom."""
        from scrapy_playwright.page import PageMethod
        return PageMethod(
            'evaluate',
            '() => window.scrollTo(0, document.body.scrollHeight)'
        )

    @staticmethod
    def random_scroll():
        """Scroll to random position."""
        from scrapy_playwright.page import PageMethod
        scroll_percent = random.randint(30, 70)
        return PageMethod(
            'evaluate',
            f'() => window.scrollTo(0, document.body.scrollHeight * {scroll_percent / 100})'
        )

    @staticmethod
    def wait_for_timeout(milliseconds: int):
        """Wait for specific time."""
        from scrapy_playwright.page import PageMethod
        return PageMethod('wait_for_timeout', milliseconds)

    @staticmethod
    def click_element(selector: str):
        """Click on element."""
        from scrapy_playwright.page import PageMethod
        return PageMethod('click', selector)

    @staticmethod
    def screenshot(path: str = None, full_page: bool = True):
        """Take screenshot."""
        from scrapy_playwright.page import PageMethod
        kwargs = {'full_page': full_page}
        if path:
            kwargs['path'] = path
        return PageMethod('screenshot', **kwargs)
