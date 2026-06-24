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
try:
    from fake_useragent import UserAgent
except ImportError:  # pragma: no cover - optional dependency
    UserAgent = None

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
        playwright_page_goto_kwargs (dict): Keyword args for Playwright `page.goto`
    """

    # Enable Playwright by default
    use_playwright = True

    # User-Agent rotator
    ua = UserAgent() if UserAgent else None

    # Fallback UA pool (used when `fake_useragent` is not available).
    FALLBACK_USER_AGENTS = [
        # Chrome (Windows)
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        # Chrome (macOS)
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        # Safari (macOS)
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15",
    ]

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
        if self.ua:
            self.current_user_agent = self.ua.random
        else:
            self.current_user_agent = random.choice(self.FALLBACK_USER_AGENTS)

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
            # Extra headers help some WAF/CDN setups treat traffic as a real browser.
            # Keep these conservative; per-spider settings can override via meta.
            'extra_http_headers': {
                'Accept-Language': 'zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
                'Upgrade-Insecure-Requests': '1',
            },
        }

        # Configure Playwright page.goto kwargs.
        #
        # NOTE:
        # `scrapy-playwright` expects this under `playwright_page_goto_kwargs`.
        # Using `networkidle` is brittle on ad-heavy news sites; prefer a faster
        # load-state and wait for specific selectors via PageMethod when needed.
        self.playwright_page_goto_kwargs = {
            'wait_until': 'domcontentloaded',
        }

        # Backward-compatible alias (older code may reference this attribute).
        # `scrapy-playwright` does NOT read this key from request meta.
        self.playwright_page_kwargs = self.playwright_page_goto_kwargs

    def get_playwright_meta(self, **kwargs) -> Dict[str, Any]:
        """
        Get Playwright metadata for request.

        This method should be called to get meta dict for Playwright requests.
        It includes fingerprint settings and can be extended with custom settings.

        Args:
            **kwargs: Additional Playwright settings to override defaults

        Supported convenience kwargs:
            wait_selector (str): If provided, prepends a `wait_for_selector` PageMethod.
            wait_selector_timeout (int): Timeout for `wait_for_selector` in ms (default: 30000).
            wait_until (str): Override `page.goto(wait_until=...)`.
            include_page (bool): Set `playwright_include_page` for access to `response.meta['playwright_page']`.

        Returns:
            dict: Meta dictionary with Playwright settings.

        Complexity:
            Time: O(m) where m is number of meta keys merged
            Space: O(m)

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
        wait_selector = kwargs.pop('wait_selector', None)
        wait_selector_timeout = int(kwargs.pop('wait_selector_timeout', 30000))
        wait_until = kwargs.pop('wait_until', None)
        include_page = kwargs.pop('include_page', None)

        goto_kwargs = self.playwright_page_goto_kwargs.copy()
        if wait_until:
            goto_kwargs['wait_until'] = wait_until

        meta: Dict[str, Any] = {
            'playwright': True,
            'playwright_context_kwargs': self.playwright_context_kwargs.copy(),
            'playwright_page_goto_kwargs': goto_kwargs,
        }

        if include_page is not None:
            meta['playwright_include_page'] = bool(include_page)

        # Merge with custom kwargs
        if kwargs:
            for key, value in kwargs.items():
                # Backward-compat: treat `playwright_page_kwargs` as `playwright_page_goto_kwargs`.
                if key == 'playwright_page_kwargs':
                    key = 'playwright_page_goto_kwargs'

                if key in meta:
                    if isinstance(meta[key], dict):
                        meta[key].update(value)
                    elif isinstance(meta[key], list) and isinstance(value, list):
                        meta[key].extend(value)
                    else:
                        meta[key] = value
                else:
                    meta[key] = value

        if wait_selector:
            try:
                from scrapy_playwright.page import PageMethod
                page_methods = meta.get('playwright_page_methods', [])
                if not isinstance(page_methods, list):
                    page_methods = [page_methods]
                page_methods.insert(0, PageMethod('wait_for_selector', wait_selector, timeout=wait_selector_timeout))
                meta['playwright_page_methods'] = page_methods
            except Exception:
                # If scrapy-playwright isn't available, just return meta without the wait.
                pass

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
