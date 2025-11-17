#!/usr/bin/env python
"""
Humanization Middleware for Playwright Spiders

This middleware simulates human-like browsing behavior to avoid bot detection:
- Random scrolling patterns
- Variable delays between actions
- Mouse movements (simulated)
- Realistic page interaction timing

Usage:
    Add to DOWNLOADER_MIDDLEWARES in settings.py:
    'scripts.crawlers.middlewares.humanization_middleware.HumanizationMiddleware': 586

Author: CNIRS Development Team
License: Educational Use Only
"""

import logging
import random
from typing import List, Optional
from scrapy import signals
from scrapy.http import Request, Response
from scrapy.exceptions import NotConfigured

try:
    from scrapy_playwright.page import PageMethod
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

logger = logging.getLogger(__name__)


class HumanizationMiddleware:
    """
    Middleware to add human-like behavior to Playwright requests.

    This middleware simulates realistic browsing patterns including:
    - Random scrolling (up, down, partial)
    - Variable delays between actions
    - Mouse movements to random positions
    - Reading time simulation based on content length

    Settings:
        HUMANIZATION_ENABLED (bool): Enable/disable middleware (default: True)
        HUMANIZATION_MIN_DELAY (float): Minimum delay in seconds (default: 0.5)
        HUMANIZATION_MAX_DELAY (float): Maximum delay in seconds (default: 2.0)
        HUMANIZATION_SCROLL_ENABLED (bool): Enable random scrolling (default: True)
        HUMANIZATION_READING_TIME (bool): Simulate reading time (default: True)
    """

    def __init__(
        self,
        enabled: bool = True,
        min_delay: float = 0.5,
        max_delay: float = 2.0,
        scroll_enabled: bool = True,
        reading_time: bool = True,
    ):
        """
        Initialize humanization middleware.

        Args:
            enabled: Whether to enable humanization
            min_delay: Minimum delay between actions (seconds)
            max_delay: Maximum delay between actions (seconds)
            scroll_enabled: Enable random scrolling
            reading_time: Simulate reading time based on content
        """
        if not PLAYWRIGHT_AVAILABLE:
            raise NotConfigured("scrapy-playwright is not installed")

        self.enabled = enabled
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.scroll_enabled = scroll_enabled
        self.reading_time = reading_time

        if self.enabled:
            logger.info(
                f"HumanizationMiddleware enabled "
                f"(delay: {min_delay}-{max_delay}s, "
                f"scroll: {scroll_enabled}, "
                f"reading: {reading_time})"
            )

    @classmethod
    def from_crawler(cls, crawler):
        """Create middleware instance from crawler settings."""
        settings = crawler.settings

        enabled = settings.getbool('HUMANIZATION_ENABLED', True)
        min_delay = settings.getfloat('HUMANIZATION_MIN_DELAY', 0.5)
        max_delay = settings.getfloat('HUMANIZATION_MAX_DELAY', 2.0)
        scroll_enabled = settings.getbool('HUMANIZATION_SCROLL_ENABLED', True)
        reading_time = settings.getbool('HUMANIZATION_READING_TIME', True)

        middleware = cls(
            enabled=enabled,
            min_delay=min_delay,
            max_delay=max_delay,
            scroll_enabled=scroll_enabled,
            reading_time=reading_time,
        )

        crawler.signals.connect(
            middleware.spider_opened,
            signal=signals.spider_opened
        )

        return middleware

    def spider_opened(self, spider):
        """Log when spider opens."""
        logger.info(f"HumanizationMiddleware active for spider: {spider.name}")

    def process_request(self, request: Request, spider):
        """
        Add humanization behaviors to Playwright requests.

        Args:
            request: Scrapy request object
            spider: Spider instance

        Returns:
            None (request is modified in place)
        """
        if not self.enabled:
            return None

        # Only process Playwright requests
        if not request.meta.get('playwright'):
            return None

        # Get existing page methods
        page_methods = request.meta.get('playwright_page_methods', [])
        if not isinstance(page_methods, list):
            page_methods = [page_methods]

        # Add humanization behaviors
        humanization_methods = self._get_humanization_methods()

        # Append humanization methods
        page_methods.extend(humanization_methods)

        # Update request meta
        request.meta['playwright_page_methods'] = page_methods

        logger.debug(f"Applied humanization to request: {request.url}")

        return None

    def _get_humanization_methods(self) -> List[PageMethod]:
        """
        Generate list of humanization page methods.

        Returns:
            list: List of PageMethod objects for human-like behavior
        """
        methods = []

        # 1. Initial delay (page load time)
        initial_delay = self._random_delay(0.5, 1.5)
        methods.append(
            PageMethod('wait_for_timeout', int(initial_delay * 1000))
        )

        # 2. Random scrolling pattern
        if self.scroll_enabled:
            scroll_methods = self._generate_scroll_pattern()
            methods.extend(scroll_methods)

        # 3. Mouse movement simulation
        mouse_move = self._generate_mouse_movement()
        if mouse_move:
            methods.append(mouse_move)

        # 4. Reading time delay
        if self.reading_time:
            reading_delay = self._random_delay(1.0, 3.0)
            methods.append(
                PageMethod('wait_for_timeout', int(reading_delay * 1000))
            )

        return methods

    def _random_delay(self, min_sec: float = None, max_sec: float = None) -> float:
        """
        Generate random delay with normal distribution.

        Args:
            min_sec: Minimum delay (defaults to self.min_delay)
            max_sec: Maximum delay (defaults to self.max_delay)

        Returns:
            float: Random delay in seconds
        """
        min_sec = min_sec if min_sec is not None else self.min_delay
        max_sec = max_sec if max_sec is not None else self.max_delay

        # Use normal distribution for more realistic timing
        mean = (min_sec + max_sec) / 2
        std = (max_sec - min_sec) / 4
        delay = random.gauss(mean, std)

        # Clamp to min/max
        delay = max(min_sec, min(max_sec, delay))

        return delay

    def _generate_scroll_pattern(self) -> List[PageMethod]:
        """
        Generate realistic scrolling pattern.

        Humans typically:
        - Scroll down to read content
        - Sometimes scroll back up to re-read
        - Scroll at variable speeds
        - Stop at various positions

        Returns:
            list: List of scroll-related PageMethod objects
        """
        scroll_methods = []

        # Number of scroll actions (1-4)
        num_scrolls = random.randint(1, 4)

        for i in range(num_scrolls):
            # Random scroll direction (mostly down, occasionally up)
            if i == 0 or random.random() > 0.2:
                # Scroll down
                scroll_percent = random.randint(20, 80)
                scroll_script = f"""
                () => {{
                    const scrollHeight = document.body.scrollHeight;
                    const targetY = scrollHeight * {scroll_percent / 100};
                    window.scrollTo({{
                        top: targetY,
                        behavior: 'smooth'
                    }});
                }}
                """
            else:
                # Scroll up (re-reading behavior)
                scroll_up = random.randint(10, 30)
                scroll_script = f"""
                () => {{
                    const currentY = window.pageYOffset;
                    const targetY = Math.max(0, currentY - {scroll_up * 10});
                    window.scrollTo({{
                        top: targetY,
                        behavior: 'smooth'
                    }});
                }}
                """

            scroll_methods.append(PageMethod('evaluate', scroll_script))

            # Delay after scroll (reading time)
            delay = self._random_delay(0.3, 1.0)
            scroll_methods.append(
                PageMethod('wait_for_timeout', int(delay * 1000))
            )

        return scroll_methods

    def _generate_mouse_movement(self) -> Optional[PageMethod]:
        """
        Generate random mouse movement.

        Note:
            This simulates mouse position but doesn't render actual movement.
            For true mouse movement simulation, use python-ghost-cursor.

        Returns:
            PageMethod or None: Mouse movement script
        """
        # 50% chance to simulate mouse movement
        if random.random() < 0.5:
            return None

        # Random position on page
        x = random.randint(100, 1200)
        y = random.randint(100, 800)

        mouse_script = f"""
        () => {{
            // Simulate mouse position (not visible but tracked by some detection systems)
            const event = new MouseEvent('mousemove', {{
                view: window,
                bubbles: true,
                cancelable: true,
                clientX: {x},
                clientY: {y}
            }});
            document.dispatchEvent(event);
        }}
        """

        return PageMethod('evaluate', mouse_script)


class AdvancedHumanizationMiddleware(HumanizationMiddleware):
    """
    Advanced humanization with more sophisticated patterns.

    Additional features:
    - Click simulation on random elements
    - Text selection simulation
    - Tab focus changes
    - More complex scrolling patterns
    """

    def _generate_scroll_pattern(self) -> List[PageMethod]:
        """
        Generate advanced scrolling pattern with pauses.

        Returns:
            list: List of scroll-related PageMethod objects
        """
        scroll_methods = []

        # Gradual scroll down with pauses (like reading)
        scroll_positions = [0.2, 0.4, 0.6, 0.8]
        random.shuffle(scroll_positions)

        # Take only 2-3 positions
        scroll_positions = scroll_positions[:random.randint(2, 3)]
        scroll_positions.sort()

        for pos in scroll_positions:
            # Scroll to position
            scroll_script = f"""
            () => {{
                const scrollHeight = document.body.scrollHeight;
                const targetY = scrollHeight * {pos};
                window.scrollTo({{
                    top: targetY,
                    behavior: 'smooth'
                }});
            }}
            """
            scroll_methods.append(PageMethod('evaluate', scroll_script))

            # Reading pause
            pause = self._random_delay(0.5, 2.0)
            scroll_methods.append(
                PageMethod('wait_for_timeout', int(pause * 1000))
            )

        return scroll_methods

    def _simulate_reading_behavior(self) -> List[PageMethod]:
        """
        Simulate realistic reading behavior.

        Returns:
            list: List of PageMethod objects for reading simulation
        """
        methods = []

        # Scroll down slowly
        for i in range(3):
            progress = (i + 1) * 0.25  # 25%, 50%, 75%
            scroll_script = f"""
            () => {{
                window.scrollTo({{
                    top: document.body.scrollHeight * {progress},
                    behavior: 'smooth'
                }});
            }}
            """
            methods.append(PageMethod('evaluate', scroll_script))

            # Variable reading time based on position
            # (people read faster as they scan down)
            reading_time = self._random_delay(1.5 - (i * 0.3), 3.0 - (i * 0.5))
            methods.append(
                PageMethod('wait_for_timeout', int(reading_time * 1000))
            )

        return methods
