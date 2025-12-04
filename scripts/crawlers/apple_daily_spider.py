#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Apple Daily News (蘋果日報) Spider

Spider for crawling news articles from Apple Daily Taiwan.
Inherits from BasePlaywrightSpider for comprehensive anti-detection.

NOTE: Apple Daily Taiwan ceased operations in May 2021. This spider is designed
for historical data or alternative Apple Daily news sources.

Target: https://tw.appledaily.com/ (archived) or https://www.nextapple.com.tw/
Categories: Politics (政治), Economics (財經), Society (社會), Entertainment (娛樂),
            Sports (體育), Life (生活), International (國際)

Features:
    - Playwright-based browser automation
    - Anti-detection fingerprinting
    - Date range crawling
    - Category-based filtering
    - Human-like behavior simulation

Usage:
    # Single category, 7 days
    scrapy runspider apple_daily_spider.py -a days=7 -a category=politics -o apple_politics.jsonl

    # All categories, custom date range
    scrapy runspider apple_daily_spider.py -a start_date=2025-11-01 -a end_date=2025-11-13 -o apple.jsonl

    # With custom settings
    scrapy runspider apple_daily_spider.py -a days=3 -s CONCURRENT_REQUESTS=1 -s DOWNLOAD_DELAY=3

Author: Information Retrieval System
Date: 2025-11-18
"""

"""
IMPORTANT NOTE ON PLAYWRIGHT REACTOR:
This spider requires asyncio reactor for Playwright.

When running standalone, set environment variable:
    export SCRAPY_REACTOR=twisted.internet.asyncioreactor.AsyncioSelectorReactor

Or ensure reactor is installed before importing this module:
    from twisted.internet import asyncioreactor
    asyncioreactor.install()
    from apple_daily_spider import AppleDailySpider
"""

import scrapy
from datetime import datetime, timedelta
import json
import logging
import re
import hashlib
from typing import Optional, List, Dict, Any
from urllib.parse import urljoin, urlparse

# Import base spider with anti-detection features
from base_playwright_spider import BasePlaywrightSpider, PlaywrightPageMethods
from scrapy_playwright.page import PageMethod

logger = logging.getLogger(__name__)


class AppleDailySpider(BasePlaywrightSpider):
    """
    Scrapy spider for Apple Daily (蘋果日報) with Playwright anti-detection.

    URL Structure (Next Apple):
        - Homepage: https://www.nextapple.com.tw/
        - Category list: https://www.nextapple.com.tw/category/{category}
        - Article page: https://www.nextapple.com.tw/article/{article_id}

    Categories:
        politics: 政治
        economy: 財經
        society: 社會
        entertainment: 娛樂
        sports: 體育
        life: 生活
        world: 國際

    Complexity:
        Time: O(D * A) where D = days, A = avg articles per day
        Space: O(A) for storing article data
    """

    name = 'apple_daily'
    allowed_domains = ['nextapple.com.tw', 'tw.appledaily.com']

    # Category mapping
    CATEGORIES = {
        'all': {'slug': '', 'name': '全部'},
        'politics': {'slug': 'politics', 'name': '政治'},
        'economy': {'slug': 'economy', 'name': '財經'},
        'society': {'slug': 'society', 'name': '社會'},
        'entertainment': {'slug': 'entertainment', 'name': '娛樂'},
        'sports': {'slug': 'sports', 'name': '體育'},
        'life': {'slug': 'life', 'name': '生活'},
        'world': {'slug': 'world', 'name': '國際'},
    }

    # Custom settings optimized for anti-detection
    custom_settings = {
        'DOWNLOAD_DELAY': 3,  # 3 seconds between requests (respectful)
        'CONCURRENT_REQUESTS_PER_DOMAIN': 1,  # One at a time
        'ROBOTSTXT_OBEY': True,
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
            'data/raw/apple_news_%(time)s.jsonl': {
                'format': 'jsonlines',
                'encoding': 'utf8',
                'store_empty': False,
                'overwrite': False,
                'indent': None,
            }
        },

        'LOG_LEVEL': 'INFO',
    }

    def __init__(self,
                 category: str = 'all',
                 days: int = 7,
                 start_date: str = None,
                 end_date: str = None,
                 *args, **kwargs):
        """
        Initialize Apple Daily spider with date range and category.

        Args:
            category: News category
            days: Number of days to crawl (from today backwards)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        """
        super().__init__(*args, **kwargs)

        # Validate and set category
        if category not in self.CATEGORIES:
            logger.warning(f"Invalid category '{category}', using 'all'")
            category = 'all'
        self.category = category
        self.category_slug = self.CATEGORIES[category]['slug']
        self.category_name = self.CATEGORIES[category]['name']

        # Set date range
        if start_date and end_date:
            try:
                self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
                self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
            except ValueError as e:
                logger.error(f"Invalid date format: {e}")
                self.end_date = datetime.now()
                self.start_date = self.end_date - timedelta(days=int(days))
        else:
            self.end_date = datetime.now()
            self.start_date = self.end_date - timedelta(days=int(days))

        logger.info("=" * 70)
        logger.info(f"Apple Daily Spider Initialized")
        logger.info("=" * 70)
        logger.info(f"Category: {self.category_name} ({self.category})")
        logger.info(f"Date Range: {self.start_date.date()} to {self.end_date.date()}")
        logger.info(f"Days: {(self.end_date - self.start_date).days + 1}")
        logger.info("=" * 70)

        # Statistics
        self.articles_scraped = 0
        self.articles_failed = 0
        self.pages_visited = 0
        self.seen_urls = set()

    def start_requests(self):
        """
        Generate start requests for Apple Daily news.

        Yields:
            scrapy.Request: Requests with Playwright enabled
        """
        # Build category URL
        if self.category == 'all':
            base_url = "https://www.nextapple.com.tw/"
        else:
            base_url = f"https://www.nextapple.com.tw/category/{self.category_slug}"

        logger.info(f"Starting with URL: {base_url}")

        yield scrapy.Request(
            url=base_url,
            callback=self.parse_list_page,
            errback=self.handle_error,
            dont_filter=True,
            meta=self.get_playwright_meta(
                playwright_page_methods=[
                    PlaywrightPageMethods.wait_for_selector('article, div.post'),
                    PlaywrightPageMethods.random_scroll(),
                    PlaywrightPageMethods.wait_for_timeout(
                        int(self.human_delay(1500, 3000))
                    ),
                ]
            )
        )

    def parse_list_page(self, response):
        """
        Parse Apple Daily news list page to extract article URLs.

        Args:
            response: Scrapy response object

        Yields:
            scrapy.Request: Requests for article detail pages
            scrapy.Request: Request for next page (pagination)
        """
        self.pages_visited += 1
        logger.info(f"Parsing list page: {response.url}")

        # Extract article links - try multiple selectors for different layouts
        article_selectors = response.css(
            'article a, '
            'div.post a, '
            'h2 a, '
            'div.entry-title a, '
            'a[href*="/article/"]'
        )

        articles_found = 0
        for article_sel in article_selectors:
            article_url = article_sel.css('::attr(href)').get()

            if not article_url:
                continue

            # Convert to absolute URL
            article_url = response.urljoin(article_url)

            # Filter: only news articles
            if not self._is_news_article(article_url):
                continue

            # Skip if already seen
            if article_url in self.seen_urls:
                continue
            self.seen_urls.add(article_url)

            articles_found += 1

            # Request article detail page
            yield scrapy.Request(
                url=article_url,
                callback=self.parse_article,
                errback=self.handle_error,
                meta=self.get_playwright_meta(
                    playwright_page_methods=[
                        PlaywrightPageMethods.wait_for_selector(
                            'article, div.entry-content'
                        ),
                        PlaywrightPageMethods.wait_for_timeout(
                            int(self.human_delay(800, 1500))
                        ),
                    ]
                ),
                dont_filter=False,
            )

        logger.info(f"Found {articles_found} articles on list page")

        # Pagination: check for next page link
        next_page = response.css(
            'a.next, '
            'a[rel="next"], '
            'div.pagination a[href*="page"]::attr(href)'
        ).get()

        if next_page and articles_found > 0 and self.pages_visited < 10:
            next_url = response.urljoin(next_page)
            logger.info(f"Following pagination to: {next_url}")

            yield scrapy.Request(
                url=next_url,
                callback=self.parse_list_page,
                errback=self.handle_error,
                dont_filter=True,
                meta=self.get_playwright_meta(
                    playwright_page_methods=[
                        PlaywrightPageMethods.wait_for_selector('article'),
                        PlaywrightPageMethods.wait_for_timeout(
                            int(self.human_delay(2000, 4000))
                        ),
                    ]
                )
            )

    def parse_article(self, response):
        """
        Parse individual Apple Daily article page.

        Article structure (varies by site version):
            - Title: <h1 class="entry-title"> or <h1>
            - Content: <div class="entry-content"> <p>
            - Date: <time> or <span class="date">
            - Author: <span class="author"> or <div class="author-name">
            - Category: from breadcrumb or URL

        Args:
            response: Scrapy response object

        Yields:
            dict: Article data in standardized format
        """
        try:
            logger.debug(f"Parsing article: {response.url}")

            # Extract date first to filter
            date_text = self._extract_first(response, [
                'time::attr(datetime)',
                'time::text',
                'span.date::text',
                'div.post-date::text',
                'meta[property="article:published_time"]::attr(content)',
            ])

            published_date = self._parse_publish_date(date_text)
            if published_date:
                pub_date_obj = datetime.strptime(published_date, '%Y-%m-%d')
                if not (self.start_date <= pub_date_obj <= self.end_date):
                    logger.debug(f"Article date {published_date} out of range, skipping")
                    return

            # Initialize article data
            article = {
                'article_id': self._generate_article_id(response.url),
                'url': response.url,
                'source': 'Apple Daily',
                'source_name': '蘋果日報',
                'crawled_at': datetime.now().isoformat(),
            }

            # Extract title
            title_selectors = [
                'h1.entry-title::text',
                'h1::text',
                'div.post-title h1::text',
                'meta[property="og:title"]::attr(content)',
            ]
            article['title'] = self._extract_first(response, title_selectors)

            # Extract content paragraphs
            content_selectors = [
                'div.entry-content p::text',
                'article p::text',
                'div.post-content p::text',
                'div.article-content p::text',
            ]

            content_paragraphs = []
            for selector in content_selectors:
                paragraphs = response.css(selector).getall()
                if paragraphs:
                    content_paragraphs = paragraphs
                    break

            article['content'] = ' '.join([self._clean_text(p) for p in content_paragraphs if p])

            # Extract author
            author_selectors = [
                'span.author::text',
                'div.author-name::text',
                'meta[name="author"]::attr(content)',
                'span.byline::text',
            ]
            article['author'] = self._extract_first(response, author_selectors) or '蘋果日報'

            # Extract publish date
            article['published_date'] = published_date

            # Extract category
            breadcrumb = response.css('div.breadcrumb a::text, nav.breadcrumb a::text').getall()
            if breadcrumb and len(breadcrumb) > 1:
                article['category'] = breadcrumb[-1].strip()
                article['category_name'] = breadcrumb[-1].strip()
            else:
                article['category'] = self.category
                article['category_name'] = self.category_name

            # Extract tags
            tag_selectors = response.css(
                'div.tags a::text, '
                'div.post-tags a::text, '
                'a[rel="tag"]::text'
            ).getall()
            article['tags'] = [self._clean_text(tag) for tag in tag_selectors if tag]

            # Extract image URL
            image_selectors = [
                'article img::attr(src)',
                'div.entry-content img::attr(src)',
                'div.featured-image img::attr(src)',
                'meta[property="og:image"]::attr(content)',
            ]
            image_url = self._extract_first(response, image_selectors)
            article['image_url'] = response.urljoin(image_url) if image_url else None

            # Validation
            if not article['title']:
                logger.warning(f"Missing title for {response.url}")
                self.articles_failed += 1
                return

            if not article['content'] or len(article['content']) < 100:
                logger.warning(f"Content too short ({len(article.get('content', ''))} chars) for {response.url}")
                self.articles_failed += 1
                return

            # Success
            self.articles_scraped += 1
            logger.info(f"✓ Article #{self.articles_scraped}: {article['title'][:50]}...")

            yield article

        except Exception as e:
            logger.error(f"Error parsing article {response.url}: {e}", exc_info=True)
            self.articles_failed += 1

    def handle_error(self, failure):
        """Handle request errors with detailed logging."""
        request = failure.request
        logger.error("=" * 70)
        logger.error(f"Request failed: {request.url}")
        logger.error(f"Error type: {failure.type.__name__}")
        logger.error(f"Error value: {failure.value}")
        logger.error("=" * 70)
        self.articles_failed += 1

    def closed(self, reason):
        """Log spider closure statistics."""
        super().closed(reason)

        total_articles = self.articles_scraped + self.articles_failed
        success_rate = (self.articles_scraped / total_articles * 100) if total_articles > 0 else 0

        logger.info("=" * 70)
        logger.info("Apple Daily Spider Statistics")
        logger.info("=" * 70)
        logger.info(f"Pages visited: {self.pages_visited}")
        logger.info(f"Articles scraped: {self.articles_scraped}")
        logger.info(f"Articles failed: {self.articles_failed}")
        logger.info(f"Success rate: {success_rate:.1f}%")
        logger.info(f"Date range: {self.start_date.date()} to {self.end_date.date()}")
        logger.info(f"Category: {self.category_name}")
        logger.info("=" * 70)

    # ========== Utility Methods ==========

    def _is_news_article(self, url: str) -> bool:
        """
        Check if URL is a news article.

        Args:
            url: Article URL

        Returns:
            bool: True if URL is a news article
        """
        if not url:
            return False

        # Next Apple article URLs contain "/article/"
        return '/article/' in url or '/post/' in url

    def _generate_article_id(self, url: str) -> str:
        """Generate unique article ID from URL."""
        return hashlib.md5(url.encode('utf-8')).hexdigest()[:16]

    def _extract_first(self, response, selectors: List[str]) -> Optional[str]:
        """Try multiple selectors and return first non-empty result."""
        for selector in selectors:
            result = response.css(selector).get()
            if result:
                return self._clean_text(result)
        return None

    def _clean_text(self, text: Optional[str]) -> str:
        """Clean and normalize text."""
        if not text:
            return ""

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Strip
        text = text.strip()

        return text

    def _parse_publish_date(self, date_text: Optional[str]) -> Optional[str]:
        """
        Parse publish date to ISO format (YYYY-MM-DD).

        Args:
            date_text: Raw date string

        Returns:
            str: ISO format date or None
        """
        if not date_text:
            return None

        # Use parent class method if available
        parsed = self.parse_date_from_text(date_text)
        if parsed:
            return parsed

        # Apple Daily specific formats
        try:
            patterns = [
                r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})',  # YYYY-MM-DD or YYYY/MM/DD
            ]

            for pattern in patterns:
                match = re.search(pattern, date_text)
                if match:
                    year, month, day = match.groups()
                    try:
                        date_obj = datetime(int(year), int(month), int(day))
                        return date_obj.strftime('%Y-%m-%d')
                    except ValueError:
                        continue

        except Exception as e:
            logger.warning(f"Failed to parse date '{date_text}': {e}")

        return None


# Standalone execution
if __name__ == '__main__':
    from scrapy.crawler import CrawlerProcess
    from scrapy.utils.project import get_project_settings

    # Install asyncio reactor first
    from twisted.internet import asyncioreactor
    asyncioreactor.install()

    settings = get_project_settings()
    settings.update({
        'DOWNLOAD_DELAY': 3,
        'CONCURRENT_REQUESTS_PER_DOMAIN': 1,
        'ROBOTSTXT_OBEY': True,
        'LOG_LEVEL': 'INFO',

        # Playwright
        'DOWNLOAD_HANDLERS': {
            "http": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
            "https": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
        },
        'PLAYWRIGHT_BROWSER_TYPE': 'chromium',
    })

    process = CrawlerProcess(settings)

    # Example: Crawl politics category for last 3 days
    process.crawl(
        AppleDailySpider,
        category='politics',
        days=3
    )

    process.start()
