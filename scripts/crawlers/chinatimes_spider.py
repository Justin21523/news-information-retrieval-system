#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
China Times News Spider (中時新聞網)

Spider for crawling news articles from China Times (中國時報/中時新聞網).
Traditional HTTP crawler without Playwright (static HTML pages).

Target: https://www.chinatimes.com/
Categories: Politics (政治), Economics (財經), Society (社會), Entertainment (娛樂),
            Sports (體育), Life (生活), Opinion (言論), Tech (科技)

Features:
    - Traditional HTTP crawler (fast, low resource)
    - Date range crawling
    - Category-based filtering
    - Efficient parsing with lxml

Usage:
    # Single category, 7 days
    scrapy runspider chinatimes_spider.py -a days=7 -a category=politic -o chinatimes.jsonl

    # All categories, custom date range
    scrapy runspider chinatimes_spider.py -a start_date=2025-11-01 -a end_date=2025-11-18 -o output.jsonl

Author: Information Retrieval System
Date: 2025-11-18
"""

import scrapy
from datetime import datetime, timedelta
import json
import logging
import re
import hashlib
from typing import Optional, List, Dict, Any
from urllib.parse import urljoin, urlparse

logger = logging.getLogger(__name__)

try:
    # When imported as a package module.
    from scripts.crawlers.base_playwright_spider import BasePlaywrightSpider
except ImportError:
    # When executed via `scrapy runspider scripts/crawlers/chinatimes_spider.py`.
    try:
        from base_playwright_spider import BasePlaywrightSpider
    except ImportError:  # pragma: no cover
        BasePlaywrightSpider = scrapy.Spider

try:
    from scrapy_playwright.page import PageMethod
except ImportError:  # pragma: no cover
    PageMethod = None


def should_abort_request(request) -> bool:
    """
    Abort non-essential resource requests to speed up Playwright navigation.

    Args:
        request: Playwright request object.

    Returns:
        bool: True to abort the request.

    Complexity:
        Time: O(1)
        Space: O(1)
    """
    try:
        return request.resource_type in {"image", "media", "font"}
    except Exception:
        return False


class ChinaTimesSpider(BasePlaywrightSpider):
    """
    Scrapy spider for China Times News (中時新聞網).

    URL Structure:
        - Homepage: https://www.chinatimes.com/
        - Category: https://www.chinatimes.com/realtimenews/?chdtv
        - Article: https://www.chinatimes.com/realtimenews/{article_id}

    Categories:
        politic: 政治
        opinion: 言論
        life: 生活
        star: 娛樂
        money: 財經
        society: 社會
        hottopic: 熱門
        tube: 快點TV
        sports: 體育
        chinese: 兩岸
        global: 國際
        tech: 科技

    Complexity:
        Time: O(D * A) where D = days, A = avg articles per day
        Space: O(A) for storing article data
    """

    name = 'chinatimes_news'
    allowed_domains = ['chinatimes.com', 'www.chinatimes.com']
    handle_httpstatus_list = [403, 429, 500, 502, 503, 504]

    # Category mapping
    CATEGORIES = {
        'all': {'slug': '', 'name': '全部'},
        'politic': {'slug': 'politic', 'name': '政治'},
        'opinion': {'slug': 'opinion', 'name': '言論'},
        'life': {'slug': 'life', 'name': '生活'},
        'star': {'slug': 'star', 'name': '娛樂'},
        'money': {'slug': 'money', 'name': '財經'},
        'society': {'slug': 'society', 'name': '社會'},
        'hottopic': {'slug': 'hottopic', 'name': '熱門'},
        'tube': {'slug': 'tube', 'name': '快點TV'},
        'sports': {'slug': 'sports', 'name': '體育'},
        'chinese': {'slug': 'chinese', 'name': '兩岸'},
        'global': {'slug': 'global', 'name': '國際'},
        'tech': {'slug': 'tech', 'name': '科技'},
    }

    # Custom settings
    custom_settings = {
        'TWISTED_REACTOR': 'twisted.internet.asyncioreactor.AsyncioSelectorReactor',
        'DOWNLOAD_DELAY': 1.5,  # balance speed and throttling
        'CONCURRENT_REQUESTS_PER_DOMAIN': 2,  # Can handle 2 concurrent
        'ROBOTSTXT_OBEY': True,
        'RETRY_TIMES': 3,
        'RETRY_HTTP_CODES': [500, 502, 503, 504, 408, 429],

        # User-Agent rotation
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',

        # Optional Playwright fallback (used when `use_playwright=auto/1`).
        'DOWNLOAD_HANDLERS': {
            "http": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
            "https": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
        },
        'PLAYWRIGHT_BROWSER_TYPE': 'chromium',
        'PLAYWRIGHT_LAUNCH_OPTIONS': {
            'headless': True,
            'timeout': 60000,
            'args': [
                '--disable-blink-features=AutomationControlled',
                '--disable-dev-shm-usage',
                '--no-sandbox',
            ],
        },
        'PLAYWRIGHT_DEFAULT_NAVIGATION_TIMEOUT': 60000,
        'PLAYWRIGHT_ABORT_REQUEST': should_abort_request,
        'DOWNLOADER_MIDDLEWARES': {
            'scripts.crawlers.middlewares.stealth_middleware.StealthMiddleware': 585,
            'scripts.crawlers.middlewares.humanization_middleware.HumanizationMiddleware': 586,
        },
        'PLAYWRIGHT_STEALTH_ENABLED': True,
        'HUMANIZATION_ENABLED': False,

        # Output settings
        'FEEDS': {
            '/mnt/c/data/information-retrieval/raw/chinatimes_news_%(time)s.jsonl': {
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
                 max_pages: int = None,
                 max_links_per_page: int = None,
                 use_playwright: str = 'auto',
                 *args, **kwargs):
        """
        Initialize China Times spider with date range and category.

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

        self.max_links_per_page: Optional[int]
        if max_links_per_page is not None:
            max_links_per_page = int(max_links_per_page)
            self.max_links_per_page = max_links_per_page if max_links_per_page > 0 else None
        else:
            self.max_links_per_page = None

        # Pagination guard: auto-scale for larger historical windows.
        # This is a safety limit; date filtering in parse_article will further reduce output.
        window_days = max(1, int((self.end_date - self.start_date).days) + 1)
        if max_pages is not None:
            self.max_pages = int(max_pages)
        else:
            self.max_pages = min(2000, max(10, window_days * 3))

        mode = str(use_playwright).strip().lower() if use_playwright is not None else 'auto'
        if mode in {'1', 'true', 'yes', 'always'}:
            self.playwright_mode = 'always'
        elif mode in {'0', 'false', 'no', 'never'}:
            self.playwright_mode = 'never'
        else:
            self.playwright_mode = 'auto'

        logger.info("=" * 70)
        logger.info(f"China Times Spider Initialized")
        logger.info("=" * 70)
        logger.info(f"Category: {self.category_name} ({self.category})")
        logger.info(f"Date Range: {self.start_date.date()} to {self.end_date.date()}")
        logger.info(f"Days: {(self.end_date - self.start_date).days + 1}")
        logger.info(f"Max pages: {self.max_pages}")
        logger.info(f"Max links per page: {self.max_links_per_page or 'unlimited'}")
        logger.info(f"Playwright mode: {self.playwright_mode}")
        logger.info("=" * 70)

        # Statistics
        self.articles_scraped = 0
        self.articles_failed = 0
        self.pages_visited = 0
        self.seen_urls = set()

    def start_requests(self):
        """
        Generate start requests for China Times news.

        Yields:
            scrapy.Request: HTTP requests
        """
        base_url = "https://chinatimes.com/realtimenews/"

        logger.info(f"Starting with URL: {base_url}")

        request_meta = {}
        if self.playwright_mode == 'always':
            request_meta = self._get_playwright_meta(wait_ms=1200)

        yield scrapy.Request(
            url=base_url,
            callback=self.parse_list_page,
            errback=self.handle_error,
            dont_filter=True,
            meta=request_meta,
        )

    def _get_playwright_meta(self, wait_ms: int = 1000) -> Dict[str, Any]:
        """Build Playwright meta for fallback requests (O(1) time/space)."""
        if hasattr(self, 'get_playwright_meta'):
            methods = []
            if PageMethod and wait_ms:
                methods = [PageMethod('wait_for_timeout', int(wait_ms))]
            return self.get_playwright_meta(playwright_page_methods=methods)

        meta: Dict[str, Any] = {'playwright': True}
        if PageMethod and wait_ms:
            meta['playwright_page_methods'] = [PageMethod('wait_for_timeout', int(wait_ms))]
        return meta

    def parse_list_page(self, response):
        """
        Parse China Times news list page to extract article URLs.

        Args:
            response: Scrapy response object

        Yields:
            scrapy.Request: Requests for article detail pages
            scrapy.Request: Request for next page (pagination)
        """
        self.pages_visited += 1
        logger.info(f"Parsing list page: {response.url}")

        if response.status in {403, 429, 500, 502, 503, 504}:
            logger.warning(f"Non-200 list page (status={response.status}): {response.url}")
            if self.playwright_mode == 'auto' and not response.meta.get('playwright') and not response.meta.get('playwright_fallback'):
                meta = {
                    'playwright_fallback': True,
                    **self._get_playwright_meta(wait_ms=2500),
                }
                yield scrapy.Request(
                    url=response.url,
                    callback=self.parse_list_page,
                    errback=self.handle_error,
                    dont_filter=True,
                    meta=meta,
                )
            return

        # Extract article links
        article_selectors = response.css(
            'h3.title a, '
            'h2.title a, '
            'div.article-list a, '
            'ul.article-list a, '
            'a[href*="/realtimenews/"]'
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
                dont_filter=False,
                meta=(self._get_playwright_meta(wait_ms=1200) if response.meta.get('playwright') or self.playwright_mode == 'always' else {}),
            )

            if self.max_links_per_page is not None and articles_found >= self.max_links_per_page:
                break

        logger.info(f"Found {articles_found} articles on list page")

        # Pagination
        next_page = response.css('a.page-link[rel="next"]::attr(href), a.next::attr(href)').get()

        if next_page and articles_found > 0 and self.pages_visited < self.max_pages:
            next_url = response.urljoin(next_page)
            logger.info(f"Following pagination to: {next_url}")

            yield scrapy.Request(
                url=next_url,
                callback=self.parse_list_page,
                errback=self.handle_error,
                dont_filter=True,
                meta=(self._get_playwright_meta(wait_ms=1200) if response.meta.get('playwright') or self.playwright_mode == 'always' else {}),
            )

    def parse_article(self, response):
        """
        Parse individual China Times article page.

        Article structure:
            - Title: <h1 class="article-title">
            - Content: <div class="article-body"> <p>
            - Date: <time> or <div class="meta-info-value">
            - Author: <div class="author">
            - Category: from breadcrumb or URL

        Args:
            response: Scrapy response object

        Yields:
            dict: Article data in standardized format
        """
        if response.status in {403, 429, 500, 502, 503, 504}:
            logger.warning(f"Non-200 article response (status={response.status}): {response.url}")
            if (
                self.playwright_mode in {'auto', 'always'}
                and not response.meta.get('playwright')
                and not response.meta.get('playwright_fallback')
            ):
                meta = {
                    'playwright_fallback': True,
                    **self._get_playwright_meta(wait_ms=2500),
                }
                yield scrapy.Request(
                    url=response.url,
                    callback=self.parse_article,
                    errback=self.handle_error,
                    dont_filter=True,
                    meta=meta,
                )
                return
            self.articles_failed += 1
            return

        try:
            logger.debug(f"Parsing article: {response.url}")

            # Extract date first to filter
            date_text = self._extract_first(response, [
                'time::attr(datetime)',
                'time::text',
                'div.meta-info-value::text',
                'span.date::text',
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
                'source': 'China Times',
                'source_name': '中時新聞網',
                'crawled_at': datetime.now().isoformat(),
            }

            # Extract title
            title_selectors = [
                'h1.article-title::text',
                'h1::text',
                'header h1::text',
                'meta[property="og:title"]::attr(content)',
            ]
            article['title'] = self._extract_first(response, title_selectors)

            # Extract content paragraphs
            content_selectors = [
                'div.article-body p::text',
                'article p::text',
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
                'div.author::text',
                'span.author::text',
                'div.reporter::text',
                'meta[name="author"]::attr(content)',
            ]
            article['author'] = self._extract_first(response, author_selectors) or '中時新聞網'

            # Extract publish date
            article['published_date'] = published_date

            # Extract category
            breadcrumb = response.css('div.breadcrumb a::text, nav.breadcrumb a::text').getall()
            if breadcrumb and len(breadcrumb) > 1:
                article['category'] = breadcrumb[-1].strip()
                article['category_name'] = breadcrumb[-1].strip()
            else:
                # Extract from URL
                category = self._extract_category_from_url(response.url)
                article['category'] = category
                article['category_name'] = self.CATEGORIES.get(category, {}).get('name', category)

            # Category filtering (best-effort; ChinaTimes list pages are not strictly category-scoped).
            if self.category != 'all':
                detected = self._extract_category_from_url(response.url)
                if detected != self.category:
                    return

            # Extract tags
            tag_selectors = response.css(
                'div.article-hash-tag a::text, '
                'div.keywords a::text, '
                'a[rel="tag"]::text'
            ).getall()
            article['tags'] = [self._clean_text(tag) for tag in tag_selectors if tag]

            # Extract image URL
            image_selectors = [
                'figure.article-main-img img::attr(src)',
                'article img::attr(src)',
                'div.article-body img::attr(src)',
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
        """Handle request errors with detailed logging and Playwright fallback."""
        from scrapy.spidermiddlewares.httperror import HttpError
        from twisted.internet.error import DNSLookupError, TimeoutError, TCPTimedOutError, ConnectError

        request = failure.request

        # Best-effort Playwright retry for connectivity/WAF failures.
        if (
            self.playwright_mode == 'auto'
            and not request.meta.get('playwright')
            and not request.meta.get('playwright_fallback')
        ):
            should_retry = False
            if failure.check(DNSLookupError, TimeoutError, TCPTimedOutError, ConnectError):
                should_retry = True
            elif failure.check(HttpError):
                response = failure.value.response
                if response is not None and response.status in {403, 429, 500, 502, 503, 504}:
                    should_retry = True

            if should_retry:
                logger.warning(f"Retrying with Playwright fallback: {request.url}")
                meta = {
                    **request.meta,
                    'playwright_fallback': True,
                    **self._get_playwright_meta(wait_ms=2500),
                }
                yield request.replace(dont_filter=True, meta=meta)
                return

        logger.error("=" * 70)
        logger.error(f"Request failed: {request.url}")
        logger.error(f"Error type: {failure.type.__name__}")
        logger.error(f"Error value: {failure.value}")
        logger.error("=" * 70)
        self.articles_failed += 1

    def closed(self, reason):
        """Log spider closure statistics."""
        total_articles = self.articles_scraped + self.articles_failed
        success_rate = (self.articles_scraped / total_articles * 100) if total_articles > 0 else 0

        logger.info("=" * 70)
        logger.info("China Times Spider Statistics")
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
        """Check if URL is a news article."""
        if not url:
            return False
        return '/realtimenews/' in url and re.search(r'/\d{8}\d+', url)

    def _extract_category_from_url(self, url: str) -> str:
        """Extract category from article URL."""
        for category_code, cat_info in self.CATEGORIES.items():
            if category_code == 'all':
                continue
            if cat_info['slug'] in url:
                return category_code
        return 'unknown'

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
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _parse_publish_date(self, date_text: Optional[str]) -> Optional[str]:
        """Parse publish date to ISO format (YYYY-MM-DD)."""
        if not date_text:
            return None

        try:
            patterns = [
                r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})',  # YYYY-MM-DD or YYYY/MM/DD
                r'(\d{4})年(\d{1,2})月(\d{1,2})日',  # Chinese format
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

    settings = get_project_settings()
    settings.update({
        'DOWNLOAD_DELAY': 2,
        'CONCURRENT_REQUESTS_PER_DOMAIN': 2,
        'ROBOTSTXT_OBEY': True,
        'LOG_LEVEL': 'INFO',
    })

    process = CrawlerProcess(settings)
    process.crawl(ChinaTimesSpider, category='politic', days=3)
    process.start()
