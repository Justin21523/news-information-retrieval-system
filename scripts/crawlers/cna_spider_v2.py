#!/usr/bin/env python
"""
CNA News Spider v2 (Playwright + Anti-Detection)

This spider crawls news from Central News Agency (CNA) using Playwright
with comprehensive anti-detection mechanisms.

Strategy:
1. Crawl from category pages (aipl, aie, ahel, etc.)
2. Use Playwright for JavaScript rendering
3. Apply anti-detection (stealth + humanization)
4. Fallback to RSS feeds if needed

Usage:
    scrapy crawl cna_v2 -a start_date=2024-01-01 -a end_date=2024-01-31

Author: CNIRS Development Team
License: Educational Use Only
"""

import scrapy
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from urllib.parse import urljoin
from scrapy_playwright.page import PageMethod

from .base_playwright_spider import BasePlaywrightSpider, PlaywrightPageMethods

logger = logging.getLogger(__name__)


class CNANewsSpiderV2(BasePlaywrightSpider):
    """
    CNA News Spider using Playwright with anti-detection.

    Features:
        - Category-based crawling
        - Date filtering
        - Playwright rendering
        - Anti-bot detection
        - RSS fallback support
    """

    name = 'cna_v2'
    allowed_domains = ['cna.com.tw']

    # CNA category codes
    # https://www.cna.com.tw/list/<category>.aspx
    CATEGORIES = {
        'aipl': '政治',          # Politics
        'aie': '財經',           # Finance & Economics
        'ahel': '生活',          # Life
        'ait': '科技',           # Technology
        'asoc': '社會',          # Society
        'acul': '文化',          # Culture
        'aspt': '運動',          # Sports
        'amov': '娛樂',          # Entertainment
        'aopl': '國際',          # International
        'acn': '兩岸',           # Cross-Strait
        'aloc': '地方',          # Local
    }

    custom_settings = {
        'CONCURRENT_REQUESTS_PER_DOMAIN': 1,  # Very conservative
        'DOWNLOAD_DELAY': 3,  # 3 seconds between requests
        'AUTOTHROTTLE_ENABLED': True,
        'AUTOTHROTTLE_START_DELAY': 3,
        'AUTOTHROTTLE_MAX_DELAY': 15,
    }

    def __init__(self, start_date: str = None, end_date: str = None,
                categories: str = None, max_articles: int = None, *args, **kwargs):
        """
        Initialize CNA spider.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            categories: Comma-separated category codes (e.g., 'aipl,aie')
            max_articles: Maximum number of articles to crawl per category
        """
        super().__init__(*args, **kwargs)

        # Parse dates
        if start_date:
            self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        else:
            # Default: last 7 days
            self.start_date = datetime.now() - timedelta(days=7)

        if end_date:
            self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
        else:
            self.end_date = datetime.now()

        # Parse categories
        if categories:
            cat_list = [c.strip() for c in categories.split(',')]
            self.categories = {k: v for k, v in self.CATEGORIES.items() if k in cat_list}
        else:
            self.categories = self.CATEGORIES

        self.max_articles = int(max_articles) if max_articles else None

        # Statistics
        self.stats = {
            'articles_scraped': 0,
            'articles_filtered': 0,
            'errors': 0,
        }

        logger.info(f"CNA Spider initialized:")
        logger.info(f"  Date range: {self.start_date.date()} to {self.end_date.date()}")
        logger.info(f"  Categories: {list(self.categories.keys())}")
        logger.info(f"  Max articles per category: {self.max_articles or 'unlimited'}")

    def start_requests(self):
        """
        Generate start requests for all categories.

        Yields:
            scrapy.Request: Requests for category list pages
        """
        for category_code, category_name in self.categories.items():
            url = f"https://www.cna.com.tw/list/{category_code}.aspx"

            logger.info(f"Starting category: {category_name} ({category_code})")

            # Use Playwright with anti-detection
            yield scrapy.Request(
                url=url,
                callback=self.parse_category_page,
                meta=self.get_playwright_meta(
                    playwright_page_methods=[
                        PlaywrightPageMethods.wait_for_selector('div.mainList', timeout=10000),
                        PlaywrightPageMethods.random_scroll(),
                        PlaywrightPageMethods.wait_for_timeout(self.human_delay(1, 2) * 1000),
                    ]
                ),
                cb_kwargs={'category': category_code, 'category_name': category_name, 'page': 1},
                errback=self.errback_close_page,
            )

    async def errback_close_page(self, failure):
        """
        Handle errors and close Playwright page.

        Args:
            failure: Failure object
        """
        self.stats['errors'] += 1
        logger.error(f"Request failed: {failure.request.url}")
        logger.error(f"Error: {failure.value}")

        page = failure.request.meta.get('playwright_page')
        if page:
            await page.close()

    def parse_category_page(self, response, category: str, category_name: str, page: int):
        """
        Parse category list page to extract article links.

        Args:
            response: Scrapy response
            category: Category code
            category_name: Category name
            page: Page number

        Yields:
            scrapy.Request: Requests for individual articles
        """
        logger.info(f"Parsing {category_name} page {page}: {response.url}")

        # Extract article links
        # CNA uses div.mainList > ul > li > a structure
        article_links = response.css('div.mainList ul li a::attr(href)').getall()

        if not article_links:
            # Try alternative selectors
            article_links = response.css('article.item a::attr(href)').getall()

        if not article_links:
            # Try generic news selectors
            article_links = response.css('a[href*="/news/"]::attr(href)').getall()

        logger.info(f"Found {len(article_links)} article links on page {page}")

        articles_yielded = 0

        for link in article_links:
            # Build absolute URL
            article_url = urljoin(response.url, link)

            # Filter by URL pattern (CNA articles: /news/<category>/<ID>)
            if '/news/' not in article_url:
                continue

            # Check if max articles reached
            if self.max_articles and articles_yielded >= self.max_articles:
                logger.info(f"Reached max articles limit for {category_name}")
                break

            articles_yielded += 1

            # Crawl article with Playwright
            yield scrapy.Request(
                url=article_url,
                callback=self.parse_article,
                meta=self.get_playwright_meta(
                    playwright_page_methods=[
                        PlaywrightPageMethods.wait_for_selector('article, div.article', timeout=10000),
                        PlaywrightPageMethods.scroll_to_bottom(),
                        PlaywrightPageMethods.wait_for_timeout(self.human_delay(0.5, 1.5) * 1000),
                    ]
                ),
                cb_kwargs={'category': category, 'category_name': category_name},
                errback=self.errback_close_page,
            )

        # Check for next page (pagination)
        # CNA pagination: ?page=2, ?page=3, etc.
        next_page = response.css('a.pageNext::attr(href)').get()
        if next_page and articles_yielded > 0:
            next_url = urljoin(response.url, next_page)
            yield scrapy.Request(
                url=next_url,
                callback=self.parse_category_page,
                meta=self.get_playwright_meta(),
                cb_kwargs={'category': category, 'category_name': category_name, 'page': page + 1},
                errback=self.errback_close_page,
            )

    def parse_article(self, response, category: str, category_name: str):
        """
        Parse individual news article.

        Args:
            response: Scrapy response
            category: Category code
            category_name: Category name

        Yields:
            dict: Article data
        """
        try:
            # Extract article data
            article = self._extract_article_data(response, category, category_name)

            if not article:
                logger.warning(f"Failed to extract article data: {response.url}")
                self.stats['errors'] += 1
                return

            # Date filtering
            if article.get('published_date'):
                try:
                    pub_date = datetime.strptime(article['published_date'], '%Y-%m-%d')
                    if not (self.start_date <= pub_date <= self.end_date):
                        logger.debug(f"Article outside date range: {response.url}")
                        self.stats['articles_filtered'] += 1
                        return
                except ValueError:
                    logger.warning(f"Invalid date format: {article['published_date']}")

            self.stats['articles_scraped'] += 1
            logger.info(f"Scraped article {self.stats['articles_scraped']}: {article['title'][:50]}...")

            yield article

        except Exception as e:
            logger.error(f"Error parsing article {response.url}: {e}", exc_info=True)
            self.stats['errors'] += 1

    def _extract_article_data(self, response, category: str, category_name: str) -> Optional[Dict[str, Any]]:
        """
        Extract article data from response.

        Args:
            response: Scrapy response
            category: Category code
            category_name: Category name

        Returns:
            dict: Article data or None if extraction fails
        """
        # Title
        title = self.extract_text(
            response,
            css='h1.title, article h1, div.article_header h1::text'
        )

        if not title:
            return None

        # Content
        # CNA uses div.paragraph or article .content
        content_parts = response.css('div.paragraph p::text, article .content p::text').getall()
        content = '\n'.join([p.strip() for p in content_parts if p.strip()])

        if not content:
            # Try alternative selector
            content = self.extract_text(response, css='article::text, div.article::text')

        # Date
        date_text = self.extract_text(
            response,
            css='time::text, span.date::text, div.updatetime::text'
        )
        published_date = self.parse_date_from_text(date_text) if date_text else None

        # Author
        author = self.extract_text(
            response,
            css='span.author::text, div.author::text, span.reporter::text'
        )

        # Tags/Keywords
        tags = response.css('a.tag::text, a.keyword::text, meta[name="keywords"]::attr(content)').getall()
        if not tags:
            # Try meta keywords
            meta_keywords = response.css('meta[name="keywords"]::attr(content)').get()
            if meta_keywords:
                tags = [k.strip() for k in meta_keywords.split(',')]

        # Article ID (from URL)
        article_id = None
        match = re.search(r'/news/[a-z]+/(\d+)', response.url)
        if match:
            article_id = match.group(1)

        return {
            'source': 'cna',
            'article_id': article_id,
            'url': response.url,
            'title': title,
            'content': content,
            'category': category,
            'category_name': category_name,
            'published_date': published_date,
            'author': author,
            'tags': tags,
            'crawled_at': datetime.now().isoformat(),
        }

    def closed(self, reason):
        """
        Log statistics when spider closes.

        Args:
            reason: Closure reason
        """
        super().closed(reason)
        logger.info("=" * 70)
        logger.info("CNA Spider Statistics:")
        logger.info(f"  Articles scraped: {self.stats['articles_scraped']}")
        logger.info(f"  Articles filtered (date): {self.stats['articles_filtered']}")
        logger.info(f"  Errors: {self.stats['errors']}")
        logger.info("=" * 70)
