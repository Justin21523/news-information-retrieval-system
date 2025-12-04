#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ETtoday News Cloud Spider (東森新聞雲)

Spider for crawling news articles from ETtoday News Cloud.
Playwright-based crawler for dynamic content rendering.

Target: https://www.ettoday.net/
Categories: Politics (政治), Society (社會), Local (地方), Finance (財經),
            Entertainment (影劇), Sports (運動), International (國際),
            Life (生活), Health (健康), Travel (旅遊), Tech (科技),
            Pets (寵物), Gaming (電競), Consumer (消費)

Features:
    - Playwright browser automation (anti-detection)
    - Date range crawling
    - Category-based filtering
    - High-volume instant news handling

Usage:
    # Single category, 7 days
    scrapy runspider ettoday_spider.py -a days=7 -a category=politics -o ettoday.jsonl

    # All categories, custom date range
    scrapy runspider ettoday_spider.py -a start_date=2025-11-01 -a end_date=2025-11-18 -o output.jsonl

IMPORTANT NOTE ON PLAYWRIGHT REACTOR:
This spider requires asyncio reactor for Playwright.

When running standalone, the reactor is installed in the __main__ block.
When importing as a module, ensure reactor is installed before import:
    from twisted.internet import asyncioreactor
    asyncioreactor.install()
    from ettoday_spider import ETtodaySpider

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

# Import base spider
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.crawlers.base_playwright_spider import BasePlaywrightSpider

logger = logging.getLogger(__name__)


class ETtodaySpider(BasePlaywrightSpider):
    """
    Scrapy spider for ETtoday News Cloud (東森新聞雲).

    URL Structure:
        - Homepage: https://www.ettoday.net/
        - Category: https://www.ettoday.net/news/{category}/
        - Article: https://www.ettoday.net/news/{date}/{article_id}.htm

    Categories:
        politics: 政治
        society: 社會
        local: 地方
        finance: 財經
        china: 大陸
        world: 國際
        entertainment: 影劇
        sports: 運動
        life: 生活
        health: 健康
        travel: 旅遊
        tech: 科技
        pets: 寵物
        gaming: 電競
        consumer: 消費

    Complexity:
        Time: O(D * A) where D = days, A = avg articles per day
        Space: O(A) for storing article data
    """

    name = 'ettoday_news'
    allowed_domains = ['ettoday.net', 'www.ettoday.net']

    # Category mapping
    CATEGORIES = {
        'all': {'slug': '', 'name': '全部'},
        'politics': {'slug': 'politics', 'name': '政治'},
        'society': {'slug': 'society', 'name': '社會'},
        'local': {'slug': 'local', 'name': '地方'},
        'finance': {'slug': 'finance', 'name': '財經'},
        'china': {'slug': 'china', 'name': '大陸'},
        'world': {'slug': 'world', 'name': '國際'},
        'entertainment': {'slug': 'star', 'name': '影劇'},
        'sports': {'slug': 'sports', 'name': '運動'},
        'life': {'slug': 'life', 'name': '生活'},
        'health': {'slug': 'health', 'name': '健康'},
        'travel': {'slug': 'travel', 'name': '旅遊'},
        'tech': {'slug': 'tech', 'name': '科技'},
        'pets': {'slug': 'pets', 'name': '寵物'},
        'gaming': {'slug': 'game', 'name': '電競'},
        'consumer': {'slug': 'consumer', 'name': '消費'},
    }

    # Custom settings
    custom_settings = {
        'DOWNLOAD_DELAY': 3,  # 3 seconds between requests (high-volume site)
        'CONCURRENT_REQUESTS_PER_DOMAIN': 1,  # Single request at a time
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
            'args': [
                '--disable-blink-features=AutomationControlled',
                '--disable-dev-shm-usage',
                '--no-sandbox',
            ]
        },
        'PLAYWRIGHT_DEFAULT_NAVIGATION_TIMEOUT': 60000,

        # Output settings
        'FEEDS': {
            'data/raw/ettoday_news_%(time)s.jsonl': {
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
        Initialize ETtoday spider with date range and category.

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
        logger.info(f"ETtoday Spider Initialized")
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
        Generate start requests for ETtoday news.

        Yields:
            scrapy.Request: Playwright-enabled HTTP requests
        """
        # Build category URL
        if self.category == 'all':
            base_url = "https://www.ettoday.net/news/news-list.htm"
        else:
            base_url = f"https://www.ettoday.net/news/{self.category_slug}/"

        logger.info(f"Starting with URL: {base_url}")

        yield scrapy.Request(
            url=base_url,
            callback=self.parse_list_page,
            errback=self.handle_error,
            dont_filter=True,
            meta=self.get_playwright_meta(wait_selector='div.part_list_2')
        )

    def parse_list_page(self, response):
        """
        Parse ETtoday news list page to extract article URLs.

        Args:
            response: Scrapy response object

        Yields:
            scrapy.Request: Requests for article detail pages
            scrapy.Request: Request for next page (pagination)
        """
        self.pages_visited += 1
        logger.info(f"Parsing list page: {response.url}")

        # Extract article links
        article_selectors = response.css(
            'div.part_list_2 h2 a, '
            'div.part_list_2 h3 a, '
            'div.piece h2 a, '
            'div.piece h3 a, '
            'a[href*="/news/"]'
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
                meta=self.get_playwright_meta(wait_selector='article.story')
            )

        logger.info(f"Found {articles_found} articles on list page")

        # Pagination - ETtoday uses page numbers
        next_page = response.css('div.pages a.tipBarL::attr(href), a.next::attr(href)').get()

        if next_page and articles_found > 0 and self.pages_visited < 10:
            next_url = response.urljoin(next_page)
            logger.info(f"Following pagination to: {next_url}")

            yield scrapy.Request(
                url=next_url,
                callback=self.parse_list_page,
                errback=self.handle_error,
                dont_filter=True,
                meta=self.get_playwright_meta(wait_selector='div.part_list_2')
            )

    def parse_article(self, response):
        """
        Parse individual ETtoday article page.

        Article structure:
            - Title: <h1 class="title"> or <header h1>
            - Content: <div class="story"> <p>
            - Date: <time> or <span class="date">
            - Author: <div class="author"> or meta tag
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
                'div.news_time::text',
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
                'source': 'ETtoday',
                'source_name': '東森新聞雲',
                'crawled_at': datetime.now().isoformat(),
            }

            # Extract title
            title_selectors = [
                'h1.title::text',
                'header h1::text',
                'h1::text',
                'meta[property="og:title"]::attr(content)',
            ]
            article['title'] = self._extract_first(response, title_selectors)

            # Extract content paragraphs
            content_selectors = [
                'div.story p::text',
                'article.story p::text',
                'div.news_content p::text',
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
                'div.writer::text',
                'meta[name="author"]::attr(content)',
            ]
            article['author'] = self._extract_first(response, author_selectors) or '東森新聞雲'

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

            # Extract tags
            tag_selectors = response.css(
                'div.tag a::text, '
                'div.keywords a::text, '
                'a[rel="tag"]::text'
            ).getall()
            article['tags'] = [self._clean_text(tag) for tag in tag_selectors if tag]

            # Extract image URL
            image_selectors = [
                'figure.photo img::attr(src)',
                'div.story img::attr(src)',
                'article img::attr(src)',
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
        total_articles = self.articles_scraped + self.articles_failed
        success_rate = (self.articles_scraped / total_articles * 100) if total_articles > 0 else 0

        logger.info("=" * 70)
        logger.info("ETtoday Spider Statistics")
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
        # ETtoday article pattern: /news/{yyyymmdd}/{article_id}.htm
        return '/news/' in url and re.search(r'/\d{8}/\d+\.htm', url)

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
    # Install asyncio reactor for Playwright
    from twisted.internet import asyncioreactor
    asyncioreactor.install()

    from scrapy.crawler import CrawlerProcess
    from scrapy.utils.project import get_project_settings

    settings = get_project_settings()
    settings.update({
        'DOWNLOAD_DELAY': 3,
        'CONCURRENT_REQUESTS_PER_DOMAIN': 1,
        'ROBOTSTXT_OBEY': True,
        'LOG_LEVEL': 'INFO',
    })

    process = CrawlerProcess(settings)
    process.crawl(ETtodaySpider, category='pets', days=3)
    process.start()
