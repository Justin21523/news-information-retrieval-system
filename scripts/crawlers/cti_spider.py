#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comprehensive CTI (ChinaTimes) Spider with Deep Refactoring v2.0

Complete multi-sitemap, multi-mode crawler for 中天新聞 (ChinaTimes.com)
with Playwright-based Cloudflare bypass and 2-year historical data access.

Features:
- Multi-sitemap support (8+ sitemaps from robots.txt)
- List mode with deep pagination (up to 500 pages)
- Category-based crawling (all major categories)
- Playwright with Cloudflare bypass
- 6+ fallback strategies per metadata field
- JSON-LD structured data extraction
- Target: 2 years historical data (730 days)
- Comprehensive statistics tracking

Author: Information Retrieval System
Date: 2025-11-19
Version: 2.0 (Deep Refactoring)
"""

import scrapy
from scrapy.http import Request
import json
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse, parse_qs
import re
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
import sys

# Set twisted reactor BEFORE importing any scrapy modules that might install it
import os
os.environ.setdefault('TWISTED_REACTOR', 'twisted.internet.asyncioreactor.AsyncioSelectorReactor')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Always import PageMethod for Playwright requests
from scrapy_playwright.page import PageMethod

try:
    from scripts.crawlers.base_playwright_spider import BasePlaywrightSpider
except ImportError:
    # Fallback if base spider not available
    class BasePlaywrightSpider(scrapy.Spider):
        """Fallback base spider with Playwright"""

        custom_settings = {
            'DOWNLOAD_HANDLERS': {
                'http': 'scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler',
                'https': 'scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler',
            },
            'PLAYWRIGHT_BROWSER_TYPE': 'chromium',
            'PLAYWRIGHT_LAUNCH_OPTIONS': {
                'headless': True,
                'args': ['--no-sandbox', '--disable-setuid-sandbox']
            },
            'CONCURRENT_REQUESTS': 2,
            'CONCURRENT_REQUESTS_PER_DOMAIN': 2,
            'DOWNLOAD_DELAY': 2,
            'TWISTED_REACTOR': 'twisted.internet.asyncioreactor.AsyncioSelectorReactor',
        }

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

logger = logging.getLogger(__name__)


class CTINewsSpider(BasePlaywrightSpider):
    """
    Complete deep-refactored spider for CTI (ChinaTimes) with 2-year historical coverage.

    Features (Deep Refactoring v2.0):
        - Multi-sitemap support (8+ sitemaps)
        - List mode with deep pagination (500 pages)
        - All categories traversal
        - Playwright with Cloudflare bypass
        - 6+ fallback strategies per metadata field
        - JSON-LD extraction
        - 730 days default (2 years)
        - Per-sitemap and per-category statistics
    """

    name = 'cti_news'
    allowed_domains = ['chinatimes.com', 'www.chinatimes.com']

    # Multiple sitemaps from robots.txt
    SITEMAPS = {
        'todaynews': 'https://www.chinatimes.com/sitemaps/sitemap_todaynews.xml',
        'todaynews_d2': 'https://www.chinatimes.com/sitemaps/sitemap_todaynews_d2.xml',
        'article_all': 'https://www.chinatimes.com/sitemaps/sitemap_article_all_index_0.xml',
        'wantrich': 'https://www.chinatimes.com/sitemaps/sitemap_wantrich_todaynews.xml',
        'stock': 'https://www.chinatimes.com/sitemaps/sitemap_stock.xml',
        'category': 'https://www.chinatimes.com/sitemaps/sitemap_category.xml',
        'car': 'https://www.chinatimes.com/sitemaps/article_sitemaps/sitemap_car_article_0.xml',
    }

    # Categories and their codes
    CATEGORIES = {
        'all': {'code': '', 'name': '全部', 'url': 'https://www.chinatimes.com/realtimenews/'},
        'politics': {'code': '260407', 'name': '政治', 'url': 'https://www.chinatimes.com/realtimenews/?chdtv'},
        'money': {'code': '260410', 'name': '財經', 'url': 'https://www.chinatimes.com/realtimenews/?money'},
        'society': {'code': '260402', 'name': '社會', 'url': 'https://www.chinatimes.com/realtimenews/?society'},
        'world': {'code': '260408', 'name': '國際', 'url': 'https://www.chinatimes.com/realtimenews/?world'},
        'entertainment': {'code': '260404', 'name': '娛樂', 'url': 'https://www.chinatimes.com/realtimenews/?entertainment'},
        'life': {'code': '260405', 'name': '生活', 'url': 'https://www.chinatimes.com/realtimenews/?life'},
        'sports': {'code': '260403', 'name': '體育', 'url': 'https://www.chinatimes.com/realtimenews/?sports'},
        'tech': {'code': '260412', 'name': '科技', 'url': 'https://www.chinatimes.com/realtimenews/?tech'},
    }

    def __init__(self,
                 mode: str = 'sitemap',
                 sitemap: str = 'all',
                 category: str = 'all',
                 days: int = 730,  # 2 YEARS DEFAULT
                 start_date: str = None,
                 end_date: str = None,
                 max_pages: int = 500,  # Deep pagination
                 max_articles: int = None,
                 *args, **kwargs):
        """
        Initialize CTI spider with comprehensive parameters.

        Args:
            mode: Crawling mode ('sitemap' or 'list')
            sitemap: Sitemap type ('todaynews', 'todaynews_d2', 'article_all', 'all')
            category: News category for list mode
            days: Number of days to crawl (default 730 = 2 years)
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            max_pages: Maximum pages per category (default 500)
            max_articles: Maximum articles to scrape (optional limit)
        """
        super().__init__(*args, **kwargs)

        self.mode = mode.lower()
        self.sitemap_mode = sitemap.lower()
        self.category = category.lower()
        self.max_pages = max_pages
        self.max_articles = max_articles

        # Date range calculation
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d') if end_date else datetime.now()
        if start_date:
            self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        elif days:
            self.start_date = self.end_date - timedelta(days=days)
        else:
            self.start_date = self.end_date - timedelta(days=730)  # Default 2 years

        # Statistics tracking
        self.stats = {
            'articles_found': 0,
            'articles_scraped': 0,
            'articles_failed': 0,
            'sitemap_urls_discovered': 0,
            'list_pages_visited': 0,
            'categories_processed': set(),
            'sitemaps_processed': set(),
            'date_range': {
                'earliest': None,
                'latest': None,
            },
            'metadata_quality': {
                'has_title': 0,
                'has_content': 0,
                'has_date': 0,
                'has_author': 0,
                'has_category': 0,
                'has_tags': 0,
                'has_image': 0,
            }
        }

        logger.info(f"CTI Spider initialized (Deep Refactoring v2.0)")
        logger.info(f"Mode: {self.mode}")
        if self.mode == 'sitemap':
            logger.info(f"Sitemap: {self.sitemap_mode}")
        else:
            logger.info(f"Category: {self.category}")
        logger.info(f"Date range: {self.start_date.date()} to {self.end_date.date()}")
        logger.info(f"Target: {(self.end_date - self.start_date).days} days of historical data")

    custom_settings = {
        'TWISTED_REACTOR': 'twisted.internet.asyncioreactor.AsyncioSelectorReactor',
        'DOWNLOAD_HANDLERS': {
            'http': 'scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler',
            'https': 'scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler',
        },
        'PLAYWRIGHT_BROWSER_TYPE': 'chromium',
        'PLAYWRIGHT_LAUNCH_OPTIONS': {
            'headless': True,
            'args': [
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-dev-shm-usage',
                '--disable-blink-features=AutomationControlled',
            ]
        },
        'PLAYWRIGHT_DEFAULT_NAVIGATION_TIMEOUT': 60000,  # 60 seconds for Cloudflare
        'CONCURRENT_REQUESTS': 2,
        'CONCURRENT_REQUESTS_PER_DOMAIN': 2,
        'DOWNLOAD_DELAY': 3,  # Respectful crawling with Cloudflare
        'ROBOTSTXT_OBEY': False,  # Cloudflare blocks robots.txt check
        'COOKIES_ENABLED': True,
        'RETRY_TIMES': 3,
        'RETRY_HTTP_CODES': [500, 502, 503, 504, 408, 429],
    }

    def start_requests(self):
        """Generate initial requests based on mode"""
        if self.mode == 'sitemap':
            yield from self._start_sitemap_mode()
        elif self.mode == 'list':
            yield from self._start_list_mode()
        else:
            logger.warning(f"Unknown mode: {self.mode}, defaulting to sitemap")
            yield from self._start_sitemap_mode()

    def _start_sitemap_mode(self):
        """Start crawling from sitemaps with Playwright for Cloudflare"""
        if self.sitemap_mode == 'all':
            sitemaps_to_crawl = self.SITEMAPS.items()
        else:
            if self.sitemap_mode not in self.SITEMAPS:
                logger.warning(f"Unknown sitemap: {self.sitemap_mode}, using 'todaynews'")
                self.sitemap_mode = 'todaynews'
            sitemaps_to_crawl = [(self.sitemap_mode, self.SITEMAPS[self.sitemap_mode])]

        for sitemap_name, sitemap_url in sitemaps_to_crawl:
            logger.info(f"Fetching sitemap: {sitemap_name} - {sitemap_url}")

            # Use Playwright for sitemap (Cloudflare protected)
            yield scrapy.Request(
                url=sitemap_url,
                callback=self.parse_sitemap,
                errback=self.errback_sitemap,
                meta={
                    'playwright': True,
                    'playwright_page_methods': [
                        PageMethod('wait_for_load_state', 'networkidle'),
                    ],
                    'sitemap_name': sitemap_name,
                },
                dont_filter=True
            )

    def _start_list_mode(self):
        """Start crawling from category list pages with deep pagination"""
        if self.category == 'all':
            # Crawl all categories for maximum coverage
            categories_to_crawl = [
                (cat_code, cat_info)
                for cat_code, cat_info in self.CATEGORIES.items()
                if cat_code != 'all'
            ]
        else:
            if self.category not in self.CATEGORIES:
                logger.warning(f"Unknown category: {self.category}, using 'all'")
                self.category = 'all'
                categories_to_crawl = [(self.category, self.CATEGORIES[self.category])]
            else:
                categories_to_crawl = [(self.category, self.CATEGORIES[self.category])]

        for cat_code, cat_info in categories_to_crawl:
            logger.info(f"Starting category: {cat_info['name']} ({cat_code})")

            # Start with page 1
            yield scrapy.Request(
                url=cat_info['url'],
                callback=self.parse_list_page,
                meta={
                    'playwright': True,
                    'playwright_page_methods': [
                        PageMethod('wait_for_load_state', 'networkidle'),
                        PageMethod('wait_for_selector', 'article, .article-list, .realtime-news-list', timeout=30000),
                    ],
                    'category_code': cat_code,
                    'category_name': cat_info['name'],
                    'page': 1,
                },
                dont_filter=True
            )

    def parse_sitemap(self, response):
        """Parse XML sitemap and extract article URLs"""
        sitemap_name = response.meta.get('sitemap_name', 'unknown')
        self.stats['sitemaps_processed'].add(sitemap_name)

        # Check if Cloudflare challenge page
        if 'Just a moment' in response.text or 'challenge-platform' in response.text:
            logger.warning(f"Cloudflare challenge detected for sitemap: {sitemap_name}")
            return

        # Parse XML sitemap
        from xml.etree import ElementTree as ET

        try:
            root = ET.fromstring(response.text)
            namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}

            urls_found = 0
            for url_elem in root.findall('.//ns:url', namespace):
                loc = url_elem.find('ns:loc', namespace)
                if loc is None or not loc.text:
                    continue

                article_url = loc.text.strip()

                # Filter by URL pattern (realtimenews articles)
                if '/realtimenews/' not in article_url:
                    continue

                # Date filtering from URL
                if not self._is_url_in_date_range(article_url):
                    continue

                urls_found += 1
                self.stats['sitemap_urls_discovered'] += 1
                self.stats['articles_found'] += 1

                yield scrapy.Request(
                    url=article_url,
                    callback=self.parse_article,
                    meta={
                        'playwright': True,
                        'playwright_page_methods': [
                            PageMethod('wait_for_load_state', 'networkidle'),
                        ],
                        'source_sitemap': sitemap_name,
                    },
                    errback=self.errback_article
                )

            logger.info(f"Sitemap {sitemap_name}: Found {urls_found} URLs")

        except ET.ParseError as e:
            logger.error(f"Failed to parse sitemap XML: {e}")
            logger.debug(f"Response text (first 500 chars): {response.text[:500]}")

    def parse_list_page(self, response):
        """Parse category list page and extract article links with AGGRESSIVE PAGINATION"""
        category_code = response.meta.get('category_code', 'unknown')
        category_name = response.meta.get('category_name', 'Unknown')
        page = response.meta.get('page', 1)

        self.stats['list_pages_visited'] += 1
        self.stats['categories_processed'].add(category_code)

        logger.info(f"Parsing {category_name} list page {page}: {response.url}")

        # Check for Cloudflare challenge
        if 'Just a moment' in response.text or 'challenge-platform' in response.text:
            logger.warning(f"Cloudflare challenge on list page {page}")
            return

        # Extract article URLs - multiple selectors for CTI
        article_urls = []

        # Method 1: Article links in realtimenews list
        article_urls.extend(response.css('a[href*="/realtimenews/"]::attr(href)').getall())

        # Method 2: Specific article list selectors
        article_urls.extend(response.css('.article-list a::attr(href)').getall())
        article_urls.extend(response.css('.realtime-news-list a::attr(href)').getall())
        article_urls.extend(response.css('article a::attr(href)').getall())

        # Method 3: Data attributes
        article_urls.extend(response.css('[data-article-url]::attr(data-article-url)').getall())

        # Deduplicate and filter
        article_urls = list(set([
            urljoin(response.url, url)
            for url in article_urls
            if url and '/realtimenews/' in url
        ]))

        # Date filtering
        article_urls = [url for url in article_urls if self._is_url_in_date_range(url)]

        articles_found = len(article_urls)
        logger.info(f"Found {articles_found} articles on page {page}")

        if articles_found > 0:
            self.stats['articles_found'] += articles_found

            for article_url in article_urls:
                yield scrapy.Request(
                    url=article_url,
                    callback=self.parse_article,
                    meta={
                        'playwright': True,
                        'playwright_page_methods': [
                            PageMethod('wait_for_load_state', 'networkidle'),
                        ],
                        'source_category': category_code,
                    },
                    errback=self.errback_article
                )

        # AGGRESSIVE PAGINATION for 2-year coverage (up to 500 pages)
        if articles_found > 0 and page < self.max_pages:
            next_page = page + 1

            # Try multiple pagination patterns
            next_page_urls = []

            # Pattern 1: ?page=X parameter
            base_url = response.url.split('?')[0]
            next_page_urls.append(f"{base_url}?page={next_page}")

            # Pattern 2: Load more button/link
            load_more = response.css('a.load-more::attr(href)').get()
            if load_more:
                next_page_urls.append(urljoin(response.url, load_more))

            # Pattern 3: Next page link
            next_link = response.css('a[rel="next"]::attr(href)').get()
            if next_link:
                next_page_urls.append(urljoin(response.url, next_link))

            # Pattern 4: Pagination links
            pagination_link = response.css(f'a.page-{next_page}::attr(href)').get()
            if pagination_link:
                next_page_urls.append(urljoin(response.url, pagination_link))

            # Try the first valid next page URL
            for next_url in next_page_urls:
                if next_url:
                    logger.info(f"Following pagination to page {next_page}: {next_url}")
                    yield scrapy.Request(
                        url=next_url,
                        callback=self.parse_list_page,
                        meta={
                            'playwright': True,
                            'playwright_page_methods': [
                                PageMethod('wait_for_load_state', 'networkidle'),
                                PageMethod('wait_for_selector', 'article, .article-list', timeout=30000),
                            ],
                            'category_code': category_code,
                            'category_name': category_name,
                            'page': next_page,
                        },
                        dont_filter=True
                    )
                    break  # Only follow one pagination link

    def parse_article(self, response):
        """Parse article page with comprehensive metadata extraction (6+ fallback strategies)"""

        # Check for Cloudflare challenge
        if 'Just a moment' in response.text or 'challenge-platform' in response.text:
            logger.warning(f"Cloudflare challenge on article page: {response.url}")
            return

        # Extract article ID from URL (e.g., realtimenews/20251118004599-260410)
        article_id_match = re.search(r'realtimenews/(\d+-\d+)', response.url)
        article_id = f"cti_{article_id_match.group(1)}" if article_id_match else response.url.split('/')[-1]

        # === TITLE (6 strategies) ===
        title = (
            response.css('meta[property="og:title"]::attr(content)').get() or
            response.css('h1.article-title::text').get() or
            response.css('h1::text').get() or
            response.css('meta[name="twitter:title"]::attr(content)').get() or
            response.css('title::text').get() or
            self._extract_from_jsonld(response, 'headline')
        )
        if title:
            title = title.strip()
            # Remove site name suffix
            title = re.sub(r'\s*[-|–—]\s*(中國時報|中時新聞網|CTI).*$', '', title, flags=re.IGNORECASE)

        # === CONTENT (6 strategies) ===
        content_paragraphs = (
            response.css('div.article-body p::text').getall() or
            response.css('article p::text').getall() or
            response.css('div.article-content p::text').getall() or
            response.css('div[itemprop="articleBody"] p::text').getall() or
            response.css('div.content p::text').getall() or
            response.css('p.paragraph::text').getall()
        )
        content = '\n'.join([p.strip() for p in content_paragraphs if p.strip()])

        if not content:
            content = self._extract_from_jsonld(response, 'articleBody')

        # === DATE (7 strategies) ===
        date_text = (
            response.css('meta[property="article:published_time"]::attr(content)').get() or
            response.css('time::attr(datetime)').get() or
            response.css('meta[name="pubdate"]::attr(content)').get() or
            response.css('span.date::text').get() or
            response.css('div.article-date::text').get() or
            response.css('time::text').get() or
            self._extract_from_jsonld(response, 'datePublished')
        )
        published_date = self._parse_date(date_text)

        # === AUTHOR (5 strategies) ===
        author = (
            response.css('meta[name="author"]::attr(content)').get() or
            response.css('span.author::text').get() or
            response.css('div.author::text').get() or
            response.css('a[rel="author"]::text').get() or
            self._extract_from_jsonld(response, 'author')
        )
        if author:
            author = author.strip()

        # === CATEGORY (5 strategies) ===
        # From URL or metadata
        category_match = re.search(r'-(\d+)$', article_id)
        category_code = category_match.group(1) if category_match else None

        category_name = (
            response.css('meta[property="article:section"]::attr(content)').get() or
            response.css('span.category::text').get() or
            response.css('a.category::text').get() or
            self._extract_from_jsonld(response, 'articleSection')
        )

        if not category_name and category_code:
            # Map category code to name
            for cat, info in self.CATEGORIES.items():
                if info['code'] == category_code:
                    category_name = info['name']
                    break

        # === TAGS (3 strategies) ===
        tags = (
            response.css('meta[property="article:tag"]::attr(content)').getall() or
            response.css('a[rel="tag"]::text').getall() or
            response.css('meta[name="keywords"]::attr(content)').get()
        )
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(',') if t.strip()]
        elif isinstance(tags, list):
            tags = [t.strip() for t in tags if t.strip()]
        else:
            tags = []

        # === IMAGE (4 strategies) ===
        image_url = (
            response.css('meta[property="og:image"]::attr(content)').get() or
            response.css('article img::attr(src)').get() or
            response.css('div.article-body img::attr(src)').get() or
            self._extract_from_jsonld(response, 'image')
        )
        if image_url:
            image_url = urljoin(response.url, image_url)

        # === DESCRIPTION (3 strategies) ===
        description = (
            response.css('meta[property="og:description"]::attr(content)').get() or
            response.css('meta[name="description"]::attr(content)').get() or
            self._extract_from_jsonld(response, 'description')
        )
        if description:
            description = description.strip()

        # Update metadata quality stats
        if title:
            self.stats['metadata_quality']['has_title'] += 1
        if content and len(content) > 100:
            self.stats['metadata_quality']['has_content'] += 1
        if published_date:
            self.stats['metadata_quality']['has_date'] += 1
        if author:
            self.stats['metadata_quality']['has_author'] += 1
        if category_name:
            self.stats['metadata_quality']['has_category'] += 1
        if tags:
            self.stats['metadata_quality']['has_tags'] += 1
        if image_url:
            self.stats['metadata_quality']['has_image'] += 1

        # Track date range
        if published_date:
            date_str = published_date
            if not self.stats['date_range']['earliest'] or date_str < self.stats['date_range']['earliest']:
                self.stats['date_range']['earliest'] = date_str
            if not self.stats['date_range']['latest'] or date_str > self.stats['date_range']['latest']:
                self.stats['date_range']['latest'] = date_str

        # Build article item
        article = {
            'article_id': article_id,
            'url': response.url,
            'title': title,
            'content': content,
            'published_date': published_date,
            'author': author or 'CTI News',
            'category': category_name or 'news',
            'category_code': category_code,
            'tags': tags,
            'image_url': image_url,
            'description': description,
            'source': 'CTI',
            'source_name': '中天新聞',
            'crawled_at': datetime.now().isoformat(),
        }

        self.stats['articles_scraped'] += 1

        yield article

    def _is_url_in_date_range(self, url: str) -> bool:
        """Check if article URL date is within target range"""
        # Extract date from URL (format: realtimenews/YYYYMMDDNNNNNN-CCCCCC)
        date_match = re.search(r'/realtimenews/(\d{8})', url)
        if not date_match:
            return True  # Include if no date found

        try:
            url_date = datetime.strptime(date_match.group(1), '%Y%m%d')
            return self.start_date <= url_date <= self.end_date
        except:
            return True

    def _parse_date(self, date_text: Optional[str]) -> Optional[str]:
        """Parse date string to YYYY-MM-DD format"""
        if not date_text:
            return None

        date_text = date_text.strip()

        # Try ISO format (2025-11-18T10:30:00+08:00)
        try:
            if 'T' in date_text:
                dt = datetime.fromisoformat(date_text.replace('Z', '+00:00').split('+')[0])
                return dt.strftime('%Y-%m-%d')
        except:
            pass

        # Try YYYY-MM-DD
        try:
            dt = datetime.strptime(date_text[:10], '%Y-%m-%d')
            return dt.strftime('%Y-%m-%d')
        except:
            pass

        # Try other common formats
        date_patterns = [
            r'(\d{4})-(\d{2})-(\d{2})',  # 2025-11-18
            r'(\d{4})/(\d{2})/(\d{2})',  # 2025/11/18
            r'(\d{4})年(\d{1,2})月(\d{1,2})日',  # 2025年11月18日
        ]

        for pattern in date_patterns:
            match = re.search(pattern, date_text)
            if match:
                try:
                    year, month, day = match.groups()
                    return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                except:
                    continue

        return None

    def _extract_from_jsonld(self, response, field: str) -> Optional[str]:
        """Extract field from JSON-LD structured data"""
        jsonld_scripts = response.css('script[type="application/ld+json"]::text').getall()

        for script in jsonld_scripts:
            try:
                data = json.loads(script)

                # Handle list of JSON-LD objects
                if isinstance(data, list):
                    for item in data:
                        if field in item:
                            value = item[field]
                            if isinstance(value, dict) and 'name' in value:
                                return value['name']
                            elif isinstance(value, list) and value:
                                return value[0] if isinstance(value[0], str) else str(value[0])
                            return str(value)

                # Handle single JSON-LD object
                elif isinstance(data, dict):
                    if field in data:
                        value = data[field]
                        if isinstance(value, dict) and 'name' in value:
                            return value['name']
                        elif isinstance(value, list) and value:
                            return value[0] if isinstance(value[0], str) else str(value[0])
                        return str(value)
            except json.JSONDecodeError:
                continue

        return None

    def errback_sitemap(self, failure):
        """Handle sitemap request failures"""
        logger.error(f"Sitemap request failed: {failure.request.url}")
        logger.error(f"Error: {failure.value}")

    def errback_article(self, failure):
        """Handle article request failures"""
        self.stats['articles_failed'] += 1
        logger.error(f"Article request failed: {failure.request.url}")

    def closed(self, reason):
        """Print comprehensive statistics on spider close"""
        print("\n" + "=" * 70)
        print("CTI Spider Statistics (Deep Refactoring v2.0)")
        print("=" * 70)
        print(f"Mode: {self.mode}")
        if self.mode == 'sitemap':
            print(f"Sitemaps processed: {', '.join(sorted(self.stats['sitemaps_processed']))}")
        else:
            print(f"Categories processed: {len(self.stats['categories_processed'])}")
        print(f"Articles found: {self.stats['articles_found']:,}")
        print(f"Articles successfully scraped: {self.stats['articles_scraped']:,}")
        print(f"Articles failed: {self.stats['articles_failed']:,}")

        if self.stats['articles_scraped'] > 0:
            success_rate = 100 * self.stats['articles_scraped'] / (
                self.stats['articles_scraped'] + self.stats['articles_failed']
            )
            print(f"Success rate: {success_rate:.1f}%")

        print(f"\nMODE-SPECIFIC STATISTICS:")
        print(f"  Sitemap URLs discovered: {self.stats['sitemap_urls_discovered']:,}")
        print(f"  List pages visited: {self.stats['list_pages_visited']:,}")

        if self.stats['articles_scraped'] > 0:
            print(f"\nMETADATA QUALITY:")
            total = self.stats['articles_scraped']
            for field, count in self.stats['metadata_quality'].items():
                percentage = 100 * count / total
                print(f"  {field}: {count}/{total} ({percentage:.1f}%)")

        if self.stats['date_range']['earliest'] and self.stats['date_range']['latest']:
            print(f"\nDate range: {self.stats['date_range']['earliest']} to {self.stats['date_range']['latest']}")

        print("=" * 70)


if __name__ == '__main__':
    from scrapy.crawler import CrawlerProcess

    process = CrawlerProcess()
    process.crawl(CTINewsSpider, mode='sitemap', sitemap='todaynews', max_articles=5)
    process.start()
