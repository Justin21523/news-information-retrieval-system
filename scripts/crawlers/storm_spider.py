#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Storm Media Spider (風傳媒) - Deep Refactoring v2.0

Comprehensive multi-mode crawler for Storm Media with Playwright optimization.
Deep refactoring with comprehensive metadata extraction and robust error handling.

Target: https://www.storm.mg/
Sitemap: https://www.storm.mg/sitemap/news (~476 articles)

Discovery (2025-11-18 - Deep Analysis):
    - Sitemap available with recent articles
    - Extremely sparse sequential ID space (0.01% fill rate)
    - Requires Playwright for JavaScript rendering
    - ID range: 5,442,108 → 11,082,041 (5.6M span)
    - 9 categories with Chinese/English mapping

URL Structure:
    Article: https://www.storm.mg/article/{article_id}
    Category: https://www.storm.mg/category/{category_id}

Categories:
    politics (政治), finance (財經), lifestyle (生活), evaluation (評論),
    international (國際), cross_strait (兩岸), video (影音),
    long_form (長文), people (人物)

Features (Deep Refactoring):
    - 4-mode strategy (sitemap/list/sequential/hybrid)
    - Comprehensive metadata extraction (6+ fallback strategies per field)
    - Optimized Playwright usage with anti-detection
    - Smart 404 handling for sparse ID space
    - Detailed statistics tracking
    - JSON-LD structured data extraction

Author: Information Retrieval System
Date: 2025-11-18
Version: 2.0 (Deep Refactoring)
"""

import scrapy
from datetime import datetime, timedelta
import json
import logging
import re
import hashlib
import xml.etree.ElementTree as ET
from typing import Optional, List, Dict, Any
from urllib.parse import urljoin

# Set twisted reactor BEFORE importing any scrapy modules that might install it
import os
os.environ.setdefault('TWISTED_REACTOR', 'twisted.internet.asyncioreactor.AsyncioSelectorReactor')

# Always import PageMethod for Playwright requests
from scrapy_playwright.page import PageMethod

# Import base spider
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.crawlers.base_playwright_spider import BasePlaywrightSpider

logger = logging.getLogger(__name__)


class StormMediaSpider(BasePlaywrightSpider):
    """
    Comprehensive multi-mode spider for Storm Media (風傳媒).

    Features (Deep Refactoring v2.0):
        - 4-mode crawling strategy (sitemap/list/sequential/hybrid)
        - Comprehensive metadata extraction with 6+ fallback strategies
        - Optimized Playwright with anti-detection
        - Smart 404 handling for extremely sparse ID space
        - JSON-LD structured data extraction
        - Detailed per-mode statistics tracking

    Usage:
        # Sitemap mode (recommended for recent articles)
        scrapy runspider storm_spider.py -a mode=sitemap -o storm.jsonl

        # List mode (category browsing)
        scrapy runspider storm_spider.py -a mode=list -a category=politics -a days=7

        # Sequential mode (historical, very sparse)
        scrapy runspider storm_spider.py -a mode=sequential -a start_id=11080000 -a end_id=11082000

        # Hybrid mode (sitemap + list)
        scrapy runspider storm_spider.py -a mode=hybrid -a days=7

    Parameters:
        mode (str): Crawling mode - 'sitemap', 'list', 'sequential', 'hybrid'
        category (str): News category (for list mode)
        days (int): Number of days to crawl (for list/hybrid modes)
        start_id (int): Start article ID (for sequential mode)
        end_id (int): End article ID (for sequential mode)
        max_articles (int): Maximum articles to scrape (for testing)

    Complexity:
        - Sitemap mode: O(S) where S = sitemap size (~476)
        - List mode: O(D * A) where D = days, A = articles per day
        - Sequential mode: O(E - S) where E = end_id, S = start_id (very sparse)
        - Hybrid mode: O(S + D * A)
    """

    name = 'storm_media'
    allowed_domains = ['storm.mg', 'www.storm.mg', 'apis.storm.mg']

    # Category mapping (English → Chinese)
    CATEGORIES = {
        'all': {'slug': '', 'id': 0, 'name': '全部'},
        'politics': {'slug': 'politics', 'id': 118549, 'name': '政治'},
        'finance': {'slug': 'finance', 'id': 118550, 'name': '財經'},
        'lifestyle': {'slug': 'lifestyle', 'id': 118551, 'name': '生活'},
        'evaluation': {'slug': 'evaluation', 'id': 118552, 'name': '評論'},
        'international': {'slug': 'international', 'id': 118553, 'name': '國際'},
        'cross_strait': {'slug': 'cross-strait', 'id': 118554, 'name': '兩岸'},
        'video': {'slug': 'video', 'id': 123021, 'name': '影音'},
        'long_form': {'slug': 'long-form-article', 'id': 150485, 'name': '長文'},
        'people': {'slug': 'people', 'id': 151127, 'name': '人物'},
    }

    # Sitemap URL
    SITEMAP_URL = 'https://www.storm.mg/sitemap/news'

    # Known ID range (from sitemap analysis)
    EARLIEST_ID = 5442108  # Oldest in sitemap
    LATEST_ID_ESTIMATE = 11082041  # Latest observed

    # Custom settings with Playwright optimization
    custom_settings = {
        'TWISTED_REACTOR': 'twisted.internet.asyncioreactor.AsyncioSelectorReactor',
        'DOWNLOAD_DELAY': 2,  # Respectful delay for Playwright
        'CONCURRENT_REQUESTS_PER_DOMAIN': 2,  # Limited concurrency
        'CONCURRENT_REQUESTS': 4,
        'ROBOTSTXT_OBEY': True,
        'RETRY_TIMES': 2,
        'RETRY_HTTP_CODES': [500, 502, 503, 504, 408, 429],
        'HTTPERROR_ALLOW_404': True,  # Expected for sparse sequential

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
                '--disable-gpu',
            ]
        },
        'PLAYWRIGHT_DEFAULT_NAVIGATION_TIMEOUT': 60000,

        # Output settings
        'FEEDS': {
            'data/raw/storm_media_%(time)s.jsonl': {
                'format': 'jsonlines',
                'encoding': 'utf8',
                'store_empty': False,
                'overwrite': False,
                'indent': 2,
            }
        },

        'LOG_LEVEL': 'INFO',
    }

    def __init__(self,
                 mode: str = 'sitemap',
                 category: str = 'all',
                 days: int = 7,
                 start_id: int = None,
                 end_id: int = None,
                 start_date: str = None,
                 end_date: str = None,
                 max_articles: int = None,
                 *args, **kwargs):
        """
        Initialize Storm Media spider with comprehensive mode support.

        Args:
            mode: Crawling mode - 'sitemap', 'list', 'sequential', 'hybrid'
            category: News category
            days: Number of days to crawl
            start_id: Start article ID (sequential mode)
            end_id: End article ID (sequential mode)
            start_date: Start date YYYY-MM-DD
            end_date: End date YYYY-MM-DD
            max_articles: Maximum articles to scrape (testing)
        """
        super().__init__(*args, **kwargs)

        # Mode selection
        self.mode = mode.lower()
        if self.mode not in ['sitemap', 'list', 'sequential', 'hybrid']:
            logger.warning(f"Invalid mode '{mode}', using 'sitemap'")
            self.mode = 'sitemap'

        # Category
        if category not in self.CATEGORIES:
            logger.warning(f"Invalid category '{category}', using 'all'")
            category = 'all'
        self.category = category
        self.category_slug = self.CATEGORIES[category]['slug']
        self.category_id = self.CATEGORIES[category]['id']
        self.category_name = self.CATEGORIES[category]['name']

        # Date range
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

        # Sequential ID range
        self.start_id = int(start_id) if start_id else self.LATEST_ID_ESTIMATE - 1000
        self.end_id = int(end_id) if end_id else self.LATEST_ID_ESTIMATE

        # Article limit (for testing)
        self.max_articles = int(max_articles) if max_articles else None

        # Statistics
        self.articles_count = 0
        self.failed_count = 0
        self.not_found_count = 0  # 404 count
        self.skipped_date_count = 0
        self.pages_visited = 0
        self.seen_urls = set()

        # Mode-specific stats
        self.mode_stats = {
            'sitemap_urls': 0,
            'list_pages': 0,
            'sequential_tried': 0,
        }

        # Initialization log
        logger.info("=" * 70)
        logger.info("Storm Media Spider Initialized (Deep Refactoring v2.0)")
        logger.info("=" * 70)
        logger.info(f"Mode: {self.mode}")
        logger.info(f"Category: {self.category_name} ({self.category})")
        logger.info(f"Date range: {self.start_date.date()} to {self.end_date.date()}")
        if self.mode == 'sequential':
            logger.info(f"ID range: {self.start_id:,} to {self.end_id:,}")
            logger.info(f"Note: Expect >99% 404 rate due to sparse ID space")
        if self.max_articles:
            logger.info(f"Article limit: {self.max_articles}")
        logger.info("=" * 70)

    def start_requests(self):
        """
        Generate start requests based on crawling mode.

        Modes:
            - sitemap: Parse XML sitemap for article URLs
            - list: Browse category list pages
            - sequential: Try sequential article IDs
            - hybrid: Combination of sitemap + list
        """
        if self.mode == 'sitemap':
            # Sitemap mode: Parse XML sitemap
            logger.info(f"Starting sitemap mode: {self.SITEMAP_URL}")
            yield scrapy.Request(
                url=self.SITEMAP_URL,
                callback=self.parse_sitemap,
                errback=self.handle_error,
                dont_filter=True,
                # No Playwright needed for sitemap XML
                meta={'playwright': False}
            )

        elif self.mode == 'list':
            # List mode: Browse category pages
            if self.category == 'all':
                base_url = "https://www.storm.mg/articles"
            else:
                base_url = f"https://www.storm.mg/category/{self.category_id}"

            logger.info(f"Starting list mode: {base_url}")
            yield scrapy.Request(
                url=base_url,
                callback=self.parse_list_page,
                errback=self.handle_error,
                dont_filter=True,
                meta=self.get_playwright_meta(wait_selector='div.card_info')
            )

        elif self.mode == 'sequential':
            # Sequential mode: Try article IDs in range
            logger.info(f"Starting sequential mode: {self.start_id:,} to {self.end_id:,}")
            for request in self._generate_sequential_requests():
                yield request

        elif self.mode == 'hybrid':
            # Hybrid mode: Sitemap + List
            logger.info("Starting hybrid mode: sitemap + list")

            # First: Sitemap
            yield scrapy.Request(
                url=self.SITEMAP_URL,
                callback=self.parse_sitemap,
                errback=self.handle_error,
                dont_filter=True,
                meta={'playwright': False}
            )

            # Then: List pages
            if self.category == 'all':
                base_url = "https://www.storm.mg/articles"
            else:
                base_url = f"https://www.storm.mg/category/{self.category_id}"

            yield scrapy.Request(
                url=base_url,
                callback=self.parse_list_page,
                errback=self.handle_error,
                dont_filter=True,
                meta=self.get_playwright_meta(wait_selector='div.card_info')
            )

    def parse_sitemap(self, response):
        """
        Parse XML sitemap to extract article URLs.

        Storm Media sitemap structure:
            <url>
                <loc>https://www.storm.mg/article/{article_id}</loc>
            </url>
        """
        logger.info(f"Parsing sitemap: {response.url}")

        try:
            # Parse XML
            root = ET.fromstring(response.text)
            ns = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}

            # Extract all URLs
            urls = []
            for url_elem in root.findall('ns:url', ns):
                loc = url_elem.find('ns:loc', ns)
                if loc is not None and loc.text and '/article/' in loc.text:
                    urls.append(loc.text)

            logger.info(f"Found {len(urls)} articles in sitemap")
            self.mode_stats['sitemap_urls'] = len(urls)

            # Generate requests for each article
            for url in urls:
                # Check if max_articles limit reached
                if self.max_articles and self.articles_count >= self.max_articles:
                    logger.info(f"Reached max_articles limit: {self.max_articles}")
                    break

                # Skip if already seen
                if url in self.seen_urls:
                    continue
                self.seen_urls.add(url)

                yield scrapy.Request(
                    url=url,
                    callback=self.parse_article,
                    errback=self.handle_error,
                    meta=self.get_playwright_meta(wait_selector='article.article_page')
                )

        except ET.ParseError as e:
            logger.error(f"XML parsing error: {e}")
        except Exception as e:
            logger.error(f"Error parsing sitemap: {e}")

    def parse_list_page(self, response):
        """
        Parse Storm Media list page to extract article URLs.

        Handles pagination and extracts article links from card elements.
        """
        self.pages_visited += 1
        self.mode_stats['list_pages'] += 1
        logger.info(f"Parsing list page #{self.pages_visited}: {response.url}")

        # Extract article links with multiple selectors
        article_selectors = response.css(
            'div.card_info h2 a, '
            'div.card_title a, '
            'h2.card_title a, '
            'a[href*="/article/"]'
        )

        articles_found = 0
        for article_sel in article_selectors:
            article_url = article_sel.css('::attr(href)').get()

            if not article_url:
                continue

            # Convert to absolute URL
            article_url = response.urljoin(article_url)

            # Filter: only article pages
            if not self._is_article_url(article_url):
                continue

            # Skip if already seen
            if article_url in self.seen_urls:
                continue
            self.seen_urls.add(article_url)

            # Check max_articles limit
            if self.max_articles and self.articles_count >= self.max_articles:
                logger.info(f"Reached max_articles limit: {self.max_articles}")
                return

            articles_found += 1

            # Request article detail
            yield scrapy.Request(
                url=article_url,
                callback=self.parse_article,
                errback=self.handle_error,
                meta=self.get_playwright_meta(wait_selector='article.article_page')
            )

        logger.info(f"Found {articles_found} articles on list page")

        # Pagination - limit to first 10 pages to avoid infinite scroll
        if articles_found > 0 and self.pages_visited < 10:
            next_page = response.css('a.load-more::attr(href), a.next::attr(href)').get()

            if next_page:
                next_url = response.urljoin(next_page)
                logger.info(f"Following pagination to: {next_url}")

                yield scrapy.Request(
                    url=next_url,
                    callback=self.parse_list_page,
                    errback=self.handle_error,
                    dont_filter=True,
                    meta=self.get_playwright_meta(wait_selector='div.card_info')
                )

    def _generate_sequential_requests(self):
        """
        Generate requests for sequential article ID crawling.

        Note: Storm Media has extremely sparse ID space (0.01% fill rate).
        Expect >99% 404 responses in this mode.
        """
        for article_id in range(self.start_id, self.end_id + 1):
            # Check max_articles limit
            if self.max_articles and self.articles_count >= self.max_articles:
                logger.info(f"Reached max_articles limit: {self.max_articles}")
                break

            self.mode_stats['sequential_tried'] += 1

            url = f"https://www.storm.mg/article/{article_id}"

            yield scrapy.Request(
                url=url,
                callback=self.parse_article,
                errback=self.handle_error,
                meta={
                    **self.get_playwright_meta(wait_selector='article.article_page'),
                    'article_id': article_id,
                    'mode': 'sequential'
                },
                dont_filter=True
            )

    def parse_article(self, response):
        """
        Parse individual Storm Media article with comprehensive metadata extraction.

        Extraction Strategies (Deep Refactoring):
        - Title: 6 fallback strategies (og:title, h1, JSON-LD, etc.)
        - Content: 5 fallback strategies (article p, JSON-LD, div content)
        - Date: 6 fallback strategies (time, meta, JSON-LD, etc.)
        - Author: 4 fallback strategies
        - Category: Breadcrumb, meta, current category
        - Tags: Multiple sources (div.article_tag, meta)
        - Image: og:image, article images, figure
        """
        # Handle 404 in sequential mode
        if response.status == 404:
            self.not_found_count += 1
            if response.meta.get('mode') == 'sequential':
                # Expected in sequential mode
                if self.not_found_count % 1000 == 0:
                    logger.info(f"404 count: {self.not_found_count:,} (expected in sparse sequential)")
            return

        try:
            logger.debug(f"Parsing article: {response.url}")

            # Initialize article structure
            article = {
                'article_id': self._extract_id_from_url(response.url),
                'url': response.url,
                'source': 'Storm Media',
                'source_name': '風傳媒',
                'crawled_at': datetime.now().isoformat(),
            }

            # === TITLE EXTRACTION (6 strategies) ===
            title = (
                # Strategy 1: og:title (most reliable)
                response.css('meta[property="og:title"]::attr(content)').get() or
                # Strategy 2: Article title class
                response.css('h1.article_title::text').get() or
                # Strategy 3: Standard h1
                response.css('h1::text').get() or
                # Strategy 4: Header h1
                response.css('header h1::text').get() or
                # Strategy 5: Twitter card title
                response.css('meta[name="twitter:title"]::attr(content)').get() or
                # Strategy 6: Page title
                response.css('title::text').get()
            )
            article['title'] = self._clean_text(title)

            # Try JSON-LD for title if not found
            if not article['title']:
                article['title'] = self._extract_from_jsonld(response, 'headline')

            # === CONTENT EXTRACTION (5 strategies) ===
            content_paragraphs = []

            # Strategy 1: JSON-LD articleBody
            jsonld_content = self._extract_from_jsonld(response, 'articleBody')
            if jsonld_content:
                content_paragraphs = [jsonld_content]
            else:
                # Strategy 2-5: CSS selectors
                content_selectors = [
                    'article.article_page p::text',
                    'div.article_content p::text',
                    'div.article-content p::text',
                    'div[class*="content"] p::text',
                    'main p::text',
                ]

                for selector in content_selectors:
                    paragraphs = response.css(selector).getall()
                    if paragraphs and len(paragraphs) > 2:  # At least 3 paragraphs
                        content_paragraphs = paragraphs
                        break

            article['content'] = ' '.join([
                self._clean_text(p) for p in content_paragraphs
                if p and len(p.strip()) > 20
            ])

            # === DATE EXTRACTION (6 strategies) ===
            date_text = (
                # Strategy 1: article:published_time
                response.css('meta[property="article:published_time"]::attr(content)').get() or
                # Strategy 2: time datetime
                response.css('time::attr(datetime)').get() or
                # Strategy 3: time text
                response.css('time::text').get() or
                # Strategy 4: info_time span
                response.css('span.info_time::text').get() or
                # Strategy 5: div.info_time
                response.css('div.info_time::text').get()
            )
            article['published_date'] = self._parse_date(date_text)

            # Strategy 6: JSON-LD
            if not article['published_date']:
                jsonld_date = self._extract_from_jsonld(response, 'datePublished')
                article['published_date'] = self._parse_date(jsonld_date)

            # Date filtering
            if article['published_date']:
                try:
                    pub_date_obj = datetime.strptime(article['published_date'], '%Y-%m-%d')
                    if not (self.start_date <= pub_date_obj <= self.end_date):
                        logger.debug(f"Article date {article['published_date']} out of range")
                        self.skipped_date_count += 1
                        return
                except:
                    pass

            # === AUTHOR EXTRACTION (4 strategies) ===
            author = (
                # Strategy 1: info_author span
                response.css('span.info_author::text').get() or
                # Strategy 2: div.info_author
                response.css('div.info_author::text').get() or
                # Strategy 3: a.info_author
                response.css('a.info_author::text').get() or
                # Strategy 4: meta author
                response.css('meta[name="author"]::attr(content)').get()
            )
            article['author'] = self._clean_text(author) if author else '風傳媒'

            # Try JSON-LD for author
            if article['author'] == '風傳媒':
                jsonld_author = self._extract_from_jsonld(response, 'author')
                if jsonld_author:
                    if isinstance(jsonld_author, dict):
                        article['author'] = jsonld_author.get('name', '風傳媒')
                    else:
                        article['author'] = str(jsonld_author)

            # === CATEGORY EXTRACTION ===
            # From breadcrumb
            breadcrumb = response.css('div.breadcrumb a::text, nav.breadcrumb a::text').getall()
            if breadcrumb and len(breadcrumb) > 1:
                article['category'] = breadcrumb[-1].strip()
                article['category_name'] = breadcrumb[-1].strip()
            else:
                # Use spider's category
                article['category'] = self.category
                article['category_name'] = self.category_name

            # === TAGS EXTRACTION ===
            tags = []

            # Source 1: article_tag div
            tag_elements = response.css('div.article_tag a::text, div.tags a::text').getall()
            tags.extend([self._clean_text(t) for t in tag_elements if t])

            # Source 2: rel="tag"
            rel_tags = response.css('a[rel="tag"]::text').getall()
            tags.extend([self._clean_text(t) for t in rel_tags if t])

            # Source 3: meta keywords
            keywords = response.css('meta[name="keywords"]::attr(content)').get()
            if keywords:
                tags.extend([k.strip() for k in keywords.split(',') if k.strip()])

            article['tags'] = list(set([t for t in tags if t and len(t) > 1]))

            # === IMAGE EXTRACTION ===
            images = []

            # Primary: og:image
            og_image = response.css('meta[property="og:image"]::attr(content)').get()
            if og_image:
                images.append(og_image)

            # Additional: article images
            image_selectors = [
                'figure.article_img img::attr(src)',
                'div.article_img img::attr(src)',
                'article img::attr(src)',
            ]
            for selector in image_selectors:
                img_urls = response.css(selector).getall()
                images.extend([response.urljoin(url) for url in img_urls if url])

            article['image_url'] = images[0] if images else None
            article['images'] = images[:5]  # Store up to 5 images

            # === DESCRIPTION ===
            description = (
                response.css('meta[property="og:description"]::attr(content)').get() or
                response.css('meta[name="description"]::attr(content)').get()
            )
            article['description'] = self._clean_text(description) if description else None

            # === VALIDATION ===
            validation_issues = []

            if not article['title']:
                validation_issues.append('no_title')

            if not article.get('content') or len(article['content']) < 100:
                validation_issues.append('insufficient_content')

            if validation_issues:
                logger.warning(f"Invalid article ({', '.join(validation_issues)}): {response.url}")
                self.failed_count += 1
                return

            # === SUCCESS ===
            self.articles_count += 1

            if self.articles_count % 10 == 0:
                logger.info(f"Progress: {self.articles_count} articles scraped")

            yield article

        except Exception as e:
            logger.error(f"Error parsing article {response.url}: {e}", exc_info=True)
            self.failed_count += 1

    def _extract_from_jsonld(self, response, field: str) -> Optional[str]:
        """Extract field from JSON-LD structured data."""
        try:
            jsonld_scripts = response.css('script[type="application/ld+json"]::text').getall()
            for script in jsonld_scripts:
                try:
                    data = json.loads(script)

                    # Handle array
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and field in item:
                                return item[field]
                    # Handle single object
                    elif isinstance(data, dict) and field in data:
                        return data[field]

                except json.JSONDecodeError:
                    continue

        except Exception as e:
            logger.debug(f"JSON-LD extraction failed for {field}: {e}")

        return None

    def handle_error(self, failure):
        """Handle request errors with detailed logging."""
        from scrapy.spidermiddlewares.httperror import HttpError
        from twisted.internet.error import DNSLookupError, TimeoutError, TCPTimedOutError

        if failure.check(HttpError):
            response = failure.value.response
            if response.status == 404:
                self.not_found_count += 1
                # Don't log every 404 in sequential mode
                if failure.request.meta.get('mode') != 'sequential':
                    logger.warning(f"HTTP 404: {failure.request.url}")
                return
            else:
                logger.error(f"HTTP {response.status}: {failure.request.url}")
        elif failure.check(DNSLookupError):
            logger.error(f"DNS lookup failed: {failure.request.url}")
        elif failure.check((TimeoutError, TCPTimedOutError)):
            logger.error(f"Timeout: {failure.request.url}")
        else:
            logger.error(f"Request failed ({failure.type.__name__}): {failure.request.url}")

        self.failed_count += 1

    def closed(self, reason):
        """Log comprehensive statistics on spider closure."""
        total_processed = self.articles_count + self.failed_count
        success_rate = (100 * self.articles_count / total_processed) if total_processed > 0 else 0

        logger.info("=" * 70)
        logger.info("Storm Media Spider Finished (Deep Refactoring v2.0)")
        logger.info("=" * 70)
        logger.info(f"Reason: {reason}")
        logger.info("")

        # Overall statistics
        logger.info("OVERALL STATISTICS:")
        logger.info(f"  Mode: {self.mode}")
        logger.info(f"  Articles successfully scraped: {self.articles_count}")
        logger.info(f"  Failed: {self.failed_count}")
        logger.info(f"  Not found (404): {self.not_found_count:,}")
        logger.info(f"  Skipped (date filter): {self.skipped_date_count}")
        logger.info(f"  Success rate: {success_rate:.2f}%")
        logger.info("")

        # Mode-specific stats
        logger.info("MODE-SPECIFIC STATISTICS:")
        if self.mode_stats['sitemap_urls'] > 0:
            logger.info(f"  Sitemap URLs discovered: {self.mode_stats['sitemap_urls']}")
        if self.mode_stats['list_pages'] > 0:
            logger.info(f"  List pages visited: {self.mode_stats['list_pages']}")
        if self.mode_stats['sequential_tried'] > 0:
            logger.info(f"  Sequential IDs tried: {self.mode_stats['sequential_tried']:,}")
            if self.not_found_count > 0:
                hit_rate = 100 * self.articles_count / (self.articles_count + self.not_found_count)
                logger.info(f"  Hit rate (excluding errors): {hit_rate:.4f}%")
        logger.info("")

        logger.info(f"Date range: {self.start_date.date()} to {self.end_date.date()}")
        logger.info(f"Category: {self.category_name}")
        logger.info("=" * 70)

    # ========== Utility Methods ==========

    def _is_article_url(self, url: str) -> bool:
        """Check if URL is a Storm Media article."""
        return bool(url and '/article/' in url and re.search(r'/article/\d+', url))

    def _extract_id_from_url(self, url: str) -> str:
        """Extract article ID from URL."""
        match = re.search(r'/article/(\d+)', url)
        if match:
            return f"STORM_{match.group(1)}"
        # Fallback: hash URL
        return f"STORM_{hashlib.md5(url.encode()).hexdigest()[:12]}"

    def _clean_text(self, text: Optional[str]) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        return text.strip()

    def _parse_date(self, date_text: Optional[str]) -> Optional[str]:
        """Parse date to YYYY-MM-DD format."""
        if not date_text:
            return None

        try:
            # ISO 8601
            if 'T' in date_text or '+' in date_text or 'Z' in date_text:
                dt = datetime.fromisoformat(date_text.replace('Z', '+00:00').split('+')[0])
                return dt.strftime('%Y-%m-%d')

            # Common patterns
            patterns = [
                r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})',  # YYYY-MM-DD or YYYY/MM/DD
                r'(\d{4})年(\d{1,2})月(\d{1,2})日',    # Chinese format
            ]

            for pattern in patterns:
                match = re.search(pattern, date_text)
                if match:
                    year, month, day = match.groups()
                    dt = datetime(int(year), int(month), int(day))
                    return dt.strftime('%Y-%m-%d')

        except Exception as e:
            logger.debug(f"Failed to parse date '{date_text}': {e}")

        return None


# Standalone execution
if __name__ == '__main__':
    # Install asyncio reactor for Playwright
    from twisted.internet import asyncioreactor
    asyncioreactor.install()

    from scrapy.crawler import CrawlerProcess

    process = CrawlerProcess({
        'LOG_LEVEL': 'INFO',
        'ROBOTSTXT_OBEY': True,
    })

    # Example usage - customize as needed:
    # mode='sitemap'     - Sitemap crawling (recommended)
    # mode='list'        - Category browsing
    # mode='sequential'  - Sequential ID (very sparse)
    # mode='hybrid'      - Sitemap + List
    # max_articles=50    - Limit for testing

    process.crawl(StormMediaSpider, mode='sitemap', max_articles=10)
    process.start()
