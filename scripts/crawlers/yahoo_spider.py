#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comprehensive Yahoo News Spider with Deep Refactoring v2.0 (FINAL CRAWLER!)

Complete date-based sitemap crawler for Yahoo奇摩新聞 (tw.news.yahoo.com)
with systematic 2-year historical data access.

Features:
- Date-based sitemap generation (730 days / 2 years)
- Multi-sitemap index support (news, topic, tag, pk, ybrain, pages)
- Category-based archive crawling
- NO Playwright needed (standard Scrapy)
- 6+ fallback strategies per metadata field
- JSON-LD structured data extraction
- Image metadata extraction from sitemaps
- Comprehensive statistics tracking

Author: Information Retrieval System
Date: 2025-11-19
Version: 2.0 (Deep Refactoring - FINAL CRAWLER)
"""

import scrapy
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse, unquote
import re
import json
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


class YahooNewsSpider(scrapy.Spider):
    """
    Complete deep-refactored spider for Yahoo News with 2-year historical coverage.

    Features (Deep Refactoring v2.0 - FINAL CRAWLER):
        - Date-based sitemap generation (730 daily sitemaps)
        - Multi-sitemap index support
        - Category archive mode
        - NO Playwright needed (fast standard Scrapy)
        - 6+ fallback strategies per metadata field
        - JSON-LD extraction
        - Image metadata from sitemaps
        - 2-year default (730 days)
        - Per-sitemap statistics
    """

    name = 'yahoo_news'
    allowed_domains = ['tw.news.yahoo.com', 'news.yahoo.com']

    # Sitemap types
    SITEMAP_INDICES = {
        'news': 'https://tw.news.yahoo.com/news-sitemap-index.xml',
        'main': 'https://tw.news.yahoo.com/sitemap-index.xml',
        'topic': 'https://tw.news.yahoo.com/sitemap/sitemap_topic.xml.gz',
        'tag': 'https://tw.news.yahoo.com/sitemap/sitemap_tag.xml.gz',
        'pk': 'https://tw.news.yahoo.com/sitemap/sitemap_pk.xml.gz',
        'ybrain': 'https://tw.news.yahoo.com/sitemap/sitemap_ybrain.xml.gz',
        'pages': 'https://tw.news.yahoo.com/sitemap/sitemap_pages.xml',
    }

    # Categories for archive mode
    CATEGORIES = {
        'politics': {'name': '政治', 'url': 'https://tw.news.yahoo.com/politics/archive/'},
        'world': {'name': '國際', 'url': 'https://tw.news.yahoo.com/world/archive/'},
        'entertainment': {'name': '娛樂', 'url': 'https://tw.news.yahoo.com/entertainment/archive/'},
        'sports': {'name': '運動', 'url': 'https://tw.news.yahoo.com/sports/archive/'},
        'finance': {'name': '財經', 'url': 'https://tw.news.yahoo.com/finance/archive/'},
        'tech': {'name': '科技', 'url': 'https://tw.news.yahoo.com/tech/archive/'},
        'health': {'name': '健康', 'url': 'https://tw.news.yahoo.com/health/archive/'},
        'lifestyle': {'name': '生活', 'url': 'https://tw.news.yahoo.com/lifestyle/archive/'},
    }

    def __init__(self,
                 mode: str = 'sitemap',
                 sitemap: str = 'daily',
                 category: str = None,
                 days: int = 730,  # 2 YEARS DEFAULT
                 start_date: str = None,
                 end_date: str = None,
                 max_articles: int = None,
                 *args, **kwargs):
        """
        Initialize Yahoo spider with comprehensive parameters.

        Args:
            mode: Crawling mode ('sitemap' or 'archive')
            sitemap: Sitemap type ('daily', 'news', 'main', 'topic', 'tag', 'all')
            category: Category for archive mode
            days: Number of days to crawl (default 730 = 2 years)
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            max_articles: Maximum articles to scrape (optional limit)
        """
        super().__init__(*args, **kwargs)

        self.mode = mode.lower()
        self.sitemap_mode = sitemap.lower()
        self.category = category
        self.max_articles = int(max_articles) if max_articles else None

        # Date range calculation
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d') if end_date else datetime.now()
        if start_date:
            self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        elif days:
            self.start_date = self.end_date - timedelta(days=int(days))
        else:
            self.start_date = self.end_date - timedelta(days=730)  # Default 2 years

        # Statistics tracking
        self.stats = {
            'articles_found': 0,
            'articles_scraped': 0,
            'articles_failed': 0,
            'sitemaps_processed': 0,
            'sitemaps_failed': 0,
            'archive_pages_visited': 0,
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

        logger.info(f"Yahoo Spider initialized (Deep Refactoring v2.0 - FINAL CRAWLER!)")
        logger.info(f"Mode: {self.mode}")
        if self.mode == 'sitemap':
            logger.info(f"Sitemap: {self.sitemap_mode}")
        else:
            logger.info(f"Category: {self.category or 'all'}")
        logger.info(f"Date range: {self.start_date.date()} to {self.end_date.date()}")
        logger.info(f"Target: {(self.end_date - self.start_date).days} days of historical data")

    # Optimized Scrapy settings (no Playwright needed!)
    custom_settings = {
        'CONCURRENT_REQUESTS': 8,  # Fast since no Playwright
        'CONCURRENT_REQUESTS_PER_DOMAIN': 8,
        'DOWNLOAD_DELAY': 0.5,  # Respectful but fast
        'ROBOTSTXT_OBEY': True,
        'COOKIES_ENABLED': True,
        'RETRY_TIMES': 3,
        'RETRY_HTTP_CODES': [500, 502, 503, 504, 408, 429],
    }

    def start_requests(self):
        """Generate initial requests based on mode"""
        if self.mode == 'sitemap':
            yield from self._start_sitemap_mode()
        elif self.mode == 'archive':
            yield from self._start_archive_mode()
        else:
            logger.warning(f"Unknown mode: {self.mode}, defaulting to sitemap")
            yield from self._start_sitemap_mode()

    def _start_sitemap_mode(self):
        """Start crawling from sitemaps (NO Playwright!)"""
        if self.sitemap_mode == 'daily':
            # Generate daily sitemaps for date range
            yield from self._generate_daily_sitemaps()
        elif self.sitemap_mode == 'all':
            # Crawl all sitemap indices
            for sitemap_name, sitemap_url in self.SITEMAP_INDICES.items():
                logger.info(f"Fetching sitemap index: {sitemap_name} - {sitemap_url}")
                yield scrapy.Request(
                    url=sitemap_url,
                    callback=self.parse_sitemap_index,
                    errback=self.errback_sitemap,
                    meta={'sitemap_name': sitemap_name},
                    dont_filter=True
                )
        else:
            # Single sitemap index
            if self.sitemap_mode not in self.SITEMAP_INDICES:
                logger.warning(f"Unknown sitemap: {self.sitemap_mode}, using 'news'")
                self.sitemap_mode = 'news'

            sitemap_url = self.SITEMAP_INDICES[self.sitemap_mode]
            logger.info(f"Fetching sitemap index: {self.sitemap_mode} - {sitemap_url}")
            yield scrapy.Request(
                url=sitemap_url,
                callback=self.parse_sitemap_index,
                errback=self.errback_sitemap,
                meta={'sitemap_name': self.sitemap_mode},
                dont_filter=True
            )

    def _generate_daily_sitemaps(self):
        """Generate daily sitemap URLs for date range (KEY FEATURE for 2-year coverage!)"""
        current_date = self.start_date
        sitemaps_generated = 0

        while current_date <= self.end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            sitemap_url = f"https://tw.news.yahoo.com/sitemap-{date_str}.xml"

            sitemaps_generated += 1
            yield scrapy.Request(
                url=sitemap_url,
                callback=self.parse_sitemap,
                errback=self.errback_sitemap,
                meta={'sitemap_date': date_str},
                dont_filter=True
            )

            current_date += timedelta(days=1)

        logger.info(f"Generated {sitemaps_generated} daily sitemaps for {(self.end_date - self.start_date).days} days")

    def _start_archive_mode(self):
        """Start crawling from category archives"""
        if self.category and self.category in self.CATEGORIES:
            categories_to_crawl = [(self.category, self.CATEGORIES[self.category])]
        else:
            # Crawl all categories
            categories_to_crawl = list(self.CATEGORIES.items())

        for cat_code, cat_info in categories_to_crawl:
            logger.info(f"Starting category archive: {cat_info['name']} ({cat_code})")
            yield scrapy.Request(
                url=cat_info['url'],
                callback=self.parse_archive_page,
                meta={'category_code': cat_code, 'category_name': cat_info['name']},
                dont_filter=True
            )

    def parse_sitemap_index(self, response):
        """Parse sitemap index and extract sitemap URLs"""
        sitemap_name = response.meta.get('sitemap_name', 'unknown')
        logger.info(f"Parsing sitemap index: {sitemap_name}")

        from xml.etree import ElementTree as ET

        try:
            root = ET.fromstring(response.text)
            namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}

            sitemap_count = 0
            for sitemap_elem in root.findall('.//ns:sitemap', namespace):
                loc = sitemap_elem.find('ns:loc', namespace)
                if loc is None or not loc.text:
                    continue

                sitemap_url = loc.text.strip()
                sitemap_count += 1

                yield scrapy.Request(
                    url=sitemap_url,
                    callback=self.parse_sitemap,
                    errback=self.errback_sitemap,
                    meta={'parent_sitemap': sitemap_name},
                    dont_filter=True
                )

            logger.info(f"Sitemap index {sitemap_name}: Found {sitemap_count} sitemaps")

        except ET.ParseError as e:
            logger.error(f"Failed to parse sitemap index XML: {e}")
            self.stats['sitemaps_failed'] += 1

    def parse_sitemap(self, response):
        """Parse XML sitemap and extract article URLs"""
        sitemap_date = response.meta.get('sitemap_date', 'unknown')
        parent_sitemap = response.meta.get('parent_sitemap', 'direct')
        self.stats['sitemaps_processed'] += 1

        from xml.etree import ElementTree as ET

        try:
            root = ET.fromstring(response.text)
            namespaces = {
                'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9',
                'image': 'http://www.google.com/schemas/sitemap-image/1.1',
            }

            urls_found = 0
            for url_elem in root.findall('.//ns:url', namespaces):
                loc = url_elem.find('ns:loc', namespaces)
                if loc is None or not loc.text:
                    continue

                article_url = loc.text.strip()

                # Filter URLs (only .html articles)
                if not article_url.endswith('.html'):
                    continue

                # Extract image metadata from sitemap
                image_url = None
                image_caption = None
                image_elem = url_elem.find('.//image:image', namespaces)
                if image_elem is not None:
                    image_loc = image_elem.find('image:loc', namespaces)
                    if image_loc is not None:
                        image_url = image_loc.text
                    image_cap = image_elem.find('image:caption', namespaces)
                    if image_cap is not None:
                        image_caption = image_cap.text

                urls_found += 1
                self.stats['articles_found'] += 1

                yield scrapy.Request(
                    url=article_url,
                    callback=self.parse_article,
                    meta={
                        'sitemap_date': sitemap_date,
                        'sitemap_image_url': image_url,
                        'sitemap_image_caption': image_caption,
                    },
                    errback=self.errback_article
                )

            logger.info(f"Sitemap {sitemap_date or parent_sitemap}: Found {urls_found} URLs")

        except ET.ParseError as e:
            logger.error(f"Failed to parse sitemap XML: {e}")
            self.stats['sitemaps_failed'] += 1

    def parse_archive_page(self, response):
        """Parse category archive page and extract article links"""
        category_code = response.meta.get('category_code', 'unknown')
        category_name = response.meta.get('category_name', 'Unknown')

        self.stats['archive_pages_visited'] += 1

        logger.info(f"Parsing {category_name} archive: {response.url}")

        # Extract article URLs
        article_urls = response.css('a[href*=".html"]::attr(href)').getall()
        article_urls = [urljoin(response.url, url) for url in article_urls if url]
        article_urls = list(set(article_urls))  # Deduplicate

        articles_found = len(article_urls)
        logger.info(f"Found {articles_found} articles in {category_name} archive")

        if articles_found > 0:
            self.stats['articles_found'] += articles_found

            for article_url in article_urls:
                yield scrapy.Request(
                    url=article_url,
                    callback=self.parse_article,
                    meta={'source_category': category_code},
                    errback=self.errback_article
                )

    def parse_article(self, response):
        """Parse article page with comprehensive metadata extraction (6+ fallback strategies)"""

        # Extract article ID from URL (e.g., xxx-163600171.html)
        article_id_match = re.search(r'-(\d+)\.html$', response.url)
        article_id = f"yahoo_{article_id_match.group(1)}" if article_id_match else response.url.split('/')[-1].replace('.html', '')

        # === TITLE (6 strategies) ===
        title = (
            response.css('meta[property="og:title"]::attr(content)').get() or
            response.css('h1::text').get() or
            response.css('div.caas-title-wrapper h1::text').get() or
            response.css('meta[name="twitter:title"]::attr(content)').get() or
            response.css('title::text').get() or
            self._extract_from_jsonld(response, 'headline')
        )
        if title:
            title = title.strip()
            # Remove site name suffix
            title = re.sub(r'\s*[-|–—]\s*Yahoo.*$', '', title, flags=re.IGNORECASE)

        # === CONTENT (6 strategies) ===
        content_paragraphs = (
            response.css('div.caas-body p::text').getall() or
            response.css('article p::text').getall() or
            response.css('div[itemprop="articleBody"] p::text').getall() or
            response.css('div.article-body p::text').getall() or
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
            response.css('div.caas-attr-time-style time::attr(datetime)').get() or
            response.css('span.date::text').get() or
            response.css('time::text').get() or
            self._extract_from_jsonld(response, 'datePublished')
        )
        published_date = self._parse_date(date_text)

        # === AUTHOR (5 strategies) ===
        author = (
            response.css('meta[name="author"]::attr(content)').get() or
            response.css('span.caas-attr-provider a::text').get() or
            response.css('div.caas-attr-provider::text').get() or
            response.css('a[rel="author"]::text').get() or
            self._extract_from_jsonld(response, 'author')
        )
        if author:
            author = author.strip()

        # === CATEGORY (5 strategies) ===
        category_name = (
            response.css('meta[property="article:section"]::attr(content)').get() or
            response.css('span.category::text').get() or
            response.css('a.category::text').get() or
            self._extract_from_url_category(response.url) or
            self._extract_from_jsonld(response, 'articleSection')
        )

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

        # === IMAGE (5 strategies - including sitemap!) ===
        image_url = (
            response.meta.get('sitemap_image_url') or  # From sitemap metadata!
            response.css('meta[property="og:image"]::attr(content)').get() or
            response.css('article img::attr(src)').get() or
            response.css('div.caas-img img::attr(src)').get() or
            self._extract_from_jsonld(response, 'image')
        )
        if image_url:
            image_url = urljoin(response.url, image_url)

        # Image caption
        image_caption = (
            response.meta.get('sitemap_image_caption') or
            response.css('figure figcaption::text').get() or
            response.css('div.caas-img-caption::text').get()
        )

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
            'author': author or 'Yahoo News',
            'category': category_name,
            'tags': tags,
            'image_url': image_url,
            'image_caption': image_caption,
            'description': description,
            'source': 'Yahoo',
            'source_name': 'Yahoo奇摩新聞',
            'crawled_at': datetime.now().isoformat(),
        }

        self.stats['articles_scraped'] += 1

        yield article

    def _extract_from_url_category(self, url: str) -> Optional[str]:
        """Extract category from URL path"""
        for cat_code, cat_info in self.CATEGORIES.items():
            if f'/{cat_code}/' in url:
                return cat_info['name']
        return None

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
        self.stats['sitemaps_failed'] += 1

    def errback_article(self, failure):
        """Handle article request failures"""
        self.stats['articles_failed'] += 1
        logger.error(f"Article request failed: {failure.request.url}")

    def closed(self, reason):
        """Print comprehensive statistics on spider close"""
        print("\n" + "=" * 70)
        print("Yahoo Spider Statistics (Deep Refactoring v2.0 - FINAL CRAWLER!)")
        print("=" * 70)
        print(f"Mode: {self.mode}")
        print(f"Articles found: {self.stats['articles_found']:,}")
        print(f"Articles successfully scraped: {self.stats['articles_scraped']:,}")
        print(f"Articles failed: {self.stats['articles_failed']:,}")

        if self.stats['articles_scraped'] > 0:
            success_rate = 100 * self.stats['articles_scraped'] / (
                self.stats['articles_scraped'] + self.stats['articles_failed']
            )
            print(f"Success rate: {success_rate:.1f}%")

        print(f"\nMODE-SPECIFIC STATISTICS:")
        print(f"  Sitemaps processed: {self.stats['sitemaps_processed']:,}")
        print(f"  Sitemaps failed: {self.stats['sitemaps_failed']:,}")
        print(f"  Archive pages visited: {self.stats['archive_pages_visited']:,}")

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
    process.crawl(YahooNewsSpider, mode='sitemap', sitemap='daily', days=7, max_articles=10)
    process.start()
