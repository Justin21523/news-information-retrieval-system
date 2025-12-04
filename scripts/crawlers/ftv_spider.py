#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FTV (民視新聞) Spider (Deep Refactoring - Complete Version)

Comprehensive Playwright-based spider for FTV News with deep pagination
and aggressive historical data crawling (targeting 2 years).

Target: https://www.ftvnews.com.tw/
Strategy: Deep pagination + All categories + Aggressive crawling

Key Features (Complete Version):
    - Deep pagination (unlimited pages for 2-year coverage)
    - All 8 categories traversal
    - Playwright optimization with anti-detection
    - 6+ fallback strategies per metadata field
    - JSON-LD structured data extraction
    - Aggressive date-based crawling
    - Per-category statistics tracking

Author: Information Retrieval System
Date: 2025-11-19
Version: 2.0 (Complete Deep Refactoring)
"""

import scrapy
from datetime import datetime, timedelta
import json
import logging
import re
import hashlib
from typing import Optional

# Set twisted reactor BEFORE importing any scrapy modules that might install it
import os
os.environ.setdefault('TWISTED_REACTOR', 'twisted.internet.asyncioreactor.AsyncioSelectorReactor')

from base_playwright_spider import BasePlaywrightSpider, PlaywrightPageMethods
from scrapy_playwright.page import PageMethod

logger = logging.getLogger(__name__)


class FTVNewsSpider(BasePlaywrightSpider):
    """
    Complete deep-refactored spider for FTV News with 2-year historical coverage.
    """

    name = 'ftv_news'
    allowed_domains = ['ftvnews.com.tw', 'www.ftvnews.com.tw']

    CATEGORIES = {
        'all': {'slug': '', 'name': '全部'},
        'politics': {'slug': 'politics', 'name': '政治'},
        'business': {'slug': 'business', 'name': '財經'},
        'society': {'slug': 'society', 'name': '社會'},
        'entertainment': {'slug': 'entertainment', 'name': '娛樂'},
        'life': {'slug': 'life', 'name': '生活'},
        'sports': {'slug': 'sports', 'name': '運動'},
        'tech': {'slug': 'tech', 'name': '科技'},
        'world': {'slug': 'world', 'name': '國際'},
    }

    custom_settings = {
        'TWISTED_REACTOR': 'twisted.internet.asyncioreactor.AsyncioSelectorReactor',
        'DOWNLOAD_DELAY': 2,
        'CONCURRENT_REQUESTS_PER_DOMAIN': 2,
        'ROBOTSTXT_OBEY': False,
        'DOWNLOAD_HANDLERS': {
            "http": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
            "https": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
        },
        'PLAYWRIGHT_BROWSER_TYPE': 'chromium',
        'PLAYWRIGHT_LAUNCH_OPTIONS': {'headless': True, 'timeout': 60000},
        'FEEDS': {'data/raw/ftv_news_%(time)s.jsonl': {'format': 'jsonlines', 'encoding': 'utf8'}},
        'LOG_LEVEL': 'INFO',
    }

    def __init__(self, category: str = 'all', days: int = 730,  # 2 years default
                 max_pages: int = 500, max_articles: int = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if category not in self.CATEGORIES:
            category = 'all'

        self.category = category
        self.category_slug = self.CATEGORIES[category]['slug']
        self.category_name = self.CATEGORIES[category]['name']

        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=int(days))
        self.max_pages = int(max_pages)
        self.max_articles = int(max_articles) if max_articles else None

        self.articles_scraped = 0
        self.articles_failed = 0
        self.pages_visited = 0
        self.seen_urls = set()

        self.category_stats = {}

        logger.info("=" * 70)
        logger.info("FTV Spider Initialized (Complete Deep Refactoring)")
        logger.info("=" * 70)
        logger.info(f"Category: {self.category_name}")
        logger.info(f"Date Range: {self.start_date.date()} to {self.end_date.date()} ({days} days)")
        logger.info(f"Max Pages: {self.max_pages}")
        if self.max_articles:
            logger.info(f"Max Articles: {self.max_articles:,}")
        logger.info("=" * 70)

    def start_requests(self):
        """Generate start requests for all categories (if 'all') or specific category."""
        if self.category == 'all':
            # Crawl all categories for maximum coverage
            for cat_code, cat_info in self.CATEGORIES.items():
                if cat_code == 'all':
                    continue

                self.category_stats[cat_code] = {'pages': 0, 'articles': 0}
                base_url = f"https://www.ftvnews.com.tw/category/{cat_info['slug']}"

                logger.info(f"Queuing category: {cat_info['name']} - {base_url}")

                yield scrapy.Request(
                    url=base_url,
                    callback=self.parse_list_page,
                    errback=self.handle_error,
                    meta={'category': cat_code, 'page': 1, **self.get_playwright_meta(
                        playwright_page_methods=[
                            PageMethod('wait_for_load_state', 'networkidle'),
                            PlaywrightPageMethods.wait_for_timeout(2000),
                        ]
                    )},
                    dont_filter=True,
                )
        else:
            self.category_stats[self.category] = {'pages': 0, 'articles': 0}
            base_url = f"https://www.ftvnews.com.tw/category/{self.category_slug}"

            yield scrapy.Request(
                url=base_url,
                callback=self.parse_list_page,
                errback=self.handle_error,
                meta={'category': self.category, 'page': 1, **self.get_playwright_meta(
                    playwright_page_methods=[
                        PageMethod('wait_for_load_state', 'networkidle'),
                        PlaywrightPageMethods.wait_for_timeout(2000),
                    ]
                )},
                dont_filter=True,
            )

    def parse_list_page(self, response):
        """Parse FTV list page with deep pagination support."""
        category = response.meta.get('category', 'unknown')
        page = response.meta.get('page', 1)

        self.pages_visited += 1
        if category in self.category_stats:
            self.category_stats[category]['pages'] += 1

        logger.info(f"Parsing list page: {response.url} (Category: {category}, Page: {page})")

        # Extract article links (comprehensive selectors)
        article_selectors = response.css(
            'a[href*="/news/detail/"], '
            'a[href*="/detail/"], '
            'div.news-item a, '
            'div.article-item a, '
            'li.news-list a, '
            'div[class*="news"] a[href*="detail"]'
        )

        articles_found = 0
        for article_sel in article_selectors:
            article_url = article_sel.css('::attr(href)').get()

            if not article_url or 'detail' not in article_url:
                continue

            article_url = response.urljoin(article_url)

            if article_url in self.seen_urls:
                continue
            self.seen_urls.add(article_url)

            articles_found += 1

            yield scrapy.Request(
                url=article_url,
                callback=self.parse_article,
                errback=self.handle_error,
                meta={'category': category, **self.get_playwright_meta(
                    playwright_page_methods=[
                        PageMethod('wait_for_load_state', 'domcontentloaded'),
                        PlaywrightPageMethods.wait_for_timeout(1500),
                    ]
                )},
                dont_filter=False,
            )

            if self.max_articles and len(self.seen_urls) >= self.max_articles:
                logger.info(f"Reached max articles limit ({self.max_articles}), stopping")
                return

        logger.info(f"Found {articles_found} articles on page {page}")

        # AGGRESSIVE PAGINATION for 2-year coverage
        if articles_found > 0 and page < self.max_pages:
            # Try multiple pagination patterns
            next_page_urls = []

            # Pattern 1: ?page=X
            next_page_urls.append(f"{response.url.split('?')[0]}?page={page + 1}")

            # Pattern 2: Load more button
            load_more_url = response.css('a.load-more::attr(href), button[data-next]::attr(data-next)').get()
            if load_more_url:
                next_page_urls.append(response.urljoin(load_more_url))

            # Pattern 3: Next page link
            next_link = response.css('a[rel="next"]::attr(href), a.next::attr(href)').get()
            if next_link:
                next_page_urls.append(response.urljoin(next_link))

            for next_url in next_page_urls[:1]:  # Use first available pattern
                logger.info(f"Following pagination to page {page + 1}: {next_url}")

                yield scrapy.Request(
                    url=next_url,
                    callback=self.parse_list_page,
                    errback=self.handle_error,
                    meta={'category': category, 'page': page + 1, **self.get_playwright_meta(
                        playwright_page_methods=[
                            PageMethod('wait_for_load_state', 'networkidle'),
                            PlaywrightPageMethods.wait_for_timeout(2000),
                        ]
                    )},
                    dont_filter=True,
                )
                break

    def parse_article(self, response):
        """
        Parse FTV article with comprehensive metadata extraction (6+ strategies per field).
        """
        try:
            category = response.meta.get('category', 'unknown')

            article = {
                'article_id': self._generate_article_id(response.url),
                'url': response.url,
                'source': 'FTV',
                'source_name': '民視新聞',
                'crawled_at': datetime.now().isoformat(),
            }

            # === TITLE: 6 strategies ===
            title = (
                response.css('meta[property="og:title"]::attr(content)').get() or
                response.css('h1.news-title::text').get() or
                response.css('h1::text').get() or
                response.css('meta[name="twitter:title"]::attr(content)').get() or
                response.css('title::text').get()
            )
            if not title:
                title = self._extract_from_jsonld(response, 'headline')
            article['title'] = self._clean_text(title) if title else None

            # === CONTENT: 6 strategies ===
            jsonld_content = self._extract_from_jsonld(response, 'articleBody')
            if jsonld_content:
                content_paragraphs = [jsonld_content]
            else:
                content_selectors = [
                    'article p::text',
                    'div.article-content p::text',
                    'div.news-content p::text',
                    'div[class*="content"] p::text',
                    'main article p::text',
                    'div.article-body p::text',
                ]

                content_paragraphs = []
                for selector in content_selectors:
                    paragraphs = response.css(selector).getall()
                    if paragraphs and len(paragraphs) >= 2:
                        content_paragraphs = paragraphs
                        break

            article['content'] = ' '.join([self._clean_text(p) for p in content_paragraphs if p])

            # === DATE: 7 strategies ===
            date_text = (
                response.css('meta[property="article:published_time"]::attr(content)').get() or
                response.css('time::attr(datetime)').get() or
                response.css('meta[name="pubdate"]::attr(content)').get() or
                response.css('span.date::text').get() or
                response.css('div.publish-time::text').get() or
                response.css('time::text').get()
            )
            if not date_text:
                date_text = self._extract_from_jsonld(response, 'datePublished')

            article['published_date'] = self._parse_publish_date(date_text)

            # Date filtering
            if article['published_date']:
                try:
                    pub_date_obj = datetime.strptime(article['published_date'], '%Y-%m-%d')
                    if not (self.start_date <= pub_date_obj <= self.end_date):
                        logger.debug(f"Article date out of range, skipping")
                        return
                except ValueError:
                    pass

            # === AUTHOR: 5 strategies ===
            author = (
                response.css('meta[name="author"]::attr(content)').get() or
                response.css('span.author::text').get() or
                response.css('div.author::text').get() or
                response.css('a.author::text').get()
            )
            if not author:
                author = self._extract_from_jsonld(response, 'author')
            article['author'] = self._clean_text(author) if author else '民視新聞'

            # === CATEGORY ===
            article['category'] = category
            article['category_name'] = self.CATEGORIES.get(category, {}).get('name', category)

            # === TAGS: 3 strategies ===
            tags = response.css('div.tags a::text, a[rel="tag"]::text, meta[name="keywords"]::attr(content)').getall()
            if not tags:
                jsonld_keywords = self._extract_from_jsonld(response, 'keywords')
                if jsonld_keywords:
                    if isinstance(jsonld_keywords, list):
                        tags = jsonld_keywords
                    elif isinstance(jsonld_keywords, str):
                        tags = [k.strip() for k in jsonld_keywords.split(',')]
            article['tags'] = [self._clean_text(t) for t in tags if t][:10]

            # === IMAGE: 4 strategies ===
            image_url = (
                response.css('meta[property="og:image"]::attr(content)').get() or
                response.css('article img::attr(src)').get() or
                response.css('div.article-image img::attr(src)').get()
            )
            if not image_url:
                image_url = self._extract_from_jsonld(response, 'image')
            article['image_url'] = response.urljoin(image_url) if image_url else None

            # === DESCRIPTION: 3 strategies ===
            description = (
                response.css('meta[property="og:description"]::attr(content)').get() or
                response.css('meta[name="description"]::attr(content)').get()
            )
            if not description:
                description = self._extract_from_jsonld(response, 'description')
            article['description'] = self._clean_text(description) if description else None

            # Validation
            if not article['title'] or not article['content'] or len(article['content']) < 50:
                logger.warning(f"Article validation failed - {response.url}")
                self.articles_failed += 1
                return

            self.articles_scraped += 1
            if category in self.category_stats:
                self.category_stats[category]['articles'] += 1

            logger.info(f"✓ Article #{self.articles_scraped}: {article['title'][:60]}...")

            yield article

        except Exception as e:
            logger.error(f"Error parsing article {response.url}: {e}", exc_info=True)
            self.articles_failed += 1

    def handle_error(self, failure):
        logger.error(f"Request failed: {failure.request.url} - {failure.type.__name__}")
        self.articles_failed += 1

    def closed(self, reason):
        super().closed(reason)

        total = self.articles_scraped + self.articles_failed
        success_rate = (self.articles_scraped / total * 100) if total > 0 else 0

        logger.info("=" * 70)
        logger.info("FTV Spider Statistics (Complete Deep Refactoring)")
        logger.info("=" * 70)
        logger.info(f"Articles successfully scraped: {self.articles_scraped}")
        logger.info(f"Articles failed: {self.articles_failed}")
        logger.info(f"Success rate: {success_rate:.1f}%")
        logger.info(f"Pages visited: {self.pages_visited}")

        if self.category_stats:
            logger.info("")
            logger.info("PER-CATEGORY STATISTICS:")
            for cat_code, stats in self.category_stats.items():
                cat_name = self.CATEGORIES[cat_code]['name']
                logger.info(f"  {cat_name}: {stats['articles']} articles, {stats['pages']} pages")

        logger.info("=" * 70)

    def _generate_article_id(self, url: str) -> str:
        match = re.search(r'/detail/([A-Z0-9]+)', url)
        if match:
            return f"ftv_{match.group(1)}"
        return hashlib.md5(url.encode('utf-8')).hexdigest()[:16]

    def _extract_from_jsonld(self, response, field: str) -> Optional[str]:
        try:
            jsonld_scripts = response.css('script[type="application/ld+json"]::text').getall()
            for script in jsonld_scripts:
                try:
                    data = json.loads(script)
                    if isinstance(data, dict):
                        value = data.get(field)
                        if value:
                            if field == 'author' and isinstance(value, dict):
                                return value.get('name')
                            if field == 'image' and isinstance(value, dict):
                                return value.get('url')
                            return value
                except json.JSONDecodeError:
                    continue
        except Exception:
            pass
        return None

    def _clean_text(self, text: Optional[str]) -> str:
        if not text:
            return ""
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _parse_publish_date(self, date_text: Optional[str]) -> Optional[str]:
        if not date_text:
            return None

        try:
            if 'T' in date_text:
                date_text = date_text.split('T')[0]

            match = re.search(r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})', date_text)
            if match:
                year, month, day = match.groups()
                date_obj = datetime(int(year), int(month), int(day))
                return date_obj.strftime('%Y-%m-%d')
        except Exception as e:
            logger.warning(f"Failed to parse date '{date_text}': {e}")

        return None


if __name__ == '__main__':
    from scrapy.crawler import CrawlerProcess
    from twisted.internet import asyncioreactor
    asyncioreactor.install()

    process = CrawlerProcess({'LOG_LEVEL': 'INFO'})
    process.crawl(FTVNewsSpider, category='politics', days=730, max_pages=100, max_articles=50)
    process.start()
