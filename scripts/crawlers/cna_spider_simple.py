#!/usr/bin/env python
"""
CNA News Spider - Simplified Version (No Playwright)

This is a lightweight version that uses regular Scrapy without Playwright.
Use this for testing and when JavaScript rendering is not required.

If CNA blocks this spider, switch to cna_spider_v2.py (with Playwright).

Usage:
    scrapy crawl cna_simple -a start_date=2024-11-01 -a end_date=2024-11-13

Author: CNIRS Development Team
License: Educational Use Only
"""

import scrapy
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from urllib.parse import urljoin

logger = logging.getLogger(__name__)


class CNANewsSpiderSimple(scrapy.Spider):
    """
    Simplified CNA News Spider without Playwright.

    This version is faster but may not work if CNA requires JavaScript.
    """

    name = 'cna_simple'
    allowed_domains = ['cna.com.tw']

    # CNA category codes
    CATEGORIES = {
        'aipl': '政治',
        'aie': '財經',
        'ahel': '生活',
        'ait': '科技',
        'asoc': '社會',
        'acul': '文化',
        'aspt': '運動',
        'amov': '娛樂',
        'aopl': '國際',
        'acn': '兩岸',
        'aloc': '地方',
    }

    custom_settings = {
        'CONCURRENT_REQUESTS_PER_DOMAIN': 2,
        'DOWNLOAD_DELAY': 2,
        'AUTOTHROTTLE_ENABLED': True,
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    }

    def __init__(self, start_date: str = None, end_date: str = None,
                categories: str = None, max_articles: int = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Parse dates
        if start_date:
            self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        else:
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

        logger.info(f"CNA Simple Spider initialized:")
        logger.info(f"  Date range: {self.start_date.date()} to {self.end_date.date()}")
        logger.info(f"  Categories: {list(self.categories.keys())}")

    def start_requests(self):
        """Generate start requests for all categories."""
        for category_code, category_name in self.categories.items():
            url = f"https://www.cna.com.tw/list/{category_code}.aspx"
            logger.info(f"Starting category: {category_name} ({category_code})")

            yield scrapy.Request(
                url=url,
                callback=self.parse_category_page,
                cb_kwargs={'category': category_code, 'category_name': category_name},
            )

    def parse_category_page(self, response, category: str, category_name: str):
        """Parse category list page to extract article links."""
        logger.info(f"Parsing {category_name}: {response.url}")

        # Extract article links - CNA uses pattern: /news/<cat>/YYYYMMDDNNNN.aspx
        article_links = response.css('a[href*="/news/"]::attr(href)').getall()

        # Filter to only news articles
        article_links = [
            link for link in article_links
            if re.match(r'/news/\w+/\d{12}\.aspx', link)
        ]

        logger.info(f"Found {len(article_links)} article links")

        articles_yielded = 0

        for link in article_links:
            article_url = urljoin(response.url, link)

            if self.max_articles and articles_yielded >= self.max_articles:
                break

            articles_yielded += 1

            yield scrapy.Request(
                url=article_url,
                callback=self.parse_article,
                cb_kwargs={'category': category, 'category_name': category_name},
            )

    def parse_article(self, response, category: str, category_name: str):
        """Parse individual news article."""
        try:
            article = self._extract_article_data(response, category, category_name)

            if not article:
                logger.warning(f"Failed to extract: {response.url}")
                self.stats['errors'] += 1
                return

            # Date filtering
            if article.get('published_date'):
                try:
                    pub_date = datetime.strptime(article['published_date'], '%Y-%m-%d')
                    if not (self.start_date <= pub_date <= self.end_date):
                        self.stats['articles_filtered'] += 1
                        return
                except ValueError:
                    pass

            self.stats['articles_scraped'] += 1
            logger.info(f"Scraped #{self.stats['articles_scraped']}: {article['title'][:50]}...")

            yield article

        except Exception as e:
            logger.error(f"Error parsing {response.url}: {e}")
            self.stats['errors'] += 1

    def _extract_article_data(self, response, category: str, category_name: str) -> Optional[Dict[str, Any]]:
        """Extract article data from response."""
        # Title
        title = response.css('h1::text, h1 span::text').get()
        if title:
            title = title.strip()

        if not title:
            return None

        # Content - CNA uses div.paragraph
        content_parts = response.css('div.paragraph p::text').getall()
        content = '\n'.join([p.strip() for p in content_parts if p.strip()])

        # Date - from meta tags (more reliable)
        date_meta = response.css('meta[property="article:published_time"]::attr(content)').get()
        if date_meta:
            # Format: 2025-11-13T05:38:00+08:00
            published_date = date_meta.split('T')[0]
        else:
            # Fallback to text extraction
            date_text = response.css('time::text, div.updatetime::text, span.date::text').get()
            published_date = self._parse_date(date_text) if date_text else None

        # Author - from meta tag
        author = response.css('meta[property="author"]::attr(content)').get()
        if not author:
            author = response.css('span.author::text, div.author::text').get()
        if author:
            author = author.strip()

        # Tags - from meta tags
        tags = response.css('meta[property="article:tag"]::attr(content)').getall()
        if not tags:
            # Fallback to keywords meta tag
            keywords_meta = response.css('meta[name="keywords"]::attr(content)').get()
            if keywords_meta:
                tags = [k.strip() for k in keywords_meta.split(',')]
        tags = [t.strip() for t in tags if t.strip()]

        # Article ID from URL
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

    def _parse_date(self, date_text: str) -> Optional[str]:
        """Parse date from text."""
        import re

        # Clean text
        date_text = date_text.strip()

        # Try: YYYY/MM/DD HH:MM
        match = re.search(r'(\d{4})/(\d{1,2})/(\d{1,2})', date_text)
        if match:
            year, month, day = match.groups()
            try:
                date_obj = datetime(int(year), int(month), int(day))
                return date_obj.strftime('%Y-%m-%d')
            except ValueError:
                pass

        # Try: YYYY-MM-DD
        match = re.search(r'(\d{4})-(\d{1,2})-(\d{1,2})', date_text)
        if match:
            year, month, day = match.groups()
            try:
                date_obj = datetime(int(year), int(month), int(day))
                return date_obj.strftime('%Y-%m-%d')
            except ValueError:
                pass

        return None

    def closed(self, reason):
        """Log statistics when spider closes."""
        logger.info("=" * 70)
        logger.info(f"Spider closed: {reason}")
        logger.info(f"  Articles scraped: {self.stats['articles_scraped']}")
        logger.info(f"  Articles filtered: {self.stats['articles_filtered']}")
        logger.info(f"  Errors: {self.stats['errors']}")
        logger.info("=" * 70)
