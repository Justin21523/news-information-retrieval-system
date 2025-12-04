#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SETN (三立新聞網) Spider (Deep Refactoring v2.0)

Sitemap-focused spider for SETN (Sanlih E-Television News) with comprehensive
metadata extraction. Historical data limited to recent articles only.

Target: https://www.setn.com/
Historical Coverage: Recent articles only (historical IDs not accessible)

Key Features (Deep Refactoring v2.0):
    - Multi-sitemap support (Google News + category sitemaps)
    - 2-mode crawling strategy (sitemap/list)
    - 6+ fallback strategies per metadata field
    - NO Playwright needed (standard Scrapy)
    - Comprehensive metadata from sitemap + article pages
    - Per-category sitemap discovery

Modes:
    sitemap:  Fast discovery from Google News + category sitemaps (recommended)
    list:     Category-based browsing (fallback)

Author: Information Retrieval System
Date: 2025-11-19
Version: 2.0 (Deep Refactoring)
"""

import scrapy
from datetime import datetime, timedelta
import json
import logging
import re
import hashlib
from typing import Optional, List, Dict, Any
from urllib.parse import urljoin, urlparse, parse_qs
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


class SETNNewsSpider(scrapy.Spider):
    """
    Sitemap-focused spider for SETN (三立新聞網).

    URL Structure:
        - Article: https://www.setn.com/News.aspx?NewsID=XXXXXX
        - Google News Sitemap: https://www.setn.com/sitemapGoogleNews.xml
        - Sitemap Index: https://www.setn.com/sitemapindex.xml
        - Category Sitemap: https://www.setn.com/sitemap.xml?PageGroupID=X

    Limitations:
        - Historical IDs not accessible (only recent articles available)
        - Must rely on sitemaps for discovery

    Complexity:
        Time: O(N) where N = articles in sitemaps
        Space: O(A) for storing article metadata
    """

    name = 'setn_news'
    allowed_domains = ['setn.com', 'www.setn.com']

    # Sitemap URLs
    SITEMAPS = {
        'google_news': 'https://www.setn.com/sitemapGoogleNews.xml',
        'index': 'https://www.setn.com/sitemapindex.xml',
        'main': 'https://www.setn.com/sitemap.xml',
    }

    # Known category PageGroupIDs
    CATEGORIES = {
        'all': {'id': None, 'name': '全部'},
        'news': {'id': 1, 'name': '政治'},
        'finance': {'id': 4, 'name': '財經'},
        'entertainment': {'id': 5, 'name': '娛樂'},
        'life': {'id': 6, 'name': '生活'},
        'sports': {'id': 8, 'name': '運動'},
        'world': {'id': 12, 'name': '國際'},
        'travel': {'id': 34, 'name': '旅遊'},
        'food': {'id': 41, 'name': '美食'},
        'health': {'id': 50, 'name': '健康'},
        'tech': {'id': 65, 'name': '科技'},
    }

    # Custom settings
    custom_settings = {
        'DOWNLOAD_DELAY': 1,
        'CONCURRENT_REQUESTS_PER_DOMAIN': 3,
        'ROBOTSTXT_OBEY': True,
        'RETRY_TIMES': 3,
        'RETRY_HTTP_CODES': [500, 502, 503, 504, 408, 429],

        'FEEDS': {
            'data/raw/setn_news_%(time)s.jsonl': {
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
                 mode: str = 'sitemap',
                 category: str = 'all',
                 days: int = 7,
                 start_date: str = None,
                 end_date: str = None,
                 sitemap: str = 'google_news',
                 max_articles: int = None,
                 *args, **kwargs):
        """
        Initialize SETN spider.

        Args:
            mode: Crawling mode ('sitemap', 'list')
            category: News category
            days: Number of days to crawl (from today backwards)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            sitemap: Sitemap type ('google_news', 'index', 'main')
            max_articles: Maximum number of articles to scrape
        """
        super().__init__(*args, **kwargs)

        # Mode validation
        valid_modes = ['sitemap', 'list']
        if mode not in valid_modes:
            logger.warning(f"Invalid mode '{mode}', using 'sitemap'")
            mode = 'sitemap'
        self.mode = mode

        # Category validation
        if category not in self.CATEGORIES:
            logger.warning(f"Invalid category '{category}', using 'all'")
            category = 'all'
        self.category = category
        self.category_id = self.CATEGORIES[category]['id']
        self.category_name = self.CATEGORIES[category]['name']

        # Sitemap type
        if sitemap not in self.SITEMAPS:
            logger.warning(f"Invalid sitemap '{sitemap}', using 'google_news'")
            sitemap = 'google_news'
        self.sitemap_type = sitemap

        # Date range setup
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

        # Max articles limit
        self.max_articles = int(max_articles) if max_articles else None

        # Statistics
        self.articles_scraped = 0
        self.articles_failed = 0
        self.seen_urls = set()

        # Mode-specific statistics
        self.mode_stats = {
            'sitemap_urls': 0,
            'category_sitemaps': 0,
        }

        # Metadata quality tracking
        self.metadata_quality = {
            'has_title': 0,
            'has_content': 0,
            'has_date': 0,
            'has_author': 0,
            'has_keywords': 0,
        }

        # Log initialization
        logger.info("=" * 70)
        logger.info("SETN Spider Initialized (Deep Refactoring v2.0)")
        logger.info("=" * 70)
        logger.info(f"Mode: {self.mode}")
        if self.mode == 'sitemap':
            logger.info(f"  Sitemap: {self.sitemap_type} ({self.SITEMAPS[self.sitemap_type]})")
        if self.category != 'all':
            logger.info(f"  Category: {self.category_name} (ID: {self.category_id})")
        logger.info(f"Date Range: {self.start_date.date()} to {self.end_date.date()}")
        logger.info(f"Days: {(self.end_date - self.start_date).days + 1}")
        if self.max_articles:
            logger.info(f"Max Articles: {self.max_articles:,}")
        logger.info("=" * 70)

    def start_requests(self):
        """
        Generate start requests based on selected mode.

        Yields:
            scrapy.Request: Requests with appropriate settings
        """
        if self.mode == 'sitemap':
            yield from self._start_sitemap_mode()
        elif self.mode == 'list':
            yield from self._start_list_mode()

    def _start_sitemap_mode(self):
        """Start requests for sitemap mode."""
        if self.sitemap_type == 'index':
            # Parse sitemap index first
            sitemap_url = self.SITEMAPS['index']
            logger.info(f"Queuing sitemap index: {sitemap_url}")

            yield scrapy.Request(
                url=sitemap_url,
                callback=self.parse_sitemap_index,
                errback=self.handle_error,
                dont_filter=True,
            )
        else:
            # Direct sitemap
            sitemap_url = self.SITEMAPS[self.sitemap_type]
            logger.info(f"Queuing sitemap: {sitemap_url}")

            yield scrapy.Request(
                url=sitemap_url,
                callback=self.parse_sitemap,
                errback=self.handle_error,
                dont_filter=True,
            )

    def _start_list_mode(self):
        """Start requests for list mode (category pages)."""
        if self.category == 'all' or self.category_id is None:
            base_url = "https://www.setn.com/"
        else:
            base_url = f"https://www.setn.com/Catalog.aspx?PageGroupID={self.category_id}"

        logger.info(f"Queuing list page: {base_url}")

        yield scrapy.Request(
            url=base_url,
            callback=self.parse_list_page,
            errback=self.handle_error,
            dont_filter=True,
        )

    def parse_sitemap_index(self, response):
        """
        Parse sitemap index to discover category sitemaps.

        Args:
            response: Scrapy response object with sitemap index XML

        Yields:
            scrapy.Request: Requests for individual category sitemaps
        """
        logger.info(f"Parsing sitemap index: {response.url}")

        try:
            root = ET.fromstring(response.text)
            namespaces = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}

            sitemaps = root.findall('.//ns:sitemap', namespaces)
            logger.info(f"Found {len(sitemaps)} category sitemaps in index")

            for sitemap_elem in sitemaps:
                loc = sitemap_elem.find('ns:loc', namespaces)
                sitemap_url = loc.text if loc is not None else None

                if not sitemap_url:
                    continue

                # Filter by category if specified
                if self.category != 'all' and self.category_id is not None:
                    if f'PageGroupID={self.category_id}' not in sitemap_url:
                        continue

                self.mode_stats['category_sitemaps'] += 1
                logger.info(f"Queuing category sitemap: {sitemap_url}")

                yield scrapy.Request(
                    url=sitemap_url,
                    callback=self.parse_sitemap,
                    errback=self.handle_error,
                    dont_filter=True,
                )

        except ET.ParseError as e:
            logger.error(f"Failed to parse sitemap index XML: {e}")
        except Exception as e:
            logger.error(f"Error parsing sitemap index: {e}", exc_info=True)

    def parse_sitemap(self, response):
        """
        Parse SETN sitemap XML to extract article URLs.

        Sitemap structure (Google News format):
            <url>
                <loc>https://www.setn.com/News.aspx?NewsID=XXXXXX</loc>
                <news:news>
                    <news:publication_date>2025-11-19T00:25:00+08:00</news:publication_date>
                    <news:title>標題</news:title>
                    <news:keywords>tag1,tag2,tag3</news:keywords>
                </news:news>
                <image:image>
                    <image:loc>https://attach.setn.com/newsimages/...</image:loc>
                </image:image>
            </url>

        Args:
            response: Scrapy response object with XML content

        Yields:
            scrapy.Request: Requests for article detail pages
        """
        logger.info(f"Parsing sitemap: {response.url}")

        try:
            root = ET.fromstring(response.text)

            # Define namespaces
            namespaces = {
                'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9',
                'news': 'http://www.google.com/schemas/sitemap-news/0.9',
                'image': 'http://www.google.com/schemas/sitemap-image/1.1',
            }

            # Extract all URLs
            urls = root.findall('.//ns:url', namespaces)
            logger.info(f"Found {len(urls)} URLs in sitemap")

            url_data = []
            for url_elem in urls:
                loc = url_elem.find('ns:loc', namespaces)
                article_url = loc.text if loc is not None else None

                if not article_url or 'NewsID=' not in article_url:
                    continue

                # Extract news metadata
                news_elem = url_elem.find('.//news:news', namespaces)
                pub_date_text = None
                title_text = None
                keywords_text = None

                if news_elem is not None:
                    pub_date_elem = news_elem.find('.//news:publication_date', namespaces)
                    if pub_date_elem is not None:
                        pub_date_text = pub_date_elem.text

                    title_elem = news_elem.find('.//news:title', namespaces)
                    if title_elem is not None:
                        title_text = title_elem.text

                    keywords_elem = news_elem.find('.//news:keywords', namespaces)
                    if keywords_elem is not None:
                        keywords_text = keywords_elem.text

                # Extract image metadata
                image_elem = url_elem.find('.//image:image', namespaces)
                image_url = None
                if image_elem is not None:
                    image_loc_elem = image_elem.find('image:loc', namespaces)
                    if image_loc_elem is not None:
                        image_url = image_loc_elem.text

                url_data.append({
                    'url': article_url,
                    'sitemap_date': pub_date_text,
                    'sitemap_title': title_text,
                    'sitemap_keywords': keywords_text,
                    'sitemap_image': image_url,
                })

            self.mode_stats['sitemap_urls'] += len(url_data)

            # Apply date filter
            if url_data:
                url_data = self._apply_date_filter(url_data)
                logger.info(f"After date filtering: {len(url_data)} URLs")

            # Queue article requests
            for item in url_data:
                article_url = item['url']

                if article_url in self.seen_urls:
                    continue
                self.seen_urls.add(article_url)

                yield scrapy.Request(
                    url=article_url,
                    callback=self.parse_article,
                    errback=self.handle_error,
                    meta={'sitemap_data': item},
                    dont_filter=False,
                )

                # Check max articles limit
                if self.max_articles and len(self.seen_urls) >= self.max_articles:
                    logger.info(f"Reached max articles limit ({self.max_articles}), stopping")
                    return

        except ET.ParseError as e:
            logger.error(f"Failed to parse sitemap XML: {e}")
        except Exception as e:
            logger.error(f"Error parsing sitemap: {e}", exc_info=True)

    def _apply_date_filter(self, url_data: List[Dict]) -> List[Dict]:
        """
        Apply date filtering to URL data.

        Args:
            url_data: List of URL dictionaries

        Returns:
            List[Dict]: Filtered URL data
        """
        if not url_data:
            return []

        filtered = []
        for item in url_data:
            if item.get('sitemap_date'):
                try:
                    # Parse ISO 8601: 2025-11-19T00:25:00+08:00
                    date_str = item['sitemap_date']
                    # Remove timezone
                    date_str = date_str.split('+')[0].split('-0')[0]
                    pub_date = datetime.fromisoformat(date_str)

                    if self.start_date <= pub_date <= self.end_date:
                        filtered.append(item)
                    continue
                except (ValueError, AttributeError):
                    pass

            # If no date or parse failed, include it
            filtered.append(item)

        return filtered

    def parse_list_page(self, response):
        """
        Parse SETN news list page to extract article URLs.

        Args:
            response: Scrapy response object

        Yields:
            scrapy.Request: Requests for article detail pages
        """
        logger.info(f"Parsing list page: {response.url}")

        # Extract article links
        article_selectors = response.css('a[href*="News.aspx?NewsID="]')

        articles_found = 0
        for article_sel in article_selectors:
            article_url = article_sel.css('::attr(href)').get()

            if not article_url:
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
                dont_filter=False,
            )

            if self.max_articles and len(self.seen_urls) >= self.max_articles:
                logger.info(f"Reached max articles limit ({self.max_articles}), stopping")
                return

        logger.info(f"Found {articles_found} articles on list page")

    def parse_article(self, response):
        """
        Parse individual SETN article page.

        Extraction Strategies (6+ fallback strategies per field):
            - Title: 6 strategies
            - Content: 5 strategies
            - Date: 6 strategies
            - Author: 4 strategies
            - Keywords: 4 strategies (including sitemap)

        Args:
            response: Scrapy response object

        Yields:
            dict: Article data in standardized format
        """
        try:
            logger.debug(f"Parsing article: {response.url}")

            # Get sitemap data if available
            sitemap_data = response.meta.get('sitemap_data', {})

            # Initialize article data
            article = {
                'article_id': self._generate_article_id(response.url),
                'url': response.url,
                'source': 'SETN',
                'source_name': '三立新聞網',
                'crawled_at': datetime.now().isoformat(),
            }

            # === TITLE: 6 strategies ===
            title = (
                response.css('meta[property="og:title"]::attr(content)').get() or
                sitemap_data.get('sitemap_title') or
                response.css('h1.news-title::text').get() or
                response.css('h1::text').get() or
                response.css('meta[name="twitter:title"]::attr(content)').get() or
                response.css('title::text').get()
            )
            article['title'] = self._clean_text(title) if title else None

            # === CONTENT: 5 strategies ===
            content_selectors = [
                'div.article-body p::text',
                'article p::text',
                'div.news-content p::text',
                'div#news-content p::text',
                'div.content-area p::text',
            ]

            content_paragraphs = []
            for selector in content_selectors:
                paragraphs = response.css(selector).getall()
                if paragraphs and len(paragraphs) >= 2:
                    content_paragraphs = paragraphs
                    break

            article['content'] = ' '.join([self._clean_text(p) for p in content_paragraphs if p])

            # === DATE: 6 strategies ===
            date_text = (
                response.css('meta[property="article:published_time"]::attr(content)').get() or
                sitemap_data.get('sitemap_date') or
                response.css('time::attr(datetime)').get() or
                response.css('span.date::text').get() or
                response.css('div.news-time::text').get() or
                response.css('meta[name="pubdate"]::attr(content)').get()
            )
            article['published_date'] = self._parse_publish_date(date_text)

            # Date filtering
            if article['published_date']:
                try:
                    pub_date_obj = datetime.strptime(article['published_date'], '%Y-%m-%d')
                    if not (self.start_date <= pub_date_obj <= self.end_date):
                        logger.debug(f"Article date {article['published_date']} out of range, skipping")
                        return
                except ValueError:
                    pass

            # === AUTHOR: 4 strategies ===
            author = (
                response.css('span.author::text').get() or
                response.css('div.author::text').get() or
                response.css('meta[name="author"]::attr(content)').get() or
                response.css('a.author::text').get()
            )
            article['author'] = self._clean_text(author) if author else '三立新聞網'

            # === KEYWORDS/TAGS: 4 strategies ===
            # Strategy 1: Sitemap keywords
            keywords = []
            if sitemap_data.get('sitemap_keywords'):
                keywords = [k.strip() for k in sitemap_data['sitemap_keywords'].split(',') if k.strip()]

            # Strategy 2: Meta keywords
            if not keywords:
                meta_keywords = response.css('meta[name="keywords"]::attr(content)').get()
                if meta_keywords:
                    keywords = [k.strip() for k in meta_keywords.split(',') if k.strip()]

            # Strategy 3: Tag links
            if not keywords:
                keywords = response.css('a.tag::text, a[rel="tag"]::text').getall()

            # Strategy 4: News tags
            if not keywords:
                keywords = response.css('div.tags a::text, span.tag::text').getall()

            article['tags'] = [self._clean_text(k) for k in keywords if k][:10]  # Limit to 10

            # === CATEGORY ===
            # Extract from URL
            parsed_url = urlparse(response.url)
            query_params = parse_qs(parsed_url.query)
            page_group_id = query_params.get('PageGroupID', [None])[0]

            if page_group_id:
                for cat_name, cat_info in self.CATEGORIES.items():
                    if cat_info['id'] == int(page_group_id):
                        article['category'] = cat_name
                        article['category_name'] = cat_info['name']
                        break
            else:
                article['category'] = 'unknown'
                article['category_name'] = '未分類'

            # === IMAGE: 3 strategies ===
            image_url = (
                response.css('meta[property="og:image"]::attr(content)').get() or
                sitemap_data.get('sitemap_image') or
                response.css('article img::attr(src)').get()
            )
            article['image_url'] = response.urljoin(image_url) if image_url else None

            # === DESCRIPTION: 2 strategies ===
            description = (
                response.css('meta[property="og:description"]::attr(content)').get() or
                response.css('meta[name="description"]::attr(content)').get()
            )
            article['description'] = self._clean_text(description) if description else None

            # === VALIDATION ===
            issues = []
            if not article['title']:
                issues.append("missing_title")
            if not article['content'] or len(article['content']) < 50:
                issues.append(f"content_too_short ({len(article.get('content', ''))} chars)")

            if issues:
                logger.warning(f"Article quality issues: {', '.join(issues)} - {response.url}")
                self.articles_failed += 1
                return

            # === METADATA QUALITY TRACKING ===
            if article['title']:
                self.metadata_quality['has_title'] += 1
            if article['content'] and len(article['content']) >= 50:
                self.metadata_quality['has_content'] += 1
            if article['published_date']:
                self.metadata_quality['has_date'] += 1
            if article['author'] and article['author'] != '三立新聞網':
                self.metadata_quality['has_author'] += 1
            if article.get('tags') and len(article['tags']) > 0:
                self.metadata_quality['has_keywords'] += 1

            # Success
            self.articles_scraped += 1
            logger.info(f"✓ Article #{self.articles_scraped}: {article['title'][:60]}...")

            yield article

        except Exception as e:
            logger.error(f"Error parsing article {response.url}: {e}", exc_info=True)
            self.articles_failed += 1

    def handle_error(self, failure):
        """Handle request errors."""
        request = failure.request
        logger.error(f"Request failed: {request.url} - {failure.type.__name__}: {failure.value}")
        self.articles_failed += 1

    def closed(self, reason):
        """Log spider closure statistics."""
        total_articles = self.articles_scraped + self.articles_failed
        success_rate = (self.articles_scraped / total_articles * 100) if total_articles > 0 else 0

        logger.info("=" * 70)
        logger.info("SETN Spider Statistics (Deep Refactoring v2.0)")
        logger.info("=" * 70)
        logger.info(f"Mode: {self.mode}")
        logger.info(f"Articles successfully scraped: {self.articles_scraped}")
        logger.info(f"Articles failed: {self.articles_failed}")
        logger.info(f"Success rate: {success_rate:.1f}%")

        logger.info("")
        logger.info("MODE-SPECIFIC STATISTICS:")
        if self.mode_stats['sitemap_urls'] > 0:
            logger.info(f"  Sitemap URLs discovered: {self.mode_stats['sitemap_urls']}")
        if self.mode_stats['category_sitemaps'] > 0:
            logger.info(f"  Category sitemaps processed: {self.mode_stats['category_sitemaps']}")

        logger.info("")
        logger.info("METADATA QUALITY:")
        if self.articles_scraped > 0:
            for field, count in self.metadata_quality.items():
                percentage = 100 * count / self.articles_scraped
                logger.info(f"  {field}: {count}/{self.articles_scraped} ({percentage:.1f}%)")

        logger.info("")
        logger.info(f"Date range: {self.start_date.date()} to {self.end_date.date()}")
        logger.info("=" * 70)

    # ========== Utility Methods ==========

    def _generate_article_id(self, url: str) -> str:
        """Generate unique article ID from URL."""
        match = re.search(r'NewsID=(\d+)', url)
        if match:
            return f"setn_{match.group(1)}"
        return hashlib.md5(url.encode('utf-8')).hexdigest()[:16]

    def _clean_text(self, text: Optional[str]) -> str:
        """Clean and normalize text."""
        if not text:
            return ""

        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text)
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

        try:
            # ISO 8601: 2025-11-19T00:25:00+08:00
            if 'T' in date_text:
                date_text = date_text.split('T')[0]

            patterns = [
                r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})',
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

    process = CrawlerProcess({
        'DOWNLOAD_DELAY': 1,
        'CONCURRENT_REQUESTS_PER_DOMAIN': 3,
        'ROBOTSTXT_OBEY': True,
        'LOG_LEVEL': 'INFO',
    })

    # Example: Sitemap mode (recommended)
    process.crawl(
        SETNNewsSpider,
        mode='sitemap',
        sitemap='google_news',
        days=7,
        max_articles=10
    )

    process.start()
