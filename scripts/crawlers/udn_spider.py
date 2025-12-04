"""
United Daily News (UDN) Spider

Spider for crawling news articles from United Daily News (聯合報).
Refactored for maximum historical data access using sequential ID strategy.

Target: https://udn.com/news/
Historical Coverage: Sparse ID space with clustered articles

Author: Information Retrieval System
Date: 2025-11-18 (Refactored)
"""

import scrapy
from datetime import datetime, timedelta
import json
import logging
import re
from typing import Optional, List
from scrapy.spidermiddlewares.httperror import HttpError
from twisted.internet.error import DNSLookupError, TimeoutError, TCPTimedOutError

logger = logging.getLogger(__name__)


class UDNNewsSpider(scrapy.Spider):
    """
    Scrapy spider for United Daily News (聯合報) with multi-mode strategy.

    Historical Discovery (2025-11-18):
        - Sequential IDs accessible but SPARSE (~40% hit rate)
        - Current range: 9,147,000 - 9,148,000+ (Nov 2025)
        - Historical IDs found: 7,800,000, 8,000,000, 8,300,000
        - Articles cluster in groups with gaps between

    URL Structure:
        - Article: https://udn.com/news/story/{story_id}/{article_id}
        - **Key Discovery:** story_id is arbitrary! Any value works for same article_id
        - List page: https://udn.com/news/index

    Modes:
        1. list: Fast crawl from list pages (recent articles, daily updates)
        2. sequential: Systematic ID-based crawl (handles sparse ID space)
        3. hybrid: Discover recent + fill historical gaps intelligently

    Categories:
        politics (政治), opinion (評論), society (社會), local (地方),
        world (國際), china (兩岸), stock (股市), etc.

    Usage:
        # List mode (recent articles)
        scrapy runspider udn_spider.py -a mode=list -o udn.jsonl

        # Sequential mode (historical crawl)
        scrapy runspider udn_spider.py \
            -a mode=sequential \
            -a start_id=7800000 \
            -a end_id=7850000 \
            -o udn_historical.jsonl

        # Hybrid mode (discover + fill gaps)
        scrapy runspider udn_spider.py -a mode=hybrid -o udn.jsonl

    Complexity:
        Time: O(N) where N = ID range (with ~60% 404s in sequential mode)
        Space: O(A) where A = articles scraped
    """

    name = 'udn_news'
    allowed_domains = ['udn.com']

    # Custom settings optimized for speed (no Playwright!)
    custom_settings = {
        'DOWNLOAD_DELAY': 1.5,  # Respectful delay
        'CONCURRENT_REQUESTS_PER_DOMAIN': 3,
        'ROBOTSTXT_OBEY': True,
        'USER_AGENT': 'CNIRS Academic Research Bot (Educational Use)',
        'HTTPERROR_ALLOW_404': True,  # Allow 404 to reach errback
        'RETRY_TIMES': 2,
        'RETRY_HTTP_CODES': [500, 502, 503, 504, 408, 429],

        # Output settings
        'FEEDS': {
            'data/raw/udn_news_%(time)s.jsonl': {
                'format': 'jsonlines',
                'encoding': 'utf8',
                'store_empty': False,
                'overwrite': False,
            }
        },
    }

    # Category mapping
    CATEGORIES = {
        'all': {'id': 'index', 'name': '全部'},
        'politics': {'id': '6638', 'name': '政治'},
        'opinion': {'id': '6643', 'name': '評論'},
        'society': {'id': '7227', 'name': '社會'},
        'local': {'id': '7228', 'name': '地方'},
        'life': {'id': '7239', 'name': '生活'},
        'world': {'id': '6809', 'name': '國際'},
        'china': {'id': '6811', 'name': '兩岸'},
        'stock': {'id': '7239', 'name': '股市'},
    }

    # Historical data range (discovered 2025-11-18)
    # Note: Sparse ID space - many gaps between valid IDs
    KNOWN_HISTORICAL_START = 7800000  # Confirmed accessible
    CURRENT_ID_ESTIMATE = 9148000  # Will be higher over time
    DEFAULT_STORY_ID = "6638"  # Politics - works for any article_id

    def __init__(self,
                 mode: str = 'list',
                 days: int = 7,
                 start_id: int = None,
                 end_id: int = None,
                 *args, **kwargs):
        """
        Initialize UDN spider with multi-mode support.

        Args:
            mode: Crawling mode ('list', 'sequential', 'hybrid')
            days: Number of days to crawl (for list/hybrid mode)
            start_id: Start article ID (for sequential mode)
            end_id: End article ID (for sequential mode)

        Example:
            >>> spider = UDNNewsSpider(mode='sequential', start_id=7800000, end_id=7850000)
            >>> spider = UDNNewsSpider(mode='list', days=7)
            >>> spider = UDNNewsSpider(mode='hybrid')
        """
        super(UDNNewsSpider, self).__init__(*args, **kwargs)

        # Set mode
        self.mode = mode.lower()
        if self.mode not in ['list', 'sequential', 'hybrid']:
            logger.warning(f"Invalid mode '{mode}', using 'list'")
            self.mode = 'list'

        # Set date range (for list/hybrid mode)
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=int(days))

        # Set ID range (for sequential/hybrid mode)
        self.start_id = int(start_id) if start_id else self.KNOWN_HISTORICAL_START
        self.end_id = int(end_id) if end_id else None  # Will auto-detect if None

        # Statistics
        self.articles_count = 0
        self.failed_count = 0
        self.not_found_count = 0
        self.seen_ids = set()

        logger.info("=" * 70)
        logger.info("UDN Spider Initialized")
        logger.info("=" * 70)
        logger.info(f"Mode: {self.mode}")
        if self.mode in ['sequential', 'hybrid']:
            logger.info(f"ID Range: {self.start_id} to {self.end_id if self.end_id else 'auto-detect'}")
            logger.info("Note: UDN has sparse ID space (~40% hit rate expected)")
        logger.info("=" * 70)

    def start_requests(self):
        """
        Generate start requests based on mode.

        Three modes:
        1. list: Crawl from homepage/list pages (fast, recent articles)
        2. sequential: Direct ID-based crawl (complete historical coverage)
        3. hybrid: Combine both strategies

        Yields:
            scrapy.Request: Initial requests
        """
        if self.mode == 'list':
            # List mode: Start with homepage
            logger.info("List mode: Starting with https://udn.com/news/index")

            yield scrapy.Request(
                url="https://udn.com/news/index",
                callback=self.parse_list_page,
                errback=self.handle_error,
                dont_filter=True
            )

        elif self.mode == 'sequential':
            # Sequential mode: Direct ID crawling
            if not self.end_id:
                # Auto-detect current max ID from homepage first
                logger.info("Sequential mode: Auto-detecting max ID from homepage...")
                yield scrapy.Request(
                    url="https://udn.com/news/index",
                    callback=self.parse_list_for_max_id,
                    errback=self.handle_error,
                    dont_filter=True
                )
            else:
                # Start sequential crawling immediately
                logger.info(f"Sequential mode: Crawling IDs {self.start_id} to {self.end_id}")
                for request in self._generate_sequential_requests():
                    yield request

        elif self.mode == 'hybrid':
            # Hybrid mode: Discover recent + fill gaps
            logger.info("Hybrid mode: Discovering recent articles + filling historical gaps...")
            yield scrapy.Request(
                url="https://udn.com/news/index",
                callback=self.parse_list_hybrid,
                errback=self.handle_error,
                dont_filter=True
            )

    def _generate_sequential_requests(self):
        """
        Generate sequential article requests by ID.

        Note: Uses arbitrary story_id (works for any article!)

        Yields:
            scrapy.Request: Article requests
        """
        for article_id in range(self.start_id, self.end_id + 1):
            # Use default story_id (arbitrary - any value works!)
            url = f"https://udn.com/news/story/{self.DEFAULT_STORY_ID}/{article_id}"

            yield scrapy.Request(
                url=url,
                callback=self.parse_article,
                errback=self.handle_error,
                meta={'article_id': article_id, 'mode': 'sequential'},
                dont_filter=True
            )

    def parse_list_page(self, response):
        """
        Parse UDN homepage/list page to extract article URLs.

        Args:
            response: Scrapy response object

        Yields:
            scrapy.Request: Article requests
        """
        logger.info(f"Parsing list page: {response.url}")

        # Extract article links - UDN format: /news/story/{story_id}/{article_id}
        article_links = response.css('a[href*="/news/story/"]::attr(href)').getall()

        articles_found = 0
        for link in article_links:
            if not link or '/news/story/' not in link:
                continue

            article_url = response.urljoin(link)

            # Extract article_id from URL
            match = re.search(r'/news/story/\d+/(\d+)', article_url)
            if match:
                article_id = int(match.group(1))
                if article_id in self.seen_ids:
                    continue
                self.seen_ids.add(article_id)

            articles_found += 1

            yield scrapy.Request(
                url=article_url,
                callback=self.parse_article,
                errback=self.handle_error,
                meta={'mode': 'list'},
                dont_filter=False
            )

        logger.info(f"Found {articles_found} articles on list page")

    def parse_list_for_max_id(self, response):
        """
        Parse homepage to auto-detect maximum article ID.

        Used in sequential mode when end_id is not specified.

        Args:
            response: Scrapy response object

        Yields:
            scrapy.Request: Sequential crawling requests
        """
        article_links = response.css('a[href*="/news/story/"]::attr(href)').getall()

        ids = []
        for link in article_links:
            match = re.search(r'/news/story/\d+/(\d+)', link)
            if match:
                ids.append(int(match.group(1)))

        if ids:
            self.end_id = max(ids)
            logger.info(f"Auto-detected maximum article ID: {self.end_id}")
        else:
            # Fallback to current estimate
            self.end_id = self.CURRENT_ID_ESTIMATE
            logger.warning(f"Could not auto-detect max ID, using estimate: {self.end_id}")

        # Now start sequential crawling
        logger.info(f"Starting sequential crawl: {self.start_id} to {self.end_id}")
        for request in self._generate_sequential_requests():
            yield request

    def parse_list_hybrid(self, response):
        """
        Parse homepage for hybrid mode: discover recent + fill gaps.

        Strategy:
        1. Extract all article IDs from homepage
        2. Crawl discovered articles
        3. Fill gaps between start_id and min discovered ID

        Args:
            response: Scrapy response object

        Yields:
            scrapy.Request: Article requests
        """
        article_links = response.css('a[href*="/news/story/"]::attr(href)').getall()

        discovered_ids = set()
        for link in article_links:
            match = re.search(r'/news/story/\d+/(\d+)', link)
            if match:
                discovered_ids.add(int(match.group(1)))

        logger.info(f"Hybrid mode: Discovered {len(discovered_ids)} article IDs from homepage")

        if not discovered_ids:
            logger.warning("No article IDs discovered, falling back to sequential mode")
            self.end_id = self.CURRENT_ID_ESTIMATE
            for request in self._generate_sequential_requests():
                yield request
            return

        max_discovered_id = max(discovered_ids)
        min_discovered_id = min(discovered_ids)

        logger.info(f"Discovered ID range: {min_discovered_id} to {max_discovered_id}")

        # 1. Crawl discovered articles
        for article_id in sorted(discovered_ids):
            url = f"https://udn.com/news/story/{self.DEFAULT_STORY_ID}/{article_id}"
            yield scrapy.Request(
                url=url,
                callback=self.parse_article,
                errback=self.handle_error,
                meta={'article_id': article_id, 'mode': 'hybrid'},
                dont_filter=True
            )

        # 2. Fill gaps from start_id to min_discovered_id
        if self.start_id < min_discovered_id:
            gap_count = min_discovered_id - self.start_id
            logger.info(f"Filling {gap_count} IDs from {self.start_id} to {min_discovered_id}")
            logger.info("Note: Expect ~60% 404s due to sparse ID space")

            for article_id in range(self.start_id, min_discovered_id):
                if article_id not in discovered_ids:
                    url = f"https://udn.com/news/story/{self.DEFAULT_STORY_ID}/{article_id}"
                    yield scrapy.Request(
                        url=url,
                        callback=self.parse_article,
                        errback=self.handle_error,
                        meta={'article_id': article_id, 'mode': 'hybrid'},
                        dont_filter=True
                    )

    def parse_article(self, response):
        """
        Parse individual UDN article page.

        Extraction Strategy:
        1. Title: h1.article-content__title
        2. Content: section.article-content__editor p
        3. Date: time.article-content__time
        4. Author: from byline
        5. Category: from URL or breadcrumb
        6. Tags: meta keywords

        Args:
            response: Scrapy response object

        Yields:
            dict: Article data
        """
        try:
            # Extract article ID
            article_id = response.meta.get('article_id')
            if not article_id:
                match = re.search(r'/news/story/\d+/(\d+)', response.url)
                article_id = int(match.group(1)) if match else None

            article = {
                'article_id': f"UDN_{article_id}",
                'url': response.url,
                'source': 'UDN',
                'source_name': '聯合報',
                'crawled_at': datetime.now().isoformat(),
            }

            # Title - multiple selectors
            title = (response.css('h1.article-content__title::text').get() or
                    response.css('h1::text').get() or
                    response.css('meta[property="og:title"]::attr(content)').get())
            article['title'] = self.clean_text(title)

            # Content paragraphs - multiple selectors
            content_paragraphs = (
                response.css('section.article-content__editor p::text').getall() or
                response.css('div.article-content p::text').getall() or
                response.css('article p::text').getall()
            )
            article['content'] = ' '.join([self.clean_text(p) for p in content_paragraphs if p])

            # Author
            author = (response.css('span.article-content__author::text').get() or
                     response.css('div.article_author::text').get() or
                     response.css('meta[name="author"]::attr(content)').get())
            article['author'] = self.clean_text(author) if author else '聯合報'

            # Publish date
            date_text = (response.css('time.article-content__time::attr(datetime)').get() or
                        response.css('time::attr(datetime)').get() or
                        response.css('meta[property="article:published_time"]::attr(content)').get())
            article['publish_date'] = self.parse_date(date_text)

            # Category from URL
            article['category'] = self.extract_category_from_url(response.url)

            # Tags
            keywords = response.css('meta[name="keywords"]::attr(content)').get()
            if keywords:
                article['tags'] = [k.strip() for k in keywords.split(',')]
            else:
                article['tags'] = []

            # Image
            image_url = (response.css('figure.article-content__image img::attr(src)').get() or
                        response.css('meta[property="og:image"]::attr(content)').get())
            article['image_url'] = response.urljoin(image_url) if image_url else None

            # Validation
            if not article['title']:
                logger.warning(f"Missing title for {response.url}")
                self.failed_count += 1
                return

            if not article['content'] or len(article['content']) < 100:
                logger.warning(f"Content too short ({len(article.get('content', ''))} chars) for {response.url}")
                self.failed_count += 1
                return

            self.articles_count += 1
            if self.articles_count % 10 == 0:
                logger.info(f"Successfully parsed {self.articles_count} articles")

            yield article

        except Exception as e:
            logger.error(f"Error parsing article {response.url}: {e}")
            self.failed_count += 1

    def handle_error(self, failure):
        """
        Handle request errors with smart 404 handling.

        Args:
            failure: Twisted failure object
        """
        if failure.check(HttpError):
            response = failure.value.response
            if response.status == 404:
                # 404 is VERY common in sequential mode due to sparse ID space
                self.not_found_count += 1
                if self.mode == 'sequential' and self.not_found_count % 1000 == 0:
                    logger.info(f"404 count: {self.not_found_count} (expected - sparse ID space)")
                return
            else:
                logger.warning(f"HTTP {response.status} for {failure.request.url}")
                self.failed_count += 1
        elif failure.check(DNSLookupError):
            logger.error(f"DNS error: {failure.request.url}")
            self.failed_count += 1
        elif failure.check(TimeoutError, TCPTimedOutError):
            logger.error(f"Timeout: {failure.request.url}")
            self.failed_count += 1
        else:
            logger.error(f"Request failed: {failure.request.url}")
            logger.error(f"Error: {repr(failure.value)}")
            self.failed_count += 1

    def closed(self, reason):
        """
        Log spider closure statistics.

        Args:
            reason: Closure reason
        """
        total_attempts = self.articles_count + self.failed_count + self.not_found_count

        logger.info("=" * 70)
        logger.info("UDN Spider Finished")
        logger.info("=" * 70)
        logger.info(f"Reason: {reason}")
        logger.info(f"Mode: {self.mode}")
        logger.info(f"Articles successfully crawled: {self.articles_count}")
        logger.info(f"Not found (404): {self.not_found_count}")
        logger.info(f"Failed requests: {self.failed_count}")
        logger.info(f"Total attempts: {total_attempts}")

        if total_attempts > 0:
            success_rate = 100 * self.articles_count / total_attempts
            logger.info(f"Overall success rate: {success_rate:.2f}%")

            if self.not_found_count > 0:
                hit_rate = 100 * self.articles_count / (self.articles_count + self.not_found_count)
                logger.info(f"Hit rate (excluding errors): {hit_rate:.2f}%")

        if self.mode == 'sequential' and self.start_id and self.end_id:
            expected_range = self.end_id - self.start_id + 1
            logger.info(f"ID range scanned: {self.start_id} to {self.end_id} ({expected_range} IDs)")

        logger.info("=" * 70)

    # ========== Utility Methods ==========

    @staticmethod
    def clean_text(text: Optional[str]) -> str:
        """
        Clean text by removing extra whitespace.

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        if not text:
            return ""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    @staticmethod
    def parse_date(date_str: Optional[str]) -> Optional[str]:
        """
        Parse date string to ISO format (YYYY-MM-DD).

        Args:
            date_str: Raw date string

        Returns:
            Date in ISO format or None
        """
        if not date_str:
            return None

        try:
            # UDN formats: ISO 8601
            formats = [
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%d %H:%M:%S',
                '%Y/%m/%d %H:%M',
                '%Y-%m-%d',
            ]

            for fmt in formats:
                try:
                    dt = datetime.strptime(date_str.strip()[:19], fmt)
                    return dt.strftime('%Y-%m-%d')
                except ValueError:
                    continue

            # Extract YYYY-MM-DD pattern
            match = re.search(r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})', date_str)
            if match:
                return f"{match.group(1)}-{match.group(2):0>2}-{match.group(3):0>2}"

        except Exception as e:
            logger.warning(f"Could not parse date: {date_str}")

        return None

    def extract_category_from_url(self, url: str) -> str:
        """
        Extract category from article URL by story_id.

        Args:
            url: Article URL

        Returns:
            Category name
        """
        # UDN URL: /news/story/{story_id}/{article_id}
        match = re.search(r'/news/story/(\d+)/', url)
        if match:
            story_id = match.group(1)
            for cat, info in self.CATEGORIES.items():
                if info['id'] == story_id:
                    return info['name']
        return '其他'


# Standalone execution
if __name__ == '__main__':
    from scrapy.crawler import CrawlerProcess
    from scrapy.utils.project import get_project_settings

    settings = get_project_settings()
    settings.update({
        'DOWNLOAD_DELAY': 1.5,
        'CONCURRENT_REQUESTS_PER_DOMAIN': 3,
        'ROBOTSTXT_OBEY': True,
        'LOG_LEVEL': 'INFO',
    })

    process = CrawlerProcess(settings)

    # Example: Sequential mode for historical crawl
    process.crawl(
        UDNNewsSpider,
        mode='sequential',
        start_id=7800000,
        end_id=7800100  # Small range for testing
    )

    process.start()
