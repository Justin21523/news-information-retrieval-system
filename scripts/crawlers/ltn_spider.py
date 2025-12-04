"""
Liberty Times Net (LTN) News Spider

Spider for crawling news articles from Liberty Times Net (自由時報).
Refactored for maximum historical data access using sequential ID strategy.

Target: https://news.ltn.com.tw/
Historical Coverage: 5+ years (2020-2025), 5+ million articles

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


class LTNNewsSpider(scrapy.Spider):
    """
    Scrapy spider for Liberty Times Net (自由時報) with multi-mode strategy.

    Historical Discovery (2025-11-18):
        - Sequential IDs 1 to 5,250,000+ are accessible
        - ID 1,000,000: 2020-09-28 (over 5 years ago!)
        - ID 4,000,000: 2022-07-21
        - ID 5,250,000: 2025-11-18 (current)
        - Estimated 5+ million articles available

    URL Structure:
        - Article: https://news.ltn.com.tw/news/{category}/breakingnews/{ID}
        - List page: https://news.ltn.com.tw/list/breakingnews/{category}

    Modes:
        1. list: Fast crawl from list pages (recent articles, daily updates)
        2. sequential: Systematic ID-based crawl (complete historical coverage)
        3. hybrid: Discover recent + fill historical gaps

    Categories:
        politics (政治), business (財經), society (社會), entertainment (娛樂),
        sports (體育), life (生活), world (國際), opinion (評論)

    Usage:
        # List mode (recent articles)
        scrapy runspider ltn_spider.py -a mode=list -a days=7 -o ltn.jsonl

        # Sequential mode (historical crawl)
        scrapy runspider ltn_spider.py \
            -a mode=sequential \
            -a start_id=1000000 \
            -a end_id=1050000 \
            -o ltn_historical.jsonl

        # Hybrid mode (discover + fill gaps)
        scrapy runspider ltn_spider.py -a mode=hybrid -a days=7 -o ltn.jsonl

    Complexity:
        Time: O(N) where N = ID range for sequential, or pages for list mode
        Space: O(A) where A = articles scraped
    """

    name = 'ltn_news'
    allowed_domains = ['news.ltn.com.tw', 'ltn.com.tw']

    # Custom settings optimized for speed (no Playwright overhead!)
    custom_settings = {
        'DOWNLOAD_DELAY': 1.5,  # Respectful delay
        'CONCURRENT_REQUESTS_PER_DOMAIN': 3,  # Faster without Playwright
        'ROBOTSTXT_OBEY': True,
        'USER_AGENT': 'CNIRS Academic Research Bot (Educational Use)',
        'HTTPERROR_ALLOW_404': True,  # Allow 404 to reach errback
        'RETRY_TIMES': 2,
        'RETRY_HTTP_CODES': [500, 502, 503, 504, 408, 429],

        # Output settings
        'FEEDS': {
            'data/raw/ltn_news_%(time)s.jsonl': {
                'format': 'jsonlines',
                'encoding': 'utf8',
                'store_empty': False,
                'overwrite': False,
            }
        },
    }

    # Category mapping
    CATEGORIES = {
        'politics': '政治',
        'business': '財經',
        'society': '社會',
        'entertainment': '娛樂',
        'sports': '體育',
        'life': '生活',
        'world': '國際',
        'opinion': '評論',
        'all': '全部',
    }

    # Historical data range (discovered 2025-11-18)
    EARLIEST_ID = 1  # ID 1 exists but very old
    KNOWN_HISTORICAL_START = 1000000  # 2020-09-28 confirmed
    CURRENT_ID_ESTIMATE = 5250000  # Will be higher over time

    def __init__(self,
                 mode: str = 'list',
                 category: str = 'all',
                 days: int = 7,
                 start_id: int = None,
                 end_id: int = None,
                 start_date: str = None,
                 end_date: str = None,
                 *args, **kwargs):
        """
        Initialize LTN spider with multi-mode support.

        Args:
            mode: Crawling mode ('list', 'sequential', 'hybrid')
            category: News category (politics/business/society/all)
            days: Number of days to crawl (for list/hybrid mode)
            start_id: Start article ID (for sequential mode)
            end_id: End article ID (for sequential mode)
            start_date: Start date in YYYY-MM-DD (for list mode)
            end_date: End date in YYYY-MM-DD (for list mode)

        Example:
            >>> spider = LTNNewsSpider(mode='sequential', start_id=1000000, end_id=1100000)
            >>> spider = LTNNewsSpider(mode='list', category='politics', days=7)
            >>> spider = LTNNewsSpider(mode='hybrid', days=30)
        """
        super(LTNNewsSpider, self).__init__(*args, **kwargs)

        # Set mode
        self.mode = mode.lower()
        if self.mode not in ['list', 'sequential', 'hybrid']:
            logger.warning(f"Invalid mode '{mode}', using 'list'")
            self.mode = 'list'

        # Validate and set category
        if category not in self.CATEGORIES:
            logger.warning(f"Invalid category '{category}', using 'all'")
            category = 'all'
        self.category = category
        self.category_name = self.CATEGORIES[category]

        # Set date range (for list/hybrid mode)
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

        # Set ID range (for sequential/hybrid mode)
        self.start_id = int(start_id) if start_id else self.KNOWN_HISTORICAL_START
        self.end_id = int(end_id) if end_id else None  # Will auto-detect if None

        # Statistics
        self.articles_count = 0
        self.failed_count = 0
        self.not_found_count = 0
        self.seen_ids = set()

        logger.info("=" * 70)
        logger.info("LTN Spider Initialized")
        logger.info("=" * 70)
        logger.info(f"Mode: {self.mode}")
        logger.info(f"Category: {self.category_name} ({self.category})")
        if self.mode in ['list', 'hybrid']:
            logger.info(f"Date Range: {self.start_date.date()} to {self.end_date.date()}")
        if self.mode in ['sequential', 'hybrid']:
            logger.info(f"ID Range: {self.start_id} to {self.end_id if self.end_id else 'auto-detect'}")
        logger.info("=" * 70)

    def start_requests(self):
        """
        Generate start requests based on mode.

        Three modes:
        1. list: Crawl from list pages (fast, recent articles)
        2. sequential: Direct ID-based crawl (complete historical coverage)
        3. hybrid: Combine both strategies

        Yields:
            scrapy.Request: Initial requests
        """
        if self.mode == 'list':
            # List mode: Start with news list pages
            if self.category == 'all':
                base_url = "https://news.ltn.com.tw/list/breakingnews"
            else:
                base_url = f"https://news.ltn.com.tw/list/breakingnews/{self.category}"

            logger.info(f"List mode: Starting with {base_url}")

            yield scrapy.Request(
                url=base_url,
                callback=self.parse_list_page,
                errback=self.handle_error,
                meta={'page': 1},
                dont_filter=True
            )

        elif self.mode == 'sequential':
            # Sequential mode: Direct ID crawling
            if not self.end_id:
                # Auto-detect current max ID from list page first
                logger.info("Sequential mode: Auto-detecting max ID from list page...")
                yield scrapy.Request(
                    url="https://news.ltn.com.tw/list/breakingnews",
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
                url="https://news.ltn.com.tw/list/breakingnews",
                callback=self.parse_list_hybrid,
                errback=self.handle_error,
                dont_filter=True
            )

    def _generate_sequential_requests(self):
        """
        Generate sequential article requests by ID.

        Yields:
            scrapy.Request: Article requests
        """
        for article_id in range(self.start_id, self.end_id + 1):
            # Use 'politics' as default category (IDs work across categories)
            url = f"https://news.ltn.com.tw/news/politics/breakingnews/{article_id}"

            yield scrapy.Request(
                url=url,
                callback=self.parse_article,
                errback=self.handle_error,
                meta={'article_id': article_id, 'mode': 'sequential'},
                dont_filter=True
            )

    def parse_list_page(self, response):
        """
        Parse news list page to extract article URLs.

        Args:
            response: Scrapy response object

        Yields:
            scrapy.Request: Article and pagination requests
        """
        page_num = response.meta.get('page', 1)
        logger.info(f"Parsing list page {page_num}: {response.url}")

        # Extract article links
        article_links = response.css('div.boxTitle a::attr(href), ul.list li a::attr(href)').getall()

        articles_found = 0
        for link in article_links:
            if not link or '/breakingnews/' not in link:
                continue

            article_url = response.urljoin(link)

            # Extract ID from URL
            match = re.search(r'/breakingnews/(\d+)', article_url)
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

        logger.info(f"Found {articles_found} articles on page {page_num}")

        # Pagination: Try next page (limit to 10 pages for efficiency)
        if articles_found > 0 and page_num < 10:
            next_page = page_num + 1
            if '?' in response.url:
                next_url = re.sub(r'page=\d+', f'page={next_page}', response.url)
                if f'page={next_page}' not in next_url:
                    next_url = f"{response.url}&page={next_page}"
            else:
                next_url = f"{response.url}?page={next_page}"

            logger.info(f"Following pagination to page {next_page}")

            yield scrapy.Request(
                url=next_url,
                callback=self.parse_list_page,
                errback=self.handle_error,
                meta={'page': next_page},
                dont_filter=True
            )

    def parse_list_for_max_id(self, response):
        """
        Parse list page to auto-detect maximum article ID.

        Used in sequential mode when end_id is not specified.

        Args:
            response: Scrapy response object

        Yields:
            scrapy.Request: Sequential crawling requests
        """
        article_links = response.css('a[href*="/breakingnews/"]::attr(href)').getall()

        ids = []
        for link in article_links:
            match = re.search(r'/breakingnews/(\d+)', link)
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
        Parse list page for hybrid mode: discover recent + fill gaps.

        Strategy:
        1. Extract all article IDs from list page
        2. Crawl discovered articles
        3. Fill gaps between start_id and min discovered ID

        Args:
            response: Scrapy response object

        Yields:
            scrapy.Request: Article requests
        """
        article_links = response.css('a[href*="/breakingnews/"]::attr(href)').getall()

        discovered_ids = set()
        for link in article_links:
            match = re.search(r'/breakingnews/(\d+)', link)
            if match:
                discovered_ids.add(int(match.group(1)))

        logger.info(f"Hybrid mode: Discovered {len(discovered_ids)} article IDs from list page")

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
            url = f"https://news.ltn.com.tw/news/politics/breakingnews/{article_id}"
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

            for article_id in range(self.start_id, min_discovered_id):
                if article_id not in discovered_ids:
                    url = f"https://news.ltn.com.tw/news/politics/breakingnews/{article_id}"
                    yield scrapy.Request(
                        url=url,
                        callback=self.parse_article,
                        errback=self.handle_error,
                        meta={'article_id': article_id, 'mode': 'hybrid'},
                        dont_filter=True
                    )

    def parse_article(self, response):
        """
        Parse individual LTN article page.

        Extraction Strategy:
        1. Title: h1 tag (multiple fallbacks)
        2. Content: div.text p (multiple fallbacks)
        3. Date: span.time, div.time, or URL
        4. Author: div.author, span.reporter
        5. Category: from URL
        6. Tags: div.keyword a
        7. Image: div.text img

        Args:
            response: Scrapy response object

        Yields:
            dict: Article data
        """
        try:
            # Extract article ID
            article_id = response.meta.get('article_id')
            if not article_id:
                match = re.search(r'/breakingnews/(\d+)', response.url)
                article_id = int(match.group(1)) if match else None

            article = {
                'article_id': f"LTN_{article_id}",
                'url': response.url,
                'source': 'LTN',
                'source_name': '自由時報',
                'crawled_at': datetime.now().isoformat(),
            }

            # Title - multiple selectors
            title = (response.css('h1::text').get() or
                    response.css('div.articlebody h1::text').get() or
                    response.css('div.whitecon h1::text').get() or
                    response.css('article h1::text').get())
            article['title'] = self.clean_text(title)

            # Content paragraphs - multiple selectors
            content_paragraphs = (
                response.css('div.text p::text').getall() or
                response.css('div.articlebody p::text').getall() or
                response.css('div.content p::text').getall() or
                response.css('article p::text').getall()
            )
            article['content'] = ' '.join([self.clean_text(p) for p in content_paragraphs if p])

            # Author
            author = (response.css('div.author::text').get() or
                     response.css('span.reporter::text').get() or
                     response.css('div.news_editor::text').get())
            article['author'] = self.clean_text(author) if author else '自由時報'

            # Publish date
            date_text = (response.css('span.time::text').get() or
                        response.css('div.time::text').get() or
                        response.css('span.date::text').get())
            article['publish_date'] = self.parse_date(date_text)

            # Category from URL
            article['category'] = self.extract_category_from_url(response.url)
            article['category_name'] = self.CATEGORIES.get(article['category'], article['category'])

            # Tags
            tags = response.css('div.keyword a::text, div.tag a::text').getall()
            article['tags'] = [self.clean_text(tag) for tag in tags if tag]

            # Image
            image_url = (response.css('div.text img::attr(data-src)').get() or
                        response.css('div.text img::attr(src)').get() or
                        response.css('div.articlebody img::attr(src)').get())
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
                # 404 is expected in sequential mode (many IDs don't exist)
                self.not_found_count += 1
                if self.mode == 'sequential' and self.not_found_count % 1000 == 0:
                    logger.info(f"404 count: {self.not_found_count} (expected in sequential mode)")
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
        logger.info("LTN Spider Finished")
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
            # LTN formats: "2025/11/17 15:30", "2025-11-17 15:30:00"
            formats = [
                '%Y/%m/%d %H:%M',
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%dT%H:%M:%S',
                '%Y/%m/%d',
                '%Y-%m-%d',
            ]

            for fmt in formats:
                try:
                    dt = datetime.strptime(date_str.strip()[:19], fmt)
                    return dt.strftime('%Y-%m-%d')
                except ValueError:
                    continue

            # Extract YYYY-MM-DD or YYYY/MM/DD pattern
            match = re.search(r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})', date_str)
            if match:
                return f"{match.group(1)}-{match.group(2):0>2}-{match.group(3):0>2}"

        except Exception as e:
            logger.warning(f"Could not parse date: {date_str}")

        return None

    @staticmethod
    def extract_category_from_url(url: str) -> str:
        """
        Extract category from article URL.

        Args:
            url: Article URL

        Returns:
            Category code
        """
        # LTN URL: /news/{category}/...
        categories = ['politics', 'business', 'society', 'entertainment',
                     'sports', 'life', 'world', 'opinion']

        for category in categories:
            if f'/news/{category}/' in url:
                return category
        return 'unknown'


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
        LTNNewsSpider,
        mode='sequential',
        start_id=1000000,
        end_id=1010000  # Small range for testing
    )

    process.start()
