"""
PTS News Spider

Spider for crawling news articles from Public Television Service (PTS).

Author: CNIRS Development Team
License: Educational Use Only
"""

import scrapy
from datetime import datetime, timedelta
import json
import logging
import re
import html
from typing import Optional, List
from scrapy.spidermiddlewares.httperror import HttpError
from twisted.internet.error import DNSLookupError, TimeoutError, TCPTimedOutError

logger = logging.getLogger(__name__)


class PTSNewsSpider(scrapy.Spider):
    """
    Scrapy spider for crawling PTS (Public Television Service) news articles.

    Target: https://news.pts.org.tw/
    Date Range: 2022-01-01 to 2024-12-31
    Categories: All news categories

    Usage:
        scrapy runspider pts_spider.py -o output.jsonl

        # With date range
        scrapy runspider pts_spider.py \
            -a start_date=2024-01-01 \
            -a end_date=2024-12-31 \
            -o pts_2024.jsonl
    """

    name = 'pts_news'
    allowed_domains = ['news.pts.org.tw']

    # Custom settings
    custom_settings = {
        'DOWNLOAD_DELAY': 1.5,  # Reduced for efficiency
        'CONCURRENT_REQUESTS_PER_DOMAIN': 3,  # Increased for sequential crawling
        'ROBOTSTXT_OBEY': True,
        'USER_AGENT': 'CNIRS Academic Research Bot (Educational Use)',
        'HTTPERROR_ALLOW_404': True,  # Allow 404 responses to reach errback
        'FEEDS': {
            'data/raw/pts_news_%(time)s.jsonl': {
                'format': 'jsonlines',
                'encoding': 'utf8',
                'store_empty': False,
                'overwrite': False,
            }
        }
    }

    # Historical data availability
    # PTS article ID range: ~690000 (Apr 2024) to current (~781900 as of Nov 2025)
    # Time span: Approximately 19 months (1.6 years) of historical data
    EARLIEST_ID = 690000  # April 2024
    CURRENT_ID_ESTIMATE = 782000  # Will be higher over time

    def __init__(self, days: int = 7, start_id: int = None, end_id: int = None,
                 max_articles: int = 500, mode: str = 'dailynews', *args, **kwargs):
        """
        Initialize spider with crawling parameters.

        Args:
            days: Number of recent days to crawl (default: 7, used for dailynews mode)
            start_id: Starting article ID for sequential crawl (default: EARLIEST_ID)
            end_id: Ending article ID for sequential crawl (default: auto-detect from /dailynews)
            max_articles: Maximum articles to crawl (default: 500)
            mode: Crawling mode - 'dailynews', 'sequential', or 'hybrid' (default: 'dailynews')

        Modes:
            - dailynews: Fast crawl of recent articles from /dailynews page (~25 articles)
            - sequential: Systematic crawl by sequential IDs (for historical data)
            - hybrid: Combine dailynews + sequential ID fill-in

        Note:
            PTS keeps approximately 19 months of historical data (Apr 2024 onwards).
            Sequential mode can access ~85,000-90,000 articles.
        """
        super(PTSNewsSpider, self).__init__(*args, **kwargs)

        self.days = int(days)
        self.max_articles = int(max_articles)
        self.mode = mode

        # For sequential ID crawling
        self.start_id = int(start_id) if start_id else self.EARLIEST_ID
        self.end_id = int(end_id) if end_id else None  # Auto-detect if not provided

        logger.info(f"PTS Spider initialized in '{self.mode}' mode")
        if self.mode == 'sequential':
            logger.info(f"Sequential ID range: {self.start_id} to {self.end_id or 'auto-detect'}")
        else:
            logger.info(f"Target: Recent {self.days} days")

        # Statistics
        self.articles_count = 0
        self.failed_count = 0
        self.not_found_count = 0

    def start_requests(self):
        """
        Generate start requests based on crawling mode.

        Modes:
            - dailynews: Extract recent articles from /dailynews page
            - sequential: Systematic crawl by article ID range
            - hybrid: Combine dailynews discovery + sequential fill-in

        Yields:
            scrapy.Request: Requests for dailynews page or article pages
        """
        if self.mode == 'dailynews':
            # Mode 1: Fast crawl of recent articles
            logger.info("Mode: dailynews - Discovering recent articles from /dailynews")
            yield scrapy.Request(
                url="https://news.pts.org.tw/dailynews",
                callback=self.parse_dailynews,
                errback=self.handle_error,
                dont_filter=True
            )

        elif self.mode == 'sequential':
            # Mode 2: Sequential ID crawling for historical data
            # Auto-detect end_id from /dailynews if not specified
            if not self.end_id:
                logger.info("Auto-detecting latest article ID from /dailynews")
                yield scrapy.Request(
                    url="https://news.pts.org.tw/dailynews",
                    callback=self.parse_dailynews_for_max_id,
                    errback=self.handle_error,
                    dont_filter=True
                )
            else:
                # Start sequential crawling immediately
                self._start_sequential_crawl()
                for request in self._generate_sequential_requests():
                    yield request

        elif self.mode == 'hybrid':
            # Mode 3: Hybrid - discover recent + fill gaps
            logger.info("Mode: hybrid - Combining dailynews + sequential crawling")
            # First, discover recent articles
            yield scrapy.Request(
                url="https://news.pts.org.tw/dailynews",
                callback=self.parse_dailynews_hybrid,
                errback=self.handle_error,
                dont_filter=True
            )
        else:
            logger.error(f"Unknown mode: {self.mode}. Falling back to dailynews.")
            yield scrapy.Request(
                url="https://news.pts.org.tw/dailynews",
                callback=self.parse_dailynews,
                errback=self.handle_error,
                dont_filter=True
            )

    def _start_sequential_crawl(self):
        """Log start of sequential crawling."""
        logger.info(f"Starting sequential ID crawling: {self.start_id} to {self.end_id}")
        logger.info(f"Expected range: ~{self.end_id - self.start_id + 1} article IDs")

    def _generate_sequential_requests(self):
        """
        Generate sequential article requests.

        Yields:
            scrapy.Request: Requests for sequential article IDs
        """
        for article_id in range(self.start_id, self.end_id + 1):
            url = f"https://news.pts.org.tw/article/{article_id}"
            yield scrapy.Request(
                url=url,
                callback=self.parse_article,
                errback=self.handle_error,
                meta={'article_id': article_id, 'mode': 'sequential'},
                dont_filter=True
            )

    def parse_dailynews(self, response):
        """
        Parse /dailynews page to extract article URLs.

        Args:
            response: Scrapy response object

        Yields:
            scrapy.Request: Requests for individual article pages
        """
        # Extract article links: href="https://news.pts.org.tw/article/{ID}"
        article_links = response.css('a[href*="/article/"]::attr(href)').getall()

        if not article_links:
            logger.warning("No articles found on /dailynews page")
            return

        # Deduplicate and extract IDs
        seen_ids = set()
        for link in article_links:
            # Extract ID from URL
            match = re.search(r'/article/(\d+)', link)
            if match:
                article_id = int(match.group(1))
                if article_id not in seen_ids:
                    seen_ids.add(article_id)
                    article_url = f"https://news.pts.org.tw/article/{article_id}"

                    yield scrapy.Request(
                        url=article_url,
                        callback=self.parse_article,
                        errback=self.handle_error,
                        meta={'article_id': article_id},
                        dont_filter=True
                    )

        logger.info(f"Found {len(seen_ids)} unique articles on /dailynews")

    def parse_dailynews_for_max_id(self, response):
        """
        Parse /dailynews to extract maximum article ID for sequential crawling.

        Args:
            response: Scrapy response object

        Yields:
            scrapy.Request: Sequential crawling requests after detecting max ID
        """
        article_links = response.css('a[href*="/article/"]::attr(href)').getall()

        if not article_links:
            logger.warning("No articles found on /dailynews for max ID detection")
            # Fallback to estimated current ID
            self.end_id = self.CURRENT_ID_ESTIMATE
        else:
            # Extract all IDs and find maximum
            ids = []
            for link in article_links:
                match = re.search(r'/article/(\d+)', link)
                if match:
                    ids.append(int(match.group(1)))

            if ids:
                self.end_id = max(ids)
                logger.info(f"Auto-detected maximum article ID: {self.end_id}")
            else:
                self.end_id = self.CURRENT_ID_ESTIMATE
                logger.warning(f"Could not detect max ID, using estimate: {self.end_id}")

        # Now start sequential crawling
        self._start_sequential_crawl()
        for request in self._generate_sequential_requests():
            yield request

    def parse_dailynews_hybrid(self, response):
        """
        Parse /dailynews for hybrid mode: discover recent + fill gaps.

        Args:
            response: Scrapy response object

        Yields:
            scrapy.Request: Requests for discovered articles + gap filling
        """
        article_links = response.css('a[href*="/article/"]::attr(href)').getall()

        if not article_links:
            logger.warning("No articles found on /dailynews (hybrid mode)")
            return

        # Extract and deduplicate IDs
        discovered_ids = set()
        for link in article_links:
            match = re.search(r'/article/(\d+)', link)
            if match:
                discovered_ids.add(int(match.group(1)))

        if not discovered_ids:
            logger.warning("No valid article IDs extracted (hybrid mode)")
            return

        max_discovered_id = max(discovered_ids)
        min_discovered_id = min(discovered_ids)

        logger.info(f"Hybrid mode: Discovered {len(discovered_ids)} articles")
        logger.info(f"ID range: {min_discovered_id} to {max_discovered_id}")

        # Strategy: Crawl discovered IDs + fill gaps from start_id
        # 1. First, yield discovered articles
        for article_id in sorted(discovered_ids):
            url = f"https://news.pts.org.tw/article/{article_id}"
            yield scrapy.Request(
                url=url,
                callback=self.parse_article,
                errback=self.handle_error,
                meta={'article_id': article_id, 'mode': 'hybrid_discovered'},
                dont_filter=True
            )

        # 2. Then, fill gaps from start_id to min_discovered_id
        if self.start_id < min_discovered_id:
            gap_count = min_discovered_id - self.start_id
            logger.info(f"Filling {gap_count} gaps from {self.start_id} to {min_discovered_id}")

            for article_id in range(self.start_id, min_discovered_id):
                if article_id not in discovered_ids:  # Skip already discovered
                    url = f"https://news.pts.org.tw/article/{article_id}"
                    yield scrapy.Request(
                        url=url,
                        callback=self.parse_article,
                        errback=self.handle_error,
                        meta={'article_id': article_id, 'mode': 'hybrid_gap'},
                        dont_filter=True
                    )

    def parse_article(self, response):
        """
        Parse individual article page.

        Args:
            response: Scrapy response object

        Yields:
            dict: Article data
        """
        try:
            # Extract article ID from meta or URL
            article_id = response.meta.get('article_id')
            if not article_id:
                # Extract from URL: /article/781889
                match = re.search(r'/article/(\d+)', response.url)
                article_id = int(match.group(1)) if match else None

            # Extract article metadata
            article = {
                'article_id': f"PTS_{article_id}" if article_id else f"PTS_{response.url.split('/')[-1]}",
                'url': response.url,
                'source': 'PTS',
                'source_name': '公視新聞',
                'crawled_at': datetime.now().isoformat(),
            }

            # Title - using meta property as primary source (more reliable)
            title = response.css('meta[property="og:title"]::attr(content)').get()
            if not title:
                title = response.css('h1::text').get()
            if not title:
                title = response.css('title::text').get()
                if title:
                    # Remove " | 公視新聞網 PNN" suffix
                    title = title.replace('｜ 公視新聞網 PNN', '').replace('| 公視新聞網 PNN', '')
            article['title'] = self.clean_text(title)

            # Content - multiple fallback strategies
            # Strategy 1: JSON-LD structured data
            content_json = response.css('script[type="application/ld+json"]::text').get()
            content_paragraphs = []
            if content_json:
                try:
                    data = json.loads(content_json)
                    if isinstance(data, dict) and 'articleBody' in data:
                        content_paragraphs = [data['articleBody']]
                except:
                    pass

            # Strategy 2: Direct paragraph extraction
            if not content_paragraphs:
                # Try various selectors observed in PTS HTML
                content_paragraphs = response.css('div.article-content p::text').getall()
            if not content_paragraphs:
                content_paragraphs = response.css('article p::text').getall()
            if not content_paragraphs:
                # Last resort: get from meta description
                meta_desc = response.css('meta[name="description"]::attr(content)').get()
                if meta_desc:
                    content_paragraphs = [meta_desc]

            article['content'] = ' '.join([self.clean_text(p) for p in content_paragraphs])

            # Author - from meta property
            author = response.css('meta[name="author"]::attr(content)').get()
            if not author:
                # Try extracting from JSON-LD
                if content_json:
                    try:
                        data = json.loads(content_json)
                        if isinstance(data, dict) and 'author' in data:
                            if isinstance(data['author'], list) and len(data['author']) > 0:
                                author = data['author'][0].get('name', '')
                            elif isinstance(data['author'], dict):
                                author = data['author'].get('name', '')
                    except:
                        pass
            article['author'] = self.clean_text(author) if author else ''

            # Publish date - from meta property (most reliable)
            publish_date = response.css('meta[property="article:published_time"]::attr(content)').get()
            if not publish_date:
                publish_date = response.css('meta[property="pubdate"]::attr(content)').get()
            article['publish_date'] = self.parse_date(publish_date)

            # Category - from meta or breadcrumb
            category = response.css('meta[property="article:section"]::attr(content)').get()
            if not category:
                breadcrumb = response.css('nav[aria-label="breadcrumb"] a::text, div.breadcrumb a::text').getall()
                if breadcrumb and len(breadcrumb) > 1:
                    category = breadcrumb[1].strip()
            article['category'] = self.clean_text(category) if category else '其他'

            # Tags/Keywords
            keywords = response.css('meta[name="keywords"]::attr(content)').get()
            if keywords:
                article['tags'] = [k.strip() for k in keywords.split(',')]
            else:
                tags = response.css('div.tag a::text, div.tags a::text').getall()
                article['tags'] = [tag.strip() for tag in tags] if tags else []

            # Image URL - from meta property
            image_url = response.css('meta[property="og:image"]::attr(content)').get()
            if not image_url:
                image_url = response.css('figure img::attr(src), article img::attr(src)').get()
            article['image_url'] = response.urljoin(image_url) if image_url else None

            # Validation
            if not article['title'] or not article['content']:
                logger.warning(f"Missing title or content for {response.url}")
                self.failed_count += 1
                return

            if len(article['content']) < 100:  # Increased minimum length
                logger.warning(f"Content too short ({len(article['content'])} chars) for {response.url}")
                self.failed_count += 1
                return

            self.articles_count += 1
            if self.articles_count % 10 == 0:  # Log every 10 articles
                logger.info(f"Successfully parsed {self.articles_count} articles")
            else:
                logger.debug(f"Successfully parsed article #{self.articles_count}: {article['title'][:50]}")

            yield article

        except Exception as e:
            logger.error(f"Error parsing article {response.url}: {e}")
            self.failed_count += 1

    def handle_error(self, failure):
        """
        Handle request errors with detailed classification.

        Args:
            failure: Twisted failure object
        """
        # Check if it's a 404 (expected for non-existent article IDs in sequential mode)
        if failure.check(HttpError):
            response = failure.value.response
            if response.status == 404:
                # 404 is expected when article ID doesn't exist (sequential mode)
                self.not_found_count += 1
                if self.mode == 'sequential' and self.not_found_count % 500 == 0:
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
        Called when spider is closed.

        Args:
            reason: Reason for closing
        """
        total_attempts = self.articles_count + self.failed_count + self.not_found_count
        logger.info("=" * 70)
        logger.info("PTS Spider Finished")
        logger.info("=" * 70)
        logger.info(f"Mode: {self.mode}")
        logger.info(f"Reason: {reason}")
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

    # Utility methods

    @staticmethod
    def clean_text(text: Optional[str]) -> str:
        """Clean text by removing extra whitespace and unescaping HTML entities."""
        if not text:
            return ""
        # Decode HTML entities like &#20160; -> 什
        text = html.unescape(text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    @staticmethod
    def parse_date(date_str: Optional[str]) -> Optional[str]:
        """Parse date string to ISO format."""
        if not date_str:
            return None

        try:
            formats = [
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%d %H:%M:%S',
                '%Y/%m/%d %H:%M',
                '%Y-%m-%d',
                '%Y/%m/%d',
            ]

            for fmt in formats:
                try:
                    dt = datetime.strptime(date_str.strip()[:19], fmt)
                    return dt.strftime('%Y-%m-%d')
                except ValueError:
                    continue

            # Extract date pattern
            match = re.search(r'(\d{4})[/-](\d{2})[/-](\d{2})', date_str)
            if match:
                return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"

            logger.warning(f"Could not parse date: {date_str}")
            return None

        except Exception as e:
            logger.error(f"Error parsing date '{date_str}': {e}")
            return None

    @staticmethod
    def translate_category(category: str) -> str:
        """Translate category from English to Chinese."""
        category_map = {
            'politics': '政治',
            'finance': '財經',
            'local': '社會',
            'international': '國際',
            'culture': '文化',
            'education': '教育',
            'life': '生活',
            'welfare': '福利',
            'environment': '環境',
        }
        return category_map.get(category.lower(), category)


# For standalone execution
if __name__ == '__main__':
    from scrapy.crawler import CrawlerProcess
    from scrapy.utils.project import get_project_settings

    settings = get_project_settings()
    settings.update({
        'DOWNLOAD_DELAY': 2,
        'CONCURRENT_REQUESTS_PER_DOMAIN': 2,
        'ROBOTSTXT_OBEY': True,
        'USER_AGENT': 'CNIRS Academic Research Bot (Educational Use)',
        'LOG_LEVEL': 'INFO',
    })

    process = CrawlerProcess(settings)
    process.crawl(
        PTSNewsSpider,
        start_date='2024-01-01',
        end_date='2024-01-31'
    )
    process.start()
