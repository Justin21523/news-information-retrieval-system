"""
CNA News Spider

Spider for crawling news articles from Central News Agency (CNA).

Author: CNIRS Development Team
License: Educational Use Only
"""

import scrapy
from datetime import datetime, timedelta
import json
import logging
import re
from typing import Optional, List

logger = logging.getLogger(__name__)


class CNANewsSpider(scrapy.Spider):
    """
    Scrapy spider for crawling CNA (Central News Agency) news articles.

    Target: https://www.cna.com.tw/
    Date Range: 2022-01-01 to 2024-12-31
    Categories: All (政治, 財經, 科技, 社會, 國際, 生活, 文化, 運動)

    Usage:
        scrapy runspider cna_spider.py -o output.jsonl

        # With date range
        scrapy runspider cna_spider.py \
            -a start_date=2024-01-01 \
            -a end_date=2024-12-31 \
            -o cna_2024.jsonl
    """

    name = 'cna_news'
    allowed_domains = ['cna.com.tw']

    # Custom settings (can be overridden by scrapy_settings.py)
    custom_settings = {
        'DOWNLOAD_DELAY': 2,  # 2 seconds between requests
        'CONCURRENT_REQUESTS_PER_DOMAIN': 2,
        'ROBOTSTXT_OBEY': True,
        'USER_AGENT': 'CNIRS Academic Research Bot (Educational Use)',
        'FEEDS': {
            'data/raw/cna_news_%(time)s.jsonl': {
                'format': 'jsonlines',
                'encoding': 'utf8',
                'store_empty': False,
                'overwrite': False,
            }
        }
    }

    def __init__(self, start_date: str = '2022-01-01', end_date: str = '2024-12-31', *args, **kwargs):
        """
        Initialize spider with date range.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        """
        super(CNANewsSpider, self).__init__(*args, **kwargs)

        try:
            self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
            self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError as e:
            logger.error(f"Invalid date format: {e}")
            self.start_date = datetime(2022, 1, 1)
            self.end_date = datetime(2024, 12, 31)

        logger.info(f"CNA Spider initialized: {self.start_date.date()} to {self.end_date.date()}")

        # Statistics
        self.articles_count = 0
        self.failed_count = 0

    def start_requests(self):
        """
        Generate start requests for all dates in range.

        Yields:
            scrapy.Request: Requests for daily news list pages
        """
        current_date = self.start_date

        while current_date <= self.end_date:
            # CNA daily news list URL format:
            # https://www.cna.com.tw/list/aall/YYYYMMDD.aspx
            date_str = current_date.strftime('%Y%m%d')
            url = f"https://www.cna.com.tw/list/aall/{date_str}.aspx"

            yield scrapy.Request(
                url=url,
                callback=self.parse_list_page,
                errback=self.handle_error,
                meta={'date': current_date.strftime('%Y-%m-%d')},
                dont_filter=True
            )

            current_date += timedelta(days=1)

    def parse_list_page(self, response):
        """
        Parse daily news list page to extract article URLs.

        Args:
            response: Scrapy response object

        Yields:
            scrapy.Request: Requests for individual article pages
        """
        date = response.meta.get('date', 'unknown')

        # Extract article links
        # CNA structure: <div class="mainList"><a href="/news/...">
        article_links = response.css('div.mainList ul li a::attr(href)').getall()

        if not article_links:
            logger.warning(f"No articles found for date {date}")
            return

        logger.info(f"Found {len(article_links)} articles for date {date}")

        for link in article_links:
            # Convert relative URL to absolute URL
            article_url = response.urljoin(link)

            # Check if it's a news article (not video/photo gallery)
            if '/news/' in article_url:
                yield scrapy.Request(
                    url=article_url,
                    callback=self.parse_article,
                    errback=self.handle_error,
                    meta={'date': date}
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
            # Extract article metadata
            article = {
                'url': response.url,
                'source': 'CNA',
                'crawled_at': datetime.now().isoformat(),
            }

            # Title
            title = response.css('h1.centralContent span::text').get()
            if not title:
                title = response.css('h1::text').get()
            article['title'] = self.clean_text(title)

            # Content paragraphs
            content_paragraphs = response.css('div.paragraph p::text').getall()
            if not content_paragraphs:
                # Fallback: try different selectors
                content_paragraphs = response.css('article p::text').getall()
            article['content'] = ' '.join([self.clean_text(p) for p in content_paragraphs])

            # Author
            author = response.css('div.author::text').get()
            if not author:
                author = response.css('div.editor::text').get()
            article['author'] = self.clean_text(author)

            # Publish date
            publish_date = response.css('div.updatetime span::text').get()
            if not publish_date:
                publish_date = response.css('time::attr(datetime)').get()
            article['publish_date'] = self.parse_date(publish_date)

            # Category
            breadcrumb = response.css('div.breadcrumb a::text').getall()
            if breadcrumb:
                article['category'] = breadcrumb[-1].strip()
            else:
                # Try to extract from URL
                article['category'] = self.extract_category_from_url(response.url)

            # Tags/Keywords (if available)
            tags = response.css('div.keywordTag a::text').getall()
            article['tags'] = [tag.strip() for tag in tags] if tags else []

            # Image URL (main image)
            image_url = response.css('div.fullPic img::attr(src)').get()
            if not image_url:
                image_url = response.css('article img::attr(src)').get()
            article['image_url'] = response.urljoin(image_url) if image_url else None

            # Validation
            if not article['title'] or not article['content']:
                logger.warning(f"Missing title or content for {response.url}")
                self.failed_count += 1
                return

            if len(article['content']) < 50:
                logger.warning(f"Content too short for {response.url}")
                self.failed_count += 1
                return

            self.articles_count += 1
            logger.debug(f"Successfully parsed article #{self.articles_count}: {article['title'][:50]}")

            yield article

        except Exception as e:
            logger.error(f"Error parsing article {response.url}: {e}")
            self.failed_count += 1

    def handle_error(self, failure):
        """
        Handle request errors.

        Args:
            failure: Twisted failure object
        """
        logger.error(f"Request failed: {failure.request.url}")
        logger.error(f"Error: {repr(failure.value)}")
        self.failed_count += 1

    def closed(self, reason):
        """
        Called when spider is closed.

        Args:
            reason: Reason for closing
        """
        logger.info("=" * 70)
        logger.info("CNA Spider Finished")
        logger.info("=" * 70)
        logger.info(f"Reason: {reason}")
        logger.info(f"Articles crawled: {self.articles_count}")
        logger.info(f"Failed requests: {self.failed_count}")
        logger.info(f"Success rate: {100 * self.articles_count / max(1, self.articles_count + self.failed_count):.1f}%")
        logger.info("=" * 70)

    # Utility methods

    @staticmethod
    def clean_text(text: Optional[str]) -> str:
        """
        Clean text by removing extra whitespace and special characters.

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        text = text.strip()

        return text

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
            # Try different formats
            formats = [
                '%Y/%m/%d %H:%M',     # 2024/03/15 10:30
                '%Y-%m-%d %H:%M:%S',  # 2024-03-15 10:30:00
                '%Y-%m-%dT%H:%M:%S',  # ISO 8601
                '%Y/%m/%d',           # 2024/03/15
                '%Y-%m-%d',           # 2024-03-15
            ]

            for fmt in formats:
                try:
                    dt = datetime.strptime(date_str.strip()[:19], fmt)
                    return dt.strftime('%Y-%m-%d')
                except ValueError:
                    continue

            # If all formats fail, try to extract YYYY-MM-DD pattern
            match = re.search(r'(\d{4})[/-](\d{2})[/-](\d{2})', date_str)
            if match:
                return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"

            logger.warning(f"Could not parse date: {date_str}")
            return None

        except Exception as e:
            logger.error(f"Error parsing date '{date_str}': {e}")
            return None

    @staticmethod
    def extract_category_from_url(url: str) -> str:
        """
        Extract category from URL.

        Args:
            url: Article URL

        Returns:
            Category name
        """
        # CNA URL structure: /news/CATEGORY/YYYYMMDDXXXXXX.aspx
        category_map = {
            'aipl': '政治',
            'aie': '財經',
            'ait': '科技',
            'asoc': '社會',
            'aopl': '國際',
            'ahel': '生活',
            'acul': '文化',
            'aspt': '運動',
        }

        for code, name in category_map.items():
            if f'/news/{code}/' in url:
                return name

        return '其他'


# For standalone execution
if __name__ == '__main__':
    from scrapy.crawler import CrawlerProcess
    from scrapy.utils.project import get_project_settings

    # Load settings
    settings = get_project_settings()

    # Override with custom settings
    settings.update({
        'DOWNLOAD_DELAY': 2,
        'CONCURRENT_REQUESTS_PER_DOMAIN': 2,
        'ROBOTSTXT_OBEY': True,
        'USER_AGENT': 'CNIRS Academic Research Bot (Educational Use)',
        'LOG_LEVEL': 'INFO',
    })

    # Create crawler process
    process = CrawlerProcess(settings)

    # Start crawling
    # Example: crawl January 2024
    process.crawl(
        CNANewsSpider,
        start_date='2024-01-01',
        end_date='2024-01-31'
    )

    # Run
    process.start()
