"""
TechNews Spider

Spider for crawling tech news articles from TechNews (科技新報).

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


class TechNewsSpider(scrapy.Spider):
    """
    Scrapy spider for crawling TechNews (科技新報) articles.

    Target: https://technews.tw/
    Date Range: 2022-01-01 to 2024-12-31
    Focus: Technology news (AI, 5G, semiconductors, etc.)

    Usage:
        scrapy runspider technews_spider.py -o output.jsonl

        # With date range
        scrapy runspider technews_spider.py \
            -a start_date=2024-01-01 \
            -a end_date=2024-12-31 \
            -o technews_2024.jsonl
    """

    name = 'technews'
    allowed_domains = ['technews.tw']

    # Custom settings
    custom_settings = {
        'DOWNLOAD_DELAY': 2,
        'CONCURRENT_REQUESTS_PER_DOMAIN': 2,
        'ROBOTSTXT_OBEY': True,
        'USER_AGENT': 'CNIRS Academic Research Bot (Educational Use)',
        'FEEDS': {
            'data/raw/technews_%(time)s.jsonl': {
                'format': 'jsonlines',
                'encoding': 'utf8',
                'store_empty': False,
                'overwrite': False,
            }
        }
    }

    # TechNews categories
    CATEGORIES = [
        'ai',              # AI 人工智慧
        '5g',              # 5G
        'semiconductor',   # 半導體
        'iot',             # IoT 物聯網
        'cloud',           # 雲端
        'big-data',        # 大數據
        'cybersecurity',   # 資安
        'fintech',         # 金融科技
        'startup',         # 新創
        'tech-policy',     # 科技政策
    ]

    def __init__(self, start_date: str = '2022-01-01', end_date: str = '2024-12-31', *args, **kwargs):
        """
        Initialize spider with date range.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        """
        super(TechNewsSpider, self).__init__(*args, **kwargs)

        try:
            self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
            self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError as e:
            logger.error(f"Invalid date format: {e}")
            self.start_date = datetime(2022, 1, 1)
            self.end_date = datetime(2024, 12, 31)

        logger.info(f"TechNews Spider initialized: {self.start_date.date()} to {self.end_date.date()}")

        # Statistics
        self.articles_count = 0
        self.failed_count = 0

    def start_requests(self):
        """
        Generate start requests using date archive URLs.

        Yields:
            scrapy.Request: Requests for monthly archive pages
        """
        # TechNews date archive URL format:
        # https://technews.tw/YYYY/MM/
        # or category: https://technews.tw/category/{category}/

        current_date = self.start_date

        while current_date <= self.end_date:
            # Monthly archive URL
            year = current_date.year
            month = current_date.month

            url = f"https://technews.tw/{year}/{month:02d}/"

            yield scrapy.Request(
                url=url,
                callback=self.parse_monthly_archive,
                errback=self.handle_error,
                meta={'year': year, 'month': month, 'page': 1},
                dont_filter=True
            )

            # Move to next month
            if month == 12:
                current_date = datetime(year + 1, 1, 1)
            else:
                current_date = datetime(year, month + 1, 1)

    def parse_monthly_archive(self, response):
        """
        Parse monthly archive page to extract article URLs.

        Args:
            response: Scrapy response object

        Yields:
            scrapy.Request: Requests for individual article pages
        """
        year = response.meta.get('year')
        month = response.meta.get('month')
        page = response.meta.get('page', 1)

        # Extract article links
        # TechNews structure: <article><h3><a href="...">
        article_links = response.css('article h3 a::attr(href)').getall()

        if not article_links:
            # Try alternative selector
            article_links = response.css('div.post-item h2 a::attr(href)').getall()

        if not article_links:
            logger.warning(f"No articles found for {year}-{month:02d} page {page}")
            return

        logger.info(f"Found {len(article_links)} articles for {year}-{month:02d} page {page}")

        for link in article_links:
            article_url = response.urljoin(link)

            yield scrapy.Request(
                url=article_url,
                callback=self.parse_article,
                errback=self.handle_error
            )

        # Check pagination
        # TechNews uses /page/N/ for pagination
        next_page_link = response.css('a.next.page-numbers::attr(href)').get()
        if next_page_link and page < 10:  # Limit to 10 pages per month
            yield scrapy.Request(
                url=response.urljoin(next_page_link),
                callback=self.parse_monthly_archive,
                errback=self.handle_error,
                meta={'year': year, 'month': month, 'page': page + 1},
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
            # Extract article metadata
            article = {
                'url': response.url,
                'source': 'TechNews',
                'crawled_at': datetime.now().isoformat(),
            }

            # Title
            title = response.css('h1.entry-title::text').get()
            if not title:
                title = response.css('h1::text').get()
            article['title'] = self.clean_text(title)

            # Content
            content_paragraphs = response.css('div.entry-content p::text').getall()
            if not content_paragraphs:
                content_paragraphs = response.css('article.post-content p::text').getall()
            article['content'] = ' '.join([self.clean_text(p) for p in content_paragraphs])

            # Author
            author = response.css('span.author a::text').get()
            if not author:
                author = response.css('div.author-name::text').get()
            article['author'] = self.clean_text(author)

            # Publish date
            publish_date = response.css('time::attr(datetime)').get()
            if not publish_date:
                publish_date = response.css('span.date::text').get()
            article['publish_date'] = self.parse_date(publish_date)

            # Category
            categories = response.css('div.entry-meta span.cat-links a::text').getall()
            if categories:
                article['category'] = self.translate_category(categories[0].strip())
            else:
                article['category'] = '科技'

            # Tags
            tags = response.css('div.entry-meta span.tag-links a::text').getall()
            article['tags'] = [tag.strip() for tag in tags] if tags else []

            # Image URL
            image_url = response.css('div.entry-content img::attr(src)').get()
            if not image_url:
                image_url = response.css('figure.featured-image img::attr(src)').get()
            article['image_url'] = response.urljoin(image_url) if image_url else None

            # Date filtering
            if article['publish_date']:
                article_date = datetime.strptime(article['publish_date'], '%Y-%m-%d')
                if article_date < self.start_date or article_date > self.end_date:
                    logger.debug(f"Article outside date range: {article['publish_date']}")
                    return

            # Validation
            if not article['title'] or not article['content']:
                logger.warning(f"Missing title or content for {response.url}")
                self.failed_count += 1
                return

            if len(article['content']) < 100:  # TechNews articles are usually longer
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
        """Handle request errors."""
        logger.error(f"Request failed: {failure.request.url}")
        logger.error(f"Error: {repr(failure.value)}")
        self.failed_count += 1

    def closed(self, reason):
        """Called when spider is closed."""
        logger.info("=" * 70)
        logger.info("TechNews Spider Finished")
        logger.info("=" * 70)
        logger.info(f"Reason: {reason}")
        logger.info(f"Articles crawled: {self.articles_count}")
        logger.info(f"Failed requests: {self.failed_count}")
        if self.articles_count + self.failed_count > 0:
            logger.info(f"Success rate: {100 * self.articles_count / (self.articles_count + self.failed_count):.1f}%")
        logger.info("=" * 70)

    # Utility methods

    @staticmethod
    def clean_text(text: Optional[str]) -> str:
        """Clean text by removing extra whitespace."""
        if not text:
            return ""
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
                '%Y/%m/%d',
                '%Y-%m-%d',
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
        """Translate category to Chinese."""
        category_map = {
            'ai': 'AI 人工智慧',
            '5g': '5G 通訊',
            'semiconductor': '半導體',
            'iot': '物聯網',
            'cloud': '雲端運算',
            'big-data': '大數據',
            'cybersecurity': '資訊安全',
            'fintech': '金融科技',
            'startup': '新創',
            'tech-policy': '科技政策',
        }
        return category_map.get(category.lower(), '科技')


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
        TechNewsSpider,
        start_date='2024-01-01',
        end_date='2024-01-31'
    )
    process.start()
