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
from typing import Optional, List

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
        'DOWNLOAD_DELAY': 2,
        'CONCURRENT_REQUESTS_PER_DOMAIN': 2,
        'ROBOTSTXT_OBEY': True,
        'USER_AGENT': 'CNIRS Academic Research Bot (Educational Use)',
        'FEEDS': {
            'data/raw/pts_news_%(time)s.jsonl': {
                'format': 'jsonlines',
                'encoding': 'utf8',
                'store_empty': False,
                'overwrite': False,
            }
        }
    }

    # PTS news categories
    CATEGORIES = [
        'politics',     # 政治
        'finance',      # 財經
        'local',        # 生活
        'international',# 國際
        'culture',      # 文化
        'education',    # 教育
        'life',         # 生活
        'welfare',      # 福利
        'environment',  # 環境
    ]

    def __init__(self, start_date: str = '2022-01-01', end_date: str = '2024-12-31', *args, **kwargs):
        """
        Initialize spider with date range.

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        """
        super(PTSNewsSpider, self).__init__(*args, **kwargs)

        try:
            self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
            self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
        except ValueError as e:
            logger.error(f"Invalid date format: {e}")
            self.start_date = datetime(2022, 1, 1)
            self.end_date = datetime(2024, 12, 31)

        logger.info(f"PTS Spider initialized: {self.start_date.date()} to {self.end_date.date()}")

        # Statistics
        self.articles_count = 0
        self.failed_count = 0

    def start_requests(self):
        """
        Generate start requests for archive pages.

        Yields:
            scrapy.Request: Requests for category/date archive pages
        """
        # PTS uses paginated list pages
        # Approach: Start from recent and go backwards
        # URL format: https://news.pts.org.tw/article/{article_id}
        # List page: https://news.pts.org.tw/category/{category}?page={page}

        for category in self.CATEGORIES:
            # Start from page 1
            url = f"https://news.pts.org.tw/category/{category}"

            yield scrapy.Request(
                url=url,
                callback=self.parse_list_page,
                errback=self.handle_error,
                meta={'category': category, 'page': 1},
                dont_filter=True
            )

    def parse_list_page(self, response):
        """
        Parse category list page to extract article URLs.

        Args:
            response: Scrapy response object

        Yields:
            scrapy.Request: Requests for individual article pages
        """
        category = response.meta.get('category', 'unknown')
        page = response.meta.get('page', 1)

        # Extract article links
        # PTS structure: <article class="media-item"><a href="/article/{id}">
        article_links = response.css('article.media-item a::attr(href)').getall()

        if not article_links:
            # Try alternative selector
            article_links = response.css('div.article-list a.item::attr(href)').getall()

        if not article_links:
            logger.warning(f"No articles found for {category} page {page}")
            return

        logger.info(f"Found {len(article_links)} articles for {category} page {page}")

        for link in article_links:
            article_url = response.urljoin(link)

            yield scrapy.Request(
                url=article_url,
                callback=self.parse_article,
                errback=self.handle_error,
                meta={'category': category}
            )

        # Check if there's a next page
        # Limit: Only crawl pages within date range
        # For now, crawl first 50 pages per category (adjustable)
        if page < 50:
            next_page_url = f"https://news.pts.org.tw/category/{category}?page={page + 1}"

            yield scrapy.Request(
                url=next_page_url,
                callback=self.parse_list_page,
                errback=self.handle_error,
                meta={'category': category, 'page': page + 1},
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
                'source': 'PTS',
                'crawled_at': datetime.now().isoformat(),
            }

            # Title
            title = response.css('h1.article-title::text').get()
            if not title:
                title = response.css('h1::text').get()
            article['title'] = self.clean_text(title)

            # Content
            content_paragraphs = response.css('div.article-content p::text').getall()
            if not content_paragraphs:
                content_paragraphs = response.css('article.content p::text').getall()
            article['content'] = ' '.join([self.clean_text(p) for p in content_paragraphs])

            # Author/Editor
            author = response.css('div.article-info span.reporter::text').get()
            if not author:
                author = response.css('span.author::text').get()
            article['author'] = self.clean_text(author)

            # Publish date
            publish_date = response.css('div.article-info time::attr(datetime)').get()
            if not publish_date:
                publish_date = response.css('time::text').get()
            article['publish_date'] = self.parse_date(publish_date)

            # Category
            category_from_meta = response.meta.get('category', '')
            breadcrumb = response.css('nav.breadcrumb a::text').getall()
            if breadcrumb and len(breadcrumb) > 1:
                article['category'] = self.translate_category(breadcrumb[1].strip())
            else:
                article['category'] = self.translate_category(category_from_meta)

            # Tags
            tags = response.css('div.tags a::text').getall()
            article['tags'] = [tag.strip() for tag in tags] if tags else []

            # Image URL
            image_url = response.css('figure.article-image img::attr(src)').get()
            if not image_url:
                image_url = response.css('article img::attr(src)').get()
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
        """Handle request errors."""
        logger.error(f"Request failed: {failure.request.url}")
        logger.error(f"Error: {repr(failure.value)}")
        self.failed_count += 1

    def closed(self, reason):
        """Called when spider is closed."""
        logger.info("=" * 70)
        logger.info("PTS Spider Finished")
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
