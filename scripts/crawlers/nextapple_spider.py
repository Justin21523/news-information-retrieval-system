"""
NextApple News Spider (壹蘋新聞網)

Comprehensive multi-sitemap crawler for NextApple (former Apple Daily Taiwan).
Deep refactoring with complete metadata extraction and robust error handling.

Target: https://news.nextapple.com/
Sitemaps:
    - News: https://apis.nextapple.tw/api/xml/site-map/news (~9,500 articles)
    - Editors: https://apis.nextapple.tw/api/xml/site-map/editors
    - Topics: https://apis.nextapple.tw/api/xml/site-map/topics

Discovery (2025-11-18 - Deep Analysis):
    - Multi-sitemap strategy provides comprehensive coverage
    - Hash-based article IDs (32-char hex) - sequential crawling impossible
    - No Playwright needed - direct access with standard Scrapy
    - 9 categories with Chinese mapping
    - Date-based URL structure enables filtering
    - Rich metadata available via og: tags and JSON-LD

URL Structure:
    Article: https://news.nextapple.com/{category}/{YYYYMMDD}/{32-char-hash}
    Example: https://news.nextapple.com/international/20251118/ABC123...DEF

Categories:
    international (國際), life (生活), sports (體育), aitech (AI科技),
    gadget (3C), property (房地產), finance (財經), entertainment (娛樂),
    local (地方)

Author: Information Retrieval System
Date: 2025-11-18
Version: 2.0 (Deep Refactoring)
"""

import scrapy
from datetime import datetime, timedelta
import logging
import re
import json
import xml.etree.ElementTree as ET
from typing import Optional, Dict, List
from scrapy.spidermiddlewares.httperror import HttpError
from twisted.internet.error import DNSLookupError, TimeoutError, TCPTimedOutError

logger = logging.getLogger(__name__)


class NextAppleSpider(scrapy.Spider):
    """
    Comprehensive multi-sitemap spider for NextApple (壹蘋新聞網).

    Features (Deep Refactoring):
        - Multi-sitemap support (news, editors, topics)
        - Comprehensive metadata extraction with 5+ fallback strategies per field
        - Date-based filtering for targeted collection
        - Robust error handling with detailed statistics
        - JSON-LD and meta property extraction
        - Category enrichment with Chinese names
        - Image and tag extraction
        - Detailed logging and progress tracking

    Usage:
        # All sitemaps (comprehensive crawl)
        scrapy runspider nextapple_spider.py -o nextapple_all.jsonl

        # News sitemap only (default)
        scrapy runspider nextapple_spider.py -a sitemap=news -o nextapple_news.jsonl

        # Editors' picks
        scrapy runspider nextapple_spider.py -a sitemap=editors -o nextapple_editors.jsonl

        # All sitemaps with date filter (last 7 days)
        scrapy runspider nextapple_spider.py -a sitemap=all -a days=7 -o nextapple_recent.jsonl

        # Specific date range
        scrapy runspider nextapple_spider.py -a start_date=2024-11-01 -a end_date=2024-11-18

    Parameters:
        sitemap (str): Which sitemap(s) to crawl - 'news', 'editors', 'topics', 'all'
        days (int): Filter articles from last N days
        start_date (str): Start date (YYYY-MM-DD format)
        end_date (str): End date (YYYY-MM-DD format)
        max_articles (int): Maximum articles to scrape (for testing)
    """

    name = 'nextapple'
    allowed_domains = ['news.nextapple.com', 'apis.nextapple.tw']

    custom_settings = {
        'DOWNLOAD_DELAY': 1.5,
        'CONCURRENT_REQUESTS_PER_DOMAIN': 3,
        'CONCURRENT_REQUESTS': 8,
        'ROBOTSTXT_OBEY': True,
        'USER_AGENT': 'CNIRS Academic Research Bot (Educational Use)',
        'RETRY_TIMES': 3,
        'RETRY_HTTP_CODES': [500, 502, 503, 504, 408, 429],
        'HTTPERROR_ALLOW_404': True,
        'FEEDS': {
            'data/raw/nextapple_news_%(time)s.jsonl': {
                'format': 'jsonlines',
                'encoding': 'utf8',
                'store_empty': False,
                'overwrite': False,
                'indent': 2,
            }
        },
    }

    # Multi-sitemap support
    SITEMAPS = {
        'news': 'https://apis.nextapple.tw/api/xml/site-map/news',
        'editors': 'https://apis.nextapple.tw/api/xml/site-map/editors',
        'topics': 'https://apis.nextapple.tw/api/xml/site-map/topics',
    }

    def __init__(self, sitemap: str = 'news', days: int = None,
                 start_date: str = None, end_date: str = None,
                 max_articles: int = None, *args, **kwargs):
        super(NextAppleSpider, self).__init__(*args, **kwargs)

        # Sitemap selection
        self.sitemap_mode = sitemap.lower()
        if self.sitemap_mode not in ['news', 'editors', 'topics', 'all']:
            logger.warning(f"Invalid sitemap '{sitemap}', using 'news'")
            self.sitemap_mode = 'news'

        # Date filtering
        self.days = int(days) if days else None
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d') if start_date else None
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d') if end_date else None

        # Apply days filter if specified
        if self.days and not self.start_date:
            self.start_date = datetime.now() - timedelta(days=self.days)
            self.end_date = datetime.now()

        # Article limit for testing
        self.max_articles = int(max_articles) if max_articles else None

        # Statistics tracking
        self.articles_count = 0
        self.failed_count = 0
        self.skipped_date_count = 0
        self.sitemap_stats = {}  # Track per-sitemap statistics

        # Initialization log
        logger.info("=" * 70)
        logger.info("NextApple Spider Initialized (Deep Refactoring v2.0)")
        logger.info("=" * 70)
        logger.info(f"Sitemap mode: {self.sitemap_mode}")
        if self.start_date and self.end_date:
            logger.info(f"Date range: {self.start_date.date()} to {self.end_date.date()}")
        elif self.days:
            logger.info(f"Date filter: Last {self.days} days")
        if self.max_articles:
            logger.info(f"Article limit: {self.max_articles}")
        logger.info("=" * 70)

    def start_requests(self):
        """
        Fetch sitemap(s) and extract article URLs.

        Supports single or multi-sitemap crawling based on sitemap_mode parameter.
        """
        if self.sitemap_mode == 'all':
            # Crawl all sitemaps
            for sitemap_name, sitemap_url in self.SITEMAPS.items():
                logger.info(f"Queuing sitemap: {sitemap_name} ({sitemap_url})")
                yield scrapy.Request(
                    url=sitemap_url,
                    callback=self.parse_sitemap,
                    errback=self.handle_error,
                    meta={'sitemap_name': sitemap_name},
                    dont_filter=True
                )
        else:
            # Crawl specific sitemap
            sitemap_url = self.SITEMAPS[self.sitemap_mode]
            logger.info(f"Queuing sitemap: {self.sitemap_mode} ({sitemap_url})")
            yield scrapy.Request(
                url=sitemap_url,
                callback=self.parse_sitemap,
                errback=self.handle_error,
                meta={'sitemap_name': self.sitemap_mode},
                dont_filter=True
            )

    def parse_sitemap(self, response):
        """
        Parse XML sitemap and extract article URLs with comprehensive filtering.

        Handles:
        - XML namespace parsing
        - Date-based filtering (days or date range)
        - Article limit enforcement
        - Per-sitemap statistics tracking
        """
        sitemap_name = response.meta.get('sitemap_name', 'unknown')
        logger.info(f"Parsing sitemap: {sitemap_name} ({response.url})")

        try:
            # Parse XML with namespace support
            root = ET.fromstring(response.text)
            ns = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}

            # Extract all URLs with lastmod dates
            url_data = []
            for url_elem in root.findall('ns:url', ns):
                loc = url_elem.find('ns:loc', ns)
                lastmod = url_elem.find('ns:lastmod', ns)

                if loc is not None and loc.text:
                    url_data.append({
                        'url': loc.text,
                        'lastmod': lastmod.text if lastmod is not None else None
                    })

            logger.info(f"Found {len(url_data)} total URLs in {sitemap_name} sitemap")

            # Initialize sitemap statistics
            if sitemap_name not in self.sitemap_stats:
                self.sitemap_stats[sitemap_name] = {
                    'total_urls': len(url_data),
                    'filtered_urls': 0,
                    'scraped': 0,
                    'failed': 0
                }

            # Apply date filtering
            filtered_urls = self._apply_date_filter(url_data)
            self.sitemap_stats[sitemap_name]['filtered_urls'] = len(filtered_urls)

            logger.info(f"After filtering: {len(filtered_urls)} URLs from {sitemap_name}")

            # Apply article limit if specified
            if self.max_articles and len(filtered_urls) > self.max_articles:
                filtered_urls = filtered_urls[:self.max_articles]
                logger.info(f"Limited to {self.max_articles} articles (testing mode)")

            # Generate requests for each article
            for url_info in filtered_urls:
                # Stop if max_articles limit reached globally
                if self.max_articles and self.articles_count >= self.max_articles:
                    logger.info(f"Reached max_articles limit: {self.max_articles}")
                    break

                yield scrapy.Request(
                    url=url_info['url'],
                    callback=self.parse_article,
                    errback=self.handle_error,
                    meta={
                        'sitemap_name': sitemap_name,
                        'lastmod': url_info.get('lastmod')
                    }
                )

        except ET.ParseError as e:
            logger.error(f"XML parsing error in {sitemap_name}: {e}")
        except Exception as e:
            logger.error(f"Error parsing sitemap {sitemap_name}: {e}")

    def _apply_date_filter(self, url_data: List[Dict]) -> List[Dict]:
        """
        Apply date filtering to URL data.

        Supports:
        - Date range filtering (start_date to end_date)
        - URL-based date extraction (/YYYYMMDD/ pattern)
        - lastmod-based filtering
        """
        if not self.start_date and not self.end_date:
            return url_data  # No filtering

        filtered = []
        for url_info in url_data:
            url = url_info['url']

            # Strategy 1: Extract date from URL pattern (/20251118/)
            match = re.search(r'/(\d{8})/', url)
            if match:
                date_str = match.group(1)
                try:
                    article_date = datetime.strptime(date_str, '%Y%m%d')

                    # Check if within date range
                    if self.start_date and article_date < self.start_date:
                        self.skipped_date_count += 1
                        continue
                    if self.end_date and article_date > self.end_date:
                        self.skipped_date_count += 1
                        continue

                    filtered.append(url_info)
                    continue

                except ValueError:
                    pass  # Invalid date, try next strategy

            # Strategy 2: Use lastmod from sitemap
            if url_info.get('lastmod'):
                try:
                    lastmod_date = datetime.fromisoformat(url_info['lastmod'].replace('Z', '+00:00'))
                    lastmod_date = lastmod_date.replace(tzinfo=None)  # Remove timezone

                    if self.start_date and lastmod_date < self.start_date:
                        self.skipped_date_count += 1
                        continue
                    if self.end_date and lastmod_date > self.end_date:
                        self.skipped_date_count += 1
                        continue

                    filtered.append(url_info)
                    continue

                except (ValueError, AttributeError):
                    pass

            # If no date found or parsing failed, include by default
            filtered.append(url_info)

        return filtered

    def parse_article(self, response):
        """
        Parse individual article page with comprehensive metadata extraction.

        Extraction Strategies (Deep Refactoring):
        - Title: 6 fallback strategies (og:title, h1, JSON-LD, etc.)
        - Content: 5 fallback strategies (article p, JSON-LD, div content, etc.)
        - Date: 6 fallback strategies (meta, time, JSON-LD, URL, etc.)
        - Author: 4 fallback strategies
        - Category: URL extraction with Chinese mapping
        - Tags: Multiple sources (meta keywords, article tags, etc.)
        - Images: og:image, article images, featured image
        """
        sitemap_name = response.meta.get('sitemap_name', 'unknown')

        try:
            # Initialize article structure
            article = {
                'article_id': self._extract_id_from_url(response.url),
                'url': response.url,
                'source': 'NextApple',
                'source_name': '壹蘋新聞網',
                'sitemap': sitemap_name,
                'crawled_at': datetime.now().isoformat(),
            }

            # === TITLE EXTRACTION (6 strategies) ===
            title = (
                # Strategy 1: og:title (most reliable)
                response.css('meta[property="og:title"]::attr(content)').get() or
                # Strategy 2: Standard h1
                response.css('h1::text').get() or
                # Strategy 3: Article title class
                response.css('h1.article-title::text').get() or
                response.css('div.article-header h1::text').get() or
                # Strategy 4: Twitter card title
                response.css('meta[name="twitter:title"]::attr(content)').get() or
                # Strategy 5: Page title (fallback)
                response.css('title::text').get()
            )
            article['title'] = self.clean_text(title)

            # Try JSON-LD for title if still not found
            if not article['title']:
                article['title'] = self._extract_from_jsonld(response, 'headline')

            # === CONTENT EXTRACTION (5 strategies) ===
            content_paragraphs = []

            # Strategy 1: JSON-LD articleBody
            jsonld_content = self._extract_from_jsonld(response, 'articleBody')
            if jsonld_content:
                content_paragraphs = [jsonld_content]
            else:
                # Strategy 2: Standard article tags
                content_paragraphs = (
                    response.css('article p::text').getall() or
                    # Strategy 3: Article content divs
                    response.css('div.article-content p::text').getall() or
                    response.css('div.article-body p::text').getall() or
                    # Strategy 4: Generic content divs
                    response.css('div[class*="content"] p::text').getall() or
                    # Strategy 5: Any paragraph in main
                    response.css('main p::text').getall()
                )

            # Clean and join content
            article['content'] = ' '.join([self.clean_text(p) for p in content_paragraphs if p and len(p.strip()) > 10])

            # === DATE EXTRACTION (6 strategies) ===
            date_text = (
                # Strategy 1: article:published_time (most reliable)
                response.css('meta[property="article:published_time"]::attr(content)').get() or
                # Strategy 2: time datetime attribute
                response.css('time::attr(datetime)').get() or
                # Strategy 3: og:updated_time
                response.css('meta[property="og:updated_time"]::attr(content)').get() or
                # Strategy 4: datePublished meta
                response.css('meta[itemprop="datePublished"]::attr(content)').get() or
                # Strategy 5: lastmod from sitemap
                response.meta.get('lastmod')
            )
            article['publish_date'] = self.parse_date(date_text)

            # Strategy 6: Extract from URL as final fallback
            if not article['publish_date']:
                match = re.search(r'/(\d{4})(\d{2})(\d{2})/', response.url)
                if match:
                    article['publish_date'] = f"{match.group(1)}-{match.group(2)}-{match.group(3)}"

            # Try JSON-LD for date if still not found
            if not article['publish_date']:
                jsonld_date = self._extract_from_jsonld(response, 'datePublished')
                article['publish_date'] = self.parse_date(jsonld_date)

            # === CATEGORY EXTRACTION ===
            article['category'] = self._extract_category(response.url)
            article['category_code'] = self._extract_category_code(response.url)

            # === AUTHOR EXTRACTION (4 strategies) ===
            author = (
                # Strategy 1: author meta tag
                response.css('meta[name="author"]::attr(content)').get() or
                # Strategy 2: Article byline
                response.css('span.author::text').get() or
                response.css('div.author-name::text').get() or
                # Strategy 3: itemprop author
                response.css('span[itemprop="author"]::text').get()
            )
            article['author'] = self.clean_text(author) if author else '壹蘋新聞網'

            # Try JSON-LD for author if not found
            if article['author'] == '壹蘋新聞網':
                jsonld_author = self._extract_from_jsonld(response, 'author')
                if jsonld_author:
                    if isinstance(jsonld_author, dict):
                        article['author'] = jsonld_author.get('name', '壹蘋新聞網')
                    else:
                        article['author'] = str(jsonld_author)

            # === TAGS EXTRACTION (comprehensive) ===
            tags = []

            # Source 1: meta keywords
            keywords = response.css('meta[name="keywords"]::attr(content)').get()
            if keywords:
                tags.extend([k.strip() for k in keywords.split(',') if k.strip()])

            # Source 2: article:tag meta tags
            article_tags = response.css('meta[property="article:tag"]::attr(content)').getall()
            tags.extend(article_tags)

            # Source 3: Tag links/spans in article
            tag_elements = response.css('a.tag::text, span.tag::text').getall()
            tags.extend([self.clean_text(t) for t in tag_elements])

            # Deduplicate and filter
            article['tags'] = list(set([t for t in tags if t and len(t) > 1]))

            # === IMAGE EXTRACTION (comprehensive) ===
            images = []

            # Primary image: og:image
            og_image = response.css('meta[property="og:image"]::attr(content)').get()
            if og_image:
                images.append(og_image)

            # Additional images
            article_images = response.css('article img::attr(src), div.article-content img::attr(src)').getall()
            images.extend([img for img in article_images if img and not img.endswith('.gif')])

            article['image_url'] = images[0] if images else None
            article['images'] = images[:5]  # Store up to 5 images

            # === DESCRIPTION ===
            description = (
                response.css('meta[property="og:description"]::attr(content)').get() or
                response.css('meta[name="description"]::attr(content)').get()
            )
            article['description'] = self.clean_text(description) if description else None

            # === VALIDATION ===
            is_valid = True
            validation_issues = []

            if not article['title']:
                validation_issues.append('no_title')
                is_valid = False

            if not article.get('content') or len(article['content']) < 50:
                validation_issues.append('insufficient_content')
                is_valid = False

            if not article.get('publish_date'):
                validation_issues.append('no_date')
                # Don't fail, but log warning
                logger.warning(f"No date found for: {response.url}")

            if not is_valid:
                logger.warning(f"Invalid article ({', '.join(validation_issues)}): {response.url}")
                self.failed_count += 1
                if sitemap_name in self.sitemap_stats:
                    self.sitemap_stats[sitemap_name]['failed'] += 1
                return

            # === SUCCESS ===
            self.articles_count += 1
            if sitemap_name in self.sitemap_stats:
                self.sitemap_stats[sitemap_name]['scraped'] += 1

            # Progress logging
            if self.articles_count % 50 == 0:
                logger.info(f"Progress: {self.articles_count} articles scraped")

            yield article

        except Exception as e:
            logger.error(f"Error parsing {response.url}: {e}")
            self.failed_count += 1
            if sitemap_name in self.sitemap_stats:
                self.sitemap_stats[sitemap_name]['failed'] += 1

    def _extract_from_jsonld(self, response, field: str) -> Optional[str]:
        """
        Extract field from JSON-LD structured data.

        Supports NewsArticle, Article, and other schema.org types.
        """
        try:
            jsonld_scripts = response.css('script[type="application/ld+json"]::text').getall()
            for script in jsonld_scripts:
                try:
                    data = json.loads(script)

                    # Handle array of items
                    if isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict) and field in item:
                                return item[field]
                    # Handle single item
                    elif isinstance(data, dict) and field in data:
                        return data[field]

                except json.JSONDecodeError:
                    continue

        except Exception as e:
            logger.debug(f"JSON-LD extraction failed for {field}: {e}")

        return None

    def handle_error(self, failure):
        """
        Handle request errors with detailed logging.

        Tracks HTTP errors, network errors, and timeouts separately.
        """
        sitemap_name = failure.request.meta.get('sitemap_name', 'unknown')

        if failure.check(HttpError):
            response = failure.value.response
            logger.warning(f"HTTP {response.status}: {failure.request.url}")
        elif failure.check(DNSLookupError):
            logger.error(f"DNS lookup failed: {failure.request.url}")
        elif failure.check((TimeoutError, TCPTimedOutError)):
            logger.error(f"Timeout: {failure.request.url}")
        else:
            logger.error(f"Request failed ({failure.type}): {failure.request.url}")

        self.failed_count += 1
        if sitemap_name in self.sitemap_stats:
            self.sitemap_stats[sitemap_name]['failed'] += 1

    def closed(self, reason):
        """
        Log comprehensive statistics on spider closure.

        Includes:
        - Overall statistics
        - Per-sitemap breakdown
        - Success rates
        - Data quality metrics
        """
        logger.info("=" * 70)
        logger.info("NextApple Spider Finished (Deep Refactoring v2.0)")
        logger.info("=" * 70)
        logger.info(f"Reason: {reason}")
        logger.info("")

        # Overall statistics
        logger.info("OVERALL STATISTICS:")
        logger.info(f"  Articles successfully scraped: {self.articles_count}")
        logger.info(f"  Failed: {self.failed_count}")
        logger.info(f"  Skipped (date filter): {self.skipped_date_count}")

        total_processed = self.articles_count + self.failed_count
        if total_processed > 0:
            success_rate = 100 * self.articles_count / total_processed
            logger.info(f"  Success rate: {success_rate:.2f}%")

        logger.info("")

        # Per-sitemap statistics
        if self.sitemap_stats:
            logger.info("PER-SITEMAP BREAKDOWN:")
            for sitemap_name, stats in self.sitemap_stats.items():
                logger.info(f"  {sitemap_name}:")
                logger.info(f"    Total URLs in sitemap: {stats['total_urls']}")
                logger.info(f"    After filtering: {stats['filtered_urls']}")
                logger.info(f"    Successfully scraped: {stats['scraped']}")
                logger.info(f"    Failed: {stats['failed']}")

                if stats['filtered_urls'] > 0:
                    sitemap_success = 100 * stats['scraped'] / stats['filtered_urls']
                    logger.info(f"    Success rate: {sitemap_success:.2f}%")

        logger.info("=" * 70)

    # Utility methods

    @staticmethod
    def clean_text(text: Optional[str]) -> str:
        if not text:
            return ""
        return re.sub(r'\s+', ' ', text).strip()

    @staticmethod
    def parse_date(date_str: Optional[str]) -> Optional[str]:
        if not date_str:
            return None
        try:
            # ISO 8601
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00').split('+')[0])
            return dt.strftime('%Y-%m-%d')
        except:
            # Extract YYYY-MM-DD
            match = re.search(r'(\d{4})-(\d{2})-(\d{2})', date_str)
            return f"{match.group(1)}-{match.group(2)}-{match.group(3)}" if match else None

    @staticmethod
    def _extract_category(url: str) -> str:
        """
        Extract category Chinese name from URL.

        URL format: https://news.nextapple.com/{category}/{YYYYMMDD}/{hash}
        """
        categories = {
            'international': '國際',
            'life': '生活',
            'sports': '體育',
            'aitech': 'AI科技',
            'gadget': '3C',
            'property': '房地產',
            'finance': '財經',
            'entertainment': '娛樂',
            'local': '地方',
        }
        for code, name in categories.items():
            if f'/{code}/' in url:
                return name
        return '其他'

    @staticmethod
    def _extract_category_code(url: str) -> str:
        """
        Extract category code from URL.

        Returns English category code for programmatic use.
        """
        categories = [
            'international', 'life', 'sports', 'aitech', 'gadget',
            'property', 'finance', 'entertainment', 'local'
        ]
        for code in categories:
            if f'/{code}/' in url:
                return code
        return 'other'

    @staticmethod
    def _extract_id_from_url(url: str) -> str:
        """
        Extract article ID (hash) from URL.

        NextApple uses 32-character hexadecimal hashes for article IDs.
        Format: NEXTAPPLE_{32-char-hex}
        """
        # Try to match 32-char hex hash
        match = re.search(r'/([A-F0-9]{32})$', url, re.IGNORECASE)
        if match:
            return f"NEXTAPPLE_{match.group(1).upper()}"

        # Fallback: use last URL segment
        last_segment = url.rstrip('/').split('/')[-1]
        return f"NEXTAPPLE_{last_segment.upper()}"


if __name__ == '__main__':
    from scrapy.crawler import CrawlerProcess

    process = CrawlerProcess({
        'LOG_LEVEL': 'INFO',
        'ROBOTSTXT_OBEY': True,
    })

    # Example usage - customize parameters as needed:
    # sitemap='news'      - News sitemap only (default)
    # sitemap='editors'   - Editors' picks
    # sitemap='topics'    - Topic collections
    # sitemap='all'       - All sitemaps
    # days=7              - Last 7 days
    # max_articles=50     - Limit for testing

    process.crawl(NextAppleSpider, sitemap='news', days=7, max_articles=50)
    process.start()
