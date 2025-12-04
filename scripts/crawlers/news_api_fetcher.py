#!/usr/bin/env python
"""
News API Fetcher - 新聞 API 資料獲取器

Integrates with multiple news APIs to fetch articles:
- NewsAPI: Global news aggregator with 150,000+ sources
- GDELT: Global Database of Events, Language, and Tone

Features:
    - Multi-source API integration
    - Rate limiting and error handling
    - Unified data format
    - Automatic retries
    - Caching support

Usage:
    # Fetch from NewsAPI
    python scripts/crawlers/news_api_fetcher.py --source newsapi --query "台灣" --days 7

    # Fetch from GDELT
    python scripts/crawlers/news_api_fetcher.py --source gdelt --query "Taiwan" --days 1

    # Fetch from both
    python scripts/crawlers/news_api_fetcher.py --source all --query "AI" --days 3

Author: CNIRS Development Team
License: Educational Use Only
"""

import os
import sys
import json
import time
import logging
import argparse
import hashlib
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from urllib.parse import quote

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class NewsAPIFetcher:
    """
    Fetcher for NewsAPI.org

    Requires API key (free tier: 100 requests/day, 1 month history)
    Get your API key at: https://newsapi.org/register

    Attributes:
        api_key: NewsAPI API key
        base_url: API endpoint
        rate_limit: Requests per second
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize NewsAPI fetcher.

        Args:
            api_key: NewsAPI API key (or set NEWSAPI_KEY environment variable)
        """
        self.api_key = api_key or os.getenv('NEWSAPI_KEY')
        if not self.api_key:
            logger.warning("NewsAPI key not found. Set NEWSAPI_KEY environment variable or pass api_key parameter.")
            logger.warning("Get free API key at: https://newsapi.org/register")

        self.base_url = "https://newsapi.org/v2"
        self.rate_limit = 0.5  # 0.5 seconds between requests (2 req/sec)
        self.last_request_time = 0

    def _rate_limit_wait(self):
        """Wait to respect rate limits."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()

    def fetch_everything(
        self,
        query: str,
        from_date: str,
        to_date: str,
        language: str = 'zh',
        sort_by: str = 'publishedAt',
        page_size: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Fetch articles using /everything endpoint.

        Args:
            query: Search query (supports AND, OR, NOT operators)
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            language: Language code (zh, en, etc.)
            sort_by: Sort by publishedAt, relevancy, or popularity
            page_size: Results per page (max 100)

        Returns:
            List of article dictionaries

        Example:
            >>> fetcher = NewsAPIFetcher(api_key='your_key')
            >>> articles = fetcher.fetch_everything('台灣', '2025-11-10', '2025-11-17')
            >>> len(articles)
            45
        """
        if not self.api_key:
            logger.error("Cannot fetch: NewsAPI key not set")
            return []

        endpoint = f"{self.base_url}/everything"

        all_articles = []
        page = 1

        while True:
            self._rate_limit_wait()

            params = {
                'q': query,
                'from': from_date,
                'to': to_date,
                'language': language,
                'sortBy': sort_by,
                'pageSize': page_size,
                'page': page,
                'apiKey': self.api_key
            }

            try:
                logger.info(f"Fetching NewsAPI page {page} for query '{query}'...")
                response = requests.get(endpoint, params=params, timeout=30)
                response.raise_for_status()

                data = response.json()

                if data.get('status') != 'ok':
                    logger.error(f"API error: {data.get('message', 'Unknown error')}")
                    break

                articles = data.get('articles', [])

                if not articles:
                    logger.info("No more articles found")
                    break

                all_articles.extend(articles)
                logger.info(f"  Fetched {len(articles)} articles (total: {len(all_articles)})")

                # Check if we've reached the end
                total_results = data.get('totalResults', 0)
                if len(all_articles) >= total_results or len(articles) < page_size:
                    break

                page += 1

            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed: {e}")
                break
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON: {e}")
                break

        logger.info(f"✓ Fetched total {len(all_articles)} articles from NewsAPI")
        return all_articles

    def fetch_top_headlines(
        self,
        country: str = 'tw',
        category: Optional[str] = None,
        query: Optional[str] = None,
        page_size: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Fetch top headlines using /top-headlines endpoint.

        Args:
            country: Country code (tw, us, etc.)
            category: Category (business, entertainment, health, science, sports, technology)
            query: Search query
            page_size: Results per page (max 100)

        Returns:
            List of article dictionaries
        """
        if not self.api_key:
            logger.error("Cannot fetch: NewsAPI key not set")
            return []

        endpoint = f"{self.base_url}/top-headlines"

        self._rate_limit_wait()

        params = {
            'country': country,
            'pageSize': page_size,
            'apiKey': self.api_key
        }

        if category:
            params['category'] = category
        if query:
            params['q'] = query

        try:
            logger.info(f"Fetching top headlines from {country}...")
            response = requests.get(endpoint, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            if data.get('status') != 'ok':
                logger.error(f"API error: {data.get('message', 'Unknown error')}")
                return []

            articles = data.get('articles', [])
            logger.info(f"✓ Fetched {len(articles)} top headlines")
            return articles

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            return []


class GDELTFetcher:
    """
    Fetcher for GDELT Project API

    GDELT (Global Database of Events, Language, and Tone) is a free,
    open platform that monitors news media in print, broadcast, and web
    formats from around the world.

    No API key required - completely free!

    Attributes:
        base_url: GDELT API endpoint
        rate_limit: Requests per second
    """

    def __init__(self):
        """Initialize GDELT fetcher."""
        self.base_url = "https://api.gdeltproject.org/api/v2"
        self.rate_limit = 1.0  # 1 second between requests
        self.last_request_time = 0

    def _rate_limit_wait(self):
        """Wait to respect rate limits."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()

    def fetch_doc_search(
        self,
        query: str,
        mode: str = 'artlist',
        max_records: int = 250,
        timespan: str = '7d',
        sort_by: str = 'DateDesc'
    ) -> List[Dict[str, Any]]:
        """
        Fetch articles using GDELT DOC 2.0 API.

        Args:
            query: Search query (supports AND, OR, NOT, NEAR operators)
            mode: artlist (articles), timeline, or wordcloud
            max_records: Maximum number of results (max 250)
            timespan: Time span (e.g., '7d', '24h', '1month')
            sort_by: DateDesc or HybridRel

        Returns:
            List of article dictionaries

        Example:
            >>> fetcher = GDELTFetcher()
            >>> articles = fetcher.fetch_doc_search('Taiwan AI', timespan='7d')
            >>> len(articles)
            150
        """
        endpoint = f"{self.base_url}/doc/doc"

        self._rate_limit_wait()

        params = {
            'query': query,
            'mode': mode,
            'maxrecords': max_records,
            'timespan': timespan,
            'sort': sort_by,
            'format': 'json'
        }

        try:
            logger.info(f"Fetching GDELT articles for query '{query}' (timespan: {timespan})...")
            response = requests.get(endpoint, params=params, timeout=60)
            response.raise_for_status()

            # GDELT returns newline-delimited JSON
            articles = []
            for line in response.text.strip().split('\n'):
                if line:
                    try:
                        article = json.loads(line)
                        articles.append(article)
                    except json.JSONDecodeError:
                        continue

            logger.info(f"✓ Fetched {len(articles)} articles from GDELT")
            return articles

        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            return []


class UnifiedNewsFetcher:
    """
    Unified interface for fetching news from multiple APIs.

    Normalizes data from different sources into a consistent format.
    """

    def __init__(self, newsapi_key: Optional[str] = None):
        """
        Initialize unified fetcher.

        Args:
            newsapi_key: NewsAPI API key
        """
        self.newsapi = NewsAPIFetcher(api_key=newsapi_key)
        self.gdelt = GDELTFetcher()

    def fetch_all(
        self,
        query: str,
        days: int = 7,
        sources: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch from all available sources.

        Args:
            query: Search query
            days: Number of days to fetch
            sources: List of sources to fetch from (['newsapi', 'gdelt'] or None for all)

        Returns:
            List of normalized article dictionaries
        """
        if sources is None:
            sources = ['newsapi', 'gdelt']

        all_articles = []

        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        from_date_str = start_date.strftime('%Y-%m-%d')
        to_date_str = end_date.strftime('%Y-%m-%d')

        # Fetch from NewsAPI
        if 'newsapi' in sources:
            logger.info("=" * 60)
            logger.info("Fetching from NewsAPI...")
            logger.info("=" * 60)
            newsapi_articles = self.newsapi.fetch_everything(
                query=query,
                from_date=from_date_str,
                to_date=to_date_str
            )

            # Normalize NewsAPI format
            for article in newsapi_articles:
                normalized = self._normalize_newsapi(article)
                all_articles.append(normalized)

        # Fetch from GDELT
        if 'gdelt' in sources:
            logger.info("=" * 60)
            logger.info("Fetching from GDELT...")
            logger.info("=" * 60)
            gdelt_articles = self.gdelt.fetch_doc_search(
                query=query,
                timespan=f'{days}d'
            )

            # Normalize GDELT format
            for article in gdelt_articles:
                normalized = self._normalize_gdelt(article)
                all_articles.append(normalized)

        # Deduplicate by URL
        all_articles = self._deduplicate(all_articles)

        logger.info("=" * 60)
        logger.info(f"✓ Total fetched: {len(all_articles)} unique articles")
        logger.info("=" * 60)

        return all_articles

    def _normalize_newsapi(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize NewsAPI article format."""
        return {
            'article_id': self._generate_id(article.get('url', '')),
            'title': article.get('title', ''),
            'content': article.get('content', '') or article.get('description', ''),
            'url': article.get('url', ''),
            'published_date': article.get('publishedAt', '').split('T')[0] if article.get('publishedAt') else '',
            'author': article.get('author', '') or article.get('source', {}).get('name', ''),
            'source': 'newsapi',
            'source_name': article.get('source', {}).get('name', ''),
            'category': 'unknown',
            'category_name': '未分類',
            'raw_data': article
        }

    def _normalize_gdelt(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize GDELT article format."""
        return {
            'article_id': self._generate_id(article.get('url', '')),
            'title': article.get('title', ''),
            'content': article.get('title', ''),  # GDELT doesn't provide full content
            'url': article.get('url', ''),
            'published_date': article.get('seendate', '')[:10] if article.get('seendate') else '',
            'author': article.get('domain', ''),
            'source': 'gdelt',
            'source_name': article.get('domain', ''),
            'category': 'unknown',
            'category_name': '未分類',
            'language': article.get('language', ''),
            'tone': article.get('tone', ''),
            'raw_data': article
        }

    def _generate_id(self, url: str) -> str:
        """Generate article ID from URL."""
        if not url:
            return hashlib.md5(str(time.time()).encode()).hexdigest()[:16]
        return hashlib.md5(url.encode()).hexdigest()[:16]

    def _deduplicate(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Deduplicate articles by URL."""
        seen_urls = set()
        unique_articles = []

        for article in articles:
            url = article.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_articles.append(article)

        logger.info(f"Removed {len(articles) - len(unique_articles)} duplicates")
        return unique_articles

    def save_to_jsonl(self, articles: List[Dict[str, Any]], output_file: str):
        """Save articles to JSONL file."""
        output_path = project_root / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for article in articles:
                f.write(json.dumps(article, ensure_ascii=False) + '\n')

        logger.info(f"✓ Saved {len(articles)} articles to {output_path}")


def main():
    """Main function for CLI usage."""
    parser = argparse.ArgumentParser(
        description='Fetch news articles from multiple APIs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch Taiwan news from all sources for last 7 days
  python scripts/crawlers/news_api_fetcher.py --query "台灣" --days 7

  # Fetch only from NewsAPI
  python scripts/crawlers/news_api_fetcher.py --query "AI" --days 3 --source newsapi

  # Fetch with custom output file
  python scripts/crawlers/news_api_fetcher.py --query "COVID" --days 1 --output data/raw/covid_news.jsonl

Environment Variables:
  NEWSAPI_KEY    NewsAPI API key (get free key at https://newsapi.org/register)
        """
    )

    parser.add_argument(
        '--query', '-q',
        type=str,
        required=True,
        help='Search query'
    )

    parser.add_argument(
        '--days', '-d',
        type=int,
        default=7,
        help='Number of days to fetch (default: 7)'
    )

    parser.add_argument(
        '--source', '-s',
        type=str,
        choices=['all', 'newsapi', 'gdelt'],
        default='all',
        help='Source to fetch from (default: all)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='data/raw/api_news.jsonl',
        help='Output JSONL file (default: data/raw/api_news.jsonl)'
    )

    parser.add_argument(
        '--newsapi-key',
        type=str,
        help='NewsAPI API key (or set NEWSAPI_KEY environment variable)'
    )

    args = parser.parse_args()

    # Determine sources
    if args.source == 'all':
        sources = ['newsapi', 'gdelt']
    else:
        sources = [args.source]

    # Initialize fetcher
    fetcher = UnifiedNewsFetcher(newsapi_key=args.newsapi_key)

    # Fetch articles
    articles = fetcher.fetch_all(
        query=args.query,
        days=args.days,
        sources=sources
    )

    # Save to file
    if articles:
        fetcher.save_to_jsonl(articles, args.output)

        # Print summary
        print("\n" + "=" * 60)
        print("FETCH SUMMARY")
        print("=" * 60)
        print(f"Query:         {args.query}")
        print(f"Days:          {args.days}")
        print(f"Sources:       {', '.join(sources)}")
        print(f"Total articles: {len(articles)}")
        print(f"Output file:   {args.output}")
        print("=" * 60)
    else:
        logger.warning("No articles fetched")


if __name__ == '__main__':
    main()
