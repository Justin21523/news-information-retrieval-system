#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CTI News Spider (中天新聞網 / ctinews.com)

This spider crawls CTI News articles from ctinews.com using the public sitemaps
advertised in robots.txt:
  - https://ctinews.com/rss/sitemap.xml
  - https://ctinews.com/rss/sitemap-news.xml

The previous implementation incorrectly targeted chinatimes.com and relied on
Playwright/Cloudflare workarounds, which caused instability and timeouts.

Features:
  - Sitemap-based discovery (fast + comprehensive)
  - Date range filtering via sitemap lastmod/publication_date and URL patterns
  - JSON-LD (NewsArticle) extraction for robust metadata/content parsing

Usage:
  scrapy runspider scripts/crawlers/cti_spider.py -a days=1 -O out.jsonl
  scrapy runspider scripts/crawlers/cti_spider.py -a sitemap=news -a start_date=2025-01-01 -a end_date=2025-12-31

Complexity:
  - Sitemap parsing: O(N) URLs
  - Article parsing: O(1) per article (bounded selectors/JSON-LD blocks)
"""

from __future__ import annotations

import json
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Iterable

import scrapy

from scripts.crawlers.utils.jobdir import has_pending_requests

logger = logging.getLogger(__name__)


def _safe_int(value, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_date(date_text: Optional[str]) -> Optional[str]:
    if not date_text:
        return None

    # Common ISO-ish formats
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M%z"):
        try:
            dt = datetime.strptime(date_text.strip(), fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue

    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", date_text)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"

    return None


def _clean_text(text: Optional[str]) -> str:
    if not text:
        return ""
    return " ".join(str(text).split()).strip()


@dataclass(frozen=True)
class SitemapEntry:
    url: str
    lastmod: Optional[str] = None
    news_publication_date: Optional[str] = None


class CTINewsSpider(scrapy.Spider):
    """Sitemap-based CTI News spider."""

    name = "cti_news"
    allowed_domains = ["ctinews.com", "www.ctinews.com", "storage.ctinews.com"]

    SITEMAPS = {
        "all": "https://ctinews.com/rss/sitemap.xml",
        "news": "https://ctinews.com/rss/sitemap-news.xml",
    }

    custom_settings = {
        "DOWNLOAD_DELAY": 0.8,
        "CONCURRENT_REQUESTS": 8,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 4,
        "ROBOTSTXT_OBEY": True,
        "RETRY_TIMES": 3,
        "RETRY_HTTP_CODES": [500, 502, 503, 504, 408, 429],
        "LOG_LEVEL": "INFO",
        "FEEDS": {
            "/mnt/c/data/information-retrieval/raw/cti_news_%(time)s.jsonl": {
                "format": "jsonlines",
                "encoding": "utf8",
                "store_empty": False,
                "overwrite": False,
                "indent": None,
            }
        },
    }

    def __init__(
        self,
        sitemap: str = "news",
        days: int = 7,
        start_date: str = None,
        end_date: str = None,
        max_articles: int = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        sitemap = str(sitemap).strip().lower() if sitemap else "news"
        if sitemap not in self.SITEMAPS:
            logger.warning(f"Invalid sitemap '{sitemap}', using 'news'")
            sitemap = "news"
        self.sitemap_mode = sitemap

        self.end_date = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.now()
        if start_date:
            self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        else:
            self.start_date = self.end_date - timedelta(days=_safe_int(days, default=7))

        self.max_articles = _safe_int(max_articles)
        self._articles_scraped = 0

        logger.info("=" * 70)
        logger.info("CTI News Spider Initialized (ctinews.com)")
        logger.info("=" * 70)
        logger.info(f"Sitemap: {self.sitemap_mode} ({self.SITEMAPS[self.sitemap_mode]})")
        logger.info(f"Date range: {self.start_date.date()} to {self.end_date.date()}")
        if self.max_articles:
            logger.info(f"Max articles: {self.max_articles}")
        logger.info("=" * 70)

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super().from_crawler(crawler, *args, **kwargs)
        spider._resume_from_jobdir = has_pending_requests(crawler.settings.get("JOBDIR"))
        if spider._resume_from_jobdir:
            logger.info(f"Detected JOBDIR resume queue; skipping start_requests seeding (JOBDIR={crawler.settings.get('JOBDIR')})")
        return spider

    def start_requests(self):
        if getattr(self, "_resume_from_jobdir", False):
            return

        yield scrapy.Request(
            url=self.SITEMAPS[self.sitemap_mode],
            callback=self.parse_sitemap,
            errback=self.handle_error,
            dont_filter=True,
        )

    def parse_sitemap(self, response):
        entries = list(self._iter_sitemap_entries(response.text))
        logger.info(f"Sitemap entries: {len(entries)}")

        start = self.start_date.date()
        end = self.end_date.date()

        for entry in entries:
            if self.max_articles and self._articles_scraped >= self.max_articles:
                break

            if "/news/items/" not in entry.url:
                continue

            date_str = _parse_date(entry.news_publication_date) or _parse_date(entry.lastmod)
            if date_str:
                try:
                    pub_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                    if not (start <= pub_date <= end):
                        continue
                except ValueError:
                    pass

            yield scrapy.Request(
                url=entry.url,
                callback=self.parse_article,
                errback=self.handle_error,
                dont_filter=True,
                meta={
                    "sitemap_lastmod": entry.lastmod,
                    "sitemap_publication_date": entry.news_publication_date,
                },
            )

    def _iter_sitemap_entries(self, xml_text: str) -> Iterable[SitemapEntry]:
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as e:
            logger.error(f"Failed to parse CTI sitemap XML: {e}")
            return []

        ns = {
            "ns": "http://www.sitemaps.org/schemas/sitemap/0.9",
            "news": "http://www.google.com/schemas/sitemap-news/0.9",
        }

        for url_elem in root.findall("ns:url", ns):
            loc = url_elem.find("ns:loc", ns)
            if loc is None or not loc.text:
                continue

            lastmod = None
            lastmod_elem = url_elem.find("ns:lastmod", ns)
            if lastmod_elem is not None and lastmod_elem.text:
                lastmod = lastmod_elem.text.strip()

            pub_date = None
            pub_date_elem = url_elem.find(".//news:publication_date", ns)
            if pub_date_elem is not None and pub_date_elem.text:
                pub_date = pub_date_elem.text.strip()

            yield SitemapEntry(url=loc.text.strip(), lastmod=lastmod, news_publication_date=pub_date)

    def parse_article(self, response):
        try:
            if response.status != 200:
                return

            jsonld = self._extract_newsarticle_jsonld(response)
            title = _clean_text(
                (jsonld.get("headline") if jsonld else None)
                or response.css('meta[property="og:title"]::attr(content)').get()
                or response.css("h1::text").get()
            )

            content = ""
            if jsonld and isinstance(jsonld.get("articleBody"), str):
                content = _clean_text(jsonld.get("articleBody"))
            if not content:
                paragraphs = response.css("main [itemprop=\"articleBody\"] p::text, article p::text").getall()
                content = " ".join(_clean_text(p) for p in paragraphs if p)

            date_text = (
                response.css('meta[property="article:published_time"]::attr(content)').get()
                or (jsonld.get("datePublished") if jsonld else None)
                or response.meta.get("sitemap_publication_date")
                or response.meta.get("sitemap_lastmod")
            )
            published_date = _parse_date(date_text)

            author = None
            if jsonld:
                author_val = jsonld.get("author")
                if isinstance(author_val, dict):
                    author = author_val.get("name")
                elif isinstance(author_val, str):
                    author = author_val
            author = _clean_text(
                author
                or response.css('meta[name="author"]::attr(content)').get()
                or response.css(".cti-author-name::text").get()
            ) or "中天新聞網"

            category = None
            if jsonld and isinstance(jsonld.get("articleSection"), str):
                category = jsonld.get("articleSection")
            category = _clean_text(category) or "unknown"

            tags = []
            if jsonld and isinstance(jsonld.get("keywords"), list):
                tags = [str(t) for t in jsonld.get("keywords") if str(t).strip()]
            if not tags:
                keywords = response.css('meta[name="keywords"]::attr(content)').get()
                if keywords:
                    tags = [k.strip() for k in keywords.split(",") if k.strip()]

            image_url = response.css('meta[property="og:image"]::attr(content)').get()

            article = {
                "article_id": self._extract_id_from_url(response.url),
                "url": response.url,
                "source": "CTI",
                "source_name": "中天新聞網",
                "crawled_at": datetime.now().isoformat(),
                "title": title,
                "content": content,
                "published_date": published_date,
                "author": author,
                "category": category,
                "category_name": category,
                "tags": tags,
                "image_url": image_url,
            }

            if not article["title"] or not article["content"] or len(article["content"]) < 100:
                return

            self._articles_scraped += 1
            yield article

        except Exception as e:
            logger.error(f"Error parsing article {response.url}: {e}", exc_info=True)

    def _extract_newsarticle_jsonld(self, response) -> dict:
        scripts = response.css('script[type="application/ld+json"]::text').getall()
        for script in scripts:
            try:
                data = json.loads(script)
            except Exception:
                continue

            candidates = data if isinstance(data, list) else [data]
            for item in candidates:
                if isinstance(item, dict) and item.get("@type") == "NewsArticle":
                    return item
        return {}

    @staticmethod
    def _extract_id_from_url(url: str) -> str:
        match = re.search(r"/news/items/([A-Za-z0-9]+)$", url)
        return f"cti_{match.group(1)}" if match else re.sub(r"\\W+", "_", url)[-32:]

    def handle_error(self, failure):
        req = failure.request
        logger.error(f"Request failed: {req.url} ({failure.type.__name__}: {failure.value})")
