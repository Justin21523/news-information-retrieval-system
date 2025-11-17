"""
CNIRS News Crawlers Package

This package contains Scrapy spiders for crawling news from various sources.

Available Spiders:
    - CNANewsSpider: Central News Agency (CNA)
    - PTSNewsSpider: Public Television Service (PTS)
    - TechNewsSpider: TechNews (科技新報)

Author: CNIRS Development Team
License: Educational Use Only
"""

from .cna_spider import CNANewsSpider
from .pts_spider import PTSNewsSpider
from .technews_spider import TechNewsSpider

__all__ = [
    'CNANewsSpider',
    'PTSNewsSpider',
    'TechNewsSpider',
]
