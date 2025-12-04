# CTI (ChinaTimes) News Spider (Deep Refactoring v2.0)

## Overview

Comprehensive multi-sitemap, multi-mode spider for **CTI 中天新聞** (ChinaTimes.com) with Playwright-based Cloudflare bypass and 2-year historical data access.

**Target:** https://www.chinatimes.com/
**Historical Coverage:** 2+ years (date-based URL filtering)
**Version:** 2.0 (Deep Refactoring - 2025-11-19)

---

## Key Features (Deep Refactoring v2.0)

### 🚀 Multi-Sitemap Strategy

| Sitemap | Description | Coverage | Use Case |
|---------|-------------|----------|----------|
| **todaynews** | Today's news (XML) | ⚡ Most recent | Current articles (recommended) |
| **todaynews_d2** | 2-day news (XML) | 📅 Last 2 days | Recent coverage |
| **article_all** | All articles index | 📚 Comprehensive | Historical discovery |
| **wantrich** | Financial news | 💰 Finance | Finance category |
| **stock** | Stock market news | 📈 Market | Stock news |
| **category** | Category sitemaps | 🗂️ Organized | Category-based |
| **car** | Automotive news | 🚗 Auto | Car category |
| **all** | All sitemaps above | 🌐 Complete | Maximum coverage |

### 📊 Crawling Modes

| Mode | Description | Speed | Use Case |
|------|-------------|-------|----------|
| **sitemap** | XML sitemap-based | ⚡ Fast | Recent articles (recommended) |
| **list** | Category list browsing | 🐢 Slower | Category-specific + deep history |

### 🎯 Comprehensive Metadata Extraction

**6+ Fallback Strategies Per Field:**

- **Title:** og:title, h1.article-title, h1, twitter:title, title tag, JSON-LD headline
- **Content:** div.article-body p, article p, div.article-content p, itemprop articleBody, div.content p, p.paragraph, JSON-LD articleBody
- **Date:** article:published_time, time datetime, pubdate, span.date, div.article-date, time text, JSON-LD datePublished
- **Author:** meta author, span.author, div.author, rel=author, JSON-LD author
- **Category:** article:section, URL extraction, span.category, a.category, JSON-LD articleSection
- **Tags:** article:tag, rel=tag, keywords
- **Images:** og:image, article img, div.article-body img, JSON-LD image
- **Description:** og:description, meta description, JSON-LD description

### ⚡ Playwright with Cloudflare Bypass

- **Cloudflare Protection:** Full Playwright support with anti-detection
- **Browser Automation:** Chromium with stealth mode
- **Concurrent Control:** 2 requests per domain for stability
- **Timeout Handling:** 60s navigation timeout for Cloudflare challenges
- **Cookie Support:** Full cookie management for session persistence

### 📈 2-Year Historical Target

- **Default:** 730 days (2 years)
- **Date Filtering:** URL-based pre-filtering (YYYYMMDD in URL)
- **Deep Pagination:** Up to 500 pages per category
- **All Categories:** 9 major categories (politics, money, society, world, entertainment, life, sports, tech)

### 📊 Comprehensive Statistics Tracking

- Per-sitemap detailed metrics
- Per-category statistics
- Metadata quality reporting
- Date range tracking
- Success rate monitoring

---

## Installation

```bash
# 1. Ensure dependencies installed
pip install scrapy scrapy-playwright

# 2. Install Playwright browsers (first time only)
playwright install chromium

# 3. Verify installation
python scripts/crawlers/cti_spider.py --help
```

---

## Usage

### Mode 1: Sitemap (Recommended - Fastest)

```bash
# Today's news (most efficient)
scrapy runspider scripts/crawlers/cti_spider.py \
    -a mode=sitemap \
    -a sitemap=todaynews \
    -a max_articles=100 \
    -o data/raw/cti_today.jsonl

# 2-day coverage
scrapy runspider scripts/crawlers/cti_spider.py \
    -a mode=sitemap \
    -a sitemap=todaynews_d2 \
    -o data/raw/cti_2days.jsonl

# All sitemaps (comprehensive)
scrapy runspider scripts/crawlers/cti_spider.py \
    -a mode=sitemap \
    -a sitemap=all \
    -o data/raw/cti_all_sitemaps.jsonl
```

### Mode 2: List (Category-Based with Deep Pagination)

```bash
# Politics category, last 30 days
scrapy runspider scripts/crawlers/cti_spider.py \
    -a mode=list \
    -a category=politics \
    -a days=30 \
    -o data/raw/cti_politics.jsonl

# All categories, 2 years (DEEP COVERAGE)
scrapy runspider scripts/crawlers/cti_spider.py \
    -a mode=list \
    -a category=all \
    -a days=730 \
    -a max_pages=500 \
    -o data/raw/cti_2years_all.jsonl
```

---

## Testing

```bash
# Run comprehensive tests (4 test cases)
cd /mnt/c/web-projects/information-retrieval
source activate ai_env
python scripts/crawlers/test_cti.py

# Tests include:
# 1. Sitemap mode (todaynews)
# 2. Sitemap mode (todaynews_d2)
# 3. List mode (politics category)
# 4. Metadata extraction quality
```

Expected output:
```
======================================================================
CTI SPIDER COMPREHENSIVE TEST (Deep Refactoring v2.0)
======================================================================
Key Features:
  - Multi-sitemap support (8+ sitemaps from robots.txt)
  - List mode with deep pagination (500 pages)
  - Playwright with Cloudflare bypass
  - 6+ fallback strategies per metadata field
  - JSON-LD structured data extraction
  - 2-year historical target (730 days)
  - Comprehensive statistics tracking
======================================================================

✓ TEST PASSED - CTI spider working (sitemap:todaynews)!
✓ TEST PASSED - 2-day sitemap working!
✓ TEST PASSED - List mode working (politics category)!
✓ TEST PASSED - Comprehensive metadata extraction working!

======================================================================
TEST SUMMARY
======================================================================
Tests passed: 4/4
  ✓ PASS - sitemap_todaynews
  ✓ PASS - sitemap_todaynews_d2
  ✓ PASS - list_mode
  ✓ PASS - metadata_extraction
```

---

## Parameters

### Mode Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | str | `sitemap` | Crawling mode (sitemap/list) |
| `sitemap` | str | `all` | Sitemap type (todaynews/todaynews_d2/article_all/all) |
| `category` | str | `all` | News category (for list mode) |

### Date Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `days` | int | `730` | Number of days to crawl (2 years default) |
| `start_date` | str | None | Start date in YYYY-MM-DD format |
| `end_date` | str | None | End date in YYYY-MM-DD format |

### Pagination Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_pages` | int | `500` | Maximum pages per category (deep pagination) |

### General Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_articles` | int | None | Maximum number of articles to scrape (optional limit) |

---

## Categories

| Category | Code | Chinese Name | URL |
|----------|------|--------------|-----|
| all | '' | 全部 | /realtimenews/ |
| politics | 260407 | 政治 | /realtimenews/?chdtv |
| money | 260410 | 財經 | /realtimenews/?money |
| society | 260402 | 社會 | /realtimenews/?society |
| world | 260408 | 國際 | /realtimenews/?world |
| entertainment | 260404 | 娛樂 | /realtimenews/?entertainment |
| life | 260405 | 生活 | /realtimenews/?life |
| sports | 260403 | 體育 | /realtimenews/?sports |
| tech | 260412 | 科技 | /realtimenews/?tech |

---

## Output Format (JSONL)

```json
{
  "article_id": "cti_20251118004599-260410",
  "url": "https://www.chinatimes.com/realtimenews/20251118004599-260410",
  "title": "台股大漲！外資回補加持 台積電ADR飆漲",
  "content": "台股18日在外資回補及台積電領軍下大漲...",
  "published_date": "2025-11-18",
  "author": "中時新聞網",
  "category": "財經",
  "category_code": "260410",
  "tags": ["台股", "外資", "台積電"],
  "image_url": "https://images.chinatimes.com/...",
  "description": "台股18日在外資回補及台積電領軍下...",
  "source": "CTI",
  "source_name": "中天新聞",
  "crawled_at": "2025-11-18T23:50:00"
}
```

---

## Performance

### Sitemap Mode (Recommended)
- **Speed:** ~3-5 articles/minute (with Cloudflare)
- **Efficiency:** Playwright required for Cloudflare bypass
- **Best For:** Recent articles, daily updates

### List Mode
- **Speed:** ~2-4 articles/minute (depends on pagination)
- **Efficiency:** Deep pagination for historical coverage
- **Best For:** Category-specific crawling, 2-year historical data

---

## URL Structure

CTI uses date-based URL pattern:
```
realtimenews/YYYYMMDDNNNNNN-CCCCCC
```

Where:
- `YYYYMMDD`: Publication date (e.g., 20251118 = 2025-11-18)
- `NNNNNN`: Sequential number within day (e.g., 004599)
- `CCCCCC`: Category code (e.g., 260410 = 財經)

Example: `realtimenews/20251118004599-260410`

---

## Cloudflare Handling

CTI uses Cloudflare protection. The spider handles it via:

1. **Playwright Integration:** Full browser automation
2. **Anti-Detection:** Stealth mode, no automation flags
3. **Timeout Management:** 60s navigation timeout
4. **Challenge Detection:** Automatic "Just a moment..." detection
5. **Session Persistence:** Cookie management

---

## Improvements (v2.0 vs v1.0)

| Feature | Before (v1.0) | After (v2.0) | Improvement |
|---------|---------------|--------------|-------------|
| **Sitemaps** | ❌ None | ✅ 8+ sitemaps | Multi-source discovery |
| **Modes** | Basic list | 2 modes (sitemap/list) | Flexible strategy |
| **Cloudflare** | ❌ Not handled | ✅ Playwright bypass | Full access |
| **Metadata** | Basic (3-4 strategies) | 6+ strategies per field | Comprehensive |
| **JSON-LD** | ❌ Not supported | ✅ Full support | Structured data |
| **Deep Pagination** | Limited | Up to 500 pages | 2-year coverage |
| **Categories** | Limited | All 9 categories | Complete coverage |
| **Date Filtering** | Post-fetch only | Pre-fetch (URL-based) | Efficient |
| **Statistics** | Basic counts | Comprehensive tracking | Detailed insights |

---

## Troubleshooting

### Issue: Playwright not installed
```bash
playwright install chromium
```

### Issue: Cloudflare challenge timeout
Increase timeout in settings:
```python
'PLAYWRIGHT_DEFAULT_NAVIGATION_TIMEOUT': 120000  # 2 minutes
```

### Issue: Slow crawling
- Use `mode=sitemap` for fastest results
- Adjust `CONCURRENT_REQUESTS` carefully (Cloudflare sensitive)
- Use `max_articles` parameter to limit scope

### Issue: Too many Cloudflare challenges
- Reduce `CONCURRENT_REQUESTS` to 1
- Increase `DOWNLOAD_DELAY` to 5 seconds
- Use `max_articles` to limit requests

---

## Mass Data Collection Commands

### Collect Last 30 Days (Recommended)
```bash
scrapy runspider scripts/crawlers/cti_spider.py \
    -a mode=sitemap \
    -a sitemap=all \
    -a days=30 \
    -o data/raw/cti_30days.jsonl \
    -s LOG_LEVEL=INFO
```

### Collect 2 Years Per Category (Deep Historical)
```bash
for cat in politics money society world entertainment life sports tech; do
    echo "Collecting category: $cat (2 years)"
    scrapy runspider scripts/crawlers/cti_spider.py \
        -a mode=list \
        -a category=$cat \
        -a days=730 \
        -a max_pages=500 \
        -o data/raw/cti_${cat}_2years.jsonl \
        -s LOG_LEVEL=INFO
    sleep 60  # Pause between categories to avoid rate limiting
done
```

### Collect All Sitemaps (Comprehensive)
```bash
scrapy runspider scripts/crawlers/cti_spider.py \
    -a mode=sitemap \
    -a sitemap=all \
    -o data/raw/cti_all_sitemaps.jsonl \
    -s LOG_LEVEL=INFO
```

---

## Statistics Reporting

Spider provides comprehensive statistics on closure:

```
======================================================================
CTI Spider Statistics (Deep Refactoring v2.0)
======================================================================
Mode: sitemap
Sitemaps processed: todaynews, todaynews_d2
Articles found: 1,234
Articles successfully scraped: 1,200
Articles failed: 34
Success rate: 97.2%

MODE-SPECIFIC STATISTICS:
  Sitemap URLs discovered: 1,234
  List pages visited: 0

METADATA QUALITY:
  has_title: 1200/1200 (100.0%)
  has_content: 1200/1200 (100.0%)
  has_date: 1180/1200 (98.3%)
  has_author: 950/1200 (79.2%)
  has_category: 1200/1200 (100.0%)
  has_tags: 800/1200 (66.7%)
  has_image: 1100/1200 (91.7%)

Date range: 2025-11-16 to 2025-11-18
======================================================================
```

---

## Notes

1. **Cloudflare protection:** Playwright required for ALL requests
2. **Sitemap mode recommended** for most use cases (faster, efficient)
3. **List mode for historical:** Deep pagination up to 500 pages for 2-year coverage
4. **Rate limiting:** Cloudflare sensitive - use respectful delays
5. **Date filtering:** Pre-applied at URL level for efficiency
6. **Metadata quality:** High (95%+ on core fields)
7. **2-year historical target:** Achievable via list mode with deep pagination

---

## Author

Information Retrieval System
Date: 2025-11-19
Version: 2.0 (Deep Refactoring)
