# Crawler Code Analysis Report

## 1. Architecture Overview

The crawling system is built on **Scrapy + Playwright** with a layered architecture:

```
┌─────────────────────────────────────────────────┐
│  Orchestration Layer                             │
│  mass_collect.py (999 lines)                     │
│  run_crawlers.py                                  │
│  health_check.py, test_single_crawler.py         │
├─────────────────────────────────────────────────┤
│  Spider Layer (15 news source spiders)           │
│  cna_spider(_v2/_simple), pts_spider,            │
│  ltn_spider, udn_spider, ettoday_spider,         │
│  ftv_spider, tvbs_spider, chinatimes_spider,     │
│  storm_spider, setn_spider, cti_spider,          │
│  nextapple_spider, apple_daily_spider,           │
│  technews_spider, yahoo_spider                   │
├─────────────────────────────────────────────────┤
│  Base Class Layer                                │
│  base_playwright_spider.py                       │
│  - Fingerprint randomization                     │
│  - UA/viewport/timezone rotation                 │
│  - Date parsing (Chinese + ROC year)             │
├─────────────────────────────────────────────────┤
│  Middleware Layer                                │
│  stealth_middleware.py                           │
│  humanization_middleware.py                      │
│  - JS injection to hide navigator.webdriver      │
│  - Chrome runtime patching                       │
│  - WebGL/vendor spoofing                         │
│  - Scroll/mouse/reading simulation               │
├─────────────────────────────────────────────────┤
│  Utility Layer                                   │
│  utils/jobdir.py  (resume detection)             │
│  utils/validate_dataset.py                       │
│  news_api_fetcher.py (NewsAPI + GDELT)           │
└─────────────────────────────────────────────────┘
```

## 2. Data Flow Pipeline

```
[Start] → mass_collect.py (orchestrator)
    ↓
[Configure] per-source settings (concurrency, delay, Playwright on/off)
    ↓
[Spawn] Scrapy CrawlerProcess or subprocess per spider
    ↓
[Discovery] sitemap / list pages / sequential ID enumeration
    ↓
[Fetch] HTTP request or Playwright browser page
    ↓
[Middleware] stealth JS injection + humanization (scroll, mouse, delay)
    ↓
[Parse] CSS selector fallback chain (6+ strategies per field)
    ↓
[Normalize] unified JSONL schema
    ↓
[Output] /mnt/c/data/information-retrieval/raw/{source}_news_{timestamp}.jsonl
    ↓
[Post-process] merge_jsonl.py → preprocess_batch.py → build_indexes.py
```

## 3. Unified Output Schema

Every spider produces the same JSONL schema:

```json
{
  "article_id": "unique_id_or_url_hash",
  "url": "https://...",
  "source": "cna|pts|ltn|udn|...",
  "source_name": "中央社|公視|自由時報|...",
  "title": "...",
  "content": "...",
  "author": "...",
  "publish_date": "2024-01-15",
  "category": "politics|finance|tech|...",
  "category_name": "政治|財經|科技|...",
  "tags": ["tag1", "tag2"],
  "image_url": "https://...",
  "crawled_at": "2024-01-15T10:30:00+08:00"
}
```

## 4. Crawling Strategies (4 Patterns)

### 4.1 Sitemap-Based (Fastest, No Playwright)
**Used by**: CTI, NextApple, Yahoo, SETN, TVBS(sitemap mode)
```python
# Pattern:
# 1. Fetch XML sitemap (e.g., https://site.com/sitemap.xml)
# 2. Parse <url><loc>...</loc></url> entries
# 3. Filter by date range from URL or <lastmod>
# 4. Fetch each article URL → parse content
# No browser needed; pure HTTP + XML parsing
# Rate: 4-8 concurrent requests, 0.5-1.5s delay
```

### 4.2 List Page + Pagination
**Used by**: CNA v2, ETtoday, ChinaTimes, Storm(list mode)
```python
# Pattern:
# 1. Start from category list page (e.g., /list/politics.aspx)
# 2. Extract article links via CSS selectors
# 3. Follow pagination (?page=N or /page/N/)
# 4. For each article link: fetch detail page → parse
# Usually requires Playwright for JS-rendered pages
# Rate: 1-2 concurrent, 1.5-3s delay
```

### 4.3 Sequential ID Enumeration
**Used by**: LTN, UDN, PTS, Storm(sequential mode)
```python
# Pattern:
# 1. Discover max article ID from /dailynews or similar page
# 2. Enumerate URLs: https://site.com/article/{ID}
# 3. Handle 404s (sparse ID space: 0.01%-40% hit rate)
# 4. Parse each valid article page
# No Playwright needed (static HTML)
# Rate: 3-8 concurrent, 0.5-1.5s delay
# Key insight: some sites have predictable URL patterns
```

### 4.4 Hybrid (List + Sequential Fill)
**Used by**: UDN, PTS, TVBS, Storm
```python
# Pattern:
# 1. Crawl list pages for recent articles (fast discovery)
# 2. Identify gaps in ID space
# 3. Fill gaps with sequential enumeration
# Best coverage but most complex
```

## 5. Anti-Detection Mechanisms

### 5.1 Browser Fingerprint (BasePlaywrightSpider)
| Feature | Implementation |
|---------|---------------|
| User-Agent | `fake_useragent` library or 3-UA fallback pool |
| Viewport | Random from 6 resolutions (1366x768 to 2560x1440) |
| Device scale | Random: 1, 1.5, or 2 |
| Timezone | Asia/Taipei |
| Locale | zh-TW |
| Touch support | Random True/False |
| Extra headers | Accept-Language, Upgrade-Insecure-Requests |

### 5.2 Stealth JS Injection (StealthMiddleware)
Injected via `PageMethod('add_init_script', script=...)` before page load:
- `navigator.webdriver` → `undefined`
- `window.chrome.runtime` → mock object
- `navigator.permissions.query` → spoofed
- `navigator.plugins` → 3 fake plugins (Chrome PDF, Native Client)
- `navigator.languages` → `['zh-TW', 'zh', 'en-US', 'en']`
- WebGL `getParameter` → spoof Intel Inc. vendor
- `Function.prototype.toString` → return `[native code]` strings
- Delete `navigator.__proto__.webdriver`
- Modernizr stub

### 5.3 Behavioral Humanization (HumanizationMiddleware)
- **Scrolling**: 1-4 scroll actions per page, 20-80% positions, occasional scroll-up (re-reading)
- **Mouse movement**: Random X/Y dispatch at 50% probability
- **Delays**: Gaussian distribution (configurable min/max)
- **Reading time**: 1-3s delay based on content
- **Advanced mode**: Gradual scroll with reading pauses, click simulation, text selection

### 5.4 Rate Limiting (Per-Site)
| Site | Concurrency | Delay | AutoThrottle |
|------|-------------|-------|-------------|
| CNA | 1 | 3s | Yes (3-15s) |
| ETtoday | 1 | 3s | Yes |
| UDN | 3 | 1.5s | No |
| PTS | 8 | 0.5s | No |
| Yahoo | 8 | 0.5s | No |
| CTI | 4 | 0.8s | No |
| FTV | 1 | 1s | No |

## 6. Content Extraction Pattern

Every spider uses a **fallback chain** (6+ strategies per field):

```python
# Title extraction (example from TVBS):
# Priority 1: JSON-LD headline
json_ld.get('headline')
# Priority 2: og:title meta tag
response.css('meta[property="og:title"]::attr(content)').get()
# Priority 3: h1.element
response.css('h1.article-title::text').get()
# Priority 4: title tag
response.css('title::text').get()
# Priority 5: header h1
response.css('header h1::text').get()
# Priority 6: fallback default
''

# Content extraction follows same pattern:
# JSON-LD articleBody → CSS selectors → text fallback
```

## 7. Key Reusable Components for Other Projects

### 7.1 `BasePlaywrightSpider` → Standalone Package
**What to extract**: The entire class (~200 lines)
**Dependencies**: `scrapy`, `scrapy-playwright`, `fake_useragent`
**How to reuse**:
```python
# In new project:
from your_package import BasePlaywrightSpider

class MySpider(BasePlaywrightSpider):
    name = 'my_spider'
    allowed_domains = ['example.com']
    
    def start_requests(self):
        yield scrapy.Request(
            'https://example.com/page',
            meta=self.get_playwright_meta(
                wait_selector='div.content',
                wait_selector_timeout=15000
            ),
            callback=self.parse
        )
```

### 7.2 `StealthMiddleware` → Drop-in Middleware
**What to extract**: The full JS injection script (~150 lines)
**How to reuse**: Add to any Scrapy project's `DOWNLOADER_MIDDLEWARES`:
```python
DOWNLOADER_MIDDLEWARES = {
    'your_package.StealthMiddleware': 585,
}
```

### 7.3 `HumanizationMiddleware` → Drop-in Middleware
**What to extract**: Full class (~200 lines)
**How to reuse**: Add after stealth middleware:
```python
DOWNLOADER_MIDDLEWARES = {
    'your_package.StealthMiddleware': 585,
    'your_package.HumanizationMiddleware': 586,
}
```

### 7.4 Crawl Strategy Patterns
For any new site, follow this decision tree:
1. **Check for sitemap** → `robots.txt` → `/sitemap.xml` → fastest, no Playwright
2. **Check for list pages** → category pages with pagination → moderate speed
3. **Check for URL patterns** → sequential ID enumeration → if sparse, use hybrid
4. **If JS-rendered** → use Playwright via `BasePlaywrightSpider`
5. **If anti-bot** → add `StealthMiddleware` + `HumanizationMiddleware`

### 7.5 Mass Collection Framework
**What to extract**: `mass_collect.py` (~999 lines)
**Features to reuse**:
- Date range calculation (`--days N`, `--months N`, `--date-range START END`)
- Per-source configuration registry
- Parallel execution with `ThreadPoolExecutor`
- JOBDIR resume with corruption detection
- Per-task timeout + retry with backoff
- Progress tracking and statistics

## 8. File Inventory

| Category | Files | Total Lines |
|----------|-------|-------------|
| Base classes | 1 | ~350 |
| Spiders | 15 (+ 2 variants) | ~3,500 |
| Middlewares | 2 | ~400 |
| Utilities | 3 | ~200 |
| Orchestration | 4 | ~1,800 |
| **Total** | **25+ files** | **~6,250 lines** |

## 9. Dependencies

```
scrapy==2.11.0              # Core framework
scrapy-playwright==0.0.34   # Playwright integration
playwright==1.40.0          # Browser automation
fake-useragent==1.4.0       # UA rotation
requests==2.31.0            # HTTP (API fetcher)
beautifulsoup4==4.12.2      # HTML parsing
lxml==4.9.3                 # XML parser
scrapy-rotating-proxies==0.6.2  # Proxy rotation (optional)
apscheduler==3.10.4         # Task scheduling
```

## 10. Data Warehouse Paths (Updated)

All data now resides at `/mnt/c/data/information-retrieval/`:

```
/mnt/c/data/information-retrieval/
├── raw/                    # Crawled JSONL files (per-spider output)
├── processed/              # Cleaned/deduplicated data
├── indexes/                # Inverted indexes, BERT embeddings
├── index_50k/              # 50K article index
├── indexes_10k/            # 10K article index + BERT
├── index_test/             # Test index (89 docs)
├── evaluation/             # QRELS, test queries
├── stats/                  # Build/query/stats reports
├── test/                   # Test datasets
├── test_results/           # Module test output
├── datasets/               # Lexicon, stopwords, mini datasets
└── configs/                # Scrapy settings, logging config
```
