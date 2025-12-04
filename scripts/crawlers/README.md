# 新聞爬蟲系統 News Crawler System

台灣主要新聞媒體爬蟲系統，支援多平台新聞採集、反爬蟲機制、自動化排程。

## 功能特性 Features

- ✅ **多平台整合**: 整合多個台灣主要新聞媒體
- ✅ **Playwright 反爬蟲**: 瀏覽器指紋隨機化、User-Agent 輪換
- ✅ **統一管理介面**: 單一命令執行多個爬蟲
- ✅ **API 整合**: NewsAPI、GDELT 等免費新聞 API
- ⏳ **自動化排程**: APScheduler 定時爬取（開發中）
- ⏳ **增量爬取**: 智能去重與增量更新（開發中）

## 已支援媒體 Supported Sources

| 媒體名稱 | 代碼 | 類型 | 狀態 |
|---------|------|------|------|
| 中央社 CNA | `cna` | 傳統爬蟲 | ✅ 完成 |
| 自由時報 LTN | `ltn` | Playwright | ✅ 完成 |
| 公視 PTS | `pts` | 傳統爬蟲 | ✅ 完成 |
| 聯合報 UDN | `udn` | Playwright | ✅ 完成 |
| 蘋果日報 Apple Daily | `apple` | Playwright | ✅ 完成 |
| TVBS新聞 TVBS News | `tvbs` | Playwright | ✅ 完成 |
| 中時新聞網 China Times | `chinatimes` | 傳統爬蟲 | ✅ 完成 |
| 東森新聞雲 ETtoday | `ettoday` | Playwright | ✅ 完成 |
| 風傳媒 Storm Media | `storm` | Playwright | ✅ 完成 |
| NewsAPI | - | API | ✅ 完成 |
| GDELT | - | API | ✅ 完成 |

## 安裝 Installation

### 1. 安裝 Python 依賴

```bash
# 安裝核心依賴
pip install -r requirements.txt

# 或手動安裝爬蟲相關套件
pip install scrapy==2.11.0 \
            scrapy-playwright==0.0.34 \
            playwright==1.40.0 \
            fake-useragent==1.4.0 \
            requests==2.31.0 \
            beautifulsoup4==4.12.2 \
            lxml==4.9.3
```

### 2. 安裝 Playwright 瀏覽器

```bash
# 安裝 Chromium（推薦）
playwright install chromium

# 或安裝所有瀏覽器
playwright install
```

### 3. 驗證安裝

```bash
# 檢查 Scrapy
scrapy version

# 檢查 Playwright
playwright --version

# 列出可用爬蟲
python scripts/crawlers/run_crawlers.py --list
```

## 使用方式 Usage

### 統一管理介面 (推薦)

```bash
# 基本用法：執行單一爬蟲
python scripts/crawlers/run_crawlers.py --crawler ltn --days 7

# 執行多個爬蟲
python scripts/crawlers/run_crawlers.py --crawler ltn,cna --days 3

# 執行所有爬蟲
python scripts/crawlers/run_crawlers.py --all --days 7

# 指定日期範圍
python scripts/crawlers.py --crawler ltn \
    --start-date 2025-11-01 \
    --end-date 2025-11-13

# 指定分類（如支援）
python scripts/crawlers/run_crawlers.py --crawler ltn \
    --days 3 --category politics
```

### 直接使用 Scrapy (進階)

```bash
# LTN 自由時報
scrapy runspider scripts/crawlers/ltn_spider.py \
    -a days=7 \
    -a category=politics \
    -o data/raw/ltn_politics.jsonl

# CNA 中央社
scrapy runspider scripts/crawlers/cna_spider.py \
    -a start_date=2025-11-01 \
    -a end_date=2025-11-13 \
    -o data/raw/cna_news.jsonl
```

### 新聞 API 擷取

```bash
# GDELT (免費，無需 API key)
python scripts/crawlers/news_api_fetcher.py \
    --query "Taiwan AI" \
    --days 7 \
    --source gdelt \
    --output data/raw/gdelt_taiwan.jsonl

# NewsAPI (需要免費 API key)
export NEWSAPI_KEY="your_api_key_here"
python scripts/crawlers/news_api_fetcher.py \
    --query "台灣" \
    --days 7 \
    --source newsapi \
    --output data/raw/newsapi_taiwan.jsonl

# 整合多個來源
python scripts/crawlers/news_api_fetcher.py \
    --query "台灣 AI" \
    --days 7 \
    --source all \
    --output data/raw/unified_news.jsonl
```

## 輸出格式 Output Format

所有爬蟲輸出統一的 JSONL 格式：

```json
{
  "article_id": "a1b2c3d4e5f6g7h8",
  "title": "文章標題",
  "content": "文章內容...",
  "url": "https://news.example.com/article/123",
  "published_date": "2025-11-17",
  "author": "記者姓名",
  "source": "LTN",
  "source_name": "自由時報",
  "category": "politics",
  "category_name": "政治",
  "tags": ["AI", "科技", "台灣"],
  "image_url": "https://...",
  "crawled_at": "2025-11-17T10:30:00"
}
```

## 爬蟲設定 Configuration

### 反爬蟲設定

所有 Playwright 爬蟲預設啟用以下反爬蟲機制：

1. **User-Agent 輪換**: 隨機切換瀏覽器標識
2. **視窗尺寸隨機化**: 模擬不同螢幕解析度
3. **人性化延遲**: 隨機延遲 1-3 秒
4. **指紋隨機化**: 時區、語言、設備資訊
5. **Stealth 模式**: 隱藏自動化標記

### 自訂設定

在爬蟲腳本中調整 `custom_settings`:

```python
custom_settings = {
    'DOWNLOAD_DELAY': 3,              # 請求間隔（秒）
    'CONCURRENT_REQUESTS_PER_DOMAIN': 1,  # 並發請求數
    'RETRY_TIMES': 3,                 # 重試次數
}
```

## 資料處理流程 Data Pipeline

```
1. 爬取 (Crawling)
   └─> data/raw/*.jsonl

2. 預處理 (Preprocessing)
   └─> python scripts/preprocess_news.py
   └─> data/preprocessed/*.jsonl

3. 建立索引 (Indexing)
   └─> python scripts/build_field_index.py
   └─> data/indexes/field_index.pkl

4. 查詢 (Querying)
   └─> Flask API /api/advanced_search
```

## 開發指南 Development

### 新增爬蟲

1. 繼承 `BasePlaywrightSpider`:

```python
from base_playwright_spider import BasePlaywrightSpider

class NewSpider(BasePlaywrightSpider):
    name = 'new_spider'
    allowed_domains = ['news.example.com']

    def start_requests(self):
        yield scrapy.Request(
            url='https://news.example.com',
            callback=self.parse,
            meta=self.get_playwright_meta()
        )

    def parse(self, response):
        # 提取文章連結
        # 解析文章內容
        yield article_data
```

2. 註冊到 `run_crawlers.py`:

```python
CRAWLERS = {
    'new': CrawlerConfig(
        name='new',
        spider_class='NewSpider',
        module_path='new_spider',
        description='新媒體 New Media',
        requires_playwright=True
    ),
}
```

### 測試爬蟲

```bash
# 限制爬取項目數（測試用）
scrapy runspider scripts/crawlers/ltn_spider.py \
    -a days=1 \
    -s CLOSESPIDER_ITEMCOUNT=5 \
    -o test_output.jsonl

# 啟用詳細日誌
scrapy runspider scripts/crawlers/ltn_spider.py \
    -a days=1 \
    -s LOG_LEVEL=DEBUG
```

## 常見問題 FAQ

### Q1: 爬蟲被封鎖怎麼辦？

**A:** 調整以下設定：

```bash
# 增加延遲
-s DOWNLOAD_DELAY=5

# 減少並發
-s CONCURRENT_REQUESTS_PER_DOMAIN=1

# 啟用代理（需配置代理池）
-s ROTATING_PROXY_ENABLED=True
```

### Q2: Playwright 瀏覽器啟動失敗？

**A:** 重新安裝瀏覽器：

```bash
playwright install --force chromium
```

### Q3: 如何設定 NewsAPI key？

**A:** 設定環境變數：

```bash
# Linux/Mac
export NEWSAPI_KEY="your_key_here"

# Windows CMD
set NEWSAPI_KEY=your_key_here

# Windows PowerShell
$env:NEWSAPI_KEY="your_key_here"
```

### Q4: 輸出檔案在哪裡？

**A:** 預設輸出位置：

- 爬蟲輸出: `data/raw/<spider_name>_news_<timestamp>.jsonl`
- API 輸出: 由 `--output` 參數指定
- 預處理輸出: `data/preprocessed/`
- 索引檔案: `data/indexes/`

## 效能優化 Performance

### 記憶體優化

```python
# 使用 JSONL streaming 而非一次載入
with open('data.jsonl', 'r') as f:
    for line in f:
        doc = json.loads(line)
        process(doc)
```

### 速度優化

```bash
# 增加並發（注意：可能被封鎖）
-s CONCURRENT_REQUESTS=16
-s CONCURRENT_REQUESTS_PER_DOMAIN=2

# 停用不必要的中介軟體
-s DOWNLOADER_MIDDLEWARES={'scrapy.downloadermiddlewares.useragent.UserAgentMiddleware': None}
```

## 授權與限制 License

- **用途**: 僅供學術研究與教育用途
- **限制**: 請遵守各新聞網站的 robots.txt 與服務條款
- **責任**: 使用者需自行承擔爬蟲使用責任

## 技術支援 Support

- **問題回報**: GitHub Issues
- **文檔**: `docs/guides/`
- **範例**: `scripts/crawlers/`

---

**Last Updated**: 2025-11-18
**Version**: 1.1.0
**Maintainer**: Information Retrieval System Development Team
