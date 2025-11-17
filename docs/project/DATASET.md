# 資料集規劃文件 (Dataset Planning Document)

**中文新聞智能檢索系統 (CNIRS)**
**Chinese News Intelligent Retrieval System**

---

## 目錄 (Table of Contents)

1. [資料集概述](#1-資料集概述-dataset-overview)
2. [資料來源](#2-資料來源-data-sources)
3. [資料收集策略](#3-資料收集策略-data-collection-strategy)
4. [資料結構設計](#4-資料結構設計-data-structure-design)
5. [資料預處理流程](#5-資料預處理流程-data-preprocessing-pipeline)
6. [資料品質控制](#6-資料品質控制-data-quality-control)
7. [資料集統計分析](#7-資料集統計分析-dataset-statistics)
8. [評估資料集](#8-評估資料集-evaluation-dataset)
9. [資料儲存方案](#9-資料儲存方案-data-storage-solution)
10. [資料更新策略](#10-資料更新策略-data-update-strategy)

---

## 1. 資料集概述 (Dataset Overview)

### 1.1 資料集規模

| 項目 | 目標值 | 說明 |
|-----|-------|------|
| **文檔總數** | 30,000 - 50,000 篇 | 中型規模，適合學術研究與教學 |
| **時間範圍** | 2022-01-01 ~ 2024-12-31 | 3年期間，涵蓋疫後復甦與AI發展 |
| **語言** | 繁體中文 | 台灣正體中文新聞 |
| **領域分布** | 多領域平衡 | 科技、政治、經濟、社會、體育等 |
| **平均文章長度** | 500-2000 字 | 典型新聞文章長度 |
| **儲存空間** | ~2-5 GB (原始) | 未壓縮文本 + metadata |

### 1.2 資料集目標

**學術研究目標**:
- 比較傳統 IR (Boolean, TF-IDF, BM25) 與現代 NLP (BERT) 效能差異
- 評估中文 NER、關鍵字提取、主題建模在真實新聞上的表現
- 研究查詢擴展 (Rocchio, Synonym) 對檢索效果的影響
- 分析不同聚類演算法在新聞分類上的效果

**教學目標**:
- 提供真實、豐富的中文文本資料
- 支援完整 IR 系統開發流程 (索引→檢索→評估)
- 展示 NLP 技術在實際場景的應用

---

## 2. 資料來源 (Data Sources)

### 2.1 主要新聞來源

#### A. 中央社 (CNA - Central News Agency)

**網站**: https://www.cna.com.tw/
**授權**: 新聞內容僅供學術研究使用

**優勢**:
- 官方通訊社，內容可靠
- 涵蓋全領域新聞
- 標準繁體中文，文法正確
- 提供 RSS Feed

**爬取頻道**:
- 政治 (Politics)
- 財經 (Finance)
- 科技 (Technology)
- 社會 (Society)
- 國際 (International)
- 生活 (Lifestyle)
- 文化 (Culture)
- 運動 (Sports)

**預計收集**: 20,000 篇

---

#### B. 公視新聞網 (PTS News)

**網站**: https://news.pts.org.tw/
**授權**: 公共電視，教育用途友善

**優勢**:
- 深度報導，文章長度適中
- 高品質新聞內容
- 多元議題涵蓋
- 無廣告干擾

**預計收集**: 10,000 篇

---

#### C. 科技新報 (TechNews)

**網站**: https://technews.tw/
**授權**: 部分內容 CC 授權

**優勢**:
- 專注科技領域
- 深入技術分析
- AI、5G、半導體等熱門主題
- 適合測試專業術語處理

**預計收集**: 5,000 篇

---

### 2.2 備用來源 (Optional)

| 來源 | 領域 | 預計篇數 | 備註 |
|-----|------|---------|------|
| **聯合新聞網** | 綜合 | 5,000 | 若主要來源不足 |
| **自由時報** | 綜合 | 5,000 | 政治色彩較重 |
| **天下雜誌** | 財經/深度 | 2,000 | 長文深度報導 |
| **風傳媒** | 評論/分析 | 2,000 | 觀點多元 |

---

### 2.3 資料收集合法性聲明

**重要聲明**:
1. 本專案**僅供學術研究與教育用途**，不做商業使用
2. 遵守各新聞網站的 `robots.txt` 爬蟲協議
3. 設定合理的爬蟲速率 (1-2 秒/請求)，避免對伺服器造成負擔
4. 尊重著作權，僅儲存必要的文本與 metadata
5. 若使用公開資料集 (如 TTDS, Taiwan Text Dataset)，遵守其授權條款

---

## 3. 資料收集策略 (Data Collection Strategy)

### 3.1 爬蟲設計 (Web Scraping)

#### 3.1.1 技術選擇

```python
# 使用 Scrapy 框架
import scrapy
from scrapy.crawler import CrawlerProcess
from datetime import datetime, timedelta

class CNANewsSpider(scrapy.Spider):
    name = 'cna_news'
    allowed_domains = ['cna.com.tw']

    # 起始 URL (依日期範圍)
    def start_requests(self):
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2024, 12, 31)

        # 每日新聞列表頁
        current_date = start_date
        while current_date <= end_date:
            url = f"https://www.cna.com.tw/list/aall/{current_date.strftime('%Y%m%d')}.aspx"
            yield scrapy.Request(url, callback=self.parse_list_page)
            current_date += timedelta(days=1)

    # 解析列表頁
    def parse_list_page(self, response):
        article_links = response.css('div.mainList a::attr(href)').getall()
        for link in article_links:
            yield response.follow(link, callback=self.parse_article)

    # 解析文章頁
    def parse_article(self, response):
        yield {
            'url': response.url,
            'title': response.css('h1.centralContent span::text').get(),
            'content': ' '.join(response.css('div.paragraph p::text').getall()),
            'author': response.css('div.author::text').get(),
            'publish_date': response.css('div.updatetime span::text').get(),
            'category': response.css('div.breadcrumb a::text').getall()[-1],
            'source': 'CNA',
            'crawled_at': datetime.now().isoformat()
        }
```

#### 3.1.2 爬蟲配置 (Scrapy Settings)

```python
# settings.py
BOT_NAME = 'cnirs_crawler'

# 遵守 robots.txt
ROBOTSTXT_OBEY = True

# 下載延遲 (秒)
DOWNLOAD_DELAY = 2  # 每 2 秒 1 個請求

# 併發限制
CONCURRENT_REQUESTS = 4
CONCURRENT_REQUESTS_PER_DOMAIN = 2

# User-Agent
USER_AGENT = 'CNIRS Academic Research Bot (contact: your-email@example.com)'

# 自動節流 (AutoThrottle)
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 2
AUTOTHROTTLE_MAX_DELAY = 10
AUTOTHROTTLE_TARGET_CONCURRENCY = 2.0

# 輸出格式
FEEDS = {
    'data/raw/cna_news_%(time)s.jsonl': {
        'format': 'jsonlines',
        'encoding': 'utf8',
        'store_empty': False,
    }
}

# 中介軟體
DOWNLOADER_MIDDLEWARES = {
    'scrapy.downloadermiddlewares.useragent.UserAgentMiddleware': None,
    'scrapy.downloadermiddlewares.retry.RetryMiddleware': 90,
    'scrapy_fake_useragent.middleware.RandomUserAgentMiddleware': 400,
}

# 重試設定
RETRY_TIMES = 3
RETRY_HTTP_CODES = [500, 502, 503, 504, 522, 524, 408, 429]
```

---

### 3.2 RSS Feed 訂閱 (Alternative)

```python
# 使用 feedparser 訂閱 RSS
import feedparser
from datetime import datetime
import time

def collect_from_rss(rss_url: str, max_articles: int = 100) -> list:
    """Collect articles from RSS feed."""
    feed = feedparser.parse(rss_url)
    articles = []

    for entry in feed.entries[:max_articles]:
        article = {
            'url': entry.link,
            'title': entry.title,
            'summary': entry.get('summary', ''),
            'publish_date': entry.get('published', ''),
            'source': feed.feed.get('title', 'Unknown'),
            'crawled_at': datetime.now().isoformat()
        }
        articles.append(article)

        # 獲取完整內容 (需額外請求)
        # full_content = fetch_full_content(entry.link)
        # article['content'] = full_content

    return articles

# CNA RSS Feeds
CNA_RSS_FEEDS = {
    'politics': 'https://www.cna.com.tw/rss/aipl.xml',
    'finance': 'https://www.cna.com.tw/rss/aife.xml',
    'technology': 'https://www.cna.com.tw/rss/ait.xml',
    'society': 'https://www.cna.com.tw/rss/asoc.xml',
    'international': 'https://www.cna.com.tw/rss/aopl.xml',
    'lifestyle': 'https://www.cna.com.tw/rss/ahel.xml',
    'sports': 'https://www.cna.com.tw/rss/aspt.xml',
}
```

---

### 3.3 資料收集時程

```
Week 1-2: 爬蟲開發與測試
├─ Day 1-3: 爬蟲設計與實作 (CNA, PTS, TechNews)
├─ Day 4-5: 爬蟲測試與除錯 (小規模測試: 100-500 篇)
└─ Day 6-7: 正式爬取 (全量: 30,000-50,000 篇)

Week 3: 資料清洗與驗證
├─ Day 1-3: 資料去重、格式統一
├─ Day 4-5: 品質檢查 (缺失值、異常值)
└─ Day 6-7: 資料統計分析與可視化

Week 4: 資料預處理與索引建立
├─ Day 1-2: 斷詞、NER
├─ Day 3-4: 關鍵字提取、主題建模
├─ Day 5-6: 索引建立 (Inverted, Positional, TF-IDF)
└─ Day 7: BERT 嵌入生成
```

---

## 4. 資料結構設計 (Data Structure Design)

### 4.1 原始資料格式 (Raw Data Format)

#### JSON Lines (.jsonl)

```json
{
  "url": "https://www.cna.com.tw/news/ait/202403150123.aspx",
  "title": "OpenAI 發布 GPT-5 引領 AI 新紀元",
  "content": "OpenAI 今日正式發布最新一代語言模型 GPT-5，號稱在推理能力、多模態理解、以及長文本處理上有顯著突破。執行長 Sam Altman 表示...",
  "author": "張三",
  "publish_date": "2024-03-15T10:30:00+08:00",
  "category": "科技",
  "source": "CNA",
  "tags": ["AI", "OpenAI", "GPT-5", "人工智慧"],
  "image_url": "https://imgcdn.cna.com.tw/...",
  "crawled_at": "2024-03-16T08:15:23+08:00"
}
```

---

### 4.2 預處理後資料格式 (Preprocessed Data Format)

```json
{
  "doc_id": 12345,
  "url": "https://www.cna.com.tw/news/ait/202403150123.aspx",
  "title": "OpenAI 發布 GPT-5 引領 AI 新紀元",
  "content": "OpenAI 今日正式發布最新一代語言模型 GPT-5...",
  "author": "張三",
  "publish_date": "2024-03-15",
  "category": "科技",
  "source": "CNA",

  // 預處理欄位
  "tokens_title": ["OpenAI", "發布", "GPT-5", "引領", "AI", "新", "紀元"],
  "tokens_content": ["OpenAI", "今日", "正式", "發布", ...],
  "tokens_count": 456,

  // NER 實體
  "entities": [
    {"text": "OpenAI", "type": "ORGANIZATION", "start": 0, "end": 6},
    {"text": "GPT-5", "type": "PRODUCT", "start": 10, "end": 15},
    {"text": "Sam Altman", "type": "PERSON", "start": 156, "end": 166}
  ],

  // 關鍵字 (ensemble of 4 methods)
  "keywords": [
    {"word": "GPT-5", "score": 0.89, "method": "ensemble"},
    {"word": "OpenAI", "score": 0.76, "method": "ensemble"},
    {"word": "語言模型", "score": 0.68, "method": "ensemble"},
    {"word": "推理能力", "score": 0.55, "method": "ensemble"}
  ],

  // 主題建模
  "topic_lda": {
    "topic_id": 3,
    "topic_label": "人工智慧與科技",
    "probability": 0.82
  },
  "topic_bert": {
    "topic_id": 7,
    "topic_label": "AI 模型發展",
    "probability": 0.91
  },

  // 句法分析 (SVO)
  "svo_triples": [
    {"subject": "OpenAI", "verb": "發布", "object": "GPT-5"},
    {"subject": "Sam Altman", "verb": "表示", "object": null}
  ],

  // 摘要
  "summary": "OpenAI 發布最新語言模型 GPT-5，在推理能力與多模態理解上有顯著突破。",

  // 索引欄位
  "tfidf_vector": [0.0, 0.12, 0.0, 0.34, ...],  // Sparse vector
  "bert_embedding": [0.012, -0.045, 0.123, ...],  // Dense vector (768-dim)

  // Metadata
  "preprocessed_at": "2024-03-17T14:20:00+08:00",
  "indexed_at": "2024-03-17T15:00:00+08:00"
}
```

---

### 4.3 資料庫 Schema (SQLite)

```sql
-- 主表: 新聞文章
CREATE TABLE news (
    -- 基本欄位
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    author TEXT,
    publish_date DATE NOT NULL,
    category TEXT,
    source TEXT NOT NULL,

    -- 預處理欄位 (JSON 格式)
    tokens_title TEXT,           -- JSON: ["token1", "token2", ...]
    tokens_content TEXT,          -- JSON: ["token1", ...]
    tokens_count INTEGER,

    entities TEXT,                -- JSON: [{"text": "...", "type": "...", ...}, ...]
    keywords TEXT,                -- JSON: [{"word": "...", "score": 0.xx}, ...]

    topic_lda_id INTEGER,
    topic_lda_prob REAL,
    topic_bert_id INTEGER,
    topic_bert_prob REAL,

    svo_triples TEXT,             -- JSON: [{"subject": "...", "verb": "...", "object": "..."}, ...]
    summary TEXT,

    -- 索引欄位 (參考路徑，實際儲存在檔案)
    tfidf_vector_path TEXT,       -- e.g., "indexes/tfidf/12345.pkl"
    bert_embedding_path TEXT,     -- e.g., "indexes/bert/12345.npy"

    -- Metadata
    crawled_at TIMESTAMP,
    preprocessed_at TIMESTAMP,
    indexed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- 索引
    INDEX idx_publish_date (publish_date),
    INDEX idx_category (category),
    INDEX idx_source (source),
    INDEX idx_topic_lda (topic_lda_id),
    INDEX idx_topic_bert (topic_bert_id)
);

-- 主題模型表 (LDA)
CREATE TABLE topics_lda (
    id INTEGER PRIMARY KEY,
    label TEXT,
    top_words TEXT,               -- JSON: [{"word": "...", "weight": 0.xx}, ...]
    document_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 主題模型表 (BERTopic)
CREATE TABLE topics_bert (
    id INTEGER PRIMARY KEY,
    label TEXT,
    top_words TEXT,
    document_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 查詢日誌表
CREATE TABLE query_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    model TEXT,                   -- "boolean", "tfidf", "bm25", "bert"
    num_results INTEGER,
    response_time REAL,           -- seconds
    user_ip TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_created_at (created_at)
);

-- 相關性回饋表 (用於 Rocchio)
CREATE TABLE relevance_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id INTEGER,
    doc_id INTEGER,
    is_relevant BOOLEAN,          -- 1 = relevant, 0 = non-relevant
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (query_id) REFERENCES query_log(id),
    FOREIGN KEY (doc_id) REFERENCES news(id)
);
```

---

## 5. 資料預處理流程 (Data Preprocessing Pipeline)

### 5.1 預處理流程圖

```
原始新聞 (Raw News)
        │
        ▼
┌───────────────────────┐
│  1. 文本清洗           │
│  - 移除 HTML 標籤      │
│  - 統一編碼 (UTF-8)    │
│  - 去除特殊符號        │
└───────┬───────────────┘
        │
        ▼
┌───────────────────────┐
│  2. 斷詞 (Jieba/CKIP) │
│  - 標題斷詞            │
│  - 內容斷詞            │
│  - 去除停用詞 (可選)   │
└───────┬───────────────┘
        │
        ▼
┌───────────────────────┐
│  3. NER 實體識別       │
│  - CKIP Transformers  │
│  - 提取人名/地名/機構  │
└───────┬───────────────┘
        │
        ▼
┌───────────────────────┐
│  4. 關鍵字提取         │
│  - TextRank           │
│  - YAKE               │
│  - KeyBERT            │
│  - RAKE               │
│  - Ensemble (融合)    │
└───────┬───────────────┘
        │
        ▼
┌───────────────────────┐
│  5. 主題建模           │
│  - LDA (Gensim)       │
│  - BERTopic           │
└───────┬───────────────┘
        │
        ▼
┌───────────────────────┐
│  6. 句法分析 (可選)    │
│  - SuPar Dependency   │
│  - SVO 三元組提取      │
└───────┬───────────────┘
        │
        ▼
┌───────────────────────┐
│  7. 摘要生成           │
│  - Lead-3 Baseline    │
│  - Key Sentence       │
└───────┬───────────────┘
        │
        ▼
┌───────────────────────┐
│  8. 索引建立           │
│  - Inverted Index     │
│  - Positional Index   │
│  - TF-IDF Vectors     │
│  - BERT Embeddings    │
└───────┬───────────────┘
        │
        ▼
   儲存至資料庫 (SQLite)
```

---

### 5.2 預處理腳本 (Preprocessing Script)

```python
# scripts/preprocess_news.py
import json
import logging
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

from src.ir.text.tokenizer import ChineseTokenizer
from src.ir.ner.ckip_ner import CKIPEntityRecognizer
from src.ir.keyextract.ensemble import KeywordEnsemble
from src.ir.topic.lda import LDATopicModel
from src.ir.topic.bertopic_model import BERTopicModel
from src.ir.syntax.parser import SVOExtractor
from src.ir.summarize.static import LeadKSummarizer, KeySentenceSummarizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsPreprocessor:
    def __init__(self):
        logger.info("Initializing preprocessing modules...")

        self.tokenizer = ChineseTokenizer(engine='jieba')
        self.ner = CKIPEntityRecognizer()
        self.keyword_extractor = KeywordEnsemble()
        self.lda = LDATopicModel(num_topics=10)
        self.bertopic = BERTopicModel()
        self.svo_extractor = SVOExtractor(tokenizer_engine='jieba')
        self.summarizer = KeySentenceSummarizer(num_sentences=3)

        logger.info("All modules loaded successfully")

    def preprocess_single(self, article: Dict) -> Dict:
        """Preprocess a single news article."""
        # 1. Tokenization
        tokens_title = self.tokenizer.tokenize(article['title'])
        tokens_content = self.tokenizer.tokenize(article['content'])

        # 2. NER
        entities = self.ner.recognize(article['content'])

        # 3. Keyword Extraction
        keywords = self.keyword_extractor.extract(
            article['content'],
            top_k=10,
            use_ner_boost=True
        )

        # 4. Topic Modeling (will be fitted on entire corpus first)
        # topic_lda = self.lda.transform(tokens_content)
        # topic_bert = self.bertopic.transform(article['content'])

        # 5. SVO Extraction (optional, expensive)
        # svo_triples = self.svo_extractor.extract(article['content'][:500])

        # 6. Summarization
        summary = self.summarizer.summarize(article['content'])

        # Update article
        preprocessed = article.copy()
        preprocessed.update({
            'tokens_title': tokens_title,
            'tokens_content': tokens_content,
            'tokens_count': len(tokens_content),
            'entities': [e.to_dict() for e in entities],
            'keywords': [{'word': w, 'score': s} for w, s in keywords],
            'summary': summary,
            # 'svo_triples': [t.to_dict() for t in svo_triples],
        })

        return preprocessed

    def preprocess_batch(self, articles: List[Dict], batch_size: int = 32) -> List[Dict]:
        """Preprocess a batch of articles."""
        logger.info(f"Preprocessing {len(articles)} articles...")

        preprocessed_articles = []
        for article in tqdm(articles, desc="Preprocessing"):
            try:
                preprocessed = self.preprocess_single(article)
                preprocessed_articles.append(preprocessed)
            except Exception as e:
                logger.error(f"Error preprocessing {article.get('url', 'unknown')}: {e}")

        return preprocessed_articles

    def fit_topic_models(self, articles: List[Dict]):
        """Fit topic models on entire corpus."""
        logger.info("Fitting topic models on corpus...")

        # Prepare data
        token_docs = [a['tokens_content'] for a in articles]
        text_docs = [a['content'] for a in articles]

        # Fit LDA
        logger.info("Fitting LDA...")
        self.lda.fit(token_docs)

        # Fit BERTopic
        logger.info("Fitting BERTopic...")
        self.bertopic.fit(text_docs)

        logger.info("Topic models fitted successfully")

    def assign_topics(self, articles: List[Dict]) -> List[Dict]:
        """Assign topics to articles."""
        logger.info("Assigning topics to articles...")

        for article in tqdm(articles, desc="Assigning topics"):
            # LDA
            topic_dist = self.lda.transform(article['tokens_content'])
            if topic_dist:
                topic_id, prob = max(topic_dist, key=lambda x: x[1])
                article['topic_lda_id'] = topic_id
                article['topic_lda_prob'] = prob

            # BERTopic
            topic_id, prob = self.bertopic.transform(article['content'])
            article['topic_bert_id'] = topic_id
            article['topic_bert_prob'] = prob

        return articles

def main():
    # Load raw data
    input_path = Path("data/raw/cna_news_2024.jsonl")
    output_path = Path("data/preprocessed/cna_news_2024_preprocessed.jsonl")

    logger.info(f"Loading data from {input_path}...")
    articles = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            articles.append(json.loads(line))

    logger.info(f"Loaded {len(articles)} articles")

    # Initialize preprocessor
    preprocessor = NewsPreprocessor()

    # Step 1: Basic preprocessing
    articles = preprocessor.preprocess_batch(articles)

    # Step 2: Fit topic models
    preprocessor.fit_topic_models(articles)

    # Step 3: Assign topics
    articles = preprocessor.assign_topics(articles)

    # Save preprocessed data
    logger.info(f"Saving preprocessed data to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for article in articles:
            f.write(json.dumps(article, ensure_ascii=False) + '\n')

    logger.info("Preprocessing complete!")

if __name__ == '__main__':
    main()
```

---

## 6. 資料品質控制 (Data Quality Control)

### 6.1 品質檢查項目

| 檢查項目 | 標準 | 處理方式 |
|---------|------|---------|
| **缺失值** | title, content 不可為空 | 移除缺失文章 |
| **重複文章** | 根據 URL 或 content 去重 | 保留最早爬取的版本 |
| **文章長度** | content 長度 > 100 字 | 移除過短文章 |
| **編碼錯誤** | 必須為有效 UTF-8 | 修復或移除 |
| **HTML 殘留** | 不應包含 `<tag>` | 清除 HTML 標籤 |
| **日期格式** | ISO 8601 格式 | 統一格式化 |
| **分類有效性** | 在預定義分類列表中 | 修正或標記為 "其他" |

### 6.2 資料清洗腳本

```python
# scripts/clean_data.py
import re
import json
from bs4 import BeautifulSoup
from datetime import datetime
from collections import Counter
import logging

logger = logging.getLogger(__name__)

VALID_CATEGORIES = {
    '政治', '財經', '科技', '社會', '國際', '生活', '文化', '運動', '其他'
}

def clean_html(text: str) -> str:
    """Remove HTML tags from text."""
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text(separator=' ', strip=True)

def normalize_date(date_str: str) -> str:
    """Normalize date to YYYY-MM-DD format."""
    try:
        # Try multiple formats
        for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%Y年%m月%d日']:
            try:
                dt = datetime.strptime(date_str[:10], fmt)
                return dt.strftime('%Y-%m-%d')
            except:
                continue
        return None
    except:
        return None

def is_valid_article(article: dict) -> bool:
    """Check if article meets quality standards."""
    # Check required fields
    if not article.get('title') or not article.get('content'):
        return False

    # Check content length
    if len(article['content']) < 100:
        return False

    # Check encoding
    try:
        article['content'].encode('utf-8')
    except:
        return False

    return True

def clean_article(article: dict) -> dict:
    """Clean a single article."""
    # Clean HTML
    article['title'] = clean_html(article['title'])
    article['content'] = clean_html(article['content'])

    # Normalize date
    if article.get('publish_date'):
        article['publish_date'] = normalize_date(article['publish_date'])

    # Normalize category
    if article.get('category') not in VALID_CATEGORIES:
        article['category'] = '其他'

    # Remove extra whitespace
    article['title'] = re.sub(r'\s+', ' ', article['title']).strip()
    article['content'] = re.sub(r'\s+', ' ', article['content']).strip()

    return article

def deduplicate(articles: list) -> list:
    """Remove duplicate articles."""
    seen_urls = set()
    seen_contents = set()
    unique_articles = []

    for article in articles:
        # Check URL
        if article.get('url') in seen_urls:
            continue

        # Check content (first 200 chars as fingerprint)
        content_fingerprint = article['content'][:200]
        if content_fingerprint in seen_contents:
            continue

        seen_urls.add(article.get('url'))
        seen_contents.add(content_fingerprint)
        unique_articles.append(article)

    logger.info(f"Removed {len(articles) - len(unique_articles)} duplicates")
    return unique_articles

def main():
    input_path = "data/raw/cna_news_raw.jsonl"
    output_path = "data/cleaned/cna_news_cleaned.jsonl"

    # Load articles
    articles = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            articles.append(json.loads(line))

    logger.info(f"Loaded {len(articles)} articles")

    # Clean articles
    cleaned_articles = []
    for article in articles:
        if is_valid_article(article):
            cleaned_article = clean_article(article)
            cleaned_articles.append(cleaned_article)

    logger.info(f"Valid articles: {len(cleaned_articles)}")

    # Deduplicate
    cleaned_articles = deduplicate(cleaned_articles)

    # Save
    with open(output_path, 'w', encoding='utf-8') as f:
        for article in cleaned_articles:
            f.write(json.dumps(article, ensure_ascii=False) + '\n')

    logger.info(f"Saved {len(cleaned_articles)} cleaned articles")

if __name__ == '__main__':
    main()
```

---

## 7. 資料集統計分析 (Dataset Statistics)

### 7.1 基礎統計

```python
# scripts/analyze_dataset.py
import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path

def analyze_dataset(data_path: str):
    """Generate dataset statistics."""
    # Load data
    articles = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            articles.append(json.loads(line))

    df = pd.DataFrame(articles)

    # Basic statistics
    stats = {
        'Total Articles': len(df),
        'Unique Sources': df['source'].nunique(),
        'Date Range': f"{df['publish_date'].min()} ~ {df['publish_date'].max()}",
        'Avg Content Length': df['content'].str.len().mean(),
        'Median Content Length': df['content'].str.len().median(),
        'Avg Tokens': df['tokens_count'].mean() if 'tokens_count' in df else 'N/A',
    }

    print("\n=== Dataset Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")

    # Category distribution
    print("\n=== Category Distribution ===")
    category_counts = df['category'].value_counts()
    print(category_counts)

    # Source distribution
    print("\n=== Source Distribution ===")
    source_counts = df['source'].value_counts()
    print(source_counts)

    # Date distribution
    df['publish_date'] = pd.to_datetime(df['publish_date'])
    df['year_month'] = df['publish_date'].dt.to_period('M')
    monthly_counts = df['year_month'].value_counts().sort_index()

    print("\n=== Monthly Distribution (Top 10) ===")
    print(monthly_counts.head(10))

    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Category pie chart
    category_counts.plot(kind='pie', ax=axes[0, 0], autopct='%1.1f%%')
    axes[0, 0].set_title('Category Distribution')
    axes[0, 0].set_ylabel('')

    # Source bar chart
    source_counts.plot(kind='bar', ax=axes[0, 1])
    axes[0, 1].set_title('Source Distribution')
    axes[0, 1].set_xlabel('Source')
    axes[0, 1].set_ylabel('Count')

    # Content length histogram
    df['content'].str.len().hist(bins=50, ax=axes[1, 0])
    axes[1, 0].set_title('Content Length Distribution')
    axes[1, 0].set_xlabel('Length (characters)')
    axes[1, 0].set_ylabel('Frequency')

    # Monthly timeline
    monthly_counts.plot(kind='line', ax=axes[1, 1])
    axes[1, 1].set_title('Monthly Article Count')
    axes[1, 1].set_xlabel('Month')
    axes[1, 1].set_ylabel('Count')

    plt.tight_layout()
    plt.savefig('data/analysis/dataset_statistics.png', dpi=300)
    print("\nVisualization saved to data/analysis/dataset_statistics.png")

if __name__ == '__main__':
    analyze_dataset('data/preprocessed/cna_news_2024_preprocessed.jsonl')
```

### 7.2 預期統計結果

| 統計項目 | 預期值 |
|---------|--------|
| **總文章數** | 30,000 - 50,000 |
| **資料來源** | 3-5 個 (CNA, PTS, TechNews, ...) |
| **時間範圍** | 2022-01-01 ~ 2024-12-31 (3 年) |
| **分類分布** | 科技 (25%), 政治 (20%), 財經 (15%), 社會 (15%), 其他 (25%) |
| **平均文章長度** | 800-1200 字 |
| **中位數文章長度** | 600-900 字 |
| **平均詞數 (tokens)** | 400-600 |

---

## 8. 評估資料集 (Evaluation Dataset)

### 8.1 測試查詢集 (Test Query Set)

為了評估檢索系統效能，需要建立一組**測試查詢 (queries)** 與對應的**相關文檔 (qrels)**。

#### 8.1.1 查詢來源

```python
# 50 個測試查詢範例
TEST_QUERIES = [
    # 科技類 (15 queries)
    {"query_id": "Q01", "text": "人工智慧發展趨勢", "category": "科技"},
    {"query_id": "Q02", "text": "ChatGPT 語言模型", "category": "科技"},
    {"query_id": "Q03", "text": "台積電 3奈米製程", "category": "科技"},
    {"query_id": "Q04", "text": "電動車市場競爭", "category": "科技"},
    {"query_id": "Q05", "text": "5G 網路建設", "category": "科技"},

    # 政治類 (10 queries)
    {"query_id": "Q06", "text": "總統大選民調", "category": "政治"},
    {"query_id": "Q07", "text": "兩岸關係發展", "category": "政治"},

    # 財經類 (10 queries)
    {"query_id": "Q08", "text": "台股指數波動", "category": "財經"},
    {"query_id": "Q09", "text": "央行利率政策", "category": "財經"},

    # 社會類 (10 queries)
    {"query_id": "Q10", "text": "少子化問題", "category": "社會"},

    # 混合查詢 (5 queries)
    {"query_id": "Q11", "text": "疫情對經濟影響", "category": "混合"},
    # ... 共 50 queries
]
```

#### 8.1.2 相關性標註 (Relevance Judgments)

```python
# QRELS 格式: query_id, doc_id, relevance_score (0-2)
# 0 = 不相關, 1 = 部分相關, 2 = 高度相關

QRELS = [
    {"query_id": "Q01", "doc_id": 1234, "relevance": 2},
    {"query_id": "Q01", "doc_id": 1567, "relevance": 2},
    {"query_id": "Q01", "doc_id": 2890, "relevance": 1},
    {"query_id": "Q01", "doc_id": 3456, "relevance": 1},
    {"query_id": "Q01", "doc_id": 4789, "relevance": 0},
    # ... 每個 query 標註 20-50 篇文檔
]
```

#### 8.1.3 標註流程

**方式 1: 半自動標註**
1. 對每個查詢，使用 BM25 檢索 top-100 文檔
2. 人工審閱前 50 篇，標註相關性 (0/1/2)
3. 若相關文檔 < 10 篇，使用不同檢索模型擴充候選集

**方式 2: Pooling 方法**
1. 對每個查詢，使用 4 種模型 (Boolean, TF-IDF, BM25, BERT) 各取 top-20
2. 合併去重，形成候選池 (pool)
3. 人工標註候選池中所有文檔

**標註指南**:
- **2 (高度相關)**: 文檔核心內容直接回答查詢
- **1 (部分相關)**: 文檔部分內容與查詢相關
- **0 (不相關)**: 文檔與查詢無關或僅提及關鍵字

---

### 8.2 QRELS 檔案格式

```txt
# data/evaluation/qrels.txt (TREC 格式)
# query_id  0  doc_id  relevance

Q01 0 1234 2
Q01 0 1567 2
Q01 0 2890 1
Q01 0 3456 1
Q01 0 4789 0
Q02 0 2345 2
Q02 0 3456 1
...
```

---

## 9. 資料儲存方案 (Data Storage Solution)

### 9.1 分層儲存策略

```
data/
├── raw/                          # 原始爬取資料 (保留)
│   ├── cna_news_2022.jsonl       (10 MB)
│   ├── cna_news_2023.jsonl       (10 MB)
│   └── cna_news_2024.jsonl       (10 MB)
│
├── cleaned/                      # 清洗後資料
│   └── news_cleaned.jsonl        (25 MB)
│
├── preprocessed/                 # 預處理資料
│   └── news_preprocessed.jsonl   (50 MB, 包含 NLP 欄位)
│
├── database/                     # SQLite 資料庫
│   └── cnirs.db                  (100 MB, 結構化 metadata)
│
├── indexes/                      # 索引檔案
│   ├── inverted_index.pkl        (20 MB)
│   ├── positional_index.pkl      (50 MB)
│   ├── tfidf/                    # TF-IDF 向量 (稀疏矩陣)
│   │   └── tfidf_matrix.pkl      (30 MB)
│   └── bert/                     # BERT 嵌入 (密集向量)
│       ├── embeddings.npy        (500 MB for 30k docs * 768 dim * 4 bytes)
│       └── faiss_index.bin       (500 MB, FAISS 索引)
│
├── models/                       # 訓練的模型
│   ├── lda_model.pkl             (10 MB)
│   ├── bertopic_model.pkl        (50 MB)
│   └── word2vec.bin              (100 MB, 若使用)
│
├── evaluation/                   # 評估資料
│   ├── queries.json              (10 KB)
│   └── qrels.txt                 (100 KB)
│
└── analysis/                     # 統計分析結果
    ├── dataset_statistics.png
    └── statistics.json

Total: ~1.5 GB (不含 Hugging Face 模型快取)
```

### 9.2 大檔案處理

#### BERT Embeddings (使用 HDF5)

```python
import h5py
import numpy as np

def save_embeddings_hdf5(embeddings: np.ndarray, filepath: str):
    """Save BERT embeddings to HDF5 format."""
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('embeddings', data=embeddings,
                         compression='gzip', compression_opts=9)

def load_embeddings_hdf5(filepath: str) -> np.ndarray:
    """Load BERT embeddings from HDF5 format."""
    with h5py.File(filepath, 'r') as f:
        return f['embeddings'][:]

# Usage
embeddings = model.encode(documents)  # (30000, 768)
save_embeddings_hdf5(embeddings, 'data/indexes/bert/embeddings.h5')
```

#### FAISS 索引 (加速相似度搜尋)

```python
import faiss
import numpy as np

def build_faiss_index(embeddings: np.ndarray, use_gpu: bool = False) -> faiss.Index:
    """Build FAISS index for fast similarity search."""
    d = embeddings.shape[1]  # Dimension (768)

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    # Create index
    if use_gpu:
        index = faiss.IndexFlatIP(d)  # Inner Product = Cosine Similarity
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    else:
        index = faiss.IndexFlatIP(d)

    # Add vectors
    index.add(embeddings)

    return index

def search_faiss(index: faiss.Index, query_embedding: np.ndarray, k: int = 10):
    """Search top-k similar documents."""
    faiss.normalize_L2(query_embedding.reshape(1, -1))
    distances, indices = index.search(query_embedding.reshape(1, -1), k)
    return indices[0], distances[0]

# Usage
index = build_faiss_index(embeddings)
faiss.write_index(index, 'data/indexes/bert/faiss_index.bin')
```

---

## 10. 資料更新策略 (Data Update Strategy)

### 10.1 增量更新 (Incremental Update)

```python
# scripts/incremental_update.py
from datetime import datetime, timedelta

def incremental_crawl(last_update_date: str):
    """Crawl news since last update."""
    start_date = datetime.fromisoformat(last_update_date)
    end_date = datetime.now()

    # Crawl new articles
    new_articles = crawl_news_by_daterange(start_date, end_date)

    # Preprocess
    new_articles = preprocess(new_articles)

    # Update indexes
    update_inverted_index(new_articles)
    update_tfidf_vectors(new_articles)
    update_bert_embeddings(new_articles)

    # Update database
    insert_to_database(new_articles)

    # Update topic models (optional, retrain or update)
    update_topic_models(new_articles)

def update_inverted_index(new_articles: list):
    """Add new articles to existing inverted index."""
    index = load_inverted_index('data/indexes/inverted_index.pkl')

    for article in new_articles:
        doc_id = article['id']
        tokens = article['tokens_content']
        index.add_document(doc_id, tokens)

    save_inverted_index(index, 'data/indexes/inverted_index.pkl')
```

### 10.2 定期重建 (Periodic Rebuild)

```bash
# cron job: 每週日凌晨 2:00 執行增量更新
0 2 * * 0 /usr/bin/python3 /app/scripts/incremental_update.py

# 每月 1 號凌晨 3:00 重建主題模型
0 3 1 * * /usr/bin/python3 /app/scripts/rebuild_topics.py

# 每季度重建所有索引 (季度末)
0 4 1 1,4,7,10 * /usr/bin/python3 /app/scripts/rebuild_all_indexes.py
```

---

## 總結 (Summary)

本資料集規劃文件詳細說明了 **CNIRS** 的資料收集、處理、儲存與更新策略：

1. **資料來源**: CNA、PTS、TechNews (30,000-50,000 篇新聞)
2. **資料收集**: Scrapy 爬蟲 + RSS Feed 訂閱
3. **資料結構**: JSON Lines (原始) → SQLite (結構化) → Pickle/HDF5 (索引)
4. **預處理流程**: 斷詞 → NER → 關鍵字 → 主題 → 索引
5. **品質控制**: 去重、清洗、驗證
6. **評估資料集**: 50 測試查詢 + QRELS 相關性標註
7. **儲存方案**: 分層儲存 (~1.5 GB)
8. **更新策略**: 增量更新 (每週) + 定期重建 (每月/季)

此資料集規劃確保系統擁有**高品質、多樣化、可評估**的中文新聞資料，支援完整的 IR 研究與開發。

---

**文件版本**: v1.0
**最後更新**: 2025-11-13
**作者**: CNIRS 開發團隊
**授權**: Educational Use
