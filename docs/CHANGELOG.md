# 變更紀錄 *Changelog*

本文件記錄專案開發過程中的所有重要變更。

格式基於 [Keep a Changelog](https://keepachangelog.com/zh-TW/1.0.0/)，
版本號遵循 [語意化版本](https://semver.org/lang/zh-TW/)。

---

## [Unreleased]
### [2025-11-18] 新聞爬蟲系統大規模擴充與測試框架建立

#### ✅ 新增 - Phase 1: 擴充媒體來源

**新增 4 個主要新聞媒體爬蟲**:
1. **TVBS 新聞爬蟲** (`tvbs_spider.py`) ⚠️
   - 12 個新聞分類 (政治、財經、娛樂、社會等)
   - Playwright 動態網頁爬取
   - 狀態: 需調整 (timeout 問題)

2. **中時新聞網爬蟲** (`chinatimes_spider.py`) ⭐⭐⭐⭐⭐
   - 13 個新聞分類
   - 傳統 HTTP 爬蟲 (無需 Playwright)
   - 測試結果: **100% 成功率**，19 篇文章，47 秒
   - 評價: 最穩定的爬蟲

3. **東森新聞雲爬蟲** (`ettoday_spider.py`) ⭐⭐⭐⭐
   - 16 個新聞分類 (含寵物、電競等特色分類)
   - Playwright 動態網頁爬取
   - 測試結果: 發現 67 篇文章，成功爬取 3 篇

4. **風傳媒爬蟲** (`storm_spider.py`) 🔄
   - 10 個新聞分類 (含深度報導、長文)
   - Playwright 動態網頁爬取
   - 狀態: 測試中

#### ✅ 新增 - Phase 2: 企業級測試框架

**完整的自動化測試系統** (pytest + 健康檢查):

1. **pytest 單元測試框架** (`tests/crawlers/`)
   - `conftest.py`: 共享 fixtures 和測試配置
     * 爬蟲註冊表 (9 個媒體)
     * 文章結構驗證模板
     * 日期範圍生成器
     * Reactor 配置處理
   - `test_crawlers_unit.py`: 20+ 單元測試
     * TestCrawlerInitialization: 初始化測試
     * TestCrawlerUtilities: 工具方法測試 (ID 生成、文字清理、日期解析)
     * TestArticleValidation: 文章數據驗證
     * TestCrawlerConfiguration: 配置檢查 (robots.txt, Playwright 設定)
   - 參數化測試: 自動測試所有爬蟲

2. **健康檢查系統** (`scripts/crawlers/health_check.py`)
   - 自動化監控所有爬蟲狀態
   - 生成 HTML 視覺化報告 (健康百分比、彩色卡片、詳細表格)
   - 生成 JSON 結構化報告
   - 支援快速檢查模式 (`--quick`)
   - 可指定特定爬蟲檢查
   - 180 秒超時保護
   - CLI: `python scripts/crawlers/health_check.py --html-report`

3. **單一爬蟲測試工具** (`test_single_crawler.py`)
   - 獨立測試個別爬蟲
   - 自動處理 Playwright reactor 安裝
   - 詳細統計報告 (成功率、文章數、執行時間)

#### ✅ 新增 - 統一爬蟲管理更新

**UDN 聯合報爬蟲**: 完善實作，支援 Playwright、分類過濾、日期範圍查詢
**Apple Daily 蘋果日報爬蟲**: 全新實作，支援 Next Apple 網站爬取
**Twisted Reactor 配置**: 修復 Playwright 的 asyncio reactor 衝突問題
**統一爬蟲管理**: 更新 `run_crawlers.py` 註冊所有爬蟲（**9 個媒體來源**）

#### 🔧 修復
- 修復 LTN 和 UDN 爬蟲的 reactor 安裝問題
- 改善錯誤處理和日誌記錄
- 優化 Playwright 配置參數
- 修復 TVBS 爬蟲 PageMethod 使用錯誤 (多次迭代)
- 統一輸出格式 (JSONL 標準)

#### 📝 文檔
- 更新 `scripts/crawlers/README.md`: 支援媒體列表、技術架構、反爬蟲機制
- 新增 `tests/crawlers/README.md`: 完整測試文檔
  * 快速開始指南
  * pytest 使用範例
  * 健康檢查系統說明
  * CI/CD 整合範例
  * 常見問題 FAQ
- 添加安裝與使用說明
- 記錄已知問題與解決方案

#### 📊 測試結果總結

**測試覆蓋**: 4 個新爬蟲
- ✅ **中時新聞網**: 19 篇文章，100% 成功率，47 秒 (⭐ 最佳表現)
- ✅ **東森新聞雲**: 67 篇發現，3 篇爬取，90 秒
- ❌ **TVBS 新聞**: Playwright 超時問題 (待修復)
- 🔄 **風傳媒**: 測試進行中

**整體統計**:
- **支援媒體數**: 9 個台灣主要新聞媒體
- **爬蟲類型**: 3 個傳統爬蟲 + 6 個 Playwright 爬蟲
- **測試數量**: 20+ 單元測試
- **新增程式碼**: ~3,000+ 行
- **文檔字數**: ~10,000+ 字

#### 🎯 技術亮點

**反爬蟲機制**:
- User-Agent 輪換
- 視窗尺寸隨機化
- 人性化延遲 (Gaussian 分佈)
- 瀏覽器指紋隨機化
- Stealth 模式

**錯誤處理**:
- 自動重試機制 (3 次)
- 超時設置 (60 秒)
- 詳細日誌記錄
- 統計數據追蹤

**輸出格式**:
- 統一 JSONL 格式
- 包含 article_id, title, content, url, source, published_date, author, category, tags, image_url, crawled_at

**檔案變更**:
- **新增 (10+ 檔案)**:
  * `scripts/crawlers/tvbs_spider.py`
  * `scripts/crawlers/chinatimes_spider.py`
  * `scripts/crawlers/ettoday_spider.py`
  * `scripts/crawlers/storm_spider.py`
  * `scripts/crawlers/test_single_crawler.py`
  * `scripts/crawlers/health_check.py`
  * `tests/crawlers/conftest.py`
  * `tests/crawlers/test_crawlers_unit.py`
  * `tests/crawlers/README.md`
  * `apple_daily_spider.py`, `test_udn.py`

- **修改**:
  * `scripts/crawlers/run_crawlers.py` (註冊 4 個新爬蟲)
  * `scripts/crawlers/README.md` (更新媒體列表與技術說明)
  * `udn_spider.py`, `ltn_spider.py`

**目前支援媒體** (9 個):
1. CNA（中央社）
2. LTN（自由時報）
3. PTS（公視）
4. UDN（聯合報）
5. Apple Daily（蘋果日報）
6. **TVBS 新聞** ⭐ 新增
7. **中時新聞網** ⭐ 新增
8. **東森新聞雲** ⭐ 新增
9. **風傳媒** ⭐ 新增

#### ⚠️ 已知問題

1. **TVBS 爬蟲超時**: Playwright `wait_for_selector` 持續超時，需要:
   - 改用 `networkidle` 等待策略，或
   - 轉為傳統爬蟲（如果是靜態 HTML），或
   - 暫時標記為 optional

2. **Playwright 效能**:
   - Playwright 爬蟲速度較慢 (~25-30 秒/篇)
   - 傳統爬蟲快 10 倍 (~2.5 秒/篇)
   - 建議: 優先使用傳統爬蟲，Playwright 僅用於必要的動態網站

3. **Twisted Reactor 衝突**:
   - 已在測試腳本中妥善處理
   - 必須在導入 spider 前安裝 asyncioreactor

---


### 計劃新增 *Planned*
- [ ] Docker 容器化部署
- [ ] 搜尋歷史記錄
- [ ] 文檔詳情彈窗
- [ ] 進階篩選功能

---

## [1.2.0] - 2025-11-17

### 新增 *Added*

#### 🌐 Phase 5: Web UI 開發 - 完成

**Phase 5 達成率**: 100% ✅
**Web 應用**: Flask + HTML/CSS/JavaScript
**API Endpoints**: 7 個
**程式碼量**: ~1,309 lines

##### 1. Flask 後端應用
- ✅ `app_simple.py` (256 lines) - 輕量級 Flask 應用
  - RESTful API 設計
  - 統一檢索系統整合
  - CORS 跨域支援
  - JSON API 回應格式
  - 延遲載入優化
  - 完整錯誤處理

**API Endpoints**:
- `GET  /` - 主搜尋頁面
- `GET  /compare` - 模型對比頁面
- `GET  /about` - 關於頁面
- `POST /api/search` - 搜尋 API
- `POST /api/compare` - 模型對比 API
- `GET  /api/document/<id>` - 文檔詳情
- `GET  /api/stats` - 系統統計

##### 2. 前端 HTML 模板
- ✅ `templates/search.html` (77 lines) - 主搜尋介面
  - 模型選擇器 (Boolean, TF-IDF, BM25, BERT)
  - Top-K 結果數量控制
  - Boolean 運算子選擇
  - 系統統計面板
  - 結果展示與高亮

- ✅ `templates/compare.html` (58 lines) - 模型對比頁面
  - 多模型複選
  - 並排結果展示
  - 效能指標比較

- ✅ `templates/about.html` (138 lines) - 系統說明頁面
  - 專案簡介
  - 模型詳細說明
  - 評估結果展示
  - 技術架構說明

##### 3. CSS 樣式設計
- ✅ `static/css/style.css` (445 lines)
  - 現代化扁平設計
  - 響應式佈局 (手機/平板/桌面)
  - 漸層色彩系統 (藍色主題)
  - 卡片式 UI 組件
  - 平滑動畫效果
  - 高亮顯示查詢詞

**設計特色**:
- Primary Color: #2563eb
- Card-based Layout
- Grid/Flexbox 響應式
- Smooth Transitions
- Loading Animations

##### 4. JavaScript 互動功能
- ✅ `static/js/search.js` (158 lines)
  - 載入系統統計
  - 搜尋功能實作
  - 結果展示與高亮
  - Enter 鍵快速搜尋
  - 動態選項顯示
  - 錯誤處理

- ✅ `static/js/compare.js` (177 lines)
  - 多模型並行對比
  - 複選框模型選擇
  - 並排結果展示
  - 效能比較表格
  - 視覺化對比

**互動特性**:
- Async/Await API 呼叫
- 動態 DOM 操作
- 查詢詞高亮 (<mark>)
- 平滑滾動
- 載入狀態管理

##### 5. 功能特性
**搜尋功能**:
- 4 種檢索模型切換
- Top-K 可調整 (5-100)
- Boolean 運算子支援
- 查詢詞高亮
- 響應時間顯示
- 元資料展示

**模型對比**:
- 並行多模型搜尋
- 並排結果顯示
- 效能指標比較 (結果數、時間、分數)
- 視覺化比較表格

**系統資訊**:
- 文檔總數: 121
- 詞彙數: 10,248
- 可用模型: 4 種
- 索引載入狀態

##### 6. 測試結果
- ✅ Web 應用成功啟動
- ✅ 搜尋功能測試通過
- ✅ 模型對比功能正常
- ✅ API Endpoints 可用
- ✅ 響應式設計驗證

**啟動命令**:
```bash
python app_simple.py --host 0.0.0.0 --port 5000
```

**訪問 URL**:
- Main: http://localhost:5000/
- Compare: http://localhost:5000/compare
- About: http://localhost:5000/about

### 文檔 *Documentation*
- ✅ `data/stats/phase5_summary.txt` - Phase 5 完成總結

---

## [1.1.0] - 2025-11-17

### 新增 *Added*

#### 🎯 Phase 4: 檢索系統整合與評估 - 完成

**Phase 4 達成率**: 100% ✅
**實驗配置**: 15 queries × 4 models = 60 runs
**評估數據**: 347 QRELS judgments
**最佳模型**: BM25 (MAP=0.3708, nDCG=0.4064)

##### 1. 統一檢索 API
- ✅ `scripts/unified_retrieval.py` (577 lines)
  - `UnifiedRetrieval` class - 整合所有檢索模型
  - `SearchResult` dataclass - 標準化結果格式
  - Boolean retrieval (AND/OR operators)
  - TF-IDF ranking (Vector Space Model)
  - BM25 ranking (Probabilistic model)
  - BERT semantic search (multilingual)
  - 統一 `search()` 介面
  - 命令列工具支援

##### 2. 端到端評估系統
- ✅ `scripts/evaluate_retrieval.py` (760 lines)
  - `RetrievalEvaluator` class
  - **Mean Average Precision (MAP)** implementation
  - **Normalized Discounted Cumulative Gain (nDCG)**
  - **Precision@K** (K = 5, 10, 20)
  - **Recall@K** (K = 5, 10, 20)
  - **F1@K** metrics
  - TREC QRELS format support (4-column standard)
  - Batch evaluation for all models
  - Response time measurement
  - JSON + Text report generation

##### 3. 評估實驗結果
- ✅ `data/results/evaluation_results.json` - 完整評估數據
- ✅ `data/results/evaluation_report.txt` - 評估摘要表格
- ✅ `data/stats/phase4_summary.txt` - Phase 4 總結報告

**模型比較結果**:
```
Model        MAP     nDCG    P@5     P@10    P@20    R@10    F1@10   Time(s)
TFIDF     0.3581   0.4012  0.4400  0.3267  0.2267  0.3288  0.2817  0.0013
BM25      0.3708   0.4064  0.4400  0.3467  0.2400  0.3447  0.2982  0.0001
BERT      0.1586   0.2306  0.2133  0.2133  0.1667  0.1631  0.1590  0.7722
```

**關鍵發現**:
- BM25 在所有精確度指標均表現最佳
- TF-IDF 表現接近 BM25，速度僅略慢
- BERT 在小規模關鍵字查詢表現較弱（適合大規模語義查詢）
- BM25/TF-IDF 響應時間 < 2ms，BERT 約 770ms

### 修正 *Fixed*
- 🔧 QRELS 載入器支援 TREC 標準格式 (4-column)
- 🔧 評估腳本正確解析 whitespace-separated QRELS
- 🔧 nDCG 計算使用 log2(rank+1) discount factor
- 🔧 統一檢索 API 的文檔 ID 映射

---

## [1.0.0] - 2025-11-17

### 新增 *Added*

#### 🎉 Phase 3: 資料預處理與索引建構 - 完成

**Phase 3 達成率**: 100% ✅
**處理資料**: 121 篇 CNA 新聞 (2025-11-07 to 2025-11-13)
**總處理時間**: 35.46 秒
**總儲存空間**: ~5 MB

##### 1. NLP 預處理管線
- ✅ `scripts/preprocess_news.py` - 整合中文 NLP 處理
  - Jieba 中文分詞 (48,412 tokens)
  - CKIP NER 實體識別 (5,493 entities, 18 種類型)
  - TextRank 關鍵詞提取 (605 keywords)
  - Lead-k 自動摘要
  - 處理速度: 0.265 秒/篇, 100% 成功率

##### 2. 搜尋索引建構
- ✅ `scripts/build_indexes.py` - 4 種索引類型
  - Inverted Index (354 KB) - 10,248 terms, 26,307 postings
  - Positional Index (350 KB) - 39,133 positions
  - TF-IDF Vectors (535 KB) - L2 normalized
  - BM25 Index (185 KB) - k1=1.5, b=0.75
  - 建構速度: 1,103.9 docs/second

##### 3. BERT 語義向量
- ✅ `scripts/build_bert_embeddings.py` - 語義檢索向量
  - 模型: paraphrase-multilingual-MiniLM-L12-v2 (384 dim)
  - GPU 加速 (CUDA) - 39.3 docs/second
  - 輸出: bert_embeddings.npy (182 KB, 121 x 384)

##### 4. SQLite 資料庫
- ✅ `scripts/build_database.py` - 結構化儲存
  - cnirs.db (1.47 MB) - 121 rows
  - news 表 + FTS5 全文檢索
  - 包含 NLP 欄位 (tokens, entities, keywords, summary)

##### 5. 評估資料集
- ✅ `scripts/create_test_queries.py` - 測試查詢與 QRELS
  - 15 個測試查詢 (5 types: simple, entity, phrase, multi, topic)
  - 347 個相關性判斷 (64 highly relevant, 57 relevant)
  - TREC 標準格式

##### 完整資料結構
```
data/
├── preprocessed/cna_mvp_preprocessed.jsonl  (1.2 MB)
├── indexes/ (1.6 MB total)
│   ├── inverted_index.pkl, positional_index.pkl
│   ├── tfidf_vectors.pkl, bm25_index.pkl
│   └── bert_embeddings.npy
├── database/cnirs.db (1.47 MB)
├── evaluation/ (test_queries.txt, qrels.txt)
└── stats/ (8 個統計報告)
```

### 技術特色
- **中文處理**: Jieba + CKIP Transformers
- **多模型索引**: Boolean, TF-IDF, BM25, BERT 並行
- **GPU 加速**: BERT embeddings 3.08 秒完成
- **完整評估**: 15 queries + 347 qrels

---

## [0.9.0] - 2025-11-13

### 新增 *Added*

#### 📊 期末專案規劃 (Final Project Planning)

本版本完成期末專案的完整規劃與文檔撰寫，確立了**中文新聞智能檢索系統 (CNIRS)** 的開發方向。

##### 專案提案 (PROPOSAL.md)
- **文件**: `docs/project/PROPOSAL.md` (738 行)
- **專案名稱**: Chinese News Intelligent Retrieval System (CNIRS)
- **核心目標**: 整合傳統 IR 與現代 NLP 技術，建立可比較的中文新聞檢索系統
- **10 大功能模組**:
  1. F1: 傳統檢索 (Boolean, TF-IDF, BM25)
  2. F2: BERT 語意搜尋
  3. F3: NER 實體識別 (CKIP)
  4. F4: 主題建模 (LDA + BERTopic)
  5. F5: 關鍵字提取 (TextRank, YAKE, KeyBERT, RAKE)
  6. F6: 句法分析 (SuPar Dependency + SVO)
  7. F7: Rocchio 查詢擴展
  8. F8: 同義詞與語義擴展
  9. F9: 自動摘要
  10. F10: 文檔聚類
- **資料規模**: 30,000-50,000 篇中文新聞 (2022-2024)
- **資料來源**: 中央社 (CNA), 公視新聞 (PTS), 科技新報 (TechNews)
- **評估計畫**: MAP, nDCG@10, P@5, R@10
- **時程規劃**: 6 週開發時程
- **使用者介面**: CLI + Web UI (Flask)

##### 系統架構文件 (ARCHITECTURE.md)
- **文件**: `docs/project/ARCHITECTURE.md` (1400+ 行)
- **架構設計**: 四層架構
  - 使用者介面層 (User Interface Layer)
  - 應用邏輯層 (Application Logic Layer)
  - IR 核心層 (IR Core Layer)
  - 資料儲存層 (Data Storage Layer)
- **核心模組設計**:
  - 傳統 IR 模組 (M1-M7) 完整設計
  - 現代 NLP 模組 (Phase 1-5) 完整設計
  - 新增模組 (BM25, BERT Search, Query Expansion)
- **資料流設計**:
  - 索引建立流程 (Indexing Pipeline)
  - 查詢處理流程 (Query Processing Pipeline)
  - 分析流程 (Analysis Pipeline)
- **API 設計**:
  - RESTful API 規範 (5+ 端點)
  - CLI 命令列介面設計
  - Request/Response 格式定義
- **資料庫 Schema**: SQLite 完整設計 (news, topics, query_log, feedback)
- **模組相依性**: 完整依賴關係圖與說明
- **部署架構**: Docker 容器化 + Nginx + PostgreSQL
- **效能考量**:
  - 快取策略 (LRU, Query Cache, Embedding Cache)
  - 批次處理優化
  - 索引壓縮 (Variable Byte Encoding)
- **擴展性設計**:
  - 水平擴展 (分散式索引, Ray 分散式編碼)
  - 垂直擴展 (GPU 加速, 多程序)
- **安全性設計**: 輸入驗證, 資料加密, API 限流

##### 資料集規劃文件 (DATASET.md)
- **文件**: `docs/project/DATASET.md` (1200+ 行)
- **資料集規模**: 30,000-50,000 篇, 2022-2024, 繁體中文
- **主要資料來源**:
  - 中央社 (CNA): 20,000 篇
  - 公視新聞 (PTS): 10,000 篇
  - 科技新報 (TechNews): 5,000 篇
- **爬蟲策略**:
  - Scrapy 框架實作
  - 遵守 robots.txt
  - Download delay: 2 秒
  - RSS Feed 訂閱 (備用)
- **資料結構設計**:
  - 原始格式: JSON Lines (.jsonl)
  - 預處理格式: JSON with NLP fields
  - 資料庫: SQLite Schema 設計
- **預處理流程**: 8 步驟
  1. 文本清洗
  2. 斷詞 (Jieba/CKIP)
  3. NER 實體識別
  4. 關鍵字提取
  5. 主題建模
  6. 句法分析 (可選)
  7. 摘要生成
  8. 索引建立
- **品質控制**:
  - 去重 (URL + content fingerprint)
  - 清洗 (HTML, encoding, 長度)
  - 驗證 (完整性、格式)
- **評估資料集**:
  - 50 測試查詢 (科技 15, 政治 10, 財經 10, 社會 10, 混合 5)
  - QRELS 相關性標註 (0/1/2)
  - Pooling 方法建立候選池
- **儲存方案**: 分層儲存 (~1.5 GB)
  - 原始資料: JSONL
  - 結構化資料: SQLite
  - 索引: Pickle
  - BERT 嵌入: HDF5 + FAISS
- **更新策略**:
  - 增量更新 (每週)
  - 定期重建 (每月/季)

##### 開發任務清單 (TODO.md)
- **文件**: `docs/project/TODO.md` (1000+ 行)
- **專案進度總覽**: 8 個階段規劃
  - Phase 0: 專案規劃 ✅ (100%)
  - Phase 1: 傳統 IR 模組 (M1-M7) ✅ (100%)
  - Phase 2: 現代 NLP 模組 (Phase 1-5) ✅ (100%)
  - Phase 3: 資料收集與預處理 ⏳ (0%)
  - Phase 4: 新增檢索模組 ⏸️ (0%)
  - Phase 5: Web UI 開發 ⏸️ (0%)
  - Phase 6: 整合與測試 ⏸️ (0%)
  - Phase 7: 評估與優化 ⏸️ (0%)
  - Phase 8: 文檔與展示 ⏸️ (0%)
- **詳細任務分解**: 每個階段細分為多個任務與子任務
- **時程安排**: 22 週完整規劃
- **工具與腳本清單**: 30+ 腳本規劃
- **預期產出**: 每個任務明確定義產出物

#### 🧠 Phase 5: 句法分析 (Syntactic Parsing) - 完成

本階段實作中文句法分析功能，包括依存句法分析與 SVO 三元組提取。

##### 核心實作 (parser.py)
- **檔案**: `src/ir/syntax/parser.py` (680+ 行)
- **核心類別**:
  1. `DependencyEdge`: 依存邊資料結構
  2. `SVOTriple`: SVO 三元組資料結構
  3. `DependencyParser`: SuPar 依存句法分析器封裝
  4. `SVOExtractor`: SVO 三元組提取器
  5. `SyntaxAnalyzer`: 統一句法分析介面

##### DependencyParser (依存句法分析器)
- **技術**: SuPar (State-of-the-art Parser)
  - 模型: `biaffine-dep-zh` (中文依存句法分析)
  - 架構: Biaffine Attention
- **核心方法**:
  - `parse()`: 解析文本為依存邊 (O(n³) 複雜度)
  - `parse_batch()`: 批次解析
  - `get_dependency_tree()`: 獲取依存樹
  - `get_root_verb()`: 提取根動詞
- **依存關係類型**:
  - nsubj (主語), dobj (賓語), root (根)
  - nmod (名詞修飾), amod (形容詞修飾)
  - ccomp (補語從句), xcomp (開放式補語)
- **PyTorch 2.6+ 相容性修復**:
  - 問題: `weights_only=True` 預設值導致模型載入失敗
  - 解決: Monkey patch `torch.load` 使用 `weights_only=False`
  - 影響: 確保在 PyTorch 2.6+ 環境正常運作

##### SVOExtractor (SVO 三元組提取器)
- **核心功能**: 從依存句法樹提取 Subject-Verb-Object 三元組
- **提取策略**:
  1. 識別根動詞 (relation='root')
  2. 查找主語 (nsubj, nsubjpass)
  3. 查找賓語 (dobj, attr, ccomp)
  4. 組合為 SVO 三元組
- **支援部分 SVO**: 允許無賓語的 SV 結構 (可選)
- **置信度評分**:
  - 完整 SVO: confidence = 1.0
  - 部分 SV: confidence = 0.7
- **批次提取**: 支援批次處理提升效率
- **複雜度**: O(n³) 解析 + O(n) 提取

##### SyntaxAnalyzer (統一分析介面)
- **整合功能**:
  - 依存句法分析
  - SVO 三元組提取
  - 統一輸出格式 (Dict)
- **輸出欄位**:
  - `text`: 原始文本
  - `tokens`: 詞彙列表
  - `dependency_edges`: 依存邊列表
  - `svo_triples`: SVO 三元組列表
  - `root_verb`: 根動詞
  - `num_edges`, `num_triples`: 統計數量
- **批次分析**: 支援批次處理

##### 測試 (test_syntax.py)
- **檔案**: `tests/test_syntax.py` (500+ 行)
- **測試格式**: 直接執行 (非 pytest)
  ```bash
  conda activate ai_env
  python tests/test_syntax.py
  ```
- **測試覆蓋**: 17 個測試函式
  1. DependencyParser 測試 (6 個)
     - 初始化測試
     - 簡單句子解析
     - 複雜句子解析
     - 批次解析
     - 依存樹獲取
     - 根動詞提取
  2. SVOExtractor 測試 (6 個)
     - 初始化測試
     - 簡單 SVO 提取
     - 複雜 SVO 提取
     - 批次 SVO 提取
     - 部分 SVO 測試
     - 所有關係提取
  3. SyntaxAnalyzer 測試 (3 個)
     - 初始化測試
     - 綜合分析
     - 批次分析
  4. 整合測試 (2 個)
     - SVO 轉字典
     - 完整工作流程
- **測試結果**: 17/17 通過 (100%)
- **測試資料**: 中文簡單句、複雜句、批次測試

##### 文檔 (SYNTAX_PARSING_GUIDE.md)
- **檔案**: `docs/guides/SYNTAX_PARSING_GUIDE.md`
- **內容**:
  - 理論背景 (依存句法、Biaffine Attention)
  - 實作細節 (3 個核心類別)
  - 使用範例與應用場景
  - 效能分析與限制
  - PyTorch 2.6 相容性問題說明

##### 應用場景
- **查詢理解**: 提取查詢的主語、動詞、賓語
- **文檔摘要**: 識別核心句子結構
- **關係抽取**: 識別實體間關係 (用於知識圖譜)
- **問答系統**: 理解問題結構
- **語義搜尋**: 句法特徵增強檢索

### 修改 *Changed*

#### 專案方向調整
- **原方向**: 學術論文搜尋引擎
- **新方向**: 中文新聞智能檢索系統
- **調整原因**:
  - 新聞資料更易取得 (CNA, PTS 等公開來源)
  - 更適合展示傳統 IR vs 現代 NLP 差異
  - 應用場景更廣泛且實用
  - 評估更容易 (新聞查詢更直觀)

#### 技術棧確定
- **Web 框架**: 選擇 Flask (簡單易學)
- **資料庫**: SQLite (開發), PostgreSQL (生產)
- **前端**: Bootstrap 5 + Chart.js
- **部署**: Docker + Nginx

### 修復 *Fixed*

#### PyTorch 2.6+ 相容性問題
- **問題**: SuPar 模型載入失敗
  ```
  WeightsUnpickler error: Unsupported global: GLOBAL supar.utils.config.Config
  was not an allowed global by default
  ```
- **原因**: PyTorch 2.6 將 `torch.load` 的 `weights_only` 預設值改為 `True`
- **解決方案**: Monkey patch `torch.load` 臨時使用 `weights_only=False`
  ```python
  original_load = torch.load
  def patched_load(*args, **kwargs):
      kwargs['weights_only'] = False
      return original_load(*args, **kwargs)
  torch.load = patched_load
  ```
- **影響**: 在 PyTorch 2.6+ (含 2.7.1) 環境可正常載入 SuPar 模型

#### ROOT 關係大小寫問題
- **問題**: `get_root_verb()` 返回 `None`
- **原因**: 程式碼檢查 `relation == 'ROOT'`，但 SuPar 返回小寫 `'root'`
- **修復**: 改為 `relation.lower() == 'root'`
- **位置**: `src/ir/syntax/parser.py` 兩處

#### 測試案例調整
- **問題**: "張三吃蘋果" 被 Jieba 錯誤斷詞為 ["張三吃", "蘋果"]
- **原因**: Jieba 預設詞典問題
- **修復**: 改用 "我喜歡你" (正確斷詞為 ["我", "喜歡", "你"])
- **影響**: `test_svo_extract_simple()` 測試通過

### 文檔 *Documentation*

#### 新增文檔
- ✅ `docs/project/PROPOSAL.md` (738 行) - 期末專案提案
- ✅ `docs/project/ARCHITECTURE.md` (1400+ 行) - 系統架構文件
- ✅ `docs/project/DATASET.md` (1200+ 行) - 資料集規劃文件
- ✅ `docs/project/TODO.md` (1000+ 行) - 開發任務清單
- ✅ `docs/guides/SYNTAX_PARSING_GUIDE.md` - 句法分析指南

#### 更新文檔
- ✅ `docs/CHANGELOG.md` - 本變更紀錄

### 專案進度 *Progress*

#### 已完成模組 (100%)
- ✅ Phase 0: 專案規劃
  - PROPOSAL.md, ARCHITECTURE.md, DATASET.md, TODO.md
- ✅ Phase 1: 傳統 IR 模組 (M1-M7)
  - Boolean Retrieval, Inverted Index, Positional Index
  - Vector Space Model (TF-IDF)
  - Evaluation Metrics (MAP, nDCG)
  - Rocchio Query Expansion
  - Clustering & Summarization
- ✅ Phase 2: 現代 NLP 模組 (Phase 1-5)
  - Chinese Tokenization (Jieba, CKIP, PKUSeg)
  - Named Entity Recognition (CKIP Transformers)
  - Keyword Extraction (TextRank, YAKE, KeyBERT, RAKE + Ensemble)
  - Topic Modeling (LDA, BERTopic)
  - Syntactic Parsing (SuPar Dependency + SVO)

#### 待開始模組 (0%)
- ⏸️ Phase 3: 資料收集與預處理 (Week 12-13)
- ⏸️ Phase 4: 新增檢索模組 (Week 14-15)
- ⏸️ Phase 5: Web UI 開發 (Week 16-17)
- ⏸️ Phase 6: 整合與測試 (Week 18-19)
- ⏸️ Phase 7: 評估與優化 (Week 20-21)
- ⏸️ Phase 8: 文檔與展示 (Week 22)

#### 整體完成度
- **核心模組**: ~60% (所有 IR/NLP 模組完成)
- **資料準備**: 0% (待開始)
- **系統整合**: 0% (待開始)
- **評估優化**: 0% (待開始)
- **總體進度**: ~30%

### 下一步 *Next Steps*

根據 TODO.md 規劃，接下來的工作重點：

#### Phase 3: 資料收集與預處理 (Week 12-13)
1. **爬蟲開發** (Day 1-3)
   - 實作 CNA, PTS, TechNews 爬蟲
   - Scrapy 框架設定
   - 錯誤處理與重試機制

2. **爬蟲測試** (Day 4-5)
   - 小規模測試 (100-500 篇)
   - 驗證資料完整性
   - 效能測試

3. **正式收集** (Day 6-7)
   - 爬取 30,000-50,000 篇新聞
   - 監控進度與錯誤

4. **資料清洗** (Week 13, Day 1-3)
   - HTML 清除
   - 去重
   - 格式統一

5. **資料分析** (Day 4-5)
   - 統計分析
   - 視覺化

6. **NLP 預處理** (Day 6-7)
   - 批次斷詞、NER
   - 關鍵字、主題建模
   - 索引建立

7. **評估資料集建立** (Day 7)
   - 50 測試查詢設計
   - QRELS 標註

---

## [0.8.0] - 2025-11-13

### 新增 *Added*

#### 📝 自動摘要 (Automatic Summarization)

##### 靜態摘要 (Static Summarization)
- **核心實作** `src/ir/summarize/static.py` (202 statements, 80% coverage)
  - `StaticSummarizer` 類別：靜態摘要引擎
  - Lead-k 摘要（提取前 k 句）
  - TF-IDF 關鍵句提取
  - 查詢導向摘要
  - 多文件摘要

- **Lead-k 摘要**
  - **策略**: 提取文件開頭前 k 個句子
  - **適用場景**: 新聞文章、技術文件（重要資訊前置）
  - **優點**: 簡單高效、基線方法
  - **複雜度**: O(n) 其中 n 為文件長度
  - **範例**:
    ```python
    summarizer = StaticSummarizer()
    summary = summarizer.lead_k_summarization(text, k=3)
    print(summary.text)  # 前三句話
    ```

- **TF-IDF 關鍵句提取**
  - **演算法**: 基於 TF-IDF 分數選擇重要句子
  - **評分公式**:
    ```
    sentence_score = (Σ TF-IDF(term)) / sentence_length

    其中 TF-IDF(term) = TF(term) × IDF(term)
    IDF(term) = log(N / df(term))
    ```
  - **位置偏差 (Position Bias)**:
    ```
    final_score = base_score × (1 + 0.5 × position_weight)
    position_weight = 1 / (1 + sentence_position)
    ```
  - **特點**:
    - 內容重要性：高 TF-IDF 的詞項表示重要概念
    - 位置加權：前面的句子獲得額外權重
    - 長度正規化：避免偏向長句子
    - 順序保留：選擇後按原始位置排序
  - **複雜度**: O(n × m + n log n)
    - n = 句子數
    - m = 平均詞項數
  - **範例**:
    ```python
    summary = summarizer.key_sentence_extraction(
        text, k=5, use_position_bias=True
    )
    for sent in summary.sentences:
        print(f"[{sent.position}] {sent.text} (score={sent.score:.3f})")
    ```

- **查詢導向摘要 (Query-Focused Summarization)**
  - **策略**: 提取與查詢最相關的句子
  - **相似度計算**: Cosine similarity between query and sentences
    ```
    similarity(query, sentence) = |Q ∩ S| / sqrt(|Q| × |S|)

    其中:
    - Q: 查詢詞項集合
    - S: 句子詞項集合
    - |Q ∩ S|: 交集大小
    ```
  - **適用**: 針對特定主題生成摘要
  - **複雜度**: O(n × m) 其中 n=句子數, m=平均詞項數

- **多文件摘要 (Multi-Document Summarization)**
  - **策略**: 從多個文件中提取代表性句子
  - **多樣性控制**: 避免冗餘資訊
    ```
    diversity_check: similarity(candidate, existing) < threshold
    使用 Jaccard 相似度: J(A,B) = |A ∩ B| / |A ∪ B|
    ```
  - **貪婪選擇演算法**:
    1. 對所有句子計算 TF-IDF 分數
    2. 按分數降序排列候選句子
    3. 依序選擇與已選句子差異足夠的句子
    4. 重複直到達到目標數量 k
  - **複雜度**: O(d × n × m + s²)
    - d = 文件數
    - n = 平均句子數
    - s = 選擇的句子數
  - **範例**:
    ```python
    summary = summarizer.multi_document_summarization(
        documents, k=10, diversity_threshold=0.5
    )
    ```

- **核心功能**:
  - `segment_sentences()`: 句子分割
    - 使用正規表達式分割句子（.!?）
    - 長度過濾（min_sentence_length, max_sentence_length）
  - `compute_term_frequencies()`: 計算詞項頻率
  - `compute_idf()`: 計算 IDF 分數
  - `score_sentence_tfidf()`: TF-IDF 句子評分
  - `_sentence_similarity()`: Jaccard 句子相似度

- **資料結構**:
  - `Sentence`: 句子表示
    - `text`: 原始文本
    - `position`: 位置（0-indexed）
    - `doc_id`: 文件識別碼
    - `tokens`: 詞項列表
    - `score`: 重要性分數
  - `Summary`: 摘要結果
    - `sentences`: 選擇的句子列表
    - `method`: 摘要方法
    - `compression_ratio`: 壓縮比率
    - `original_length`: 原始句子數
    - `text`: 純文本摘要（property）

##### 動態摘要 (Dynamic Summarization) - KWIC
- **核心實作** `src/ir/summarize/dynamic.py` (214 statements, 76% coverage)
  - `KWICGenerator` 類別：KWIC 產生器
  - KeyWord In Context (KWIC) 片段生成
  - 多種視窗策略
  - 結果快取機制

- **KWIC (KeyWord In Context)**
  - **定義**: 提取關鍵字周圍的上下文視窗
  - **應用場景**:
    - 搜尋結果片段（Search snippets）
    - 關鍵字高亮顯示
    - 上下文預覽
  - **視窗類型**:
    1. **Fixed Window** (固定視窗):
       - 固定字元數量的上下文
       - 在詞邊界截斷
       - 最快速、最簡單
    2. **Sentence Window** (句子視窗):
       - 提取包含關鍵字的完整句子
       - 尋找句子邊界（.!?）
       - 最完整的上下文
    3. **Adaptive Window** (自適應視窗):
       - 在自然斷點處調整（, ; : -）
       - 保留完整短語
       - 平衡完整性與長度
  - **複雜度**: O(n × m)
    - n = 文本長度
    - m = 查詢詞項數

- **快取機制 (Caching)**
  - **策略**: LRU (Least Recently Used) 淘汰
  - **快取鍵**: (query, text_hash)
  - **效能提升**: 重複查詢達到 O(1)
  - **快取管理**:
    ```python
    generator = KWICGenerator(enable_cache=True, max_cache_size=1000)

    # 查詢統計
    stats = generator.get_cache_stats()
    print(f"Cache size: {stats['size']}/{stats['max_size']}")

    # 清空快取
    generator.clear_cache()
    ```

- **高亮與格式化**
  - **Markdown**: `**keyword**`
  - **ANSI**: `\033[1;31mkeyword\033[0m` (紅色粗體)
  - **HTML**: `<mark>keyword</mark>`
  - **範例**:
    ```python
    result = generator.generate(text, "machine learning")
    output = generator.format_results(result, highlight_style='markdown')
    print(output)
    # 輸出: ...with **machine** **learning** algorithms...
    ```

- **多文件 KWIC**
  - 跨多個文件搜尋關鍵字
  - 保留文件來源資訊
  - 限制每文件匹配數
  - **範例**:
    ```python
    result = generator.generate_multi(
        documents, query="keyword", max_matches_per_doc=3
    )
    ```

- **核心功能**:
  - `generate()`: 單文件 KWIC 生成
  - `generate_multi()`: 多文件 KWIC 生成
  - `_find_keyword_matches()`: 尋找所有匹配
  - `_extract_context()`: 提取上下文視窗
  - `_extract_fixed_window()`: 固定視窗提取
  - `_extract_sentence_window()`: 句子視窗提取
  - `_extract_adaptive_window()`: 自適應視窗提取
  - `format_results()`: 格式化顯示結果

- **資料結構**:
  - `KWICMatch`: KWIC 匹配
    - `keyword`: 匹配的關鍵字
    - `position`: 文件中的位置
    - `left_context`: 左側上下文
    - `right_context`: 右側上下文
    - `doc_id`: 文件識別碼
    - `snippet`: 格式化片段（property）
  - `KWICResult`: KWIC 結果
    - `matches`: 匹配列表
    - `query`: 原始查詢
    - `num_documents`: 搜尋文件數
    - `cache_hit`: 是否命中快取
    - `num_matches`: 匹配總數（property）

#### 🛠️ CLI 摘要工具
- **摘要工具** `scripts/summarize_doc.py`
  - **輸入選項**:
    - `--input`: 單一文件路徑
    - `--input-dir`: 多文件目錄

  - **摘要方法** (`--method`):
    - `lead-k`: Lead-k 摘要
    - `key-sentence`: 關鍵句提取
    - `query-focused`: 查詢導向摘要
    - `multi-doc`: 多文件摘要
    - `kwic`: KWIC 片段生成

  - **通用參數**:
    - `--k`: 提取句子數（預設 3）
    - `--query`: 查詢字串（query-focused, kwic 必需）
    - `--output`: 輸出檔案（預設 stdout）

  - **靜態摘要選項**:
    - `--position-bias` / `--no-position-bias`: 位置偏差
    - `--min-sentence-length`: 最小句子長度（預設 5）
    - `--max-sentence-length`: 最大句子長度（預設 100）

  - **KWIC 選項**:
    - `--window`: 視窗大小（預設 50）
    - `--window-type`: 視窗類型（fixed/sentence/adaptive）
    - `--max-matches`: 最大匹配數
    - `--highlight-style`: 高亮樣式（markdown/ansi/html）
    - `--case-sensitive`: 大小寫敏感

  - **多文件選項**:
    - `--diversity-threshold`: 多樣性閾值（預設 0.5）

  - **顯示選項**:
    - `--verbose`: 詳細輸出
    - `--show-scores`: 顯示句子分數

- **使用範例**:
  ```bash
  # Lead-k 摘要
  python scripts/summarize_doc.py --input article.txt --method lead-k --k 5

  # 關鍵句提取（無位置偏差）
  python scripts/summarize_doc.py --input doc.txt --method key-sentence --k 3 --no-position-bias --show-scores

  # 查詢導向摘要
  python scripts/summarize_doc.py --input paper.txt --method query-focused --query "neural networks deep learning" --k 5

  # KWIC 固定視窗
  python scripts/summarize_doc.py --input text.txt --method kwic --query "machine learning" --window 40 --highlight-style ansi

  # KWIC 句子視窗
  python scripts/summarize_doc.py --input doc.txt --method kwic --query "algorithm" --window-type sentence

  # 多文件摘要
  python scripts/summarize_doc.py --input-dir articles/ --method multi-doc --k 10 --diversity-threshold 0.6

  # 輸出到檔案
  python scripts/summarize_doc.py --input doc.txt --method key-sentence --k 5 --output summary.txt
  ```

#### ✅ 測試與驗證
- **測試檔案** `tests/test_summarization.py` (28 tests, 100% pass rate)
  - **TestStaticSummarization** (8 tests):
    - `test_sentence_segmentation`: 句子分割
    - `test_lead_k_summarization`: Lead-k 摘要
    - `test_key_sentence_extraction`: 關鍵句提取
    - `test_query_focused_summarization`: 查詢導向
    - `test_multi_document_summarization`: 多文件摘要
    - `test_empty_text`: 空文本處理
    - `test_compute_term_frequencies`: TF 計算
    - `test_compute_idf`: IDF 計算

  - **TestKWICGeneration** (15 tests):
    - `test_basic_kwic_generation`: 基本 KWIC 生成
    - `test_kwic_match_structure`: 匹配結構驗證
    - `test_kwic_case_insensitive`: 大小寫不敏感
    - `test_kwic_case_sensitive`: 大小寫敏感
    - `test_kwic_sentence_window`: 句子視窗
    - `test_kwic_adaptive_window`: 自適應視窗
    - `test_kwic_multiple_matches`: 多重匹配
    - `test_kwic_max_matches`: 匹配數限制
    - `test_kwic_multi_document`: 多文件 KWIC
    - `test_kwic_no_match`: 無匹配情況
    - `test_kwic_cache`: 快取機制
    - `test_kwic_cache_clear`: 快取清除
    - `test_kwic_formatting`: 格式化輸出
    - `test_kwic_snippet_property`: 片段屬性
    - `test_get_snippets`: 獲取片段方法

  - **TestEdgeCases** (5 tests):
    - `test_single_word_document`: 單詞文件
    - `test_very_long_sentence`: 超長句子過濾
    - `test_special_characters`: 特殊字元處理
    - `test_unicode_text`: Unicode 文本
    - `test_kwic_empty_query`: 空查詢

- **測試結果**:
  ```
  ============================= test session starts ==============================
  tests/test_summarization.py::TestStaticSummarization (8 tests) PASSED
  tests/test_summarization.py::TestKWICGeneration (15 tests) PASSED
  tests/test_summarization.py::TestEdgeCases (5 tests) PASSED

  ============================== 28 passed in 4.33s ===============================
  ```

- **測試覆蓋率**:
  - `static.py`: 202 statements, **80% coverage**
  - `dynamic.py`: 214 statements, **76% coverage**

### 技術細節 *Technical Details*

#### 摘要方法比較
| 方法 | 複雜度 | 優點 | 缺點 | 適用場景 |
|------|--------|------|------|----------|
| Lead-k | O(n) | 簡單快速、效果佳 | 假設前置重要性 | 新聞、技術文件 |
| TF-IDF | O(n×m + n log n) | 內容感知、可配置 | 需要統計計算 | 通用文件摘要 |
| Query-Focused | O(n×m) | 針對性強 | 需要查詢輸入 | 搜尋結果、QA |
| Multi-Doc | O(d×n×m + s²) | 跨文件整合 | 計算成本高 | 主題摘要、新聞彙整 |
| KWIC | O(n×m) | 即時生成、上下文豐富 | 不是完整摘要 | 搜尋片段、預覽 |

#### 評估指標
摘要品質通常使用 **ROUGE** (Recall-Oriented Understudy for Gisting Evaluation)：
- **ROUGE-N**: N-gram 重疊
- **ROUGE-L**: 最長公共子序列
- **ROUGE-S**: Skip-bigram 共現

本專案實作的是**提取式摘要** (Extractive Summarization)，從原文選擇句子；
相對的，**生成式摘要** (Abstractive Summarization) 使用 LLM 生成新句子。

#### 效能最佳化
1. **句子分割**: 使用正規表達式，O(n) 一次掃描
2. **TF-IDF 計算**: 快取詞項頻率避免重複計算
3. **KWIC 快取**: LRU 策略，重複查詢 O(1)
4. **多樣性檢查**: Jaccard 相似度，O(1) 集合操作

### 使用範例 *Usage Examples*

#### Python API 使用

**靜態摘要**:
```python
from src.ir.summarize.static import StaticSummarizer

text = """
Machine learning is a subset of artificial intelligence.
It enables computers to learn from data without explicit programming.
Deep learning is a type of machine learning based on neural networks.
Neural networks consist of layers of interconnected nodes.
Applications include image recognition and natural language processing.
"""

summarizer = StaticSummarizer()

# Lead-k 摘要
summary = summarizer.lead_k_summarization(text, k=3)
print(f"Lead-3 summary ({summary.compression_ratio:.1%} compression):")
print(summary.text)

# 關鍵句提取
summary = summarizer.key_sentence_extraction(text, k=2, use_position_bias=True)
for i, sent in enumerate(summary.sentences):
    print(f"[{i+1}] (score={sent.score:.3f}): {sent.text}")

# 查詢導向摘要
summary = summarizer.query_focused_summarization(text, "neural networks", k=2)
print(f"Query-focused summary: {summary.text}")
```

**動態摘要 (KWIC)**:
```python
from src.ir.summarize.dynamic import KWICGenerator

text = """
Machine learning is a powerful technology that enables computers to learn
from data. Deep learning uses neural networks to achieve state-of-the-art
results. Many applications of machine learning exist in industry.
"""

# 固定視窗 KWIC
generator = KWICGenerator(window_size=30, window_type='fixed')
result = generator.generate(text, "machine learning")

for i, match in enumerate(result.matches):
    print(f"{i+1}. {match.snippet}")
# 輸出: ...enables **machine** **learning** to achieve...

# 句子視窗 KWIC
generator = KWICGenerator(window_type='sentence')
result = generator.generate(text, "neural networks")
output = generator.format_results(result, highlight_style='markdown')
print(output)

# 快取演示
result1 = generator.generate(text, "learning")  # Cache miss
result2 = generator.generate(text, "learning")  # Cache hit
print(f"Cache hit: {result2.cache_hit}")  # True
```

### 效能 *Performance*

#### 複雜度總結
| 功能 | 時間複雜度 | 空間複雜度 | 備註 |
|------|-----------|-----------|------|
| Lead-k | O(n) | O(k) | n=文件長度, k=摘要長度 |
| TF-IDF 摘要 | O(n×m + n log n) | O(n + v) | m=平均詞數, v=詞彙量 |
| Query-Focused | O(n×m) | O(n) | 相似度計算 |
| Multi-Doc | O(d×n×m + s²) | O(d×n) | d=文件數, s=選擇句子數 |
| KWIC (無快取) | O(n×m) | O(k×w) | w=視窗大小 |
| KWIC (有快取) | O(1) | O(c×k×w) | c=快取大小 |

#### 可擴展性
- **Lead-k**: 可處理任意長度文件，記憶體效率高
- **TF-IDF**: 適合中等規模文件（< 10,000 句）
- **KWIC**: 即時生成，適合線上搜尋系統
- **建議**: 大規模文件集使用 Lead-k 或分段處理

### 已知限制 *Limitations*
- **提取式限制**: 只能選擇原文句子，無法改寫或生成新內容
- **語義理解**: 基於統計方法，無深層語義理解
- **跨句推理**: 無法處理需要多句推理的摘要
- **句子分割**: 對於複雜標點符號可能誤判
- **多語言**: 中文句子分割可能不準確（需專門處理）
- **未實作**: ROUGE 評估、生成式摘要、主題模型

### 參考文獻 *References*
- Manning et al., "Introduction to Information Retrieval", Chapter 21, 23
- Luhn (1958). "The Automatic Creation of Literature Abstracts"
- Edmundson (1969). "New Methods in Automatic Extracting"
- Lin (2004). "ROUGE: A Package for Automatic Evaluation of Summaries"

---

## [0.7.0] - 2025-11-12

### 新增 *Added*

#### 📊 分群演算法 (Clustering Algorithms)

##### 文件分群 (Document Clustering)
- **核心實作** `src/ir/cluster/doc_cluster.py` (234 statements, 68% coverage)
  - `DocumentClusterer` 類別：文件分群引擎
  - 階層式聚合分群（Hierarchical Agglomerative Clustering, HAC）
  - K-means 平坦分群（K-means Flat Clustering）
  - 多種相似度度量與連結方法
  - Silhouette 分數評估

- **階層式聚合分群 (HAC)**
  - **演算法**: 自底向上（bottom-up）合併策略
  - **連結方法 (Linkage)**:
    - `single`: 單一連結（最大相似度）- 適合鏈狀群集
    - `complete`: 完全連結（最小相似度）- 適合緊密群集
    - `average`: 平均連結（平均相似度）- 折衷方案
  - **複雜度**:
    - Time: O(n²) 相似度矩陣 + O(n² log n) 優先佇列合併
    - Space: O(n²) 相似度矩陣

- **K-means 分群**
  - **演算法**: 迭代式質心更新
  - **特點**:
    - 隨機初始化質心
    - 文件分配至最近質心
    - 重新計算質心（平均向量）
    - 收斂檢查（質心移動 < tolerance）
  - **複雜度**: O(k × n × i × d)
    - k: 群集數量
    - n: 文件數量
    - i: 迭代次數（預設最多 100）
    - d: 向量維度

- **相似度度量 (Similarity Metrics)**:
  - **Cosine**: `cos(v1, v2) = (v1 · v2) / (||v1|| × ||v2||)`
    - 適合高維稀疏向量
    - 角度相似度，忽略長度
  - **Euclidean**: `dist(v1, v2) = sqrt(Σ(v1_i - v2_i)²)`
    - 歐幾里得距離
    - 考慮向量長度差異
  - **Jaccard**: `J(A, B) = |A ∩ B| / |A ∪ B|`
    - 集合相似度
    - 適合二元特徵

- **核心功能**:
  - `hierarchical_clustering()`: HAC 實作
    - 參數：k（群集數）、linkage（連結方法）、similarity_metric
    - 返回 `ClusteringResult` 包含樹狀結構與最終群集
  - `kmeans_clustering()`: K-means 實作
    - 參數：k、max_iterations、tolerance、random_seed
    - 支援固定隨機種子以重現結果
  - `cosine_similarity()`: 餘弦相似度計算
  - `euclidean_distance()`: 歐氏距離計算
  - `jaccard_similarity()`: Jaccard 係數計算
  - `evaluate_clusters()`: Silhouette 分數評估
    - 測量群集凝聚度（cohesion）與分離度（separation）
    - 分數範圍 [-1, 1]，越高越好

- **資料結構**:
  - `Cluster`: 群集容器
    - `cluster_id`: 唯一識別碼
    - `doc_ids`: 文件 ID 列表
    - `centroid`: 質心向量（K-means）
    - `size`: 群集大小
  - `ClusteringResult`: 分群結果
    - `num_clusters`: 群集數量
    - `clusters`: Cluster 物件列表
    - `assignments`: 文件到群集的映射
    - `dendrogram`: 樹狀圖（HAC）

##### 詞項分群 (Term Clustering)
- **核心實作** `src/ir/cluster/term_cluster.py` (177 statements, 48% coverage)
  - `TermClusterer` 類別：詞項分群引擎
  - 基於字串相似度的分群
  - 基於共現模式的分群
  - Star 分群演算法

- **編輯距離 (Edit Distance)**
  - **演算法**: Levenshtein 距離
  - **定義**: 將字串 s1 轉換成 s2 的最少編輯次數
  - **操作**: 插入、刪除、替換
  - **實作**: 動態規劃（Dynamic Programming）
  - **複雜度**:
    - Time: O(m × n) 其中 m, n 為字串長度
    - Space: O(m × n) DP 表格
  - **範例**:
    - `edit_distance("kitten", "sitting")` → 3
    - `edit_distance("color", "colour")` → 1

- **Star 分群演算法**
  - **策略**: 貪婪式選擇「星狀中心」
  - **步驟**:
    1. 計算每個詞項的潛力（potential）= 相似詞項數量
    2. 選擇最高潛力的未分群詞項作為中心
    3. 將所有相似詞項分配到此群集
    4. 移除已分群詞項並重複
  - **複雜度**: O(n²) 相似度矩陣 + O(n²) 分群過程
  - **適用**: 同義詞群集、拼寫變體

- **共現分群 (Co-occurrence Clustering)**
  - **基於**: 詞項在文件中的共同出現頻率
  - **假設**: 經常共現的詞項語義相關
  - **複雜度**: O(n² × d) 其中 d 為文件數

- **字串相似度度量**:
  - `edit_distance()`: Levenshtein 距離
  - `normalized_edit_distance()`: 正規化至 [0, 1]
  - `prefix_similarity()`: 前綴相似度

- **核心功能**:
  - `star_clustering()`: Star 分群
    - 參數：similarity_threshold（預設 0.7）、similarity_metric
    - 返回 `TermCluster` 列表
  - `edit_distance_clustering()`: 編輯距離分群
    - 參數：max_distance（預設 2）
    - 簡單貪婪策略
  - `cooccurrence_clustering()`: 共現分群
    - 參數：documents（文件詞項集合）、min_cooccurrence
    - 建立共現矩陣並分群

- **資料結構**:
  - `TermCluster`: 詞項群集
    - `cluster_id`: 唯一識別碼
    - `terms`: 詞項列表
    - `center`: 代表詞項（星狀中心）
    - `size`: 群集大小

#### 🛠️ CLI 分群工具
- **文件分群工具** `scripts/cluster_docs.py`
  - `--index`: VSM 索引檔案路徑（必需）
  - `--algorithm`: 分群演算法（必需）
    - `hac`: 階層式聚合分群
    - `kmeans`: K-means 分群
  - `--k`: 群集數量（必需）
  - `--linkage`: HAC 連結方法（預設 `complete`）
    - 選項：`single`, `complete`, `average`
  - `--max-iterations`: K-means 最大迭代次數（預設 100）
  - `--seed`: K-means 隨機種子（可選）
  - `--verbose`: 顯示詳細輸出

- **使用範例**:
  ```bash
  # 階層式完全連結分群
  python scripts/cluster_docs.py --index vsm_index.json --algorithm hac --k 5

  # K-means 分群（固定隨機種子）
  python scripts/cluster_docs.py --index vsm_index.json --algorithm kmeans --k 5 --seed 42

  # 單一連結分群
  python scripts/cluster_docs.py --index vsm_index.json --algorithm hac --k 10 --linkage single
  ```

#### ✅ 測試與驗證
- **測試檔案** `tests/test_clustering.py` (5 tests, 100% pass rate)
  - `TestDocumentClustering`:
    - `test_cosine_similarity`: 驗證餘弦相似度計算
    - `test_hac_clustering`: 驗證 HAC 分群結果
    - `test_kmeans_clustering`: 驗證 K-means 分群（固定種子）
  - `TestTermClustering`:
    - `test_edit_distance`: 驗證編輯距離計算
    - `test_star_clustering`: 驗證 Star 分群演算法

- **測試結果**:
  ```
  ============================= test session starts ==============================
  tests/test_clustering.py::TestDocumentClustering::test_cosine_similarity PASSED
  tests/test_clustering.py::TestDocumentClustering::test_hac_clustering PASSED
  tests/test_clustering.py::TestDocumentClustering::test_kmeans_clustering PASSED
  tests/test_clustering.py::TestTermClustering::test_edit_distance PASSED
  tests/test_clustering.py::TestTermClustering::test_star_clustering PASSED

  ============================== 5 passed in 4.02s ===============================
  ```

- **測試覆蓋率**:
  - `doc_cluster.py`: 234 statements, **68% coverage**
  - `term_cluster.py`: 177 statements, **48% coverage**

### 技術細節 *Technical Details*

#### 分群演算法選擇指南
- **HAC Complete-link**:
  - 優點：產生緊密、高品質群集
  - 缺點：對雜訊敏感、計算成本高
  - 適用：小規模資料（< 10,000 文件）、需要樹狀結構

- **HAC Single-link**:
  - 優點：可發現任意形狀群集
  - 缺點：易產生鏈狀效應（chaining）
  - 適用：非球形群集、探索性分析

- **K-means**:
  - 優點：高效、可擴展至大規模資料
  - 缺點：需預設 k、對初始化敏感、假設球形群集
  - 適用：大規模文件集（> 10,000）、已知群集數

- **Star Clustering**:
  - 優點：無需預設群集數、適合同義詞發現
  - 缺點：貪婪策略、對閾值敏感
  - 適用：詞彙標準化、拼寫修正

#### Silhouette 分數解讀
- **計算公式**:
  ```
  s(i) = (b(i) - a(i)) / max(a(i), b(i))

  其中:
  - a(i): 文件 i 與同群集內其他文件的平均距離（凝聚度）
  - b(i): 文件 i 與最近鄰群集的平均距離（分離度）
  ```

- **分數解讀**:
  - **0.7 - 1.0**: 強結構（strong structure）
  - **0.5 - 0.7**: 合理結構（reasonable structure）
  - **0.25 - 0.5**: 弱結構（weak structure）
  - **< 0.25**: 無明顯結構（no substantial structure）

### 使用範例 *Usage Examples*

#### Python API 使用

**文件分群**:
```python
from src.ir.cluster.doc_cluster import DocumentClusterer

# 準備文件向量（TF-IDF 加權）
documents = {
    0: {"term1": 0.5, "term2": 0.3},
    1: {"term1": 0.4, "term2": 0.4},
    2: {"term3": 0.6, "term4": 0.5}
}

clusterer = DocumentClusterer()

# HAC 分群
result = clusterer.hierarchical_clustering(
    documents, k=2, linkage='complete'
)
print(f"Created {result.num_clusters} clusters")
for cluster in result.clusters:
    print(f"Cluster {cluster.cluster_id}: {cluster.doc_ids}")

# K-means 分群
result = clusterer.kmeans_clustering(
    documents, k=2, random_seed=42
)

# 評估品質
score = clusterer.evaluate_clusters(documents, result)
print(f"Silhouette score: {score:.3f}")
```

**詞項分群**:
```python
from src.ir.cluster.term_cluster import TermClusterer

terms = ["color", "colour", "colored", "paint", "painted"]
clusterer = TermClusterer()

# 編輯距離計算
dist = clusterer.edit_distance("color", "colour")
print(f"Edit distance: {dist}")  # 輸出: 1

# Star 分群
clusters = clusterer.star_clustering(
    terms, similarity_threshold=0.7
)
for cluster in clusters:
    print(f"Cluster {cluster.cluster_id} (center: {cluster.center}):")
    print(f"  Terms: {cluster.terms}")

# 編輯距離分群
clusters = clusterer.edit_distance_clustering(
    terms, max_distance=2
)
```

### 效能 *Performance*

#### 複雜度總結
| 演算法 | 時間複雜度 | 空間複雜度 | 備註 |
|--------|-----------|-----------|------|
| HAC | O(n² log n) | O(n²) | n = 文件數 |
| K-means | O(k·n·i·d) | O(n·d + k·d) | k=群集數, i=迭代次數, d=維度 |
| Star Clustering | O(n²·m) | O(n²) | m = 平均詞項長度 |
| Edit Distance | O(m·n) | O(m·n) | m, n = 字串長度 |

#### 可擴展性
- **HAC**: 適合 < 10,000 文件（O(n²) 記憶體瓶頸）
- **K-means**: 可擴展至百萬級文件（線性記憶體）
- **建議**: 大規模資料使用 K-means；需層次結構時用 HAC

### 已知限制 *Limitations*
- HAC 記憶體需求：O(n²) 相似度矩陣限制大規模應用
- K-means 對初始化敏感：不同隨機種子可能產生不同結果
- Star clustering 貪婪策略：無法保證全域最優
- 未實作 DBSCAN、Spectral Clustering 等進階演算法

### 參考文獻 *References*
- Manning et al., "Introduction to Information Retrieval", Chapter 16-17
- Steinbach, Karypis, Kumar (2000). "A Comparison of Document Clustering Techniques"
- Lloyd (1982). "Least squares quantization in PCM" (K-means 原始論文)

---

## [0.6.0] - 2025-11-12

### 新增 *Added*

#### 🔄 Rocchio 查詢擴展 (Query Expansion with Rocchio)
- **核心實作** `src/ir/ranking/rocchio.py` (121 statements, 74% coverage)
  - `RocchioExpander` 類別：Rocchio 演算法實作
  - 經典 Rocchio 公式支援
  - 擬相關回饋（Pseudo-Relevance Feedback）
  - 明確相關回饋（Explicit Relevance Feedback）
  - 詞項選擇與過濾
  - 查詢向量修改與重新加權

- **Rocchio 公式**
  ```
  Q_new = α × Q_orig + β × (1/|D_r|) × Σ D_r - γ × (1/|D_nr|) × Σ D_nr

  其中:
  - Q_orig: 原始查詢向量
  - D_r: 相關文件集合
  - D_nr: 非相關文件集合
  - α, β, γ: 調整參數（典型值：α=1.0, β=0.75, γ=0.15）
  ```

- **核心功能**
  - `expand_query()`: 使用 Rocchio 演算法擴展查詢
    - 結合原始查詢、相關文件、非相關文件
    - 計算新查詢向量權重
    - 過濾負權重與低權重詞項
    - 選擇 Top-K 擴展詞項
  - `expand_with_pseudo_feedback()`: 擬相關回饋擴展
    - 假設 Top-K 檢索結果為相關文件
    - 自動選擇相關/非相關文件分界
    - 無需人工標註
  - `reweight_query()`: 查詢向量重新加權
    - 結合原始與擴展詞項
    - 可選正規化
  - `get_top_expansion_terms()`: 獲取 Top-K 擴展詞項
    - 按權重降序排列
    - 使用 heap 最佳化
  - `set_parameters()`: 動態調整 α, β, γ 參數

- **資料結構**
  - `ExpandedQuery`: 擴展結果容器
    - `original_terms`: 原始查詢詞項
    - `expanded_terms`: 新增詞項
    - `all_terms`: 合併詞項列表
    - `term_weights`: 每個詞項的權重
    - `num_relevant`, `num_nonrelevant`: 使用的文件數

- **參數調整**
  - **α (alpha)**: 原始查詢權重（預設 1.0）
    - 強調使用者原始意圖
    - 防止查詢漂移（query drift）
  - **β (beta)**: 相關文件權重（預設 0.75）
    - 從正例學習
    - 加入相關文件的特徵詞
  - **γ (gamma)**: 非相關文件權重（預設 0.15）
    - 避免負例
    - 降低非相關詞項權重
  - **max_expansion_terms**: 最大擴展詞項數（預設 10）
  - **min_term_weight**: 最小詞項權重閾值（預設 0.1）

#### 🛠️ CLI 查詢擴展工具
- **擴展工具** `scripts/expand_query.py` (500+ 行)
  - `--query`: 指定查詢字串
  - `--mode`: 擴展模式
    - `pseudo`: 擬相關回饋（自動）
    - `explicit`: 明確相關回饋（需要標註）
  - `--index`: VSM 索引檔案路徑
  - `--topk`: 檢索文件數量（預設 20）
  - `--num-relevant`: 視為相關的前 K 個文件（預設 10）
  - `--num-nonrelevant`: 視為非相關的文件數（預設 0）

- **Rocchio 參數選項**
  - `--alpha`: 原始查詢權重（預設 1.0）
  - `--beta`: 相關文件權重（預設 0.75）
  - `--gamma`: 非相關文件權重（預設 0.15）
  - `--max-terms`: 最大擴展詞項數（預設 10）
  - `--min-weight`: 最小詞項權重（預設 0.1）

- **明確回饋選項**
  - `--relevant`: 相關文件 ID 檔案
  - `--nonrelevant`: 非相關文件 ID 檔案

- **輸出選項**
  - `--no-rerank`: 跳過擴展查詢的重新檢索
  - `--verbose`: 顯示詳細輸出

- **使用範例**
  ```bash
  # 擬相關回饋（自動）
  python scripts/expand_query.py --query "information retrieval" \
      --mode pseudo --index vsm_index.json --topk 10

  # 明確相關回饋（人工標註）
  python scripts/expand_query.py --query "vector space model" \
      --mode explicit --index vsm_index.json --relevant rel_docs.txt

  # 自訂參數
  python scripts/expand_query.py --query "search engine" \
      --mode pseudo --index vsm_index.json \
      --alpha 1.0 --beta 0.8 --gamma 0.2 --max-terms 15

  # 包含非相關文件
  python scripts/expand_query.py --query "IR" --mode explicit \
      --index vsm_index.json --relevant rel.txt --nonrelevant nonrel.txt
  ```

#### ✅ 完整測試套件
- **測試檔案** (30 個測試案例全部通過)
  - `tests/test_rocchio.py` (30 tests, 74% coverage)
    - `TestBasicExpansion` (4 tests)
      - 基本查詢擴展
      - 原始詞項追蹤
      - 擴展詞項驗證
    - `TestWithNonRelevantDocs` (2 tests)
      - 包含非相關文件的擴展
      - 權重降低驗證
    - `TestPseudoRelevanceFeedback` (3 tests)
      - 擬相關回饋基本功能
      - 包含非相關文件
      - 空文件處理
    - `TestParameters` (4 tests)
      - α, β, γ 參數效果測試
      - 參數動態設定
    - `TestExpansionControl` (3 tests)
      - 最大擴展詞項限制
      - 最小權重閾值
      - 零擴展測試
    - `TestReweighting` (3 tests)
      - 查詢重新加權
      - 正規化/非正規化
    - `TestTopExpansionTerms` (3 tests)
      - Top-K 詞項選擇
      - 權重排序驗證
    - `TestEdgeCases` (5 tests)
      - 無相關文件
      - 空查詢向量
      - 負權重過濾
    - `TestRocchioFormula` (2 tests)
      - 完整公式驗證
      - 文件平均計算
    - `TestIntegration` (1 test)
      - 完整工作流測試

- **測試結果**
  - ✅ 30/30 測試通過 (100% pass rate)
  - ✅ 執行時間：4.14 秒
  - ✅ 覆蓋率：74% (121 statements, 32 missed)
    - 未覆蓋：demo 函式、部分錯誤處理

### 技術特性 *Technical Highlights*

#### Rocchio 演算法原理
- **向量空間修正**: 基於相關性回饋修改查詢向量
- **正例學習**: 相關文件貢獻正向權重
- **負例避免**: 非相關文件貢獻負向權重
- **查詢漂移控制**: α 參數保持原始查詢意圖

#### 擬相關回饋 (Pseudo-Relevance Feedback)
- **自動化**: 無需人工標註
- **假設**: Top-K 檢索結果為相關文件
- **優點**:
  - 自動改善檢索效果
  - 無需使用者互動
  - 適合批次查詢
- **風險**:
  - 初始檢索品質影響大
  - 可能引入查詢漂移

#### 明確相關回饋 (Explicit Relevance Feedback)
- **使用者參與**: 需要人工標註相關/非相關
- **優點**:
  - 更準確的回饋
  - 可包含非相關文件資訊
  - 檢索效果提升更顯著
- **缺點**:
  - 需要使用者互動
  - 標註成本高

#### 查詢擴展效果
- **詞彙不匹配問題**: 解決同義詞、相關詞缺失
- **召回率提升**: 加入相關詞項增加相關文件數
- **精確率風險**: 不當擴展可能降低精確率
- **參數調校**: α, β, γ 需根據場景調整

#### 效能指標
- **擴展時間**: O(|D_r| × V + |D_nr| × V) where V is vocabulary size
- **Top-K 選擇**: O(V × log(k)) using heap
- **空間複雜度**: O(V) for expanded vector
- **整合開銷**: 與 VSM 檢索時間相當

### 應用場景 *Use Cases*

1. **互動式搜尋引擎** - 使用者點擊相關文件後改善結果
2. **個人化推薦** - 基於使用者歷史相關文件擴展查詢
3. **批次查詢最佳化** - 擬相關回饋自動改善檢索品質
4. **領域專業搜尋** - 加入領域詞彙擴展
5. **多語言檢索** - 擴展跨語言同義詞

### 範例 *Examples*

#### Python API 使用

```python
from src.ir.ranking.rocchio import RocchioExpander

# 初始化擴展器
expander = RocchioExpander(
    alpha=1.0,    # 原始查詢權重
    beta=0.75,    # 相關文件權重
    gamma=0.15,   # 非相關文件權重
    max_expansion_terms=10,
    min_term_weight=0.1
)

# 原始查詢向量
query_vector = {
    "information": 0.8,
    "retrieval": 0.6
}

# 相關文件向量
relevant_docs = [
    {"information": 0.5, "retrieval": 0.7, "system": 0.3},
    {"information": 0.6, "search": 0.4, "engine": 0.3},
    {"retrieval": 0.5, "document": 0.4, "index": 0.3}
]

# 擴展查詢
expanded = expander.expand_query(query_vector, relevant_docs)

print(f"Original terms: {expanded.original_terms}")
# ['information', 'retrieval']

print(f"Expanded terms: {expanded.expanded_terms}")
# ['system', 'search', 'engine', 'document', ...]

print(f"Term weights:")
for term in expanded.all_terms[:5]:
    print(f"  {term}: {expanded.term_weights[term]:.4f}")
```

#### 擬相關回饋範例

```python
# 假設 Top-10 為相關文件
top_documents = [...]  # 從檢索結果獲取

expanded = expander.expand_with_pseudo_feedback(
    query_vector,
    top_documents,
    num_relevant=10,      # 前10個視為相關
    num_nonrelevant=5     # 第11-15個視為非相關
)

# 獲取 Top-5 擴展詞項
top_terms = expander.get_top_expansion_terms(expanded, k=5)
for term, weight in top_terms:
    print(f"{term}: {weight:.4f}")
```

#### 與 VSM 整合

```python
from src.ir.retrieval.vsm import VectorSpaceModel
from src.ir.ranking.rocchio import RocchioExpander

# 建立 VSM
vsm = VectorSpaceModel()
vsm.build_index(documents)

# 初始檢索
query = "information retrieval"
initial_result = vsm.search(query, topk=20)

# 獲取 Top-K 文件向量
top_doc_vectors = [
    vsm.get_document_vector(doc_id)
    for doc_id in initial_result.doc_ids[:10]
]

# 建立查詢向量
from collections import defaultdict
query_tokens = vsm.inverted_index.tokenizer(query)
query_tf = defaultdict(int)
for token in query_tokens:
    query_tf[token] += 1

query_vector = vsm.term_weighting.vectorize(
    dict(query_tf), tf_scheme='l', idf_scheme='n', normalize='c'
)

# Rocchio 擴展
expander = RocchioExpander()
expanded = expander.expand_with_pseudo_feedback(
    query_vector, top_doc_vectors, num_relevant=10
)

# 使用擴展查詢重新檢索
expanded_query_str = " ".join(expanded.all_terms)
final_result = vsm.search(expanded_query_str, topk=20)

print(f"Original: {initial_result.num_results} results")
print(f"Expanded: {final_result.num_results} results")
```

#### 參數調校範例

```python
# 實驗不同參數配置
configs = [
    (1.0, 0.75, 0.15, "Standard"),
    (1.0, 1.0, 0.0, "Positive only"),
    (1.0, 0.5, 0.5, "Balanced pos/neg"),
    (2.0, 0.5, 0.1, "High original weight"),
]

for alpha, beta, gamma, desc in configs:
    expander.set_parameters(alpha, beta, gamma)
    expanded = expander.expand_query(query_vector, relevant_docs)

    print(f"{desc} (α={alpha}, β={beta}, γ={gamma}):")
    print(f"  Expansion terms: {len(expanded.expanded_terms)}")
    print(f"  Top terms: {expanded.expanded_terms[:3]}")
```

### CLI 使用範例

```bash
# 基本擬相關回饋
python scripts/expand_query.py \
    --query "information retrieval" \
    --mode pseudo \
    --index test_vsm_index.json \
    --topk 10 \
    --num-relevant 5

# 輸出:
# Query: "information retrieval"
# Mode: Pseudo-relevance feedback (top-10 documents)
# ============================================================
#
# 1. Initial Retrieval:
#    Retrieved 9 documents
#
# 4. Query Expansion (Rocchio):
#    Relevant docs: 5
#    Original terms: 2
#    Expanded terms: 8
#
# 5. Top Expansion Terms:
#    1. system: 0.0750
#    2. search: 0.0625
#    3. engine: 0.0562
#    ...

# 明確相關回饋
# 先創建相關文件列表
echo "0\n2\n5" > relevant_docs.txt

python scripts/expand_query.py \
    --query "vector space model" \
    --mode explicit \
    --index test_vsm_index.json \
    --relevant relevant_docs.txt

# 自訂參數（更激進的擴展）
python scripts/expand_query.py \
    --query "search" \
    --mode pseudo \
    --index test_vsm_index.json \
    --alpha 1.0 \
    --beta 1.0 \
    --gamma 0.0 \
    --max-terms 20 \
    --min-weight 0.05 \
    --verbose
```

### 已知限制 *Known Limitations*

1. **覆蓋率**: 74%（目標 80%+）
   - 未覆蓋：demo 函式、CLI 部分錯誤處理
   - 計劃增加整合測試

2. **查詢漂移風險**: 擬相關回饋可能引入不相關詞項
   - 解決方案：調整 α 參數保持原始查詢權重
   - 限制擴展詞項數量

3. **初始檢索品質依賴**: 擬相關回饋效果受初始結果影響
   - 初始結果差 → 擴展品質差
   - 建議結合多種擴展策略

4. **參數調校**: α, β, γ 需根據資料集調整
   - 無通用最佳參數
   - 需要實驗驗證

5. **計算開銷**: 需要額外的文件向量提取與計算
   - 對大規模資料集可能較慢
   - 可考慮快取文件向量

### 整合性說明

**與 Phase 4 (VSM) 整合**：
- Rocchio 使用 VSM 的文件向量與查詢向量
- 擴展後的查詢可直接用於 VSM 檢索
- CLI 工具無縫整合 VSM 索引

**與 Phase 5 (Evaluation) 整合**：
- 使用 MAP, nDCG 評估擴展效果
- 比較擴展前後的檢索性能
- 實驗範例:
  ```python
  # 原始查詢評估
  original_result = vsm.search(query)
  original_ap = metrics.average_precision(
      original_result.doc_ids, relevant_set
  )

  # 擴展查詢評估
  expanded_result = vsm.search(expanded_query)
  expanded_ap = metrics.average_precision(
      expanded_result.doc_ids, relevant_set
  )

  improvement = (expanded_ap - original_ap) / original_ap * 100
  print(f"AP improvement: {improvement:.2f}%")
  ```

**實驗應用**：
- 可用於作業/報告的查詢擴展實驗
- 比較不同參數配置的效果
- 分析擴展詞項的品質

### 下一步計劃 *Next Steps*

- [ ] 提升測試覆蓋率至 80%+ (增加 CLI 測試)
- [ ] 實作其他查詢擴展方法（共現分析、詞嵌入）
- [ ] 整合至期末專案搜尋引擎
- [ ] 新增查詢擴展效果自動評估
- [ ] 實作自適應參數調整
- [ ] 支援增量式回饋（多輪擴展）

---

## [0.5.0] - 2025-11-12

### 新增 *Added*

#### 📈 評估指標模組 (Evaluation Metrics)
- **核心實作** `src/ir/eval/metrics.py` (186 statements, 73% coverage)
  - `Metrics` 類別：完整 IR 評估指標計算器
  - 支援二元相關性評估（Binary Relevance）
  - 支援分級相關性評估（Graded Relevance）
  - 單查詢與多查詢評估
  - 標準 TREC 評估格式支援

- **二元相關性指標 (Binary Relevance)**
  - `precision()`: 精確率 = |Retrieved ∩ Relevant| / |Retrieved|
  - `recall()`: 召回率 = |Retrieved ∩ Relevant| / |Relevant|
  - `f_measure()`: F-measure (β可調整，預設F1)
  - `precision_at_k()`: Precision@K (前K個結果的精確率)
  - `recall_at_k()`: Recall@K (前K個結果的召回率)

- **排序檢索指標 (Ranked Retrieval)**
  - `average_precision()`: 平均精確率 (AP)
    - 公式：AP = (1/|Relevant|) × Σ(P@k × rel(k))
    - 強調相關文件早出現
  - `mean_average_precision()`: 平均平均精確率 (MAP)
    - 多查詢 AP 的平均值
    - IR 系統比較的標準指標
  - `reciprocal_rank()`: 倒數排名 (RR)
    - RR = 1 / (第一個相關文件的排名)
    - 適用於導航型查詢
  - `mean_reciprocal_rank()`: 平均倒數排名 (MRR)

- **分級相關性指標 (Graded Relevance)**
  - `dcg_at_k()`: 折扣累積增益 (DCG@K)
    - 公式：DCG = Σ(2^rel(i) - 1) / log₂(i + 1)
    - 支援 0-5 分相關性分級
  - `ndcg_at_k()`: 正規化 DCG (nDCG@K)
    - nDCG = DCG / IDCG (理想排序的DCG)
    - 範圍 [0, 1]，1為完美排序
    - 分級相關性評估的標準指標

- **核心功能**
  - `evaluate_query()`: 單查詢完整評估
    - 計算所有二元與分級指標
    - 支援多個 k 值 (預設 5, 10, 20)
    - 回傳完整指標字典
  - `evaluate_run()`: 批次查詢評估
    - 多查詢指標平均
    - 支援 TREC 格式評估
    - Per-query 與 aggregated 結果

- **資料結構**
  - `EvaluationResult`: 評估結果容器
    - precision, recall, f1, ap, rr, ndcg
    - num_relevant, num_retrieved, num_relevant_retrieved

#### 🛠️ CLI 評估工具
- **評估工具** `scripts/eval_run.py` (400+ 行)
  - `--results`: 載入系統檢索結果
  - `--qrels`: 載入相關性判斷（qrels）
  - `--relevance`: 載入分級相關性分數（可選）
  - `--k-values`: 指定 P@k, R@k, nDCG@k 的 k 值
  - `--per-query`: 顯示每個查詢的詳細評估
  - `--output`: 輸出結果至檔案（JSON/CSV/TXT）

- **支援格式**
  - JSON 格式：
    - Results: `{"q1": [doc1, doc2, ...], ...}`
    - Qrels: `{"q1": [rel_doc1, rel_doc2, ...], ...}`
    - Graded: `{"q1": {"doc1": 3, "doc2": 2, ...}, ...}`
  - TREC 格式：
    - Results: `query_id Q0 doc_id rank score run_id`
    - Qrels: `query_id 0 doc_id relevance`

- **使用範例**
  ```bash
  # 基本評估
  python scripts/eval_run.py --results run.json --qrels qrels.json

  # 指定 k 值
  python scripts/eval_run.py --results run.json --qrels qrels.json --k-values 5,10,20,100

  # Per-query 分析
  python scripts/eval_run.py --results run.json --qrels qrels.json --per-query

  # 分級相關性評估 (nDCG)
  python scripts/eval_run.py --results run.json --qrels qrels.json --relevance grades.json

  # 輸出至 CSV
  python scripts/eval_run.py --results run.json --qrels qrels.json --output eval.csv
  ```

#### 📊 示範資料
- **範例結果** `datasets/mini/sample_results.json`
  - 3 個查詢的檢索結果
  - 每個查詢回傳 10 個文件
- **範例 Qrels** `datasets/mini/sample_qrels.json`
  - 對應的相關性判斷
  - 二元相關性格式

#### ✅ 完整測試套件
- **測試檔案** (44 個測試案例全部通過)
  - `tests/test_metrics.py` (44 tests, 73% coverage)
    - `TestPrecisionRecall` (6 tests)
      - 精確率與召回率計算
      - 邊界條件測試
    - `TestFMeasure` (4 tests)
      - F1, F2, F0.5 測試
      - Beta 參數測試
    - `TestPrecisionAtK` (4 tests)
      - P@k 計算
      - k 值超過結果數量
    - `TestRecallAtK` (2 tests)
      - R@k 計算
    - `TestAveragePrecision` (5 tests)
      - 完美排序、交錯排序、最差排序
      - 無相關文件測試
    - `TestMeanAveragePrecision` (3 tests)
      - 多查詢 MAP 計算
    - `TestReciprocalRank` (4 tests)
      - 不同排名的 RR 計算
    - `TestMeanReciprocalRank` (1 test)
      - 多查詢 MRR 計算
    - `TestDCG` (3 tests)
      - DCG 計算與驗證
    - `TestNDCG` (4 tests)
      - 完美/最差排序 nDCG
      - 正規化驗證
    - `TestEvaluateQuery` (2 tests)
      - 單查詢完整評估
      - 分級相關性評估
    - `TestEvaluateRun` (2 tests)
      - 批次評估
    - `TestEdgeCases` (4 tests)
      - 空結果、空相關、無重疊、完全重疊

- **測試結果**
  - ✅ 44/44 測試通過 (100% pass rate)
  - ✅ 執行時間：3.71 秒
  - ✅ 覆蓋率：73% (186 statements, 50 missed)
    - 未覆蓋：demo 函式、部分錯誤處理

### 技術特性 *Technical Highlights*

#### 評估指標設計
- **二元 vs 分級相關性**：同時支援兩種評估模式
- **排序感知**：AP, MAP, nDCG 考慮文件順序
- **位置折扣**：nDCG 使用對數折扣函數
- **標準化**：nDCG 正規化至 [0, 1]

#### AP 與 MAP 計算
- **Average Precision 公式**：
  ```
  AP = (1/R) × Σ(P(k) × rel(k))
  其中 R = 相關文件總數
       P(k) = 前k個文件的精確率
       rel(k) = 第k個文件是否相關
  ```
- **MAP**: 多個查詢的 AP 平均值
- **用途**: IR 系統排序品質的標準指標

#### nDCG 計算
- **DCG 公式**：
  ```
  DCG@K = Σ(i=1 to k) (2^rel(i) - 1) / log₂(i + 1)
  ```
- **IDCG**: 理想排序（按相關性降序）的 DCG
- **nDCG**: DCG / IDCG （正規化）
- **特性**:
  - 考慮相關性分級（0-5分）
  - 位置折扣（越後面折扣越大）
  - 正規化使不同查詢可比較

#### 效能指標
- **Precision/Recall**: O(k) where k = 檢索文件數
- **AP**: O(k) 單次掃描計算
- **MAP**: O(Q×k) where Q = 查詢數
- **nDCG**: O(k×log(k)) 需排序計算 IDCG
- **空間複雜度**: O(1) 串流計算

### 應用場景 *Use Cases*

1. **IR 系統評估** - 使用 MAP, nDCG 比較系統性能
2. **參數調校** - 觀察 P@k, R@k 曲線選擇最佳參數
3. **排序品質分析** - 使用 AP 評估單查詢排序品質
4. **學術研究** - 標準評估指標用於論文實驗
5. **TREC 競賽** - 符合 TREC 格式的評估工具

### 範例 *Examples*

#### Python API 使用

```python
from src.ir.eval.metrics import Metrics

metrics = Metrics()

# 二元相關性評估
retrieved = [1, 2, 3, 4, 5]
relevant = {1, 3, 5}

p = metrics.precision(retrieved, relevant)
r = metrics.recall(retrieved, relevant)
f1 = metrics.f_measure(p, r)
ap = metrics.average_precision(retrieved, relevant)

print(f"Precision: {p:.3f}")  # 0.600
print(f"Recall: {r:.3f}")     # 1.000
print(f"F1: {f1:.3f}")        # 0.750
print(f"AP: {ap:.3f}")        # 0.756

# 分級相關性評估 (nDCG)
relevance_scores = {1: 3, 2: 0, 3: 2, 4: 0, 5: 3}
ndcg_5 = metrics.ndcg_at_k(retrieved, relevance_scores, k=5)
print(f"nDCG@5: {ndcg_5:.3f}")  # 0.868

# 多查詢評估 (MAP)
results = {
    'q1': [1, 2, 3, 4],
    'q2': [5, 6, 7, 8],
}
qrels = {
    'q1': {1, 3},
    'q2': {6, 8},
}
map_score = metrics.mean_average_precision(results, qrels)
print(f"MAP: {map_score:.3f}")
```

#### 完整評估工作流

```python
# 單查詢完整評估
eval_result = metrics.evaluate_query(
    retrieved=[1, 2, 3, 4, 5],
    relevant={1, 3, 5},
    relevance_scores={1: 3, 2: 0, 3: 2, 4: 0, 5: 3},
    k_values=[3, 5, 10]
)

for metric_name, score in sorted(eval_result.items()):
    print(f"{metric_name}: {score:.3f}")

# 輸出:
# ap: 0.756
# f1: 0.750
# ndcg@3: 0.658
# ndcg@5: 0.868
# p@3: 0.667
# p@5: 0.600
# precision: 0.600
# ...
```

#### 批次評估

```python
# 多查詢批次評估
results = {
    'q1': [1, 2, 3, 4, 5],
    'q2': [6, 7, 8, 9, 10],
    'q3': [11, 12, 13, 14, 15]
}

qrels = {
    'q1': {1, 3, 5},
    'q2': {7, 9},
    'q3': {11, 15}
}

# 批次評估
aggregated = metrics.evaluate_run(
    results, qrels, k_values=[5, 10, 20]
)

print(f"MAP: {aggregated['map']:.3f}")
print(f"MRR: {aggregated['mrr']:.3f}")
print(f"P@10: {aggregated['p@10']:.3f}")
```

### CLI 使用範例

```bash
# 基本評估
python scripts/eval_run.py \
    --results datasets/mini/sample_results.json \
    --qrels datasets/mini/sample_qrels.json

# 輸出:
# ============================================================
# Aggregated Evaluation Results
# ============================================================
# ap                  : 0.581799
# f1                  : 0.529915
# map                 : 0.581799
# mrr                 : 0.833333
# p@10                : 0.366667
# p@5                 : 0.466667
# ...

# Per-query 詳細評估
python scripts/eval_run.py \
    --results datasets/mini/sample_results.json \
    --qrels datasets/mini/sample_qrels.json \
    --per-query

# 輸出至 CSV
python scripts/eval_run.py \
    --results datasets/mini/sample_results.json \
    --qrels datasets/mini/sample_qrels.json \
    --output evaluation_results.csv

# 自訂 k 值
python scripts/eval_run.py \
    --results datasets/mini/sample_results.json \
    --qrels datasets/mini/sample_qrels.json \
    --k-values 3,5,10,20,100
```

### 已知限制 *Known Limitations*

1. **覆蓋率**: 73%（目標 80%+）
   - 未覆蓋：demo 函式、CLI 載入函式、部分錯誤處理
   - 計劃增加 CLI 整合測試

2. **進階指標**: 未實作部分進階指標
   - 已實作：P, R, F1, AP, MAP, RR, MRR, DCG, nDCG
   - 未實作：ERR, RBP, bpref, infAP
   - 可依需求擴充

3. **統計顯著性**: 未實作統計檢定
   - 未來可加入 paired t-test, Wilcoxon signed-rank test
   - 用於判斷系統改進是否顯著

4. **大規模評估**: 未針對大規模評估最佳化
   - 目前適用於數千至數萬查詢
   - 超大規模需考慮分散式計算

### 整合性說明

**與 Phase 4 (VSM) 整合**：
- VSM 的 `search()` 回傳格式可直接用於評估
- `VSMResult` 的 `doc_ids` 對應 `retrieved`
- 整合範例:
  ```python
  vsm_result = vsm.search("query", topk=10)
  ap = metrics.average_precision(vsm_result.doc_ids, relevant_set)
  ```

**與 Phase 3 (Boolean Retrieval) 整合**：
- Boolean 的 `QueryResult` 也可用於評估
- 布林檢索主要評估 Precision/Recall
- 排序後可評估 AP/MAP

**準備 Phase 6 (Query Expansion)**：
- 使用評估指標比較擴展前後效果
- MAP/nDCG 作為查詢擴展的目標函數
- 評估擴展策略的有效性

### 下一步計劃 *Next Steps*

- [ ] 提升測試覆蓋率至 80%+ (增加 CLI 測試)
- [ ] 實作進階指標 (ERR, RBP, bpref)
- [ ] 實作統計顯著性檢定
- [ ] 新增 Precision-Recall 曲線繪製
- [ ] 實作交叉驗證評估
- [ ] 整合至 Phase 6 查詢擴展實驗

---

## [0.4.0] - 2025-11-12

### 新增 *Added*

#### 📊 詞項權重計算 (Term Weighting)
- **核心實作** `src/ir/index/term_weighting.py` (137 statements, 52% coverage)
  - `TermWeighting` 類別：TF-IDF 權重計算引擎
  - 支援多種 TF (Term Frequency) 方案
  - 支援多種 IDF (Inverse Document Frequency) 方案
  - 向量正規化與餘弦相似度計算
  - 文件頻率（DF）統計與 IDF 預計算
  - Bonus: BM25 評分演算法

- **TF 計算方案**
  - `'n'` (natural): 原始詞頻 (raw count)
  - `'l'` (logarithmic): 1 + log₁₀(count)
  - `'a'` (augmented): 0.5 + 0.5 × (count / max_count)
  - `'b'` (boolean): 1 if present, 0 otherwise

- **IDF 計算方案**
  - `'n'` (none): 不使用 IDF (固定為 1.0)
  - `'t'` (standard): log₁₀(N / df)
  - `'p'` (probabilistic): log₁₀((N - df) / df)

- **核心功能**
  - `build_from_index()`: 從倒排索引建立統計資料 (O(V))
  - `tf()`: 計算 TF 值 (O(1) for n/l/b, O(|doc|) for a)
  - `idf_value()`: 獲取 IDF 值 (O(1) 查表)
  - `tf_idf()`: 計算 TF-IDF 權重
  - `vectorize()`: 將文件轉為加權向量 (O(|doc|))
  - `cosine_similarity()`: 餘弦相似度 (O(min(|v1|, |v2|)))
  - `euclidean_distance()`: 歐氏距離 (O(|v1| + |v2|))
  - `get_top_terms()`: Top-K 高權重詞項 (O(|doc| × log(k)))
  - `bm25_score()`: BM25 評分 (bonus implementation)

#### 🔢 向量空間模型 (Vector Space Model)
- **核心實作** `src/ir/retrieval/vsm.py` (146 statements, 60% coverage)
  - `VectorSpaceModel` 類別：向量空間檢索引擎
  - 文件與查詢的向量表示
  - 基於餘弦相似度的文件排序
  - 預計算文件向量以提升查詢效率
  - Top-K 堆積最佳化檢索
  - 彈性權重方案配置（ltc/lnc）

- **權重方案 (Weighting Schemes)**
  - 標準方案表示法：`[tf][idf][norm]` (三字元碼)
  - 文件預設：`ltc` (log TF, standard IDF, cosine norm)
  - 查詢預設：`lnc` (log TF, no IDF, cosine norm)
  - 可自訂其他方案：`atc`, `nnc`, `bnn` 等

- **核心功能**
  - `build_index()`: 建立索引與預計算文件向量 (O(T + D×V))
  - `search()`: 執行 VSM 查詢檢索 (O(|q| + C + k×log(k)))
    - 查詢向量化
    - 候選文件篩選（僅包含查詢詞項的文件）
    - 餘弦相似度計算
    - Top-K 堆積排序（使用 heapq.nlargest）
  - `set_weighting_scheme()`: 設定權重方案
  - `get_document_vector()`: 獲取文件向量
  - `similarity()`: 計算文件間相似度 (O(min(V₁, V₂)))
  - `get_similar_documents()`: 尋找相似文件 (O(D×V + k×log(k)))

- **資料結構**
  - `VSMResult`: 查詢結果容器
    - `doc_ids`: 排序後的文件 ID 列表
    - `scores`: 文件相似度分數字典
    - `query`: 原始查詢字串
    - `num_results`: 結果數量
  - 文件向量：`{term: weight}` 稀疏表示法
  - 查詢向量：`{term: weight}` 稀疏表示法

#### 🛠️ CLI 命令列工具
- **VSM 檢索工具** `scripts/vsm_search.py` (85 行)
  - `--build`: 從 JSON 文件建立 VSM 索引
    - 載入文件與 metadata
    - 建立倒排索引
    - 計算 TF-IDF 權重
    - 預計算文件向量
    - 儲存索引至檔案
  - `--search`: 執行 VSM 檢索查詢
    - 載入已建立的索引
    - 執行查詢並計算相似度
    - 顯示 Top-K 排序結果
    - 顯示文件標題與分數
  - `--index`: 指定索引檔案路徑（必要參數）
  - `--topk`: 限制回傳結果數量（預設 10）

- **使用範例**
  ```bash
  # 建立索引
  python scripts/vsm_search.py --build \
      --input datasets/mini/ir_documents.json \
      --index vsm_index.json

  # 執行查詢
  python scripts/vsm_search.py --search "information retrieval" \
      --index vsm_index.json --topk 10

  # 查詢檢索模型
  python scripts/vsm_search.py --search "boolean vector model" \
      --index vsm_index.json
  ```

#### ✅ 完整測試套件
- **測試檔案** (15 個測試案例全部通過)
  - `tests/test_term_weighting.py` (8 tests)
    - `TestTFCalculation` (3 tests)
      - 自然 TF 計算
      - 對數 TF 計算
      - 布林 TF 計算
    - `TestIDFCalculation` (2 tests)
      - 標準 IDF 計算
      - 無 IDF 模式
    - `TestVectorization` (1 test)
      - 完整向量化流程（TF-IDF + 正規化）
    - `TestCosineSimilarity` (2 tests)
      - 相同向量相似度（應為 1.0）
      - 正交向量相似度（應為 0.0）

  - `tests/test_vsm.py` (7 tests)
    - `TestVSMBasic` (3 tests)
      - 索引建立測試
      - 基本查詢測試
      - 空查詢邊界測試
    - `TestRanking` (2 tests)
      - 結果排序驗證（降序）
      - Top-K 限制測試
    - `TestDocumentSimilarity` (2 tests)
      - 文件間相似度計算
      - 自相似度測試（應為 1.0）

- **測試結果**
  - ✅ 15/15 測試通過 (100% pass rate)
  - ✅ 執行時間：3.29 秒
  - ✅ 覆蓋率統計：
    - `term_weighting.py`: 52% (137 statements)
    - `vsm.py`: 60% (146 statements)
    - 整體 Phase 4 模組：56% 平均覆蓋率

### 技術特性 *Technical Highlights*

#### 向量空間模型設計
- **向量表示**: 稀疏向量 `{term: weight}` (僅儲存非零權重)
- **權重方案**: 彈性的三字元方案表示法 (tf-idf-norm)
- **預計算最佳化**: 文件向量預先計算並快取
- **Top-K 檢索**: 使用 min-heap 實現高效 Top-K (O(n + k×log(k)))

#### TF-IDF 計算
- **TF 變種**: 支援 natural, log, augmented, boolean 四種
- **IDF 變種**: 支援 none, standard, probabilistic 三種
- **正規化**: 支援 cosine (L2) 與 none 兩種
- **組合彈性**: 任意 TF-IDF-Norm 組合（如 ltc, lnc, atc, bnn）

#### 餘弦相似度計算
- **公式**: cos(θ) = (v₁ · v₂) / (||v₁|| × ||v₂||)
- **稀疏最佳化**: 僅計算共同詞項的點積
- **向量已正規化**: 若使用 cosine norm，則 ||v|| = 1
- **時間複雜度**: O(min(|v₁|, |v₂|)) 透過稀疏表示

#### 查詢處理流程
1. **查詢詞元化**: 使用與建立索引相同的 tokenizer
2. **計算查詢 TF**: 統計查詢詞頻
3. **查詢向量化**: 使用 lnc 或 ltc 方案
4. **候選文件篩選**: 僅取包含查詢詞項的文件
5. **相似度計算**: 計算查詢與候選文件的餘弦相似度
6. **Top-K 排序**: 使用 heapq 回傳前 K 個結果

#### 效能指標
- **索引建立**: O(T + D×V) where T = 總詞數, D = 文件數, V = 詞彙大小
- **文件向量計算**: O(D×V_avg) where V_avg = 平均文件詞彙數
- **查詢處理**: O(|q| + C + k×log(k)) where |q| = 查詢長度, C = 候選文件數, k = Top-K
- **相似度計算**: O(min(|v₁|, |v₂|)) 稀疏向量點積
- **空間複雜度**: O(D×V_avg) 儲存文件向量

### 應用場景 *Use Cases*

1. **學術文獻搜尋** - TF-IDF 排序提升檢索品質
2. **文件推薦系統** - 基於餘弦相似度的相似文件推薦
3. **資訊檢索評估** - 標準 VSM baseline
4. **查詢擴展準備** - 提供相似文件基礎（for Rocchio）
5. **混合檢索系統** - 結合布林檢索與 VSM 排序

### 範例 *Examples*

#### Python API 使用

```python
from src.ir.retrieval.vsm import VectorSpaceModel

# 建立索引
documents = [
    "information retrieval systems are important",
    "vector space model for information retrieval",
    "boolean retrieval model",
    "tf idf weighting scheme",
    "cosine similarity for ranking documents"
]

vsm = VectorSpaceModel()
vsm.build_index(documents)

# 執行查詢
result = vsm.search("information retrieval", topk=3)
print(f"Found {result.num_results} results")
for i, doc_id in enumerate(result.doc_ids, 1):
    score = result.scores[doc_id]
    print(f"{i}. Document {doc_id}: {score:.4f}")

# 計算文件相似度
sim = vsm.similarity(0, 1)
print(f"Doc 0 vs Doc 1: {sim:.4f}")

# 尋找相似文件
similar = vsm.get_similar_documents(0, topk=3)
for doc_id, sim in similar:
    print(f"Doc {doc_id}: {sim:.4f}")
```

#### 自訂權重方案

```python
from src.ir.retrieval.vsm import VectorSpaceModel

vsm = VectorSpaceModel()
vsm.build_index(documents)

# 使用 augmented TF + standard IDF + cosine norm
vsm.set_weighting_scheme(doc_scheme='atc', query_scheme='atc')

# 使用 boolean TF (無 IDF 無正規化)
vsm.set_weighting_scheme(doc_scheme='bnn', query_scheme='bnn')

# 標準方案（文件 ltc, 查詢 lnc）
vsm.set_weighting_scheme(doc_scheme='ltc', query_scheme='lnc')
```

#### 詞項權重計算

```python
from src.ir.index.term_weighting import TermWeighting
from src.ir.index.inverted_index import InvertedIndex

# 建立索引
inv_index = InvertedIndex()
inv_index.build(["hello world", "world peace", "hello peace world"])

# 建立 TermWeighting
tw = TermWeighting()
tw.build_from_index(inv_index)

# 計算 TF-IDF
doc = {"hello": 3, "world": 2}
tfidf = tw.tf_idf("hello", doc, tf_scheme='l', idf_scheme='t')
print(f"TF-IDF: {tfidf:.4f}")

# 向量化文件
vec = tw.vectorize(doc, tf_scheme='l', idf_scheme='t', normalize='c')
print(f"Vector: {vec}")

# 餘弦相似度
v1 = {"hello": 0.6, "world": 0.8}
v2 = {"hello": 0.8, "world": 0.6}
sim = tw.cosine_similarity(v1, v2)
print(f"Similarity: {sim:.4f}")
```

### CLI 使用範例

```bash
# 建立索引
python scripts/vsm_search.py --build \
    --input datasets/mini/ir_documents.json \
    --index ir_vsm_index.json

# Output:
# Loading documents from datasets/mini/ir_documents.json...
# Index built: 15 documents
# Vocabulary: 127 terms
# Index saved to ir_vsm_index.json

# 執行查詢
python scripts/vsm_search.py --search "information retrieval" \
    --index ir_vsm_index.json --topk 5

# Output:
# Query: information retrieval
# Found 5 results
#
# 1. Document 0 (score: 0.8532)
#    Title: Information Retrieval Systems
# 2. Document 1 (score: 0.7124)
#    Title: Boolean Retrieval Model
# ...

# 查詢檢索模型
python scripts/vsm_search.py --search "vector space model" \
    --index ir_vsm_index.json

# 查詢評估指標
python scripts/vsm_search.py --search "precision recall evaluation" \
    --index ir_vsm_index.json --topk 3
```

### 已知限制 *Known Limitations*

1. **覆蓋率不足**：52-60%（目標 80%）
   - 未覆蓋：demo 函式、部分錯誤處理、邊界情況
   - 計劃增加更多單元測試

2. **權重方案**：未實作所有理論方案
   - 已實作：n, l, a, b (TF) + n, t, p (IDF) + n, c (Norm)
   - 未實作：d (document frequency TF), L (avg TF normalization)

3. **查詢擴展**：尚未整合 Rocchio 演算法
   - 目前僅支援單輪查詢
   - v0.6.0 將實作擬相關回饋與查詢擴展

4. **效能最佳化**：未實作進階索引壓縮
   - 目前使用原始稀疏向量
   - 未來可考慮向量量化或 LSH

5. **記憶體使用**：預計算所有文件向量
   - 大規模資料集可能佔用較多記憶體
   - 未來考慮 lazy evaluation 或分塊載入

### 整合性說明

**與 Phase 3 (Boolean Retrieval) 整合**：
- VSM 使用 Phase 3 的 `InvertedIndex` 作為基礎索引
- VSM 的 tokenizer 與 InvertedIndex 相容
- 可結合布林查詢篩選 + VSM 排序（未來功能）

**準備 Phase 5 (Evaluation Metrics)**：
- VSM 提供排序結果（doc_ids + scores）
- 結果格式可直接用於 Precision/Recall/MAP/nDCG 計算
- 為評估實驗提供 baseline 系統

**準備 Phase 6 (Query Expansion)**：
- VSM 提供 `get_similar_documents()` 用於擬相關回饋
- 文件向量可用於 Rocchio 演算法的向量運算
- 權重方案可調整以配合查詢擴展策略

### 下一步計劃 *Next Steps*

- [ ] 提升測試覆蓋率至 80%+
- [ ] 實作更多 TF-IDF 變種（如 document frequency TF）
- [ ] 實作向量空間模型的其他相似度度量（如 Dice, Jaccard）
- [ ] 優化大規模資料集的記憶體使用
- [ ] 整合 Rocchio 查詢擴展（v0.6.0）
- [ ] 實作混合檢索（Boolean + VSM ranking）

---

## [0.3.0] - 2025-11-12

### 新增 *Added*

#### 🗂️ 倒排索引 (Inverted Index)
- **核心實作** `src/ir/index/inverted_index.py` (160 statements, 69% coverage)
  - `InvertedIndex` 類別：高效文件檢索索引
  - 詞項到文件的映射 (term → [(doc_id, term_freq), ...])
  - 詞頻統計（TF）與文件頻率（DF）
  - 布林操作：交集（AND）、聯集（OR）、差集（NOT）
  - 可自訂 tokenizer（預設：小寫 + 非字母數字分割）
  - JSON 格式儲存/載入
  - 批次文件處理與增量索引

- **核心功能**
  - `build()`: 批次建立索引 (O(T) 時間複雜度)
  - `add_document()`: 增量添加文件 (O(n) 單文件)
  - `get_postings()`: 獲取詞項的 posting list (O(1) 平均)
  - `intersect()`: 合併演算法交集 (O(n+m) 時間)
  - `union()`: 聯集操作 (O(n+m) 時間)
  - `negate()`: 差集操作 (O(D+k) 時間)
  - `get_stats()`: 索引統計資訊

#### 📍 位置索引 (Positional Index)
- **核心實作** `src/ir/index/positional_index.py` (184 statements, 60% coverage)
  - `PositionalIndex` 類別：位置感知索引
  - 詞項到位置的映射 (term → {doc_id: [pos1, pos2, ...]})
  - 支援詞組查詢（phrase query）
  - 支援鄰近查詢（proximity search）
  - 支援視窗查詢（window query）

- **查詢功能**
  - `phrase_query()`: 精確詞組匹配 (O(k * min(p1,p2)))
  - `proximity_query()`: 鄰近搜尋 NEAR/k (O(k * p1 * p2))
  - `window_query()`: 視窗內多詞匹配 (O(k * w))
  - `get_positions()`: 獲取詞項位置 (O(1) 平均)

#### 🔍 布林查詢引擎 (Boolean Query Engine)
- **核心實作** `src.ir.retrieval/boolean.py` (191 statements, 65% coverage)
  - `BooleanQueryEngine` 類別：完整查詢處理引擎
  - 支援 AND, OR, NOT 布林操作
  - 支援詞組查詢 ("exact phrase")
  - 支援括號分組 ((term1 OR term2) AND term3)
  - 查詢解析器（Shunting Yard 演算法）
  - 查詢最佳化（term ordering）
  - 結果排序（基於 TF）

- **查詢語法**
  - 布林操作: `information AND retrieval`
  - 聯集操作: `boolean OR vector`
  - 差集操作: `NOT extraction`
  - 詞組查詢: `"information retrieval"`
  - 複雜查詢: `(boolean OR vector) AND model AND NOT extraction`
  - 括號優先級: `(term1 OR term2) AND (term3 OR term4)`

- **查詢處理流程**
  1. 詞組提取（保留引號內空格）
  2. 詞元化（識別操作符與括號）
  3. 中綴轉後綴（Reverse Polish Notation）
  4. 堆疊式求值（stack-based evaluation）
  5. 可選結果排序

#### 🛠️ CLI 命令列工具
- **檢索工具** `scripts/boolean_search.py` (450+ 行)
  - `--build`: 從文件建立索引
    - 支援純文字檔（一行一文件）
    - 支援 JSON 檔（含 metadata）
    - 自動建立倒排索引與位置索引
  - `--query`: 執行檢索查詢
    - 布林查詢支援
    - 詞組查詢支援
    - 結果排序選項
  - `--interactive`: 互動式查詢模式
    - REPL 介面
    - 內建 help 與 stats 指令
    - 即時查詢執行

- **使用範例**
  ```bash
  # 建立索引
  python scripts/boolean_search.py --build --input docs.txt --index my_index.json

  # 簡單查詢
  python scripts/boolean_search.py --query "information AND retrieval" --index my_index.json

  # 詞組查詢
  python scripts/boolean_search.py --query '"vector space model"' --index my_index.json

  # 複雜查詢
  python scripts/boolean_search.py --query "(boolean OR vector) AND model" --index my_index.json

  # 互動模式
  python scripts/boolean_search.py --interactive --index my_index.json
  ```

#### 📊 示範資料集
- **IR 文件集** `datasets/mini/ir_documents.json`
  - 15 篇資訊檢索相關文件
  - 涵蓋主題：
    - 檢索模型（Boolean, VSM）
    - 索引結構（Inverted Index, Positional Index）
    - 排序機制（TF-IDF, PageRank）
    - 評估指標（Precision, Recall）
    - 進階技術（Query Expansion, Clustering, Summarization）
    - 特殊案例（三聚氰胺事件、中文 IR）
  - 每份文件包含：text, title, doc_id, category

#### ✅ 完整測試套件
- **測試檔案** (31 個測試案例全部通過)
  - `tests/test_inverted_index.py` (13 tests)
    - 基礎索引建立與查詢
    - 布林操作（交集、聯集、差集）
    - 統計計算與儲存/載入
    - 邊界條件測試

  - `tests/test_positional_index.py` (9 tests)
    - 位置索引建立
    - 詞組查詢測試
    - 鄰近查詢測試
    - 儲存/載入功能

  - `tests/test_boolean.py` (9 tests)
    - 簡單查詢（單詞、AND、OR、NOT）
    - 詞組查詢測試
    - 複雜查詢（巢狀操作）
    - 結果排序測試

- **測試結果**
  - ✅ 31/31 測試通過 (100% pass rate)
  - ✅ 執行時間：2.88 秒
  - ✅ 覆蓋率統計：
    - `inverted_index.py`: 69% (160 statements)
    - `positional_index.py`: 60% (184 statements)
    - `boolean.py`: 65% (191 statements)
    - 整體 Phase 3 模組：65% 平均覆蓋率

### 技術特性 *Technical Highlights*

#### 倒排索引設計
- **資料結構**: `{term: [(doc_id, term_freq), ...]}`
- **排序**: Posting lists 按 doc_id 排序（利於合併）
- **最佳化**: 合併演算法 (merge algorithm) 用於交集/聯集
- **空間複雜度**: O(V + P) where V=vocabulary, P=postings

#### 位置索引設計
- **資料結構**: `{term: {doc_id: [position1, position2, ...]}}`
- **詞組查詢**: 檢查連續位置 (position + 1, position + 2, ...)
- **鄰近查詢**: 計算位置距離 |pos1 - pos2| <= k
- **視窗查詢**: 滑動視窗檢查所有詞項

#### 查詢解析演算法
- **Shunting Yard 演算法**: 中綴轉後綴表示法
- **操作符優先級**: NOT(3) > AND(2) > OR(1)
- **堆疊求值**: 後綴表達式求值
- **詞組處理**: 預先提取並使用 placeholder

#### 效能指標
- **倒排索引建立**: O(T) where T = 總詞數
- **詞項查詢**: O(1) 平均（雜湊表查找）
- **交集操作**: O(n + m) 合併演算法
- **詞組查詢**: O(k * min(p1, p2)) where k = 候選文件數
- **鄰近查詢**: O(k * p1 * p2) 雙層迴圈

### 應用場景 *Use Cases*

1. **文件檢索系統** - 傳統搜尋引擎核心
2. **學術文獻搜尋** - 支援精確詞組與複雜查詢
3. **法律文件檢索** - 布林查詢適合專業檢索
4. **企業知識庫** - 結構化查詢支援
5. **程式碼搜尋** - 精確匹配與鄰近搜尋

### 範例 *Examples*

```python
from src.ir.index.inverted_index import InvertedIndex
from src.ir.index.positional_index import PositionalIndex
from src.ir.retrieval.boolean import BooleanQueryEngine

# 建立索引
documents = [
    "information retrieval systems",
    "boolean retrieval models",
    "vector space models"
]

inv_index = InvertedIndex()
inv_index.build(documents)

pos_index = PositionalIndex()
pos_index.build(documents)

# 建立查詢引擎
engine = BooleanQueryEngine(inv_index, pos_index)

# 簡單查詢
result = engine.query("information AND retrieval")
print(result.doc_ids)  # [0]

# 詞組查詢
result = engine.query('"information retrieval"')
print(result.doc_ids)  # [0]

# 複雜查詢
result = engine.query("(boolean OR vector) AND models")
print(result.doc_ids)  # [1, 2]
```

### CLI 使用範例

```bash
# 建立索引
python scripts/boolean_search.py --build \
    --input datasets/mini/ir_documents.json \
    --index ir_index.json

# 簡單查詢
python scripts/boolean_search.py \
    --query "information AND retrieval" \
    --index ir_index.json

# 詞組查詢
python scripts/boolean_search.py \
    --query '"vector space model"' \
    --index ir_index.json

# 複雜查詢
python scripts/boolean_search.py \
    --query '(boolean OR vector) AND model AND NOT extraction' \
    --index ir_index.json --rank

# 互動模式
python scripts/boolean_search.py --interactive --index ir_index.json
```

### 已知限制 *Known Limitations*

1. **查詢解析器**：不支援萬用字元查詢（wildcard queries）
   - 計劃在 v0.4.0 實作

2. **排序機制**：目前僅基於簡單 TF 排序
   - v0.4.0 將實作完整 TF-IDF 與 cosine similarity

3. **測試覆蓋率**：60-69%（目標 80%）
   - 未覆蓋：demo 函式、部分錯誤處理、儲存/載入邊界情況

4. **記憶體使用**：位置索引佔用較大記憶體
   - 未來考慮壓縮技術（如 gap encoding）

### 下一步計劃 *Next Steps*

- [ ] 提升測試覆蓋率至 80%+
- [ ] 實作萬用字元查詢 (wildcard queries)
- [ ] 實作拼寫校正 (spelling correction)
- [ ] 優化記憶體使用（posting list 壓縮）
- [ ] 新增查詢日誌記錄

---

## [0.2.0] - 2025-11-12

### 新增 *Added*

#### 🎵 CSoundex 中文諧音編碼模組
- **核心實作** `src/ir/text/csoundex.py` (208 行，78% 測試覆蓋率)
  - `CSoundex` 類別：中文語音編碼引擎
  - 支援漢字轉拼音（pypinyin + 內建字典）
  - 拼音正規化（去聲調、分離聲母韻母）
  - 聲母/韻母分群編碼（基於發音部位與方法）
  - 編碼格式：`[首字母][聲母碼][韻母碼][聲調碼(可選)]`
  - LRU 快取機制（maxsize=10000）
  - 批次編碼支援（`encode_batch()`）

- **編碼功能**
  - `encode_character()`: 單字元編碼（O(1) 快取時間）
  - `encode()`: 文字串編碼（支援中英混合、標點處理）
  - `encode_batch()`: 批次編碼（O(n*m) 時間複雜度）
  - `get_pinyin()`: 拼音查詢（字典優先，pypinyin 備用）
  - `normalize_pinyin()`: 拼音正規化與組件分離

- **相似度計算**
  - `similarity()`: 語音相似度計算（3 種模式）
    - `exact`: 精確匹配（二元判斷）
    - `fuzzy`: 模糊匹配（字元級相似度，預設）
    - `weighted`: 位置加權匹配（前面字元權重較高）
  - `find_similar()`: 相似文本搜尋（閾值過濾 + Top-K）

#### 📖 拼音字典資源
- **基礎字典** `datasets/lexicon/basic_pinyin.tsv`
  - 格式：`字元 TAB 拼音(含聲調)`
  - 內容：500+ 常用漢字（百家姓、常用詞、IR 術語）
  - 涵蓋：姓名（張王李趙）、時間、方位、數字、顏色
  - 特殊詞彙：「三聚氰胺」事件相關字、資訊檢索術語
  - 同音字範例：張/章/彰 (zhang1)、李/理/裡 (li3)
  - 異體字範例：裏/裡、台/臺

#### 🛠️ CLI 命令列工具
- **編碼工具** `scripts/csoundex_encode.py` (400+ 行)
  - `--text`: 直接編碼文字
  - `--file`: 批次處理檔案
  - `--stdin`: 管線輸入模式
  - `--tone`: 包含聲調資訊
  - `--show-original`: 顯示原文與編碼
  - `--output`: 指定輸出檔案

- **相似度搜尋**
  - `--similar <查詢>`: 尋找相似文本
  - `--threshold <閾值>`: 設定相似度門檻（預設 0.6）
  - `--topk <數量>`: 限制回傳結果數量

- **進階功能**
  - `--matrix <輸出檔>`: 計算相似度矩陣（CSV 格式）
  - `--cache-info`: 顯示快取統計資訊
  - `--config`: 自訂配置檔路徑
  - `--lexicon`: 自訂字典檔路徑
  - `--verbose`: 詳細輸出模式

#### ✅ 完整測試套件
- **測試檔案** `tests/test_csoundex.py` (600+ 行，43 個測試案例)
  - ✅ 所有測試通過（43/43 passed）
  - ✅ 測試覆蓋率：78% (208 statements, 45 missed)
  - ✅ 執行時間：3.30 秒

- **測試類別**
  1. `TestBasicEncoding` (4 tests) - 基礎編碼功能
     - 單字元編碼、多字元編碼、聲調處理
     - 常見姓氏編碼驗證（王李趙錢孫周吳鄭）

  2. `TestHomophoneMatching` (3 tests) - 同音字匹配
     - 精確同音字（張/章/彰 → Z89）
     - 不同聲調變化（詩/時/史/試）
     - 李/理/裡 同音群組

  3. `TestVariantCharacters` (1 test) - 異體字處理
     - 繁簡異體（裏/裡、台/臺）

  4. `TestMixedText` (4 tests) - 混合文本處理
     - 純中文、中英混合、標點符號、數字

  5. `TestRealWorldExamples` (3 tests) - 實際應用場景
     - 三聚氰胺事件（S99 J75...）
     - 資訊檢索術語
     - 中文姓名編碼

  6. `TestSimilarity` (5 tests) - 相似度計算
     - 精確匹配、同音相似度、部分相似度
     - 加權相似度、零相似度

  7. `TestBatchProcessing` (3 tests) - 批次處理
     - 批次編碼、相似文本搜尋、Top-K 限制

  8. `TestEdgeCases` (9 tests) - 邊界條件
     - 空字串、單字元、純空白、純標點
     - 純英文、未知字元

  9. `TestConfiguration` (3 tests) - 配置與初始化
     - 預設配置、自訂配置、字典載入

  10. `TestNormalization` (4 tests) - 拼音正規化
      - 帶聲調、無聲調、大寫、零聲母

  11. `TestPerformance` (3 tests) - 效能與快取
      - 快取驗證、大批次編碼、編碼速度（< 1秒）

  12. `TestIntegration` (3 tests) - 整合測試
      - 姓名匹配工作流、去重工作流、查詢擴展

### 技術特性 *Technical Highlights*

#### 編碼設計
- **聲母分群** (21 聲母 → 10 群組 0-9)
  - 0: 零聲母（純元音）
  - 1: 雙唇音 (b/p)
  - 2: 唇齒音 (f)
  - 3: 雙唇鼻音 (m)
  - 4: 舌尖中音 (d/t)
  - 5: 舌尖鼻音/邊音 (n/l)
  - 6: 舌根音 (g/k/h)
  - 7: 舌面音 (j/q/x)
  - 8: 捲舌音 (zh/ch/sh/r)
  - 9: 平舌音 (z/c/s)

- **韻母分群** (38 韻母 → 10 群組 0-9)
  - 0: 零韻母
  - 1: 主元音 a (a/ia/ua)
  - 2: 主元音 o (o/uo)
  - 3: 主元音 e (e/ie/ue/ve)
  - 4: 元音 i
  - 5: 元音 u
  - 6: 元音 ü (v/u:)
  - 7: 複韻母韻尾 i (ai/ei/ui/uai)
  - 8: 複韻母韻尾 u (ao/ou/iu/iao)
  - 9: 鼻韻母 (an/en/ang/eng/ing/ong/...)

#### 效能指標
- **時間複雜度**
  - 單字元編碼：O(1) (with LRU cache)
  - 文字串編碼：O(n) where n = 字元數
  - 批次編碼：O(n*m) where n = 文本數, m = 平均長度
  - 相似度計算：O(min(len1, len2))

- **空間複雜度**
  - 配置載入：O(1)
  - 字典載入：O(d) where d = 字典大小
  - LRU 快取：O(cache_size) = O(10000)

- **快取效能**
  - 最大快取：10,000 個字元編碼
  - 實測命中率：> 95% (重複字元場景)
  - 快取清除：支援手動清除 (`clear_cache()`)

### 應用場景 *Use Cases*

1. **姓名模糊搜尋** - 處理同音異字姓名
2. **去重系統** - 基於語音相似度的文本去重
3. **查詢擴展** - 自動擴展同音詞彙
4. **拼寫糾錯** - 基於語音的錯誤偵測
5. **語音搜尋** - 支援諧音搜尋功能

### 範例 *Examples*

```python
from src.ir.text.csoundex import CSoundex

# 初始化
csoundex = CSoundex()

# 基礎編碼
csoundex.encode("張三")  # → "Z89 S99"
csoundex.encode("三聚氰胺")  # → "S99 J75 Q79 A91"

# 同音字檢測
code1 = csoundex.encode("張三")
code2 = csoundex.encode("章三")
print(code1 == code2)  # → True (Z89 S99)

# 相似度計算
sim = csoundex.similarity("張偉", "章偉", mode='fuzzy')
print(sim)  # → 1.0 (完全相同語音)

# 尋找相似名稱
candidates = ["張偉", "章偉", "張維", "李偉"]
results = csoundex.find_similar("張偉", candidates, threshold=0.5)
# → [("張偉", 1.0), ("章偉", 1.0), ("張維", 0.5)]
```

### CLI 使用範例

```bash
# 直接編碼
python scripts/csoundex_encode.py --text "信息檢索"

# 批次處理
python scripts/csoundex_encode.py --file names.txt --output codes.txt

# 尋找相似名稱
python scripts/csoundex_encode.py --similar "張偉" --file database.txt --threshold 0.6 --topk 10

# 計算相似度矩陣
python scripts/csoundex_encode.py --file names.txt --matrix similarity.csv
```

### 已知限制 *Known Limitations*

1. **多音字處理**：目前採用「最常見讀音」策略，準確率約 85%
   - 未來版本將支援基於詞彙脈絡的多音字判斷（需要分詞）

2. **字典覆蓋**：基礎字典包含 500+ 常用字
   - 罕見字依賴 pypinyin 備用
   - 可自訂字典擴充

3. **方言支援**：目前僅支援普通話拼音
   - 粵語、閩南語、客家話等方言為實驗性功能（未實作）

4. **測試覆蓋率**：78% (目標 80%)
   - 未覆蓋：錯誤處理邊界、demo 函式

### 下一步計劃 *Next Steps*

- [ ] 提升測試覆蓋率至 85%+
- [ ] 實作基於詞彙的多音字處理
- [ ] 擴充拼音字典至 5000 字
- [ ] 優化相似度計算演算法
- [ ] 新增簡繁轉換支援

---

## [0.1.0] - 2025-11-12

### 新增 *Added*

#### 📁 專案架構
- 建立完整專案目錄結構：
  - `src/ir/{text,index,retrieval,eval,ranking,cluster,summarize}/` - 核心模組
  - `scripts/` - CLI 工具目錄
  - `tests/` - 測試目錄
  - `configs/` - 設定檔目錄
  - `datasets/{mini,lexicon}/` - 資料集目錄
  - `logs/` - 日誌目錄
- 建立所有模組的 `__init__.py` 檔案（共 10 個）
- 新增 `CLAUDE.md` - Claude Code 開發指引
- 新增 `README.md` - 英文專案簡介（Quick Start, API Examples）
- 新增 `docs/README.md` - 專案總覽文件（繁體中文，15,000 字）
- 新增 `docs/CHANGELOG.md` - 變更紀錄（本檔案）
- 新增 `docs/PROJECT_ROADMAP.md` - 完整開發路線圖（v0.1.0 - v1.0.0）

#### 📚 文件系統
- 建立 `docs/exams/midterm/` 期中考準備資料
  - `OUTLINE.md` - 結構化題綱（7 個題目）
  - `DRAFT.md` - 完整擬答（12,000 字）
- 建立 `docs/hw/template/` 作業報告範本
  - `REPORT_TEMPLATE.md` - 標準報告格式（9 章節）
- 建立 `docs/project/` 期末專案文件
  - `PROPOSAL.md` - 專案提案（學術搜尋引擎設計）
  - `REPORT.md` - 專案報告範本（25 頁結構）
- 建立 `docs/guides/` 實作指南
  - `IMPLEMENTATION.md` - 詳細實作指南（6 模組，包含程式碼範例）
  - `CSOUNDEX_DESIGN.md` - CSoundex 技術設計文件（20,000 字）
  - `CSOUNDEX.md` - CSoundex 快速指南

#### ⚙️ 設定檔與依賴管理
- 建立 `requirements.txt` - Python 3.10 相容依賴清單
  - 核心：pypinyin 0.49.0, jieba 0.42.1, numpy 1.24.3, scipy 1.10.1
  - 測試：pytest 7.4.0 + 擴充套件（cov, mock, timeout）
  - 開發工具：pylint, black, mypy, flake8
  - 資料處理：pyyaml, tqdm, pandas
- 建立 `configs/csoundex.yaml` - CSoundex 完整配置
  - 聲母分群規則（21 個聲母 → 10 群組）
  - 韻母分群規則（38 個韻母 → 10 群組）
  - 多音字處理策略、編碼模式、快取設定
- 建立 `configs/logging.yaml` - 日誌系統配置
  - 多層級處理器（console, file, error_file, performance_file）
  - 輪轉日誌（10 MB, 3 備份）
  - 模組化日誌記錄器
- 建立 `pytest.ini` - 測試框架配置
  - 測試發現規則（test_*.py）
  - 覆蓋率報告（HTML, term, XML）
  - 自訂標記（unit, integration, performance, slow, csoundex 等）
  - 超時設定（300 秒）
- 建立 `.gitignore` - 完整忽略規則
  - Python 編譯檔案、虛擬環境
  - IDE 設定檔、作業系統暫存檔
  - 測試報告、日誌檔案
  - 資料集檔案（保留 mini/ 與 lexicon/）

#### 🧪 測試架構
- 建立 `tests/` 目錄結構
- 配置 pytest 測試框架（已驗證可運行）
- 準備測試標記系統（unit, integration, performance 等）

#### 🗂️ 資料集準備
- 建立 `datasets/mini/` - 小型測試資料集目錄
- 建立 `datasets/lexicon/` - 詞典資源目錄（待補充 basic_pinyin.tsv）

### ✅ 開發環境設定
- 安裝所有 Python 依賴於 `ai_env` conda 環境
- 驗證 pytest 7.4.0 可正常運行
- 驗證專案結構完整性

---

## [未來版本規劃]

### v0.2.0 - CSoundex 模組（預計 Week 4）
- [ ] 實作 `src/ir/text/csoundex.py`
- [ ] 建立 `scripts/csoundex_encode.py` CLI 工具
- [ ] 撰寫 `tests/test_csoundex.py` 單元測試
- [ ] 建立 `configs/csoundex.yaml` 設定檔
- [ ] 準備 `datasets/lexicon/basic_pinyin.tsv` 拼音字典
- [ ] 撰寫 `docs/guides/CSOUNDEX.md` 詳細文件

**目標功能**：
- 中文字轉拼音
- 拼音正規化（去聲調、小寫化）
- 聲母分群編碼
- 同音字匹配
- 混合文字處理（中英文、標點符號）

### v0.3.0 - 布林檢索（預計 Week 5-6）
- [ ] 實作 `src/ir/index/inverted_index.py` - 倒排索引
- [ ] 實作 `src/ir/index/positional_index.py` - 位置索引
- [ ] 實作 `src/ir/retrieval/boolean.py` - 布林查詢引擎
- [ ] 建立 `scripts/boolean_search.py` CLI 工具
- [ ] 撰寫測試案例
- [ ] 更新實作指南

**目標功能**：
- AND/OR/NOT 操作
- 詞組查詢（phrase query）
- 查詢最佳化
- 位置資訊儲存

### v0.4.0 - 向量空間模型（預計 Week 7-8）
- [ ] 實作 `src/ir/index/term_weighting.py` - TF-IDF 計算
- [ ] 實作 `src/ir/retrieval/vsm.py` - VSM 檢索引擎
- [ ] 建立 `scripts/vsm_search.py` CLI 工具
- [ ] 實作多種權重方案（ltc, lnc, etc.）
- [ ] Top-K 堆積最佳化
- [ ] 撰寫效能測試

**目標功能**：
- TF-IDF 權重計算
- 餘弦相似度
- 多種正規化方案
- 高效 Top-K 檢索

### v0.5.0 - 評估指標（預計 Week 9）
- [ ] 實作 `src/ir/eval/metrics.py`
  - Precision, Recall, F-measure
  - Average Precision (AP)
  - Mean Average Precision (MAP)
  - Normalized Discounted Cumulative Gain (nDCG)
- [ ] 建立 `scripts/eval_run.py` 批次評估工具
- [ ] 支援多種輸出格式（JSON, CSV, TXT）
- [ ] 撰寫評估報告產生器

### v0.6.0 - 查詢擴展（預計 Week 10-11）
- [ ] 實作 `src/ir/ranking/rocchio.py` - Rocchio 演算法
- [ ] 建立 `scripts/expand_query.py` CLI 工具
- [ ] 實作擬相關回饋（pseudo-relevance feedback）
- [ ] 實作明確回饋（explicit feedback）
- [ ] 參數調校介面（α, β, γ）

### v0.7.0 - 分群演算法（預計 Week 12-13）
- [ ] 實作 `src/ir/cluster/doc_cluster.py`
  - K-means
  - Hierarchical clustering (single-link, complete-link)
  - Star clustering
- [ ] 實作 `src/ir/cluster/term_cluster.py`
  - 字串相似度分群
  - 編輯距離計算
- [ ] 建立視覺化工具
- [ ] 撰寫分群評估指標

### v0.8.0 - 自動摘要（預計 Week 14）
- [ ] 實作 `src/ir/summarize/static.py`
  - Lead-K 摘要
  - 關鍵句萃取
  - 位置加權
- [ ] 實作 `src/ir/summarize/dynamic.py`
  - KWIC (KeyWord In Context)
  - 視窗快取機制
  - 多查詢詞高亮
- [ ] 建立摘要品質評估

### v1.0.0 - 期末專案（預計 Week 16-18）
- [ ] 整合所有模組
- [ ] 建立 Web UI 介面
- [ ] 實作欄位搜尋（標題、作者、年份）
- [ ] 實作分面瀏覽
- [ ] 效能最佳化
- [ ] 完整系統測試
- [ ] 撰寫使用者手冊
- [ ] 錄製展示影片

---

## 變更類型說明

變更類型遵循以下分類：

- **新增 *Added***：新功能、新模組、新文件
- **修改 *Changed***：既有功能的變更
- **棄用 *Deprecated***：即將移除的功能
- **移除 *Removed***：已移除的功能
- **修正 *Fixed***：錯誤修正
- **安全性 *Security***：安全性相關更新

---

## 版本號規則

採用語意化版本 `MAJOR.MINOR.PATCH`：

- **MAJOR**（主版本號）：不相容的 API 變更
- **MINOR**（次版本號）：向下相容的功能新增
- **PATCH**（修訂號）：向下相容的錯誤修正

範例：
- `0.1.0` → `0.2.0`：新增 CSoundex 模組（新功能）
- `0.2.0` → `0.2.1`：修正 CSoundex 編碼錯誤（bug fix）
- `0.9.0` → `1.0.0`：完整期末專案（重大里程碑）

---

## 如何記錄變更

### 每次變更後必須更新本檔案

```bash
# 1. 完成程式碼修改
git add src/ir/text/csoundex.py

# 2. 更新 CHANGELOG.md
vim docs/CHANGELOG.md
# 在 [Unreleased] 區段新增變更項目

# 3. 一起提交
git add docs/CHANGELOG.md
git commit -m "feat(csoundex): implement phonetic encoding

- Add pinyin conversion
- Add consonant grouping
- Support mixed Chinese/English text
- Update CHANGELOG.md"
```

### 記錄格式範例

```markdown
## [0.2.0] - 2025-11-15

### 新增 *Added*

#### CSoundex 模組
- 實作 `src/ir/text/csoundex.py` - 中文諧音編碼核心功能
  - 支援漢字轉拼音（使用 pypinyin 函式庫）
  - 拼音正規化（去聲調、轉小寫）
  - 聲母分群編碼（0-9 共 10 群）
  - 輸出格式：`[首字母][3 位數字]`（如 `Z800` 代表「張/章/彰」）
- 建立 `scripts/csoundex_encode.py` - 命令列編碼工具
  - 支援 `--text` 直接編碼文字
  - 支援 `--file` 批次處理檔案
  - 支援 `--stdin` 管線輸入
- 新增 `configs/csoundex.yaml` - 聲母分群規則設定
- 準備 `datasets/lexicon/basic_pinyin.tsv` - 基礎拼音字典（5000 常用字）

#### 測試與文件
- 撰寫 `tests/test_csoundex.py` - 完整單元測試
  - 測試同音字匹配（張/章 → Z800）
  - 測試異形字處理（裡/裏）
  - 測試標點符號容錯
  - 測試混合中英文文本
  - 測試邊界條件（空字串、單字、超長文本）
- 更新 `docs/guides/CSOUNDEX.md` - CSoundex 詳細實作指南
- 新增使用範例至 README.md

### 修改 *Changed*
- 調整 `configs/csoundex.yaml` 韻母分群規則
  - 將 `ü` 併入元音組（群組 0）
  - 分離捲舌音 `zh/ch/sh/r`（群組 8）與平舌音 `z/c/s`（群組 9）

### 修正 *Fixed*
- 修正多音字預設取音問題（如「行」優先取 `xing` 而非 `hang`）
- 修正 UTF-8 編碼在 Windows 環境的相容性問題
```

---

## 參考連結

- [Keep a Changelog](https://keepachangelog.com/zh-TW/1.0.0/)
- [語意化版本](https://semver.org/lang/zh-TW/)
- [Conventional Commits](https://www.conventionalcommits.org/zh-hant/v1.0.0/)

---

**最後更新**：2025-11-12
**維護者**：[您的姓名/學號]
