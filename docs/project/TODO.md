# 開發任務清單 (Development Task List)

**中文新聞智能檢索系統 (CNIRS)**
**Chinese News Intelligent Retrieval System**

---

## 專案進度總覽 (Project Progress Overview)

| 階段 | 狀態 | 完成度 | 開始日期 | 預計完成 |
|-----|------|--------|---------|---------|
| **Phase 0: 專案規劃** | ✅ 完成 | 100% | Week 1 | Week 1 |
| **Phase 1: 傳統 IR 模組 (M1-M7)** | ✅ 完成 | 100% | Week 2-6 | Week 6 |
| **Phase 2: 現代 NLP 模組 (Phase 1-5)** | ✅ 完成 | 100% | Week 7-11 | Week 11 |
| **Phase 3: 資料收集與預處理** | ⏳ 進行中 | 0% | Week 12 | Week 13 |
| **Phase 4: 新增檢索模組** | ⏸️ 待開始 | 0% | Week 14 | Week 15 |
| **Phase 5: Web UI 開發** | ⏸️ 待開始 | 0% | Week 16 | Week 17 |
| **Phase 6: 整合與測試** | ⏸️ 待開始 | 0% | Week 18 | Week 19 |
| **Phase 7: 評估與優化** | ⏸️ 待開始 | 0% | Week 20 | Week 21 |
| **Phase 8: 文檔與展示** | ⏸️ 待開始 | 0% | Week 22 | Week 22 |

**圖示說明**:
- ✅ 完成 (Completed)
- ⏳ 進行中 (In Progress)
- ⏸️ 待開始 (Pending)
- ❌ 已取消 (Cancelled)
- 🔄 需修訂 (Needs Revision)

---

## Phase 0: 專案規劃 (Week 1) ✅

### 0.1 專案文檔撰寫 ✅

- [x] **PROPOSAL.md** - 期末專案提案
  - 專案動機與目標
  - 系統功能規劃 (10 大功能)
  - 技術架構設計
  - 資料來源與規模
  - 評估計畫
  - 6 週工作時程
  - 風險評估

- [x] **ARCHITECTURE.md** - 系統架構文件
  - 四層架構設計
  - 核心模組設計 (M1-M7 + Phase 1-5)
  - 資料流設計
  - API 設計 (RESTful + CLI)
  - 模組相依性分析
  - 部署架構 (Docker)
  - 效能考量與快取策略
  - 擴展性設計 (水平/垂直)
  - 安全性設計

- [x] **DATASET.md** - 資料集規劃文件
  - 資料來源 (CNA, PTS, TechNews)
  - 爬蟲策略與技術選擇
  - 資料結構設計 (JSON, SQLite)
  - 預處理流程
  - 品質控制
  - 評估資料集 (queries + qrels)
  - 儲存方案 (分層儲存)
  - 更新策略

- [x] **TODO.md** - 開發任務清單 (本文件)

- [x] **更新 CHANGELOG.md**

---

## Phase 1: 傳統 IR 模組 (M1-M7) ✅

### M1: Boolean Retrieval (布林檢索) ✅

- [x] **實作 `src/ir/retrieval/boolean.py`**
  - [x] `BooleanRetrieval` 類別
  - [x] 查詢解析器 (AND/OR/NOT)
  - [x] Posting list 合併演算法
  - [x] 短語查詢支援

- [x] **測試 `tests/test_boolean.py`**
  - [x] 基本 AND/OR/NOT 查詢
  - [x] 複雜布林運算式
  - [x] 短語查詢

- [x] **CLI 工具 `scripts/boolean_search.py`**

### M2-M3: Inverted Index & Positional Index ✅

- [x] **實作 `src/ir/index/inverted_index.py`**
  - [x] `InvertedIndex` 類別
  - [x] 建立索引方法
  - [x] 查詢 posting lists
  - [x] 序列化/反序列化

- [x] **實作 `src/ir/index/positional_index.py`**
  - [x] `PositionalIndex` 類別
  - [x] 短語查詢方法
  - [x] 鄰近查詢方法

- [x] **測試**
  - [x] `tests/test_inverted_index.py`
  - [x] `tests/test_positional_index.py`

### M4: Vector Space Model ✅

- [x] **實作 `src/ir/retrieval/vsm.py`**
  - [x] `VectorSpaceModel` 類別
  - [x] TF-IDF 計算
  - [x] Cosine Similarity
  - [x] Top-k 檢索

- [x] **測試 `tests/test_vsm.py`**

- [x] **CLI 工具 `scripts/tfidf_search.py`**

### M5: Evaluation Metrics ✅

- [x] **實作 `src/ir/eval/metrics.py`**
  - [x] Precision@k, Recall@k
  - [x] F-measure
  - [x] Average Precision (AP)
  - [x] Mean Average Precision (MAP)
  - [x] nDCG@k

- [x] **測試 `tests/test_metrics.py`**

- [x] **CLI 工具 `scripts/evaluate.py`**

### M6: Rocchio Query Expansion ✅

- [x] **實作 `src/ir/ranking/rocchio.py`**
  - [x] `RocchioExpansion` 類別
  - [x] Rocchio 演算法
  - [x] Pseudo-relevance feedback
  - [x] Explicit relevance feedback

- [x] **測試 `tests/test_rocchio.py`**

- [x] **CLI 工具 `scripts/expand_query.py`**

### M7: Clustering & Summarization ✅

- [x] **實作 `src/ir/cluster/`**
  - [x] K-means clustering
  - [x] Hierarchical clustering
  - [x] Document clustering

- [x] **實作 `src/ir/summarize/`**
  - [x] Lead-k summarization
  - [x] Key sentence extraction
  - [x] KWIC (KeyWord In Context)

- [x] **測試**
  - [x] `tests/test_cluster.py`
  - [x] `tests/test_summarize.py`

---

## Phase 2: 現代 NLP 模組 (Phase 1-5) ✅

### Phase 1: Chinese Tokenization ✅

- [x] **實作 `src/ir/text/tokenizer.py`**
  - [x] 支援 Jieba
  - [x] 支援 CKIP
  - [x] 支援 PKUSeg
  - [x] 統一介面

- [x] **測試 `tests/test_tokenizer.py`**

### Phase 2: Named Entity Recognition ✅

- [x] **實作 `src/ir/ner/ckip_ner.py`**
  - [x] CKIP Transformers 整合
  - [x] Entity 類別定義
  - [x] 批次識別
  - [x] 實體過濾

- [x] **測試 `tests/test_ner.py`**

- [x] **文檔 `docs/guides/NER_GUIDE.md`**

### Phase 3: Keyword Extraction ✅

- [x] **實作 `src/ir/keyextract/`**
  - [x] TextRank (`textrank.py`)
  - [x] YAKE (`yake_extractor.py`)
  - [x] KeyBERT (`keybert_extractor.py`)
  - [x] RAKE (`rake_extractor.py`)
  - [x] Ensemble 融合 (`ensemble.py`)

- [x] **2025 新增功能**
  - [x] 位置權重 (Position Weighting)
  - [x] NER 實體增強 (Entity Boosting)

- [x] **測試 `tests/test_keyextract.py`**

- [x] **文檔 `docs/guides/KEYWORD_EXTRACTION_GUIDE.md`**

### Phase 4: Topic Modeling ✅

- [x] **實作 `src/ir/topic/`**
  - [x] LDA (`lda.py`)
  - [x] BERTopic (`bertopic_model.py`)

- [x] **測試**
  - [x] `tests/test_lda.py`
  - [x] `tests/test_bertopic.py`

- [x] **文檔 `docs/guides/TOPIC_MODELING_GUIDE.md`**

### Phase 5: Syntactic Parsing ✅

- [x] **實作 `src/ir/syntax/parser.py`**
  - [x] `DependencyParser` (SuPar 整合)
  - [x] `SVOExtractor` (SVO 三元組提取)
  - [x] `SyntaxAnalyzer` (統一介面)
  - [x] PyTorch 2.6+ 相容性修復

- [x] **測試 `tests/test_syntax.py`** (17 個測試, 直接執行)

- [x] **文檔 `docs/guides/SYNTAX_PARSING_GUIDE.md`**

---

## Phase 3: 資料收集與預處理 (Week 12-13) ⏳

### 3.1 爬蟲開發 (Week 12, Day 1-3) ⏸️

- [ ] **實作 `scripts/crawlers/cna_spider.py`**
  - [ ] Scrapy Spider 設定
  - [ ] 列表頁解析
  - [ ] 文章頁解析
  - [ ] 日期範圍過濾 (2022-2024)
  - [ ] 錯誤處理與重試

- [ ] **實作 `scripts/crawlers/pts_spider.py`**
  - [ ] PTS 新聞網爬蟲

- [ ] **實作 `scripts/crawlers/technews_spider.py`**
  - [ ] 科技新報爬蟲

- [ ] **爬蟲配置 `configs/scrapy_settings.py`**
  - [ ] User-Agent 設定
  - [ ] Download delay (2 秒)
  - [ ] AutoThrottle 啟用
  - [ ] ROBOTSTXT_OBEY = True

**預計產出**:
- 500-1000 篇測試資料 (驗證爬蟲正確性)

---

### 3.2 爬蟲測試與除錯 (Week 12, Day 4-5) ⏸️

- [ ] **小規模測試**
  - [ ] 爬取 100-500 篇文章
  - [ ] 驗證資料完整性
  - [ ] 檢查 encoding 問題
  - [ ] 測試去重功能

- [ ] **效能測試**
  - [ ] 測量爬取速度 (篇/小時)
  - [ ] 檢查記憶體使用
  - [ ] 確認不會被封鎖

**預計產出**:
- 爬蟲測試報告
- 除錯後的穩定版爬蟲

---

### 3.3 正式資料收集 (Week 12, Day 6-7) ⏸️

- [ ] **執行全量爬取**
  - [ ] CNA: 目標 20,000 篇
  - [ ] PTS: 目標 10,000 篇
  - [ ] TechNews: 目標 5,000 篇

- [ ] **監控爬取進度**
  - [ ] 每小時檢查進度
  - [ ] 記錄錯誤與失敗 URL
  - [ ] 自動重試失敗請求

**預計產出**:
- `/mnt/c/data/information-retrieval/raw/cna_news_2022-2024.jsonl` (20,000 篇)
- `/mnt/c/data/information-retrieval/raw/pts_news_2022-2024.jsonl` (10,000 篇)
- `/mnt/c/data/information-retrieval/raw/technews_2022-2024.jsonl` (5,000 篇)

---

### 3.4 資料清洗 (Week 13, Day 1-3) ⏸️

- [ ] **實作 `scripts/clean_data.py`**
  - [ ] HTML 標籤清除
  - [ ] 日期格式統一
  - [ ] 分類名稱正規化
  - [ ] 內容長度過濾 (> 100 字)
  - [ ] 去重 (URL + content fingerprint)

- [ ] **執行清洗**
  - [ ] 處理所有原始資料
  - [ ] 生成清洗報告 (移除數量、原因)

**預計產出**:
- `data/cleaned/news_cleaned.jsonl` (30,000-35,000 篇)
- 清洗報告 (`data/analysis/cleaning_report.txt`)

---

### 3.5 資料統計分析 (Week 13, Day 4-5) ⏸️

- [ ] **實作 `scripts/analyze_dataset.py`**
  - [ ] 基礎統計 (篇數、來源、分類)
  - [ ] 文章長度分布
  - [ ] 時間分布
  - [ ] 詞頻統計

- [ ] **生成視覺化**
  - [ ] 分類分布圓餅圖
  - [ ] 來源分布長條圖
  - [ ] 文章長度直方圖
  - [ ] 時間序列圖

**預計產出**:
- `data/analysis/dataset_statistics.json`
- `data/analysis/dataset_statistics.png`

---

### 3.6 資料預處理 (Week 13, Day 6-7) ⏸️

- [ ] **實作 `scripts/preprocess_news.py`**
  - [ ] 整合所有 NLP 模組
  - [ ] 批次斷詞 (Jieba)
  - [ ] 批次 NER (CKIP)
  - [ ] 批次關鍵字提取 (Ensemble)
  - [ ] 主題建模 (LDA + BERTopic)
  - [ ] 摘要生成

- [ ] **執行預處理**
  - [ ] 處理全部 30,000+ 篇
  - [ ] 使用 GPU 加速 (若可用)
  - [ ] 進度監控

**預計產出**:
- `data/preprocessed/news_preprocessed.jsonl`
- 預處理時間統計

**預估時間**:
- 斷詞: 30 秒 (30k 篇)
- NER: 2-3 小時 (GPU) / 10+ 小時 (CPU)
- 關鍵字: 1 小時
- 主題建模: 30 分鐘 (LDA), 1 小時 (BERTopic)
- **總計: ~4-6 小時 (GPU)**

---

### 3.7 索引建立 (Week 13, Day 7) ⏸️

- [ ] **實作 `scripts/build_indexes.py`**
  - [ ] 建立 Inverted Index
  - [ ] 建立 Positional Index
  - [ ] 計算 TF-IDF Vectors
  - [ ] 生成 BERT Embeddings
  - [ ] 建立 FAISS 索引 (加速 BERT 搜尋)

- [ ] **執行索引建立**

**預計產出**:
- `/mnt/c/data/information-retrieval/indexes/inverted_index.pkl`
- `/mnt/c/data/information-retrieval/indexes/positional_index.pkl`
- `/mnt/c/data/information-retrieval/indexes/tfidf/tfidf_matrix.pkl`
- `/mnt/c/data/information-retrieval/indexes/bert/embeddings.h5`
- `/mnt/c/data/information-retrieval/indexes/bert/faiss_index.bin`

---

### 3.8 資料庫建立 (Week 13, Day 7) ⏸️

- [ ] **實作 `scripts/build_database.py`**
  - [ ] 建立 SQLite schema
  - [ ] 匯入預處理資料
  - [ ] 建立索引 (publish_date, category, source)

- [ ] **執行匯入**

**預計產出**:
- `data/database/cnirs.db` (~100 MB)

---

### 3.9 評估資料集建立 (Week 13, Day 7) ⏸️

- [ ] **設計測試查詢**
  - [ ] 50 個查詢 (科技 15, 政治 10, 財經 10, 社會 10, 混合 5)
  - [ ] 涵蓋不同查詢類型 (短查詢、長查詢、專業術語)

- [ ] **實作 `scripts/create_qrels.py`**
  - [ ] 使用 Pooling 方法
  - [ ] BM25 + BERT 各取 top-20
  - [ ] 合併候選池

- [ ] **人工標註**
  - [ ] 標註相關性 (0/1/2)
  - [ ] 每個查詢至少 20 篇標註

**預計產出**:
- `/mnt/c/data/information-retrieval/evaluation/queries.json`
- `/mnt/c/data/information-retrieval/evaluation/qrels.txt`

---

## Phase 4: 新增檢索模組 (Week 14-15) ⏸️

### 4.1 BM25 Retrieval (Week 14, Day 1-2) ⏸️

- [ ] **實作 `src/ir/retrieval/bm25.py`**
  - [ ] `BM25Retrieval` 類別
  - [ ] BM25 評分公式
  - [ ] Top-k 檢索
  - [ ] 可調參數 (k1, b)

- [ ] **測試 `tests/test_bm25.py`**
  - [ ] 基本檢索測試
  - [ ] 參數調整測試
  - [ ] 與 TF-IDF 比較

- [ ] **CLI 工具 `scripts/bm25_search.py`**

**預計產出**:
- BM25 檢索模組
- 效能測試報告

---

### 4.2 BERT Semantic Search (Week 14, Day 3-5) ⏸️

- [ ] **實作 `src/ir/retrieval/bert_search.py`**
  - [ ] `BERTSemanticSearch` 類別
  - [ ] Sentence-Transformers 整合
  - [ ] 文檔編碼方法
  - [ ] 查詢編碼方法
  - [ ] FAISS 加速搜尋
  - [ ] 批次處理

- [ ] **模型選擇**
  - [ ] 測試 3-5 個中文語意搜尋模型
  - [ ] `paraphrase-multilingual-MiniLM-L12-v2`
  - [ ] `sentence-transformers/distiluse-base-multilingual-cased-v2`
  - [ ] 選擇最佳模型

- [ ] **測試 `tests/test_bert_search.py`**
  - [ ] 編碼測試
  - [ ] 相似度計算測試
  - [ ] 效能測試 (速度、準確度)

- [ ] **CLI 工具 `scripts/bert_search.py`**

**預計產出**:
- BERT 語意搜尋模組
- 模型比較報告
- BERT embeddings (已在 Phase 3 生成)

---

### 4.3 Query Expansion (Week 14, Day 6-7) ⏸️

- [ ] **實作 `src/ir/expansion/synonym_expansion.py`**
  - [ ] 同義詞詞典載入 (HowNet 或自建)
  - [ ] 同義詞擴展方法
  - [ ] 擴展詞數量控制

- [ ] **實作 `src/ir/expansion/semantic_expansion.py`**
  - [ ] Word2Vec 模型整合
  - [ ] 語義相似詞查找
  - [ ] 閾值控制

- [ ] **整合 Rocchio (已有)**
  - [ ] 統一查詢擴展介面

- [ ] **測試**
  - [ ] `tests/test_synonym_expansion.py`
  - [ ] `tests/test_semantic_expansion.py`

**預計產出**:
- 查詢擴展模組
- 同義詞詞典 (若自建)

---

### 4.4 整合搜尋引擎 (Week 15, Day 1-3) ⏸️

- [ ] **實作 `src/ir/engine/search_engine.py`**
  - [ ] `SearchEngine` 類別 (統一介面)
  - [ ] 支援多模型檢索
  - [ ] 結果合併與排序
  - [ ] 查詢擴展整合
  - [ ] 快取機制

- [ ] **實作查詢解析器**
  - [ ] 布林運算子識別
  - [ ] 短語查詢識別
  - [ ] 欄位查詢 (title:, author:, date:)

- [ ] **實作結果聚合器**
  - [ ] 多模型結果合併
  - [ ] 重排序 (Re-ranking)
  - [ ] 去重

**預計產出**:
- 統一搜尋引擎介面
- 支援 4 種檢索模型 + 查詢擴展

---

### 4.5 CLI 統一介面 (Week 15, Day 4-5) ⏸️

- [ ] **實作 `scripts/search.py` (統一搜尋工具)**
  ```bash
  python scripts/search.py --query "人工智慧" --model tfidf --top-k 10
  python scripts/search.py --query "AI" --models all --compare
  python scripts/search.py --query "機器學習" --expand rocchio
  ```

- [ ] **實作 `scripts/index.py` (索引管理工具)**
  ```bash
  python scripts/index.py --build --input data/news.jsonl
  python scripts/index.py --update --incremental
  python scripts/index.py --rebuild
  ```

- [ ] **實作 `scripts/analyze.py` (分析工具)**
  ```bash
  python scripts/analyze.py --task topics --num-topics 10
  python scripts/analyze.py --task clustering --num-clusters 5
  python scripts/analyze.py --task wordcloud
  ```

**預計產出**:
- 統一 CLI 工具集
- 使用文檔

---

## Phase 5: Web UI 開發 (Week 16-17) ⏸️

### 5.1 後端 API 開發 (Week 16, Day 1-3) ⏸️

- [ ] **選擇 Web 框架**
  - [ ] Flask (簡單) 或 FastAPI (高效能)
  - [ ] 決定: **Flask** (學習曲線低)

- [ ] **實作 `app/app.py` (Flask 主程式)**
  - [ ] 路由設定
  - [ ] CORS 設定
  - [ ] 錯誤處理

- [ ] **實作 API 端點**
  - [ ] `POST /api/v1/search` - 搜尋
  - [ ] `GET /api/v1/document/<id>` - 獲取文檔
  - [ ] `GET /api/v1/analysis/topics` - 主題分析
  - [ ] `GET /api/v1/analysis/clusters` - 聚類分析
  - [ ] `POST /api/v1/evaluation/compare` - 模型比較

- [ ] **實作 `app/services/`**
  - [ ] `search_service.py` - 搜尋服務
  - [ ] `analysis_service.py` - 分析服務
  - [ ] `evaluation_service.py` - 評估服務

- [ ] **API 測試**
  - [ ] 使用 Postman/curl 測試
  - [ ] 編寫 API 測試 (`tests/test_api.py`)

**預計產出**:
- RESTful API (5+ 端點)
- API 文檔 (Swagger/OpenAPI)

---

### 5.2 前端開發 (Week 16, Day 4-7) ⏸️

- [ ] **技術選擇**
  - [ ] Bootstrap 5 (UI 框架)
  - [ ] Chart.js / Plotly.js (圖表)
  - [ ] jQuery (AJAX)

- [ ] **頁面設計**
  - [ ] `templates/index.html` - 首頁/搜尋頁
  - [ ] `templates/results.html` - 搜尋結果頁
  - [ ] `templates/document.html` - 文檔詳情頁
  - [ ] `templates/analysis.html` - 分析儀表板
  - [ ] `templates/comparison.html` - 模型比較頁

- [ ] **實作 `static/css/style.css`**
  - [ ] 自定義樣式
  - [ ] 響應式設計

- [ ] **實作 `static/js/app.js`**
  - [ ] AJAX 搜尋請求
  - [ ] 結果動態渲染
  - [ ] 圖表繪製 (主題分布、聚類視覺化)

**預計產出**:
- 5 個主要頁面
- 響應式 UI

---

### 5.3 功能實作 (Week 17, Day 1-4) ⏸️

#### 搜尋頁面功能

- [ ] **基本搜尋**
  - [ ] 查詢輸入框
  - [ ] 模型選擇 (Boolean, TF-IDF, BM25, BERT)
  - [ ] Top-k 設定
  - [ ] 進階選項 (展開/收合)

- [ ] **進階搜尋**
  - [ ] 日期範圍過濾 (date picker)
  - [ ] 分類過濾 (多選)
  - [ ] 來源過濾
  - [ ] 查詢擴展選項 (Rocchio, Synonym)

- [ ] **搜尋結果**
  - [ ] 文章卡片展示 (標題、摘要、日期、來源)
  - [ ] 關鍵字高亮
  - [ ] NER 實體標記 (不同顏色)
  - [ ] 分頁

#### 文檔詳情頁面

- [ ] **文檔內容**
  - [ ] 完整文章顯示
  - [ ] Metadata (作者、日期、來源、分類)

- [ ] **NLP 分析結果**
  - [ ] 實體列表 (PERSON, ORG, LOC)
  - [ ] 關鍵字列表
  - [ ] 主題標籤 (LDA, BERTopic)
  - [ ] SVO 三元組

#### 分析儀表板

- [ ] **主題分析**
  - [ ] LDA 主題分布圓餅圖
  - [ ] BERTopic 主題分布
  - [ ] 主題詞雲
  - [ ] 主題時間序列

- [ ] **聚類分析**
  - [ ] 文檔聚類視覺化 (2D scatter plot using t-SNE/UMAP)
  - [ ] 聚類大小分布
  - [ ] 聚類關鍵詞

- [ ] **統計資訊**
  - [ ] 資料集概覽 (總篇數、來源、分類)
  - [ ] 詞頻統計 (Top-50 詞彙)
  - [ ] 實體統計 (Top-50 人名、機構)

#### 模型比較頁面

- [ ] **評估指標比較**
  - [ ] 輸入 queries 與 qrels
  - [ ] 計算 MAP, nDCG@10, P@5, R@10
  - [ ] 表格展示
  - [ ] 長條圖比較

- [ ] **單一查詢比較**
  - [ ] 選擇查詢
  - [ ] 4 種模型結果並排顯示
  - [ ] 相同文檔標記
  - [ ] Venn diagram (結果交集)

**預計產出**:
- 完整功能的 Web UI
- 5 個主要頁面全部實作完成

---

### 5.4 UI/UX 優化 (Week 17, Day 5-7) ⏸️

- [ ] **效能優化**
  - [ ] AJAX 異步載入
  - [ ] 結果快取
  - [ ] 延遲載入 (Lazy loading)

- [ ] **使用者體驗**
  - [ ] Loading 動畫
  - [ ] 錯誤提示 (Toast notifications)
  - [ ] 快捷鍵 (Enter 搜尋)
  - [ ] 搜尋歷史

- [ ] **視覺優化**
  - [ ] 配色調整
  - [ ] 圖示添加 (Font Awesome)
  - [ ] 動畫效果

- [ ] **跨瀏覽器測試**
  - [ ] Chrome
  - [ ] Firefox
  - [ ] Safari
  - [ ] Edge

**預計產出**:
- 優化後的使用者介面
- 跨瀏覽器相容性報告

---

## Phase 6: 整合與測試 (Week 18-19) ⏸️

### 6.1 系統整合 (Week 18, Day 1-3) ⏸️

- [ ] **模組整合測試**
  - [ ] 搜尋引擎 ↔ Web API
  - [ ] Web API ↔ 前端
  - [ ] 資料庫 ↔ 搜尋引擎
  - [ ] 快取 ↔ 各模組

- [ ] **端到端測試**
  - [ ] 使用者搜尋流程 (輸入查詢 → 顯示結果)
  - [ ] 文檔詳情查看
  - [ ] 分析功能
  - [ ] 模型比較

- [ ] **錯誤處理測試**
  - [ ] 無效查詢
  - [ ] 網路錯誤
  - [ ] 模型載入失敗
  - [ ] 資料庫連線失敗

**預計產出**:
- 整合測試報告
- Bug 清單

---

### 6.2 效能測試 (Week 18, Day 4-5) ⏸️

- [ ] **檢索效能測試**
  - [ ] Boolean: < 100ms
  - [ ] TF-IDF: < 200ms
  - [ ] BM25: < 200ms
  - [ ] BERT: < 2s (30k 文檔)

- [ ] **負載測試**
  - [ ] 10 並發用戶
  - [ ] 50 並發用戶
  - [ ] 響應時間測量
  - [ ] 記憶體使用測量

- [ ] **使用工具**
  - [ ] Apache Bench (ab)
  - [ ] Locust
  - [ ] Python cProfile

**預計產出**:
- 效能測試報告
- 瓶頸分析

---

### 6.3 Bug 修復與優化 (Week 18, Day 6-7 + Week 19, Day 1-3) ⏸️

- [ ] **修復整合測試發現的 Bug**
  - [ ] 按優先級排序
  - [ ] 高優先級: 系統崩潰、資料錯誤
  - [ ] 中優先級: 功能異常
  - [ ] 低優先級: UI 小問題

- [ ] **效能優化**
  - [ ] 針對瓶頸優化
  - [ ] 快取策略調整
  - [ ] 資料庫查詢優化
  - [ ] 前端資源壓縮

**預計產出**:
- Bug 修復記錄
- 優化後的系統

---

### 6.4 使用者測試 (Week 19, Day 4-5) ⏸️

- [ ] **邀請 3-5 位使用者**
  - [ ] 同學、老師、朋友

- [ ] **測試任務**
  - [ ] 任務 1: 搜尋 "人工智慧發展"
  - [ ] 任務 2: 比較 4 種檢索模型
  - [ ] 任務 3: 查看主題分析
  - [ ] 任務 4: 查看文檔詳情與 NER

- [ ] **收集反饋**
  - [ ] 問卷調查
  - [ ] 介面易用性
  - [ ] 功能完整性
  - [ ] 改進建議

**預計產出**:
- 使用者測試報告
- 改進清單

---

### 6.5 最終調整 (Week 19, Day 6-7) ⏸️

- [ ] **根據使用者反饋調整**
  - [ ] UI 調整
  - [ ] 功能微調
  - [ ] 文字修正

- [ ] **最終測試**
  - [ ] 完整流程測試
  - [ ] 確保所有功能正常

**預計產出**:
- 可展示的完整系統

---

## Phase 7: 評估與優化 (Week 20-21) ⏸️

### 7.1 評估實驗設計 (Week 20, Day 1-2) ⏸️

- [ ] **實驗設定**
  - [ ] 測試查詢: 50 queries
  - [ ] 評估指標: MAP, nDCG@10, P@5, P@10, R@10
  - [ ] 模型: Boolean, TF-IDF, BM25, BERT
  - [ ] 查詢擴展: Baseline vs Rocchio vs Synonym

- [ ] **實驗腳本**
  - [ ] `scripts/run_evaluation.py`
  - [ ] 自動化評估流程
  - [ ] 結果儲存 (JSON)

**預計產出**:
- 評估實驗計畫

---

### 7.2 執行評估實驗 (Week 20, Day 3-5) ⏸️

- [ ] **實驗 1: 基礎檢索模型比較**
  - [ ] Boolean vs TF-IDF vs BM25 vs BERT
  - [ ] 計算所有評估指標
  - [ ] 統計顯著性測試 (t-test)

- [ ] **實驗 2: 查詢擴展效果**
  - [ ] Baseline (無擴展)
  - [ ] Rocchio
  - [ ] Synonym
  - [ ] Rocchio + Synonym

- [ ] **實驗 3: NLP 模組貢獻**
  - [ ] 加入 NER 增強 vs 不加入
  - [ ] 加入主題資訊 vs 不加入
  - [ ] 加入句法資訊 vs 不加入

- [ ] **實驗 4: 參數調整**
  - [ ] BM25: k1, b
  - [ ] Rocchio: α, β, γ
  - [ ] Top-k: 5, 10, 20, 50

**預計產出**:
- 評估結果 (JSON/CSV)
- 原始實驗資料

---

### 7.3 結果分析 (Week 20, Day 6-7) ⏸️

- [ ] **實作 `scripts/analyze_results.py`**
  - [ ] 讀取評估結果
  - [ ] 計算統計量 (mean, std, median)
  - [ ] 統計顯著性分析
  - [ ] 生成視覺化圖表

- [ ] **生成圖表**
  - [ ] 模型比較長條圖 (MAP, nDCG)
  - [ ] 查詢擴展效果圖
  - [ ] 參數影響曲線圖
  - [ ] Precision-Recall 曲線

- [ ] **錯誤分析**
  - [ ] 找出失敗查詢 (MAP = 0)
  - [ ] 分析失敗原因
  - [ ] Case study (2-3 個查詢)

**預計產出**:
- 評估分析報告
- 視覺化圖表 (10+ 張)

---

### 7.4 系統優化 (Week 21, Day 1-3) ⏸️

- [ ] **基於評估結果優化**
  - [ ] 調整最佳參數
  - [ ] 改進失敗查詢的處理
  - [ ] 優化查詢擴展策略

- [ ] **效能優化**
  - [ ] BERT 搜尋加速 (FAISS 參數調整)
  - [ ] 快取命中率提升
  - [ ] 資料庫索引優化

- [ ] **重新評估**
  - [ ] 驗證優化效果
  - [ ] 記錄改進幅度

**預計產出**:
- 優化後的系統
- 優化效果報告

---

### 7.5 對比實驗 (Week 21, Day 4-5) ⏸️

- [ ] **傳統 IR vs 現代 NLP**
  - [ ] 傳統: Boolean, TF-IDF, BM25
  - [ ] 現代: BERT Semantic Search
  - [ ] 分析各自優勢與劣勢

- [ ] **速度 vs 準確度權衡**
  - [ ] TF-IDF (快) vs BERT (準確)
  - [ ] 混合策略: TF-IDF 初篩 + BERT 重排

- [ ] **不同文本長度的影響**
  - [ ] 短文本 (< 200 字)
  - [ ] 中文本 (200-1000 字)
  - [ ] 長文本 (> 1000 字)

**預計產出**:
- 對比實驗報告
- 傳統 vs 現代分析

---

## Phase 8: 文檔與展示 (Week 22) ⏸️

### 8.1 期末報告撰寫 (Week 22, Day 1-3) ⏸️

- [ ] **更新 `docs/project/REPORT.md`**
  - [ ] 1. 題目與目標
  - [ ] 2. 理論背景
  - [ ] 3. 方法設計
  - [ ] 4. 實作細節
  - [ ] 5. 實驗設計
  - [ ] 6. 結果與分析
  - [ ] 7. 限制與未來工作
  - [ ] 8. 重現步驟

- [ ] **圖表製作**
  - [ ] 系統架構圖 (高解析度 SVG)
  - [ ] 資料流程圖
  - [ ] 評估結果圖表 (10+ 張)
  - [ ] UI 截圖 (5+ 張)

- [ ] **程式碼整理**
  - [ ] 清理註解
  - [ ] 統一程式碼風格
  - [ ] 移除測試/除錯程式碼

**預計產出**:
- 期末報告 (30-50 頁, Markdown)

---

### 8.2 展示準備 (Week 22, Day 4-5) ⏸️

- [ ] **簡報製作**
  - [ ] 使用 Google Slides 或 PowerPoint
  - [ ] 10-15 分鐘簡報
  - [ ] 內容:
    - [ ] 專案動機
    - [ ] 系統架構
    - [ ] 核心功能展示
    - [ ] 實驗結果
    - [ ] Demo 影片或現場展示

- [ ] **Demo 準備**
  - [ ] 準備 3-5 個展示查詢
  - [ ] 練習展示流程
  - [ ] 錄製 Demo 影片 (備用)

- [ ] **問答準備**
  - [ ] 預想可能問題
  - [ ] 準備回答

**預計產出**:
- 簡報 (PPT/PDF)
- Demo 影片 (可選)

---

### 8.3 文檔導出 (Week 22, Day 6) ⏸️

- [ ] **導出 DOCX/PDF**
  - [ ] 使用 Pandoc 或 `scripts/format_to_docx.py`
  - [ ] 確保格式正確
  - [ ] 調整頁邊距、行距、字體

- [ ] **命名規範**
  - [ ] `ACI2025FinalReport_<學號>.docx`
  - [ ] `ACI2025FinalReport_<學號>.pdf`

**預計產出**:
- 期末報告 (DOCX + PDF)

---

### 8.4 最終檢查 (Week 22, Day 7) ⏸️

- [ ] **系統檢查**
  - [ ] 所有功能正常運作
  - [ ] 無明顯 Bug
  - [ ] Web UI 可訪問

- [ ] **文檔檢查**
  - [ ] 報告完整性
  - [ ] 無錯別字
  - [ ] 圖表清晰

- [ ] **程式碼檢查**
  - [ ] README.md 完整
  - [ ] requirements.txt 正確
  - [ ] 可重現性

- [ ] **提交準備**
  - [ ] 壓縮專案 (排除大檔案)
  - [ ] 上傳至 GitHub (若需要)
  - [ ] 準備提交

**預計產出**:
- 可提交的完整專案

---

## 附錄：工具與腳本清單 (Appendix: Tools & Scripts)

### 資料收集與處理

| 腳本 | 功能 | 狀態 |
|-----|------|------|
| `scripts/crawlers/cna_spider.py` | CNA 新聞爬蟲 | ⏸️ 待開發 |
| `scripts/crawlers/pts_spider.py` | PTS 新聞爬蟲 | ⏸️ 待開發 |
| `scripts/crawlers/technews_spider.py` | 科技新報爬蟲 | ⏸️ 待開發 |
| `scripts/clean_data.py` | 資料清洗 | ⏸️ 待開發 |
| `scripts/analyze_dataset.py` | 資料集統計分析 | ⏸️ 待開發 |
| `scripts/preprocess_news.py` | NLP 預處理 | ⏸️ 待開發 |
| `scripts/build_indexes.py` | 索引建立 | ⏸️ 待開發 |
| `scripts/build_database.py` | 資料庫建立 | ⏸️ 待開發 |
| `scripts/create_qrels.py` | 建立評估資料集 | ⏸️ 待開發 |

### 檢索與分析

| 腳本 | 功能 | 狀態 |
|-----|------|------|
| `scripts/search.py` | 統一搜尋工具 | ⏸️ 待開發 |
| `scripts/index.py` | 索引管理工具 | ⏸️ 待開發 |
| `scripts/analyze.py` | 分析工具 | ⏸️ 待開發 |
| `scripts/boolean_search.py` | Boolean 檢索 | ✅ 完成 |
| `scripts/tfidf_search.py` | TF-IDF 檢索 | ✅ 完成 |
| `scripts/bm25_search.py` | BM25 檢索 | ⏸️ 待開發 |
| `scripts/bert_search.py` | BERT 語意搜尋 | ⏸️ 待開發 |
| `scripts/expand_query.py` | 查詢擴展 | ✅ 完成 |

### 評估

| 腳本 | 功能 | 狀態 |
|-----|------|------|
| `scripts/evaluate.py` | 評估工具 | ✅ 完成 |
| `scripts/run_evaluation.py` | 自動化評估 | ⏸️ 待開發 |
| `scripts/analyze_results.py` | 結果分析 | ⏸️ 待開發 |

### Web 應用

| 檔案 | 功能 | 狀態 |
|-----|------|------|
| `app/app.py` | Flask 主程式 | ⏸️ 待開發 |
| `app/services/search_service.py` | 搜尋服務 | ⏸️ 待開發 |
| `app/services/analysis_service.py` | 分析服務 | ⏸️ 待開發 |
| `templates/index.html` | 搜尋頁 | ⏸️ 待開發 |
| `templates/results.html` | 結果頁 | ⏸️ 待開發 |
| `templates/analysis.html` | 分析頁 | ⏸️ 待開發 |

---

## 總結 (Summary)

本 TODO 清單詳細規劃了 **CNIRS** 的完整開發流程，涵蓋 **8 個階段、22 週的工作計畫**：

- **Phase 0**: 專案規劃 ✅
- **Phase 1**: 傳統 IR 模組 ✅
- **Phase 2**: 現代 NLP 模組 ✅
- **Phase 3**: 資料收集與預處理 (進行中)
- **Phase 4**: 新增檢索模組 (待開始)
- **Phase 5**: Web UI 開發 (待開始)
- **Phase 6**: 整合與測試 (待開始)
- **Phase 7**: 評估與優化 (待開始)
- **Phase 8**: 文檔與展示 (待開始)

目前已完成 **Phase 0-2** (約 50% 核心模組)，接下來將進入 **資料收集與系統整合階段**。

---

**文件版本**: v1.0
**最後更新**: 2025-11-13
**作者**: CNIRS 開發團隊
**授權**: Educational Use
