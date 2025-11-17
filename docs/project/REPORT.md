# 期末專案報告：學術論文搜尋引擎

**專案名稱**：AcademicSearch - 輕量級學術論文檢索系統
**英文名稱**：AcademicSearch - Lightweight Academic Paper Retrieval System

**課程**：LIS5033 自動分類與索引 *Automatic Classification and Indexing*
**學期**：2024-2025
**團隊成員**：[學號1] [姓名1], [學號2] [姓名2]
**報告日期**：2025-01-15

**專案網址**：https://github.com/username/academic-search
**展示影片**：https://youtu.be/xxxxx

---

## 摘要 *Executive Summary*

[150-200 字總結報告內容]

本專案開發了一個輕量級學術論文檢索系統 AcademicSearch，整合多種資訊檢索技術，包括倒排索引、TF-IDF 排序、分面瀏覽與查詢擴展。系統索引 20,000 筆 arXiv 論文，提供自然語言查詢、多維度排序（相關性+引用數+時效性）與視覺化探索功能。評估結果顯示，系統在測試集上達到 MAP 0.68、nDCG@10 0.72，查詢回應時間 P95 為 145ms，滿足即時檢索需求。使用者研究顯示，相較於基準系統，本系統的任務完成率提升 23%，平均滿意度達 4.2/5。本專案展示了課程所學 IR 技術在真實應用場景的整合與實踐。

**關鍵詞**：資訊檢索 *Information Retrieval*、學術搜尋 *Academic Search*、倒排索引 *Inverted Index*、TF-IDF、分面瀏覽 *Faceted Search*

---

## 目錄

1. [緒論](#1-緒論)
2. [文獻回顧](#2-文獻回顧)
3. [系統設計](#3-系統設計)
4. [實作細節](#4-實作細節)
5. [實驗與評估](#5-實驗與評估)
6. [使用者研究](#6-使用者研究)
7. [討論與反思](#7-討論與反思)
8. [結論與未來工作](#8-結論與未來工作)
9. [參考文獻](#9-參考文獻)
10. [附錄](#10-附錄)

---

## 1. 緒論 *Introduction*

### 1.1 研究背景與動機

學術研究的首要步驟是文獻回顧（*Literature Review*），研究者需要系統性地檢索、篩選與分析領域內的相關論文。根據統計，博士生在撰寫論文過程中平均需閱讀 300-500 篇文獻 [引用來源]。然而，隨著學術出版的爆炸性成長——以 arXiv 為例，2023 年單年新增論文超過 200,000 篇——研究者面臨嚴重的資訊過載問題。

現有的學術搜尋平台（如 Google Scholar, Microsoft Academic, Semantic Scholar）雖然功能強大，但存在以下痛點：

1. **查詢語法門檻**：進階檢索需要學習布林運算子與欄位限定語法
2. **結果泛濫**：熱門主題的查詢結果動輒數萬筆，缺乏有效篩選機制
3. **領域泛化**：通用平台無法針對特定研究社群優化排序與推薦
4. **個人化不足**：未考慮研究者的專業背景與研究興趣

### 1.2 研究問題

本專案針對上述挑戰，提出以下研究問題：

**RQ1**：如何設計直覺的檢索介面，降低學術搜尋的使用門檻？
**RQ2**：如何整合多維度排序因子（相關性、引用數、時效性），提升結果品質？
**RQ3**：如何透過分面瀏覽與視覺化，支援探索式文獻發現？
**RQ4**：自行實作的輕量級 IR 系統，能否在效能與品質上達到實用標準？

### 1.3 專案目標與貢獻

**目標**：
- ✅ 建立涵蓋 20,000 筆論文的全文索引與檢索系統
- ✅ 實作 TF-IDF 排序、多維度排序與查詢擴展
- ✅ 提供分面瀏覽（作者、年份、會議）與引用網路視覺化
- ✅ 查詢回應時間 < 200ms（P95）
- ✅ MAP > 0.65, nDCG@10 > 0.70

**貢獻**：
1. **整合性實踐**：將課程學習的 IR 技術（索引、檢索、排序、評估）整合為完整系統
2. **開源實作**：提供可重現的輕量級學術搜尋引擎參考實作
3. **使用者導向設計**：透過使用者研究驗證系統可用性與有效性
4. **教學資源**：完整的程式碼與文件可作為 IR 課程的實作範例

### 1.4 報告結構

本報告組織如下：第 2 節回顧相關文獻與技術基礎；第 3 節描述系統架構設計；第 4 節詳述核心模組的實作細節；第 5 節報告離線評估實驗；第 6 節呈現使用者研究結果；第 7 節討論研究發現與限制；第 8 節總結並展望未來工作。

---

## 2. 文獻回顧 *Related Work*

### 2.1 學術搜尋引擎

#### 2.1.1 商業系統

**Google Scholar** [1] 是最廣泛使用的學術搜尋平台，採用 PageRank 變體（Academic Rank）整合引用關係進行排序。其優勢在於涵蓋範圍廣（跨學科）、索引更新快速，但缺點是排序演算法不透明、缺乏進階篩選功能。

**Microsoft Academic** [2]（已於 2021 年停止服務）曾提供知識圖譜（*Knowledge Graph*）整合作者、機構、期刊等實體，支援多維度探索。其停止服務突顯了學術搜尋的商業挑戰與開源替代方案的重要性。

**Semantic Scholar** [3] 由 Allen Institute for AI 開發，特色是使用 NLP 技術萃取論文的語意資訊（研究方法、資料集、實驗結果），並提供引用上下文（*Citation Context*）分析。然而其僅涵蓋部分學科（CS, Bio 為主）。

#### 2.1.2 領域專用系統

**PubMed** [4] 針對生物醫學領域，提供 MeSH（Medical Subject Headings）控制詞彙與複雜的布林查詢語法，適合專業研究者進行高召回率檢索。

**arXiv.org** [5] 是物理、數學、電腦科學的預印本平台，提供基本的分類瀏覽與關鍵字搜尋，但檢索功能相對陽春，缺乏排序與推薦。

### 2.2 資訊檢索核心技術

#### 2.2.1 索引建構

**倒排索引** *Inverted Index* 是現代搜尋引擎的基礎資料結構 [6]。Manning 等人（2008）詳細討論了索引建構的最佳化策略，包括分塊排序（*Block Sorting*）、合併（*Merge*）與壓縮（*Compression*）技術 [7]。

**位置索引** *Positional Index* 儲存詞彙在文件中的位置資訊，支援詞組查詢（*Phrase Query*）與鄰近查詢（*Proximity Query*），代價是索引大小增加 2-4 倍 [8]。

#### 2.2.2 排序模型

**TF-IDF** 是經典的統計權重方案，假設詞頻（*Term Frequency*）高且文件頻率（*Document Frequency*）低的詞彙更重要。餘弦相似度（*Cosine Similarity*）衡量查詢與文件向量的夾角，實現排序 [9]。

**BM25** 是對 TF-IDF 的改良，加入文件長度正規化與飽和函數（*Saturation Function*），在 TREC 競賽中表現優異 [10]。然而實作複雜度較高，本專案採用 TF-IDF 以平衡效能與實作成本。

**Learning to Rank** 使用機器學習（如 RankSVM, LambdaMART）訓練排序模型，整合多個特徵（內容相似度、引用數、作者權威性等）[11]。本專案採用簡化的線性組合模型。

#### 2.2.3 查詢擴展

**Rocchio 演算法** [12] 透過相關回饋（*Relevance Feedback*）修改查詢向量，向相關文件靠近、遠離不相關文件。**擬相關回饋**（*Pseudo-Relevance Feedback*）假設初次檢索的前 K 筆為相關，自動執行查詢擴展，無需使用者標記 [13]。

#### 2.2.4 評估指標

IR 系統的評估包含效能（*Efficiency*）與品質（*Effectiveness*）兩面向。效能指標如索引建構時間、查詢回應時間、記憶體使用量；品質指標如 Precision, Recall, MAP, nDCG [14]。

### 2.3 視覺化與互動設計

**分面瀏覽** *Faceted Search* 允許使用者從多個維度（年份、作者、主題）篩選結果，降低認知負荷 [15]。Tunkelang (2009) 詳細討論了分面設計的最佳實踐 [16]。

**引用網路視覺化**：透過圖形呈現論文間的引用關係，幫助使用者理解領域知識結構 [17]。D3.js 與 Cytoscape.js 是常用的視覺化工具 [18]。

### 2.4 研究定位

本專案與現有工作的差異：
- **相較於商業系統**：輕量級、開源、可客製化，適合教學與研究
- **相較於 arXiv**：提供進階排序、分面瀏覽與視覺化功能
- **相較於理論研究**：強調系統實作與使用者評估，而非演算法創新

---

## 3. 系統設計 *System Design*

### 3.1 系統架構

[插入系統架構圖 SVG]

**四層架構**：
1. **展示層** *Presentation Layer*：Web UI（HTML + CSS + JavaScript）
2. **應用層** *Application Layer*：Flask REST API
3. **核心層** *Core Layer*：IR 模組（索引、檢索、排序）
4. **資料層** *Data Layer*：SQLite 資料庫 + JSON 索引

### 3.2 使用者介面設計

[插入 UI 截圖]

**核心元件**：
- **搜尋框**：支援自然語言查詢與進階語法（欄位限定）
- **結果列表**：顯示標題、作者、年份、摘要預覽、引用數
- **分面側欄**：年份、會議、作者、主題的篩選器
- **視覺化面板**：引用網路圖、主題分布圖

### 3.3 工作流程

**索引建構流程**：
```
原始資料（JSON）
  → 資料清理與正規化
  → 分詞與停用詞過濾
  → 建構倒排索引
  → 計算 TF-IDF 權重
  → 預計算文件向量
  → 儲存索引（JSON/Pickle）
  → 匯入元資料至 SQLite
```

**查詢處理流程**：
```
使用者輸入查詢
  → 查詢解析（欄位限定、布林運算子）
  → 分詞與正規化
  → 索引查找（檢索候選文件）
  → 計算相關性分數（TF-IDF）
  → 多維度排序（相關性+引用數+時效性）
  → 分面統計計算
  → 回傳結果（JSON）
  → 前端渲染
```

### 3.4 技術選型理由

[參見提案第 3.2 節，此處可補充實作後的經驗]

---

## 4. 實作細節 *Implementation*

### 4.1 索引建構模組

#### 4.1.1 倒排索引

**資料結構**：
```python
inverted_index = {
    "term": [
        {"doc_id": 1, "tf": 0.05, "positions": [10, 25, 67]},
        {"doc_id": 5, "tf": 0.08, "positions": [3, 45]},
        ...
    ],
    ...
}
```

**演算法**：
```python
def build_inverted_index(documents):
    """
    Build inverted index with positional information.

    Complexity:
        Time: O(T) where T is total tokens
        Space: O(V + P) where V is vocab size, P is postings size

    Optimizations:
        1. Use generator to process documents in batches (memory efficient)
        2. Pre-allocate dictionary with expected vocab size
        3. Use list comprehension for token extraction
    """
    index = defaultdict(list)
    for doc_id, doc in enumerate(documents):
        tokens = tokenize(doc['title'] + ' ' + doc['abstract'])
        term_positions = defaultdict(list)

        for pos, token in enumerate(tokens):
            term_positions[token].append(pos)

        for term, positions in term_positions.items():
            tf = len(positions) / len(tokens)
            index[term].append({
                'doc_id': doc_id,
                'tf': tf,
                'positions': positions
            })

    return dict(index)
```

**效能最佳化**：
- 使用 `defaultdict` 減少鍵存在性檢查
- 批次處理文件（每批 1000 筆）避免記憶體溢出
- 建構完成後轉為 `dict` 並 pickle 序列化加速載入

**結果**：
- 20,000 筆論文：建構時間 18.3 秒
- 索引大小：42.5 MB（原始資料 85 MB，壓縮率 50%）

#### 4.1.2 TF-IDF 計算

**IDF 公式**：
```
IDF(t) = log(N / df(t))
```
其中 N 是文件總數，df(t) 是包含詞彙 t 的文件數。

**實作**：
```python
def compute_idf(inverted_index, num_docs):
    """Compute IDF for all terms."""
    idf = {}
    for term, postings in inverted_index.items():
        df = len(postings)
        idf[term] = math.log(num_docs / df)
    return idf


def compute_tfidf_vectors(inverted_index, idf):
    """Precompute TF-IDF vectors for all documents."""
    doc_vectors = defaultdict(dict)

    for term, postings in inverted_index.items():
        for posting in postings:
            doc_id = posting['doc_id']
            tf = posting['tf']
            tfidf = tf * idf[term]
            doc_vectors[doc_id][term] = tfidf

    # Normalize vectors (cosine normalization)
    for doc_id, vector in doc_vectors.items():
        norm = math.sqrt(sum(v**2 for v in vector.values()))
        doc_vectors[doc_id] = {t: v/norm for t, v in vector.items()}

    return doc_vectors
```

**最佳化**：預計算並儲存所有文件的 TF-IDF 向量，查詢時只需計算查詢向量與文件向量的內積。

### 4.2 查詢處理模組

#### 4.2.1 查詢解析

**支援語法**：
```
# 自然語言查詢
machine learning neural networks

# 布林運算子
machine AND learning

# 欄位限定
title:"information retrieval" author:Manning year:2010-2020

# 組合查詢
title:"deep learning" AND (venue:NeurIPS OR venue:ICML) year:2020-
```

**解析器實作**（簡化版）：
```python
import re

def parse_query(query_string):
    """
    Parse query string into structured query object.

    Returns:
        {
            'terms': ['machine', 'learning'],
            'field_queries': {'title': ['deep learning']},
            'year_range': (2020, 2024),
            'boolean_op': 'AND'
        }
    """
    # Extract field queries
    field_pattern = r'(\w+):"([^"]+)"'
    field_queries = {}
    for field, value in re.findall(field_pattern, query_string):
        field_queries[field] = value
        query_string = query_string.replace(f'{field}:"{value}"', '')

    # Extract year range
    year_pattern = r'year:(\d{4})-(\d{4})'
    year_match = re.search(year_pattern, query_string)
    year_range = None
    if year_match:
        year_range = (int(year_match.group(1)), int(year_match.group(2)))
        query_string = re.sub(year_pattern, '', query_string)

    # Extract terms
    tokens = query_string.lower().split()
    terms = [t for t in tokens if t not in ['and', 'or', 'not']]

    # Detect boolean operator
    boolean_op = 'AND' if 'AND' in query_string else 'OR'

    return {
        'terms': terms,
        'field_queries': field_queries,
        'year_range': year_range,
        'boolean_op': boolean_op
    }
```

#### 4.2.2 相關性排序

**多維度排序公式**：
```
score(q, d) = α × relevance(q, d) + β × log(citations(d) + 1) + γ × recency(d)
```

**實作**：
```python
def compute_final_score(query_vector, doc_id, doc_vector, metadata,
                       alpha=0.6, beta=0.3, gamma=0.1):
    """
    Compute final ranking score with multiple factors.

    Args:
        query_vector: Query TF-IDF vector
        doc_vector: Document TF-IDF vector
        metadata: Document metadata (citations, year)
        alpha, beta, gamma: Weight parameters

    Returns:
        Final score (float)
    """
    # Relevance score (cosine similarity)
    relevance = cosine_similarity(query_vector, doc_vector)

    # Citation score (log-scaled)
    citations = metadata.get('citations', 0)
    citation_score = math.log(citations + 1)

    # Recency score (inverse age)
    year = metadata.get('year', 2000)
    age = 2024 - year
    recency_score = 1.0 / (1 + age)

    # Normalize each component to [0, 1]
    relevance_norm = relevance  # Already in [0, 1]
    citation_norm = min(citation_score / 10, 1.0)  # Assume max log(citations)=10
    recency_norm = recency_score

    # Weighted combination
    final_score = (alpha * relevance_norm +
                   beta * citation_norm +
                   gamma * recency_norm)

    return final_score
```

**參數調校**：使用網格搜索（*Grid Search*）在驗證集上最佳化 α, β, γ：
```
Best params: α=0.65, β=0.25, γ=0.10 (MAP=0.68)
```

### 4.3 分面瀏覽模組

**分面計算**：
```python
def compute_facets(doc_ids, metadata_db):
    """
    Compute facet counts for a set of documents.

    Args:
        doc_ids: List of document IDs in result set
        metadata_db: SQLite connection to metadata

    Returns:
        {
            'year': {2024: 15, 2023: 23, ...},
            'venue': {'NeurIPS': 8, 'ICML': 5, ...},
            'author': {'Manning, C.': 3, ...}
        }
    """
    facets = {
        'year': Counter(),
        'venue': Counter(),
        'author': Counter()
    }

    for doc_id in doc_ids:
        doc = metadata_db.get_document(doc_id)
        facets['year'][doc['year']] += 1
        facets['venue'][doc['venue']] += 1
        for author in doc['authors']:
            facets['author'][author] += 1

    # Sort and limit
    for facet_name in facets:
        facets[facet_name] = dict(facets[facet_name].most_common(20))

    return facets
```

### 4.4 後端 API 設計

**RESTful API 端點**：

```python
# Flask application
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/search', methods=['GET'])
def search():
    """
    Search endpoint.

    Query params:
        q: query string
        topk: number of results (default 20)
        offset: pagination offset (default 0)
        facets: comma-separated facet names

    Returns:
        {
            'results': [list of papers],
            'total': total result count,
            'facets': facet counts,
            'query_time': milliseconds
        }
    """
    query = request.args.get('q', '')
    topk = int(request.args.get('topk', 20))
    offset = int(request.args.get('offset', 0))

    start_time = time.time()

    # Execute search
    results = search_engine.search(query, topk=topk+offset)
    results = results[offset:offset+topk]

    # Compute facets
    doc_ids = [r['doc_id'] for r in results]
    facets = compute_facets(doc_ids, metadata_db)

    query_time = (time.time() - start_time) * 1000

    return jsonify({
        'results': results,
        'total': len(results),
        'facets': facets,
        'query_time': query_time
    })


@app.route('/api/document/<int:doc_id>', methods=['GET'])
def get_document(doc_id):
    """Get full document details."""
    doc = metadata_db.get_document(doc_id)
    return jsonify(doc)


@app.route('/api/recommend/<int:doc_id>', methods=['GET'])
def recommend(doc_id):
    """Get recommended papers for a given paper."""
    recommendations = recommender.get_similar(doc_id, topk=10)
    return jsonify({'recommendations': recommendations})
```

### 4.5 前端實作

**技術棧**：HTML5 + Bootstrap 5 + jQuery + Chart.js

**關鍵功能**：
1. **即時搜尋建議**（Auto-completion）：使用 jQuery Autocomplete
2. **分面篩選**：點選側邊欄分面，動態更新查詢並重新檢索
3. **無限捲動**（Infinite Scroll）：捲動到底部自動載入下一頁結果
4. **圖表視覺化**：使用 Chart.js 繪製年份分布圖

---

## 5. 實驗與評估 *Experiments & Evaluation*

### 5.1 實驗設定

#### 5.1.1 資料集

**資料來源**：arXiv 資料集（Kaggle）
**資料規模**：
- 原始資料：1.7M 篇論文
- 抽樣策略：選取 CS 類別（cs.*）的最近 5 年論文
- 最終資料集：20,147 篇論文
- 資料分割：訓練集 18,000 筆（用於索引）、驗證集 2,147 筆（用於測試）

**資料統計**：
- 平均標題長度：9.2 詞
- 平均摘要長度：156.3 詞
- 詞彙表大小：45,278 個唯一詞彙
- 年份分布：2020 (3,821), 2021 (4,105), 2022 (4,562), 2023 (4,289), 2024 (3,370)

#### 5.1.2 測試集建構

**查詢集設計**：
- 來源：從驗證集隨機抽取 50 篇論文的標題作為查詢
- 查詢長度：2-5 個詞
- 查詢類型：20 個一般查詢、15 個專有名詞查詢、15 個多詞查詢

**相關性標註**：
- 標註者：2 位研究生（資訊科學背景）
- 標註流程：對每個查詢，檢視前 20 筆結果，標記相關/不相關
- 標註一致性：Cohen's Kappa = 0.78（高一致性）
- 衝突解決：討論後達成共識

#### 5.1.3 基準系統

**Baseline 1：BM25**
- 使用 Elasticsearch 實作
- 預設參數：k1=1.2, b=0.75

**Baseline 2：純 TF-IDF**
- 未整合引用數與時效性
- 僅使用餘弦相似度排序

**Our System：Multi-factor TF-IDF**
- TF-IDF + 引用數 + 時效性
- 參數：α=0.65, β=0.25, γ=0.10

### 5.2 評估指標

**品質指標**：
- Precision@10：前 10 筆的精確率
- Recall@50：前 50 筆的召回率
- MAP (Mean Average Precision)：排序品質綜合指標
- nDCG@10：考慮排序位置的品質指標

**效能指標**：
- 索引建構時間
- 查詢回應時間（P50, P95, P99）
- 記憶體使用量
- 索引檔案大小

### 5.3 檢索品質結果

#### 5.3.1 整體效能

| 系統 | P@10 | Recall@50 | MAP | nDCG@10 |
|------|------|-----------|-----|---------|
| BM25 | 0.72 | 0.65 | 0.66 | 0.71 |
| Pure TF-IDF | 0.68 | 0.62 | 0.64 | 0.68 |
| **Our System** | **0.74** | **0.69** | **0.68** | **0.72** |

**觀察**：
- ✅ 多維度排序（Our System）在所有指標上優於純 TF-IDF
- ✅ nDCG@10 提升顯著（+4%），顯示排序品質改善
- ✅ Recall@50 提升 7%，表示相關文件更容易被檢索到

#### 5.3.2 不同查詢類型表現

| 查詢類型 | 數量 | BM25 MAP | Our System MAP | 改善 |
|---------|------|---------|---------------|------|
| 一般查詢 | 20 | 0.68 | 0.72 | +5.9% |
| 專有名詞 | 15 | 0.70 | 0.71 | +1.4% |
| 長查詢（4+詞） | 15 | 0.60 | 0.65 | +8.3% |

**分析**：
- 長查詢改善最明顯（+8.3%），顯示 TF-IDF 對多詞查詢的優勢
- 專有名詞查詢改善有限，因為名稱匹配已高度精確
- 一般查詢的穩定改善顯示多維度排序的普適性

#### 5.3.3 案例研究

**案例 1：成功案例**
```
查詢："neural machine translation attention mechanism"

Our System Top 3:
1. "Attention Is All You Need" (Vaswani et al., 2017) - Cited 50,000+
2. "Neural Machine Translation by Jointly Learning to Align and Translate" (Bahdanau et al., 2015)
3. "Effective Approaches to Attention-based Neural Machine Translation" (Luong et al., 2015)

BM25 Top 3:
1. 同上第 1 筆
2. [Recent low-citation paper on attention]
3. 同上第 2 筆

分析：Our System 成功將經典高引用論文排在前列，BM25 混入低相關的近期論文。
```

**案例 2：失敗案例**
```
查詢："BERT fine-tuning"

Our System Top 1:
"BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2019)

使用者期望：
關於 fine-tuning 技巧的論文（非 BERT 原始論文）

分析：系統過度重視高引用數，將原始論文排最前。改進方向：考慮查詢意圖（原理 vs. 應用）。
```

### 5.4 效能評估

#### 5.4.1 索引建構效能

| 文件數量 | 建構時間 (秒) | 索引大小 (MB) | 吞吐量 (docs/s) |
|---------|--------------|--------------|----------------|
| 5,000 | 4.7 | 10.8 | 1,064 |
| 10,000 | 9.8 | 21.5 | 1,020 |
| 20,000 | 18.3 | 42.5 | 1,093 |

**觀察**：
- 建構時間接近線性（O(n)），符合理論複雜度
- 吞吐量穩定在 1,000+ docs/s，顯示良好的可擴展性

#### 5.4.2 查詢回應時間

| 統計量 | 時間 (ms) |
|--------|-----------|
| P50（中位數） | 87 |
| P95 | 145 |
| P99 | 203 |
| 最大值 | 312 |

**分布分析**：
- 95% 的查詢在 145ms 內完成，滿足即時性要求（< 200ms）
- P99 (203ms) 略超目標，主要是複雜查詢（5+ 詞）與 OR 運算
- 最大值 (312ms) 來自極端案例（10 詞 OR 查詢）

**效能瓶頸分析**（使用 cProfile）：
1. **Cosine similarity 計算**：佔 42% CPU 時間
   - 優化方向：使用 NumPy 向量化運算
2. **分面統計計算**：佔 28% CPU 時間
   - 優化方向：預計算常見分面、使用快取
3. **SQLite 查詢**：佔 18% CPU 時間
   - 優化方向：批次查詢、增加索引

---

## 6. 使用者研究 *User Study*

### 6.1 研究設計

**目標**：驗證系統的可用性（*Usability*）與有效性（*Effectiveness*）

**參與者**：
- 招募：10 位資訊相關科系研究生
- 條件：有撰寫過文獻回顧經驗
- 報酬：300 元/人

**實驗設計**：受試者內設計（*Within-subject Design*）
- 每位參與者使用兩個系統：Our System vs. Baseline (Google Scholar)
- 順序平衡（5 人先用 Our System,5 人先用 Google Scholar）

**任務設計**：
1. **任務 1**：找出「圖神經網路」領域的 3 篇經典論文
2. **任務 2**：找出 2021 年後關於「BERT fine-tuning」的論文
3. **任務 3**：找出特定作者（如 Yann LeCun）在「深度學習」領域的論文

**測量指標**：
- 任務完成時間
- 任務成功率（找到正確論文的比例）
- 主觀滿意度（1-5 分量表）：
  - 易用性（Ease of Use）
  - 結果品質（Result Quality）
  - 介面設計（Interface Design）
  - 整體滿意度（Overall Satisfaction）

### 6.2 結果

#### 6.2.1 任務效能

| 指標 | Our System | Google Scholar | 改善 |
|------|-----------|----------------|------|
| 平均完成時間 | 3.2 分鐘 | 4.5 分鐘 | **-28.9%** |
| 任務成功率 | 93.3% | 76.7% | **+21.6%** |

**統計檢定**：
- 完成時間：配對 t 檢定，t(9) = 3.45, p = 0.007（顯著）
- 成功率：McNemar 檢定，p = 0.031（顯著）

**分析**：
- Our System 的分面篩選功能顯著加速任務 2（限定年份）
- 任務 3 中，Our System 的作者欄位搜尋表現優異

#### 6.2.2 主觀滿意度

| 維度 | Our System | Google Scholar |
|------|-----------|----------------|
| 易用性 | 4.3 | 3.8 |
| 結果品質 | 4.4 | 4.1 |
| 介面設計 | 4.2 | 3.5 |
| **整體滿意度** | **4.2** | **3.7** |

**質性回饋**：
- ✅ "分面篩選很直覺，不用記複雜語法"（P3）
- ✅ "引用數顯示很有用，快速識別重要論文"（P7）
- ⚠️ "希望能匯出 BibTeX"（P2, P5, P9）
- ⚠️ "搜尋建議有時不準確"（P6）

---

## 7. 討論與反思 *Discussion & Reflection*

### 7.1 研究發現

**RQ1：直覺的檢索介面**
- 分面瀏覽有效降低使用門檻，使用者研究顯示任務完成時間減少 29%
- 欄位搜尋（author:, year:）比布林運算子更易理解

**RQ2：多維度排序**
- 整合引用數與時效性使 MAP 提升 4%，nDCG@10 提升 4%
- 引用數對經典論文檢索幫助大，但可能壓抑新興研究（trade-off）

**RQ3：探索式發現**
- 視覺化功能（引用網路圖）受使用者好評，但實作複雜度高
- 分面瀏覽是最實用的探索工具

**RQ4：輕量級系統的可行性**
- 查詢回應時間 P95 = 145ms，滿足即時性要求
- 20K 論文規模下，自行實作的效能可與商業系統相比
- 但擴展至百萬級需要分散式架構

### 7.2 限制與挑戰

**資料規模限制**：
- 當前系統僅處理 20K 論文，遠小於 Google Scholar 的規模
- 記憶體限制（索引完全載入）難以擴展至百萬級

**演算法簡化**：
- 使用線性組合的多維度排序，未採用 Learning to Rank
- 查詢擴展僅實作 Rocchio，未整合語意擴展（Word2Vec）

**評估資料集小**：
- 僅 50 個測試查詢，可能無法涵蓋所有查詢類型
- 相關性標註由 2 位標註者完成，規模有限

**使用者研究規模**：
- 僅 10 位參與者，統計檢定力（*Statistical Power*）有限
- 參與者同質性高（資訊科系研究生），缺乏多樣性

### 7.3 經驗教訓

**技術層面**：
1. **提前最佳化是萬惡之源**：初期花過多時間優化索引壓縮，後來發現 20K 規模不需要
2. **測試驅動開發**：先寫測試案例再實作，大幅減少除錯時間
3. **版本控制**：頻繁 commit 拯救多次誤刪代碼的慘劇

**專案管理**：
1. **時程預估不足**：前端開發花費時間超出預期（原估 1 週,實際 2 週）
2. **文件的重要性**：清晰的 API 文件使前後端並行開發順暢

**學習收穫**：
1. 深入理解 IR 理論與實務的差距（理論 O(n) vs. 實務常數項）
2. 體會完整系統開發的挑戰（不只是演算法，還有工程、UI、評估）
3. 使用者研究的價值（發現多項自己未察覺的可用性問題）

---

## 8. 結論與未來工作 *Conclusion & Future Work*

### 8.1 總結

本專案成功開發了一個輕量級學術論文檢索系統 AcademicSearch，整合倒排索引、TF-IDF 排序、多維度排序、分面瀏覽等 IR 技術。實驗評估顯示，系統在 20,000 筆論文上達到 MAP 0.68、nDCG@10 0.72，查詢回應時間 P95 為 145ms，滿足即時檢索需求。使用者研究顯示，相較於基準系統（Google Scholar），本系統的任務完成時間減少 29%，成功率提升 22%，整體滿意度達 4.2/5。

本專案展示了課程所學 IR 理論在真實應用場景的整合與實踐，證明自行實作的輕量級系統在中小規模資料集上的可行性。

### 8.2 未來工作

**短期改進**（1-2 個月）：
- [ ] 實作 BibTeX/RIS 匯出功能
- [ ] 改進搜尋建議準確性（使用 Trie 或字首樹）
- [ ] 加入使用者帳戶與收藏功能
- [ ] 優化效能（向量化計算、快取熱門查詢）

**中期擴展**（3-6 個月）：
- [ ] 整合更多資料來源（DBLP, Semantic Scholar API）
- [ ] 實作 Learning to Rank 模型
- [ ] 加入個人化推薦（基於閱讀歷史）
- [ ] 支援多語言（中文論文檢索）

**長期研究方向**（6 個月以上）：
- [ ] 探索神經檢索模型（BERT-based Dense Retrieval）
- [ ] 建立分散式索引架構（Elasticsearch, Solr）
- [ ] 開發行動版應用（React Native）
- [ ] 發表系統論文於 SIGIR, CIKM 等會議

### 8.3 最後的話

這個專案是我在資訊檢索課程中最具挑戰性也最有成就感的學習經驗。從最初的倒排索引實作，到最終的完整系統，每個階段都讓我對 IR 的理解更深一層。特別感謝授課教師的指導，以及參與使用者研究的同學們的寶貴回饋。

希望這個開源專案能為未來學習 IR 的同學提供實作參考，也期待在學術搜尋引擎領域看到更多創新。

---

## 9. 參考文獻 *References*

[1] Google Scholar. https://scholar.google.com

[2] Sinha, A., et al. (2015). An Overview of Microsoft Academic Service (MAS) and Applications. *WWW*.

[3] Semantic Scholar. https://www.semanticscholar.org

[4] PubMed. https://pubmed.ncbi.nlm.nih.gov

[5] arXiv.org. https://arxiv.org

[6] Zobel, J., & Moffat, A. (2006). Inverted files for text search engines. *ACM Computing Surveys*, 38(2), 6.

[7] Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.

[8] Büttcher, S., Clarke, C. L., & Cormack, G. V. (2016). *Information Retrieval: Implementing and Evaluating Search Engines*. MIT Press.

[9] Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. *Information Processing & Management*, 24(5), 513-523.

[10] Robertson, S., & Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. *Foundations and Trends in Information Retrieval*, 3(4), 333-389.

[11] Liu, T. Y. (2009). Learning to rank for information retrieval. *Foundations and Trends in Information Retrieval*, 3(3), 225-331.

[12] Rocchio, J. J. (1971). Relevance feedback in information retrieval. *The SMART Retrieval System*, 313-323.

[13] Xu, J., & Croft, W. B. (1996). Query expansion using local and global document analysis. *SIGIR*.

[14] Järvelin, K., & Kekäläinen, J. (2002). Cumulated gain-based evaluation of IR techniques. *ACM Transactions on Information Systems*, 20(4), 422-446.

[15] Hearst, M. A., et al. (2002). Finding the flow in web site search. *Communications of the ACM*, 45(9), 42-49.

[16] Tunkelang, D. (2009). *Faceted Search*. Morgan & Claypool Publishers.

[17] Chen, C. (2006). CiteSpace II: Detecting and visualizing emerging trends and transient patterns in scientific literature. *Journal of the American Society for Information Science and Technology*, 57(3), 359-377.

[18] Bostock, M., Ogievetsky, V., & Heer, J. (2011). D³ data-driven documents. *IEEE Transactions on Visualization and Computer Graphics*, 17(12), 2301-2309.

---

## 10. 附錄 *Appendices*

### 附錄 A：系統截圖

[插入完整的 UI 截圖集]

### 附錄 B：使用者研究問卷

[附上完整的問卷內容]

### 附錄 C：API 文件

**完整 API 規格**：https://github.com/username/academic-search/blob/main/docs/API.md

### 附錄 D：程式碼統計

```
Language                 files          blank        comment           code
-------------------------------------------------------------------------------
Python                      25           1847           2134           5823
JavaScript                   8            456            312           1987
HTML                         5            123             45            876
CSS                          3             89             12            456
Markdown                     7            234              0           1456
-------------------------------------------------------------------------------
SUM:                        48           2749           2503          10598
-------------------------------------------------------------------------------
```

**測試覆蓋率**：
```
Name                                 Stmts   Miss  Cover
--------------------------------------------------------
src/ir/index/inverted_index.py        156      8    95%
src/ir/retrieval/search.py             234     18    92%
src/ir/ranking/multi_factor.py         98      5    95%
src/ir/facets/facets.py               112     12    89%
--------------------------------------------------------
TOTAL                                2847    234    92%
```

---

**報告完成日期**：2025-01-15
**總頁數**：25 頁
**總字數**：約 15,000 字（中英文混合）

**團隊成員貢獻聲明**：
- [姓名1]：索引建構 (40%)、查詢引擎 (30%)、實驗評估 (15%)、報告撰寫 (15%)
- [姓名2]：資料庫設計 (20%)、前端UI (30%)、視覺化 (20%)、使用者研究 (15%)、報告撰寫 (15%)

**誠信聲明**：本報告為團隊成員共同完成，所有引用文獻均已標註，無抄襲或造假行為。

---

**指導教師評語欄**：

評分：______ / 100

評語：

<br><br><br>

---

**簽名**：__________________
**日期**：__________________
