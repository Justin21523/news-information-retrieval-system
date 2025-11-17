# IR System API 文檔 (API Documentation)

完整的資訊檢索系統 REST API 文檔。

**Base URL**: `http://localhost:5001`

---

## 📊 系統統計 (System Stats)

### GET `/api/stats`

獲取系統統計資訊。

**Response**:
```json
{
    "documents": 121,
    "vocabulary_size": 8478,
    "avg_doc_length": 245.6,
    "total_terms": 29734
}
```

---

## 🔍 檢索 API (Search APIs)

### 1. Boolean Search (布林檢索)

**Endpoint**: `POST /api/search/boolean`

**Request**:
```json
{
    "query": "台灣 AND 經濟",
    "limit": 10
}
```

**支援運算子**:
- `AND`, `OR`, `NOT`
- `NEAR/n`: 鄰近查詢 (e.g., `資訊 NEAR/3 檢索`)
- 括號: `(台灣 OR 中國) AND 經濟`
- 欄位查詢: `title:AI`, `category:科技`
- 日期範圍: `published_date:[2025-11-01 TO 2025-11-13]`
- 通配符: `info*`, `te?t`

**Response**:
```json
{
    "query": "台灣 AND 經濟",
    "results": [
        {
            "doc_id": 5,
            "title": "...",
            "snippet": "...",
            "url": "...",
            "date": "2025-11-13",
            "category": "財經"
        }
    ],
    "total": 15,
    "execution_time": 0.023
}
```

---

### 2. VSM Search (向量空間模型)

**Endpoint**: `POST /api/search/vsm`

使用 TF-IDF 和餘弦相似度排序。

**Request**:
```json
{
    "query": "人工智慧發展",
    "limit": 10
}
```

**Response**:
```json
{
    "query": "人工智慧發展",
    "model": "VSM",
    "results": [
        {
            "doc_id": 12,
            "title": "...",
            "snippet": "...",
            "score": 0.8542,
            "url": "...",
            "date": "2025-11-12",
            "category": "科技"
        }
    ],
    "total": 10,
    "execution_time": 0.045
}
```

---

### 3. BM25 Search (BM25 排序)

**Endpoint**: `POST /api/search/bm25`

使用 BM25 機率排序函數 (k1=1.5, b=0.75)。

**Request**:
```json
{
    "query": "深度學習應用",
    "limit": 10
}
```

**Response**:
```json
{
    "query": "深度學習應用",
    "model": "BM25",
    "results": [
        {
            "doc_id": 8,
            "title": "...",
            "snippet": "...",
            "score": 15.2345,
            "url": "...",
            "date": "2025-11-11",
            "category": "科技"
        }
    ],
    "total": 10,
    "parameters": {
        "k1": 1.5,
        "b": 0.75,
        "delta": 0.0
    },
    "execution_time": 0.038
}
```

---

### 4. Language Model Search (語言模型檢索)

**Endpoint**: `POST /api/search/lm`

使用查詢可能性模型 (Query Likelihood) 與 Dirichlet 平滑 (μ=2000)。

**Request**:
```json
{
    "query": "機器學習應用",
    "limit": 10
}
```

**Response**:
```json
{
    "query": "機器學習應用",
    "model": "Language Model",
    "results": [
        {
            "doc_id": 15,
            "title": "...",
            "snippet": "...",
            "score": -12.3456,
            "url": "...",
            "date": "2025-11-10",
            "category": "科技"
        }
    ],
    "total": 10,
    "parameters": {
        "smoothing": "dirichlet",
        "lambda": 0.7,
        "mu": 2000,
        "delta": 0.7
    },
    "execution_time": 0.052
}
```

---

### 5. Hybrid Search (混合排序)

**Endpoint**: `POST /api/search/hybrid`

結合多個檢索模型 (BM25 + VSM + LM + BERT*) 的混合排序。

**Request**:
```json
{
    "query": "自然語言處理",
    "limit": 10,
    "fusion_method": "rrf"
}
```

**融合方法** (`fusion_method`):
- `linear`: 線性組合 (需要分數正規化)
- `rrf`: Reciprocal Rank Fusion (推薦,預設)
- `combsum`: 分數總和
- `combmnz`: 分數總和 × 匹配數量

**Response**:
```json
{
    "query": "自然語言處理",
    "model": "Hybrid",
    "results": [
        {
            "doc_id": 22,
            "title": "...",
            "snippet": "...",
            "score": 0.0234,
            "url": "...",
            "date": "2025-11-09",
            "category": "科技"
        }
    ],
    "total": 10,
    "fusion_method": "rrf",
    "weights": {
        "bm25": 0.333,
        "vsm": 0.333,
        "lm": 0.334
    },
    "component_scores": {
        "bm25": [15.23, 12.45, 10.67, ...],
        "vsm": [0.85, 0.78, 0.72, ...],
        "lm": [-10.2, -11.5, -12.3, ...]
    },
    "execution_time": 0.125
}
```

---

## 📄 文檔操作 (Document Operations)

### Get Document Details (獲取文檔詳情)

**Endpoint**: `GET /api/document/<doc_id>`

**Example**: `GET /api/document/5`

**Response**:
```json
{
    "id": 5,
    "title": "...",
    "content": "...",
    "url": "...",
    "published_date": "2025-11-13",
    "category_name": "財經",
    "author": "...",
    "summary": "...",
    "tags": ["台灣", "經濟", "出口"]
}
```

---

### Summarize Document (文檔摘要)

**Endpoint**: `POST /api/summarize/<doc_id>`

**Request**:
```json
{
    "method": "lead_k",
    "k": 3,
    "keyword": "台灣"
}
```

**方法** (`method`):
- `lead_k`: 前 k 句
- `key_sentence`: 關鍵句提取
- `kwic`: 關鍵詞上下文 (需要 `keyword`)

**Response (lead_k/key_sentence)**:
```json
{
    "method": "lead_k",
    "k": 3,
    "summary": [
        "第一句摘要...",
        "第二句摘要...",
        "第三句摘要..."
    ]
}
```

**Response (KWIC)**:
```json
{
    "method": "kwic",
    "keyword": "台灣",
    "contexts": [
        {
            "left": "...上下文左邊...",
            "keyword": "台灣",
            "right": "...上下文右邊...",
            "position": 45
        }
    ]
}
```

---

## 🔧 進階功能 (Advanced Features)

### Query Expansion (查詢擴展)

**Endpoint**: `POST /api/expand_query`

使用 Rocchio 演算法進行查詢擴展。

**Request**:
```json
{
    "query": "人工智慧",
    "relevant_docs": [0, 1, 2]
}
```

**Response**:
```json
{
    "original_query": "人工智慧",
    "expansion_terms": [
        {"term": "機器學習", "weight": 0.8542},
        {"term": "深度學習", "weight": 0.7234},
        {"term": "神經網路", "weight": 0.6891}
    ],
    "relevant_docs": [0, 1, 2]
}
```

---

### Document Clustering (文檔分群)

**Endpoint**: `POST /api/cluster`

**Request**:
```json
{
    "n_clusters": 3,
    "method": "hierarchical",
    "doc_ids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
}
```

**方法** (`method`):
- `hierarchical`: 階層式分群
- `kmeans`: K-means 分群

**Response**:
```json
{
    "method": "hierarchical",
    "n_clusters": 3,
    "clusters": [
        {
            "cluster_id": 0,
            "size": 5,
            "doc_ids": [0, 2, 4, 7, 9],
            "documents": [
                {"doc_id": 0, "title": "..."},
                {"doc_id": 2, "title": "..."}
            ]
        }
    ]
}
```

---

## 📊 語言模型分析 (Language Model Analysis)

### Collocation Extraction (詞彙共現分析)

**Endpoint**: `POST /api/analyze/collocation`

提取顯著的詞彙組合 (bigrams)。

**Request**:
```json
{
    "measure": "pmi",
    "topk": 20
}
```

**統計量測** (`measure`):
- `pmi`: Pointwise Mutual Information
- `llr`: Log-Likelihood Ratio
- `chi_square`: Chi-Square (χ²)
- `t_score`: T-Score
- `dice`: Dice Coefficient

**Response**:
```json
{
    "measure": "pmi",
    "topk": 20,
    "collocations": [
        {
            "bigram": "人工 智慧",
            "freq": 25,
            "pmi": 8.5432,
            "llr": 156.23,
            "chi_square": 234.56,
            "t_score": 4.89,
            "dice": 0.7654
        }
    ]
}
```

---

### N-gram Analysis (N-gram 分析)

**Endpoint**: `POST /api/analyze/ngram`

計算文本的語言模型機率或困惑度 (perplexity)。

**Request (Perplexity)**:
```json
{
    "text": "資訊檢索系統",
    "calculate": "perplexity"
}
```

**Request (Probability)**:
```json
{
    "text": "機器學習應用",
    "calculate": "probability"
}
```

**Response (Perplexity)**:
```json
{
    "text": "資訊檢索系統",
    "perplexity": 45.2341,
    "n": 2,
    "smoothing": "jm"
}
```

**Response (Probability)**:
```json
{
    "text": "機器學習應用",
    "probability": 1.2345e-08,
    "log_probability": -18.2134,
    "n": 2,
    "smoothing": "jm"
}
```

---

## 🎯 完整檢索範例 (Complete Examples)

### Example 1: 多模型比較

```python
import requests

BASE_URL = "http://localhost:5001"
query = "人工智慧發展"

# 比較不同檢索模型
models = [
    ('boolean', '/api/search/boolean'),
    ('vsm', '/api/search/vsm'),
    ('bm25', '/api/search/bm25'),
    ('lm', '/api/search/lm'),
    ('hybrid', '/api/search/hybrid')
]

for model_name, endpoint in models:
    response = requests.post(
        BASE_URL + endpoint,
        json={'query': query, 'limit': 5}
    )
    result = response.json()

    print(f"\n{model_name.upper()}:")
    print(f"  Results: {result['total']}")
    print(f"  Time: {result['execution_time']:.3f}s")

    if 'results' in result:
        for i, doc in enumerate(result['results'][:3], 1):
            print(f"  {i}. {doc['title'][:50]}...")
```

### Example 2: 詞彙共現分析

```python
# 提取顯著的詞彙組合
response = requests.post(
    "http://localhost:5001/api/analyze/collocation",
    json={'measure': 'pmi', 'topk': 10}
)

collocations = response.json()['collocations']

print("Top 10 Collocations (PMI):")
for col in collocations:
    print(f"  {col['bigram']}: PMI={col['pmi']:.2f}, freq={col['freq']}")
```

### Example 3: 混合排序自訂權重

```python
# 使用線性組合,自訂權重
response = requests.post(
    "http://localhost:5001/api/search/hybrid",
    json={
        'query': '深度學習',
        'limit': 10,
        'fusion_method': 'linear'
        # 注意: 權重目前在初始化時設定,未來可支援動態調整
    }
)

result = response.json()
print(f"Fusion: {result['fusion_method']}")
print(f"Weights: {result['weights']}")
print(f"Top result: {result['results'][0]['title']}")
```

---

## ⚙️ 模型參數說明

### BM25 參數
- **k1** (default: 1.5): 詞頻飽和參數 (1.2 - 2.0)
- **b** (default: 0.75): 長度正規化參數 (0 - 1)
- **delta** (default: 0.0): BM25+ 參數

### Language Model 參數
- **smoothing**: 平滑方法 (`jm`, `dirichlet`, `absolute`)
- **lambda_param** (default: 0.7): Jelinek-Mercer λ (0 - 1)
- **mu_param** (default: 2000): Dirichlet μ (500 - 5000)

### Hybrid Ranker 參數
- **fusion_method**: 融合策略 (`linear`, `rrf`, `combsum`, `combmnz`)
- **normalization**: 分數正規化 (`minmax`, `zscore`, `none`)
- **weights**: 各 ranker 權重 (自動正規化為總和 = 1)

### N-gram Model 參數
- **n** (default: 2): N-gram 階數
- **smoothing**: 平滑方法 (`laplace`, `jm`, `dirichlet`)

---

## 🚀 快速開始

1. **啟動伺服器**:
```bash
python app.py
```

2. **測試 API**:
```bash
# 健康檢查
curl http://localhost:5001/api/stats

# 簡單查詢
curl -X POST http://localhost:5001/api/search/bm25 \
  -H "Content-Type: application/json" \
  -d '{"query": "台灣經濟", "limit": 5}'
```

3. **查看文檔**:
瀏覽器開啟 `http://localhost:5001/` 查看 Web UI。

---

## 📝 錯誤處理

所有 API 在錯誤時返回:
```json
{
    "error": "錯誤訊息描述"
}
```

常見 HTTP 狀態碼:
- `400`: 請求參數錯誤
- `404`: 資源不存在
- `500`: 伺服器內部錯誤

---

## 🔤 關鍵字提取 API (Keyword Extraction APIs)

### POST `/api/extract/keywords`

從文本中提取關鍵字,支援多種演算法。

**Request**:
```json
{
    "text": "機器學習是人工智慧的重要分支...",
    "method": "textrank",
    "topk": 10,
    "use_pos_filter": false,
    "use_ner_boost": false
}
```

**Parameters**:
- `text` (string, required): 輸入文本
- `method` (string): 提取方法
  - `textrank`: TextRank 圖式演算法 (預設)
  - `yake`: YAKE 統計方法
  - `keybert`: KeyBERT 語義方法 (需要 sentence-transformers)
  - `rake`: RAKE 快速關鍵字提取
- `topk` (int): 返回前 k 個關鍵字 (預設: 10)
- `use_pos_filter` (bool): 是否只保留名詞和動詞 (預設: false)
- `use_ner_boost` (bool): 是否增強命名實體權重 (預設: false, 僅 TextRank)

**Response**:
```json
{
    "method": "textrank",
    "topk": 10,
    "keywords": [
        {
            "keyword": "機器學習",
            "score": 0.2341,
            "frequency": 3,
            "positions": [5, 42, 98]
        },
        {
            "keyword": "人工智慧",
            "score": 0.1872,
            "frequency": 2,
            "positions": [15, 67]
        }
    ],
    "execution_time": 0.156
}
```

---

## 📊 主題建模 API (Topic Modeling APIs)

### POST `/api/extract/topics`

從文檔集合中提取潛在主題。

**Request**:
```json
{
    "documents": ["doc1...", "doc2...", "doc3..."],
    "method": "lda",
    "n_topics": 5,
    "model_params": {
        "iterations": 50,
        "passes": 10
    }
}
```

**Parameters**:
- `documents` (array, required): 文檔列表 (最少 3 篇)
- `method` (string): 主題建模方法
  - `lda`: Latent Dirichlet Allocation (預設)
  - `bertopic`: BERTopic (基於 BERT embeddings)
- `n_topics` (int): 主題數量 (預設: 5)
- `model_params` (object): 模型特定參數
  - LDA: `iterations`, `passes`
  - BERTopic: `calculate_probabilities`

**Response (LDA)**:
```json
{
    "method": "lda",
    "n_topics": 5,
    "topics": [
        {
            "topic_id": 0,
            "words": [
                {"word": "學習", "prob": 0.0523},
                {"word": "模型", "prob": 0.0431},
                {"word": "訓練", "prob": 0.0389}
            ]
        }
    ],
    "topic_proportions": [
        {"Topic": 0, "Words": "學習, 模型, 訓練", "Proportion": 0.24}
    ],
    "document_topics": [
        {
            "doc_index": 0,
            "topics": [[0, 0.65], [1, 0.25], [2, 0.10]]
        }
    ],
    "metrics": {
        "perplexity": 145.23,
        "coherence": 0.4521
    }
}
```

---

## 🔍 模式挖掘 API (Pattern Mining API)

### POST `/api/extract/patterns`

使用 PAT-tree 提取頻繁模式。

**Request**:
```json
{
    "texts": ["text1...", "text2...", "text3..."],
    "min_pattern_length": 2,
    "max_pattern_length": 5,
    "min_frequency": 2,
    "topk": 20,
    "use_mi_score": true
}
```

**Parameters**:
- `texts` (array, required): 文本列表
- `min_pattern_length` (int): 最小模式長度 (token 數, 預設: 2)
- `max_pattern_length` (int): 最大模式長度 (預設: 5)
- `min_frequency` (int): 最小出現頻率 (預設: 2)
- `topk` (int): 返回前 k 個模式 (預設: 20)
- `use_mi_score` (bool): 使用 Mutual Information 排序 (預設: true)

**Response**:
```json
{
    "patterns": [
        {
            "pattern": "機器學習",
            "tokens": ["機器", "學習"],
            "frequency": 5,
            "mi_score": 8.543,
            "positions": [0, 15, 42, 67, 89]
        }
    ],
    "statistics": {
        "total_tokens": 450,
        "unique_tokens": 156,
        "total_nodes": 892
    },
    "parameters": {
        "min_pattern_length": 2,
        "max_pattern_length": 5,
        "min_frequency": 2,
        "use_mi_score": true
    }
}
```

---

## 🏷️ 命名實體識別 API (NER API)

### POST `/api/analyze/ner`

識別文本中的命名實體。

**Request**:
```json
{
    "text": "台積電在台灣新竹科學園區成立於1987年,創辦人是張忠謀",
    "entity_types": ["PERSON", "ORG", "GPE", "LOC", "DATE"]
}
```

**Parameters**:
- `text` (string, required): 輸入文本
- `entity_types` (array, optional): 要識別的實體類型
  - `PERSON`: 人名
  - `ORG`: 組織機構
  - `GPE`: 地緣政治實體 (國家、城市)
  - `LOC`: 地點
  - `DATE`: 日期
  - `QUANTITY`: 數量
  - `CARDINAL`: 基數
  - 省略則識別所有類型

**Response**:
```json
{
    "text": "台積電在台灣新竹科學園區成立於1987年,創辦人是張忠謀",
    "entities": [
        {
            "text": "台積電",
            "type": "ORG",
            "start": 0,
            "end": 3,
            "confidence": 0.9876
        },
        {
            "text": "台灣",
            "type": "GPE",
            "start": 4,
            "end": 6,
            "confidence": 0.9654
        },
        {
            "text": "張忠謀",
            "type": "PERSON",
            "start": 23,
            "end": 26,
            "confidence": 0.9912
        }
    ],
    "entity_count": 3,
    "entity_types": ["ORG", "GPE", "PERSON"],
    "entities_by_type": {
        "ORG": ["台積電"],
        "GPE": ["台灣", "新竹"],
        "PERSON": ["張忠謀"]
    }
}
```

---

## 🌳 句法分析 API (Syntax Analysis API)

### POST `/api/analyze/syntax`

進行句法分析,提取依存關係或 SVO 三元組。

**Request**:
```json
{
    "text": "台積電在台灣生產先進的半導體晶片",
    "analysis_type": "svo"
}
```

**Parameters**:
- `text` (string, required): 輸入文本
- `analysis_type` (string): 分析類型
  - `svo`: 提取 Subject-Verb-Object 三元組 (預設)
  - `dependencies`: 完整依存句法分析

**Response (SVO)**:
```json
{
    "text": "台積電在台灣生產先進的半導體晶片",
    "analysis_type": "svo",
    "triples": [
        {
            "subject": "台積電",
            "verb": "生產",
            "object": "晶片",
            "confidence": 0.92
        }
    ],
    "triple_count": 1
}
```

**Response (Dependencies)**:
```json
{
    "text": "台積電在台灣生產先進的半導體晶片",
    "analysis_type": "dependencies",
    "dependencies": [
        {
            "head": "生產",
            "relation": "nsubj",
            "dependent": "台積電",
            "head_pos": "VV",
            "dep_pos": "Nb"
        },
        {
            "head": "生產",
            "relation": "dobj",
            "dependent": "晶片",
            "head_pos": "VV",
            "dep_pos": "Na"
        }
    ],
    "dependency_count": 5
}
```

---

## 📄 文檔綜合分析 API (Document Analysis API)

### GET `/api/document/<doc_id>/analysis`

獲取文檔的綜合語言學分析結果。

**Request**:
```
GET /api/document/0/analysis
```

**Response**:
```json
{
    "doc_id": 0,
    "title": "台積電宣布新技術突破",
    "analysis": {
        "keywords": [
            {"word": "台積電", "score": 0.3421},
            {"word": "技術", "score": 0.2876},
            {"word": "突破", "score": 0.2341}
        ],
        "entities": [
            {"text": "台積電", "type": "ORG"},
            {"text": "台灣", "type": "GPE"}
        ],
        "linguistic": {
            "tokens": ["台積電", "宣布", "新", "技術", "突破"],
            "pos_tags": ["Nb", "VE", "A", "Na", "VJ"],
            "sentence_count": 5,
            "word_count": 245
        }
    }
}
```

---

## 📋 API 使用範例 (Usage Examples)

### Python 範例

```python
import requests

BASE_URL = "http://localhost:5001"

# 1. 關鍵字提取
response = requests.post(
    f"{BASE_URL}/api/extract/keywords",
    json={
        "text": "機器學習是人工智慧的重要分支",
        "method": "textrank",
        "topk": 5
    }
)
keywords = response.json()['keywords']

# 2. 主題建模
response = requests.post(
    f"{BASE_URL}/api/extract/topics",
    json={
        "documents": ["doc1...", "doc2...", "doc3..."],
        "method": "lda",
        "n_topics": 3
    }
)
topics = response.json()['topics']

# 3. 命名實體識別
response = requests.post(
    f"{BASE_URL}/api/analyze/ner",
    json={
        "text": "台積電在新竹成立",
        "entity_types": ["ORG", "LOC"]
    }
)
entities = response.json()['entities']
```

### curl 範例

```bash
# 關鍵字提取
curl -X POST http://localhost:5001/api/extract/keywords \
  -H "Content-Type: application/json" \
  -d '{
    "text": "人工智慧發展迅速",
    "method": "textrank",
    "topk": 5
  }'

# NER
curl -X POST http://localhost:5001/api/analyze/ner \
  -H "Content-Type: application/json" \
  -d '{
    "text": "台積電在台灣新竹成立",
    "entity_types": ["ORG", "GPE"]
  }'
```

---

## 🎯 API 效能指標 (Performance Metrics)

| API 端點 | 平均響應時間 | 複雜度 | 備註 |
|---------|------------|--------|------|
| `/api/extract/keywords` (TextRank) | ~200ms | O(V²+I×V) | V=詞彙數, I=迭代次數 |
| `/api/extract/keywords` (YAKE) | ~100ms | O(n×m) | n=文本長度, m=n-gram |
| `/api/extract/topics` (LDA) | ~2-5s | O(K×D×N×I) | K=主題數, D=文檔數 |
| `/api/extract/patterns` | ~500ms | O(n²) | n=文本長度 |
| `/api/analyze/ner` | ~300ms | O(n) | 使用 CKIP Transformers |
| `/api/analyze/syntax` | ~400ms | O(n²) | 依存句法分析 |

---

## 🎯 推薦系統 API (Recommendation System APIs)

### 1. 內容推薦 - 相似文檔 (Content-Based - Similar Documents)

**Endpoint**: `POST /api/recommend/similar`

基於內容相似性推薦相關文檔 (Content similarity recommendations)。

**Request**:
```json
{
    "doc_id": 5,
    "top_k": 10,
    "use_embeddings": false,
    "apply_diversity": true,
    "diversity_lambda": 0.3
}
```

**參數說明**:
- `doc_id`: 源文檔 ID (required)
- `top_k`: 返回結果數量 (default: 10)
- `use_embeddings`: 使用 BERT embeddings (default: false, 使用 TF-IDF)
- `apply_diversity`: 應用多樣性重排序 MMR (default: true)
- `diversity_lambda`: 多樣性參數 λ ∈ [0,1] (default: 0.3)

**Response**:
```json
{
    "doc_id": 5,
    "method": "content_based_similarity",
    "recommendations": [
        {
            "doc_id": 12,
            "score": 0.8542,
            "title": "相關新聞標題...",
            "category": "科技",
            "similarity": 0.8542,
            "reason": "High content similarity"
        }
    ],
    "parameters": {
        "top_k": 10,
        "use_embeddings": false,
        "apply_diversity": true
    },
    "computation_time": 0.023
}
```

**Python 範例**:
```python
import requests

response = requests.post('http://localhost:5001/api/recommend/similar', json={
    "doc_id": 5,
    "top_k": 10,
    "use_embeddings": False,
    "apply_diversity": True
})

recs = response.json()['recommendations']
for rec in recs[:3]:
    print(f"Doc {rec['doc_id']}: {rec['title'][:50]} (score: {rec['score']:.4f})")
```

**curl 範例**:
```bash
curl -X POST http://localhost:5001/api/recommend/similar \
  -H "Content-Type: application/json" \
  -d '{"doc_id": 5, "top_k": 10, "apply_diversity": true}'
```

---

### 2. 個人化推薦 (Personalized Recommendations)

**Endpoint**: `POST /api/recommend/personalized`

基於閱讀歷史的個人化推薦 (Personalized recommendations based on reading history)。

**Request**:
```json
{
    "reading_history": [0, 1, 5, 10, 15],
    "top_k": 10,
    "use_embeddings": false,
    "diversity_lambda": 0.3
}
```

**參數說明**:
- `reading_history`: 已閱讀文檔 ID 列表 (required, 非空)
- `top_k`: 返回結果數量 (default: 10)
- `use_embeddings`: 使用 BERT embeddings (default: false)
- `diversity_lambda`: 多樣性參數 (default: 0.3)

**Response**:
```json
{
    "reading_history": [0, 1, 5, 10, 15],
    "method": "personalized_content_based",
    "recommendations": [
        {
            "doc_id": 23,
            "score": 0.7892,
            "title": "推薦文章標題...",
            "category": "財經",
            "reason": "Matches your reading profile"
        }
    ],
    "user_profile_docs": 5,
    "computation_time": 0.018
}
```

**Python 範例**:
```python
# 基於用戶閱讀歷史推薦
reading_history = [0, 1, 5, 10, 15]  # 用戶已讀文檔

response = requests.post('http://localhost:5001/api/recommend/personalized', json={
    "reading_history": reading_history,
    "top_k": 10,
    "use_embeddings": False
})

recs = response.json()['recommendations']
print(f"Based on {len(reading_history)} documents you've read:")
for i, rec in enumerate(recs[:5], 1):
    print(f"{i}. {rec['title']} (score: {rec['score']:.4f})")
```

---

### 3. 熱門推薦 (Trending Recommendations)

**Endpoint**: `GET /api/recommend/trending`

推薦熱門或最新文檔 (Trending/popular documents)。

**Query Parameters**:
- `top_k`: 返回結果數量 (default: 10)
- `time_window_hours`: 時間窗口(小時) (default: 168 = 7天)
- `category`: 可選類別篩選

**Request**:
```bash
GET /api/recommend/trending?top_k=10&time_window_hours=168&category=科技
```

**Response**:
```json
{
    "method": "trending",
    "recommendations": [
        {
            "doc_id": 45,
            "score": 0.9123,
            "title": "最新科技新聞...",
            "category": "科技",
            "published_date": "2025-11-13",
            "reason": "Trending in last 7 days"
        }
    ],
    "time_window_hours": 168,
    "category_filter": "科技",
    "computation_time": 0.005
}
```

**Python 範例**:
```python
# 獲取最近7天的熱門科技新聞
response = requests.get('http://localhost:5001/api/recommend/trending', params={
    "top_k": 10,
    "time_window_hours": 168,
    "category": "科技"
})

trending = response.json()['recommendations']
for doc in trending:
    print(f"{doc['title']} (發布: {doc['published_date']})")
```

---

### 4. 協同過濾 - 基於用戶 (Collaborative Filtering - User-Based)

**Endpoint**: `POST /api/recommend/cf/user-based`

基於相似用戶的協同過濾推薦 (User-based collaborative filtering)。

**Request**:
```json
{
    "user_id": 0,
    "top_k": 10,
    "n_neighbors": 20,
    "similarity_metric": "cosine"
}
```

**參數說明**:
- `user_id`: 用戶 ID (required)
- `top_k`: 返回結果數量 (default: 10)
- `n_neighbors`: 考慮的相似用戶數 (default: 20)
- `similarity_metric`: 相似度度量 `cosine` 或 `pearson` (default: cosine)

**Response**:
```json
{
    "user_id": 0,
    "method": "user_based_cf",
    "recommendations": [
        {
            "doc_id": 123,
            "score": 0.8512,
            "title": "協同過濾推薦文章...",
            "category": "體育",
            "reason": "Users similar to you liked this"
        }
    ],
    "n_neighbors_found": 15,
    "parameters": {
        "top_k": 10,
        "n_neighbors": 20,
        "similarity_metric": "cosine"
    },
    "computation_time": 0.023
}
```

**演算法說明**:
- 計算用戶相似度矩陣 (User similarity matrix)
- 找出 k 個最相似用戶 (k-nearest neighbors)
- 聚合相似用戶喜歡的項目 (Aggregate items from similar users)
- **複雜度**: O(U²) 用戶相似度計算, O(k×I) 推薦生成

**Python 範例**:
```python
# User-based CF 推薦
response = requests.post('http://localhost:5001/api/recommend/cf/user-based', json={
    "user_id": 0,
    "top_k": 10,
    "n_neighbors": 20,
    "similarity_metric": "cosine"
})

recs = response.json()['recommendations']
print(f"Found {len(recs)} recommendations based on similar users")
for rec in recs[:5]:
    print(f"  - {rec['title']} (score: {rec['score']:.4f})")
```

---

### 5. 協同過濾 - 基於項目 (Collaborative Filtering - Item-Based)

**Endpoint**: `POST /api/recommend/cf/item-based`

基於項目相似性的協同過濾推薦 (Item-based collaborative filtering)。

**Request**:
```json
{
    "user_id": 0,
    "top_k": 10,
    "n_neighbors": 50,
    "similarity_metric": "cosine"
}
```

**參數說明**:
- `user_id`: 用戶 ID (required)
- `top_k`: 返回結果數量 (default: 10)
- `n_neighbors`: 每個項目考慮的相似項目數 (default: 50)
- `similarity_metric`: 相似度度量 - `cosine`, `adjusted_cosine`, `jaccard` (default: cosine)

**Response**:
```json
{
    "user_id": 0,
    "method": "item_based_cf",
    "recommendations": [
        {
            "doc_id": 456,
            "score": 0.9201,
            "title": "相似項目推薦...",
            "category": "娛樂",
            "reason": "Similar to items you liked"
        }
    ],
    "parameters": {
        "top_k": 10,
        "n_neighbors": 50,
        "similarity_metric": "cosine"
    },
    "computation_time": 0.018
}
```

**演算法說明**:
- 計算項目相似度矩陣 (Item similarity matrix)
- 對於用戶已互動的項目,找出相似項目 (Find similar items)
- 聚合並排序候選項目 (Aggregate and rank candidates)
- **優勢**: 項目相似度可預計算,查詢效率高
- **複雜度**: O(I²) 項目相似度計算, O(u×k) 推薦生成

**Python 範例**:
```python
# Item-based CF (通常比 User-based 更穩定)
response = requests.post('http://localhost:5001/api/recommend/cf/item-based', json={
    "user_id": 0,
    "top_k": 10,
    "n_neighbors": 50,
    "similarity_metric": "adjusted_cosine"
})

recs = response.json()['recommendations']
for rec in recs:
    print(f"{rec['doc_id']}: {rec['title']} ({rec['score']:.4f})")
```

---

### 6. 矩陣分解推薦 (Matrix Factorization)

**Endpoint**: `POST /api/recommend/cf/matrix-factorization`

基於矩陣分解的協同過濾 (Matrix factorization: SVD or ALS)。

**Request**:
```json
{
    "user_id": 0,
    "top_k": 10,
    "n_factors": 50,
    "method": "svd"
}
```

**參數說明**:
- `user_id`: 用戶 ID (required)
- `top_k`: 返回結果數量 (default: 10)
- `n_factors`: 潛在因子維度 (default: 50)
- `method`: 方法選擇 - `svd` (奇異值分解) 或 `als` (交替最小二乘) (default: svd)

**Response**:
```json
{
    "user_id": 0,
    "method": "matrix_factorization_svd",
    "recommendations": [
        {
            "doc_id": 789,
            "score": 0.8834,
            "title": "潛在因子推薦...",
            "category": "政治",
            "reason": "Predicted based on latent factors"
        }
    ],
    "parameters": {
        "top_k": 10,
        "n_factors": 50,
        "method": "svd"
    },
    "computation_time": 0.156
}
```

**演算法說明**:

**SVD (Singular Value Decomposition)**:
- R ≈ U × Σ × V^T
- U: 用戶潛在因子矩陣 (User latent factors)
- V: 項目潛在因子矩陣 (Item latent factors)
- Σ: 奇異值對角矩陣 (Singular values)
- **優點**: 數學嚴謹,快速計算
- **複雜度**: O(min(U,I)²×max(U,I))

**ALS (Alternating Least Squares)**:
- 交替優化用戶和項目因子 (Alternately optimize user and item factors)
- 目標函數: min ||R - U×V^T||² + λ(||U||² + ||V||²)
- **優點**: 處理隱式反饋,可並行化
- **複雜度**: O(n_iter × n_factors × n_ratings)

**Python 範例**:
```python
# SVD 矩陣分解
response_svd = requests.post('http://localhost:5001/api/recommend/cf/matrix-factorization', json={
    "user_id": 0,
    "top_k": 10,
    "n_factors": 50,
    "method": "svd"
})

# ALS 矩陣分解
response_als = requests.post('http://localhost:5001/api/recommend/cf/matrix-factorization', json={
    "user_id": 0,
    "top_k": 10,
    "n_factors": 50,
    "method": "als"
})

print("SVD Recommendations:")
for rec in response_svd.json()['recommendations'][:5]:
    print(f"  {rec['title']} ({rec['score']:.4f})")

print("\nALS Recommendations:")
for rec in response_als.json()['recommendations'][:5]:
    print(f"  {rec['title']} ({rec['score']:.4f})")
```

---

### 7. 混合推薦系統 (Hybrid Recommender)

**Endpoint**: `POST /api/recommend/hybrid`

結合內容和協同過濾的混合推薦 (Hybrid recommendations combining content-based and CF)。

**Request**:
```json
{
    "user_id": 0,
    "doc_id": 5,
    "top_k": 10,
    "fusion_method": "weighted",
    "content_weight": 0.5,
    "cf_weight": 0.4,
    "popularity_weight": 0.1,
    "use_embeddings": false
}
```

**參數說明**:
- `user_id`: 用戶 ID (required)
- `doc_id`: 當前文檔 ID (optional, 提供上下文)
- `top_k`: 返回結果數量 (default: 10)
- `fusion_method`: 融合方法 - `weighted`, `cascade`, `switching` (default: weighted)
- `content_weight`: 內容權重 (weighted 方法, default: 0.5)
- `cf_weight`: CF 權重 (weighted 方法, default: 0.4)
- `popularity_weight`: 熱度權重 (weighted 方法, default: 0.1)
- `use_embeddings`: 使用 BERT embeddings (default: false)

**Response**:
```json
{
    "user_id": 0,
    "method": "hybrid_weighted",
    "recommendations": [
        {
            "doc_id": 123,
            "score": 0.8734,
            "title": "混合推薦文章...",
            "category": "科技",
            "content_score": 0.8523,
            "cf_score": 0.9201,
            "popularity_score": 0.7845,
            "reason": "Combined content similarity and collaborative filtering"
        }
    ],
    "fusion_method": "weighted",
    "parameters": {
        "top_k": 10,
        "content_weight": 0.5,
        "cf_weight": 0.4,
        "popularity_weight": 0.1,
        "use_embeddings": false
    },
    "computation_time": 0.045
}
```

**融合方法說明**:

**1. Weighted Fusion (加權融合)**:
```
final_score = w_c × content_score + w_cf × cf_score + w_p × popularity_score
```
- 線性組合各個分數 (Linear combination of scores)
- 權重可調整以平衡不同策略
- **優點**: 簡單直觀,可解釋性強

**2. Cascade Fusion (級聯融合)**:
- Stage 1: 內容推薦生成候選集 (高召回率)
- Stage 2: CF 重排序候選集 (高精確度)
- Stage 3: 熱度作為 tiebreaker
- **優點**: 充分利用各方法優勢

**3. Switching Strategy (切換策略)**:
- 新用戶 (<5 互動): 使用內容推薦 (解決冷啟動)
- 活躍用戶 (≥5 互動): 使用協同過濾
- **優點**: 動態適應用戶狀態

**Python 範例**:
```python
# Weighted 混合推薦
response = requests.post('http://localhost:5001/api/recommend/hybrid', json={
    "user_id": 0,
    "doc_id": 5,
    "top_k": 10,
    "fusion_method": "weighted",
    "content_weight": 0.5,
    "cf_weight": 0.4,
    "popularity_weight": 0.1
})

recs = response.json()['recommendations']
for rec in recs[:5]:
    print(f"{rec['title']}")
    print(f"  Overall: {rec['score']:.4f}")
    print(f"  Content: {rec['content_score']:.4f}, CF: {rec['cf_score']:.4f}, Pop: {rec['popularity_score']:.4f}")

# Cascade 混合推薦
response = requests.post('http://localhost:5001/api/recommend/hybrid', json={
    "user_id": 0,
    "fusion_method": "cascade",
    "top_k": 10
})

# Switching 策略 (自動選擇方法)
response = requests.post('http://localhost:5001/api/recommend/hybrid', json={
    "user_id": 0,
    "fusion_method": "switching",
    "top_k": 10
})
```

---

### 8. 記錄用戶互動 (Record User Interaction)

**Endpoint**: `POST /api/interaction/record`

記錄用戶與文檔的互動行為 (Record user interaction with documents)。

**Request**:
```json
{
    "user_id": 0,
    "doc_id": 123,
    "interaction_type": "read",
    "duration": 45.5,
    "timestamp": "2025-11-14T10:30:00"
}
```

**參數說明**:
- `user_id`: 用戶 ID (required)
- `doc_id`: 文檔 ID (required)
- `interaction_type`: 互動類型 - `click`, `read`, `like`, `share` (default: click)
- `duration`: 持續時間(秒) (optional)
- `timestamp`: 時間戳 ISO 格式 (optional, 默認當前時間)

**Response**:
```json
{
    "status": "success",
    "interaction_id": 42,
    "message": "Interaction recorded"
}
```

**Python 範例**:
```python
# 記錄用戶點擊
requests.post('http://localhost:5001/api/interaction/record', json={
    "user_id": 0,
    "doc_id": 123,
    "interaction_type": "click"
})

# 記錄用戶閱讀(帶停留時間)
requests.post('http://localhost:5001/api/interaction/record', json={
    "user_id": 0,
    "doc_id": 123,
    "interaction_type": "read",
    "duration": 45.5
})

# 記錄用戶點贊
requests.post('http://localhost:5001/api/interaction/record', json={
    "user_id": 0,
    "doc_id": 123,
    "interaction_type": "like"
})
```

---

### 9. 獲取用戶互動歷史 (Get User Interaction History)

**Endpoint**: `GET /api/interaction/history`

獲取用戶的互動歷史記錄 (Retrieve user interaction history)。

**Query Parameters**:
- `user_id`: 用戶 ID (required)
- `limit`: 返回記錄數 (default: 50)

**Request**:
```bash
GET /api/interaction/history?user_id=0&limit=50
```

**Response**:
```json
{
    "user_id": 0,
    "interactions": [
        {
            "interaction_id": 42,
            "doc_id": 123,
            "interaction_type": "read",
            "duration": 45.5,
            "timestamp": "2025-11-14T10:30:00"
        },
        {
            "interaction_id": 41,
            "doc_id": 120,
            "interaction_type": "click",
            "duration": 0,
            "timestamp": "2025-11-14T10:25:00"
        }
    ],
    "total": 142,
    "returned": 50
}
```

**Python 範例**:
```python
# 獲取用戶互動歷史
response = requests.get('http://localhost:5001/api/interaction/history', params={
    "user_id": 0,
    "limit": 50
})

history = response.json()
print(f"User {history['user_id']} has {history['total']} interactions")

for interaction in history['interactions'][:10]:
    print(f"{interaction['timestamp']}: {interaction['interaction_type']} on doc {interaction['doc_id']}")

# 提取閱讀歷史用於個性化推薦
reading_history = [i['doc_id'] for i in history['interactions'] if i['interaction_type'] == 'read']
print(f"Read {len(reading_history)} documents")
```

---

## 🎯 推薦系統性能指標 (Recommendation System Performance)

| API 端點 | 平均響應時間 | 複雜度 | 說明 |
|---------|-------------|--------|------|
| `/api/recommend/similar` | ~20-30ms | O(D) | D=文檔數,使用預計算向量 |
| `/api/recommend/personalized` | ~15-25ms | O(h×D) | h=歷史長度 |
| `/api/recommend/trending` | ~5-10ms | O(D log D) | 簡單排序 |
| `/api/recommend/cf/user-based` | ~20-50ms | O(U×k) | U=用戶數,k=鄰居數 |
| `/api/recommend/cf/item-based` | ~15-30ms | O(u×k) | u=用戶歷史,k=鄰居數 |
| `/api/recommend/cf/matrix-factorization` | ~100-200ms | O(n_factors×I) | 包含訓練時間 |
| `/api/recommend/hybrid` | ~40-80ms | 取決於融合方法 | 組合多個推薦器 |
| `/api/interaction/record` | <5ms | O(1) | 簡單插入操作 |
| `/api/interaction/history` | <10ms | O(n log n) | n=互動數 |

**優化建議**:
1. **預計算**: TF-IDF 向量、BERT embeddings、項目相似度矩陣
2. **快取**: 熱門文檔推薦、用戶相似度
3. **索引**: 用戶互動記錄建立索引
4. **批處理**: MF 訓練可離線執行
5. **採樣**: 大規模數據集使用採樣加速

---

## 📦 完整 API 清單 (Complete API List)

### 檢索 APIs (Retrieval)
1. `POST /api/search/boolean` - 布林檢索
2. `POST /api/search/vsm` - 向量空間模型
3. `POST /api/search/bm25` - BM25 排序
4. `POST /api/search/lm` - 語言模型檢索
5. `POST /api/search/hybrid` - 混合排序

### 文本分析 APIs (Text Analysis)
6. `POST /api/extract/keywords` - 關鍵字提取 ⭐ NEW
7. `POST /api/extract/topics` - 主題建模 ⭐ NEW
8. `POST /api/extract/patterns` - 模式挖掘 ⭐ NEW

### 語言處理 APIs (NLP)
9. `POST /api/analyze/ner` - 命名實體識別 ⭐ NEW
10. `POST /api/analyze/syntax` - 句法分析 ⭐ NEW
11. `POST /api/analyze/collocation` - 詞彙共現分析
12. `POST /api/analyze/ngram` - N-gram 分析

### 文檔 APIs (Document)
13. `GET /api/document/<id>` - 獲取文檔
14. `GET /api/document/<id>/analysis` - 文檔綜合分析 ⭐ NEW
15. `POST /api/summarize/<id>` - 文檔摘要
16. `POST /api/expand_query` - 查詢擴展
17. `POST /api/cluster` - 文檔聚類

### 推薦系統 APIs (Recommendation) ⭐ NEW
18. `POST /api/recommend/similar` - 內容推薦 (相似文檔)
19. `POST /api/recommend/personalized` - 個人化推薦
20. `GET /api/recommend/trending` - 熱門推薦
21. `POST /api/recommend/cf/user-based` - 協同過濾 (基於用戶)
22. `POST /api/recommend/cf/item-based` - 協同過濾 (基於項目)
23. `POST /api/recommend/cf/matrix-factorization` - 矩陣分解 (SVD/ALS)
24. `POST /api/recommend/hybrid` - 混合推薦系統
25. `POST /api/interaction/record` - 記錄用戶互動
26. `GET /api/interaction/history` - 用戶互動歷史

### 系統 APIs (System)
27. `GET /api/stats` - 系統統計

**總計**: 27 個 API 端點
**v2.0 新增**: 5 個進階 NLP API
**v3.0 新增**: 9 個推薦系統 API (Content-Based, Collaborative Filtering, Hybrid)

---

**作者**: Information Retrieval System
**版本**: v3.0 (Recommendation System Edition)
**日期**: 2025-11-14
**License**: Educational Use
