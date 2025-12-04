# Faceted Search 整合指南

## 概述

已完成實作 Faceted Search (分面搜尋) 功能模組，支援多維度篩選和動態分面計數。

**實作位置**: `src/ir/facet/`

**測試狀態**: ✅ 31/31 測試通過 (68-80% 覆蓋率)

---

## 核心模組

### 1. `facet_engine.py` - 分面計算引擎

**主要類別**:
- `FacetEngine`: 分面計算引擎
- `FacetResult`: 分面結果容器
- `FacetValue`: 單一分面值 (值 + 計數)

**支援的分面類型**:
1. **Term Facets** (離散值分面)
   - 新聞來源、分類、作者等
   - 自動排序 (按計數降序)

2. **Date Range Facets** (日期範圍分面)
   - 按月份/年份分組
   - 支援自定義日期格式

3. **Numeric Range Facets** (數值範圍分面)
   - 分數區間、數值範圍
   - 可自定義 bucket

### 2. `facet_filter.py` - 篩選條件管理

**主要類別**:
- `FacetFilter`: 篩選管理器 (支援 AND/OR 邏輯)
- `FilterCondition`: 篩選條件
- `RangeFilter`: 範圍篩選
- `FilterOperator`: 8 種運算子 (EQUALS, IN, RANGE, GT, LT, etc.)

**支援的篩選邏輯**:
- **AND 邏輯**: 跨欄位條件必須全部符合
- **OR 邏輯**: 同欄位內多選 (checkbox 風格)

---

## 快速開始

### 基本使用範例

```python
from src.ir.facet import (
    FacetEngine,
    FacetFilter,
    FilterCondition,
    FilterOperator,
    RangeFilter
)

# 1. 初始化引擎並配置分面
engine = FacetEngine()
engine.configure_facet("source", "新聞來源", "term")
engine.configure_facet("category", "分類", "term")
engine.configure_facet("pub_date", "發布日期", "date_range", date_format="%Y-%m")

# 2. 從搜尋結果建立分面
search_results = [
    {"id": "1", "source": "CNA", "category": "politics", "pub_date": "2024-11-15"},
    {"id": "2", "source": "UDN", "category": "finance", "pub_date": "2024-11-16"},
    # ...
]

facets = engine.build_facets(search_results)

# 3. 查看分面結果
print(f"來源分面:")
for fv in facets["source"].values[:5]:
    print(f"  {fv.label}: {fv.count} 篇")

# 4. 建立篩選器
filter_mgr = FacetFilter()

# 多選來源 (CNA OR UDN)
filter_mgr.add_condition(
    FilterCondition("source", FilterOperator.IN, ["CNA", "UDN"])
)

# 單選分類
filter_mgr.add_condition(
    FilterCondition("category", FilterOperator.EQUALS, "politics")
)

# 日期範圍
filter_mgr.add_condition(
    RangeFilter("pub_date", "2024-11-01", "2024-11-30")
)

# 5. 應用篩選
filtered_docs = filter_mgr.filter(search_results)
print(f"篩選後: {len(filtered_docs)} 篇文章")
```

---

## Flask API 整合

### API 端點設計

#### 1. `POST /api/facets` - 取得分面資訊

**請求格式**:
```json
{
  "query": "台灣 經濟",
  "model": "bm25",
  "top_k": 100,
  "facet_fields": ["source", "category", "pub_date"]
}
```

**回應格式**:
```json
{
  "success": true,
  "total_results": 156,
  "facets": {
    "source": {
      "field_name": "source",
      "display_name": "新聞來源",
      "facet_type": "term",
      "total_docs": 156,
      "values": [
        {"value": "CNA", "count": 45, "label": "中央社"},
        {"value": "UDN", "count": 38, "label": "聯合新聞網"},
        {"value": "LTN", "count": 32, "label": "自由時報"}
      ]
    },
    "category": {
      "field_name": "category",
      "display_name": "分類",
      "facet_type": "term",
      "values": [
        {"value": "finance", "count": 89, "label": "財經"},
        {"value": "politics", "count": 45, "label": "政治"},
        {"value": "international", "count": 22, "label": "國際"}
      ]
    },
    "pub_date": {
      "field_name": "pub_date",
      "display_name": "發布月份",
      "facet_type": "date_range",
      "values": [
        {"value": "2024-11", "count": 120, "label": "2024-11"},
        {"value": "2024-10", "count": 36, "label": "2024-10"}
      ]
    }
  }
}
```

#### 2. `POST /api/search/faceted` - 帶篩選的搜尋

**請求格式**:
```json
{
  "query": "台灣 經濟",
  "model": "bm25",
  "top_k": 20,
  "filters": {
    "source": ["CNA", "UDN"],
    "category": "finance",
    "pub_date": ["2024-11-01", "2024-11-30"]
  }
}
```

**回應格式**:
```json
{
  "success": true,
  "query": "台灣 經濟",
  "total_results": 45,
  "filtered_results": 23,
  "results": [
    {
      "doc_id": "...",
      "title": "...",
      "score": 0.85,
      "source": "CNA",
      "category": "finance",
      "pub_date": "2024-11-15"
    }
  ],
  "active_filters": {
    "count": 3,
    "conditions": [...]
  }
}
```

### Flask 實作範例

在 `app_simple.py` 中添加以下端點：

```python
# Global facet engine (lazy loading)
_facet_engine_cache = None

def get_facet_engine():
    """Get or initialize facet engine."""
    global _facet_engine_cache

    if _facet_engine_cache is None:
        engine = FacetEngine()

        # Configure news-specific facets
        engine.configure_facet("source", "新聞來源", "term", max_values=20)
        engine.configure_facet("category", "分類", "term")
        engine.configure_facet("pub_date", "發布月份", "date_range", date_format="%Y-%m")
        engine.configure_facet("author", "作者", "term", max_values=15)

        _facet_engine_cache = engine
        logger.info("Facet engine initialized")

    return _facet_engine_cache


@app.route('/api/facets', methods=['POST'])
def api_facets():
    """
    Get facet information from search results.

    POST body:
        {
            "query": str,
            "model": str (default: "bm25"),
            "top_k": int (default: 100),
            "facet_fields": list (optional, all if not specified)
        }
    """
    try:
        data = request.get_json()

        if not data or 'query' not in data:
            return jsonify({'success': False, 'error': 'Missing query'}), 400

        query = data.get('query', '').strip()
        model = data.get('model', 'bm25')
        top_k = min(data.get('top_k', 100), app.config['MAX_RESULTS'])
        facet_fields = data.get('facet_fields', None)

        # Perform search
        ret = get_retriever()
        results = ret.search(query, model=model, top_k=top_k)

        # Convert SearchResult to dict format for facet engine
        documents = [
            {
                'id': r.doc_id,
                'title': r.title,
                'source': r.metadata.get('source', ''),
                'category': r.metadata.get('category', ''),
                'pub_date': r.metadata.get('published_date', ''),
                'author': r.metadata.get('author', '')
            }
            for r in results
        ]

        # Build facets
        engine = get_facet_engine()
        facets = engine.build_facets(documents, field_name=facet_fields)

        # Format response
        facets_data = {}
        for field_name, facet_result in facets.items():
            facets_data[field_name] = {
                'field_name': facet_result.field_name,
                'display_name': facet_result.display_name,
                'facet_type': facet_result.facet_type,
                'total_docs': facet_result.total_docs,
                'values': [
                    {
                        'value': fv.value,
                        'count': fv.count,
                        'label': fv.label
                    }
                    for fv in facet_result.values
                ]
            }

        return jsonify({
            'success': True,
            'total_results': len(documents),
            'facets': facets_data
        })

    except Exception as e:
        logger.error(f"Facets error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/search/faceted', methods=['POST'])
def api_faceted_search():
    """
    Search with faceted filtering.

    POST body:
        {
            "query": str,
            "model": str,
            "top_k": int,
            "filters": {
                "source": ["CNA", "UDN"],  // Multi-select
                "category": "finance",      // Single select
                "pub_date": ["2024-11-01", "2024-11-30"]  // Range
            }
        }
    """
    try:
        data = request.get_json()

        if not data or 'query' not in data:
            return jsonify({'success': False, 'error': 'Missing query'}), 400

        query = data.get('query', '').strip()
        model = data.get('model', 'bm25')
        top_k = min(data.get('top_k', 20), app.config['MAX_RESULTS'])
        filters = data.get('filters', {})

        # Perform search
        ret = get_retriever()
        results = ret.search(query, model=model, top_k=top_k*5)  # Get more for filtering

        # Convert to document format
        documents = [
            {
                'id': r.doc_id,
                'title': r.title,
                'snippet': r.snippet,
                'score': r.score,
                'rank': r.rank,
                'source': r.metadata.get('source', ''),
                'category': r.metadata.get('category', ''),
                'pub_date': r.metadata.get('published_date', ''),
                'author': r.metadata.get('author', ''),
                'url': r.metadata.get('url', ''),
                'metadata': r.metadata
            }
            for r in results
        ]

        # Apply filters if provided
        if filters:
            filter_mgr = FacetFilter()

            for field, value in filters.items():
                if isinstance(value, list):
                    if len(value) == 2 and field in ['pub_date', 'score']:
                        # Range filter
                        filter_mgr.add_condition(
                            RangeFilter(field, value[0], value[1])
                        )
                    else:
                        # Multi-select filter
                        filter_mgr.add_condition(
                            FilterCondition(field, FilterOperator.IN, value)
                        )
                else:
                    # Single value filter
                    filter_mgr.add_condition(
                        FilterCondition(field, FilterOperator.EQUALS, value)
                    )

            documents = filter_mgr.filter(documents)

        # Limit to top_k after filtering
        documents = documents[:top_k]

        return jsonify({
            'success': True,
            'query': query,
            'total_results': len(results),
            'filtered_results': len(documents),
            'results': documents,
            'active_filters': {
                'count': len(filters),
                'filters': filters
            }
        })

    except Exception as e:
        logger.error(f"Faceted search error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500
```

---

## API 測試

執行測試腳本驗證 API 端點：

```bash
# 1. 啟動 Flask 應用
python app_simple.py --port 5000

# 2. 在另一個終端執行測試
python test_facet_api.py
```

### 測試腳本說明

`test_facet_api.py` 會測試:
1. **`/api/facets`**: 取得分面資訊
2. **`/api/search/faceted`**: 執行帶篩選的搜尋

測試輸出範例:
```
===========================================================
Testing /api/facets endpoint
===========================================================

✅ /api/facets endpoint working!
Total results: 50

Facets available: ['source', 'category', 'category_name', 'pub_date', 'author']

新聞來源 facet:
  中央社: 25 篇
  聯合新聞網: 15 篇
  ...

===========================================================
Testing /api/search/faceted endpoint
===========================================================

--- Test 1: Filter by source (CNA) ---
✅ Filter test passed!
Total results before filter: 100
Filtered results: 25
Active filters: 1
...
```

### 手動測試 (使用 curl)

**測試 `/api/facets`**:
```bash
curl -X POST http://localhost:5000/api/facets \
  -H "Content-Type: application/json" \
  -d '{"query": "台灣", "model": "bm25", "top_k": 50}'
```

**測試 `/api/search/faceted`**:
```bash
curl -X POST http://localhost:5000/api/search/faceted \
  -H "Content-Type: application/json" \
  -d '{
    "query": "經濟",
    "model": "bm25",
    "top_k": 10,
    "filters": {
      "source": ["中央社"],
      "category": "afe"
    }
  }'
```

---

## 前端整合

### HTML/JavaScript 範例

```html
<!-- Left Sidebar: Facet Filters -->
<div class="facet-sidebar">
    <h3>篩選條件</h3>

    <!-- Source Facet -->
    <div class="facet-group">
        <h4>新聞來源</h4>
        <div id="source-facets" class="facet-checkboxes">
            <!-- Dynamically populated -->
        </div>
    </div>

    <!-- Category Facet -->
    <div class="facet-group">
        <h4>分類</h4>
        <div id="category-facets" class="facet-checkboxes">
            <!-- Dynamically populated -->
        </div>
    </div>

    <!-- Date Range Facet -->
    <div class="facet-group">
        <h4>發布月份</h4>
        <div id="date-facets" class="facet-checkboxes">
            <!-- Dynamically populated -->
        </div>
    </div>

    <button onclick="clearFilters()">清除所有篩選</button>
</div>

<script>
// Fetch and display facets
async function loadFacets(query) {
    const response = await fetch('/api/facets', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            query: query,
            top_k: 100
        })
    });

    const data = await response.json();

    // Render source facets
    const sourceDiv = document.getElementById('source-facets');
    sourceDiv.innerHTML = data.facets.source.values.map(fv => `
        <label>
            <input type="checkbox" name="source" value="${fv.value}">
            ${fv.label} (${fv.count})
        </label>
    `).join('');

    // Similar for other facets...
}

// Perform filtered search
async function searchWithFilters() {
    const query = document.getElementById('search-query').value;
    const filters = {
        source: Array.from(document.querySelectorAll('input[name="source"]:checked'))
                     .map(cb => cb.value),
        category: document.querySelector('input[name="category"]:checked')?.value,
        // ...
    };

    const response = await fetch('/api/search/faceted', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            query: query,
            filters: filters,
            top_k: 20
        })
    });

    const data = await response.json();
    displayResults(data.results);
}
</script>
```

---

## 效能考量

### 複雜度分析

**FacetEngine**:
- `build_facets()`: O(N × F) - N 文檔, F 分面欄位
- `filter_documents()`: O(N × C) - N 文檔, C 條件數
- 空間: O(U) - U 為唯一分面值數量

**最佳實踐**:
1. 限制 top_k 文檔數量 (建議 100-500)
2. 使用 `max_values` 限制分面值數量
3. 快取分面結果 (相同查詢)
4. 按需載入分面 (只計算可見分面)

---

## 測試

執行測試:
```bash
pytest tests/test_facet.py -v
```

測試涵蓋:
- ✅ Term facets 建立
- ✅ Date range facets 分組
- ✅ 篩選條件 (單選/多選/範圍)
- ✅ Filter 組合邏輯
- ✅ 邊界條件處理

---

## 未來擴展

可能的增強功能:
1. **Hierarchical Facets** (階層式分面)
   - 支援樹狀結構 (例如：地區 > 城市)

2. **Search-within-Facets** (分面內搜尋)
   - 當分面值過多時支援搜尋

3. **Dynamic Range Buckets** (動態範圍分組)
   - 根據數據分布自動決定 bucket

4. **Facet Caching** (分面快取)
   - Redis 快取熱門查詢的分面結果

5. **Facet-based Query Suggestions** (基於分面的查詢建議)
   - 根據分面組合推薦相關查詢

---

**作者**: Claude Code
**日期**: 2025-11-21
**版本**: 1.0
