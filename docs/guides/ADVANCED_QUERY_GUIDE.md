# 進階查詢系統使用指南
# Advanced Query System Guide

## 📚 概述 (Overview)

本指南說明如何使用進階查詢系統進行複雜的多欄位查詢，類似圖書館資料庫的查詢功能。

**核心功能**:
- 欄位特定查詢 (Field-specific queries)
- 布林運算子 (AND, OR, NOT)
- 範圍查詢 (Range queries for dates)
- 括號分組 (Parentheses grouping)
- 結構化查詢 API (Structured query API)

---

## 🔧 核心組件 (Core Components)

### 1. Query Parser

解析類似 Google/圖書館的查詢語法。

**支援的語法**:
```
field:value                           # 欄位查詢
field:value1 AND field2:value2       # AND 組合
field:value1 OR field2:value2        # OR 組合
NOT field:value                       # 否定
(field1:val1 OR field1:val2) AND field2:val3  # 括號分組
field:[start TO end]                  # 範圍查詢
```

### 2. Query Executor

執行解析後的查詢樹，使用 FieldIndexer 進行實際搜尋。

### 3. Field Indexer

為每個 metadata 欄位建立獨立的倒排索引。

**支援的欄位**:
- `title` - 標題 (tokenized)
- `content` - 內容 (tokenized)
- `category` - 分類代碼 (exact match)
- `category_name` - 分類名稱 (tokenized)
- `tags` - 標籤 (multi-value)
- `published_date` - 發布日期 (range queries)
- `author` - 作者 (tokenized)
- `source` - 來源 (exact match)

---

## 💻 程式化使用 (Programmatic Usage)

### 基礎範例

```python
from src.ir.query import parse_query, QueryExecutor
from src.ir.index.field_indexer import FieldIndexer

# 1. 建立 Field Index
from src.ir.text.chinese_tokenizer import ChineseTokenizer

tokenizer = ChineseTokenizer(engine='jieba')
field_indexer = FieldIndexer(tokenizer=tokenizer.tokenize)

# 載入文檔並建立索引
documents = [...]  # 你的文檔列表
field_indexer.build(documents)

# 2. 解析查詢
query_node = parse_query("title:台灣 AND category:aipl")

# 3. 執行查詢
executor = QueryExecutor(field_indexer, documents)
results = executor.execute(query_node, top_k=20)

# 4. 處理結果
for result in results:
    print(f"Doc ID: {result.doc_id}")
    print(f"Score: {result.score}")
    print(f"Matched fields: {result.matched_fields}")
```

### 進階範例

#### 範例 1: 複雜布林查詢

```python
# 查詢：標題包含"台灣"或"中國"，且分類為政治，但不包含體育
query = "(title:台灣 OR title:中國) AND category:aipl AND NOT category:sports"

node = parse_query(query)
results = executor.execute(node)

print(f"找到 {len(results)} 篇文章")
```

#### 範例 2: 日期範圍查詢

```python
# 查詢：2025年11月1日到11月13日之間的新聞
query = "date:[2025-11-01 TO 2025-11-13]"

node = parse_query(query)
results = executor.execute(node)
```

#### 範例 3: 多欄位組合查詢

```python
# 查詢：標題含"經濟"，作者為"記者"，且2025年11月發布
query = "title:經濟 AND author:記者 AND date:[2025-11-01 TO 2025-11-30]"

node = parse_query(query)
results = executor.execute(node)
```

#### 範例 4: 結構化查詢 API

```python
# 使用 JSON 格式的結構化查詢
conditions = [
    {"field": "title", "operator": "contains", "value": "台灣"},
    {"field": "category", "operator": "equals", "value": "aipl"},
    {"field": "published_date", "operator": "between", "value": ["2025-11-01", "2025-11-13"]}
]

results = executor.execute_structured_query(
    conditions=conditions,
    logic="AND",
    top_k=20
)
```

---

## 🌐 API 整合範例 (API Integration Example)

### Flask API Endpoint

以下是如何在 Flask app 中整合進階查詢系統：

```python
# app_simple.py

from flask import Flask, request, jsonify
from src.ir.query import parse_query, QueryExecutor
from src.ir.index.field_indexer import FieldIndexer
import pickle

app = Flask(__name__)

# 載入 Field Index (啟動時載入一次)
with open('data/indexes/field_index.pkl', 'rb') as f:
    field_indexer = pickle.load(f)

# 載入文檔
import json
documents = []
with open('data/preprocessed/cna_mvp_preprocessed.jsonl', 'r') as f:
    for line in f:
        if line.strip():
            documents.append(json.loads(line))

# 建立 QueryExecutor
query_executor = QueryExecutor(field_indexer, documents)


@app.route('/api/advanced_search', methods=['POST'])
def api_advanced_search():
    """
    進階查詢 API

    Request JSON:
        {
            "query": "title:台灣 AND category:政治",  // 查詢字串 (Option 1)
            // OR
            "conditions": [                           // 結構化查詢 (Option 2)
                {"field": "title", "operator": "contains", "value": "台灣"},
                {"field": "category", "operator": "equals", "value": "aipl"}
            ],
            "logic": "AND",                          // 條件組合邏輯
            "top_k": 20,                             // 最多回傳幾筆
            "sort_by": "date",                       // 排序欄位
            "sort_order": "desc"                     // 排序順序
        }

    Response JSON:
        {
            "success": true,
            "query": "...",
            "total": 15,
            "results": [
                {
                    "doc_id": 0,
                    "article_id": "202511120135",
                    "title": "...",
                    "content": "...",
                    "category": "aipl",
                    "published_date": "2025-11-12",
                    "score": 1.0,
                    "matched_fields": ["title", "category"]
                },
                ...
            ],
            "processing_time": 0.015
        }
    """
    import time
    start_time = time.time()

    try:
        data = request.get_json()

        # Option 1: Query string
        if 'query' in data:
            query_str = data['query']
            query_node = parse_query(query_str)
            results = query_executor.execute(
                query_node,
                top_k=data.get('top_k', None)
            )

        # Option 2: Structured query
        elif 'conditions' in data:
            results = query_executor.execute_structured_query(
                conditions=data['conditions'],
                logic=data.get('logic', 'AND'),
                top_k=data.get('top_k', None)
            )

        else:
            return jsonify({
                'success': False,
                'error': 'Must provide either "query" or "conditions"'
            }), 400

        # 轉換結果為 JSON 格式
        result_list = []
        for result in results:
            doc = documents[result.doc_id]
            result_list.append({
                'doc_id': result.doc_id,
                'article_id': doc.get('article_id'),
                'title': doc.get('title'),
                'content': doc.get('content', '')[:200] + '...',  # 摘要
                'category': doc.get('category'),
                'category_name': doc.get('category_name'),
                'published_date': doc.get('published_date'),
                'tags': doc.get('tags', []),
                'score': result.score,
                'matched_fields': result.matched_fields
            })

        # 排序 (如果需要)
        sort_by = data.get('sort_by')
        if sort_by:
            reverse = data.get('sort_order', 'desc') == 'desc'
            result_list.sort(key=lambda x: x.get(sort_by, ''), reverse=reverse)

        processing_time = time.time() - start_time

        return jsonify({
            'success': True,
            'query': data.get('query', ''),
            'total': len(result_list),
            'results': result_list,
            'processing_time': round(processing_time, 3)
        })

    except SyntaxError as e:
        return jsonify({
            'success': False,
            'error': f'Query syntax error: {str(e)}'
        }), 400

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/query_fields', methods=['GET'])
def api_query_fields():
    """
    返回可查詢的欄位列表

    Response JSON:
        {
            "fields": [
                {
                    "name": "title",
                    "display_name": "標題",
                    "type": "text",
                    "operators": ["contains", "starts_with", "equals"]
                },
                ...
            ]
        }
    """
    fields = [
        {
            "name": "title",
            "display_name": "標題",
            "type": "text",
            "operators": ["contains", "starts_with", "equals"]
        },
        {
            "name": "content",
            "display_name": "內容",
            "type": "text",
            "operators": ["contains", "starts_with", "equals"]
        },
        {
            "name": "category",
            "display_name": "分類代碼",
            "type": "exact",
            "operators": ["equals", "not_equals"]
        },
        {
            "name": "category_name",
            "display_name": "分類名稱",
            "type": "text",
            "operators": ["contains", "equals"]
        },
        {
            "name": "tags",
            "display_name": "標籤",
            "type": "multi",
            "operators": ["contains_any", "contains_all"]
        },
        {
            "name": "published_date",
            "display_name": "發布日期",
            "type": "date",
            "operators": ["equals", "before", "after", "between"]
        },
        {
            "name": "author",
            "display_name": "作者",
            "type": "text",
            "operators": ["contains", "equals"]
        },
        {
            "name": "source",
            "display_name": "來源",
            "type": "exact",
            "operators": ["equals"]
        }
    ]

    return jsonify({"fields": fields})
```

---

## 🎨 前端 UI 範例 (Frontend UI Example)

### Query Builder HTML

```html
<!-- templates/advanced_search.html -->

<div class="query-builder-container">
    <h3>進階查詢</h3>

    <div id="query-conditions">
        <!-- Condition rows will be added here -->
    </div>

    <div class="query-controls">
        <button onclick="addCondition()" class="btn btn-secondary">
            ➕ 新增條件
        </button>

        <select id="logic-operator">
            <option value="AND">AND (且)</option>
            <option value="OR">OR (或)</option>
        </select>

        <button onclick="executeQuery()" class="btn btn-primary">
            🔍 執行查詢
        </button>
    </div>

    <div id="query-results">
        <!-- Results will be displayed here -->
    </div>
</div>

<!-- Condition Row Template -->
<template id="condition-template">
    <div class="query-condition">
        <select class="field-select">
            <option value="title">標題</option>
            <option value="content">內容</option>
            <option value="category">分類</option>
            <option value="tags">標籤</option>
            <option value="published_date">發布日期</option>
            <option value="author">作者</option>
        </select>

        <select class="operator-select">
            <option value="contains">包含</option>
            <option value="equals">等於</option>
            <option value="starts_with">開頭為</option>
        </select>

        <input type="text" class="value-input" placeholder="查詢值">

        <button onclick="removeCondition(this)" class="btn btn-sm btn-danger">
            ❌
        </button>
    </div>
</template>
```

### JavaScript Implementation

```javascript
// static/js/advanced-query.js

let conditions = [];

function addCondition() {
    const template = document.getElementById('condition-template');
    const clone = template.content.cloneNode(true);

    const container = document.getElementById('query-conditions');
    container.appendChild(clone);

    updateConditions();
}

function removeCondition(button) {
    button.closest('.query-condition').remove();
    updateConditions();
}

function updateConditions() {
    conditions = [];
    const conditionElements = document.querySelectorAll('.query-condition');

    conditionElements.forEach(elem => {
        const field = elem.querySelector('.field-select').value;
        const operator = elem.querySelector('.operator-select').value;
        const value = elem.querySelector('.value-input').value;

        if (field && value) {
            conditions.push({ field, operator, value });
        }
    });
}

async function executeQuery() {
    updateConditions();

    if (conditions.length === 0) {
        alert('請至少新增一個查詢條件');
        return;
    }

    const logic = document.getElementById('logic-operator').value;

    try {
        const response = await fetch('/api/advanced_search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                conditions: conditions,
                logic: logic,
                top_k: 20,
                sort_by: 'date',
                sort_order: 'desc'
            })
        });

        const data = await response.json();

        if (data.success) {
            displayResults(data.results, data.total, data.processing_time);
        } else {
            alert(`查詢錯誤: ${data.error}`);
        }
    } catch (error) {
        console.error('Query error:', error);
        alert('查詢失敗，請稍後再試');
    }
}

function displayResults(results, total, time) {
    const container = document.getElementById('query-results');

    let html = `
        <div class="results-header">
            <p>找到 ${total} 筆結果 (${time}秒)</p>
        </div>
    `;

    results.forEach(result => {
        html += `
            <div class="result-item">
                <h4>${result.title}</h4>
                <p class="result-meta">
                    分類: ${result.category_name} |
                    日期: ${result.published_date} |
                    來源: ${result.source}
                </p>
                <p class="result-snippet">${result.content}</p>
                <p class="result-matched">
                    匹配欄位: ${result.matched_fields.join(', ')}
                </p>
            </div>
        `;
    });

    container.innerHTML = html;
}

// Initialize with one condition
window.addEventListener('DOMContentLoaded', () => {
    addCondition();
});
```

---

## 📝 查詢語法參考 (Query Syntax Reference)

### 基礎語法

| 語法 | 說明 | 範例 |
|------|------|------|
| `field:value` | 欄位包含值 | `title:台灣` |
| `field:value1 value2` | 欄位包含多個值 (OR) | `title:台灣 中國` |
| `field:"exact phrase"` | 精確片語 | `title:"台灣新聞"` |
| `field:value1 AND field2:value2` | AND 組合 | `title:台灣 AND category:政治` |
| `field:value1 OR field2:value2` | OR 組合 | `category:aipl OR category:afe` |
| `NOT field:value` | 否定 | `NOT category:sports` |
| `(...)` | 分組 | `(title:A OR title:B) AND category:C` |
| `field:[start TO end]` | 範圍 | `date:[2025-11-01 TO 2025-11-30]` |

### 支援的運算子

**文字欄位**: title, content, category_name, author
- `contains` - 包含 (tokenized search)
- `starts_with` - 開頭為
- `equals` - 精確等於

**精確欄位**: category, source, url
- `equals` - 等於
- `not_equals` - 不等於

**多值欄位**: tags
- `contains_any` - 包含任一
- `contains_all` - 包含全部

**日期欄位**: published_date
- `equals` - 等於
- `before` - 之前
- `after` - 之後
- `between` - 介於 (使用 `[start TO end]` 語法)

---

## 🚀 下一步擴展 (Next Steps for Extension)

### 1. Faceted Search (分面搜尋)

```python
@app.route('/api/facets', methods=['POST'])
def api_facets():
    """返回當前查詢結果的 facet 統計"""
    # 實作 facet 計數
    facets = {
        "category": {
            "aipl": 45,
            "afe": 23,
            "ait": 18
        },
        "date_histogram": {
            "2025-11-12": 15,
            "2025-11-11": 12
        }
    }
    return jsonify(facets)
```

### 2. Query Suggestions (查詢建議)

```python
@app.route('/api/suggest', methods=['GET'])
def api_suggest():
    """基於輸入提供查詢建議"""
    prefix = request.args.get('q', '')
    # 使用 PAT-tree 提供自動補全
    suggestions = pat_tree.starts_with(prefix)
    return jsonify({"suggestions": suggestions[:10]})
```

### 3. Saved Queries (儲存查詢)

```python
@app.route('/api/saved_queries', methods=['GET', 'POST'])
def api_saved_queries():
    """儲存和載入常用查詢"""
    # 實作查詢儲存功能
    pass
```

### 4. Query History (查詢歷史)

```javascript
// localStorage-based query history
function saveQueryHistory(query) {
    let history = JSON.parse(localStorage.getItem('queryHistory') || '[]');
    history.unshift({
        query: query,
        timestamp: new Date().toISOString()
    });
    history = history.slice(0, 10);  // Keep last 10
    localStorage.setItem('queryHistory', JSON.stringify(history));
}
```

---

## ✅ 總結 (Summary)

### 已完成功能

✓ **Query Parser** - 完整的查詢語法解析器
✓ **Query Executor** - 查詢執行引擎
✓ **Field Indexer** - 欄位索引（已存在）
✓ **Query Tree Representation** - 查詢樹資料結構
✓ **Structured Query API** - JSON 格式查詢介面

### 待實作功能

⏳ **API Endpoints** - Flask endpoints (本文檔提供範例)
⏳ **Query Builder UI** - 前端介面 (本文檔提供範例)
⏳ **Faceted Search** - 分面搜尋統計
⏳ **Query Suggestions** - 查詢自動補全
⏳ **Query History** - 查詢歷史記錄
⏳ **Result Filtering** - 結果篩選 UI

### 使用流程

1. **建立 Field Index**:
   ```bash
   python scripts/build_field_index.py
   ```

2. **整合到現有系統**:
   - 參考本文檔的 API 整合範例
   - 複製 Flask endpoint 程式碼到 `app_simple.py`
   - 載入 field_index.pkl

3. **建立前端 UI**:
   - 參考本文檔的 HTML/JavaScript 範例
   - 建立 Query Builder 組件
   - 整合到現有搜尋介面

4. **測試與調整**:
   - 測試各種查詢語法
   - 調整 UI/UX
   - 優化性能

---

**作者**: LLMProvider Tooling
**日期**: 2025-11-17
**版本**: 1.0
**狀態**: ✅ Ready for Integration
