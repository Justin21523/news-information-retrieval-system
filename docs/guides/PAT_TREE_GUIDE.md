# PAT-tree 實作指南
# PAT-tree Implementation Guide

## 📚 理論背景 (Theoretical Background)

### 什麼是 PAT-tree？

**PAT-tree** (Patricia Trie) 是一種空間優化的字典樹（Trie）資料結構，由 Donald R. Morrison 在 1968 年提出。

**核心概念**：
- **Trie**: 樹狀結構，用邊儲存字串的字元
- **Patricia**: **P**ractical **A**lgorithm **T**o **R**etrieve **I**nformation **C**oded **i**n **A**lphanumeric
- **Path Compression**: 合併只有單一子節點的路徑，減少節點數

### 與標準 Trie 的差異

```
Standard Trie:          Patricia Trie (Compressed):
    ROOT                     ROOT
    / | \                    / | \
   t  a  b                  t  a  "book"
   |  |  |                  |  |
   h  n  o                 "he" "nd"
   |  |  |                  / \
   e  d  o                 "ir" "ory"
  / \    |
 i   r   k
 r   y

台: "台灣"、"台北"、"台中"
中: "中國"、"中山"
```

### 文獻來源

1. **Morrison, D. R.** (1968). "PATRICIA—Practical Algorithm To Retrieve Information Coded in Alphanumeric". *Journal of the ACM*, 15(4), 514-534.

2. **Manning, Raghavan, Schütze** (2008). *Introduction to Information Retrieval*. Cambridge University Press.
   - Chapter 2.3: Dictionaries and tolerant retrieval
   - Chapter 3.4: Wildcard queries

3. **Baeza-Yates, R. & Ribeiro-Neto, B.** (2011). *Modern Information Retrieval* (2nd ed.).
   - Section 8.3: String matching and PAT-trees

---

## 🏗️ 實作架構 (Implementation Architecture)

### 核心資料結構

```python
@dataclass
class PatNode:
    """PAT-tree 節點"""
    label: str = ""              # 邊標籤（可多字元）
    children: Dict[str, 'PatNode'] = None  # 子節點字典
    is_terminal: bool = False    # 是否為詞彙結尾
    frequency: int = 0           # 詞頻
    doc_ids: Set[str] = None     # 包含此詞的文檔ID集合
    metadata: Dict = None        # 額外元數據
```

### 關鍵演算法

#### 1. 插入 (Insert) - O(k)

```python
def insert(self, key: str, doc_id: str = None, metadata: dict = None):
    """
    將詞彙插入 PAT-tree

    時間複雜度: O(k), k = 詞彙長度
    空間複雜度: O(k) worst case
    """
```

**步驟**：
1. 從root開始遍歷
2. 找到匹配的子邊
3. 如果邊標籤部分匹配，分裂節點（split）
4. 如果完全匹配，標記為terminal並更新統計

#### 2. 前綴搜尋 (Prefix Search) - O(k + m)

```python
def starts_with(self, prefix: str) -> List[Tuple[str, PatNode]]:
    """
    找出所有以指定前綴開頭的詞彙

    時間複雜度: O(k + m)
    - k: prefix長度
    - m: 匹配詞彙數量
    """
```

**關鍵實作細節**：
- 處理compressed edges中的部分匹配
- 例如：搜尋prefix "台" 能匹配edge "台灣"

#### 3. 關鍵詞提取 (Keyword Extraction) - O(n log n)

```python
def extract_keywords(self, top_k: int = 20, min_freq: int = 2,
                    min_doc_freq: int = 1, method: str = 'tfidf'):
    """
    從 PAT-tree 提取關鍵詞

    時間複雜度: O(n log n), n = 候選詞數量（排序）
    """
```

**支援的評分方法**：

| 方法 | 公式 | 適用情境 |
|------|------|----------|
| **TF-IDF** | `tf × idf` | 平衡詞頻與獨特性 |
| **Frequency** | `freq` | 高頻詞優先 |
| **Doc Frequency** | `doc_freq` | 廣泛分佈的詞 |
| **Combined** | `(tf × idf) × (1 + log(df + 1))` | 綜合評分 |

其中：
- `tf = freq / total_terms` (normalized term frequency)
- `idf = log((total_docs + 1) / (doc_freq + 1)) + 1`

---

## 🔧 使用方式 (Usage)

### 基本操作

```python
from src.ir.index.pat_tree import PatriciaTree

# 1. 建立 PAT-tree
tree = PatriciaTree()

# 2. 插入詞彙
tree.insert("資訊檢索", doc_id="doc1")
tree.insert("資訊系統", doc_id="doc2")
tree.insert("檢索技術", doc_id="doc1")

# 3. 前綴搜尋
matches = tree.starts_with("資訊")
# 返回: [("資訊檢索", node1), ("資訊系統", node2)]

# 4. 關鍵詞提取
keywords = tree.extract_keywords(
    top_k=20,
    min_freq=2,
    min_doc_freq=1,
    method='tfidf'
)

# 5. 統計資訊
stats = tree.get_statistics()
print(f"Total terms: {stats['total_terms']}")
print(f"Compression ratio: {stats['compression_ratio']:.2f}x")
```

### 從文檔集合建立

```python
from src.ir.index.build_pat_tree import build_pat_tree_from_documents
from src.ir.text.chinese_tokenizer import ChineseTokenizer

# 準備文檔
documents = [
    {'doc_id': 'doc1', 'content': '這是第一篇文檔'},
    {'doc_id': 'doc2', 'content': '這是第二篇文檔'},
]

# 建立 tokenizer
tokenizer = ChineseTokenizer(engine='jieba')

# 建立 PAT-tree
tree = build_pat_tree_from_documents(documents, tokenizer)
```

### Web API 使用

#### 獲取樹結構

```bash
# 無前綴（顯示整棵樹）
curl "http://localhost:5000/api/pat_tree?max_nodes=100"

# 指定前綴
curl "http://localhost:5000/api/pat_tree?prefix=台&max_nodes=10"
```

**回應格式**：
```json
{
    "success": true,
    "prefix": "台",
    "processing_time": 0.028,
    "statistics": {
        "total_terms": 49028,
        "unique_terms": 8478,
        "total_nodes": 9265,
        "compression_ratio": 2.32
    },
    "tree": {
        "label": "ROOT",
        "children": [
            {"label": "台灣", "frequency": 120, "terminal": true},
            {"label": "台北", "frequency": 85, "terminal": true}
        ]
    }
}
```

#### 提取關鍵詞

```bash
curl -X POST "http://localhost:5000/api/pat_tree_keywords" \
  -H "Content-Type: application/json" \
  -d '{
    "top_k": 20,
    "min_freq": 2,
    "min_doc_freq": 1,
    "method": "tfidf"
  }'
```

**回應格式**：
```json
{
    "success": true,
    "method": "tfidf",
    "total_candidates": 8478,
    "keywords": [
        {
            "rank": 1,
            "term": "資訊",
            "score": 0.0245,
            "tf": 0.0212,
            "idf": 1.15,
            "frequency": 1038,
            "doc_count": 104
        }
    ]
}
```

---

## 📊 性能特徵 (Performance Characteristics)

### 當前實作數據（121篇CNA新聞）

```
建構時間 (Build Time):     ~36-39 秒
總詞彙數 (Total Terms):     49,028
唯一詞彙 (Unique Terms):    8,478
樹節點數 (Tree Nodes):      9,265
壓縮率 (Compression):       2.32x
最大深度 (Max Depth):       7
平均詞頻 (Avg Frequency):   5.78
```

### 複雜度分析

| 操作 | 時間複雜度 | 空間複雜度 | 說明 |
|------|-----------|-----------|------|
| **Insert** | O(k) | O(k) | k = 詞彙長度 |
| **Search** | O(k) | O(1) | 精確搜尋 |
| **Prefix Search** | O(k + m) | O(m) | m = 匹配數量 |
| **Extract Keywords** | O(n log n) | O(n) | n = 候選詞數 |
| **Build Tree** | O(T) | O(U·M) | T = total tokens, U = unique terms, M = max term length |

### 壓縮效果

**未壓縮 Trie vs PAT-tree**：
```
Trie nodes:     8478 × 平均路徑長度 ≈ 21,000+ 節點
PAT nodes:      9,265 節點
Space saved:    ~55%
```

---

## 🎨 視覺化界面 (Visualization Interface)

訪問 `http://localhost:5000/pat_tree` 可使用完整的web界面：

### 功能模塊

1. **統計面板**
   - 總詞彙數、唯一詞彙
   - 樹節點數、最大深度
   - 壓縮率、平均詞頻

2. **樹結構視覺化**
   - 互動式樹狀圖
   - 前綴篩選
   - 可控制顯示節點數

3. **關鍵詞提取**
   - 4種評分演算法選擇
   - 可配置Top-K、最小詞頻
   - 即時顯示排名與分數

---

## 🔬 與文獻的對應關係

### Morrison's PATRICIA (1968)

**原始設計**：
- Binary branching (二元分支)
- Bit-level indexing (位元級索引)
- Skip numbers (跳躍數字指示位置)

**本實作**：
- Multi-way branching (多路分支)
- Token-level indexing (詞級索引)
- Edge labels (邊標籤儲存字串片段)

**為何不同**？
- 中文文本特性：詞彙為基本單位，非位元
- 實用性考量：token-level更直觀、易實作
- 性能優勢：對中文檢索更有效

**正確命名**：本實作更接近 **Compressed Radix Tree** 或 **Compact Trie**

### 關鍵詞提取與文獻

本實作結合了多種IR技術：

1. **TF-IDF** (Salton & Buckley, 1988)
   - Term frequency weighting
   - Inverse document frequency

2. **Document Frequency Filtering** (Luhn, 1958)
   - 過濾低頻詞（噪音）
   - 過濾極高頻詞（停用詞）

3. **Percentile Ranking**
   - 提供相對重要性指標

---

## 🚀 進階應用 (Advanced Applications)

### 1. 自動補全 (Auto-completion)

```python
def autocomplete(prefix: str, max_suggestions: int = 10):
    """利用 PAT-tree 實現自動補全"""
    matches = tree.starts_with(prefix)
    # 按詞頻排序
    ranked = sorted(matches, key=lambda x: x[1].frequency, reverse=True)
    return [term for term, _ in ranked[:max_suggestions]]
```

### 2. 拼寫校正 (Spell Correction)

結合編輯距離（Edit Distance）：
```python
def find_similar(query: str, max_distance: int = 2):
    """找出編輯距離相近的詞彙"""
    candidates = []
    for term in tree.term_stats.keys():
        dist = edit_distance(query, term)
        if dist <= max_distance:
            candidates.append((term, dist))
    return sorted(candidates, key=lambda x: x[1])
```

### 3. 主題偵測 (Topic Detection)

基於關鍵詞共現：
```python
def extract_topics(top_k_keywords: int = 50):
    """從高頻關鍵詞提取主題"""
    keywords = tree.extract_keywords(top_k=top_k_keywords, method='combined')
    # 分析關鍵詞的doc_ids overlap
    # 聚類相關詞彙形成主題
```

---

## 📝 測試與驗證 (Testing & Validation)

### 單元測試

```bash
# 運行測試
python test_prefix_debug.py

# 預期輸出
=== Tree Statistics ===
total_terms: 8
unique_terms: 8
compression_ratio: 1.45

=== Test Prefix Search ===
Searching for prefix: '台'
Found 4 matches:
  - 台灣 (freq: 1)
  - 台北 (freq: 1)
  - 台中 (freq: 1)
  - 台南 (freq: 1)
```

### API測試

```bash
# 測試prefix search
curl "http://localhost:5000/api/pat_tree?prefix=台&max_nodes=10" | jq '.success'
# 輸出: true

# 測試keyword extraction
curl -X POST "http://localhost:5000/api/pat_tree_keywords" \
  -d '{"top_k": 20}' -H "Content-Type: application/json" | jq '.keywords | length'
# 輸出: 20
```

---

## 🐛 已知限制與未來改進 (Limitations & Future Work)

### 當前限制

1. **記憶體使用**
   - 所有詞彙載入記憶體
   - 大規模語料可能需要磁碟索引

2. **建構時間**
   - 121篇文檔需要~36秒
   - 增量更新未實作

3. **並發支援**
   - 單執行緒建構
   - 查詢可並發，更新需加鎖

### 未來改進方向

1. **性能優化**
   - [ ] 實作first-character index (O(1) child lookup)
   - [ ] 並行化建構過程
   - [ ] 持久化到磁碟（序列化）

2. **功能增強**
   - [ ] C-value / NC-value 演算法（複合詞提取）
   - [ ] 增量更新支援
   - [ ] 模糊匹配（edit distance）

3. **文檔與測試**
   - [x] 完整技術文檔
   - [ ] 性能benchmark
   - [ ] 更多單元測試

---

## 📚 參考資料 (References)

1. Morrison, D. R. (1968). PATRICIA—Practical Algorithm To Retrieve Information Coded in Alphanumeric. *Journal of the ACM*.

2. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.

3. Baeza-Yates, R., & Ribeiro-Neto, B. (2011). *Modern Information Retrieval* (2nd ed.). Addison Wesley.

4. Knuth, D. E. (1997). *The Art of Computer Programming, Vol. 3: Sorting and Searching* (2nd ed.). Addison-Wesley.

5. Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. *Information Processing & Management*.

---

## 💡 總結 (Summary)

本PAT-tree實作提供：

✅ **完整功能**：插入、搜尋、前綴匹配、關鍵詞提取
✅ **高效性能**：2.32x壓縮率，O(k)操作複雜度
✅ **實用工具**：Web API、視覺化界面、多種評分方法
✅ **良好文檔**：理論背景、使用範例、性能分析

**適用情境**：
- 中文關鍵詞提取
- 自動補全系統
- 資訊檢索索引
- 文本分析工具

---

**作者**: LLMProvider Tooling
**版本**: 1.0
**更新日期**: 2025-11-17
