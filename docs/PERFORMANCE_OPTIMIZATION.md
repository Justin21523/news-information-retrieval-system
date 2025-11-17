# PAT-tree 性能優化計畫
# Performance Optimization Plan

## 📊 當前性能基線 (Current Baseline)

### 建構性能 (Build Performance)
```
數據集: 121篇CNA新聞
總詞彙: 49,028個
唯一詞彙: 8,478個
建構時間: ~36-39秒
記憶體使用: ~150MB (估計)
```

### 查詢性能 (Query Performance)
```
Prefix Search: ~0.03秒 (包含PAT-tree建構時間)
Keyword Extraction: ~0.05秒 (Top-20, TF-IDF)
Tree Visualization: ~0.03秒 (100 nodes)
```

---

## 🎯 優化目標 (Optimization Goals)

### 短期目標 (Immediate)
- [ ] Child lookup: O(n) → O(1)
- [ ] 建構時間: 36s → 25s (-30%)
- [ ] 查詢響應: 30ms → 15ms (-50%)
- [ ] 記憶體使用: 150MB → 120MB (-20%)

### 中期目標 (Medium-term)
- [ ] 支援增量更新
- [ ] 查詢結果快取
- [ ] API分頁功能
- [ ] 並行化建構

### 長期目標 (Long-term)
- [ ] 持久化到磁碟
- [ ] 分散式索引
- [ ] 即時更新支援

---

## 🔧 優化策略 (Optimization Strategies)

### 1. First-Character Index (O(1) Child Lookup)

**當前實作**：
```python
# O(n) - 遍歷所有子節點
for child_label, child_node in node.children.items():
    if child_label[0] == first_char:
        # ...
```

**優化方案**：
```python
@dataclass
class PatNode:
    children: Dict[str, 'PatNode']
    first_char_index: Dict[str, List['PatNode']]  # 新增

# O(1) - 直接查找
first_char = key[0]
candidates = node.first_char_index.get(first_char, [])
```

**預期效果**：
- Insert: 36s → 28s (-22%)
- Prefix Search: 30ms → 10ms (-67%)

---

### 2. Batch Insertion (批次插入)

**當前實作**：
```python
for term in terms:
    tree.insert(term, doc_id)  # 逐個插入
```

**優化方案**：
```python
def batch_insert(self, terms: List[Tuple[str, str]]):
    """批次插入，減少重複遍歷"""
    # 按照prefix分組
    grouped = defaultdict(list)
    for term, doc_id in terms:
        grouped[term[0]].append((term, doc_id))

    # 批次處理
    for prefix, group in grouped.items():
        self._batch_insert_group(group)
```

**預期效果**：
- 建構時間: 28s → 22s (-21%)

---

### 3. Query Result Caching (查詢快取)

**實作方案**：
```python
from functools import lru_cache

class PatriciaTree:
    def __init__(self):
        self._query_cache = {}
        self._cache_size = 1000
        self._cache_ttl = 3600  # 1 hour

    @lru_cache(maxsize=128)
    def starts_with(self, prefix: str):
        """Cached prefix search"""
        # ...
```

**快取策略**：
- LRU (Least Recently Used)
- TTL: 1 hour
- Max size: 1000 entries
- Cache hit rate target: >80%

**預期效果**：
- Prefix Search (cached): 10ms → 1ms (-90%)
- Keyword Extraction (cached): 50ms → 5ms (-90%)

---

### 4. API Pagination (分頁)

**當前問題**：
- 返回全部結果可能很大
- 前端渲染緩慢
- 網路傳輸開銷

**優化方案**：
```python
@app.route('/api/pat_tree')
def get_pat_tree():
    page = request.args.get('page', 1, type=int)
    page_size = request.args.get('page_size', 50, type=int)

    results = tree.starts_with(prefix)
    total = len(results)

    start = (page - 1) * page_size
    end = start + page_size

    return {
        'data': results[start:end],
        'total': total,
        'page': page,
        'page_size': page_size,
        'total_pages': (total + page_size - 1) // page_size
    }
```

**預期效果**：
- API響應時間: 30ms → 10ms (-67%)
- 前端渲染: 200ms → 50ms (-75%)

---

### 5. Memory Optimization (記憶體優化)

#### 5.1 String Interning
```python
# 重複字串使用同一記憶體
self.label = sys.intern(label)
```

#### 5.2 Slot-based Classes
```python
@dataclass
class PatNode:
    __slots__ = ['label', 'children', 'is_terminal',
                 'frequency', 'doc_ids', 'metadata']
```

#### 5.3 Lazy Loading
```python
class PatNode:
    def __init__(self):
        self._metadata = None  # 延遲載入

    @property
    def metadata(self):
        if self._metadata is None:
            self._metadata = {}
        return self._metadata
```

**預期效果**：
- 記憶體使用: 150MB → 100MB (-33%)

---

### 6. Parallel Construction (並行建構)

**方案A: 多進程**
```python
from multiprocessing import Pool

def build_parallel(documents, n_workers=4):
    # 分割文檔
    chunks = np.array_split(documents, n_workers)

    # 並行建立子樹
    with Pool(n_workers) as pool:
        subtrees = pool.map(build_subtree, chunks)

    # 合併子樹
    return merge_trees(subtrees)
```

**方案B: 多執行緒**
```python
from concurrent.futures import ThreadPoolExecutor

def build_threaded(documents):
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_doc, doc)
                  for doc in documents]
        results = [f.result() for f in futures]
```

**預期效果**：
- 建構時間: 22s → 8s (-64%, 4 cores)

---

## 📈 性能基準測試 (Benchmarks)

### 測試腳本

```python
import time
import psutil
import tracemalloc

class PerformanceBenchmark:
    def __init__(self):
        self.results = {}

    def measure_build_time(self, tree, documents):
        """測試建構時間"""
        start = time.time()
        for doc in documents:
            for term in tokenize(doc):
                tree.insert(term, doc['id'])
        elapsed = time.time() - start
        return elapsed

    def measure_query_time(self, tree, queries):
        """測試查詢時間"""
        times = []
        for query in queries:
            start = time.time()
            tree.starts_with(query)
            times.append(time.time() - start)
        return {
            'mean': np.mean(times),
            'p50': np.percentile(times, 50),
            'p95': np.percentile(times, 95),
            'p99': np.percentile(times, 99)
        }

    def measure_memory(self, tree):
        """測試記憶體使用"""
        tracemalloc.start()
        # Build tree
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return {
            'current_mb': current / 1024 / 1024,
            'peak_mb': peak / 1024 / 1024
        }
```

### 基準數據

| 指標 | 當前 | 目標 | 優化後 | 改善 |
|-----|------|------|--------|------|
| **建構時間** | 36s | 25s | TBD | TBD |
| **Prefix Search (cold)** | 30ms | 15ms | TBD | TBD |
| **Prefix Search (warm)** | 30ms | 1ms | TBD | TBD |
| **Keyword Extraction** | 50ms | 25ms | TBD | TBD |
| **記憶體使用** | 150MB | 120MB | TBD | TBD |
| **壓縮率** | 2.32x | 2.50x | TBD | TBD |

---

## 🔍 性能分析工具 (Profiling Tools)

### CPU Profiling
```python
import cProfile
import pstats

# Profile建構過程
cProfile.run('build_tree(documents)', 'profile_stats')
stats = pstats.Stats('profile_stats')
stats.sort_stats('cumulative').print_stats(20)
```

### Memory Profiling
```python
from memory_profiler import profile

@profile
def build_tree(documents):
    tree = PatriciaTree()
    # ...
```

### Line Profiling
```python
from line_profiler import LineProfiler

lp = LineProfiler()
lp.add_function(PatriciaTree.insert)
lp.run('build_tree(documents)')
lp.print_stats()
```

---

## 📋 實施計畫 (Implementation Plan)

### Phase 1: 核心優化 (Week 1)
- [x] 規劃優化方案
- [ ] 實作first-character index
- [ ] 批次插入優化
- [ ] 建立性能基準測試

### Phase 2: 快取與分頁 (Week 2)
- [ ] 實作查詢結果快取
- [ ] API分頁功能
- [ ] 記憶體優化

### Phase 3: 並行化 (Week 3)
- [ ] 並行建構實驗
- [ ] 執行緒安全檢查
- [ ] 效能驗證

### Phase 4: 驗證與文檔 (Week 4)
- [ ] 完整性能測試
- [ ] 更新技術文檔
- [ ] 生成性能報告

---

## 🎯 驗收標準 (Acceptance Criteria)

### 必須達成
✅ 建構時間減少 > 30%
✅ 查詢響應減少 > 50%
✅ 所有現有測試通過
✅ 無功能退化

### 期望達成
⭐ 記憶體使用減少 > 20%
⭐ Cache hit rate > 80%
⭐ API P99 latency < 100ms

---

## 📚 參考資料 (References)

1. **Optimization Techniques**:
   - Knuth, D. E. (1997). *The Art of Computer Programming, Vol. 1*
   - Cormen et al. (2009). *Introduction to Algorithms* (3rd ed.)

2. **Python Performance**:
   - Gorelick & Ozsvald (2014). *High Performance Python*
   - https://wiki.python.org/moin/PythonSpeed/PerformanceTips

3. **Caching Strategies**:
   - Podlipnig & Böszörmenyi (2003). "A survey of web cache replacement strategies"

---

**更新日期**: 2025-11-17
**版本**: 1.0
**狀態**: 🚧 進行中
