# 增量索引系統設計 (Incremental Indexing System Design)

**版本**: 1.0
**日期**: 2025-11-19
**用途**: 支援大規模新聞語料庫的動態更新

---

## 1. 系統需求

### 功能需求
- ✅ **增量索引**: 新增文章不需重建整個索引
- ✅ **快速查詢**: 次秒級回應時間
- ✅ **Field-based 索引**: 支援標題、內容、日期、來源等欄位
- ✅ **去重機制**: 自動偵測重複文章
- ✅ **版本管理**: 追蹤索引更新歷史

### 非功能需求
- **可擴展性**: 支援百萬級文章規模
- **容錯性**: 索引損壞時可恢復
- **效率**: 增量更新時間 < 完整更新時間 10%

---

## 2. 架構設計

### 2.1 索引結構

```
data/
├── index/                          # 索引主目錄
│   ├── inverted/                   # 倒排索引
│   │   ├── term_index.pkl         # 詞彙 → Postings
│   │   ├── doc_lengths.pkl        # 文檔長度（用於 BM25）
│   │   └── term_stats.pkl         # 詞彙統計（DF, IDF）
│   ├── forward/                    # 正向索引
│   │   ├── doc_metadata.pkl       # 文檔元資料
│   │   ├── doc_content.pkl        # 文檔內容（可選）
│   │   └── doc_hash.pkl           # 文檔雜湊（去重）
│   ├── field/                      # 欄位索引
│   │   ├── title_index.pkl        # 標題索引
│   │   ├── content_index.pkl      # 內容索引
│   │   ├── date_index.pkl         # 日期索引
│   │   └── source_index.pkl       # 來源索引
│   └── meta/                       # 元資料
│       ├── index_info.json        # 索引資訊（版本、統計）
│       ├── update_log.jsonl       # 更新日誌
│       └── checkpoint.json        # 檢查點（已處理檔案）
└── raw/                            # 原始資料
    └── *.jsonl                     # 爬蟲收集的新聞資料
```

### 2.2 索引流程

```
┌──────────────┐
│ 新資料來源   │ (JSONL files)
└──────┬───────┘
       │
       ├─> 1. 讀取新檔案
       │
       ├─> 2. 去重檢查 (hash comparison)
       │   │
       │   ├─ 已存在 → 跳過
       │   └─ 新文章 → 繼續
       │
       ├─> 3. 文字預處理
       │   ├─ Tokenization (jieba)
       │   ├─ Stopword removal
       │   └─ Normalization
       │
       ├─> 4. 建立增量索引
       │   ├─ 更新倒排索引
       │   ├─ 更新正向索引
       │   ├─ 更新欄位索引
       │   └─ 更新統計資訊
       │
       └─> 5. 合併索引
           ├─ 記錄檢查點
           └─ 更新元資料
```

---

## 3. 核心演算法

### 3.1 去重演算法

```python
def is_duplicate(new_doc: dict, doc_hash_index: dict) -> bool:
    """
    使用 SimHash 或 MD5 快速去重

    Time: O(1) for hash lookup
    Space: O(N) where N = number of documents
    """
    content_hash = hashlib.md5(new_doc['content'].encode()).hexdigest()

    if content_hash in doc_hash_index:
        # 精確重複
        return True

    # 可選: 使用 SimHash 檢測近似重複（標題+前100字）
    sim_hash = simhash(new_doc['title'] + new_doc['content'][:100])

    for existing_hash in doc_hash_index.values():
        if hamming_distance(sim_hash, existing_hash) < 3:  # 容忍 3-bit 差異
            return True

    return False
```

### 3.2 增量索引演算法

```python
def incremental_update(new_docs: List[dict],
                       existing_index: InvertedIndex) -> InvertedIndex:
    """
    增量更新倒排索引

    Time: O(M * T) where M = new docs, T = avg terms per doc
    Space: O(U) where U = unique terms in new docs
    """
    # 從現有索引獲取統計資訊
    N_old = existing_index.num_docs
    N_new = N_old + len(new_docs)

    # 分配新的 doc_id (連續編號)
    next_doc_id = N_old

    for doc in new_docs:
        doc_id = next_doc_id
        next_doc_id += 1

        # Tokenize
        terms = tokenize(doc['content'])
        doc_length = len(terms)

        # 更新倒排索引
        for term in set(terms):
            tf = terms.count(term)

            if term not in existing_index.postings:
                existing_index.postings[term] = []

            existing_index.postings[term].append({
                'doc_id': doc_id,
                'tf': tf,
                'positions': [i for i, t in enumerate(terms) if t == term]
            })

            # 更新 DF
            existing_index.df[term] = existing_index.df.get(term, 0) + 1

        # 更新文檔長度
        existing_index.doc_lengths[doc_id] = doc_length

    # 重新計算 IDF (僅針對新增或更新的詞彙)
    for term in existing_index.postings:
        df = existing_index.df[term]
        existing_index.idf[term] = math.log((N_new - df + 0.5) / (df + 0.5) + 1)

    existing_index.num_docs = N_new

    return existing_index
```

### 3.3 索引合併演算法

對於長時間運行的系統，定期執行索引合併以優化效能：

```python
def merge_index_segments(segments: List[IndexSegment]) -> IndexSegment:
    """
    合併多個索引片段（類似 Lucene 的 segment merging）

    Time: O(T * log(S)) where T = total postings, S = segments
    Space: O(T)
    """
    merged = IndexSegment()

    # 使用最小堆合併 postings lists
    for term in get_all_terms(segments):
        postings = []

        for segment in segments:
            if term in segment.postings:
                postings.extend(segment.postings[term])

        # 按 doc_id 排序並合併
        postings.sort(key=lambda x: x['doc_id'])
        merged.postings[term] = postings

    return merged
```

---

## 4. 查詢介面

### 4.1 支援的查詢類型

```python
# 1. 關鍵字查詢
results = search("台灣 選舉", top_k=10)

# 2. 欄位查詢
results = search("title:總統 AND content:政策", top_k=10)

# 3. 日期範圍查詢
results = search("疫情", date_range=("2024-01-01", "2024-12-31"))

# 4. 來源過濾
results = search("經濟", sources=["yahoo", "ltn", "udn"])

# 5. 組合查詢
results = search(
    query="AI 人工智慧",
    fields=["title", "content"],
    date_range=("2024-06-01", "2024-12-31"),
    sources=["yahoo", "storm"],
    top_k=20
)
```

### 4.2 排序模型

支援多種排序演算法：
- **BM25**: 預設，平衡效果最好
- **TF-IDF**: 傳統 VSM
- **Language Model**: 語言模型機率排序
- **Learning to Rank**: 機器學習排序（未來）

---

## 5. 更新策略

### 5.1 即時更新 (Real-time)

```bash
# 監控 data/raw/ 目錄，有新檔案立即索引
python scripts/index_monitor.py --watch data/raw --interval 60
```

### 5.2 批次更新 (Batch)

```bash
# 每小時/每天執行批次索引
python scripts/build_index.py --incremental --batch-size 10000
```

### 5.3 完整重建 (Full Rebuild)

```bash
# 索引損壞或需要重新分詞時
python scripts/build_index.py --full --optimize
```

---

## 6. 效能指標

### 6.1 索引效能

| 操作 | 目標時間 | 實際時間 | 備註 |
|------|---------|---------|------|
| 增量索引 (1K docs) | < 10s | TBD | |
| 增量索引 (10K docs) | < 60s | TBD | |
| 完整索引 (100K docs) | < 10min | TBD | |
| 完整索引 (1M docs) | < 2h | TBD | |

### 6.2 查詢效能

| 查詢類型 | 目標時間 | 實際時間 | 備註 |
|---------|---------|---------|------|
| 簡單關鍵字 | < 50ms | TBD | |
| 欄位查詢 | < 100ms | TBD | |
| 複雜組合查詢 | < 200ms | TBD | |

---

## 7. 容錯機制

### 7.1 檢查點系統

```json
{
  "last_processed_file": "yahoo_politics_14days.jsonl",
  "last_processed_line": 45639,
  "last_doc_id": 257136,
  "timestamp": "2025-11-19T23:31:00Z",
  "status": "completed"
}
```

### 7.2 錯誤恢復

1. **索引損壞**: 從最近的檢查點重新索引
2. **處理中斷**: 從檢查點繼續處理
3. **資料損壞**: 跳過損壞的文檔並記錄

---

## 8. 監控與維護

### 8.1 索引健康檢查

```bash
python scripts/index_health_check.py
```

輸出：
```
✓ 索引完整性: OK
✓ 檔案一致性: OK
✓ 統計資訊準確性: OK
⚠ 索引碎片化: 23% (建議合併)
```

### 8.2 索引統計

```bash
python scripts/index_stats.py
```

輸出：
```
總文檔數: 257,137
唯一詞彙數: 1,234,567
平均文檔長度: 432.5 tokens
索引大小: 2.3 GB
最後更新: 2025-11-19 23:31:00
```

---

## 9. 實作優先順序

### Phase 1: 基礎索引 (本週)
- [x] 基本倒排索引
- [x] 去重機制
- [ ] BM25 排序
- [ ] 簡單查詢介面

### Phase 2: 增量更新 (下週)
- [ ] 檢查點系統
- [ ] 增量索引演算法
- [ ] 索引合併
- [ ] 監控腳本

### Phase 3: 進階功能 (第三週)
- [ ] 欄位索引
- [ ] 複雜查詢
- [ ] 效能優化
- [ ] Web API

---

## 10. 參考資料

- Manning et al., *Introduction to Information Retrieval* (課程教科書)
- Lucene 索引架構: https://lucene.apache.org/core/
- Elasticsearch 增量索引設計
- Whoosh (Python IR library)
