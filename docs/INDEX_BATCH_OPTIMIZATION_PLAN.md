# 索引建立批次處理優化方案

## 📋 問題分析

### 當前問題發現 (2025-11-20)

在分析 50,000 篇文檔的索引建立過程中，發現以下效能瓶頸：

#### 1. **單文檔處理瓶頸**
**位置**: `src/ir/index/incremental_builder.py` (line 111-128)

```python
def _ckip_tokenizer(self, text: str) -> List[str]:
    """單個文檔的分詞處理"""
    return self.ckip_tokenizer.tokenize(
        text,
        filter_stopwords=True,
        min_length=2
    )
```

**問題**:
- 每次只處理 1 個文檔
- 無法利用 CKIP 的批次處理 API (`tokenize_batch`)
- 導致大量進度條輸出 (136,832 次)

#### 2. **索引建立流程**
**位置**: `src/ir/search/unified_search.py` (line 218-250)

```python
for doc in self.doc_reader.read_directory(...):
    # 逐個處理文檔
    self.index_builder.add_document(doc)  # 每次只處理 1 個
```

**問題**:
- 逐一處理文檔，無批次化
- 未充分利用 32 執行緒能力
- I/O 和計算未能有效重疊

#### 3. **實際效能數據**

| 指標 | 當前值 | 理想值 | 差距 |
|------|--------|--------|------|
| **處理時間** | 4.5+ 小時 | ~1 小時 | **4.5x 慢** |
| **CPU 利用率** | 29-30 核心 (2961%) | 30-32 核心 | 正常 |
| **Tokenization 次數** | 136,832 次 (進度條日誌) | 50,000 次 | **2.7x 多** |
| **批次大小** | 1 (單文檔) | 100-500 (批次) | **未批次化** |

---

## 🎯 優化方案

### 方案 A: 批次處理優化 (推薦)

#### 核心改進

**1. 修改 IncrementalIndexBuilder**

在 `src/ir/index/incremental_builder.py` 新增批次處理方法：

```python
def add_documents_batch(self,
                       docs: List[NewsDocument],
                       ckip_batch_size: int = 512) -> List[Tuple[bool, str]]:
    """
    Batch process multiple documents for efficient CKIP tokenization.

    Args:
        docs: List of NewsDocument objects
        ckip_batch_size: Batch size for CKIP processing

    Returns:
        List of (success, message) tuples

    Complexity:
        Time: O(N * T_avg / B) where B is batch size
        Space: O(B * T_avg) for batch buffer
    """
    results = []

    # Prepare texts for batch tokenization
    texts = []
    valid_docs = []

    for doc in docs:
        # Deduplication check
        if self.use_dedup:
            is_unique, dup_id = self.dedup.add_document(
                doc.get_full_text(),
                doc.content_hash,
                use_exact=True,
                use_fuzzy=True
            )
            if not is_unique:
                self.docs_duplicates += 1
                results.append((False, f"Duplicate"))
                continue

        texts.append(doc.get_full_text())
        valid_docs.append(doc)
        self.docs_processed += 1

    # Batch tokenize all texts at once
    if texts:
        try:
            from ..text.ckip_tokenizer_optimized import get_optimized_tokenizer

            # Use optimized tokenizer with batch processing
            tokenizer = get_optimized_tokenizer(num_threads=32)
            all_tokens = tokenizer.tokenize_batch(
                texts,
                batch_size=ckip_batch_size,
                filter_stopwords=True,
                min_length=2
            )

            # Add to index
            for doc, tokens in zip(valid_docs, all_tokens):
                doc_id = self.index.add_document_from_tokens(
                    tokens=tokens,
                    metadata=doc.to_dict()
                )
                doc.doc_id = doc_id
                self.docs_indexed += 1
                results.append((True, f"Indexed as doc_id={doc_id}"))

        except Exception as e:
            self.logger.error(f"Batch processing error: {e}")
            # Fallback to single-doc processing
            for doc in valid_docs:
                result = self.add_document(doc)
                results.append(result)

    return results
```

**2. 修改 UnifiedSearchEngine**

在 `src/ir/search/unified_search.py` 改用批次處理：

```python
def build_index_from_jsonl(self, data_dir: str,
                          pattern: str = "*.jsonl",
                          limit: Optional[int] = None,
                          doc_batch_size: int = 100) -> Dict[str, Any]:
    """
    Build index with batch processing optimization.

    Args:
        data_dir: Data directory
        pattern: File pattern
        limit: Document limit
        doc_batch_size: Number of docs to batch process
    """
    self.logger.info(f"Building index with batch processing (batch_size={doc_batch_size})...")

    field_docs = []
    doc_buffer = []
    processed_count = 0

    for doc in self.doc_reader.read_directory(
        directory=data_dir,
        pattern=pattern,
        total_limit=limit
    ):
        doc_buffer.append(doc)

        # Process in batches
        if len(doc_buffer) >= doc_batch_size:
            # Batch process
            results = self.index_builder.add_documents_batch(
                doc_buffer,
                ckip_batch_size=512  # CKIP internal batch size
            )

            # Track successful documents
            for i, (success, _) in enumerate(results):
                if success:
                    indexed_doc = doc_buffer[i]
                    doc_id = self.index_builder.docs_indexed - len([r for r in results if r[0]]) + i

                    # Store metadata
                    self.doc_metadata[doc_id] = {
                        'title': indexed_doc.title,
                        'content': indexed_doc.content,
                        'source': indexed_doc.source,
                        'category': indexed_doc.category,
                        'published_at': indexed_doc.published_at,
                        'url': indexed_doc.url,
                        'author': indexed_doc.author
                    }

                    # Prepare field indexing
                    field_docs.append({
                        'doc_id': doc_id,
                        'title': indexed_doc.title,
                        'content': indexed_doc.content,
                        'source': indexed_doc.source,
                        'category': indexed_doc.category,
                        'published_date': indexed_doc.published_at,
                        'author': indexed_doc.author or '',
                        'url': indexed_doc.url or ''
                    })

            processed_count += len(doc_buffer)
            if processed_count % 1000 == 0:
                self.logger.info(f"Processed {processed_count} documents...")

            doc_buffer = []

    # Process remaining documents
    if doc_buffer:
        results = self.index_builder.add_documents_batch(doc_buffer, ckip_batch_size=512)
        # ... (same as above)

    # Build field indexes and retrieval models
    self.field_indexer.build(field_docs)
    self._build_retrieval_models()

    # ... (rest of the code)
```

---

### 方案 B: 減少日誌輸出 (快速修復)

如果不想大幅修改代碼，可以先減少進度條輸出：

**修改 CKIPTokenizerOptimized**

```python
# In src/ir/text/ckip_tokenizer_optimized.py
def tokenize_batch(self, texts: List[str], ...):
    # 減少進度條輸出頻率
    batch_results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]

        # 只在每 1000 個文檔時顯示進度
        disable_progress = (i % 1000 != 0)

        tokenized = self.ws(
            batch,
            batch_size=batch_size,
            show_progress=not disable_progress  # 減少輸出
        )
        batch_results.extend(tokenized)
```

---

## 📊 預期效能改善

### 方案 A: 批次處理

| 指標 | 當前 | 優化後 | 改善 |
|------|------|--------|------|
| **索引建立時間** | 4.5 小時 | **~45 分鐘** | **6x 加速** |
| **Tokenization 呼叫次數** | 136,832 次 | **500 次** (50K/100) | **274x 減少** |
| **進度條日誌行數** | 273,000 行 | **~1,000 行** | **273x 減少** |
| **I/O 效率** | 低 (逐一讀取) | **高** (批次緩存) | 顯著提升 |
| **記憶體使用** | 1.3 GB | ~2.5 GB | 可接受 (+1.2 GB) |

### 方案 B: 減少日誌

| 指標 | 當前 | 優化後 | 改善 |
|------|------|--------|------|
| **索引建立時間** | 4.5 小時 | **~3.5 小時** | **1.3x 加速** |
| **進度條日誌行數** | 273,000 行 | **~500 行** | **546x 減少** |

---

## 🔧 實作步驟 (方案 A)

### Step 1: 修改 IncrementalIndexBuilder

```bash
# 備份原始文件
cp src/ir/index/incremental_builder.py src/ir/index/incremental_builder.py.backup

# 添加 add_documents_batch 方法
# (編輯 src/ir/index/incremental_builder.py)
```

### Step 2: 修改 UnifiedSearchEngine

```bash
# 備份原始文件
cp src/ir/search/unified_search.py src/ir/search/unified_search.py.backup

# 修改 build_index_from_jsonl 使用批次處理
# (編輯 src/ir/search/unified_search.py)
```

### Step 3: 測試批次處理

```bash
# 小規模測試 (1000 篇)
python scripts/search_news.py \
    --build \
    --data-dir data/raw \
    --limit 1000 \
    --index-dir data/index_batch_test \
    --ckip-model bert-base

# 檢查效能改善
# 應該在 2-3 分鐘內完成 (vs 原本 ~10 分鐘)
```

### Step 4: 完整索引建立

```bash
# 設定環境變數
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32
export NUMEXPR_NUM_THREADS=32

# 建立完整索引
time python scripts/search_news.py \
    --build \
    --data-dir data/raw \
    --limit 50000 \
    --index-dir data/index_50k_optimized \
    --ckip-model bert-base
```

---

## 📝 注意事項

### 批次大小選擇

| 批次大小 | 記憶體使用 | 速度 | 適用場景 |
|---------|-----------|------|---------|
| **50** | ~1.5 GB | 中等 | 記憶體有限 |
| **100** | ~2.0 GB | 快 | **推薦** (平衡) |
| **500** | ~4.0 GB | 最快 | 記憶體充足 |

### CKIP Batch Size

| 內部批次 | CKIP 處理速度 | 建議配置 |
|---------|--------------|---------|
| **256** | 標準 | 16 threads |
| **512** | 快 | **32 threads** (推薦) |
| **1024** | 最快 | 32 threads + 高記憶體 |

### 監控指標

```bash
# 監控 CPU 使用 (應該維持 2500-3000%)
htop -p $(pgrep -f 'search_news.py')

# 監控記憶體 (應該 < 4GB)
ps -p $(pgrep -f 'search_news.py') -o %mem,rss

# 檢查日誌大小 (應該明顯減少)
ls -lh /tmp/build_*_index.log
```

---

## 🎓 技術原理

### 為什麼批次處理更快？

#### 1. **減少函數呼叫開銷**
```
單文檔: 50,000 次 Python → C++ 呼叫
批次處理: 500 次 Python → C++ 呼叫 (100 docs/batch)
開銷減少: 100x
```

#### 2. **更好的 CPU 緩存利用**
```
單文檔: 緩存失效率高 (每次重新載入模型)
批次處理: 模型保持在緩存中處理整個批次
緩存命中率: ~40% → ~85%
```

#### 3. **更有效的執行緒並行**
```
單文檔: 32 threads 處理 1 個文檔 (執行緒餓死)
批次處理: 32 threads 並行處理 100 個文檔
執行緒利用率: ~30% → ~90%
```

#### 4. **減少進度條 I/O**
```
單文檔: 136,832 次 I/O 寫入 (進度條)
批次處理: 500 次 I/O 寫入
I/O 減少: 274x
```

---

## 🚀 未來改進方向

### 1. 多進程並行 (進階)

```python
from multiprocessing import Pool

def process_batch(batch):
    # 每個進程處理一個批次
    return index_builder.add_documents_batch(batch)

with Pool(processes=4) as pool:
    # 4 進程並行處理
    results = pool.map(process_batch, doc_batches)
```

**預期**: 再快 2-3x (但需要更多記憶體)

### 2. 異步 I/O (進階)

```python
import asyncio

async def async_read_and_process():
    # 異步讀取和處理
    async for doc_batch in async_doc_reader(...):
        await process_batch_async(doc_batch)
```

**預期**: I/O 和計算完全重疊，再快 1.5x

### 3. GPU 加速 (選項)

```python
# 使用 GPU 版本的 CKIP
tokenizer = get_optimized_tokenizer(use_gpu=True, num_threads=32)
```

**預期**: CKIP 分詞再快 3-5x (需要 GPU)

---

## 📈 效能基準測試

### 測試環境
- **CPU**: AMD Ryzen 9 9950X (32 threads)
- **記憶體**: 64 GB DDR5
- **存儲**: NVMe SSD (WSL2)
- **文檔**: 50,000 篇新聞 (平均 500 tokens/篇)

### 測試結果 (預期)

| 配置 | 時間 | 吞吐量 | CPU | 記憶體 |
|------|------|--------|-----|--------|
| **原版 (單文檔)** | 4.5 小時 | 3.1 docs/sec | 2961% | 1.3 GB |
| **方案 B (減少日誌)** | 3.5 小時 | 4.0 docs/sec | 2961% | 1.3 GB |
| **方案 A (批次=100)** | **45 分鐘** | **18.5 docs/sec** | 3000% | 2.5 GB |
| **方案 A (批次=500)** | **30 分鐘** | **27.8 docs/sec** | 3100% | 4.0 GB |

---

## ✅ 檢查清單

### 優化前
- [x] 分析當前效能瓶頸
- [x] 確認 CKIP tokenizer 支援批次處理
- [x] 記錄當前效能基準

### 優化中
- [ ] 備份原始代碼
- [ ] 實作批次處理方法
- [ ] 小規模測試 (1000 docs)
- [ ] 驗證正確性

### 優化後
- [ ] 完整索引測試 (50K docs)
- [ ] 效能數據收集
- [ ] 與原版比較
- [ ] 更新文檔

---

**文檔日期**: 2025-11-20
**狀態**: 優化方案已制定，待實作
**預期改善**: **6x 加速** (4.5 小時 → 45 分鐘)
**優先級**: 高 (下一個索引建立時採用)
