# 實作進度紀錄 *Worklog*

本檔案用於記錄每次實作/修改的「目的、範圍、片段程式碼、原理說明、驗證方式與下一步」。

- 程式碼內註解與 docstring **以英文為主**（符合專案規範），本檔案以**繁體中文**整理並搭配必要的雙語術語。
- 每次我在專案內進行實作，都會在此新增一筆紀錄（依日期排序）。

---

## 2025-12-26：全專案註解覆蓋率盤點與補齊（第 1 批）

### 目標

- 盤點全專案 Python 程式碼的 docstring 覆蓋率，找出缺口位置。
- 以「可讀性」為主補齊缺漏的 docstring（module / function / class），讓讀者能快速掌握用途、輸入輸出與行為。

### 盤點結果（初始狀態）

- Python 檔案數：175
- Module docstrings：166/175（94.9%）
- Function docstrings：1530/1711（89.4%）
- Class docstrings：199/235（84.7%）

缺口主要集中在：

- `tests/`：fixture functions 與 `Test*` 類別多數未加 docstring（測試名稱雖自描述，但對「學習/回顧」不夠友善）。
- `scripts/`：部分 CLI 的 `main()` 未加 docstring（不易快速理解參數/模式）。
- `src/**/__init__.py`：多為空檔，缺少 package 用途摘要。

### 本次完成

- 新增 `docs/PROGRESS.md` 作為固定「進度/原理整理」檔案，並從 `docs/README.md` 加入連結。
- 為 `src/**/__init__.py` 與 `tests/__init__.py` 補上英文 module docstring（補齊 package 用途摘要）。
- 為多個 `tests/test_*.py` 的 fixture 與 `Test*` 類別補上英文 docstring，提升可讀性與搜尋性。
- 為多個 `scripts/*` 的 `main()` 補上英文 docstring，讓 CLI 入口點更容易理解。
- 補齊少數散落的缺漏 docstring（例如 `app_simple.py` 的 nested sort key、`benchmark_pat_tree.py` 的 `__init__`、`test_app_integration.py` 的 tokenizer、`scripts/quick_test_v2.py` 的結果記錄函式）。
- `tests/test_topic.py` 於缺少（或安裝不完整的）topic modeling 依賴時，自動在 pytest 收集階段跳過，避免整體測試因可選依賴而中斷。

### 更新後狀態（本次作業結束）

以全專案 Python 檔案（`find . -name "*.py"`）重新統計：

- Python 檔案數：182
- Module docstrings：182/182（100.0%）
- Function docstrings：1637/1779（92.0%）
- Class docstrings：236/236（100.0%）

### 片段程式碼（本次採用的註解策略）

以「可搜尋、可快速掃描」為原則：對 fixture 與測試類別補上簡短英文 docstring，避免逐行註解造成噪音。

```python
@pytest.fixture
def sample_docs():
    """Return a small, deterministic document set for unit tests."""
    return [
        "information retrieval systems",
        "vector space model",
        "boolean retrieval",
    ]


@pytest.mark.unit
class TestRanking:
    """Unit tests for ranked retrieval behaviors (score ordering, top-k cutoff)."""
```

### 原理說明（為什麼優先補 docstring）

- **Docstring** 是 Python 物件的「內建說明文字」：在 REPL、IDE tooltip、`help()`、自動文件產生工具中都能直接顯示。
- 對學習 IR 系統而言，最佳的註解密度通常是「**高層摘要 + 關鍵步驟說明**」：  
  1) 模組在做什麼（module docstring）  
  2) 主要 API 的契約（function/class docstring：輸入、輸出、例外、複雜度）  
  3) 複雜流程在程式碼內用少量 inline comment 標出「為什麼」而非「做了什麼」  

### 驗證方式

- 重新統計 docstring 覆蓋率（module / function / class）。
- 執行 `pytest -m "not slow"`（若環境允許）確保只改註解不影響行為。

### 驗證結果（本次環境）

- 針對本次有改動的單元測試檔案，已執行並通過：  
  `pytest tests/test_metrics.py tests/test_rocchio.py tests/test_summarization.py tests/test_term_weighting.py tests/test_clustering.py tests/test_vsm.py`
- `pytest -m "not slow"` 在本次環境仍會因多個「可選依賴/外部環境」相關測試失敗（例如：crawler 設定、CKIP/NER/SuPar、RAKE/YAKE 等）。這些失敗與本次註解補齊無關，需另外處理依賴安裝或為測試加上更明確的 skip/marker 才能在無完整依賴的環境下全綠。

### 下一步（第 2 批建議）

- 針對 `src/ir/` 核心演算法（Boolean/VSM/BM25/Rocchio/Query Parser 等）挑選「最常閱讀的路徑」補上更教學導向的英文註解（著重 *why*、複雜度、常見陷阱）。
- 把需要額外套件/模型的測試加上 `skip` 條件與 marker（例如 `requires_data`、`slow`、`requires_optional_deps`），讓 `pytest -m "not slow"` 在基礎環境下可穩定全綠。

---

## 2025-12-26：核心檢索模組教學註解補強（第 2 批）

### 目標

- 針對你指定的核心主題補上更「教學導向」的英文註解與 docstring：  
  Boolean 查詢解析、VSM/BM25 排序、Rocchio、Query Parser。
- 在不改變行為的前提下，讓讀者能直接從程式碼理解：資料流、演算法直覺、複雜度、以及目前實作的限制。

### 本次修改範圍

- Boolean 查詢解析：`src/ir/retrieval/boolean.py`
- VSM 排序：`src/ir/retrieval/vsm.py`
- BM25 排序：`src/ir/retrieval/bm25.py`
- Rocchio 查詢擴展：`src/ir/ranking/rocchio.py`
- Query Parser：`src/ir/query/query_parser.py`

### 片段程式碼（Boolean：由 Infix → Postfix → Stack Evaluation）

```python
# 1) Parse: extract quoted phrases -> placeholder tokens (__PHRASE_0__)
# 2) Convert: infix tokens -> postfix (RPN) via Shunting Yard
# 3) Evaluate: stack-based evaluation where operands are Set[int] doc_ids
postfix = self._to_postfix(tokens)
result = self._evaluate_postfix(postfix, phrases, optimize)
```

### 原理整理（重點）

- **Boolean 查詢解析**：
  - 先把 `"quoted phrase"` 替換成 `__PHRASE_N__`，避免 tokenizer 因空白而把 phrase 拆散。
  - 用 **Shunting Yard** 將中序（infix）轉成 **RPN/postfix**，這樣 evaluation 只要用 stack 就能單趟完成。
  - evaluation 以集合運算為主：`AND`＝交集、`OR`＝聯集、`NOT`＝全集差集。
  - **NEAR/n 限制**：真正的 proximity 需要「原始詞項的位置信息」。目前 stack 主要存 `doc_id set`，因此多數情況會降級成 AND（已在程式內註記原因）。

- **VSM（TF‑IDF + Cosine Similarity）**：
  - 查詢向量與文件向量都是稀疏向量（dict）。
  - 先用倒排索引做 candidate generation（至少包含一個 query term 的文件），再算 cosine similarity，最後用 heap 取 Top‑k。

- **BM25**：
  - 以 TF 飽和（saturation）避免 tf 線性爆炸，以文件長度正規化避免長文偏好。
  - candidate generation 同樣先做 postings union，避免全庫掃描。

- **Rocchio**：
  - `αQ + β*centroid(D_r) - γ*centroid(D_nr)`：保留原始意圖、向 relevant centroid 靠近、遠離 non-relevant centroid。
  - 增加 **query drift**（cosine distance）檢查：當漂移過大時，縮減 expansion terms 數量以降低主題偏移風險。

- **Query Parser（Recursive Descent）**：
  - 以遞迴下降對應 grammar：`OR → AND → NOT → TERM`，自然得到 precedence。
  - 支援 **implicit AND**（兩個 term 相鄰視為 AND），符合一般搜尋引擎使用習慣。

### 驗證方式

- 已通過本次修改直接涵蓋的測試：  
  `pytest tests/test_boolean.py tests/test_vsm.py tests/test_rocchio.py`

### 下一步

- 你若希望更「像教科書」的解釋，我可以在 `docs/PROGRESS.md` 補上：
  - Boolean query 的 AST/RPN 圖示（含 precedence/associativity）
  - VSM 與 BM25 的逐步手算例子（小型 corpus）
  - Rocchio 的 drift 案例（何時會跑偏、如何調 α/β/γ）

---

## 2025-12-26：索引結構與加權教學註解補強（第 3 批）

### 目標

- 讓「索引」與「加權」兩個底層模組更容易閱讀：倒排索引、位置索引、TF‑IDF/餘弦相似度。
- 保持行為不變的前提下，補上英文註解把關鍵不變量（invariants）與時間複雜度寫清楚。

### 本次修改範圍

- 倒排索引：`src/ir/index/inverted_index.py`
- 位置索引：`src/ir/index/positional_index.py`
- 加權與相似度：`src/ir/index/term_weighting.py`

### 片段程式碼（倒排索引：postings list 的 merge 與二分搜尋）

Postings list 以 `doc_id` 排序後，可以用 merge 做 AND/OR，也可以用二分搜尋查詢某個文件中的 tf：

```python
idx = bisect_left(postings, (doc_id, 0))
if idx < len(postings) and postings[idx][0] == doc_id:
    return postings[idx][1]
```

### 片段程式碼（位置索引：proximity 的 two‑pointer）

兩個位置列表都已排序時，用 two-pointer 可把 proximity 從 O(p1*p2) 降到 O(p1+p2)：

```python
i = j = 0
while i < len(pos1) and j < len(pos2):
    if abs(pos1[i] - pos2[j]) <= max_distance:
        return True
    if pos1[i] < pos2[j]:
        i += 1
    else:
        j += 1
```

### 原理整理（重點）

- **倒排索引 *Inverted Index***：
  - 核心資料結構是 `term -> postings list`；postings 以 `doc_id` 排序是關鍵不變量，讓 AND/OR 可以用 merge 線性掃過。
  - `term_frequency(term, doc_id)` 若 postings 有序，可用二分搜尋從 O(k) 改為 O(log k)（k 為 postings 長度）。

- **位置索引 *Positional Index***：
  - 額外存 `term -> {doc_id: [positions...]}`，讓 phrase query / proximity query 成為可能。
  - phrase query 的核心是「位置對齊」：第一個詞出現在 pos，第二個詞要在 pos+1，第三個詞要在 pos+2…。
  - proximity query 在 positions 有序時，用 two-pointer 可以避免雙重迴圈的 O(p1*p2)。

- **TF‑IDF 與 Cosine Similarity**：
  - `vectorize()` 只輸出「出現過的詞」的稀疏向量（dict），避免建立巨大的 dense 向量。
  - cosine similarity 計算 dot product 時，迭代較小的向量可以減少 hash lookup 次數。

### 驗證方式

- 已通過：`pytest tests/test_inverted_index.py tests/test_positional_index.py tests/test_term_weighting.py`

---

## 2025-12-26：查詢最佳化與整合式搜尋流程補強（第 4 批）

### 目標

- 補齊「Top‑K 查詢最佳化」的關鍵概念註解（WAND 迴圈：pivot、threshold θ、skip）。
- 修正 `UnifiedSearchEngine` 在整合 BM25/VSM 時的回傳結構不一致問題，並補上流程註解，讓整合式搜尋可用且好讀。

### 本次修改範圍

- 查詢最佳化：`src/ir/retrieval/query_optimization.py`
- 整合式搜尋：`src/ir/search/unified_search.py`

### 片段程式碼（UnifiedSearch：把不同模型的 score 轉成同一種 mapping）

BM25 與 VSM 的 `scores` 形狀不同，先轉成共同格式 `{doc_id: score}` 才能寫出一致的排序/融合邏輯：

```python
if isinstance(scores, dict):
    return scores
if isinstance(scores, list) and isinstance(doc_ids, list):
    return {doc_id: score for doc_id, score in zip(doc_ids, scores)}
```

### 原理整理（重點）

- **WAND（Weak AND）**：
  - 維護一個 top‑k 的 min‑heap，heap 最小值就是 threshold θ（第 k 名分數）。
  - 依目前各 postings cursor 的 doc_id 排序，累加 term upper bounds 找到 pivot：
    - 若 pivot doc_id == min doc_id：此 doc 有機會超過 θ → 進行「精確打分」並更新 heap/θ
    - 否則：可以安全地把 pivot 前面的 term cursor 直接跳到 pivot doc_id（skip）

- **UnifiedSearch 的整合重點**：
  - `BM25Result.scores` 是「與 doc_ids 對齊的 list」，`VSMResult.scores` 是「以 doc_id 為 key 的 dict」；若直接混用會在 runtime 爆掉（`.values()/.items()` 或用 index 取 dict）。
  - 以 `_scores_to_dict()` 做一次正規化後：
    - `_execute_simple_query()` 可以一致地取分數
    - `_execute_hybrid_query()` 可以先各自 normalize，再做加權融合
  - Boolean 模式分兩類：  
    - **field boolean**（含 `title:` 等欄位前綴）→ 走 `QueryParser/QueryExecutor`，可回傳 matched_fields  
    - **content boolean**（純內容詞項 AND/OR/NOT）→ 走 `BooleanRetrieval`

### 驗證方式

- 靜態語法檢查：`python -m py_compile src/ir/retrieval/query_optimization.py src/ir/search/unified_search.py`
- 回歸測試（確保既有檢索核心不受影響）：`pytest tests/test_boolean.py tests/test_vsm.py`

---

## 2025-12-26：欄位索引（FieldIndexer）教學註解補強（第 5 批）

### 目標

- 讓「欄位/metadata 索引」更容易理解：每個欄位獨立的倒排索引、日期欄位的正規化策略，以及查詢端需要的 term normalization 規則。

### 本次修改範圍

- `src/ir/index/field_indexer.py`

### 片段程式碼（概念：每個欄位一份索引，集合運算在上層完成）

`FieldIndexer` 的核心想法是為每個欄位維護獨立的 `term -> set(doc_id)`，而 AND/OR/NOT 的集合運算通常交給上層（例如 `QueryExecutor`）：

```python
# field -> term -> set(doc_id)
field_index.add_term(doc_id, term)
```

### 原理整理（重點）

- **為什麼要「每個欄位一份索引」**：
  - 這類設計常見於「圖書館式檢索」（library-style IR）：title/author/tags/date 等欄位語意不同，不應混在同一個內容索引中打分或做 boolean。
  - 因為 postings 集合彼此獨立，查詢時可以很自然地做欄位限制（例如 `title:deep AND author:smith`）。

- **日期欄位正規化（lexicographic == chronological）**：
  - 將日期統一存成 `YYYY-MM-DD` 字串後，字典序會與時間先後一致，讓 range query 可以安全地用字串比較完成（已在程式內補上對應註解）。

- **term normalization（EXACT_FIELDS vs text fields）**：
  - `EXACT_FIELDS` 走「大小寫不敏感」的 exact match（index-time 已 lowercase），因此 query-time 必須同步 lowercase。
  - 其餘欄位走 tokenizer（可能拆詞、正規化），因此 query-time 也要以 tokenizer 對齊。

### 驗證方式

- 靜態語法檢查：`python -m py_compile src/ir/index/field_indexer.py`

---

## 2025-12-26：欄位查詢執行器（QueryExecutor）註解補強與日期欄位別名修正（第 6 批）

### 目標

- 讓 `QueryExecutor` 的「AST → set algebra」執行流程更容易讀懂（AND/OR/NOT 的集合運算、以及 matched_fields 的取得方式）。
- 修正文件/範例常用的 `date:[start TO end]` 與底層索引實際使用的 `published_date` 命名不一致問題（增加 alias 正規化）。

### 本次修改範圍

- `src/ir/query/query_executor.py`
- `tests/test_query_executor.py`

### 片段程式碼（AND：先交集小集合 + early-exit）

AND 的結果集合只會愈來愈小，因此先交集最小集合可以減少運算量，並在結果變空時提早停止：

```python
child_sets = [self._execute_node(child) for child in node.children]
child_sets.sort(key=len)
result = child_sets[0]
for child_set in child_sets[1:]:
    result &= child_set
    if not result:
        break
```

### 片段程式碼（Range：date → published_date alias）

文件中常用 `date:[...]`，但 metadata 索引實際是 `published_date`。在 executor 端做一次 alias 正規化即可相容：

```python
field_for_lookup = self._DATE_FIELD_ALIASES.get(field, field)
result = self.field_indexer.search_date_range(field_for_lookup, start, end)
```

### 原理整理（重點）

- **QueryExecutor = 把 QueryNode tree 轉成集合運算**：
  - leaf（FIELD/RANGE）回傳 `Set[int] doc_ids`
  - internal node（AND/OR/NOT）只做集合交集/聯集/差集
  - 這種設計非常直覺，且容易把「資料結構」（索引）與「邏輯結合」（query semantics）分層。

- **NOT 的 universe 假設**：
  - 目前以 `set(range(doc_count))` 當作全集，前提是 doc_id 是連續的 0..N-1（符合 `FieldIndexer.build()` 以 list position 指派 doc_id 的做法）。

- **matched_fields 的成本**：
  - `matched_fields` 目前採「事後回查」：對每個 doc、對每個 field leaf 重新查一次索引再做 membership。
  - 這對 UI/除錯很方便，但在 large result sets 可能變成主要瓶頸；程式內已補上註解說明可用 caching 或 membership-only API 優化。

### 驗證方式

- 靜態語法檢查：`python -m py_compile src/ir/query/query_executor.py`
- 新增單元測試：`pytest tests/test_query_executor.py`

---

## 2025-12-26：UnifiedSearch 欄位前綴偵測修正（第 7 批）

### 目標

- 修正 `UnifiedSearchEngine` 對「欄位查詢」的偵測規則，避免 `tags:`、`published_date:` 等欄位查詢被誤判成 content boolean 而走錯檢索模型。

### 本次修改範圍

- `src/ir/search/unified_search.py`

### 片段程式碼（從 FieldIndexer.supported_fields 動態取得欄位前綴）

以前用 hard-coded list 容易漏掉欄位；改為從 `FieldIndexer.supported_fields` 派生，再補上少數 alias：

```python
prefixes = [f"{field}:" for field in self.field_indexer.supported_fields]
prefixes.extend(["date:", "published_at:"])
```

### 原理整理（重點）

- `UnifiedSearch` 需要先決定「這個 query 要走哪條管線」：
  - **FIELD/field boolean**：`QueryParser` → `QueryExecutor`（FieldIndexer）
  - **content boolean**：`BooleanRetrieval`
  - **simple ranked**：BM25/VSM/HYBRID
- 先前的欄位前綴清單不完整時，像 `tags:(AI OR 人工智慧)` 這種 query 會被誤判，導致 parser/執行器沒被使用（功能上等同壞掉）。
- 以 `supported_fields` 動態派生前綴，可讓新增/調整欄位時不必同步修改多處 hard-coded list。

### 驗證方式

- 靜態語法檢查：`python -m py_compile src/ir/search/unified_search.py`

---

## 2025-12-26：Batch indexing doc_id 對齊修正（第 8 批）

### 目標

- 修正 batch indexing 下 `doc_id` 在「內容索引 / metadata / 欄位索引」之間可能錯位的問題，避免搜尋結果對到錯的文件。
- 讓 batch indexing 的回傳結果順序與輸入文件順序一致，方便上層以位置對齊處理。

### 本次修改範圍

- `src/ir/index/incremental_builder.py`
- `src/ir/search/unified_search.py`
- `src/ir/index/field_indexer.py`
- `tests/test_incremental_builder.py`
- `tests/test_field_indexer.py`

### 片段程式碼（Batch indexing：results 與輸入 docs 對齊）

先建立固定長度的 results，並用原始位置回填，確保 `results[i]` 對應 `docs[i]`：

```python
results = [(False, "Not processed")] * len(docs)
...
results[i] = (False, "Duplicate")
...
results[pos] = (True, f"Indexed as doc_id={doc_id}")
```

### 片段程式碼（UnifiedSearch：使用實際被指派的 doc.doc_id）

batch 中若有 duplicate/error，`doc_id` 不能用「batch 位置」推算；要用索引器回傳並寫回文件物件的 `doc_id`：

```python
for doc, (success, _) in zip(doc_buffer, results):
    if success and doc.doc_id is not None:
        self.doc_metadata[doc.doc_id] = {...}
        field_docs.append({"doc_id": doc.doc_id, ...})
```

### 片段程式碼（FieldIndexer：尊重輸入文件的顯式 doc_id）

`FieldIndexer.build()` 若輸入文件提供 `doc_id`，就用它做索引鍵，讓欄位索引可與內容索引共享同一套 doc_id 空間：

```python
explicit_doc_id = doc.get("doc_id")
doc_id = explicit_doc_id if isinstance(explicit_doc_id, int) else i
```

### 原理整理（重點）

- **doc_id 是跨模組的一致主鍵（primary key）**：
  - 內容索引（倒排索引 / BM25 / VSM）、欄位索引（FieldIndexer）、以及 UI/metadata 都必須用同一個 doc_id 才能正確對應同一篇文件。

- **為什麼 batch indexing 不能用位置推 doc_id**：
  - batch 中可能混入 duplicate/error，導致「輸入文件數」≠「實際新增的文件數」。
  - 若用 `batch index i` 推 `doc_id`，會把 metadata/欄位索引寫到錯的 doc_id → 查詢結果就會錯位。

- **FieldIndexer 的 doc_id 對齊策略**：
  - 若上層提供 `doc_id`（例如由內容索引器產生），FieldIndexer 直接用該 doc_id 建索引。
  - 若未提供（一般小型單元測試或簡單 demo），則維持舊行為：用 enumerate 產生 0..N-1。

### 驗證方式

- 靜態語法檢查：`python -m py_compile src/ir/index/field_indexer.py src/ir/index/incremental_builder.py src/ir/search/unified_search.py`
- 單元測試：
  - `pytest tests/test_field_indexer.py`
  - `pytest tests/test_incremental_builder.py`
  - `pytest tests/test_query_executor.py`

---

## 2025-12-26：補齊核心模組缺漏 docstring（第 9 批）

### 目標

- 針對多個核心模組中「容易被忽略」但仍會被讀到的函式，補齊簡潔的英文 docstring：
  - dataclass 的 `__post_init__`
  - `@property`（如 `size`/`length`）
  - `__str__` / `__repr__`
  - 小型 nested helper（如 phrase placeholder 的 replacer）

### 本次修改範圍

- `src/ir/search/unified_search.py`
- `src/ir/index/pat_tree.py`
- `src/ir/facet/facet_engine.py`
- `src/ir/summarize/static.py`
- `src/ir/cluster/doc_cluster.py`
- `src/ir/cluster/term_cluster.py`
- `src/ir/text/ner_extractor.py`
- `src/ir/retrieval/boolean.py`

### 片段程式碼（dataclass __post_init__ 的意圖補註）

dataclass 常見的陷阱是 mutable default；即使現有程式碼已正確初始化，也應用 docstring 把「為什麼要這樣做」寫清楚：

```python
def __post_init__(self) -> None:
    \"\"\"Initialize mutable default fields safely.\"\"\"
    if self.children is None:
        self.children = {}
```

### 原理整理（重點）

- `__post_init__`、`__repr__` 這些方法雖然小，但會直接影響：
  - IDE tooltip / `help()` 顯示內容
  - 除錯輸出可讀性
  - 讀者理解「物件初始化後的狀態不變量（invariants）」是否成立（例如 `tokens` 永遠不是 `None`）

### 驗證方式

- 靜態語法檢查：`python -m py_compile src/ir/search/unified_search.py src/ir/summarize/static.py src/ir/retrieval/boolean.py src/ir/cluster/doc_cluster.py src/ir/cluster/term_cluster.py src/ir/facet/facet_engine.py src/ir/index/pat_tree.py src/ir/text/ner_extractor.py`

---

## 2025-12-26：`src/ir/` 全面補齊缺漏函式 docstring（第 10 批）

### 目標

- 將 `src/ir/` 內剩餘尚未補齊的函式 docstring 全數補上（英文），把「核心 IR 程式碼」的函式 docstring 覆蓋率推到 100%。

### 本次修改範圍（重點模組）

- `src/ir/keyextract/*`：補齊 `__repr__` 等小型方法 docstring（方便除錯與輸出理解）。
- `src/ir/patterns/pat_tree.py`：補齊 `Pattern/PATNode/PATTree` 的 `__repr__`、`__init__`、以及內部遞迴 helper 的 docstring（含複雜度）。
- `src/ir/ranking/hybrid.py`：補齊 demo 用 `MockRanker` 方法 docstring（明確標出 query 參數僅為介面相容）。
- `src/ir/recommendation/*`：補齊結果 dataclass 的 `__repr__` docstring。
- `src/ir/syntax/parser.py`：補齊 `DependencyEdge/SVOTriple` 的 `__repr__`，以及 PyTorch 相容性 patch 的 nested helper docstring。
- `src/ir/topic/*`：補齊 topic info 的 `__str__` 與 model 的 `__repr__` docstring。

### 片段程式碼（nested helper：遞迴計數的複雜度註記）

像 `PATTree.get_statistics()` 的 `count_nodes()` 是遞迴掃樹，讀者若不知道它是 O(N) 很容易在 debug/印出時誤踩效能坑：

```python
def count_nodes(node):
    \"\"\"Count nodes in the subtree rooted at `node`.\"\"\"
    ...
```

### 盤點結果（本批結束）

以 AST 掃描 `src/ir/**/*.py` 的函式 docstring 缺口：

- `src/ir` functions without docstrings：`0`

### 驗證方式

- 靜態語法檢查：`python -m py_compile src/ir/keyextract/evaluator.py src/ir/patterns/pat_tree.py src/ir/syntax/parser.py src/ir/topic/bertopic_model.py src/ir/topic/lda_model.py`

---

## 2025-12-26：補齊 `tests/` 與 `scripts/` 缺漏函式 docstring（第 11 批）

### 目標

- 針對 AST 盤點仍缺漏 docstring 的部分，集中在 `tests/` 與 `scripts/`：
  - `tests/`：替大量 `test_*` 函式補上一行英文 docstring，讓測試本身變成「行為規格（spec）」。
  - `scripts/`：替 CLI / crawler / data 工具內的 `__init__`、`__post_init__`、nested helper 加上英文 docstring，讓未來維護者可以快速理解意圖與資料流。
- 將「全 repo」函式 docstring 缺口收斂到 0（以 AST 掃描驗證）。

### 本次修改範圍

- 測試（tests）
  - `tests/test_metrics.py`：補齊所有 `test_*` 方法 docstring（precision/recall、AP/MAP、DCG/nDCG 等）。
  - `tests/test_rocchio.py`：補齊 Rocchio expansion / reweight / edge cases 的 `test_*` docstring。
  - `tests/test_term_weighting.py`、`tests/test_vsm.py`、`tests/test_clustering.py`：補齊核心模型/工具的 `test_*` docstring。
  - `tests/test_incremental_builder.py`：補齊 stub tokenizer methods 的 docstring（避免教學閱讀斷層）。
- 腳本（scripts）
  - 索引與檢索工具：`scripts/build_indexes_from_preprocessed.py`、`scripts/unified_retrieval.py`
  - 資料處理：`scripts/preprocess_news.py`、`scripts/data/analyze_dataset.py`
  - 爬蟲與啟動器：`scripts/run_crawlers.py`、`scripts/crawlers/*`（CNA/CTI/FTV/NextApple）
  - 測試工具：`scripts/comprehensive_test.py`

### 片段程式碼（測試 docstring = 行為規格）

把每個測試的意圖寫成一行 docstring，能在 IDE tooltip / `pytest -q` 失敗訊息中快速知道「這個測試在保證什麼」：

```python
def test_precision(self, metrics, sample_retrieved, sample_relevant):
    \"\"\"Compute precision as |retrieved ∩ relevant| / |retrieved|.\"\"\"
    ...
```

### 片段程式碼（nested helper 的 docstring：工具腳本可讀性）

像 `run_crawlers.py` 的 concurrent 模式是 nested coroutine，沒有 docstring 很難從結構一眼看懂流程：

```python
@defer.inlineCallbacks
def crawl():
    \"\"\"Run selected spiders with CrawlerRunner and stop the reactor when done.\"\"\"
    ...
```

### 盤點結果（本批結束）

以 AST 掃描全專案（`src/` + `tests/` + `scripts/`）函式 docstring 缺口：

- 變更前：`112`（`tests: 93`、`scripts: 19`）
- 變更後：`0`

### 驗證方式

- 靜態語法檢查：`python -m py_compile tests/test_metrics.py tests/test_rocchio.py tests/test_term_weighting.py tests/test_vsm.py tests/test_clustering.py scripts/build_indexes_from_preprocessed.py scripts/unified_retrieval.py scripts/run_crawlers.py scripts/preprocess_news.py scripts/data/analyze_dataset.py scripts/comprehensive_test.py scripts/crawlers/cna_spider_simple.py scripts/crawlers/cti_spider.py scripts/crawlers/ftv_spider.py scripts/crawlers/nextapple_spider.py`
- 測試子集：`pytest tests/test_metrics.py tests/test_rocchio.py tests/test_term_weighting.py tests/test_vsm.py tests/test_clustering.py`

---

## 2025-12-26：核心檢索模組「教科書式」行內註解（第 12 批）

### 目標

- 你要的是「程式碼逐行註解」，讓閱讀起來像教科書：每個關鍵步驟都能直接在程式碼旁理解其 **IR 原理 / 資料結構 / 演算法流程**。
- 依專案規範：**程式碼內註解與 docstring 以英文為主**；本檔案仍以繁體中文整理脈絡與重點。
- 本批只做「註解/可讀性」提升，不改變行為（no behavior change）。

### 本次修改範圍（核心模組）

- Boolean 查詢解析與執行：`src/ir/retrieval/boolean.py`
  - 補強 parse → postfix(RPN) → stack evaluate 的逐步註解
  - 清楚標註 wildcard / field / date range / phrase / NOT universe 的假設與限制（例如 NEAR 的 positional 限制）
- Query Parser / Executor：`src/ir/query/query_parser.py`、`src/ir/query/query_executor.py`
  - Tokenization 的歧義處理（`FIELD:` 後的 token 強制視為 `VALUE`）
  - Recursive descent precedence（NOT > AND > OR）與 implicit AND 的教學式註解
  - AND 交集先小集合、NOT universe 的假設（doc_id contiguous）
- 排序模型：`src/ir/retrieval/vsm.py`、`src/ir/retrieval/bm25.py`
  - SMART weighting（ltc/lnc）與候選集合（candidate generation）流程註解
  - BM25 的 index-time / query-time 步驟註解（tf saturation + length normalization）
- Rocchio 擴展：`src/ir/ranking/rocchio.py`
  - Rocchio 三段式公式（αQ + βDr - γDnr）逐步註解
  - pseudo feedback 的切分方式與 normalize 的意義

### 片段程式碼（Boolean：pipeline 的逐步註解）

以下示意「查詢字串」如何依序被拆解與執行（實際程式碼已在模組內加上更細的行內註解）：

```python
# 1) Parse query string -> token stream (+ phrase table)
parsed = self._parse_query(query_str)

# 2) Execute boolean logic -> set(doc_id)
doc_ids = self._execute_query(parsed, optimize)

# 3) Optional: rank results (boolean set retrieval has no native ranking)
if rank_results:
    scores = self._rank_results(query_str, doc_ids)
```

### 下一步建議（你若同意我就繼續做）

- 第 13 批：索引與整合式搜尋的「教科書式行內註解」
  - `src/ir/index/inverted_index.py`、`src/ir/index/positional_index.py`、`src/ir/index/term_weighting.py`
  - `src/ir/index/field_indexer.py`、`src/ir/search/unified_search.py`、`src/ir/index/incremental_builder.py`

### 驗證方式

- 靜態語法檢查：`python -m py_compile src/ir/retrieval/boolean.py src/ir/query/query_parser.py src/ir/query/query_executor.py src/ir/retrieval/vsm.py src/ir/retrieval/bm25.py src/ir/ranking/rocchio.py`
- 測試子集：`pytest tests/test_boolean.py tests/test_query_executor.py tests/test_rocchio.py tests/test_vsm.py`

---

## 2025-12-26：Text/Tokenizer 與其他 Retrieval 模型的教科書式行內註解（第 13 批）

### 目標

- 先把「文本處理（tokenizer/stopwords）」補成教科書式逐步註解，讓你能清楚理解：
  - 為什麼要做 tokenizer normalization（空白、大小寫、字典）
  - 為什麼 IR 常用 stopwords（以及可能的副作用）
  - CKIP/Jieba 的取捨與 fallback 策略
- 再補齊 retrieval 其他模型的教科書式行內註解：
  - wildcard / fuzzy 的 expansion 原理與「防爆炸」限制
  - LM retrieval 的 smoothing 與 log-likelihood ranking
  - BIM 的 binary 表示法與 RSJ 權重（relevance feedback）

### 本次修改範圍

- Text / Tokenizer
  - `src/ir/text/chinese_tokenizer.py`：補強 engine auto-selection、backend init、batch、cache 等行內註解。
  - `src/ir/text/ckip_tokenizer.py`：補強 singleton、post-processing（min_length/stopwords/digits）、fallback 的行內註解。
  - `src/ir/text/ckip_tokenizer_optimized.py`：補強 thread pool/環境變數/torch threading 的教學式註解（強調其作用與限制）。
  - `src/ir/text/stopwords.py`：補強 stopwords 在 IR 的角色、case normalization、filter 的成本與使用時機。
- Retrieval 其他模型
  - `src/ir/retrieval/wildcard.py`：補強 wildcard → regex、fullmatch vs substring、max_expansions 的行內註解。
  - `src/ir/retrieval/fuzzy.py`：補強 Levenshtein DP table 意義、fuzzy scan 的複雜度與常見加速策略（BK-tree 等）註解。
  - `src/ir/retrieval/language_model_retrieval.py`：補強 index-time（doc model / collection model）、smoothing 選擇、log-likelihood 的行內註解；並修正 absolute discounting smoothing 的 `doc_id` 參數遺漏問題（避免 `NameError`）。
  - `src/ir/retrieval/bim.py`：補強 binary 表示法、candidate generation、RSJ smoothing 常數的行內註解。

### 片段程式碼（LM：smoothing 分流 + 絕對折扣修正）

LM retrieval 的核心是先算 `tf`、`|D|`、`P(w|C)`，再依 smoothing 方法套公式：

```python
if self.smoothing == 'absolute':
    return self._absolute_discounting_smoothing(tf, doc_length, p_collection, doc_id)
```

### 驗證方式

- 靜態語法檢查：`python -m py_compile src/ir/text/chinese_tokenizer.py src/ir/text/ckip_tokenizer.py src/ir/text/ckip_tokenizer_optimized.py src/ir/text/stopwords.py src/ir/retrieval/wildcard.py src/ir/retrieval/fuzzy.py src/ir/retrieval/language_model_retrieval.py src/ir/retrieval/bim.py`
- 測試子集（確保 wildcard 整合沒壞）：`pytest tests/test_boolean.py`

---

## 2025-12-26：索引工具模組教科書式行內註解（Compression / Dedup / Reader）（第 14 批）

### 目標

- 補強索引管線中「常被忽略但很關鍵」的三個工具模組，讓你能逐行對照理解：
  - Index compression：為什麼 postings list 先 gap encoding、再用可變長度碼（VByte/Gamma/Delta）能大幅省空間
  - Deduplication：MD5（exact）vs SimHash（near-duplicate）的原理與複雜度取捨
  - Document reader：JSONL / PostgreSQL 兩種資料來源的讀取策略與「可容錯」的資料正規化

### 本次修改範圍

- `src/ir/index/compression.py`
  - 補強 VByte 的 base-128 chunk/continuation bit 解碼心智模型（multiplier 的意義）
  - **修正** Elias Gamma 的 bitstream 格式與 decode pointer 前進邏輯，讓 encode/decode 可正確 round-trip
  - 更新 Delta 範例與註解，使其符合標準定義（先 gamma 編碼 bit-length，再附上去掉 leading 1 的 offset bits）
- `src/ir/index/deduplication.py`
  - 補強 SimHash 的「bit vote 向量」直覺（為什麼相似文本會有小 Hamming distance）
  - 補強 fuzzy 去重的複雜度與常見加速方向（banding/BK-tree/LSH buckets）
- `src/ir/index/doc_reader.py`
  - 補強 NewsDocument 欄位/主鍵（doc_id）對齊觀念
  - 補強 JSONL 容錯讀取（per-line error isolation）
  - 補強 PostgreSQL server-side cursor + `fetchmany()` 的批次讀取意義（記憶體/IO 取捨）

### 片段程式碼（Gamma：標準表示法 `0^L || binary(n)`）

Gamma 編碼最常見的教科書定義是：前置 L 個 0，再接上 n 的 binary（含 leading 1）。這樣 decode 才能靠「數 0 的數量」知道要再讀多少 bits：

```python
binary = bin(num)[2:]           # e.g., 13 -> "1101"
length = len(binary) - 1        # L = floor(log2(n))
return ("0" * length) + binary  # "0001101"
```

### 片段程式碼（SimHash：bit vote）

SimHash 的核心是把每個 token hash 轉成 bit pattern，對每個 bit 做「+1 / -1 投票」，最後看正負號決定指紋：

```python
if token_hash & (1 << i):
    v[i] += 1
else:
    v[i] -= 1
```

### 原理整理（重點）

- **Compression（壓縮）**
  - postings list 是遞增 doc_id；先做 **gap encoding** 會把大數字轉成小 gap（更容易被可變長度碼壓縮）。
  - VByte 的 decode 用 `multiplier = 1, 128, 128^2...` 逐步重建整數，是理解 base-128 little-endian 的關鍵。
  - Gamma/Delta 這類 bit-level 編碼要特別注意：**bitstream 的自描述性**（self-delimiting）必須成立，decode pointer 才能前進。

- **Deduplication（去重）**
  - Exact：MD5 像 checksum，快且 O(1) lookup；但無法抓「文字稍微改過」的近似重複。
  - Fuzzy：SimHash 用 Hamming distance 衡量「指紋差多少 bit」；但 naive 比對是 O(M)（M=已收錄文件數），大規模時需加速（banding/LSH）。

- **Document reader（資料讀取）**
  - JSONL 讀取採逐行 parse + try/except：單行壞資料不會讓整個檔案失敗。
  - PostgreSQL 用 server-side cursor + `fetchmany(batch_size)`：避免一次把全部結果拉進記憶體。

### 驗證方式

- 靜態語法檢查：`python -m py_compile src/ir/index/compression.py src/ir/index/deduplication.py src/ir/index/doc_reader.py`
- 建議快速 sanity（手動）：在 Python REPL 內做 `GammaEncoder().decode(GammaEncoder().encode([1, 5, 13]))` 應回到原序列

---

## 2025-12-26：IR 評估指標（Metrics）教科書式行內註解（第 15 批）

### 目標

- 把 `src/ir/eval/metrics.py` 補成更「教科書式」的逐步註解，讓你能一邊看程式一邊理解：
  - Binary relevance（Precision/Recall/F1）
  - Ranked metrics（AP/MAP/MRR）
  - Graded relevance（DCG/nDCG/ERR）
  - 不完整標註下的指標（Bpref）
  - Run-level 聚合（evaluate_query / evaluate_run）

### 本次修改範圍

- `src/ir/eval/metrics.py`
  - 在每個指標的「關鍵計算步驟」補上英文行內註解（為什麼這樣做、複雜度、常見定義/慣例）。
  - 補強 `ndcg_at_k()` 的 IDCG 假設（基於 `relevance_scores` 的 judged docs 集合）與「unjudged = 0」的評估直覺。
  - 補強 `evaluate_run()` 聚合流程註解（為什麼要先 collect per-query 再平均、以及 MAP/MRR/GMAP 為什麼再顯式重算）。

### 片段程式碼（AP：只在 relevant 命中時累加 P@rank）

Average Precision 會「只在命中 relevant 文件的 rank」累加 precision，因此會同時獎勵「命中」與「命中越早越好」：

```python
for rank, doc_id in enumerate(retrieved, start=1):
    if doc_id in relevant:
        num_relevant_seen += 1
        precision_sum += num_relevant_seen / rank
```

### 片段程式碼（nDCG：用 IDCG 正規化）

nDCG 的核心是用理想排序（IDCG）做 normalization，把 DCG 映射到 0..1：

```python
dcg = self.dcg_at_k(retrieved, relevance_scores, k)
ideal = sorted(relevance_scores.keys(), key=lambda d: relevance_scores[d], reverse=True)
idcg = self.dcg_at_k(ideal, relevance_scores, k)
return 0.0 if idcg == 0 else dcg / idcg
```

### 原理整理（重點）

- **Binary relevance metrics**（Precision/Recall/F1）：
  - 只需要 `relevant: Set[int]`；集合 membership 平均 O(1)，整體計算 O(k)。
  - 空集合時的慣例（P@0/Recall when |Relevant|=0）通常回 0.0，避免除以 0。

- **AP/MAP**：
  - AP 會把「未被取回的 relevant 文件」視為 0 貢獻（除以 |Relevant|），因此能同時衡量排序品質與召回不足。
  - MAP 是把有 qrels 的 query 平均；沒 qrels 的 query 通常跳過（避免不公平拉低/拉高）。

- **DCG/nDCG/ERR（graded relevance）**：
  - DCG 用 log discount 模擬「使用者越往後越不看」。
  - nDCG 用理想排序做 normalization，便於跨 query 比較。
  - ERR 用 cascade user model，把「滿意就停」的行為寫進公式（prob_continue 的累乘）。

- **Bpref**：
  - 針對 judged 不完整的情境，避免把 unjudged 當 non-relevant。
  - 核心是：對每個 relevant 文件 r，只看它前面有多少 judged non-relevant（並做 cap）。

### 驗證方式

- 靜態語法檢查：`python -m py_compile src/ir/eval/metrics.py`
- 單元測試：`pytest tests/test_metrics.py`
