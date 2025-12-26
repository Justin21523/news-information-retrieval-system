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
