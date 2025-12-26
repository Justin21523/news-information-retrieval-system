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
