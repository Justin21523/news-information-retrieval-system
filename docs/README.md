# 資訊檢索系統專案文件

## 專案簡介

本專案為 **LIS5033 - 自動分類與索引** 課程的資訊檢索系統實作，基於教科書《Introduction to Information Retrieval》（Manning, Raghavan, Schütze）實現傳統 IR 技術，包括索引建構、檢索模型、評估指標、查詢擴展、分群演算法與自動摘要等核心功能。

**課程名稱**：自動分類與索引 *Automatic Classification and Indexing*
**開課單位**：國立臺灣大學圖書資訊學系
**參考教材**：*Introduction to Information Retrieval* (Manning, Raghavan, Schütze)

---

## 文件導覽

### 📚 學習資源

- **[實作指南](guides/IMPLEMENTATION.md)** - 各模組詳細實作說明
- **[CSoundex 指南](guides/CSOUNDEX.md)** - 中文諧音編碼詳細文件
- **[變更紀錄](CHANGELOG.md)** - 專案開發歷程記錄

### 📝 考試準備

- **[期中考大綱](exams/midterm/OUTLINE.md)** - 結構化答題大綱
- **[期中考完整答案](exams/midterm/DRAFT.md)** - 完整擬答（含雙語術語）
- **[圖表資源](exams/midterm/FIGS/)** - SVG 格式示意圖

### 📊 作業報告

- **[報告範本](hw/template/REPORT_TEMPLATE.md)** - 標準化作業報告格式
- **各次作業**：`hw/HW01/`, `hw/HW02/`, ... （依課程進度建立）

### 🎯 期末專案

- **[專案提案](project/PROPOSAL.md)** - 學術搜尋引擎提案
- **[專案報告](project/REPORT.md)** - 完整系統實作報告

---

## 核心模組概覽

### M1: 布林檢索 *Boolean Retrieval*

**功能**：倒排索引建構、布林查詢處理（AND/OR/NOT）、詞組查詢
**模組位置**：
- `src/ir/index/inverted_index.py` - 倒排索引
- `src/ir/retrieval/boolean.py` - 布林檢索引擎
- `scripts/boolean_search.py` - 命令列工具

**關鍵技術**：
- 倒排索引 *Inverted Index*：詞彙 → 文件列表映射
- 詞組查詢 *Phrase Query*：位置索引支援
- 查詢最佳化：依文件頻率排序操作順序

---

### M2: 字典與容錯檢索 *Dictionary & Tolerant Retrieval*

**功能**：萬用字元查詢、拼寫校正、中文諧音匹配
**模組位置**：
- `src/ir/text/csoundex.py` - 中文諧音編碼
- `src/ir/index/dictionary.py` - 字典結構
- `scripts/csoundex_encode.py` - CSoundex CLI

**關鍵技術**：
- **CSoundex**：中文 Soundex 編碼（同音字匹配）
- 編輯距離 *Edit Distance*：拼寫校正
- 萬用字元索引 *Wildcard Index*：前綴/後綴樹

**詳細說明**：參見 [CSoundex 指南](guides/CSOUNDEX.md)

---

### M3: 向量空間模型 *Vector Space Model*

**功能**：TF-IDF 權重計算、餘弦相似度排序
**模組位置**：
- `src/ir/retrieval/vsm.py` - VSM 實作
- `src/ir/index/term_weighting.py` - TF-IDF 計算

**關鍵技術**：
- **TF-IDF**：詞頻-逆文件頻率加權
- **餘弦相似度**：向量夾角度量
- **Top-K 檢索**：堆積結構加速排序

**複雜度**：
- 索引建構：O(T)，T 為總詞數
- Top-K 檢索：O(k log k)

---

### M4: 評估指標 *Evaluation Metrics*

**功能**：精確率、召回率、F-measure、MAP、nDCG
**模組位置**：
- `src/ir/eval/metrics.py` - 評估指標實作
- `scripts/eval_run.py` - 批次評估工具

**實作指標**：

| 指標 | 中文名稱 | 適用場景 |
|------|---------|---------|
| Precision | 精確率 | 搜尋結果相關性 |
| Recall | 召回率 | 覆蓋率評估 |
| F-measure | F 分數 | 平衡指標 |
| AP | 平均精確率 | 單一查詢效能 |
| MAP | 平均平均精確率 | 整體系統效能 |
| nDCG | 標準化折損累積增益 | 考慮排序位置 |

**使用範例**：
```bash
python scripts/eval_run.py \
  --results results.json \
  --qrels qrels.txt \
  --metrics MAP,nDCG@10,P@5
```

---

### M5: 查詢擴展 *Query Expansion*

**功能**：Rocchio 演算法、相關回饋、擬相關回饋
**模組位置**：
- `src/ir/ranking/rocchio.py` - Rocchio 實作
- `scripts/expand_query.py` - 查詢擴展工具

**演算法**：
```
Q_modified = α × Q_original + β × D_relevant - γ × D_irrelevant
```

**模式**：
1. **明確回饋** *Explicit Feedback*：使用者標記相關文件
2. **擬相關回饋** *Pseudo-Relevance Feedback*：假設前 K 筆為相關

**使用範例**：
```bash
# 擬相關回饋（假設前 10 筆相關）
python scripts/expand_query.py \
  --query "資訊檢索" \
  --mode pseudo \
  --topk 10 \
  --alpha 1.0 --beta 0.75 --gamma 0.0

# 明確回饋
python scripts/expand_query.py \
  --query "機器學習" \
  --mode explicit \
  --relevant rel_docs.txt
```

---

### M6: 分群演算法 *Clustering*

**功能**：階層式分群、平面分群、詞彙分群
**模組位置**：
- `src/ir/cluster/doc_cluster.py` - 文件分群
- `src/ir/cluster/term_cluster.py` - 詞彙分群

**演算法實作**：

**階層式分群** *Hierarchical Clustering*：
- Complete-link（完全連結）
- Single-link（單一連結）
- Group-average（群組平均）

**平面分群** *Flat Clustering*：
- K-means
- Star clustering（星狀分群）

**詞彙分群**：
- 基於編輯距離的字串分群
- 基於共現矩陣的語意分群

---

### M7: 自動摘要 *Automatic Summarization*

**功能**：靜態摘要、動態摘要（KWIC）
**模組位置**：
- `src/ir/summarize/static.py` - 靜態摘要
- `src/ir/summarize/dynamic.py` - 動態摘要

**摘要類型**：

1. **靜態摘要** *Static Summary*：
   - Lead-K：取前 K 句
   - 關鍵句萃取：TF-IDF 排序
   - 位置加權：標題、首段加權

2. **動態摘要** *Dynamic Summary*：
   - KWIC (KeyWord In Context)：關鍵詞前後文視窗
   - 快取機制：避免重複計算
   - 多查詢詞高亮：不同詞彙分色顯示

---

## 開發流程

### 1️⃣ 實作前準備

```bash
# 閱讀相關文件
cat docs/guides/IMPLEMENTATION.md
cat docs/hw/HW01/REPORT.md  # 查看作業需求

# 檢查設定檔
cat configs/csoundex.yaml
```

### 2️⃣ 開發階段

```bash
# 在 src/ 撰寫模組（含英文註解）
vim src/ir/text/csoundex.py

# 建立 CLI 工具（支援 --help）
vim scripts/csoundex_encode.py

# 撰寫測試（至少 3 cases）
vim tests/test_csoundex.py

# 使用小型測試資料
ls datasets/mini/
```

### 3️⃣ 完成後作業

```bash
# 更新文件（更新既有文件，不建立新文件）
vim docs/guides/CSOUNDEX.md

# 記錄變更
vim docs/CHANGELOG.md

# 執行測試
pytest tests/ -v

# 確認 CLI 可用
python scripts/csoundex_encode.py --help
```

---

## 測試與品質控制

### 單元測試

```bash
# 執行全部測試
pytest tests/

# 執行特定測試檔案
pytest tests/test_csoundex.py -v

# 顯示覆蓋率
pytest tests/ --cov=src/ir --cov-report=html
```

### 測試標準

每個模組至少包含三類測試：
1. **正常情況**：標準輸入輸出
2. **邊界情況**：空字串、單一字元、極大輸入
3. **異常情況**：混合文字、特殊符號、編碼問題

### 程式碼品質

- ✅ 所有函式包含英文 docstring
- ✅ 包含複雜度分析（Time/Space）
- ✅ 非顯而易見的邏輯需註解
- ✅ 可直接執行的範例程式碼

---

## 報告撰寫規範

### 期中考答題格式

**檔案位置**：`docs/exams/midterm/`

**必要檔案**：
- `OUTLINE.md` - 結構化大綱（重點條列）
- `DRAFT.md` - 完整擬答（中文，雙語術語）
- `FIGS/` - SVG 格式圖表

**答題架構**（四段式）：
1. **觀點陳述**：明確表達立場
2. **例證支持**：具體案例說明
3. **反例討論**：考慮對立觀點
4. **小結總結**：整合論述

**重點主題**：
- 嵌入式搜尋 vs. 通用搜尋引擎（Facebook/Blog search vs. General search）
- 搜尋 vs. 瀏覽（Searching vs. Browsing）
- 課程模組自選主題（IR models, evaluation, clustering 等）

### 作業報告格式

**範本位置**：`docs/hw/template/REPORT_TEMPLATE.md`

**標準章節**：
1. 題目與目標（含雙語術語）
2. 理論背景（公式、演算法）
3. 方法設計（架構圖、流程圖）
4. 實作細節（模組、複雜度、參數）
5. 實驗設計（資料集、指標、基準）
6. 結果與分析（表格、圖表 SVG、錯誤分析）
7. 限制與未來工作
8. 重現步驟（完整指令）

### 期末專案要求

**核心功能**：
- ✅ 全文索引（位置索引支援）
- ✅ VSM 排序（TF-IDF + 餘弦相似度）
- ✅ 多種查詢（布林 + 詞組 + 萬用字元）
- ✅ 欄位搜尋（標題/作者/年份）
- ✅ 分面瀏覽或分群

**繳交文件**：
- `docs/project/PROPOSAL.md` - 專案提案
- `docs/project/REPORT.md` - 完整報告
- `scripts/run_demo.sh` - 可執行展示

---

## 常用指令速查

### CSoundex（中文諧音編碼）

```bash
# 編碼文字
python scripts/csoundex_encode.py --text "三聚氰胺"

# 從檔案編碼
python scripts/csoundex_encode.py --file input.txt --output encoded.txt

# 批次處理（管線）
cat names.txt | python scripts/csoundex_encode.py --stdin
```

### 布林檢索

```bash
# 簡單查詢
python scripts/boolean_search.py --query "information AND retrieval"

# 詞組查詢
python scripts/boolean_search.py --query '"vector space model"'

# 萬用字元
python scripts/boolean_search.py --query "retrie*"
```

### 向量空間檢索

```bash
# TF-IDF 排序
python scripts/vsm_search.py --query "machine learning" --topk 10

# 指定權重方案
python scripts/vsm_search.py --query "IR" --tf ltc --idf t
```

### 評估指標

```bash
# 計算 MAP 與 nDCG
python scripts/eval_run.py \
  --results results.json \
  --qrels qrels.txt \
  --metrics MAP,nDCG@10,P@5,Recall@10

# 輸出詳細報告
python scripts/eval_run.py \
  --results results.json \
  --qrels qrels.txt \
  --output eval_report.txt \
  --verbose
```

### 查詢擴展

```bash
# 擬相關回饋
python scripts/expand_query.py \
  --query "深度學習" \
  --mode pseudo \
  --topk 10

# 明確回饋
python scripts/expand_query.py \
  --query "自然語言處理" \
  --mode explicit \
  --relevant rel_docs.txt \
  --irrelevant irrel_docs.txt
```

---

## 資源連結

### 課程教材

- [課程投影片](https://www.csie.ntu.edu.tw/~sdlin/aci/)
- [教科書線上版](https://nlp.stanford.edu/IR-book/)
- [評估資料集](http://ir.dcs.gla.ac.uk/resources/test_collections/)

### 開發工具

- [Pytest 文件](https://docs.pytest.org/)
- [Argparse 教學](https://docs.python.org/3/library/argparse.html)
- [Pandoc 轉檔](https://pandoc.org/MANUAL.html)

### 參考實作

- [NLTK IR Package](https://www.nltk.org/)
- [Gensim](https://radimrehurek.com/gensim/)
- [Whoosh (Pure Python)](https://whoosh.readthedocs.io/)

---

## 問題排解

### Q1: 如何新增文件？

**❌ 錯誤做法**：建立 `file_v2.md`, `file_copy.md`
**✅ 正確做法**：更新既有檔案，並在 `CHANGELOG.md` 記錄變更

### Q2: CSoundex 如何處理多音字？

參見 [CSoundex 指南](guides/CSOUNDEX.md) 的「多音字處理策略」章節。
簡要說明：使用詞頻最高的讀音，或提供多編碼輸出模式。

### Q3: 如何輸出 DOCX/PDF 格式報告？

```bash
# 使用 pandoc（需安裝）
pandoc docs/exams/midterm/DRAFT.md \
  -o ACI2025MidTerm<學號>.docx \
  --reference-doc=template.docx

# 轉 PDF
pandoc ACI2025MidTerm<學號>.docx \
  -o ACI2025MidTerm<學號>.pdf

# 使用專案工具
python scripts/format_to_docx.py \
  --input docs/exams/midterm/DRAFT.md \
  --output ACI2025MidTerm<學號>.docx
```

### Q4: 測試失敗如何除錯？

```bash
# 詳細輸出
pytest tests/test_csoundex.py -v -s

# 只執行失敗的測試
pytest --lf

# 進入互動除錯
pytest --pdb
```

### Q5: 如何檢查程式碼覆蓋率？

```bash
# 產生 HTML 報告
pytest tests/ --cov=src/ir --cov-report=html

# 開啟報告
xdg-open htmlcov/index.html  # Linux
open htmlcov/index.html      # macOS
```

---

## 貢獻指引

本專案為課程作業專案，開發政策如下：

### 檔案管理政策（嚴格執行）

- ❌ **禁止建立**新的重複文件（相同主題）
- ❌ **禁止保留**舊版本（`_v2`, `_copy`, `_final` 等）
- ❌ **禁止建立**暫存目錄（`tmp/`, `playground/`）
- ✅ **必須更新**既有文件（不建立新文件）
- ✅ **必須記錄** `docs/CHANGELOG.md`（每次變更）
- ✅ **必須整合**相同主題內容為單一文件

### Git 工作流程

```bash
# 建立功能分支
git checkout -b feature/csoundex-implementation

# 提交變更（遵循 Conventional Commits）
git add src/ir/text/csoundex.py
git commit -m "feat(csoundex): implement Chinese phonetic encoding"

# 更新 CHANGELOG
git add docs/CHANGELOG.md
git commit -m "docs: update changelog for csoundex module"
```

### Commit 訊息格式（英文）

```
<type>(<scope>): <subject>

[optional body]
```

**類型**：
- `feat`: 新功能
- `fix`: 錯誤修正
- `docs`: 文件更新
- `test`: 測試新增/修改
- `refactor`: 重構
- `perf`: 效能改善

---

## 聯絡資訊

**課程**：LIS5033 自動分類與索引
**學期**：2024-2025 學年度
**授課教師**：[教師姓名]

**專案維護**：[您的姓名/學號]
**最後更新**：2025-11-12

---

## 授權聲明

本專案為國立臺灣大學課程作業，僅供學習研究使用。
程式碼採用 MIT License，文件採用 CC BY-NC-SA 4.0。

**MIT License** - 程式碼
**CC BY-NC-SA 4.0** - 文件與報告

---

**📌 快速開始**：參見 [實作指南](guides/IMPLEMENTATION.md)
**📌 CSoundex 教學**：參見 [CSoundex 指南](guides/CSOUNDEX.md)
**📌 變更歷程**：參見 [變更紀錄](CHANGELOG.md)
