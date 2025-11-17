# 專案開發路線圖 *Project Roadmap*

**課程**：LIS5033 自動分類與索引
**專案**：資訊檢索系統實作
**規劃日期**：2025-11-12

---

## 📊 目前進度總覽

### ✅ 已完成項目

- [x] **專案文檔系統**（v0.1.0）
  - 完整的中文文檔（60,000+ 字）
  - 期中考準備資料（大綱 + 完整擬答）
  - 實作指南與範本
  - CSoundex 詳細設計文件

### ⏳ 進行中項目

- [ ] **專案基礎架構**（當前階段）
- [ ] **核心模組實作**（後續階段）

### 📅 預計完成時間軸

```
Week 1-2  (已完成)  ✅ 文檔系統建立
Week 3-4  (當前)    🔄 專案架構 + CSoundex 實作
Week 5-6            📝 布林檢索 + VSM
Week 7-8            📝 評估指標 + 查詢擴展
Week 9-10           📝 分群 + 摘要
Week 11-14          📝 期中考準備
Week 15-18          🎯 期末專案整合
```

---

## 🎯 Phase 1: 專案基礎架構（Week 3，預計 2-3 天）

### 目標
建立完整的專案結構、依賴管理、設定檔案系統。

### 1.1 目錄結構建立

**需要建立的目錄**：
```
information-retrieval/
├── src/
│   └── ir/
│       ├── __init__.py
│       ├── text/
│       │   ├── __init__.py
│       │   └── csoundex.py          # CSoundex 核心
│       ├── index/
│       │   ├── __init__.py
│       │   ├── inverted_index.py    # 倒排索引
│       │   └── positional_index.py  # 位置索引
│       ├── retrieval/
│       │   ├── __init__.py
│       │   ├── boolean.py           # 布林檢索
│       │   └── vsm.py               # 向量空間模型
│       ├── eval/
│       │   ├── __init__.py
│       │   └── metrics.py           # 評估指標
│       ├── ranking/
│       │   ├── __init__.py
│       │   └── rocchio.py           # Rocchio 演算法
│       ├── cluster/
│       │   ├── __init__.py
│       │   ├── term_cluster.py      # 詞彙分群
│       │   └── doc_cluster.py       # 文件分群
│       └── summarize/
│           ├── __init__.py
│           ├── static.py            # 靜態摘要
│           └── dynamic.py           # 動態摘要（KWIC）
├── scripts/
│   ├── csoundex_encode.py          # CSoundex CLI
│   ├── boolean_search.py           # 布林檢索 CLI
│   ├── vsm_search.py               # VSM 檢索 CLI
│   ├── eval_run.py                 # 評估執行 CLI
│   ├── expand_query.py             # 查詢擴展 CLI
│   └── format_to_docx.py           # 報告匯出工具
├── tests/
│   ├── __init__.py
│   ├── test_csoundex.py
│   ├── test_inverted_index.py
│   ├── test_boolean.py
│   ├── test_vsm.py
│   ├── test_metrics.py
│   ├── test_rocchio.py
│   ├── test_cluster.py
│   └── test_summarize.py
├── configs/
│   ├── csoundex.yaml               # CSoundex 配置
│   └── logging.yaml                # 日誌配置
├── datasets/
│   ├── mini/                       # 小型測試資料
│   │   ├── documents.json
│   │   └── queries.txt
│   └── lexicon/
│       └── basic_pinyin.tsv        # 拼音字典
├── requirements.txt                # Python 依賴
├── setup.py                        # 套件安裝設定
├── pytest.ini                      # pytest 設定
├── .gitignore                      # Git 忽略清單
└── README.md                       # 專案簡介（英文）
```

### 1.2 requirements.txt

建立依賴清單：

```txt
# Core dependencies
pypinyin==0.49.0          # 中文拼音轉換
jieba==0.42.1             # 中文分詞
numpy==1.24.3             # 數值計算
scipy==1.10.1             # 科學計算

# Testing
pytest==7.4.0             # 測試框架
pytest-cov==4.1.0         # 測試覆蓋率
pytest-mock==3.11.1       # Mock 工具

# Documentation
pandoc==2.3               # 文件轉換（可選）
python-docx==0.8.11       # DOCX 匯出（可選）

# Development tools
pylint==2.17.5            # 程式碼檢查
black==23.7.0             # 程式碼格式化
mypy==1.5.1               # 型別檢查

# Data processing
pyyaml==6.0.1             # YAML 配置
tqdm==4.66.1              # 進度條
```

### 1.3 設定檔案

**configs/csoundex.yaml**：
```yaml
# CSoundex 配置檔

# 聲母分組
initial_groups:
  0: []                    # 零聲母
  1: [b, p]                # 雙唇音
  2: [f]                   # 唇齒音
  3: [m]                   # 雙唇鼻音
  4: [d, t]                # 舌尖中音
  5: [n, l]                # 舌尖鼻音與邊音
  6: [g, k, h]             # 舌根音
  7: [j, q, x]             # 舌面音
  8: [zh, ch, sh, r]       # 舌尖後音
  9: [z, c, s]             # 舌尖前音

# 韻母分組
final_groups:
  0: []                    # 零韻母
  1: [a, ia, ua]           # 主元音 a
  2: [o, uo]               # 主元音 o
  3: [e, ie, ue, ve]       # 主元音 e
  4: [i]                   # 元音 i
  5: [u]                   # 元音 u
  6: [v, u:]               # 元音 ü
  7: [ai, ei, ui, uai]     # 複韻母（韻尾 i）
  8: [ao, ou, iu, iao]     # 複韻母（韻尾 u）
  9: [an, en, in, un, vn, ang, eng, ing, ong, ian, uan, van, iang, uang, iong, er]  # 鼻韻母

# 拼音字典路徑
lexicon_path: datasets/lexicon/basic_pinyin.tsv

# 多音字處理策略
polyphone_strategy: most_common  # 選項: most_common, context_aware, all_variants

# 預設編碼模式
default_mode: standard           # 選項: standard, extended, loose

# 預設是否包含聲調
default_include_tone: false
```

**configs/logging.yaml**：
```yaml
version: 1
disable_existing_loggers: false

formatters:
  simple:
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  detailed:
    format: '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: simple
    stream: ext://sys.stdout

  file:
    class: logging.handlers.RotatingFileHandler
    level: DEBUG
    formatter: detailed
    filename: logs/ir_system.log
    maxBytes: 10485760  # 10MB
    backupCount: 3

root:
  level: INFO
  handlers: [console, file]
```

### 1.4 其他設定檔

**pytest.ini**：
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    -v
    --tb=short
    --strict-markers
    --cov=src/ir
    --cov-report=html
    --cov-report=term-missing
```

**.gitignore**：
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
*.egg-info/
dist/
build/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Testing
.pytest_cache/
.coverage
htmlcov/

# Logs
logs/
*.log

# Data
datasets/*.json
datasets/*.csv
!datasets/mini/
!datasets/lexicon/

# Temporary
tmp/
temp/
*.tmp
```

**README.md**（英文簡介）：
```markdown
# Information Retrieval System

An implementation of traditional IR techniques for the course LIS5033 - Automatic Classification and Indexing.

## Features

- CSoundex: Chinese phonetic encoding
- Boolean Retrieval with inverted index
- Vector Space Model (TF-IDF)
- Evaluation metrics (MAP, nDCG)
- Query expansion (Rocchio)
- Clustering & Summarization

## Quick Start

\`\`\`bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Encode Chinese text with CSoundex
python scripts/csoundex_encode.py --text "張三李四"
\`\`\`

## Documentation

Complete Chinese documentation is available in `docs/` directory.

## License

MIT License
```

### 1.5 執行步驟

```bash
# 1. 建立目錄結構
mkdir -p src/ir/{text,index,retrieval,eval,ranking,cluster,summarize}
mkdir -p scripts tests configs datasets/{mini,lexicon} logs

# 2. 建立 __init__.py 檔案
touch src/__init__.py
touch src/ir/__init__.py
touch src/ir/{text,index,retrieval,eval,ranking,cluster,summarize}/__init__.py
touch tests/__init__.py

# 3. 建立設定檔
# （將上述內容寫入對應檔案）

# 4. 安裝依賴
pip install -r requirements.txt

# 5. 驗證結構
tree -L 3
```

---

## 🎯 Phase 2: CSoundex 模組實作（Week 3-4，預計 4-5 天）

### 目標
實作完整的 CSoundex 編碼系統，包含核心演算法、CLI 工具、測試。

### 2.1 核心模組實作

**檔案**：`src/ir/text/csoundex.py`

**實作順序**：
1. ✅ 拼音轉換函式（使用 pypinyin）
2. ✅ 拼音正規化函式
3. ✅ 聲母提取與編碼
4. ✅ 韻母提取與編碼
5. ✅ 聲調提取
6. ✅ 主編碼函式（3 種模式）
7. ✅ 相似度計算函式

**預計程式碼量**：~500 行（含註解）

### 2.2 CLI 工具實作

**檔案**：`scripts/csoundex_encode.py`

**功能需求**：
```bash
# 編碼單一文字
python scripts/csoundex_encode.py --text "張三李四"

# 從檔案讀取並編碼
python scripts/csoundex_encode.py --file input.txt --output output.txt

# 指定模式
python scripts/csoundex_encode.py --text "張三" --mode standard --tone

# 批次處理
cat names.txt | python scripts/csoundex_encode.py --stdin
```

**預計程式碼量**：~150 行

### 2.3 測試實作

**檔案**：`tests/test_csoundex.py`

**測試案例**（至少 30 個）：
- 基本功能測試（10 cases）
  - 單字編碼
  - 詞彙編碼
  - 同音字匹配
- 邊界條件測試（10 cases）
  - 空字串
  - 單字元
  - 混合中英文
  - 標點符號
- 效能測試（5 cases）
  - 批次編碼速度
  - 記憶體使用
- 準確度測試（5 cases）
  - HMR 計算
  - FPR 計算

**目標覆蓋率**：> 90%

### 2.4 資料準備

**basic_pinyin.tsv**（5,000 常用字）：
```tsv
字	拼音	詞頻
的	de5	1000000
一	yi1	800000
是	shi4	750000
不	bu4	700000
了	le5	650000
...
```

**來源選項**：
1. 從開源字典提取（如 CC-CEDICT）
2. 使用教育部常用字表
3. 從 pypinyin 預設字典轉換

### 2.5 交付成果

- [ ] `src/ir/text/csoundex.py` - 核心模組（100% 測試覆蓋）
- [ ] `scripts/csoundex_encode.py` - CLI 工具（含 --help）
- [ ] `tests/test_csoundex.py` - 完整測試（30+ cases）
- [ ] `configs/csoundex.yaml` - 設定檔
- [ ] `datasets/lexicon/basic_pinyin.tsv` - 拼音字典（5000 字）
- [ ] 更新 `docs/CHANGELOG.md` - 記錄 v0.2.0 變更

---

## 🎯 Phase 3: 布林檢索系統（Week 5-6，預計 5-6 天）

### 目標
實作倒排索引與布林查詢處理系統。

### 3.1 模組實作

**src/ir/index/inverted_index.py**：
- 倒排索引建構
- 增量更新
- 索引儲存與載入
- 索引壓縮（可選）

**src/ir/index/positional_index.py**：
- 位置索引建構
- 支援詞組查詢

**src/ir/retrieval/boolean.py**：
- AND/OR/NOT 運算
- 詞組查詢處理
- 查詢解析器

**預計程式碼量**：~800 行

### 3.2 CLI 工具

**scripts/boolean_search.py**：
```bash
python scripts/boolean_search.py \
  --index index.json \
  --query "information AND retrieval"
```

### 3.3 測試資料

**datasets/mini/documents.json**（100 筆測試文件）：
```json
[
  {
    "id": 1,
    "title": "Introduction to Information Retrieval",
    "content": "Information retrieval is the activity of obtaining..."
  },
  ...
]
```

### 3.4 交付成果

- [ ] 倒排索引模組
- [ ] 位置索引模組
- [ ] 布林檢索引擎
- [ ] CLI 工具
- [ ] 測試（20+ cases）
- [ ] 測試資料集（100 documents）

---

## 🎯 Phase 4: 向量空間模型（Week 7，預計 4-5 天）

### 目標
實作 TF-IDF 權重計算與餘弦相似度排序。

### 4.1 模組實作

**src/ir/retrieval/vsm.py**：
- TF-IDF 計算
- 餘弦相似度
- Top-K 檢索
- 多種權重方案（ltc, lnc）

**預計程式碼量**：~600 行

### 4.2 CLI 工具

**scripts/vsm_search.py**：
```bash
python scripts/vsm_search.py \
  --index index.json \
  --query "machine learning" \
  --topk 10
```

### 4.3 交付成果

- [ ] VSM 檢索引擎
- [ ] CLI 工具
- [ ] 測試（15+ cases）
- [ ] 效能基準測試

---

## 🎯 Phase 5: 評估指標（Week 8，預計 3-4 天）

### 目標
實作標準 IR 評估指標。

### 5.1 模組實作

**src/ir/eval/metrics.py**：
- Precision, Recall, F-measure
- Average Precision (AP)
- Mean Average Precision (MAP)
- Normalized DCG (nDCG)

**預計程式碼量**：~400 行

### 5.2 CLI 工具

**scripts/eval_run.py**：
```bash
python scripts/eval_run.py \
  --results results.json \
  --qrels qrels.txt \
  --metrics MAP,nDCG@10,P@5
```

### 5.3 測試資料

**qrels.txt**（相關性判斷）：
```
query_id  0  doc_id  relevance
Q001      0  D123    1
Q001      0  D456    1
Q001      0  D789    0
...
```

### 5.4 交付成果

- [ ] 評估指標模組
- [ ] CLI 工具
- [ ] 測試（10+ cases）
- [ ] qrels 測試資料

---

## 🎯 Phase 6: 查詢擴展（Week 9，預計 3-4 天）

### 目標
實作 Rocchio 演算法與擬相關回饋。

### 6.1 模組實作

**src/ir/ranking/rocchio.py**：
- Rocchio 演算法
- 擬相關回饋
- 明確回饋
- 參數調校介面

**預計程式碼量**：~500 行

### 6.2 CLI 工具

**scripts/expand_query.py**：
```bash
python scripts/expand_query.py \
  --query "information retrieval" \
  --mode pseudo \
  --topk 10
```

### 6.3 交付成果

- [ ] Rocchio 模組
- [ ] CLI 工具
- [ ] 測試（10+ cases）

---

## 🎯 Phase 7: 分群演算法（Week 10，預計 4-5 天）

### 目標
實作 K-means 與階層式分群。

### 7.1 模組實作

**src/ir/cluster/doc_cluster.py**：
- K-means
- Hierarchical clustering
- Star clustering

**src/ir/cluster/term_cluster.py**：
- 字串分群
- 編輯距離計算

**預計程式碼量**：~700 行

### 7.2 交付成果

- [ ] 文件分群模組
- [ ] 詞彙分群模組
- [ ] 測試（15+ cases）

---

## 🎯 Phase 8: 自動摘要（Week 10-11，預計 3-4 天）

### 目標
實作靜態與動態摘要。

### 8.1 模組實作

**src/ir/summarize/static.py**：
- Lead-K 摘要
- 關鍵句萃取
- TF-IDF 排序

**src/ir/summarize/dynamic.py**：
- KWIC (KeyWord In Context)
- 視窗快取
- 多查詢詞高亮

**預計程式碼量**：~400 行

### 8.2 交付成果

- [ ] 靜態摘要模組
- [ ] 動態摘要模組
- [ ] 測試（10+ cases）

---

## 🎯 Phase 9: 期末專案整合（Week 15-18，預計 2-3 週）

### 目標
整合所有模組為完整的學術搜尋引擎。

### 9.1 系統整合

**整合任務**：
1. 建立統一的 API 介面
2. 開發 Web UI（Flask + Bootstrap）
3. 實作資料庫層（SQLite）
4. 整合所有檢索功能
5. 建立索引管理介面

### 9.2 資料準備

**arXiv 資料集**：
- 下載 20,000 篇論文資料
- 資料清理與預處理
- 建構索引

### 9.3 評估實驗

**實驗設計**：
1. 建構測試集（50 查詢）
2. 標註相關性判斷
3. 執行離線評估
4. 進行使用者研究（10 人）

### 9.4 交付成果

- [ ] 完整的 Web 應用
- [ ] 20,000 篇論文索引
- [ ] 評估報告
- [ ] 使用者研究報告
- [ ] 期末專案報告（25 頁）
- [ ] 展示影片（5-10 分鐘）

---

## 📅 詳細時間規劃

| 週次 | 階段 | 任務 | 產出 | 工作量 |
|------|------|------|------|--------|
| W3 | Phase 1 | 專案架構建立 | 目錄結構、配置檔 | 2-3 天 |
| W3-4 | Phase 2 | CSoundex 實作 | 核心模組、CLI、測試 | 4-5 天 |
| W5-6 | Phase 3 | 布林檢索 | 索引、檢索引擎 | 5-6 天 |
| W7 | Phase 4 | VSM 實作 | TF-IDF、排序 | 4-5 天 |
| W8 | Phase 5 | 評估指標 | MAP, nDCG | 3-4 天 |
| W9 | Phase 6 | 查詢擴展 | Rocchio | 3-4 天 |
| W10 | Phase 7 | 分群演算法 | K-means, 階層式 | 4-5 天 |
| W10-11 | Phase 8 | 自動摘要 | Lead-K, KWIC | 3-4 天 |
| W11-14 | - | **期中考準備** | 複習、擬答 | - |
| W15-18 | Phase 9 | 期末專案 | 完整系統 | 2-3 週 |

---

## ⚠️ 風險管理

### 高風險項目

| 風險 | 可能性 | 影響 | 應對策略 |
|------|--------|------|---------|
| **CSoundex 多音字處理複雜** | 高 | 中 | 先實作最高頻率策略，詞彙脈絡為進階功能 |
| **索引建構效能不佳** | 中 | 高 | 限制資料規模至 20K，使用批次處理 |
| **評估資料標註耗時** | 高 | 中 | 縮小測試集至 30-50 查詢 |
| **期末專案時間不足** | 中 | 高 | 提前完成核心模組，預留 3 週整合時間 |
| **依賴套件相容性問題** | 低 | 中 | 使用虛擬環境，固定版本號 |

### 應對原則

1. **優先順序**：核心功能 > 進階功能 > 優化
2. **時間緩衝**：每個階段預留 20% 緩衝時間
3. **彈性調整**：根據實際進度動態調整計畫
4. **文檔同步**：實作與文檔同步更新

---

## 🎯 里程碑與檢查點

### Milestone 1: 專案架構完成（Week 3 結束）
- [ ] 目錄結構建立
- [ ] 依賴安裝成功
- [ ] 測試框架可運行
- [ ] Git repository 初始化

**驗收標準**：`pytest tests/` 可執行（即使無測試案例）

### Milestone 2: CSoundex 模組完成（Week 4 結束）
- [ ] 核心編碼函式實作完成
- [ ] CLI 工具可用
- [ ] 測試覆蓋率 > 90%
- [ ] 文檔更新

**驗收標準**：
```bash
python scripts/csoundex_encode.py --text "張三" --mode standard
# 輸出: Z811 S901
```

### Milestone 3: 檢索系統完成（Week 7 結束）
- [ ] 布林檢索可用
- [ ] VSM 檢索可用
- [ ] 支援 100+ 文件索引
- [ ] 查詢回應 < 100ms

**驗收標準**：可在測試資料集上執行完整的檢索流程

### Milestone 4: 評估系統完成（Week 9 結束）
- [ ] 評估指標實作完成
- [ ] 可計算 MAP, nDCG
- [ ] 查詢擴展可用

**驗收標準**：可對檢索結果進行量化評估

### Milestone 5: 完整系統交付（Week 11 結束）
- [ ] 所有核心模組完成
- [ ] 測試覆蓋率 > 85%
- [ ] CLI 工具完整
- [ ] 文檔齊全

**驗收標準**：可提供給其他同學使用的完整 IR 工具包

### Milestone 6: 期末專案完成（Week 18 結束）
- [ ] Web 應用上線
- [ ] 20,000 篇論文可檢索
- [ ] 評估報告完成
- [ ] 最終報告提交

**驗收標準**：通過期末專案評分標準

---

## 📝 建議的工作流程

### 每個 Phase 的標準流程

1. **規劃階段**（10%）
   - 閱讀相關文檔（IMPLEMENTATION.md）
   - 設計資料結構與 API
   - 撰寫測試案例（TDD）

2. **實作階段**（60%）
   - 實作核心功能
   - 撰寫詳細註解
   - 邊寫邊測試

3. **測試階段**（20%）
   - 執行測試
   - 修正 bug
   - 達到覆蓋率目標

4. **文檔階段**（10%）
   - 更新 CHANGELOG.md
   - 撰寫使用範例
   - 更新相關文檔

### Git 工作流程

```bash
# 每個 Phase 建立新分支
git checkout -b feature/csoundex

# 頻繁 commit
git add .
git commit -m "feat(csoundex): implement phonetic encoding"

# 完成後合併到 main
git checkout main
git merge feature/csoundex

# 標記版本
git tag v0.2.0
```

---

## 🎓 學習資源

### 必讀文件
- [x] `docs/guides/IMPLEMENTATION.md` - 實作指南
- [x] `docs/guides/CSOUNDEX_DESIGN.md` - CSoundex 設計
- [ ] Manning et al. (2008) - Introduction to IR（對應章節）

### 推薦工具
- **IDE**：VS Code + Python extension
- **測試**：pytest + pytest-cov
- **除錯**：pdb / VS Code debugger
- **效能分析**：cProfile + snakeviz

### 參考實作
- **NLTK**：https://www.nltk.org/
- **Whoosh**：https://whoosh.readthedocs.io/
- **Gensim**：https://radimrehurek.com/gensim/

---

## 📊 進度追蹤

### 當前狀態
- ✅ v0.1.0: 文檔系統（已完成）
- 🔄 v0.2.0: CSoundex 模組（準備開始）
- ⏳ v0.3.0: 布林檢索（排程中）
- ⏳ v0.4.0: VSM（排程中）

### 下一步行動
1. **立即行動**（今天）：建立專案目錄結構
2. **本週目標**：完成 Phase 1 + 開始 Phase 2
3. **本月目標**：完成 CSoundex + 布林檢索

---

## 📞 協作與支援

### 需要協助時
1. 查閱 `docs/` 目錄的相關文檔
2. 參考 `docs/guides/IMPLEMENTATION.md` 的程式碼範例
3. 在 GitHub Issues 提問（若為團隊專案）
4. 詢問授課教師或助教

### 程式碼審查檢核表
- [ ] 所有函式有英文 docstring
- [ ] 包含複雜度分析（Time/Space）
- [ ] 通過 pylint 檢查
- [ ] 測試覆蓋率 > 85%
- [ ] 有實際使用範例

---

**路線圖版本**：v1.0
**最後更新**：2025-11-12
**維護者**：[您的姓名/學號]

**Let's build an amazing IR system! 🚀**
