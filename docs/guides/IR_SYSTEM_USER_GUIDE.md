# IR 系統使用指南 (Information Retrieval System User Guide)

## 目錄 (Table of Contents)

1. [系統概述](#系統概述)
2. [快速開始](#快速開始)
3. [查詢語法](#查詢語法)
4. [搜尋模式](#搜尋模式)
5. [排序模型](#排序模型)
6. [進階功能](#進階功能)
7. [效能優化](#效能優化)
8. [常見問題](#常見問題)

---

## 系統概述

本系統是一個基於 **CKIP 中文分詞** (*CKIP Chinese Word Segmentation*) 的新聞檢索系統，實作了多種經典 IR 技術：

### 核心特性

- ✅ **CKIP BERT 分詞** - 使用 BERT-based 模型進行精準中文斷詞
- ✅ **多種排序模型** - BM25, VSM (TF-IDF), Hybrid 排序
- ✅ **Boolean 檢索** - 支援 AND, OR, NOT 邏輯運算
- ✅ **欄位查詢** (*Field Query*) - 針對 title, category, source 等欄位搜尋
- ✅ **PostgreSQL 整合** - 76,668 篇新聞文章資料庫
- ✅ **即時索引** - 支援增量索引更新

### 系統架構

```
使用者查詢
    ↓
查詢解析 (Query Parser)
    ↓
CKIP 分詞 (Tokenizer)
    ↓
索引檢索 (Inverted Index)
    ↓
排序計算 (BM25/VSM/Hybrid)
    ↓
結果回傳
```

---

## 快速開始

### 1. 建立索引 (Index Building)

#### 從 JSONL 檔案建立
```bash
# 建立小型索引 (1,000 篇)
python scripts/search_news.py --build --limit 1000 --index-dir data/index_1k

# 建立中型索引 (50,000 篇)
python scripts/search_news.py --build --limit 50000 --index-dir data/index_50k

# 建立完整索引 (全部文檔)
python scripts/search_news.py --build --index-dir data/index_full
```

#### 從 PostgreSQL 建立
```bash
# 從資料庫建立索引
python scripts/search_news.py --build --from-db --db-name ir_news --index-dir data/index_db
```

### 2. 執行查詢

#### 互動式搜尋
```bash
python scripts/search_news.py --index-dir data/index_50k
```

#### 命令列查詢
```bash
# 簡單查詢
python scripts/search_news.py --query "台灣 經濟" --index-dir data/index_50k

# Boolean 查詢
python scripts/search_news.py --query "台灣 AND 經濟" --mode boolean

# 欄位查詢
python scripts/search_news.py --query "title:AI" --mode field

# 指定排序模型
python scripts/search_news.py --query "人工智慧" --model VSM --top-k 20
```

### 3. Web 介面

```bash
# 啟動 Flask Web 應用
python app_simple.py

# 瀏覽器開啟
http://localhost:5000
```

---

## 查詢語法

### 簡單查詢 (Simple Query)

**語法**: 直接輸入關鍵字，以空格分隔

```
台灣 經濟        # 搜尋包含「台灣」或「經濟」的文章
人工智慧         # 搜尋包含「人工智慧」的文章
COVID-19 疫苗    # 支援英文和數字
```

**CKIP 分詞範例**:
```
輸入: "台積電股價上漲"
分詞: ["台積電", "股價", "上漲"]
```

---

### Boolean 查詢 (Boolean Query)

**語法**: 使用 AND, OR, NOT 運算子

#### AND 運算 (交集)
```
台灣 AND 經濟           # 必須同時包含「台灣」和「經濟」
人工智慧 AND 應用       # AI 應用相關文章
```

#### OR 運算 (聯集)
```
經濟 OR 金融            # 包含「經濟」或「金融」
AI OR 機器學習          # AI 或 ML 相關
```

#### NOT 運算 (排除)
```
疫苗 AND NOT 副作用     # 疫苗新聞但排除副作用
政治 NOT 選舉           # 政治新聞但不包含選舉
```

#### 複雜組合 (使用括號)
```
(台灣 OR 中國) AND 貿易                    # 台灣或中國的貿易新聞
台灣 AND (經濟 OR 金融) AND NOT 股市       # 台灣經濟/金融但非股市
```

---

### 欄位查詢 (Field Query)

**語法**: `欄位名:查詢詞`

支援欄位:
- `title:` - 標題
- `category:` - 分類 (政治、財經、生活等)
- `source:` - 來源媒體

#### 單一欄位
```
title:台灣              # 標題包含「台灣」
category:政治           # 政治類新聞
source:ltn              # 自由時報的文章
```

#### 多欄位組合
```
title:經濟 AND category:財經              # 標題有「經濟」且分類為財經
source:yahoo AND title:AI                 # Yahoo 的 AI 相關標題
title:台灣 AND (經濟 OR 金融)             # 標題有台灣，內容有經濟或金融
```

#### 欄位 + 內容混合
```
title:疫情 AND 疫苗 AND NOT 副作用        # 標題有疫情，內容有疫苗但無副作用
source:ltn AND category:政治 AND 選舉     # 自由時報政治類選舉新聞
```

---

## 搜尋模式

### 1. SIMPLE 模式 (預設)
- 自動 CKIP 分詞
- 關鍵字查詢
- BM25/VSM 排序
- 最適合日常查詢

**使用時機**: 一般關鍵字搜尋

```python
from src.ir.search import UnifiedSearchEngine, QueryMode

engine = UnifiedSearchEngine("data/index_50k")
results = engine.search("台灣 經濟", mode=QueryMode.SIMPLE)
```

---

### 2. BOOLEAN 模式
- 支援 AND/OR/NOT
- 精確邏輯控制
- 適合專業查詢

**使用時機**: 需要精確控制搜尋邏輯

```python
results = engine.search("台灣 AND 經濟", mode=QueryMode.BOOLEAN)
```

---

### 3. FIELD 模式
- 針對特定欄位
- 支援多欄位組合
- 適合結構化查詢

**使用時機**: 已知要搜尋的欄位

```python
results = engine.search("title:AI", mode=QueryMode.FIELD)
```

---

### 4. AUTO 模式 (智慧偵測)
- 自動判斷查詢類型
- 偵測 Boolean 運算子
- 偵測欄位語法

**使用時機**: 不確定查詢類型

```python
results = engine.search(user_query, mode=QueryMode.AUTO)
```

---

## 排序模型

### 1. BM25 (Best Match 25)

**特點**:
- 考慮詞頻 (*Term Frequency*)
- 考慮文檔長度正規化
- 業界標準排序演算法

**適用情境**: 一般新聞檢索

**參數**:
- k1 = 1.5 (詞頻飽和度)
- b = 0.75 (長度正規化)

```python
results = engine.search("台灣 經濟", ranking_model=RankingModel.BM25)
```

**評分公式**:
```
score(d, q) = Σ IDF(qi) · [f(qi, d) · (k1 + 1)] / [f(qi, d) + k1 · (1 - b + b · |d| / avgdl)]
```

---

### 2. VSM (Vector Space Model)

**特點**:
- TF-IDF 加權
- Cosine 相似度
- 向量空間模型

**適用情境**: 相似文章查找

```python
results = engine.search("人工智慧", ranking_model=RankingModel.VSM)
```

**評分公式**:
```
score(d, q) = cosine(tf-idf(d), tf-idf(q))
```

---

### 3. HYBRID (混合模型)

**特點**:
- BM25 + VSM 加權平均
- 兼具兩者優點
- α = 0.7 (BM25), β = 0.3 (VSM)

**適用情境**: 追求最佳檢索品質

```python
results = engine.search("科技 創新", ranking_model=RankingModel.HYBRID)
```

---

## 進階功能

### 1. 查詢擴展 (Query Expansion)

```python
# Pseudo-Relevance Feedback
expanded_query = engine.expand_query("AI", top_k=10, num_terms=5)
results = engine.search(expanded_query)
```

### 2. 相關文章推薦

```python
# 找出與某篇文章相似的其他文章
similar_docs = engine.find_similar(doc_id=12345, top_k=10)
```

### 3. 分面瀏覽 (Faceted Search)

```python
# 取得分類統計
facets = engine.get_facets(query="台灣")
# 輸出: {'政治': 150, '經濟': 120, '社會': 80, ...}
```

### 4. 高亮顯示 (Highlighting)

```python
# 自動標記查詢詞在內容中的位置
results = engine.search("台灣 經濟", highlight=True)
for result in results:
    print(result.highlighted_content)
```

---

## 效能優化

### 索引大小與查詢速度

| 文檔數量 | 索引大小 | 平均查詢時間 | 建議用途 |
|---------|---------|------------|---------|
| 1,000   | ~50 MB  | < 50ms     | 開發測試 |
| 50,000  | ~500 MB | < 200ms    | 一般應用 |
| 全部 (76K) | ~800 MB | < 500ms    | 正式環境 |

### CKIP 分詞效能

- **單筆分詞**: ~250ms (含模型載入)
- **批次分詞**: ~100ms/篇 (batch=10)
- **加速比**: 2.5x

**優化建議**:
```python
# 使用批次分詞
tokenizer.tokenize_batch(queries)  # ✅ 快

# 避免逐筆分詞
for q in queries:
    tokenizer.tokenize(q)  # ❌ 慢
```

### 快取策略

```python
# 啟用查詢快取
engine = UnifiedSearchEngine("data/index_50k", enable_cache=True)

# 熱門查詢會自動快取
results = engine.search("台灣")  # 第一次: 200ms
results = engine.search("台灣")  # 第二次: 10ms (快取)
```

---

## 常見問題

### Q1: 為什麼搜尋「台灣經濟」沒結果？

**A**: CKIP 會將其斷詞為 `["台灣", "經濟"]`，系統會搜尋包含這兩個詞的文章。如果沒結果，可能是：
- 索引尚未建立或已損壞
- 查詢詞過於精確

**解決方案**:
```bash
# 檢查索引狀態
python scripts/search_news.py --stats --index-dir data/index_50k

# 嘗試更寬鬆的查詢
python scripts/search_news.py --query "台灣 OR 經濟"
```

---

### Q2: Boolean 查詢沒有正確過濾？

**A**: 確認：
1. 使用大寫 `AND`, `OR`, `NOT`
2. 指定 `--mode boolean`

```bash
# ✅ 正確
python scripts/search_news.py --query "台灣 AND 經濟" --mode boolean

# ❌ 錯誤 (會被當成一般關鍵字)
python scripts/search_news.py --query "台灣 and 經濟"
```

---

### Q3: 欄位查詢語法錯誤？

**A**: 檢查：
- 欄位名稱拼寫: `title:`, `category:`, `source:`
- 冒號後不要有空格
- 使用 `--mode field`

```bash
# ✅ 正確
--query "title:AI" --mode field

# ❌ 錯誤
--query "title: AI"  # 冒號後有空格
--query "標題:AI"    # 使用中文欄位名
```

---

### Q4: 查詢速度太慢？

**A**: 優化方法：
1. 減少 `--top-k` 數量 (預設 10)
2. 使用較小的索引
3. 啟用快取
4. 避免過於寬鬆的查詢 (如單字查詢)

```python
# 慢速查詢
results = engine.search("的", top_k=1000)  # ❌

# 快速查詢
results = engine.search("台灣 經濟 政策", top_k=10)  # ✅
```

---

### Q5: CKIP 分詞結果不理想？

**A**: CKIP 是基於 BERT 的模型，但仍可能有誤切。可以：

1. 查看分詞結果:
```python
from src.ir.text.ckip_tokenizer import get_tokenizer
tokenizer = get_tokenizer()
tokens = tokenizer.tokenize("台積電股價上漲")
print(tokens)  # ['台積電', '股價', '上漲']
```

2. 使用同義詞或變體查詢:
```bash
# 如果「人工智慧」分詞不佳
--query "AI OR 人工智慧 OR 機器學習"
```

---

### Q6: 如何匯出搜尋結果？

**A**: 使用 JSON 格式輸出:

```bash
python scripts/search_news.py --query "台灣" --output results.json --format json
```

或使用 Python API:
```python
results = engine.search("台灣")
import json
with open('results.json', 'w', encoding='utf-8') as f:
    json.dump([
        {
            'doc_id': r.doc_id,
            'title': r.title,
            'score': r.score,
            'source': r.source
        } for r in results
    ], f, ensure_ascii=False, indent=2)
```

---

## 系統限制

| 項目 | 限制 | 說明 |
|-----|------|-----|
| 最大索引大小 | ~1M 文檔 | 受記憶體限制 |
| 單次查詢詞數 | < 20 詞 | 過多會影響效能 |
| Top-K 上限 | 1000 | 超過會顯著變慢 |
| CKIP 輸入長度 | < 512 字元 | BERT 模型限制 |
| Boolean 巢狀深度 | < 5 層 | 複雜查詢解析限制 |

---

## 參考資料

- **CKIP Transformers**: https://github.com/ckiplab/ckip-transformers
- **BM25 論文**: Robertson & Walker (1994)
- **課程教材**: Introduction to Information Retrieval (Manning, Raghavan, Schütze)

---

## 技術支援

- **GitHub Issues**: https://github.com/your-repo/issues
- **文檔更新**: 2025-11-20
- **版本**: v1.0

---

**祝使用愉快！Happy Searching! 🔍**
