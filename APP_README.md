# Information Retrieval Web Application

## 資訊檢索系統 Web 應用程式

這是一個基於 Flask 的資訊檢索系統展示應用程式，整合了多種 IR 技術來檢索和分析中央社（CNA）新聞文章。

## 功能特色 Features

### 1. 布林檢索 Boolean Retrieval
- 支援 AND、OR、NOT 運算子
- 括號組合查詢
- 精確匹配文檔

**範例查詢**:
- `台灣 AND 經濟`
- `美國 OR 中國`
- `人工智慧 AND NOT 風險`

### 2. 向量空間模型 Vector Space Model (VSM)
- TF-IDF 權重計算
- 餘弦相似度排序
- 相關性評分展示

**範例查詢**:
- `人工智慧發展`
- `氣候變遷影響`
- `科技創新`

### 3. 查詢擴展 Query Expansion (Rocchio)
- 偽相關回饋 Pseudo-Relevance Feedback
- 自動擴充查詢詞項
- 提升檢索效能

### 4. 文檔摘要 Document Summarization
- **Lead-K**: 前 K 句摘要
- **關鍵句子提取**: Key Sentence Extraction
- **KWIC**: KeyWord In Context 上下文視窗

### 5. 文檔分群 Document Clustering
- **階層式分群**: Hierarchical Clustering
- **K-means**: Flat Clustering
- 自動文檔組織

## 系統架構 Architecture

```
Flask Web Application
├── Backend (Python)
│   ├── InvertedIndex (倒排索引)
│   ├── PositionalIndex (位置索引)
│   ├── BooleanQueryEngine (布林檢索引擎)
│   ├── VectorSpaceModel (向量空間模型)
│   ├── StaticSummarizer (靜態摘要)
│   ├── KWICGenerator (KWIC 生成器)
│   ├── RocchioExpander (Rocchio 擴展)
│   └── DocumentClusterer (文檔分群)
├── Frontend (HTML + JavaScript)
│   ├── Bootstrap 5 UI
│   ├── Font Awesome Icons
│   └── Responsive Design
└── Dataset
    └── 121 CNA news articles (data/processed/cna_mvp_cleaned.jsonl)
```

## 安裝與執行 Installation & Running

### 1. 安裝相依套件
```bash
pip install flask flask-cors
```

### 2. 確認資料集存在
```bash
ls data/processed/cna_mvp_cleaned.jsonl
```

應包含 121 篇中央社新聞文章。

### 3. 啟動應用程式
```bash
python app.py
```

### 4. 開啟瀏覽器
訪問: **http://localhost:5001**

## API 端點 API Endpoints

### GET /api/stats
取得系統統計資訊

**回應範例**:
```json
{
    "documents": 121,
    "vocabulary_size": 6710,
    "avg_doc_length": 0,
    "total_terms": 8013
}
```

### POST /api/search/boolean
執行布林檢索

**請求範例**:
```json
{
    "query": "台灣 AND 經濟",
    "limit": 10
}
```

**回應範例**:
```json
{
    "query": "台灣 AND 經濟",
    "results": [
        {
            "doc_id": 16,
            "title": "...",
            "snippet": "...",
            "url": "...",
            "date": "2025-11-11",
            "category": "政治"
        }
    ],
    "total": 2,
    "execution_time": 0.0002
}
```

### POST /api/search/vsm
執行向量空間模型檢索

**請求範例**:
```json
{
    "query": "人工智慧",
    "limit": 10
}
```

**回應範例**:
```json
{
    "query": "人工智慧",
    "results": [
        {
            "doc_id": 116,
            "title": "...",
            "snippet": "...",
            "score": 0.8542,
            "url": "...",
            "date": "2025-11-13",
            "category": "科技"
        }
    ],
    "total": 1,
    "execution_time": 0.000067
}
```

### GET /api/document/{doc_id}
取得完整文檔內容

### POST /api/summarize/{doc_id}
生成文檔摘要

**請求範例**:
```json
{
    "method": "lead_k",  // 或 "key_sentence", "kwic"
    "k": 3,
    "keyword": "台灣"  // 僅 KWIC 需要
}
```

### POST /api/expand_query
查詢擴展

**請求範例**:
```json
{
    "query": "人工智慧",
    "relevant_docs": [0, 1, 2]  // 可選
}
```

### POST /api/cluster
文檔分群

**請求範例**:
```json
{
    "n_clusters": 3,
    "method": "hierarchical",  // 或 "kmeans"
    "doc_ids": [0, 1, 2, ...]  // 可選
}
```

## 資料集資訊 Dataset Information

### 來源 Source
中央社（Central News Agency, CNA）新聞文章

### 統計數據 Statistics
- **文檔數量**: 121 篇
- **詞彙量**: 6,710 個獨特詞項
- **總詞項數**: 8,013
- **日期範圍**: 2025-11-07 至 2025-11-13
- **分類**: 政治、財經、社會、科技等

### 文章結構
```json
{
    "article_id": "202511120135",
    "title": "文章標題",
    "content": "完整內容...",
    "url": "https://www.cna.com.tw/...",
    "published_date": "2025-11-12",
    "category": "aipl",
    "category_name": "政治",
    "tags": ["標籤1", "標籤2"],
    "author": "中央通訊社"
}
```

## 初始化時間 Initialization Time

應用程式啟動時會自動建立以下索引：
- ✅ Inverted Index: ~8ms
- ✅ Positional Index: ~5ms
- ✅ Vector Space Model: ~115ms
- ✅ Total: **~130ms**

## 效能指標 Performance Metrics

### 搜尋速度
- **布林檢索**: ~0.2ms
- **VSM 檢索**: ~0.06ms
- **查詢擴展**: ~5ms
- **文檔分群**: ~50ms (30 docs)

### 記憶體使用
- **倒排索引**: ~2MB
- **文檔向量**: ~5MB
- **Total**: ~10MB

## 技術棧 Technology Stack

### Backend
- Python 3.8+
- Flask 3.1.2
- Flask-CORS 6.0.1

### Frontend
- Bootstrap 5.3.0
- Font Awesome 6.4.0
- Vanilla JavaScript (ES6)

### IR Modules
- Inverted Index & Positional Index
- TF-IDF & Vector Space Model
- Boolean Query Engine
- Rocchio Query Expansion
- Static & Dynamic Summarization
- Hierarchical & K-means Clustering

## 專案結構 Project Structure

```
.
├── app.py                      # Flask 主應用程式
├── templates/
│   └── index.html             # Web UI 介面
├── static/
│   ├── css/                   # 樣式表（內嵌在 HTML）
│   └── js/                    # JavaScript（內嵌在 HTML）
├── src/ir/                    # IR 模組
│   ├── index/                 # 索引模組
│   ├── retrieval/             # 檢索模組
│   ├── summarize/             # 摘要模組
│   ├── cluster/               # 分群模組
│   └── ranking/               # 排序模組
└── data/
    └── processed/
        └── cna_mvp_cleaned.jsonl  # 清理後的資料集
```

## 使用範例 Usage Examples

### 範例 1: 布林檢索
1. 選擇「布林檢索 Boolean」標籤
2. 輸入: `台灣 AND 經濟`
3. 點擊「搜尋」
4. 查看精確匹配的文檔列表

### 範例 2: VSM 檢索 + 查詢擴展
1. 選擇「向量空間模型 VSM」標籤
2. 輸入: `人工智慧發展`
3. 勾選「啟用查詢擴展 (Rocchio)」
4. 點擊「搜尋」
5. 查看擴展詞項和相關性排序結果

### 範例 3: 文檔摘要
1. 從搜尋結果點擊任一文檔標題
2. 在彈出視窗查看完整內容
3. 點擊「Lead-K」或「關鍵句子」按鈕生成摘要
4. 或輸入關鍵詞後點擊「KWIC」查看上下文

### 範例 4: 文檔分群
1. 選擇「文檔分群 Clustering」標籤
2. 設定分群方法（階層式/K-means）
3. 設定群組數量（例如: 3）
4. 設定文檔數量（例如: 30）
5. 點擊「執行分群」
6. 查看自動分類的文檔群組

## 疑難排解 Troubleshooting

### 問題 1: 埠口被佔用
**錯誤訊息**: `Address already in use`

**解決方法**:
```bash
# 方法 1: 更改 app.py 的埠口
# 編輯 app.py，將 port=5001 改為其他埠口

# 方法 2: 停止佔用埠口的程序
lsof -i :5001
kill -9 <PID>
```

### 問題 2: 模組導入錯誤
**錯誤訊息**: `ModuleNotFoundError`

**解決方法**:
```bash
# 確認當前目錄
pwd

# 確認在專案根目錄執行
cd /mnt/c/web-projects/information-retrieval
python app.py
```

### 問題 3: 資料集不存在
**錯誤訊息**: `FileNotFoundError: data/processed/cna_mvp_cleaned.jsonl`

**解決方法**:
```bash
# 檢查資料集是否存在
ls data/processed/cna_mvp_cleaned.jsonl

# 如果不存在，執行資料清理
python scripts/data/clean_dataset.py
```

## 未來改進 Future Improvements

### 短期 Short-term
- [ ] 添加查詢歷史記錄
- [ ] 實作分面瀏覽 Faceted Search
- [ ] 添加高亮顯示關鍵詞
- [ ] 匯出搜尋結果

### 中期 Medium-term
- [ ] 整合 SQLite 資料庫
- [ ] 實作使用者認證
- [ ] 添加個人化推薦
- [ ] 支援更多資料來源

### 長期 Long-term
- [ ] 分散式索引架構
- [ ] 即時增量索引
- [ ] 機器學習排序 LTR
- [ ] 多語言支援

## 相關文件 Related Documents

- [專案路線圖](docs/PROJECT_ROADMAP.md) - 完整開發計畫
- [模組測試報告](data/test_results/module_test_results_final.txt) - 測試結果
- [資料集統計](data/stats/cna_mvp_stats.txt) - 資料分析
- [CHANGELOG](docs/CHANGELOG.md) - 變更記錄

## 授權 License

Educational Use Only - National Taiwan University
僅供教育使用 - 國立臺灣大學

## 致謝 Acknowledgments

- **課程**: LIS5033 - Automatic Classification and Indexing
- **教材**: Introduction to Information Retrieval (Manning, Raghavan, Schütze)
- **資料來源**: 中央社 Central News Agency

---

**專案資訊 Project Info**
Author: Information Retrieval System
Version: 1.0.0
Date: 2025-11-13
