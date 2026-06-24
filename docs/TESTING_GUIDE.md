# 完整系統測試指南 (Complete System Testing Guide)

**版本**: v4.0
**日期**: 2025-11-14
**目的**: 驗證所有 29 個 API 端點正常運作

---

## 📋 測試準備 (Test Preparation)

### 1. 系統需求
- Python 3.8+
- 所有依賴已安裝: `pip install -r requirements.txt`
- 至少 4GB 可用記憶體 (CKIP Transformers 需要)
- 數據集已準備: `/mnt/c/data/information-retrieval/processed/cna_mvp_cleaned.jsonl`

### 2. 啟動伺服器

```bash
# 方法 1: 直接啟動 (推薦用於測試,可看到即時輸出)
python app.py

# 方法 2: 背景啟動
nohup python app.py > server.log 2>&1 &

# 方法 3: 使用 Gunicorn (生產環境)
gunicorn -w 1 -b 0.0.0.0:5001 --timeout 300 app:app
```

**重要**: 初始化需要 2-5 分鐘 (載入 CKIP Transformers 模型)

### 3. 確認伺服器已啟動

```bash
# 檢查伺服器是否響應
curl http://localhost:5001/api/stats

# 應該返回 JSON 格式的統計資訊
# {"documents": 121, "vocabulary_size": 8478, ...}
```

---

## 🧪 執行測試 (Running Tests)

### 快速測試 (Quick Test)

測試 7 個核心 API,確認系統基本運作:

```bash
python scripts/test_complete_system.py --quick
```

**預期輸出**:
```
✓ System Stats              0.012s
✓ Boolean Search            0.145s
✓ VSM Search                0.089s
✓ WAND Search               0.023s
✓ Similar Docs              0.034s
✓ CF Recommendation         0.056s
✓ Keyword Extract           0.234s

Quick Test Result:
  7/7 essential APIs working
  ✓ System appears operational
```

---

### 完整測試 (Full Test)

測試所有 29 個 API 端點:

```bash
python scripts/test_complete_system.py
```

**測試覆蓋**:
- ✅ 系統 APIs (1個): stats
- ✅ 檢索 APIs (7個): boolean, vsm, bm25, lm, hybrid, wand, maxscore
- ✅ 推薦 APIs (9個): similar, personalized, trending, cf×3, hybrid, interaction×2
- ✅ NLP APIs (5個): keywords, topics, patterns, ner, syntax
- ✅ 文檔 APIs (4個): document, analysis, summarize, expand_query
- ✅ 語言模型 APIs (2個): collocation, ngram

**預期輸出**:
```
TEST REPORT - 完整測試報告

Summary:
  Total APIs tested: 29
  Passed: 29
  Failed: 0
  Success rate: 100.0%
  Total execution time: 12.45s
  Average response time: 0.429s

🎉 ALL TESTS PASSED! System is fully operational.
```

---

### 詳細測試 (Verbose Mode)

顯示每個 API 的詳細響應:

```bash
python scripts/test_complete_system.py --verbose
```

---

## 🔍 手動測試 (Manual Testing)

如果自動化測試失敗,可以手動測試各個 API:

### 1. 系統統計

```bash
curl http://localhost:5001/api/stats
```

### 2. 布林檢索

```bash
curl -X POST http://localhost:5001/api/search/boolean \
  -H "Content-Type: application/json" \
  -d '{"query": "台灣 AND 經濟", "limit": 5}'
```

### 3. VSM 檢索

```bash
curl -X POST http://localhost:5001/api/search/vsm \
  -H "Content-Type: application/json" \
  -d '{"query": "人工智慧", "limit": 5}'
```

### 4. WAND 優化檢索

```bash
curl -X POST http://localhost:5001/api/search/wand \
  -H "Content-Type: application/json" \
  -d '{"query": "深度學習應用", "limit": 10}'
```

**預期響應**:
```json
{
  "query": "深度學習應用",
  "algorithm": "WAND",
  "results": [...],
  "statistics": {
    "num_scored_docs": 15,
    "num_candidate_docs": 98,
    "speedup_ratio": 6.53
  }
}
```

### 5. 內容推薦

```bash
curl -X POST http://localhost:5001/api/recommend/similar \
  -H "Content-Type: application/json" \
  -d '{"doc_id": 0, "top_k": 5, "apply_diversity": true}'
```

### 6. 協同過濾推薦

```bash
curl -X POST http://localhost:5001/api/recommend/cf/item-based \
  -H "Content-Type: application/json" \
  -d '{"user_id": 0, "top_k": 10}'
```

### 7. 混合推薦

```bash
curl -X POST http://localhost:5001/api/recommend/hybrid \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 0,
    "top_k": 10,
    "fusion_method": "weighted",
    "content_weight": 0.5,
    "cf_weight": 0.4,
    "popularity_weight": 0.1
  }'
```

### 8. 關鍵字提取

```bash
curl -X POST http://localhost:5001/api/extract/keywords \
  -H "Content-Type: application/json" \
  -d '{
    "text": "人工智慧和機器學習是現代科技的重要發展領域",
    "method": "textrank",
    "topk": 5
  }'
```

### 9. 主題建模

```bash
curl -X POST http://localhost:5001/api/extract/topics \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["人工智慧深度學習", "機器學習神經網路"],
    "method": "lda",
    "n_topics": 2
  }'
```

### 10. 命名實體識別

```bash
curl -X POST http://localhost:5001/api/analyze/ner \
  -H "Content-Type: application/json" \
  -d '{"text": "台灣位於東亞,首都是台北"}'
```

---

## 📊 測試結果解讀 (Interpreting Results)

### 成功標準

| 指標 | 標準 | 說明 |
|------|------|------|
| 成功率 | ≥ 95% | 至少 28/29 API 通過 |
| 平均響應時間 | < 1s | 大部分 API 應在 500ms 內 |
| 檢索加速比 (WAND) | > 3x | 相比 naive 檢索的加速 |
| 推薦延遲 | < 100ms | 內容推薦和 CF 推薦 |

### 性能基準

| API 類型 | 預期響應時間 | 說明 |
|---------|-------------|------|
| 簡單檢索 (Boolean) | 50-150ms | 取決於查詢複雜度 |
| VSM/BM25 | 20-80ms | 向量檢索較快 |
| WAND/MaxScore | 5-30ms | 優化後更快 |
| 內容推薦 | 20-50ms | 使用預計算向量 |
| 協同過濾 | 30-100ms | 取決於用戶數 |
| 混合推薦 | 50-150ms | 組合多個推薦器 |
| NLP 分析 | 100-500ms | CKIP/BERT 模型較慢 |

---

## 🐛 常見問題排除 (Troubleshooting)

### 問題 1: 伺服器無法啟動

**症狀**: `Connection refused` 或 `Server not running`

**解決方法**:
```bash
# 1. 檢查是否有其他進程佔用 5001 端口
lsof -i :5001
# 或
netstat -tulpn | grep 5001

# 2. 終止舊進程
pkill -f "python.*app.py"

# 3. 重新啟動
python app.py
```

### 問題 2: 初始化時間過長

**症狀**: 伺服器啟動後長時間無響應

**原因**: CKIP Transformers 模型載入需要時間

**解決方法**:
- 等待 2-5 分鐘
- 檢查記憶體是否足夠 (需要 2-4GB)
- 查看日誌: `tail -f server.log`

### 問題 3: CKIP Transformers 錯誤

**症狀**: `ModuleNotFoundError: ckip_transformers`

**解決方法**:
```bash
pip install -U ckip-transformers
```

### 問題 4: 某些 API 失敗

**症狀**: 部分 API 返回 500 錯誤

**診斷步驟**:
1. 查看詳細錯誤訊息:
   ```bash
   curl -v http://localhost:5001/api/[endpoint]
   ```

2. 檢查伺服器日誌:
   ```bash
   tail -100 server.log | grep ERROR
   ```

3. 測試依賴模組:
   ```python
   # Python 互動環境
   from src.ir.recommendation import ContentBasedRecommender
   from src.ir.retrieval.query_optimization import WANDRetrieval
   ```

### 問題 5: 推薦系統返回空結果

**原因**:
- 用戶互動數據不足
- 文檔向量未正確建立

**解決方法**:
1. 先記錄一些互動:
   ```bash
   curl -X POST http://localhost:5001/api/interaction/record \
     -H "Content-Type: application/json" \
     -d '{"user_id": 0, "doc_id": 5, "interaction_type": "read"}'
   ```

2. 使用內容推薦作為 fallback

### 問題 6: WAND/MaxScore 加速比低

**原因**:
- 數據集太小 (121 篇文章)
- 查詢詞太常見

**說明**: 在大規模數據集 (>100K 文檔) 上加速效果更明顯

---

## 📈 性能測試 (Performance Testing)

### 壓力測試

使用 Apache Bench 進行壓力測試:

```bash
# 測試 VSM 檢索 (100 requests, 10 concurrent)
ab -n 100 -c 10 -p query.json -T application/json \
   http://localhost:5001/api/search/vsm

# query.json 內容:
# {"query": "人工智慧", "limit": 10}
```

### 基準測試

```bash
# 測試腳本包含性能基準
python scripts/test_query_optimization.py --compare
```

**輸出範例**:
```
Algorithm Comparison:
BM25 time:      0.0234s (baseline)
WAND time:      0.0045s (5.2x faster)
MaxScore time:  0.0067s (3.5x faster)
```

---

## ✅ 測試檢查清單 (Test Checklist)

使用以下檢查清單確保完整測試:

### 基礎功能
- [ ] 伺服器成功啟動
- [ ] `/api/stats` 返回正確統計
- [ ] 至少一個檢索 API 正常運作

### 檢索功能 (7個 API)
- [ ] Boolean Search (布林檢索)
- [ ] VSM Search (向量空間模型)
- [ ] BM25 Ranking (BM25 排序)
- [ ] Language Model Retrieval (語言模型)
- [ ] Hybrid Search (混合檢索)
- [ ] WAND Optimization (WAND 優化)
- [ ] MaxScore Optimization (MaxScore 優化)

### 推薦系統 (9個 API)
- [ ] Similar Documents (相似文檔)
- [ ] Personalized Recommendations (個人化推薦)
- [ ] Trending Documents (熱門文檔)
- [ ] User-Based CF (基於用戶的協同過濾)
- [ ] Item-Based CF (基於項目的協同過濾)
- [ ] Matrix Factorization (矩陣分解)
- [ ] Hybrid Recommender (混合推薦)
- [ ] Interaction Recording (互動記錄)
- [ ] Interaction History (互動歷史)

### NLP 分析 (5個 API)
- [ ] Keyword Extraction (關鍵字提取)
- [ ] Topic Modeling (主題建模)
- [ ] Pattern Mining (模式挖掘)
- [ ] Named Entity Recognition (NER)
- [ ] Syntax Analysis (句法分析)

### 文檔操作 (4個 API)
- [ ] Get Document (獲取文檔)
- [ ] Document Analysis (文檔分析)
- [ ] Summarization (文檔摘要)
- [ ] Query Expansion (查詢擴展)

### 語言模型 (2個 API)
- [ ] Collocation Extraction (詞彙共現)
- [ ] N-gram Analysis (N-gram 分析)

---

## 📝 測試報告範本 (Test Report Template)

```
=================================================
系統測試報告 (System Test Report)
=================================================

測試日期: 2025-11-14
測試人員: [姓名]
系統版本: v4.0

--- 測試環境 ---
OS: [作業系統]
Python: [版本]
記憶體: [可用記憶體]
數據集: cna_mvp_cleaned.jsonl (121 篇文章)

--- 測試結果 ---
總計 API: 29
通過: [數量]
失敗: [數量]
成功率: [百分比]

--- 性能指標 ---
平均響應時間: [時間] ms
WAND 加速比: [倍數] x
推薦延遲: [時間] ms

--- 問題記錄 ---
[列出遇到的問題和解決方法]

--- 結論 ---
[系統是否可用於生產環境]

=================================================
```

---

## 🚀 持續整合測試 (CI Testing)

如需設置自動化測試,可使用以下 GitHub Actions 配置:

```yaml
# .github/workflows/test.yml
name: System Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.9'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt

    - name: Start server
      run: |
        python app.py &
        sleep 60  # Wait for initialization

    - name: Run tests
      run: |
        python scripts/test_complete_system.py
```

---

## 📞 支援與回饋 (Support & Feedback)

如果測試過程中遇到問題:

1. 查看日誌: `cat server.log`
2. 檢查文檔: `docs/API.md`
3. 參考範例: `scripts/test_*.py`
4. 提交 Issue: [GitHub Issues]

---

**測試指南版本**: v1.0
**最後更新**: 2025-11-14
**維護**: Information Retrieval System Team
