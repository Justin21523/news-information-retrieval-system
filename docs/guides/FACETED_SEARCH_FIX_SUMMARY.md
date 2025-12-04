# Faceted Search 問題修復與改善建議

## 問題診斷結果

### 1. JavaScript錯誤 ✅ 已修復

**問題**: `Cannot read properties of undefined (reading 'toUpperCase')`

**原因**: `static/js/search.js` 第 182 行直接調用 `data.model.toUpperCase()` 而未檢查 `data.model` 是否存在

**修復**:
```javascript
// 修復前
modelName.textContent = `📊 ${data.model.toUpperCase()}`;

// 修復後
modelName.textContent = `📊 ${data.model ? data.model.toUpperCase() : 'Unknown'}`;
```

**檔案**: `/mnt/c/web-projects/information-retrieval/static/js/search.js:182`

---

### 2. 文檔數量問題 🔍 已診斷

**問題**: 只顯示 121 筆文檔，而非預期的 5000+ 筆

**原因分析**:

#### 2.1 當前索引狀態
- **使用中的索引**: `data/indexes/` - 只包含 121 個文檔
- **配置位置**: `app_simple.py` 第 51 行
  ```python
  app.config['INDEX_DIR'] = project_root / 'data' / 'indexes'
  ```

#### 2.2 可用資料源
```bash
# 統計結果
總共 46 個 JSONL 檔案
最大檔案: data/raw/ltn_14days.jsonl (8,227 篇文章)
總計可用: ~10,000+ 篇文章
```

**檔案分佈**:
- `ltn_14days.jsonl`: 8,227 篇 (自由時報 14 天資料)
- 其他小型測試檔案: ~100 篇
- 正在收集中的資料: 可能更多

#### 2.3 大型索引建立狀態
以下索引建立任務正在進行中但尚未完成:
- `data/index_50k/` - 目標 50,000 篇 (僅有 meta 目錄)
- `data/index_50k_clean/` - 目標 50,000 篇 (僅有 meta 目錄)

**建立進度**: 需檢查 `/tmp/build_50k_*.log` 了解進度

---

## 改善建議

### 選項 1: 使用現有 LTN 資料建立中型索引 (推薦 ✨)

**優點**:
- 快速建立 (~5-10 分鐘)
- 8,227 篇文章足以測試 Faceted Search 功能
- 可立即使用

**步驟**:
```bash
# 1. 停止目前的 Flask 應用
pkill -f "python app_simple.py"

# 2. 建立 LTN 索引
source activate ai_env
python scripts/search_news.py --build \
  --data-file data/raw/ltn_14days.jsonl \
  --index-dir data/indexes_ltn_8k \
  --ckip-model bert-base

# 3. 更新 app_simple.py 配置
# 修改第 51 行為:
# app.config['INDEX_DIR'] = project_root / 'data' / 'indexes_ltn_8k'

# 4. 重新啟動 Flask
python app_simple.py
```

**預期結果**: 系統將顯示 8,227 筆文檔

---

### 選項 2: 等待 50k 索引完成

**優點**:
- 最完整的資料集
- 最佳測試環境

**缺點**:
- 可能需要數小時完成
- 需要持續監控進度

**檢查進度**:
```bash
# 檢查索引建立日誌
tail -f /tmp/build_50k_clean.log

# 檢查索引檔案
ls -lh data/index_50k_clean/

# 檢查處理程序
ps aux | grep search_news.py
```

**完成後步驟**:
```bash
# 更新 app_simple.py 配置
# 修改第 51 行為:
# app.config['INDEX_DIR'] = project_root / 'data' / 'index_50k_clean'

# 重新啟動 Flask
pkill -f "python app_simple.py"
python app_simple.py
```

---

### 選項 3: 合併所有 raw 資料建立完整索引

**優點**:
- 使用所有可用資料 (~10,000+ 篇)
- 平衡建立時間與資料量

**步驟**:
```bash
source activate ai_env

# 建立完整索引
python scripts/search_news.py --build \
  --data-dir data/raw \
  --index-dir data/indexes_full \
  --ckip-model bert-base

# 更新配置並重啟 (同上)
```

**預計時間**: ~15-30 分鐘

---

## 當前系統狀態總結

### ✅ 已完成項目
1. ✅ Faceted Search 後端引擎實作完成
   - `src/ir/facet/facet_engine.py`
   - `src/ir/facet/facet_filter.py`
2. ✅ 單元測試通過 (31/31)
3. ✅ API 端點實作完成
   - `/api/facets`
   - `/api/search/faceted`
4. ✅ API 端點測試通過
5. ✅ 前端 UI 組件完成
   - `static/js/facet.js`
   - `static/css/facet.css`
6. ✅ HTML 整合完成
7. ✅ Flask 依賴問題修復
8. ✅ JavaScript `.toUpperCase()` 錯誤修復

### ⚠️ 待解決項目
1. ⚠️ **索引資料量不足** (目前 121 篇，可用 8,227 篇)
2. ⚠️ **需要重新配置 INDEX_DIR** 指向較大索引
3. ⏳ **大型索引建立中** (50k 索引尚未完成)

### 🔧 建議下一步

**立即行動** (推薦):
```bash
# 方案: 使用 LTN 資料快速建立可用索引
source activate ai_env

# 建立 LTN 8k 索引
python scripts/search_news.py --build \
  --data-file data/raw/ltn_14days.jsonl \
  --index-dir data/indexes_ltn_8k \
  --ckip-model bert-base

# 等待完成後檢查
ls -lh data/indexes_ltn_8k/
```

然後修改 `app_simple.py` 第 51 行:
```python
app.config['INDEX_DIR'] = project_root / 'data' / 'indexes_ltn_8k'
```

重啟 Flask:
```bash
pkill -f "python app_simple.py"
source activate ai_env
python app_simple.py
```

---

## 測試 Faceted Search

索引建立完成並重啟 Flask 後:

### 1. 開啟瀏覽器
訪問: `http://localhost:5000`

### 2. 執行搜尋
- 輸入查詢: 「台灣經濟」
- 點擊「搜尋」按鈕

### 3. 使用 Faceted Search
- 點擊「🔽 進階篩選」按鈕
- 應該看到以下 facet 群組:
  - 📰 新聞來源
  - 🏷️ 分類
  - 📅 發布月份
  - ✍️ 作者

### 4. 測試篩選功能
- 勾選任一 facet 值 (例如: 中央社)
- 搜尋結果應自動更新
- 已套用篩選標籤應顯示在頂部
- 結果統計應更新為「找到 X 筆結果 (共 Y 筆)」

### 5. 測試多重篩選
- 同時勾選多個 facet
- 驗證 AND 邏輯正確運作
- 測試移除單一篩選功能
- 測試「清除所有篩選」按鈕

---

## 效能基準 (8k 文檔)

預期效能:
- **Facet 計算**: < 200ms
- **篩選查詢**: < 100ms
- **端到端請求**: < 300ms
- **前端渲染**: < 50ms

---

## 故障排除

### 問題: Flask 無法啟動
```bash
# 檢查端口佔用
lsof -i :5000

# 強制停止舊進程
pkill -9 -f "python app_simple.py"
```

### 問題: 索引建立失敗
```bash
# 檢查日誌
tail -100 /tmp/build_*.log

# 檢查資料檔案
head data/raw/ltn_14days.jsonl
```

### 問題: Facet 沒有顯示
1. 開啟瀏覽器開發者工具 (F12)
2. 檢查 Console 錯誤訊息
3. 檢查 Network 標籤，確認 `/api/facets` 請求成功
4. 確認回傳的 JSON 包含 `"success": true`

---

## 聯絡資訊

**文件建立日期**: 2025-11-21
**修復狀態**: JavaScript 錯誤已修復 | 索引問題需用戶選擇解決方案
**預計總完成時間**:
- 選項 1 (LTN 8k): ~10 分鐘
- 選項 2 (等待 50k): 數小時
- 選項 3 (完整索引): ~30 分鐘
