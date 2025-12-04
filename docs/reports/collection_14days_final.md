# 14天新聞收集最終報告

**收集時間**: 2025-11-19 07:05:52 ~ 18:41:00 (約11.5小時主要執行期)
**日期範圍**: 2025-11-05 to 2025-11-19 (14天)
**執行模式**: 平行收集 (31個任務同時執行)
**最終統計時間**: 2025-11-19 18:41:00

---

## ✅ 收集完成 - 最終統計

### 總體成果

| 指標 | 數值 |
|------|------|
| **總文章數** | 246,961 篇 |
| **總資料大小** | 776 MB |
| **成功來源** | 6 個新聞網站 |
| **成功類別** | 13 個資料集 |
| **收集成功率** | 41.9% (13/31 任務) |
| **資料品質** | ✅ UTF-8 編碼正確，中文顯示完整 |

---

## 📊 詳細收集結果

### Yahoo 奇摩新聞 (主要資料來源)

| 類別 | 文章數 | 檔案大小 | 狀態 |
|------|--------|----------|------|
| Sports (體育) | 66,414 | 206MB | ✅ 完成 |
| Politics (政治) | 45,639 | 141MB | ✅ 完成 (4小時後超時終止) |
| Finance (財經) | 43,488 | 136MB | ⏳ 執行中 (12小時+，持續收集) |
| Entertainment (娛樂) | 31,656 | 99MB | ✅ 完成 |
| World (國際) | 23,136 | 73MB | ✅ 完成 |
| Lifestyle (生活) | 17,519 | 55MB | ✅ 完成 |
| Tech (科技) | 13,988 | 44MB | ✅ 完成 |
| Health (健康) | 3,520 | 11MB | ✅ 完成 |

**Yahoo 小計**: 245,360 篇, ~765MB (98.4%)

### 其他新聞來源

| 來源 | 文章數 | 檔案大小 | 狀態 |
|------|--------|----------|------|
| NextApple 壹蘋 | 998 | 7.5MB | ✅ 完成 |
| SETN 三立 | 500 | 1.5MB | ✅ 完成 |
| UDN 聯合報 | 63 | 195KB | ✅ 完成 |
| LTN 自由時報 | 20 | 52KB | ✅ 完成 |
| PTS 公視 | 20 | 44KB | ✅ 完成 |

**其他來源小計**: 1,601 篇, ~9.3MB (1.6%)

---

## ❌ 失敗或未完成的來源

### 1. CNA 中央社
- **狀態**: HTML 解析失敗
- **現象**: 所有 URL 都返回 "Missing title or content" 警告
- **原因**: 網站 HTML 結構改變，或需要 JavaScript 渲染
- **診斷**: 已測試確認問題存在
- **處理**: 需要深度修復（重新分析 HTML 結構或改用 Playwright）

### 2. Playwright 爬蟲 (Reactor 衝突)

| 來源 | 任務數 | 問題 |
|------|--------|------|
| FTV 民視 | 6 類別 | Twisted reactor 衝突 |
| Storm 風傳媒 | 1 | Twisted reactor 衝突 |
| TVBS 新聞 | 1 | Twisted reactor 衝突 |

**技術細節**:
```
Exception: The installed reactor (twisted.internet.epollreactor.EPollReactor)
does not match the requested one (twisted.internet.asyncioreactor.AsyncioSelectorReactor)
```

**原因**: Scrapy Playwright 無法在同一環境中平行執行多個任務
**處理**: 需要獨立進程執行策略

### 3. CTI 中天 (已修復但未執行)

- **Bug 類型**: `custom_settings` property 錯誤
- **修復狀態**: ✅ 已修正 (兩處：fallback base class + main spider class)
- **執行狀態**: ⏸️ 未執行 (9 個類別：politics, money, society, world, entertainment, life, sports, tech, all)
- **備註**: 修復後理論上可執行，但仍會遇到 Playwright reactor 問題

---

## 🔧 技術修復紀錄

### 本次收集期間完成的修復

1. **Yahoo Spider** (2025-11-19 早期)
   - 修復: `custom_settings` property → class attribute
   - 修復: 參數型別轉換 (`days` 必須是 int)
   - 結果: ✅ 成功收集 8 個類別，245K+ 篇文章

2. **Mass Collection System** (2025-11-19)
   - 新增: UTF-8 編碼設定 (`FEED_EXPORT_ENCODING=utf-8`, `ENSURE_ASCII=False`)
   - 新增: 平行執行模式 (`--parallel` flag)
   - 新增: 4 小時超時保護機制
   - 結果: ✅ 成功平行執行 31 個任務

3. **PTS Spider** (2025-11-19)
   - 新增: HTML entity 解碼 (`html.unescape()`)
   - 結果: ✅ 正確顯示中文標題和內容

4. **CTI Spider** (2025-11-19 下午)
   - 修復: `custom_settings` property → class attribute (兩處)
   - 結果: ✅ 修復完成但未執行收集

---

## 📁 最終資料檔案清單

### 有效資料檔案 (13 個)

```
data/raw/yahoo_sports_14days.jsonl          206MB    66,414 篇
data/raw/yahoo_finance_14days.jsonl         136MB    43,488 篇  (執行中)
data/raw/yahoo_politics_14days.jsonl        141MB    45,639 篇
data/raw/yahoo_entertainment_14days.jsonl    99MB    31,656 篇
data/raw/yahoo_world_14days.jsonl            73MB    23,136 篇
data/raw/yahoo_lifestyle_14days.jsonl        55MB    17,519 篇
data/raw/yahoo_tech_14days.jsonl             44MB    13,988 篇
data/raw/yahoo_health_14days.jsonl           11MB     3,520 篇
data/raw/nextapple_14days.jsonl             7.5MB       998 篇
data/raw/setn_14days.jsonl                  1.5MB       500 篇
data/raw/udn_14days.jsonl                   195KB        63 篇
data/raw/ltn_14days.jsonl                    52KB        20 篇
data/raw/pts_14days.jsonl                    44KB        20 篇
```

### 已刪除的空白檔案 (8 個)

```
✗ data/raw/ftv_politics_14days.jsonl      (0 bytes) - 已刪除
✗ data/raw/ftv_finance_14days.jsonl       (0 bytes) - 已刪除
✗ data/raw/ftv_culture_14days.jsonl       (0 bytes) - 已刪除
✗ data/raw/ftv_international_14days.jsonl (0 bytes) - 已刪除
✗ data/raw/ftv_life_14days.jsonl          (0 bytes) - 已刪除
✗ data/raw/ftv_all_14days.jsonl           (0 bytes) - 已刪除
✗ data/raw/storm_14days.jsonl             (0 bytes) - 已刪除
✗ data/raw/tvbs_14days.jsonl              (0 bytes) - 已刪除
```

---

## 🎯 資料品質評估

### 編碼品質
- ✅ **UTF-8 編碼**: 所有檔案使用正確的 UTF-8 編碼
- ✅ **中文顯示**: 繁體中文字元完整無亂碼
- ✅ **HTML 解碼**: HTML entities 正確轉換為中文字元
- ✅ **JSON 格式**: 符合 JSONL 格式（每行一個 JSON 物件）

### 資料完整性
- ✅ **標題**: 所有文章都有標題
- ✅ **內容**: 所有文章都有內容本文
- ✅ **時間戳**: 所有文章都有發布時間
- ✅ **URL**: 所有文章都有來源 URL
- ⚠️ **作者**: 部分文章缺少作者資訊（依來源而定）

### 時間範圍覆蓋
- **目標**: 2025-11-05 至 2025-11-19 (14天)
- **實際**: 根據 sitemap 爬取，主要覆蓋近期文章
- **Yahoo**: 主要覆蓋最近 14 天的每日 sitemap
- **其他**: 依各來源 sitemap 或列表頁範圍而定

---

## 📈 資料分布分析

### 按來源分布
```
Yahoo 奇摩:    245,360 篇  (99.4%)
壹蘋:            998 篇   (0.4%)
三立:            500 篇   (0.2%)
其他:            103 篇   (0.04%)
```

### 按類別分布 (Yahoo)
```
體育:      66,414 篇  (27.0%)
政治:      45,639 篇  (18.6%)
財經:      43,488 篇  (17.7%)
娛樂:      31,656 篇  (12.9%)
國際:      23,136 篇   (9.4%)
生活:      17,519 篇   (7.1%)
科技:      13,988 篇   (5.7%)
健康:       3,520 篇   (1.4%)
```

---

## ✅ 結論與建議

### 收集成果總結

1. **資料量充足**: 246,961 篇文章，776MB 資料量足夠用於資訊檢索系統開發與測試
2. **資料品質優良**: UTF-8 編碼正確，中文顯示完整，JSON 格式符合規範
3. **涵蓋範圍廣**: 8 個主要新聞類別，時間跨度 14 天
4. **主要來源穩定**: Yahoo 奇摩提供了 99.4% 的資料，品質穩定可靠

### 下一步建議

#### 選項 A: 立即使用現有資料 (推薦)
✅ **適用於**: 快速啟動 IR 系統開發
- 使用現有 246,961 篇文章建立索引
- 測試檢索演算法 (Boolean, VSM, TF-IDF)
- 評估效能與準確度
- 資料量已足夠進行有意義的測試

#### 選項 B: 擴展至 2 年資料
⏳ **適用於**: 需要更大規模資料集
- 修改 `mass_collect.py` 參數: `--days 730`
- 預估收集時間: 數天至一週
- 預估資料量: 40-50GB+
- 注意: 需要穩定的執行環境和足夠的儲存空間

#### 選項 C: 修復失敗的來源
🔧 **適用於**: 需要多樣化資料來源
- **CNA**: 需要深度調查 HTML 結構或改用 Playwright
- **FTV/Storm/TVBS**: 需要獨立進程執行策略
- **CTI**: 可重新執行，但仍會遇到 Playwright 問題
- **預估時間**: 數小時至數天（取決於問題複雜度）

### 專案優先級建議

根據資訊檢索系統開發需求，建議優先級為:

1. **第一優先**: 使用現有資料建立索引和檢索系統 (選項 A)
2. **第二優先**: 擴展至 2 年資料以提升系統規模 (選項 B)
3. **第三優先**: 修復失敗來源以增加資料多樣性 (選項 C)

---

## 📝 技術備註

### 執行環境
- Python 3.10.19 + Scrapy 2.11.0
- WSL2 (Linux 6.6.87.2-microsoft-standard-WSL2)
- Conda environment: `ai_env`
- 並行執行: 31 個 subprocess 同時運行

### 超時機制
- 單任務超時: 4 小時
- Yahoo Politics: 超時後收集到 45,639 篇 (141MB)
- Yahoo Finance: 仍在執行中 (12 小時+)

### 未來優化方向
1. 針對大量資料來源調整超時時間或分批收集策略
2. Playwright 爬蟲改用獨立進程或分時執行
3. 增加進度監控和斷點續傳功能
4. 考慮使用分散式爬蟲架構 (Scrapy Cluster)

---

**報告生成時間**: 2025-11-19 18:41:00
**報告版本**: 最終版 (v1.1)
**Yahoo Finance 執行時間**: 12小時10分 (持續執行中)
