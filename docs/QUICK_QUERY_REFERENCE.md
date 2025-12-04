# IR 系統查詢快速參考 (Quick Query Reference)

快速查詢參考指南 - 常用查詢範例與技巧

---

## 📖 查詢語法速查表

### 簡單查詢
```
台灣 經濟               → 關鍵字查詢
人工智慧                → 單一詞彙
COVID-19 疫苗           → 支援英數字
```

### Boolean 查詢
```
台灣 AND 經濟           → 必須同時包含
經濟 OR 金融            → 包含任一即可
疫苗 AND NOT 副作用     → 排除特定詞
(台灣 OR 中國) AND 貿易 → 複雜組合
```

### 欄位查詢
```
title:AI                → 標題包含 AI
category:政治           → 政治類新聞
source:ltn              → 自由時報
title:台灣 AND category:經濟 → 多欄位組合
```

---

## 🎯 常用查詢範例

### 經濟財經類

```bash
# 台灣經濟發展
台灣 經濟 發展

# 股市相關
台股 OR 股市 OR 大盤

# 科技產業
台積電 OR TSMC OR 半導體

# 房地產
房價 OR 房市 OR 不動產
```

### 政治社會類

```bash
# 選舉相關
選舉 投票

# 政策法規
政策 AND (修法 OR 立法)

# 兩岸關係
兩岸 OR 海峽 OR 中國

# 地方政治
title:市長 OR title:縣長
```

### 科技創新類

```bash
# 人工智慧
AI OR 人工智慧 OR ChatGPT

# 5G 通訊
5G AND (基地台 OR 網路)

# 電動車
電動車 OR 特斯拉 OR Tesla

# 元宇宙
元宇宙 OR metaverse OR VR
```

### 醫療健康類

```bash
# 疫情相關
COVID-19 OR 新冠 OR 疫情

# 疫苗接種
疫苗 AND 接種

# 健保醫療
健保 OR 醫療 OR 醫院

# 排除副作用新聞
疫苗 AND NOT (副作用 OR 不良反應)
```

### 環境生態類

```bash
# 氣候變遷
氣候 OR 暖化 OR 減碳

# 能源議題
綠能 OR 太陽能 OR 風電

# 環保政策
環保 AND 政策

# 空污問題
空污 OR PM2.5 OR 空氣品質
```

---

## 🔍 進階查詢技巧

### 1. 精確標題搜尋

```bash
# 標題必須包含特定詞
title:台灣 AND title:經濟

# 標題 + 內容組合
title:AI AND (應用 OR 發展)

# 特定來源的標題搜尋
source:ltn AND title:選舉
```

### 2. 分類篩選

```bash
# 政治類
category:政治

# 財經類
category:財經 OR category:經濟

# 排除特定分類
台灣 AND NOT category:娛樂
```

### 3. 多媒體來源組合

```bash
# 主流媒體
source:ltn OR source:udn OR source:chinatimes

# 特定媒體的特定主題
source:yahoo AND category:科技

# 來源 + 關鍵字
source:ltn AND 選舉 AND 政見
```

### 4. 時事追蹤查詢

```bash
# 重大政策
政策 AND (宣布 OR 公布 OR 修正)

# 重要會議
title:記者會 OR title:說明會

# 突發新聞
title:快訊 OR title:即時

# 專訪報導
title:專訪 OR title:獨家
```

---

## 💡 查詢優化建議

### ✅ 好的查詢方式

```bash
# 1. 具體明確的關鍵字
台積電 擴廠 投資         ✅

# 2. 適當使用 Boolean
台灣 AND 經濟 AND NOT 股市  ✅

# 3. 結合欄位查詢
title:AI AND category:科技    ✅

# 4. 同義詞組合
人工智慧 OR AI OR 機器學習   ✅
```

### ❌ 應避免的查詢

```bash
# 1. 過於寬鬆的單字查詢
的                    ❌ (停用詞)
台                    ❌ (太短)

# 2. 過度複雜的 Boolean
((A AND B) OR (C AND NOT (D OR E))) AND F  ❌

# 3. 錯誤的語法
title: AI             ❌ (冒號後有空格)
Title:AI              ❌ (大寫欄位名)

# 4. 過多關鍵字
台灣 經濟 政治 社會 文化 教育 科技 醫療  ❌
```

---

## 📊 查詢模式選擇指南

| 需求 | 推薦模式 | 範例查詢 |
|------|---------|---------|
| 一般關鍵字搜尋 | SIMPLE | `台灣 經濟` |
| 精確邏輯控制 | BOOLEAN | `A AND B OR C` |
| 針對特定欄位 | FIELD | `title:AI` |
| 不確定查詢類型 | AUTO | 任何查詢 |

---

## 🎨 排序模型選擇

| 情境 | 推薦模型 | 說明 |
|------|---------|------|
| 一般新聞檢索 | BM25 | 平衡速度與品質 |
| 找相似文章 | VSM | 向量相似度 |
| 追求最佳結果 | HYBRID | BM25 + VSM |

---

## 🚀 快速上手命令

### 命令列查詢

```bash
# 基本查詢
python scripts/search_news.py --query "台灣 經濟"

# Boolean 查詢
python scripts/search_news.py --query "AI AND 應用" --mode boolean

# 欄位查詢
python scripts/search_news.py --query "title:疫情" --mode field

# 指定排序模型
python scripts/search_news.py --query "科技" --model VSM

# 增加結果數量
python scripts/search_news.py --query "台灣" --top-k 20
```

### 互動式查詢

```bash
# 啟動互動模式
python scripts/search_news.py --index-dir data/index_50k

# 執行 Demo
python scripts/demo_ir_system.py --index-dir data/index_50k

# Web 介面
python app_simple.py
```

---

## 📝 查詢範例集

### 新聞時事追蹤

```yaml
熱門話題:
  - "選舉 AND 辯論"
  - "疫苗 AND 接種率"
  - "AI AND ChatGPT"

政策法規:
  - "政策 AND (修正 OR 通過)"
  - "title:修法 AND category:政治"

經濟指標:
  - "GDP OR 經濟成長"
  - "失業率 OR 就業"
  - "CPI OR 物價"
```

### 專題研究查詢

```yaml
科技趨勢:
  - "人工智慧 AND 應用 AND NOT 風險"
  - "(5G OR 6G) AND 發展"
  - "元宇宙 AND (商機 OR 應用)"

社會議題:
  - "少子化 OR 高齡化"
  - "房價 AND (上漲 OR 下跌)"
  - "環保 AND 政策"

國際關係:
  - "兩岸 AND (貿易 OR 經濟)"
  - "美中 AND (關係 OR 競爭)"
  - "title:外交 AND category:國際"
```

---

## 🔧 疑難排解

### 沒有搜尋結果？

```bash
# 1. 檢查拼字和語法
title:AI        ✅
title: AI       ❌ (多了空格)

# 2. 嘗試更寬鬆的查詢
台灣 AND 經濟  →  台灣 OR 經濟

# 3. 使用同義詞
人工智慧  →  AI OR 人工智慧 OR 機器學習

# 4. 檢查索引狀態
python scripts/search_news.py --stats
```

### 結果太多？

```bash
# 1. 增加限制條件
台灣  →  台灣 AND 經濟 AND 政策

# 2. 使用欄位查詢
經濟  →  title:經濟

# 3. 排除不相關
經濟 AND NOT 股市

# 4. 減少 top-k
--top-k 100  →  --top-k 10
```

### 結果不相關？

```bash
# 1. 使用 Boolean 精確控制
台灣 經濟  →  台灣 AND 經濟

# 2. 欄位查詢限縮範圍
AI  →  title:AI AND category:科技

# 3. 嘗試不同排序模型
--model BM25  →  --model HYBRID

# 4. 增加更多關鍵字
AI  →  AI AND 應用 AND 發展
```

---

## 📚 相關資源

- **完整使用指南**: `docs/guides/IR_SYSTEM_USER_GUIDE.md`
- **測試查詢集**: `tests/test_queries.yaml`
- **互動式 Demo**: `python scripts/demo_ir_system.py`
- **Web 介面**: `python app_simple.py`

---

**最後更新**: 2025-11-20
**版本**: v1.0

**快樂搜尋！Happy Searching! 🔍**
