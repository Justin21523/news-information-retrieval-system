# Mass News Collection System

專業、彈性、可擴充的大規模新聞收集系統

## 功能特色

### 🎯 彈性日期範圍
- **天數**: `--days 14` (最近 14 天)
- **月數**: `--months 3` (最近 3 個月)
- **年數**: `--years 1` (最近 1 年)
- **指定範圍**: `--date-range 2024-01-01 2024-12-31`

### 📰 新聞網選擇
- **全部**: 預設收集所有 11 個新聞網
- **指定來源**: `--sources cna,pts,ltn,udn`
- **排除來源**: `--exclude-sources storm,tvbs,ftv,cti`
- **速度篩選**: `--speed-filter fast` (只收集快速來源)

### 🏷️ 類別篩選
- **支援類別的新聞網**:
  - Yahoo: politics, world, entertainment, sports, finance, tech, health, lifestyle
  - FTV: politics, finance, culture, international, life, all
  - CTI: politics, money, society, world, entertainment, life, sports, tech, all
- **使用方式**: `--categories politics,finance`

### ⚙️ 執行選項
- **CPU 優先級**: `--nice 10` (降低 CPU 優先級，0-19，越高越低優先)
- **背景執行**: `--background` (在背景執行並記錄到 log 檔)
- **乾跑模式**: `--dry-run` (只顯示命令不執行)
- **日誌等級**: `--log-level INFO|ERROR|WARNING|DEBUG`

## 使用範例

### 基本使用

```bash
# 收集最近 14 天，所有新聞網
python scripts/crawlers/mass_collect.py --days 14

# 收集最近 30 天，指定新聞網
python scripts/crawlers/mass_collect.py --days 30 --sources cna,pts,ltn,udn

# 收集最近 3 個月
python scripts/crawlers/mass_collect.py --months 3

# 收集 1 年資料
python scripts/crawlers/mass_collect.py --years 1
```

### 進階使用

```bash
# 只收集快速來源（不包含 Playwright 爬蟲）
python scripts/crawlers/mass_collect.py --days 14 --speed-filter fast

# 排除最慢的來源
python scripts/crawlers/mass_collect.py --days 14 --exclude-sources cti

# 指定日期範圍
python scripts/crawlers/mass_collect.py --date-range 2024-01-01 2024-12-31

# 收集特定類別（只從支援的新聞網）
python scripts/crawlers/mass_collect.py --days 14 --sources ftv,cti --categories politics,finance
```

### 資源控制

```bash
# 降低 CPU 優先級避免影響其他進程
python scripts/crawlers/mass_collect.py --days 14 --nice 10

# 在背景執行
python scripts/crawlers/mass_collect.py --days 14 --nice 10 --background

# 乾跑模式（預覽命令）
python scripts/crawlers/mass_collect.py --days 14 --dry-run
```

## 新聞網速度分類

### Fast (快速 - 無需 Playwright)
- CNA 中央社
- PTS 公視
- LTN 自由時報
- UDN 聯合報
- NextApple 壹蘋
- SETN 三立
- Yahoo 奇摩

### Slow (慢速 - 需要 Playwright)
- Storm 風傳媒
- TVBS 新聞
- FTV 民視

### Very Slow (極慢 - Playwright + Cloudflare)
- CTI 中天

## 輸出檔案

預設輸出到 `data/raw/` 目錄，檔名格式：
- `{source}_{suffix}.jsonl`
- 例如：`cna_14days.jsonl`, `yahoo_3months.jsonl`

自訂輸出目錄：
```bash
python scripts/crawlers/mass_collect.py --days 14 --output-dir data/custom
```

自訂檔名後綴：
```bash
python scripts/crawlers/mass_collect.py --days 14 --output-suffix 2024Q1
# 輸出：cna_2024Q1.jsonl
```

## 實際應用場景

### 場景 1：快速每日更新
```bash
# 只收集快速來源的最近 1 天
python scripts/crawlers/mass_collect.py --days 1 --speed-filter fast
```

### 場景 2：每週完整收集
```bash
# 每週收集所有來源最近 7 天
python scripts/crawlers/mass_collect.py --days 7
```

### 場景 3：月度完整歷史
```bash
# 每月初收集上個月完整資料
python scripts/crawlers/mass_collect.py --months 1
```

### 場景 4：研究特定主題
```bash
# 收集政治和財經類別資料
python scripts/crawlers/mass_collect.py --days 30 \
    --sources ftv,cti,yahoo \
    --categories politics,finance,money
```

### 場景 5：建立歷史資料庫
```bash
# 收集 2 年完整歷史（排除最慢的 CTI）
python scripts/crawlers/mass_collect.py --years 2 \
    --exclude-sources cti \
    --nice 15 \
    --background
```

## 效能預估

### 14 天收集 (所有 11 個來源)
- **快速來源** (7個): ~30-60 分鐘
- **慢速來源** (3個): ~1-2 小時
- **極慢來源** (1個): ~2-4 小時
- **總計**: 約 3-7 小時

### 1 個月收集
- **總計**: 約 6-14 小時

### 3 個月收集
- **總計**: 約 18-40 小時

### 1 年收集
- **總計**: 約 3-7 天 (建議分批執行)

## 最佳實踐

### 避免影響 GPU 資源
```bash
# 使用 nice 降低優先級
python scripts/crawlers/mass_collect.py --days 14 --nice 15
```

### 大規模長時間收集
```bash
# 使用 nohup 和背景執行
nohup python scripts/crawlers/mass_collect.py --years 1 --nice 15 &
```

### 分階段收集
```bash
# 第一階段：快速來源
python scripts/crawlers/mass_collect.py --days 30 --speed-filter fast

# 第二階段：慢速來源
python scripts/crawlers/mass_collect.py --days 30 --exclude-sources cna,pts,ltn,udn,nextapple,setn,yahoo
```

## 故障排除

### 問題：Playwright 錯誤
```bash
# 重新安裝 Playwright
playwright install chromium
playwright install-deps
```

### 問題：記憶體不足
```bash
# 分批收集或排除慢速來源
python scripts/crawlers/mass_collect.py --days 14 --speed-filter fast
```

### 問題：CTI Cloudflare 失敗
```bash
# 排除 CTI 或單獨處理
python scripts/crawlers/mass_collect.py --days 14 --exclude-sources cti
```

## 開發者資訊

- **版本**: 1.0
- **作者**: Information Retrieval System
- **日期**: 2025-11-19
- **Python 版本**: 3.8+
- **依賴**: scrapy, scrapy-playwright (選用)
