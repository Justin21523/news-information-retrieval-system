# 新聞爬蟲測試套件

完整的自動化測試系統，用於持續監控和驗證新聞爬蟲的健康狀態。

---

## 📋 目錄結構

```
tests/crawlers/
├── README.md                    # 本文件
├── conftest.py                  # pytest 配置和 fixtures
├── test_crawlers_unit.py        # 單元測試
├── integration/                 # 整合測試（待實作）
└── fixtures/                    # 測試數據

scripts/crawlers/
├── health_check.py              # 健康檢查系統 ⭐
├── test_single_crawler.py       # 單一爬蟲測試
└── test_all_crawlers.py         # 批量測試
```

---

## 🚀 快速開始

### 1. 安裝依賴

```bash
# 確保已安裝 pytest
pip install pytest pytest-cov pytest-html

# 確保已安裝爬蟲依賴
pip install -r requirements.txt
playwright install chromium
```

### 2. 運行測試

```bash
# 運行所有單元測試
pytest tests/crawlers/test_crawlers_unit.py -v

# 運行特定測試
pytest tests/crawlers/test_crawlers_unit.py::TestCrawlerInitialization -v

# 生成覆蓋率報告
pytest tests/crawlers/ --cov=scripts.crawlers --cov-report=html

# 生成 HTML 測試報告
pytest tests/crawlers/ --html=test_report.html --self-contained-html
```

---

## 🏥 健康檢查系統

### 基本用法

```bash
# 檢查所有爬蟲
python scripts/crawlers/health_check.py

# 快速檢查（每個爬蟲1項）
python scripts/crawlers/health_check.py --quick

# 檢查特定爬蟲
python scripts/crawlers/health_check.py --crawlers chinatimes,ettoday

# 生成 HTML 報告
python scripts/crawlers/health_check.py --html-report

# 生成 JSON 報告
python scripts/crawlers/health_check.py --json-report
```

### 輸出範例

**終端輸出**:
```
======================================================================
Crawler Health Check Summary
======================================================================
Timestamp: 2025-11-18T21:30:00
Test Items: 1

Total Crawlers: 9
✓ Healthy: 6
✗ Unhealthy: 2
− Skipped: 1

Overall Health: 66.7%

Detailed Results:
----------------------------------------------------------------------
✓ CNA中央社              | Items:  1 | Time:  15.2s | Working normally
✓ 中時新聞網             | Items:  1 | Time:  12.5s | Working normally
✓ 東森新聞雲             | Items:  1 | Time:  45.3s | Working normally
✗ TVBS新聞               | Items:  0 | Time: 180.0s | Timeout (> 3 minutes)
− TVBS新聞               | Items:  0 | Time:   0.0s | Crawler marked as skip
======================================================================
```

**HTML 報告**: 美觀的網頁報告，包含：
- 總體健康百分比進度條
- 彩色狀態卡片
- 詳細測試結果表格
- 自動更新時間戳

---

## 🧪 單元測試

### 測試類別

#### 1. TestCrawlerInitialization
測試爬蟲初始化和配置。

```bash
pytest tests/crawlers/test_crawlers_unit.py::TestCrawlerInitialization -v
```

**測試項目**:
- ✓ 預設參數初始化
- ✓ 自訂參數初始化
- ✓ 日期範圍配置

#### 2. TestCrawlerUtilities
測試爬蟲工具方法。

```bash
pytest tests/crawlers/test_crawlers_unit.py::TestCrawlerUtilities -v
```

**測試項目**:
- ✓ 文章 ID 生成一致性
- ✓ 文字清理功能
- ✓ 日期解析（多種格式）

#### 3. TestArticleValidation
測試文章數據驗證。

```bash
pytest tests/crawlers/test_crawlers_unit.py::TestArticleValidation -v
```

**測試項目**:
- ✓ 必要欄位完整性
- ✓ 內容最小長度（100字）
- ✓ URL 格式驗證
- ✓ 日期格式驗證（YYYY-MM-DD）

#### 4. TestCrawlerConfiguration
測試爬蟲配置設定。

```bash
pytest tests/crawlers/test_crawlers_unit.py::TestCrawlerConfiguration -v
```

**測試項目**:
- ✓ custom_settings 存在
- ✓ robots.txt 遵守設定
- ✓ Playwright 配置正確

### 參數化測試

使用 `pytest.mark.parametrize` 測試多個爬蟲：

```python
@pytest.mark.parametrize("crawler_name", [
    'cna', 'ltn', 'pts', 'chinatimes', 'ettoday'
])
def test_crawler_init_default(self, crawler_name, crawler_registry):
    # 測試代碼
```

---

## 🏷️ 測試標記 (Markers)

使用標記過濾測試：

```bash
# 只運行單元測試
pytest -m unit

# 跳過慢速測試
pytest -m "not slow"

# 只運行 Playwright 測試
pytest -m playwright

# 運行整合測試
pytest -m integration
```

**可用標記**:
- `@pytest.mark.unit` - 單元測試
- `@pytest.mark.integration` - 整合測試
- `@pytest.mark.slow` - 慢速測試
- `@pytest.mark.playwright` - 需要 Playwright 的測試

---

## 📊 測試報告

### HTML 覆蓋率報告

```bash
pytest tests/crawlers/ \
    --cov=scripts.crawlers \
    --cov-report=html \
    --cov-report=term

# 打開報告
open htmlcov/index.html  # Mac
xdg-open htmlcov/index.html  # Linux
```

### pytest-html 報告

```bash
pytest tests/crawlers/ \
    --html=reports/test_report.html \
    --self-contained-html

# 打開報告
open reports/test_report.html
```

---

## 🔄 持續集成 (CI/CD)

### GitHub Actions 範例

在 `.github/workflows/test-crawlers.yml`:

```yaml
name: Crawler Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        playwright install chromium

    - name: Run unit tests
      run: |
        pytest tests/crawlers/test_crawlers_unit.py -v

    - name: Run health check
      run: |
        python scripts/crawlers/health_check.py --quick --json-report

    - name: Upload reports
      uses: actions/upload-artifact@v2
      with:
        name: test-reports
        path: data/health_check/
```

### Cron Job 定期檢查

```bash
# 每天早上 8 點運行健康檢查
0 8 * * * cd /path/to/project && python scripts/crawlers/health_check.py --html-report
```

---

## 📝 Fixtures 使用

### 共享 Fixtures

在 `conftest.py` 中定義：

```python
@pytest.fixture(scope="session")
def test_config():
    """測試配置"""
    return {
        'test_days': 1,
        'test_items': 2,
        'timeout': 120,
    }

@pytest.fixture
def sample_article():
    """範例文章數據"""
    return {...}
```

### 使用方式

```python
def test_example(test_config, sample_article):
    days = test_config['test_days']
    title = sample_article['title']
    # 測試邏輯
```

---

## 🔍 常見問題

### Q1: 測試執行很慢？

**A**: 使用標記跳過慢速測試：
```bash
pytest -m "not slow" -v
```

### Q2: Playwright 測試失敗？

**A**: 確保瀏覽器已安裝：
```bash
playwright install chromium
```

### Q3: 如何只測試特定爬蟲？

**A**: 使用參數化過濾：
```bash
pytest -k "chinatimes" -v
```

### Q4: 測試覆蓋率太低？

**A**: 添加更多測試案例，目標 >80%：
```bash
pytest --cov=scripts.crawlers --cov-report=term-missing
```

---

## 📈 維護指南

### 添加新爬蟲測試

1. 在 `conftest.py` 的 `crawler_registry` 添加配置
2. 參數化測試會自動包含新爬蟲
3. 運行測試驗證：
   ```bash
   pytest tests/crawlers/ -v
   ```

### 更新健康檢查

1. 編輯 `scripts/crawlers/health_check.py`
2. 更新 `crawler_registry`
3. 測試：
   ```bash
   python scripts/crawlers/health_check.py --quick
   ```

### 監控測試趨勢

1. 定期運行健康檢查
2. 保存 JSON 報告
3. 分析趨勢（可寫腳本解析 JSON）

---

## 🎯 最佳實踐

1. **定期運行**: 每天至少運行一次健康檢查
2. **監控報告**: 檢查 HTML 報告，關注失敗的爬蟲
3. **快速修復**: 發現問題立即修復，避免累積
4. **版本控制**: 測試報告不納入 Git，但 JSON 數據可選擇性保留
5. **文檔更新**: 新增爬蟲時同步更新測試文檔

---

## 🔗 相關資源

- [pytest 官方文檔](https://docs.pytest.org/)
- [Scrapy 測試文檔](https://docs.scrapy.org/en/latest/topics/testing.html)
- [pytest-cov 文檔](https://pytest-cov.readthedocs.io/)

---

**Last Updated**: 2025-11-18
**Version**: 1.0.0
**Maintainer**: Information Retrieval System Development Team
