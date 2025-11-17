# PAT-tree 快速開始指南
# Quick Start Guide

## 🚀 5分鐘快速上手

### 1. 啟動服務

```bash
# 進入專案目錄
cd /mnt/c/web-projects/information-retrieval

# 啟動Flask服務器
python app_simple.py --port 5000

# 等待PAT-tree建構完成（約30-40秒）
# 看到 "PAT-tree built in XX.XXs" 即表示完成
```

### 2. 訪問Web界面

打開瀏覽器訪問：
```
http://localhost:5000/pat_tree
```

### 3. 使用功能

#### 📊 查看統計資訊
頁面載入時自動顯示：
- 總詞彙數：49,028
- 唯一詞彙：8,478
- 壓縮率：2.32x

#### 🌲 可視化樹結構
1. **輸入前綴**（例如："台"）
2. **設定最大節點數**（10-500）
3. 點擊「生成樹結構視覺化」
4. 查看匹配的詞彙樹

#### 🔑 提取關鍵詞
1. **選擇評分方法**：
   - TF-IDF（推薦）
   - 詞頻統計
   - 文檔頻率
   - 綜合評分

2. **設定參數**：
   - Top-K: 20
   - 最小詞頻: 2
   - 最小文檔頻率: 1

3. 點擊「提取關鍵詞」
4. 查看排名結果

---

## 🔧 API使用範例

### 測試Tree Visualization

```bash
# 獲取統計資訊
curl "http://localhost:5000/api/pat_tree?max_nodes=1" | jq '.statistics'

# 前綴搜尋（搜尋"台"開頭的詞）
curl "http://localhost:5000/api/pat_tree?prefix=台&max_nodes=10" | jq

# 完整樹結構（限100個節點）
curl "http://localhost:5000/api/pat_tree?max_nodes=100" | jq '.tree' | head -50
```

### 測試Keyword Extraction

```bash
# TF-IDF方法提取Top 20關鍵詞
curl -X POST "http://localhost:5000/api/pat_tree_keywords" \
  -H "Content-Type: application/json" \
  -d '{
    "top_k": 20,
    "min_freq": 2,
    "min_doc_freq": 1,
    "method": "tfidf"
  }' | jq '.keywords[:5]'

# 詞頻方法提取Top 10高頻詞
curl -X POST "http://localhost:5000/api/pat_tree_keywords" \
  -H "Content-Type: application/json" \
  -d '{
    "top_k": 10,
    "min_freq": 5,
    "method": "frequency"
  }' | jq '.keywords | map(.term)'
```

---

## 💻 程式化使用

### Python範例

```python
from src.ir.index.pat_tree import PatriciaTree

# 建立簡單的PAT-tree
tree = PatriciaTree()

# 插入詞彙
terms = ["台灣", "台北", "台中", "中國", "中山"]
for term in terms:
    tree.insert(term, doc_id="doc1")

# 前綴搜尋
matches = tree.starts_with("台")
print(f"找到 {len(matches)} 個匹配: {[t for t, _ in matches]}")
# 輸出: 找到 3 個匹配: ['台灣', '台北', '台中']

# 提取關鍵詞
keywords = tree.extract_keywords(top_k=5, method='tfidf')
for kw in keywords:
    print(f"{kw['rank']}. {kw['term']} (score: {kw['score']:.4f})")

# 查看統計
stats = tree.get_statistics()
print(f"壓縮率: {stats['compression_ratio']:.2f}x")
```

---

## 📁 文件結構

```
information-retrieval/
├── src/ir/index/
│   ├── pat_tree.py           # PAT-tree核心實作
│   └── build_pat_tree.py     # 建構工具
├── templates/
│   └── pat_tree.html         # Web界面
├── static/js/
│   └── pat-tree.js           # 前端JavaScript
├── docs/guides/
│   └── PAT_TREE_GUIDE.md     # 完整技術文檔（詳細）
└── docs/
    └── PAT_TREE_QUICKSTART.md # 本檔案（快速開始）
```

---

## ❓ 常見問題 (FAQ)

### Q: PAT-tree建構需要多久？
**A**: 對於121篇CNA新聞（約49,000個詞彙），建構時間約36-40秒。

### Q: prefix search返回null怎麼辦？
**A**:
1. 確認server已完成PAT-tree建構（檢查log）
2. 檢查prefix是否存在於語料中
3. 嘗試重新啟動server

### Q: 如何調整關鍵詞提取的敏感度？
**A**:
- 增加`min_freq`：過濾低頻詞
- 增加`min_doc_freq`：過濾只出現在少數文檔的詞
- 選擇`combined`方法：綜合多種信號

### Q: 支援哪些中文分詞工具？
**A**:
- Jieba（預設，速度快）
- CKIP（學術級，較慢）

### Q: 可以用於其他語言嗎？
**A**: 可以！只需提供相應的tokenizer。當前專注於中文，但結構上支援任何語言。

---

## 🎯 下一步

### 深入學習
閱讀完整技術文檔：`docs/guides/PAT_TREE_GUIDE.md`

### 應用場景
- ✅ 自動補全系統
- ✅ 關鍵詞提取
- ✅ 文檔索引
- ✅ 文本分析

### 進階功能
- 🔜 C-value / NC-value複合詞提取
- 🔜 增量更新支援
- 🔜 拼寫校正整合

---

## 📞 獲取幫助

- 技術文檔：`docs/guides/PAT_TREE_GUIDE.md`
- GitHub Issues: [報告問題]
- 程式碼：`src/ir/index/pat_tree.py`

**快速測試指令**：
```bash
# 驗證安裝
python -c "from src.ir.index.pat_tree import PatriciaTree; print('✓ PAT-tree installed')"

# 運行簡單測試
python test_prefix_debug.py
```

---

**祝你使用愉快！** 🎉
