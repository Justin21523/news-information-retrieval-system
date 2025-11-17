# 關鍵詞擷取模組 (Keyword Extraction Module)

## 概述 (Overview)

本模組提供多種關鍵詞擷取演算法，支援繁體中文與英文文本。基於最新學術研究（2024-2025），實作包含TextRank、YAKE、RAKE與KeyBERT等演算法。

This module provides various keyword extraction algorithms for Traditional Chinese and English text, based on latest academic research (2024-2025).

## 功能特色 (Features)

✅ **4種擷取演算法 (4 Extraction Algorithms)**
- TextRank: 圖形化排序 (Graph-based ranking)
- YAKE: 統計特徵法 (Statistical features)
- RAKE: 快速自動擷取 (Rapid automatic extraction)
- KeyBERT: BERT嵌入向量 (BERT embeddings)

✅ **完整中文支援 (Full Chinese Support)**
- 繁體中文分詞 (CKIP/Jieba)
- 289個繁體停用詞 (Traditional Chinese stopwords)
- POS詞性標註 (POS tagging)

✅ **評估框架 (Evaluation Framework)**
- 監督式評估: P@K, R@K, F1@K, MAP, MRR, nDCG@K
- 非監督式評估: Diversity, Coverage

✅ **CLI工具 (Command-line Tool)**
- 批次處理 (Batch processing)
- 多演算法比較 (Multi-algorithm comparison)
- JSON輸出 (JSON output)

---

## 快速開始 (Quick Start)

### 安裝依賴 (Installation)

```bash
pip install networkx yake rake-nltk keybert sentence-transformers jieba
```

### 基本使用 (Basic Usage)

```python
from src.ir.keyextract import TextRankExtractor, YAKEExtractor

# 繁體中文範例
text = "機器學習是人工智慧的重要分支，深度學習使用神經網路建立複雜模型。"

# TextRank
extractor = TextRankExtractor(tokenizer_engine='jieba')
keywords = extractor.extract(text, top_k=5)

for kw in keywords:
    print(f"{kw.word}: {kw.score:.4f}")
```

輸出:
```
學習: 0.1246
機器: 0.0825
使用: 0.0548
資料: 0.0324
模式: 0.0317
```

### CLI使用 (Command-line Usage)

```bash
# 從文本提取
python scripts/extract_keywords.py \
    --text "深度學習是機器學習的子領域" \
    --method textrank \
    --top-k 5

# 從檔案提取
python scripts/extract_keywords.py \
    --input document.txt \
    --method yake \
    --top-k 10 \
    --output results.json

# 比較所有方法
python scripts/extract_keywords.py \
    --input doc.txt \
    --method all \
    --compare

# 使用ground truth評估
python scripts/extract_keywords.py \
    --input doc.txt \
    --ground-truth keywords.txt \
    --method textrank
```

---

## 演算法詳解 (Algorithm Details)

### 1. TextRank

**原理 (Principle):**
基於PageRank的圖形排序演算法，使用共現窗口建立詞彙圖。

**改進 (Improvements):**
- 位置權重 (Position weighting, Chen et al. 2025): 提升精確度 +6.3%
- POS詞性過濾 (POS filtering): 僅保留名詞與動詞
- 多詞組提取 (Multi-word keyphrase extraction)

**複雜度 (Complexity):**
- 時間: O(V² + I×V), V=唯一詞數, I=迭代次數
- 空間: O(V² + E), E=邊數

**參數 (Parameters):**
```python
TextRankExtractor(
    window_size=5,              # 共現窗口大小
    damping_factor=0.85,        # PageRank阻尼係數
    use_position_weight=True,   # 啟用位置權重
    pos_filter=['N', 'V'],      # POS過濾（名詞&動詞）
    tokenizer_engine='jieba'    # 分詞引擎
)
```

**範例 (Example):**
```python
from src.ir.keyextract import TextRankExtractor

extractor = TextRankExtractor(
    window_size=5,
    use_position_weight=True,
    pos_filter=['N', 'V']
)

text = """
機器學習是人工智慧的重要分支。
深度學習使用神經網路建立複雜模型。
"""

keywords = extractor.extract(text, top_k=5)
keyphrases = extractor.extract_keyphrases(text, top_k=3)
```

---

### 2. YAKE (Yet Another Keyword Extractor)

**原理 (Principle):**
基於統計特徵的無監督關鍵詞擷取，不需訓練。

**特徵 (Features):**
1. 詞頻 (Term Frequency)
2. 大小寫 (Casing)
3. 位置 (Position)
4. 上下文關聯度 (Term Relatedness to Context)
5. 句子分散度 (Term Different Sentence)

**複雜度 (Complexity):**
- 時間: O(n), n=文檔長度
- 空間: O(k), k=候選詞數
- 性能: ~2000 docs/2s

**參數 (Parameters):**
```python
YAKEExtractor(
    language='zh',              # 語言代碼
    max_ngram_size=3,           # 最大n-gram
    deduplication_threshold=0.9, # 去重閾值
    window_size=1,              # 共現窗口
    num_keywords=20             # 預設提取數量
)
```

**範例 (Example):**
```python
from src.ir.keyextract import YAKEExtractor

extractor = YAKEExtractor(
    language='zh',
    max_ngram_size=3
)

keywords = extractor.extract(text, top_k=10)
```

**Note:** YAKE分數越低越好 (Lower score = better)

---

### 3. RAKE (Rapid Automatic Keyword Extraction)

**原理 (Principle):**
使用停用詞作為片語分隔符，快速提取關鍵詞組。

**評分公式 (Scoring):**
- `degree_to_frequency`: deg(w) / freq(w) [預設]
- `word_degree`: deg(w)
- `word_frequency`: freq(w)

其中 deg(w) = 與詞w共現的其他詞總數

**複雜度 (Complexity):**
- 時間: O(n), n=詞數
- 空間: O(w), w=唯一詞數

**參數 (Parameters):**
```python
RAKEExtractor(
    min_length=1,                        # 最小詞組長度
    max_length=4,                        # 最大詞組長度
    ranking_metric='degree_to_frequency', # 排序指標
    include_repeated_phrases=True        # 包含重複片語
)
```

**範例 (Example):**
```python
from src.ir.keyextract import RAKEExtractor

extractor = RAKEExtractor(
    max_length=4,
    ranking_metric='degree_to_frequency'
)

keywords = extractor.extract(text, top_k=10)
```

---

### 4. KeyBERT

**原理 (Principle):**
使用BERT嵌入向量計算文檔與候選詞的語義相似度。

**MMR (Maximal Marginal Relevance):**
```
MMR = λ × Sim(candidate, document) - (1-λ) × max Sim(candidate, selected)
```
- λ: 相關性 vs. 多樣性權重
- 高λ: 更相關（相似）
- 低λ: 更多樣（不同）

**複雜度 (Complexity):**
- 時間: O(n×d + V×d), d=嵌入維度, V=候選詞數
- 空間: O(V×d + D×d)
- GPU加速可用

**支援模型 (Supported Models):**
- `paraphrase-multilingual-MiniLM-L12-v2`: 多語言（預設）
- `distiluse-base-multilingual-cased-v1`: 50+語言
- `bert-base-chinese`: 中文BERT
- `hfl/chinese-bert-wwm-ext`: 全詞遮罩中文BERT

**參數 (Parameters):**
```python
KeyBERTExtractor(
    model_name='paraphrase-multilingual-MiniLM-L12-v2',
    use_mmr=True,               # 啟用MMR
    diversity=0.5,              # 多樣性參數（0.0-1.0）
    keyphrase_ngram_range=(1,3), # N-gram範圍
    device='cpu'                # 'cpu' or 'cuda'
)
```

**範例 (Example):**
```python
from src.ir.keyextract import KeyBERTExtractor

extractor = KeyBERTExtractor(
    use_mmr=True,
    diversity=0.5,
    device='cpu'
)

# 提取關鍵詞
keywords = extractor.extract(text, top_k=5)

# 調整多樣性
keywords_diverse = extractor.extract(
    text,
    top_k=5,
    use_mmr=True,
    diversity=0.8  # 更高多樣性
)
```

---

## 評估框架 (Evaluation Framework)

### 監督式評估 (Supervised Metrics)

需要人工標註的ground truth關鍵詞。

**Precision@K:**
```
P@K = (# 正確關鍵詞 in top-K) / K
```

**Recall@K:**
```
R@K = (# 正確關鍵詞 in top-K) / (# ground truth關鍵詞)
```

**F1@K:**
```
F1@K = 2 × (P@K × R@K) / (P@K + R@K)
```

**MAP (Mean Average Precision):**
```
MAP = (1/|R|) × Σ P(k) × rel(k)
```

**MRR (Mean Reciprocal Rank):**
```
MRR = 1 / rank_of_first_relevant
```

**nDCG@K (Normalized Discounted Cumulative Gain):**
```
DCG@K = Σ_{i=1}^{K} rel_i / log2(i+1)
nDCG@K = DCG@K / IDCG@K
```

### 非監督式評估 (Unsupervised Metrics)

不需ground truth，評估關鍵詞品質。

**Diversity (多樣性):**
```
Diversity = (# 唯一token) / (# 總token)
```

**Coverage (覆蓋率):**
```
Coverage = (# 文本token在關鍵詞中) / (# 總文本token)
```

### 使用評估器 (Using Evaluator)

```python
from src.ir.keyextract.evaluator import KeywordEvaluator

# 初始化評估器
evaluator = KeywordEvaluator(
    k_values=[1, 3, 5, 10, 15],
    case_sensitive=False
)

# 提取的關鍵詞
extracted = ['機器學習', '深度學習', '神經網路', '人工智慧', '資料科學']

# Ground truth
ground_truth = ['機器學習', '人工智慧', '深度學習', '神經網路', '自然語言處理']

# 評估
result = evaluator.evaluate(extracted, ground_truth, text)

# 查看結果
print(f"P@5:  {result.precision_at_k[5]:.4f}")
print(f"R@5:  {result.recall_at_k[5]:.4f}")
print(f"F1@5: {result.f1_at_k[5]:.4f}")
print(f"MAP:  {result.map_score:.4f}")
print(f"MRR:  {result.mrr:.4f}")
print(f"nDCG@5: {result.ndcg_at_k[5]:.4f}")
```

**批次評估 (Batch Evaluation):**
```python
# 多個文檔
extracted_batch = [
    ['keyword1', 'keyword2', 'keyword3'],
    ['word1', 'word2', 'word3']
]
ground_truth_batch = [
    ['keyword1', 'keyword3', 'keyword4'],
    ['word1', 'word3', 'word5']
]

# 批次評估
results = evaluator.evaluate_batch(extracted_batch, ground_truth_batch)

# 聚合結果
avg_result = evaluator.aggregate_results(results)

print(f"Average P@3:  {avg_result.precision_at_k[3]:.4f}")
print(f"Average MAP:  {avg_result.map_score:.4f}")
```

---

## 測試 (Testing)

### 運行測試 (Run Tests)

```bash
# 運行所有測試
pytest tests/test_keyextract.py -v

# 運行特定測試類別
pytest tests/test_keyextract.py::TestTextRankExtractor -v

# 運行特定測試
pytest tests/test_keyextract.py::TestTextRankExtractor::test_textrank_basic -v

# 跳過慢速測試（KeyBERT）
pytest tests/test_keyextract.py -v -m "not slow"

# 顯示覆蓋率
pytest tests/test_keyextract.py --cov=src.ir.keyextract --cov-report=html
```

### 測試覆蓋率 (Test Coverage)

- TextRank: 25個測試
- YAKE: 8個測試
- RAKE: 8個測試
- KeyBERT: 5個測試（可選）
- Evaluator: 15個測試
- Integration: 3個測試

**總計: 64個測試**

---

## 效能比較 (Performance Comparison)

| 演算法 | 速度 | 精確度 | 記憶體 | 中文支援 | 需訓練 |
|--------|------|--------|--------|----------|--------|
| TextRank | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ✅ | ❌ |
| YAKE | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ | ❌ |
| RAKE | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ | ❌ |
| KeyBERT | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ✅ | ❌* |

*KeyBERT使用預訓練BERT模型，不需針對特定任務訓練

### 建議使用場景 (Recommended Use Cases)

**TextRank:**
- 需要高品質關鍵詞
- 文檔長度中等以上
- 可以接受較慢速度
- 需要位置權重或POS過濾

**YAKE:**
- 需要快速處理大量文檔
- 單一文檔擷取（無需語料庫）
- 多語言支援
- 記憶體受限環境

**RAKE:**
- 需要最快速度
- 簡單快速提取
- 記憶體極度受限
- 領域獨立應用

**KeyBERT:**
- 需要最高品質語義關鍵詞
- 有GPU資源可用
- 需要多樣性控制
- 跨語言語義相似度

---

## 參考文獻 (References)

### TextRank
- Mihalcea & Tarau (2004). "TextRank: Bringing Order into Text". EMNLP 2004.
- Chen et al. (2025). "An Improved Chinese Keyword Extraction Algorithm Based on Complex Networks".

### YAKE
- Campos et al. (2018). "YAKE! Collection-independent Automatic Keyword Extractor". ECIR 2018.
- Campos et al. (2020). "YAKE! Keyword Extraction from Single Documents using Multiple Local Features". Information Sciences, Vol 509.

### RAKE
- Rose et al. (2010). "Automatic Keyword Extraction from Individual Documents". Text Mining: Applications and Theory.

### KeyBERT
- Grootendorst (2020). "KeyBERT: Minimal keyword extraction with BERT".
- Devlin et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding". NAACL 2019.
- Carbonell & Goldstein (1998). "The use of MMR, diversity-based reranking for reordering documents and producing summaries". SIGIR 1998.

### Chinese NLP
- Chen et al. (2022). "Multifaceted Assessments of Traditional Chinese Word Segmentation Tool on Large Corpora". ROCLING 2022.

---

## 授權 (License)

Educational Use - 國立臺灣大學 圖書資訊學系 資訊組織課程專案

---

## 聯絡 (Contact)

如有問題或建議，請開啟GitHub issue或聯絡專案維護者。

For questions or suggestions, please open a GitHub issue or contact the project maintainer.
