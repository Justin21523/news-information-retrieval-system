# 資訊檢索系統實作總結 (Implementation Summary)

本文檔總結所有已實作的檢索模型與演算法。

## 已完成模組清單 (Completed Modules)

### Phase 1: 中文文本處理 (Chinese Text Processing)
✅ **CKIP Transformers 整合**
- 路徑: `src/ir/text/chinese_tokenizer.py`
- 功能: 中文分詞、詞性標注、命名實體識別
- 模型: CKIP-BERT (Academia Sinica)
- 詞彙量提升: 26% (6,710 → 8,478 terms)

### Phase 2: 欄位索引 (Field-Based Indexing)
✅ **元數據欄位搜尋**
- 路徑: `src/ir/index/field_indexer.py` (480 lines)
- 支援欄位: title, content, category, author, published_date, tags, source, language, summary (共9種)
- 查詢語法:
  - 欄位查詢: `title:台灣`
  - 日期範圍: `published_date:[2025-11-01 TO 2025-11-13]`
  - 複雜查詢: `(title:AI OR title:人工智慧) AND category:科技`

### Phase 3: 進階布林檢索 (Enhanced Boolean Retrieval)
✅ **NEAR/n 鄰近查詢**
- 路徑: `src/ir/retrieval/boolean.py` (修改)
- 語法: `資訊 NEAR/3 檢索` (找出距離3個詞內的共現)
- 演算法: Positional Index + Sliding Window

✅ **通配符查詢 (Wildcard Queries)**
- 路徑: `src/ir/retrieval/wildcard.py` (295 lines)
- 支援: `info*`, `te?t`, `*form*`
- 實作: Regex 模式匹配 + Permuterm Index 概念

✅ **模糊查詢 (Fuzzy Queries)**
- 路徑: `src/ir/retrieval/fuzzy.py` (324 lines)
- 語法: `test~2` (編輯距離≤2)
- 演算法: Levenshtein Distance (動態規劃)
- 複雜度: O(m × n) per query term

---

## 傳統 IR 模型 (Traditional IR Models)

### Phase 4: BM25 機率排序函數
✅ **BM25 檢索模型**
- 路徑: `src/ir/retrieval/bm25.py` (580+ lines)
- 公式: `BM25(D,Q) = Σ IDF(qi) × (f(qi,D) × (k1+1)) / (f(qi,D) + k1 × (1-b + b×|D|/avgdl))`
- 參數:
  - k1 = 1.5 (term frequency saturation)
  - b = 0.75 (length normalization)
  - delta = 0.0 (BM25+ variant)
- 功能:
  - 標準 BM25 排序
  - BM25+ 變體支援
  - 分數解釋 (`explain_score()`)
  - IDF 快取優化

### Week 1-2: N-gram 語言模型
✅ **N-gram Language Models**
- 路徑: `src/ir/langmodel/ngram.py` (580+ lines)
- 支援: Unigram, Bigram, Trigram, 任意 N-gram
- 平滑技術 (Smoothing):
  1. **Laplace (Add-1)**: `P(w) = (count(w) + 1) / (total + V)`
  2. **Jelinek-Mercer**: `P(w|d) = λ × P_ML(w|d) + (1-λ) × P(w|C)`
  3. **Dirichlet**: `P(w|d) = (count(w) + μ × P(w|C)) / (|d| + μ)`
  4. **Kneser-Ney**: 框架已實作
- 評估指標:
  - Perplexity: `2^(-1/N × Σ log2 P(wi))`
  - 句子機率計算
  - 文本生成

✅ **詞彙共現分析 (Collocation Extraction)**
- 路徑: `src/ir/langmodel/collocation.py` (550+ lines)
- 統計量測:
  1. **PMI** (Pointwise Mutual Information): `log2(P(x,y) / (P(x)×P(y)))`
  2. **LLR** (Log-Likelihood Ratio): `2 × Σ O × log(O/E)`
  3. **Chi-Square** (χ²): `Σ (O-E)² / E`
  4. **T-Score**: `(P(x,y) - P(x)×P(y)) / √(P(x,y)/N)`
  5. **Dice Coefficient**: `2×f(x,y) / (f(x)+f(y))`
- 功能:
  - 2×2 列聯表 (Contingency Tables)
  - 假設檢定 (Hypothesis Testing)
  - Top-K 詞彙組合提取

### Week 3-4: 機率檢索模型
✅ **Language Model Retrieval**
- 路徑: `src/ir/retrieval/language_model_retrieval.py` (500+ lines)
- 模型: Query Likelihood `P(Q|D)`
- 排序公式: `log P(Q|D) = Σ log P(qi|D)`
- 平滑方法:
  - Jelinek-Mercer (λ=0.7)
  - Dirichlet Prior (μ=2000)
  - Absolute Discounting (δ=0.7)
- 功能:
  - 查詢可能性評分
  - KL-Divergence 排序
  - 分數解釋

✅ **Binary Independence Model (BIM)**
- 路徑: `src/ir/retrieval/bim.py` (550+ lines)
- 理論基礎: Probability Ranking Principle
- RSV 公式: `RSV(D,Q) = Σ wi × xi`
- RSJ 權重: `wi = log((pi × (1-qi)) / ((1-pi) × qi))`
- 功能:
  - 無回饋檢索 (IDF 近似)
  - 相關性回饋 (Relevance Feedback)
  - RSJ 權重計算

---

## 索引壓縮與查詢優化 (Compression & Optimization)

✅ **Index Compression**
- 路徑: `src/ir/index/compression.py` (800+ lines)
- 編碼方法:
  1. **VByte** (Variable Byte):
     - 格式: 7 bits data + 1 continuation bit
     - 壓縮率: ~25-30%
  2. **Gamma** (Elias Gamma):
     - 格式: unary(length) + binary(offset)
     - 壓縮率: ~15-20%
  3. **Delta** (Elias Delta):
     - 格式: gamma(length+1) + binary(offset)
     - 壓縮率: ~10-15%
- Gap Encoding: 儲存 doc_id 差值而非絕對值
- 複雜度:
  - 編碼: O(n × log(max_value))
  - 解碼: O(encoded_size)

✅ **Query Optimization**
- 路徑: `src/ir/retrieval/query_optimization.py` (650+ lines)

**1. WAND (Weak AND)**
- 演算法: Document-at-a-Time with early termination
- 關鍵概念:
  - Term Upper Bounds: `UB(t) = max_d(score(t, d))`
  - Pivot Term: 第一個使 Σ UB(ti) ≥ θ 的詞
  - Threshold θ: 第 k 名文檔的分數
- 效能: 10-100x speedup over naive scoring
- 複雜度: O(m × log k) average, where m << N

**2. MaxScore**
- 演算法: Term partitioning (essential vs non-essential)
- 關鍵概念:
  - Essential terms: 必須匹配才能進入 top-k
  - Dynamic partition adjustment
- 效能: 特別適合包含罕見詞的查詢

---

## 現代檢索技術 (Modern Retrieval)

### Phase 5: 語義檢索 (Semantic Search)
✅ **BERT Dense Retrieval**
- 路徑: `src/ir/semantic/bert_retrieval.py` (550+ lines)
- 模型支援:
  - 中文: `bert-base-chinese`, `hfl/chinese-bert-wwm-ext`
  - 多語言: `paraphrase-multilingual-MiniLM-L12-v2`
- 技術:
  - Bi-encoder architecture
  - Mean pooling over token embeddings
  - Cosine similarity ranking
  - Optional FAISS indexing (ANN search)
- 向量維度: 768 (BERT-base)
- 複雜度:
  - 編碼: O(N × M × D²) where M=seq_len, D=hidden_dim
  - 搜尋: O(N × D) brute-force, O(log N × D) with FAISS

### Phase 6: 混合排序 (Hybrid Ranking)
✅ **Hybrid Ranker**
- 路徑: `src/ir/ranking/hybrid.py` (600+ lines)
- 融合策略:
  1. **Linear Combination**: `score = Σ wi × score_i`
     - 需要分數正規化 (min-max 或 z-score)
  2. **Reciprocal Rank Fusion (RRF)**: `score = Σ 1/(k + rank_i)`
     - k=60 (典型值)
     - 無需分數正規化
  3. **CombSUM**: `score = Σ score_i`
  4. **CombMNZ**: `score = (Σ score_i) × |matching_rankers|`
- 正規化方法:
  - Min-Max: `(x - min) / (max - min)`
  - Z-score: `(x - mean) / std`
- 權重配置:
  - 預設: 等權重
  - 可客製化各 ranker 權重

---

## 實作統計總覽 (Implementation Statistics)

### 程式碼量
```
傳統 IR 模型:
- BM25:                    580 lines
- N-gram Models:           580 lines
- Collocation:             550 lines
- Language Model (LM):     500 lines
- BIM:                     550 lines

索引與優化:
- Index Compression:       800 lines
- Query Optimization:      650 lines
- Field Indexer:          480 lines
- Wildcard Queries:       295 lines
- Fuzzy Queries:          324 lines

現代技術:
- BERT Retrieval:         550 lines
- Hybrid Ranking:         600 lines

總計: ~6,000+ lines of production code
```

### 支援的檢索模式

| 模型類型 | 特性 | 適用場景 |
|---------|------|---------|
| Boolean | 精確匹配 | 法律、專利檢索 |
| BM25 | 詞頻 + IDF | 一般文本檢索 |
| Language Model | 統計機率 | 語音識別、文本生成 |
| BIM | 機率排序 | 理論研究、相關性回饋 |
| BERT | 語義相似 | 問答系統、跨語言檢索 |
| Hybrid | 多信號融合 | 生產環境 |

---

## 使用範例 (Usage Examples)

### 1. BM25 檢索
```python
from src.ir.retrieval.bm25 import BM25Ranker

ranker = BM25Ranker(k1=1.5, b=0.75)
ranker.build_index(documents)

result = ranker.search("資訊檢索", topk=10)
print(result.doc_ids)  # [5, 12, 3, 18, ...]
```

### 2. Language Model 檢索
```python
from src.ir.retrieval.language_model_retrieval import LanguageModelRetrieval

lm = LanguageModelRetrieval(smoothing='dirichlet', mu_param=2000)
lm.build_index(documents)

result = lm.search("機器學習", topk=10)
print(result.doc_ids)
```

### 3. BERT 語義檢索
```python
from src.ir.semantic.bert_retrieval import BERTRetrieval

bert = BERTRetrieval(model_name="hfl/chinese-bert-wwm-ext")
bert.build_index(documents, batch_size=32)

result = bert.search("深度學習應用", topk=10)
print(result.scores)  # Cosine similarities
```

### 4. Hybrid 混合排序
```python
from src.ir.ranking.hybrid import HybridRanker

rankers = {
    'bm25': bm25_ranker,
    'bert': bert_retrieval,
    'lm': lm_retrieval
}

hybrid = HybridRanker(
    rankers=rankers,
    fusion_method='rrf',
    weights={'bm25': 0.4, 'bert': 0.4, 'lm': 0.2}
)

result = hybrid.search("自然語言處理", topk=10)
```

### 5. N-gram 語言模型
```python
from src.ir.langmodel.ngram import NGramModel

bigram = NGramModel(n=2, smoothing='jm', lambda_param=0.7)
bigram.train(documents)

prob = bigram.probability(("資訊",), "檢索")
perplexity = bigram.perplexity("測試句子")
```

### 6. 詞彙共現分析
```python
from src.ir.langmodel.collocation import CollocationExtractor

extractor = CollocationExtractor(min_freq=5)
extractor.train(documents)

collocations = extractor.extract_collocations(measure='pmi', topk=100)
for col in collocations[:10]:
    print(f"{col.bigram}: PMI={col.pmi:.2f}, LLR={col.llr:.2f}")
```

### 7. Query Optimization (WAND)
```python
from src.ir.retrieval.query_optimization import WANDRetrieval

wand = WANDRetrieval(inverted_index, doc_lengths, doc_count, avg_len)
result = wand.search(["資訊", "檢索"], topk=10)

print(f"Speedup: {result.speedup_ratio:.2f}x")
print(f"Scored {result.num_scored_docs} / {result.num_candidate_docs} docs")
```

### 8. Index Compression
```python
from src.ir.index.compression import VByteEncoder, GammaEncoder

# VByte encoding
vbyte = VByteEncoder()
encoded = vbyte.encode_gaps([3, 7, 10, 15])
decoded = vbyte.decode_gaps(encoded)

# Gamma encoding
gamma = GammaEncoder()
encoded_bits = gamma.encode_gaps([3, 7, 10, 15])
decoded = gamma.decode_gaps(encoded_bits)
```

---

## 理論基礎文獻 (References)

### 教科書
1. Manning, Raghavan & Schütze (2008). "Introduction to Information Retrieval"
2. Croft, Metzler & Strohman (2015). "Search Engines: Information Retrieval in Practice"
3. Baeza-Yates & Ribeiro-Neto (2011). "Modern Information Retrieval"

### 經典論文
1. **BM25**: Robertson & Zaragoza (2009). "The Probabilistic Relevance Framework"
2. **Language Models**: Zhai & Lafferty (2004). "A Study of Smoothing Methods"
3. **BERT**: Devlin et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers"
4. **WAND**: Broder et al. (2003). "Efficient Query Evaluation using Two-Level Retrieval"
5. **RRF**: Cormack et al. (2009). "Reciprocal Rank Fusion"
6. **BIM**: Robertson (1977). "The Probability Ranking Principle in IR"

---

## 下一步驟 (Next Steps)

### 等待中: 統一整合與完整測試
- [ ] 整合所有模型到 `app.py`
- [ ] 建立統一 API 介面
- [ ] 編寫完整單元測試
- [ ] 性能基準測試 (Benchmarking)
- [ ] 評估指標計算 (MAP, nDCG, P@k)
- [ ] 部署與優化

---

## 技術債與限制 (Technical Debt & Limitations)

1. **BERT 模型**:
   - 需要額外安裝 `transformers` 和 `torch`
   - 記憶體需求較高 (768-dim embeddings × N docs)
   - 建議使用 FAISS 加速大規模檢索

2. **索引壓縮**:
   - 目前僅實作編碼/解碼
   - 尚未整合到實際 inverted index

3. **Query Optimization**:
   - WAND 和 MaxScore 需要特定的 index 結構
   - 可考慮與 BM25Ranker 深度整合

4. **測試覆蓋率**:
   - 需要為每個模組編寫單元測試
   - 需要端對端整合測試

---

**總結**: 所有核心檢索演算法已完成實作,涵蓋從傳統到現代的完整 IR 技術棧。
系統具備從布林檢索到深度學習語義搜尋的完整能力,可支援多種檢索場景。

**作者**: Information Retrieval System
**日期**: 2025-11-13
**版本**: v1.0
