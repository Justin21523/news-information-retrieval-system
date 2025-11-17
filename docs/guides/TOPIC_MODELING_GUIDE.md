# 主題模型指南 (*Topic Modeling Guide*)

## 目錄

1. [簡介](#簡介)
2. [理論背景](#理論背景)
3. [LDA 模型](#lda-模型)
4. [BERTopic 模型](#bertopic-模型)
5. [模組架構](#模組架構)
6. [使用範例](#使用範例)
7. [性能分析](#性能分析)
8. [方法選擇](#方法選擇)
9. [參數調優](#參數調優)
10. [應用場景](#應用場景)

---

## 簡介

**主題模型** (*Topic Modeling*) 是一種無監督學習技術，用於從大量文檔集合中自動發現潛在主題。本模組提供兩種主流方法的實作：

### 核心功能

- **LDA (Latent Dirichlet Allocation)**: 經典機率主題模型
- **BERTopic**: 基於神經網路的現代主題建模
- **繁體中文優化**: 整合 CKIP/Jieba 分詞
- **模型評估**: Perplexity、Coherence 指標
- **視覺化支援**: 互動式主題視覺化
- **模型持久化**: 訓練模型儲存與載入

### 快速比較

| 特性 | LDA | BERTopic |
|------|-----|----------|
| **基礎** | 機率生成模型 | 神經網路 + 聚類 |
| **速度** | 中等 (需迭代) | 快速 (預訓練) |
| **準確度** | 良好 | 優秀 |
| **可解釋性** | 高 (機率分布) | 中等 |
| **主題數** | 需預先指定 | 自動偵測 |
| **語義理解** | 基於詞頻 | 基於語義 |
| **訓練數據需求** | 較多 | 較少 |

---

## 理論背景

### 1. 主題模型核心概念

主題模型基於以下假設：

1. **文檔是主題的混合物** (*Documents as Mixtures of Topics*)
   - 每篇文檔由多個主題組成
   - 例如：一篇論文可能 60% 關於機器學習，30% 關於自然語言處理，10% 關於評估方法

2. **主題是詞彙的分布** (*Topics as Distributions over Words*)
   - 每個主題定義一個詞彙機率分布
   - 例如：「機器學習」主題可能包含 {學習(0.1), 模型(0.08), 訓練(0.07), ...}

3. **無監督發現** (*Unsupervised Discovery*)
   - 不需要標註數據
   - 自動從文本中學習主題結構

### 2. 數學基礎

#### 文檔生成過程

```
對於每篇文檔 d:
  1. 抽取主題分布 θ_d ~ Dir(α)

  對於文檔中每個位置 n:
    2. 選擇主題 z_n ~ Multinomial(θ_d)
    3. 選擇詞彙 w_n ~ Multinomial(φ_{z_n})
```

其中：
- `α`: 文檔-主題 Dirichlet 先驗
- `β (或 η)`: 主題-詞彙 Dirichlet 先驗
- `θ_d`: 文檔 d 的主題分布
- `φ_k`: 主題 k 的詞彙分布

---

## LDA 模型

### 1. 演算法原理

**LDA** 由 Blei, Ng, Jordan (2003) 提出，是最經典的主題模型。

#### 核心公式

**文檔-主題分布** (*Document-Topic Distribution*):
```
P(z_k | d) = (n_{d,k} + α) / (Σ_k n_{d,k} + K*α)
```

**主題-詞彙分布** (*Topic-Word Distribution*):
```
P(w_j | z_k) = (n_{k,j} + β) / (Σ_j n_{k,j} + V*β)
```

其中：
- `n_{d,k}`: 文檔 d 中屬於主題 k 的詞數
- `n_{k,j}`: 主題 k 中詞彙 j 的出現次數
- `K`: 主題總數
- `V`: 詞彙表大小

#### 推論方法 (*Inference Methods*)

1. **Gibbs Sampling** (吉布斯取樣)
   - 馬可夫鏈蒙地卡羅 (MCMC) 方法
   - 逐詞抽樣更新主題分配
   - 收斂速度慢但精確

2. **Variational Bayes** (變分貝氏)
   - Gensim 預設方法
   - 快速近似推論
   - 適合大規模數據

### 2. Gensim 實作細節

本模組使用 **Gensim** (`gensim.models.LdaModel`) 實作，提供：

- **線上學習** (*Online Learning*): 支援數據流式訓練
- **多核平行化**: 利用 CPU 多核心加速
- **模型更新**: 增量更新已訓練模型

#### 訓練複雜度

```python
Time Complexity:  O(K * D * N * I)
Space Complexity: O(K * V + D * K)

其中:
  K = 主題數量
  D = 文檔數量
  N = 平均文檔長度
  I = 迭代次數
  V = 詞彙表大小
```

### 3. 參數說明

```python
LDAModel(
    n_topics=10,           # 主題數量 (需手動設定)
    alpha='symmetric',     # 文檔-主題先驗
    eta='auto',            # 主題-詞彙先驗
    iterations=50,         # 每篇文檔迭代次數
    passes=10,             # 語料庫遍歷次數
    min_word_freq=2,       # 最小詞頻 (過濾稀有詞)
    max_word_freq=0.5      # 最大詞頻比例 (過濾常見詞)
)
```

**Alpha (α) 參數**:
- `symmetric`: 所有主題均勻分布 (預設)
- `asymmetric`: 允許不均勻分布
- `float`: 固定值，控制稀疏度 (越小越稀疏)

**Eta (β) 參數**:
- `auto`: 自動計算 (1/K)
- `symmetric`: 均勻詞彙分布
- `float`: 固定值

### 4. 評估指標

#### Perplexity (困惑度)

衡量模型對測試數據的預測能力，**越低越好**。

```python
Perplexity = exp(-log P(w_test | M) / N_test)
```

- 值域: [1, ∞)
- 典型範圍: 100 - 2000
- **警告**: 低 perplexity ≠ 好主題 (可能過擬合)

#### Coherence (一致性)

衡量主題詞彙的語義一致性，**越高越好**。

```python
# 四種一致性指標
C_v:    基於滑動視窗 + 間接餘弦相似度 (推薦)
U_mass: 基於文檔共現 (快速但粗糙)
C_uci:  基於點互資訊 (PMI)
C_npmi: 標準化 PMI
```

**C_v 範例**:
```python
coherence = lda.calculate_coherence(coherence_type='c_v')
# C_v ∈ [-1, 1], 通常 > 0.4 表示良好主題
```

---

## BERTopic 模型

### 1. 演算法原理

**BERTopic** (Grootendorst, 2022) 是一種現代化主題建模方法，結合：

1. **BERT 文檔嵌入** (*Document Embeddings*)
2. **UMAP 降維** (*Dimensionality Reduction*)
3. **HDBSCAN 聚類** (*Clustering*)
4. **c-TF-IDF 主題表示** (*Topic Representation*)

#### 四階段流程

```
文檔 → [BERT Embeddings] → 高維向量 (768-dim)
      ↓
      [UMAP Reduction] → 低維向量 (5-dim)
      ↓
      [HDBSCAN Clustering] → 主題分配
      ↓
      [c-TF-IDF] → 主題關鍵詞
```

### 2. 核心技術

#### (1) BERT Embeddings

使用預訓練的 **Sentence-BERT** 模型：

```python
# 推薦模型 (繁體中文)
paraphrase-multilingual-MiniLM-L12-v2  # 預設，平衡速度與效果
distiluse-base-multilingual-cased-v1   # 更快速
xlm-roberta-base                       # 更準確但較慢
```

**優勢**:
- 捕捉語義相似性 (非僅詞彙重疊)
- 多語言支援
- 預訓練模型，無需大量數據

#### (2) UMAP 降維

**UMAP** (*Uniform Manifold Approximation and Projection*) 將高維嵌入投影到低維空間。

```python
UMAP(
    n_neighbors=15,    # 局部結構大小
    n_components=5,    # 目標維度
    metric='cosine'    # 距離度量
)
```

**參數影響**:
- `n_neighbors` ↑ → 保留全域結構 (但損失局部細節)
- `n_components`: 通常 5-10 維已足夠

#### (3) HDBSCAN 聚類

**HDBSCAN** (*Hierarchical Density-Based Clustering*) 自動偵測聚類數量。

```python
HDBSCAN(
    min_cluster_size=10,           # 最小聚類大小
    metric='euclidean',            # 距離度量
    cluster_selection_method='eom' # Excess of Mass
)
```

**優勢**:
- **自動主題數**: 不需預先指定
- **異常值檢測**: 主題 -1 表示未分類文檔
- **層次結構**: 支援多層次主題

#### (4) c-TF-IDF

**Class-based TF-IDF** 用於生成主題詞彙表示。

```
c-TF-IDF(w, c) = tf(w, c) * log(N / df(w))

其中:
  tf(w, c) = 詞 w 在類別 c 中的頻率
  N = 總類別數
  df(w) = 包含詞 w 的類別數
```

相比傳統 TF-IDF，c-TF-IDF 將**每個主題視為一個大文檔**，更能突顯主題特色詞。

### 3. 訓練複雜度

```python
# BERTopic 階段複雜度
Embedding:  O(n * L)        # L = 平均文檔長度
UMAP:       O(n * log n)    # n = 文檔數
HDBSCAN:    O(n * log n)
c-TF-IDF:   O(k * V)        # k = 主題數, V = 詞彙表

Total:      O(n * L + n log n)
Space:      O(n * d + k * V)  # d = 嵌入維度
```

### 4. 參數說明

```python
BERTopicModel(
    embedding_model='paraphrase-multilingual-MiniLM-L12-v2',
    language='multilingual',   # 停用詞語言
    n_neighbors=15,            # UMAP 局部鄰居數
    n_components=5,            # UMAP 目標維度
    min_cluster_size=10,       # HDBSCAN 最小聚類大小
    min_topic_size=10,         # 主題最小文檔數
    nr_topics=None,            # 主題數 (None=自動)
    top_n_words=10,            # 每主題顯示詞數
    device='cpu'               # 'cpu' 或 'cuda'
)
```

---

## 模組架構

### 1. 類別結構

```
src/ir/topic/
│
├── __init__.py                    # 模組初始化
├── lda_model.py                   # LDA 實作
│   ├── LDAModel                   # 主類別
│   └── LDATopicInfo               # 主題資訊容器
│
└── bertopic_model.py              # BERTopic 實作
    ├── BERTopicModel              # 主類別
    └── TopicInfo                  # 主題資訊容器
```

### 2. LDAModel 類別

```python
class LDAModel:
    # 初始化與配置
    __init__(n_topics, alpha, eta, ...)

    # 訓練方法
    fit(documents) -> self
    transform(documents) -> List[List[Tuple[int, float]]]

    # 主題資訊
    get_topics() -> Dict[int, List[Tuple[str, float]]]
    get_topic_words(topic_id, top_n) -> List[Tuple[str, float]]
    get_topic_info(topic_id=None) -> Union[DataFrame, LDATopicInfo]

    # 評估方法
    calculate_perplexity(documents=None) -> float
    calculate_coherence(coherence_type='c_v') -> float

    # 模型持久化
    save(path)
    load(path) -> LDAModel

    # 內部方法
    _preprocess_documents(documents) -> List[List[str]]
```

### 3. BERTopicModel 類別

```python
class BERTopicModel:
    # 初始化與配置
    __init__(embedding_model, language, ...)

    # 訓練方法
    fit(documents, embeddings=None) -> self
    fit_transform(documents) -> Tuple[List[int], ndarray]
    transform(documents) -> Tuple[List[int], ndarray]

    # 主題資訊
    get_topics() -> Dict[int, List[Tuple[str, float]]]
    get_topic_info(topic_id=None) -> Union[DataFrame, TopicInfo]
    get_topic_words(topic_id, top_n) -> List[Tuple[str, float]]
    get_representative_docs(topic_id, n_docs) -> List[str]

    # 視覺化
    visualize_topics() -> go.Figure
    visualize_barchart(topics, n_words) -> go.Figure
    visualize_hierarchy() -> go.Figure

    # 工具方法
    get_document_topic(doc_idx) -> Tuple[int, float]
    reduce_topics(nr_topics) -> self
    find_topics(search_term, top_n) -> List[Tuple[int, float]]

    # 模型持久化
    save(path, save_embeddings=False)
    load(path, device='cpu') -> BERTopicModel
```

---

## 使用範例

### 1. LDA 基本使用

```python
from src.ir.topic import LDAModel

# 準備文檔
documents = [
    "機器學習是人工智慧的重要分支",
    "資訊檢索系統需要倒排索引",
    "自然語言處理技術應用於文本分類",
    # ... 更多文檔
]

# (1) 初始化模型
lda = LDAModel(
    n_topics=10,              # 設定主題數
    iterations=50,
    passes=10,
    tokenizer_engine='jieba'  # 使用 Jieba 分詞
)

# (2) 訓練模型
lda.fit(documents)
print(f"詞彙表大小: {len(lda.dictionary)}")

# (3) 查看主題
topics = lda.get_topics()
for topic_id, words in topics.items():
    print(f"主題 {topic_id}:")
    for word, prob in words[:5]:
        print(f"  {word}: {prob:.4f}")

# (4) 推論新文檔
new_docs = ["深度學習神經網路"]
topic_dist = lda.transform(new_docs)
print(f"主題分布: {topic_dist[0]}")

# (5) 評估模型
perplexity = lda.calculate_perplexity()
coherence = lda.calculate_coherence('c_v')
print(f"Perplexity: {perplexity:.2f}")
print(f"Coherence (C_v): {coherence:.4f}")

# (6) 儲存模型
lda.save('models/lda_topic_model')
```

### 2. LDA 進階配置

```python
# 使用 CKIP 分詞與自訂參數
lda = LDAModel(
    n_topics=20,
    alpha='asymmetric',       # 允許不均勻主題分布
    eta=0.01,                 # 稀疏主題-詞彙分布
    iterations=100,
    passes=20,
    min_word_freq=5,          # 過濾低頻詞
    max_word_freq=0.3,        # 過濾高頻詞
    top_n_words=15,
    tokenizer_engine='ckip',  # 使用 CKIP
    use_stopwords=True,
    random_state=42
)

lda.fit(documents)

# 查看特定主題詳細資訊
topic_info = lda.get_topic_info(topic_id=0)
print(topic_info)
# 輸出: Topic 0 (15.3%): 機器學習(0.084), 深度學習(0.061), ...

# 批次推論
test_docs = ["文檔1", "文檔2", "文檔3"]
topic_dists = lda.transform(test_docs)
for i, dist in enumerate(topic_dists):
    main_topic = max(dist, key=lambda x: x[1])
    print(f"文檔 {i}: 主題 {main_topic[0]} (機率={main_topic[1]:.3f})")
```

### 3. BERTopic 基本使用

```python
from src.ir.topic import BERTopicModel

# (1) 初始化模型
topic_model = BERTopicModel(
    embedding_model='paraphrase-multilingual-MiniLM-L12-v2',
    language='multilingual',
    min_cluster_size=5,       # 最小聚類大小
    min_topic_size=5,         # 最小主題大小
    top_n_words=10,
    device='cpu'              # 或 'cuda'
)

# (2) 訓練模型 (自動偵測主題數)
topics, probs = topic_model.fit_transform(documents)

print(f"發現 {len(set(topics))} 個主題")
print(f"異常值文檔: {sum(1 for t in topics if t == -1)}")

# (3) 查看主題資訊
topic_info = topic_model.get_topic_info()
print(topic_info.head())
#   Topic  Count  Name
#   0      1      25    1_學習_機器_深度
#   1      2      18    2_檢索_索引_查詢

# (4) 查看特定主題詞彙
words = topic_model.get_topic_words(topic_id=0, top_n=10)
for word, score in words:
    print(f"{word}: {score:.4f}")

# (5) 獲取代表性文檔
rep_docs = topic_model.get_representative_docs(topic_id=0, n_docs=3)
for i, doc in enumerate(rep_docs, 1):
    print(f"{i}. {doc}")

# (6) 推論新文檔
new_topics, new_probs = topic_model.transform(["新的文檔內容"])
print(f"主題: {new_topics[0]}, 機率: {new_probs[0].max():.3f}")
```

### 4. BERTopic 進階功能

```python
# (1) 視覺化主題 (需安裝 plotly)
fig = topic_model.visualize_topics()
fig.write_html('outputs/topic_map.html')

# (2) 視覺化主題關鍵詞長條圖
fig = topic_model.visualize_barchart(topics=[0, 1, 2], n_words=10)
fig.write_html('outputs/topic_barchart.html')

# (3) 視覺化主題階層結構
fig = topic_model.visualize_hierarchy()
fig.write_html('outputs/topic_hierarchy.html')

# (4) 搜尋相關主題
related_topics = topic_model.find_topics("機器學習", top_n=3)
for topic_id, similarity in related_topics:
    print(f"主題 {topic_id}: 相似度 {similarity:.3f}")

# (5) 減少主題數量
print(f"原始主題數: {len(topic_model.get_topics())}")
topic_model.reduce_topics(nr_topics=10)
print(f"減少後主題數: {len(topic_model.get_topics())}")

# (6) 查看文檔主題分配
for i in range(min(5, len(documents))):
    topic_id, prob = topic_model.get_document_topic(i)
    print(f"Doc {i}: Topic {topic_id} (prob={prob:.3f})")
    print(f"  {documents[i][:60]}...")
```

### 5. 兩種方法比較

```python
# 準備相同數據
documents = [...]

# LDA 訓練
lda = LDAModel(n_topics=10, tokenizer_engine='jieba')
lda.fit(documents)
lda_topics = lda.get_topics()

# BERTopic 訓練
bertopic = BERTopicModel(min_cluster_size=10)
bertopic.fit(documents)
bert_topics = bertopic.get_topics()

# 比較結果
print("=" * 70)
print("LDA vs BERTopic 比較")
print("=" * 70)

print(f"\nLDA:")
print(f"  主題數: {len(lda_topics)}")
print(f"  詞彙表: {len(lda.dictionary)}")
print(f"  Perplexity: {lda.calculate_perplexity():.2f}")
print(f"  Coherence: {lda.calculate_coherence('c_v'):.4f}")

print(f"\nBERTopic:")
print(f"  主題數: {len(bert_topics)}")
print(f"  異常值: {sum(1 for t in bertopic.topics if t == -1)}")

# 顯示前3個主題的關鍵詞
print("\nLDA 主題:")
for tid in range(min(3, len(lda_topics))):
    words = ", ".join([w for w, p in lda_topics[tid][:5]])
    print(f"  主題 {tid}: {words}")

print("\nBERTopic 主題:")
for tid in range(min(3, len(bert_topics))):
    if tid == -1:
        continue
    words = ", ".join([w for w, s in bert_topics[tid][:5]])
    print(f"  主題 {tid}: {words}")
```

---

## 性能分析

### 1. 時間複雜度比較

| 操作 | LDA | BERTopic |
|------|-----|----------|
| **訓練** | O(K·D·N·I) | O(n·L + n log n) |
| **推論** | O(K·N·I) | O(m·L) |
| **主題提取** | O(K·V) | O(k·V) |

其中:
- K/k: 主題數
- D: 訓練文檔數
- n: 文檔數
- N: 平均文檔長度
- L: 平均文檔長度 (字元)
- I: 迭代次數
- V: 詞彙表大小
- m: 新文檔數

### 2. 空間複雜度

```python
# LDA
Space = O(K*V + D*K)
      = 主題-詞彙矩陣 + 文檔-主題矩陣

# BERTopic
Space = O(n*d + k*V)
      = 文檔嵌入 + 主題表示
```

### 3. 效能基準測試

測試環境: Intel i7-10700, 16GB RAM, CPU only

```
文檔數: 1000 篇 (平均長度 100 字)

LDA (10 topics, 50 iterations, 10 passes):
  - 訓練時間: ~45 秒
  - 記憶體: ~200 MB
  - Coherence (C_v): 0.52

BERTopic (自動主題數, paraphrase-multilingual-MiniLM):
  - 訓練時間: ~60 秒 (含嵌入)
  - 記憶體: ~800 MB
  - 發現主題數: 12
  - 異常值: 8%

文檔數: 10000 篇:

LDA:
  - 訓練時間: ~8 分鐘
  - 記憶體: ~500 MB

BERTopic:
  - 訓練時間: ~5 分鐘 (嵌入 4 分鐘)
  - 記憶體: ~4 GB
```

**效能建議**:

1. **小型數據集** (< 1000 文檔): 兩者皆可，BERTopic 更準確
2. **中型數據集** (1K-10K): BERTopic 更快且更準確
3. **大型數據集** (> 10K): LDA 更省記憶體，BERTopic 可預計算嵌入
4. **即時推論**: BERTopic 更快 (無需迭代)

### 4. GPU 加速

BERTopic 支援 GPU 加速嵌入計算：

```python
# 使用 CUDA (需安裝 torch with CUDA)
topic_model = BERTopicModel(
    device='cuda',  # 或 'cuda:0'
    embedding_model='paraphrase-multilingual-MiniLM-L12-v2'
)

# 速度提升 (1000 文檔):
#   CPU: ~30 秒嵌入
#   GPU (RTX 3080): ~5 秒嵌入
```

---

## 方法選擇

### 1. 使用 LDA 的情境

✅ **適合**:
- 需要**高可解釋性** (明確機率分布)
- 文檔有**明確的主題混合** (如新聞、論文)
- 需要**文檔-主題機率**精確值
- **主題數已知**或可估計
- **記憶體受限**環境
- 需要**線上學習**或增量更新

❌ **不適合**:
- 文檔非常短 (< 20 詞)
- 需要捕捉**語義相似性**
- 主題數未知
- 多語言混合文本

**典型應用**:
- 學術論文分類
- 新聞文章主題追蹤
- 長文檔內容分析
- 傳統文本挖掘任務

### 2. 使用 BERTopic 的情境

✅ **適合**:
- 需要**自動偵測主題數**
- 文檔**語義相似性**重要
- **短文本** (如推文、評論)
- **多語言**文本
- 需要**視覺化**互動探索
- 有**預訓練模型**可用

❌ **不適合**:
- 記憶體極度受限 (< 2GB)
- 需要精確機率分布
- 需要增量學習
- 計算資源受限

**典型應用**:
- 社群媒體分析
- 客戶評論主題發現
- 多語言文檔聚類
- 短文本主題建模
- 探索性數據分析

### 3. 決策樹

```
需要自動偵測主題數?
├─ Yes → BERTopic
└─ No
    └─ 文檔平均長度 < 30 詞?
        ├─ Yes → BERTopic (語義嵌入更適合短文)
        └─ No
            └─ 需要精確機率分布?
                ├─ Yes → LDA
                └─ No → BERTopic (通常更準確)
```

---

## 參數調優

### 1. LDA 調優指南

#### (1) 主題數 (n_topics)

```python
# 方法1: 困惑度與一致性掃描
coherence_scores = []
perplexity_scores = []

for k in range(5, 51, 5):
    lda = LDAModel(n_topics=k, tokenizer_engine='jieba')
    lda.fit(documents)

    coherence = lda.calculate_coherence('c_v')
    perplexity = lda.calculate_perplexity()

    coherence_scores.append(coherence)
    perplexity_scores.append(perplexity)

# 選擇 coherence 最高的 k (通常 10-30 之間)
best_k = (k_values[coherence_scores.index(max(coherence_scores))])
```

**經驗法則**:
- 小數據集 (< 1000 文檔): k = 5-15
- 中數據集 (1K-10K): k = 10-30
- 大數據集 (> 10K): k = 30-100

#### (2) Alpha 參數

```python
# 稀疏分布 (每篇文檔少數主題)
lda = LDAModel(n_topics=20, alpha=0.01)  # 稀疏

# 均勻分布 (每篇文檔多個主題)
lda = LDAModel(n_topics=20, alpha='symmetric')  # α = 1/K

# 自動優化 (Gensim預設)
lda = LDAModel(n_topics=20, alpha='auto')
```

**建議值**:
- α < 1: 稀疏 (每文檔少數主題) - **推薦**
- α = 1: 均勻
- α > 1: 平滑

#### (3) Eta (β) 參數

```python
# 稀疏主題詞 (每主題少數代表詞)
lda = LDAModel(n_topics=20, eta=0.01)

# 均勻詞彙分布
lda = LDAModel(n_topics=20, eta='symmetric')

# 自動 (推薦)
lda = LDAModel(n_topics=20, eta='auto')
```

#### (4) 迭代次數

```python
# 快速原型 (不推薦生產環境)
lda = LDAModel(iterations=10, passes=5)

# 標準配置
lda = LDAModel(iterations=50, passes=10)

# 高品質 (慢)
lda = LDAModel(iterations=100, passes=20)
```

**收斂檢查**:
```python
# 監控 log_perplexity 變化
# 如果連續幾次 passes 變化 < 1%，表示已收斂
```

#### (5) 詞頻過濾

```python
# 移除稀有詞 (噪音) 和超高頻詞 (停用詞)
lda = LDAModel(
    min_word_freq=5,      # 至少出現 5 次
    max_word_freq=0.5,    # 最多出現於 50% 文檔
    use_stopwords=True    # 使用停用詞表
)
```

### 2. BERTopic 調優指南

#### (1) 嵌入模型選擇

```python
# 速度優先 (小模型)
model = BERTopicModel(
    embedding_model='distiluse-base-multilingual-cased-v1'
)

# 平衡 (推薦)
model = BERTopicModel(
    embedding_model='paraphrase-multilingual-MiniLM-L12-v2'
)

# 準確度優先 (大模型)
model = BERTopicModel(
    embedding_model='xlm-roberta-base'
)
```

#### (2) UMAP 參數

```python
# 保留局部結構 (小 n_neighbors)
model = BERTopicModel(
    n_neighbors=5,      # 細緻局部結構
    n_components=5
)

# 保留全域結構 (大 n_neighbors)
model = BERTopicModel(
    n_neighbors=50,     # 整體結構
    n_components=10
)

# 平衡 (推薦)
model = BERTopicModel(
    n_neighbors=15,
    n_components=5
)
```

**經驗法則**:
- `n_neighbors`: 5-50 之間
- `n_components`: 2-10 之間 (通常 5 已足夠)

#### (3) HDBSCAN 參數

```python
# 小主題 (更多細緻主題)
model = BERTopicModel(
    min_cluster_size=5,    # 最小聚類
    min_topic_size=5
)

# 大主題 (較少但穩定的主題)
model = BERTopicModel(
    min_cluster_size=50,
    min_topic_size=50
)
```

**異常值控制**:
- `min_cluster_size` ↑ → 異常值 ↑, 主題數 ↓
- 如果異常值 > 20%，降低 `min_cluster_size`

#### (4) 手動設定主題數

```python
# 自動偵測 (預設)
model = BERTopicModel(nr_topics=None)
model.fit(documents)

# 訓練後減少主題
model.reduce_topics(nr_topics=15)

# 或初始化時指定
model = BERTopicModel(nr_topics=15)
```

---

## 應用場景

### 1. 學術論文分類

**任務**: 自動分類數千篇論文到研究主題

```python
# 使用 LDA (明確主題數)
lda = LDAModel(
    n_topics=20,           # 已知領域有約 20 個子領域
    alpha='asymmetric',    # 論文通常專注於少數主題
    iterations=100,
    tokenizer_engine='ckip'
)

lda.fit(paper_abstracts)

# 為每篇論文分配主題
for i, paper in enumerate(papers):
    topic_dist = lda.transform([paper])[0]
    main_topic = max(topic_dist, key=lambda x: x[1])
    print(f"Paper {i}: Topic {main_topic[0]} ({main_topic[1]:.2f})")
```

### 2. 社群媒體趨勢分析

**任務**: 從推文中發現熱門話題

```python
# 使用 BERTopic (短文本 + 自動主題數)
topic_model = BERTopicModel(
    min_cluster_size=10,
    min_topic_size=10,
    top_n_words=15
)

topics, probs = topic_model.fit_transform(tweets)

# 視覺化熱門主題
topic_info = topic_model.get_topic_info()
topic_info = topic_info[topic_info['Topic'] != -1]  # 移除異常值
topic_info = topic_info.sort_values('Count', ascending=False)

print("熱門主題:")
for idx, row in topic_info.head(10).iterrows():
    print(f"{row['Topic']}: {row['Name']} ({row['Count']} 則推文)")
```

### 3. 客戶評論主題發現

**任務**: 分析產品評論的主要議題

```python
# 使用 BERTopic + 視覺化
topic_model = BERTopicModel(
    language='multilingual',
    min_cluster_size=5
)

topics, probs = topic_model.fit_transform(reviews)

# 找出負面評論的主要問題
negative_reviews = [r for r, s in zip(reviews, sentiments) if s < 0]
neg_topics, _ = topic_model.transform(negative_reviews)

# 統計負面評論的主題分布
from collections import Counter
topic_counts = Counter(neg_topics)
print("負面評論主要問題:")
for topic_id, count in topic_counts.most_common(5):
    if topic_id != -1:
        words = topic_model.get_topic_words(topic_id, top_n=5)
        print(f"  主題 {topic_id} ({count} 則): {[w for w, s in words]}")
```

### 4. 新聞文章主題追蹤

**任務**: 追蹤新聞主題隨時間演變

```python
# 使用 LDA + 時間分析
import pandas as pd

lda = LDAModel(n_topics=15, tokenizer_engine='jieba')
lda.fit(news_articles)

# 為每篇新聞分配主題
news_df = pd.DataFrame({
    'date': dates,
    'article': news_articles
})

topic_dists = lda.transform(news_articles)
news_df['main_topic'] = [max(d, key=lambda x: x[1])[0] for d in topic_dists]

# 分析每月主題分布
news_df['month'] = pd.to_datetime(news_df['date']).dt.to_period('M')
topic_trend = news_df.groupby(['month', 'main_topic']).size().unstack(fill_value=0)

print(topic_trend)
```

### 5. 多語言文檔聚類

**任務**: 聚類包含中英文的混合文檔

```python
# 使用 BERTopic 多語言嵌入
topic_model = BERTopicModel(
    embedding_model='paraphrase-multilingual-MiniLM-L12-v2',
    language='multilingual',
    min_cluster_size=10
)

# 包含中文、英文文檔
mixed_docs = [
    "Machine learning is a subset of AI",
    "機器學習是人工智慧的子領域",
    "Deep learning uses neural networks",
    "深度學習使用神經網路",
    # ...
]

topics, probs = topic_model.fit_transform(mixed_docs)

# BERT 嵌入能捕捉跨語言語義相似性
# 中英文相似文檔會被分到相同主題
```

### 6. 長文檔摘要與主題標籤

**任務**: 為長文檔生成主題標籤

```python
# 結合 LDA 主題 + TextRank 關鍵詞
from src.ir.keyextract import TextRankExtractor

lda = LDAModel(n_topics=30, tokenizer_engine='ckip')
lda.fit(long_documents)

textrank = TextRankExtractor(tokenizer_engine='ckip')

for doc in long_documents:
    # 主題分配
    topic_dist = lda.transform([doc])[0]
    top3_topics = sorted(topic_dist, key=lambda x: x[1], reverse=True)[:3]

    # 關鍵詞提取
    keywords = textrank.extract(doc, top_k=5)

    print(f"文檔摘要:")
    print(f"  主題: {[f'T{t}({p:.2f})' for t, p in top3_topics]}")
    print(f"  關鍵詞: {[kw.word for kw in keywords]}")
```

---

## 參考文獻

### LDA

1. **原始論文**:
   - Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). "Latent Dirichlet Allocation". *Journal of Machine Learning Research*, 3, 993-1022.

2. **變分推論**:
   - Hoffman, M., Bach, F. R., & Blei, D. M. (2010). "Online Learning for Latent Dirichlet Allocation". *NIPS 2010*.

3. **評估指標**:
   - Röder, M., Both, A., & Hinneburg, A. (2015). "Exploring the Space of Topic Coherence Measures". *WSDM 2015*.

### BERTopic

1. **原始論文**:
   - Grootendorst, M. (2022). "BERTopic: Neural topic modeling with a class-based TF-IDF procedure". *arXiv:2203.05794*.

2. **BERT 嵌入**:
   - Reimers, N., & Gurevych, I. (2019). "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks". *EMNLP 2019*.

3. **UMAP**:
   - McInnes, L., Healy, J., & Melville, J. (2018). "UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction". *arXiv:1802.03426*.

4. **HDBSCAN**:
   - McInnes, L., & Healy, J. (2017). "Accelerated Hierarchical Density Based Clustering". *IEEE ICDM Workshop 2017*.

### 工具與函式庫

- **Gensim**: https://radimrehurek.com/gensim/
- **BERTopic**: https://maartengr.github.io/BERTopic/
- **Sentence-Transformers**: https://www.sbert.net/
- **CKIP Transformers**: https://github.com/ckiplab/ckip-transformers

---

## 故障排除

### 常見問題

#### Q1: LDA perplexity 很高但 coherence 也很高？

**原因**: Perplexity 和 Coherence 不完全相關。
**解決方案**: 優先關注 Coherence (更符合人類判斷)。

#### Q2: BERTopic 找到太多小主題？

**原因**: `min_cluster_size` 太小。
**解決方案**: 增加 `min_cluster_size` 和 `min_topic_size` 參數。

```python
model = BERTopicModel(
    min_cluster_size=20,   # 增加到 20
    min_topic_size=20
)
```

#### Q3: BERTopic 異常值 (Topic -1) 太多？

**原因**: 文檔太分散或參數設定太嚴格。
**解決方案**:
1. 降低 `min_cluster_size`
2. 使用 `reduce_topics()` 合併主題
3. 檢查文檔品質

#### Q4: 記憶體不足錯誤

**LDA 解決方案**:
- 增加 `min_word_freq` 減少詞彙表大小
- 減少 `n_topics`

**BERTopic 解決方案**:
- 預計算嵌入並儲存 (避免重複計算)
- 使用較小的嵌入模型
- 分批處理大型數據集

#### Q5: 中文主題詞彙混亂？

**原因**: 分詞不佳或停用詞不足。
**解決方案**:
1. 使用 CKIP 代替 Jieba
2. 擴充停用詞表
3. 增加 `min_word_freq`

```python
lda = LDAModel(
    tokenizer_engine='ckip',   # 使用 CKIP
    use_stopwords=True,
    min_word_freq=5            # 過濾低頻詞
)
```

---

**最後更新**: 2025-11-13
**版本**: 1.0
**作者**: Information Retrieval System
