# 資訊檢索系統實作指南 *Implementation Guide*

本文件提供各核心模組的詳細實作說明，包括理論基礎、演算法步驟、程式碼範例、複雜度分析與常見陷阱。

---

## 目錄

1. [布林檢索系統](#1-布林檢索系統)
2. [向量空間模型](#2-向量空間模型)
3. [評估指標系統](#3-評估指標系統)
4. [查詢擴展](#4-查詢擴展-rocchio)
5. [分群演算法](#5-分群演算法)
6. [自動摘要](#6-自動摘要)

---

## 1. 布林檢索系統

### 1.1 理論基礎

布林檢索（*Boolean Retrieval*）是最古老也最基礎的 IR 模型，基於集合論（*Set Theory*）與布林代數（*Boolean Algebra*），支援 AND、OR、NOT 三種基本運算。

**核心概念**：
- **倒排索引** *Inverted Index*：詞彙 → 文件列表的映射
- **文件頻率** *Document Frequency (DF)*：包含某詞彙的文件數量
- **詞組查詢** *Phrase Query*：需要位置資訊支援

### 1.2 倒排索引建構

#### 演算法步驟

```python
def build_inverted_index(documents: List[str]) -> Dict[str, List[int]]:
    """
    Build an inverted index from a list of documents.

    Args:
        documents: List of raw text documents

    Returns:
        Dictionary mapping term -> list of document IDs

    Complexity:
        Time: O(T) where T is total number of tokens
        Space: O(V + P) where V is vocabulary size, P is postings size

    Example:
        >>> docs = ["hello world", "world peace", "hello peace"]
        >>> index = build_inverted_index(docs)
        >>> index["world"]
        [0, 1]
    """
    index = {}

    for doc_id, doc in enumerate(documents):
        # Tokenization: split by whitespace (簡化版)
        tokens = doc.lower().split()

        # Remove duplicates in same document
        unique_tokens = set(tokens)

        for token in unique_tokens:
            if token not in index:
                index[token] = []
            index[token].append(doc_id)

    return index
```

#### 進階：位置索引

支援詞組查詢需要儲存位置資訊：

```python
def build_positional_index(documents: List[str]) -> Dict[str, List[Tuple[int, List[int]]]]:
    """
    Build a positional inverted index.

    Returns:
        Dictionary mapping term -> [(doc_id, [positions]), ...]

    Example:
        >>> docs = ["to be or not to be"]
        >>> index = build_positional_index(docs)
        >>> index["to"]
        [(0, [0, 4])]
        >>> index["be"]
        [(0, [1, 5])]
    """
    index = {}

    for doc_id, doc in enumerate(documents):
        tokens = doc.lower().split()

        for position, token in enumerate(tokens):
            if token not in index:
                index[token] = []

            # Find if this doc_id already exists
            doc_entry = None
            for entry in index[token]:
                if entry[0] == doc_id:
                    doc_entry = entry
                    break

            if doc_entry:
                doc_entry[1].append(position)
            else:
                index[token].append((doc_id, [position]))

    return index
```

### 1.3 查詢處理

#### AND 查詢最佳化

```python
def boolean_and(postings_lists: List[List[int]]) -> List[int]:
    """
    Compute AND of multiple postings lists.

    Optimization: Process in order of increasing list length.

    Complexity:
        Time: O(n1 + n2 + ... + nk) where ni is length of i-th list
        Space: O(min(n1, n2, ..., nk))
    """
    if not postings_lists:
        return []

    # Sort by length (shortest first)
    postings_lists = sorted(postings_lists, key=len)

    result = postings_lists[0][:]

    for postings in postings_lists[1:]:
        result = merge_and(result, postings)
        if not result:  # Early termination
            break

    return result


def merge_and(list1: List[int], list2: List[int]) -> List[int]:
    """
    Merge two sorted postings lists with AND operation.
    """
    result = []
    i, j = 0, 0

    while i < len(list1) and j < len(list2):
        if list1[i] == list2[j]:
            result.append(list1[i])
            i += 1
            j += 1
        elif list1[i] < list2[j]:
            i += 1
        else:
            j += 1

    return result
```

#### 詞組查詢

```python
def phrase_query(positional_index: Dict, phrase: List[str]) -> List[int]:
    """
    Process phrase query using positional index.

    Args:
        positional_index: Positional index
        phrase: List of tokens in phrase

    Returns:
        List of document IDs containing the phrase

    Example:
        >>> phrase_query(index, ["to", "be"])
        [0]  # doc 0 contains "to be"
    """
    if not phrase:
        return []

    # Get postings for first term
    if phrase[0] not in positional_index:
        return []

    candidates = positional_index[phrase[0]]

    result = []

    for doc_id, positions in candidates:
        # Check if phrase exists in this document
        for start_pos in positions:
            match = True

            for i, term in enumerate(phrase[1:], start=1):
                if term not in positional_index:
                    match = False
                    break

                # Check if term appears at position start_pos + i
                found = False
                for d_id, pos_list in positional_index[term]:
                    if d_id == doc_id and (start_pos + i) in pos_list:
                        found = True
                        break

                if not found:
                    match = False
                    break

            if match:
                result.append(doc_id)
                break  # Found in this doc, no need to check other positions

    return result
```

### 1.4 常見陷阱

❌ **錯誤**：未排序 postings lists 就執行合併
✅ **正確**：確保所有 postings lists 已排序（建構時或查詢時排序）

❌ **錯誤**：忽略大小寫（`Hello` ≠ `hello`）
✅ **正確**：在索引建構時統一正規化（lowercase, remove accents）

❌ **錯誤**：位置索引儲存全域位置
✅ **正確**：每個文件的位置從 0 開始計數

---

## 2. 向量空間模型

### 2.1 理論基礎

向量空間模型（*Vector Space Model, VSM*）將文件與查詢表示為高維向量，透過向量相似度衡量相關性。

**核心公式**：

**TF-IDF 權重**：
```
tf(t, d) = freq(t, d) / max_freq(d)  # 正規化詞頻
idf(t) = log(N / df(t))              # 逆文件頻率
weight(t, d) = tf(t, d) × idf(t)     # TF-IDF 權重
```

**餘弦相似度**：
```
sim(q, d) = (q · d) / (|q| × |d|)
         = Σ(q_i × d_i) / sqrt(Σq_i²) × sqrt(Σd_i²)
```

### 2.2 實作步驟

#### Step 1: 計算 TF-IDF 權重

```python
import math
from collections import Counter
from typing import Dict, List

def compute_tf(document: List[str]) -> Dict[str, float]:
    """
    Compute normalized term frequency.

    Args:
        document: List of tokens

    Returns:
        Dictionary mapping term -> TF value

    Example:
        >>> compute_tf(["hello", "world", "hello"])
        {'hello': 1.0, 'world': 0.5}
    """
    freq = Counter(document)
    max_freq = max(freq.values()) if freq else 1

    return {term: count / max_freq for term, count in freq.items()}


def compute_idf(documents: List[List[str]]) -> Dict[str, float]:
    """
    Compute inverse document frequency.

    Args:
        documents: List of tokenized documents

    Returns:
        Dictionary mapping term -> IDF value

    Complexity:
        Time: O(T) where T is total tokens
        Space: O(V) where V is vocabulary size
    """
    N = len(documents)
    df = Counter()

    for doc in documents:
        unique_terms = set(doc)
        for term in unique_terms:
            df[term] += 1

    idf = {}
    for term, doc_freq in df.items():
        idf[term] = math.log(N / doc_freq)

    return idf


def compute_tfidf(documents: List[List[str]]) -> List[Dict[str, float]]:
    """
    Compute TF-IDF vectors for all documents.

    Returns:
        List of TF-IDF dictionaries, one per document
    """
    idf = compute_idf(documents)
    tfidf_vectors = []

    for doc in documents:
        tf = compute_tf(doc)
        tfidf = {term: tf_val * idf[term]
                 for term, tf_val in tf.items()}
        tfidf_vectors.append(tfidf)

    return tfidf_vectors
```

#### Step 2: 餘弦相似度計算

```python
def cosine_similarity(vec1: Dict[str, float], vec2: Dict[str, float]) -> float:
    """
    Compute cosine similarity between two sparse vectors.

    Args:
        vec1, vec2: Sparse vectors as dictionaries

    Returns:
        Cosine similarity score in [0, 1]

    Complexity:
        Time: O(min(|vec1|, |vec2|))
        Space: O(1)

    Example:
        >>> v1 = {'hello': 0.5, 'world': 0.3}
        >>> v2 = {'hello': 0.4, 'world': 0.6}
        >>> cosine_similarity(v1, v2)
        0.933...
    """
    # Dot product
    dot_product = sum(vec1.get(term, 0) * vec2.get(term, 0)
                      for term in set(vec1) | set(vec2))

    # Magnitudes
    mag1 = math.sqrt(sum(val ** 2 for val in vec1.values()))
    mag2 = math.sqrt(sum(val ** 2 for val in vec2.values()))

    if mag1 == 0 or mag2 == 0:
        return 0.0

    return dot_product / (mag1 * mag2)
```

#### Step 3: Top-K 檢索

```python
import heapq

def retrieve_top_k(query: List[str],
                   documents: List[List[str]],
                   k: int = 10) -> List[Tuple[int, float]]:
    """
    Retrieve top-k documents for a query using TF-IDF + Cosine similarity.

    Args:
        query: Tokenized query
        documents: List of tokenized documents
        k: Number of top results to return

    Returns:
        List of (doc_id, score) tuples, sorted by score descending

    Complexity:
        Time: O(N × M + N log k) where N is num docs, M is avg doc length
        Space: O(N × M) for TF-IDF storage
    """
    # Compute TF-IDF for documents
    tfidf_docs = compute_tfidf(documents)

    # Compute TF-IDF for query (treat as a document)
    idf = compute_idf(documents)
    tf_query = compute_tf(query)
    tfidf_query = {term: tf_val * idf.get(term, 0)
                   for term, tf_val in tf_query.items()}

    # Compute similarities
    scores = []
    for doc_id, tfidf_doc in enumerate(tfidf_docs):
        score = cosine_similarity(tfidf_query, tfidf_doc)
        scores.append((doc_id, score))

    # Return top-k using heap for efficiency
    top_k = heapq.nlargest(k, scores, key=lambda x: x[1])

    return top_k
```

### 2.3 權重方案變體

**標準權重方案**：

| 符號 | TF | IDF | 正規化 |
|------|----|----|-------|
| n | 原始頻率 | 無 | 無 |
| l | log(1 + freq) | 無 | 無 |
| t | 正規化 freq/max | log(N/df) | 餘弦正規化 |

**常用組合**：
- **ltc**：文件用 (log tf, idf, cosine norm)
- **lnc**：查詢用 (log tf, no idf, cosine norm)

```python
def compute_ltc_weight(tf: float, idf: float) -> float:
    """
    Compute ltc weight: log TF, IDF, cosine normalization.
    """
    if tf == 0:
        return 0.0
    return (1 + math.log(tf)) * idf
```

### 2.4 最佳化技巧

**🚀 優化 1：僅計算查詢詞彙出現的文件**

```python
def retrieve_top_k_optimized(query: List[str],
                             inverted_index: Dict[str, List[int]],
                             tfidf_docs: List[Dict[str, float]],
                             k: int = 10) -> List[Tuple[int, float]]:
    """
    Optimized retrieval: only compute scores for docs containing query terms.

    Speedup: 10-100x for large document collections.
    """
    # Find candidate documents (union of postings)
    candidates = set()
    for term in set(query):
        if term in inverted_index:
            candidates.update(inverted_index[term])

    # Compute TF-IDF for query
    idf = compute_idf_from_index(inverted_index)
    tfidf_query = compute_query_tfidf(query, idf)

    # Compute scores only for candidates
    scores = []
    for doc_id in candidates:
        score = cosine_similarity(tfidf_query, tfidf_docs[doc_id])
        scores.append((doc_id, score))

    return heapq.nlargest(k, scores, key=lambda x: x[1])
```

**🚀 優化 2：預計算文件向量的模長**

```python
def precompute_doc_norms(tfidf_docs: List[Dict[str, float]]) -> List[float]:
    """
    Precompute document vector magnitudes for faster cosine computation.
    """
    return [math.sqrt(sum(val ** 2 for val in doc.values()))
            for doc in tfidf_docs]
```

---

## 3. 評估指標系統

### 3.1 Precision, Recall, F-measure

```python
def compute_precision(retrieved: Set[int], relevant: Set[int]) -> float:
    """
    Compute precision.

    Args:
        retrieved: Set of retrieved document IDs
        relevant: Set of relevant document IDs

    Returns:
        Precision score in [0, 1]

    Formula:
        Precision = |retrieved ∩ relevant| / |retrieved|
    """
    if not retrieved:
        return 0.0
    return len(retrieved & relevant) / len(retrieved)


def compute_recall(retrieved: Set[int], relevant: Set[int]) -> float:
    """
    Compute recall.

    Formula:
        Recall = |retrieved ∩ relevant| / |relevant|
    """
    if not relevant:
        return 0.0
    return len(retrieved & relevant) / len(relevant)


def compute_f_measure(precision: float, recall: float, beta: float = 1.0) -> float:
    """
    Compute F-measure.

    Args:
        precision, recall: P and R scores
        beta: Weight factor (beta=1 for F1, beta=2 emphasizes recall)

    Formula:
        F_β = (1 + β²) × (P × R) / (β² × P + R)
    """
    if precision == 0 and recall == 0:
        return 0.0
    return (1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall)
```

### 3.2 Average Precision (AP) 與 MAP

```python
def compute_average_precision(ranked_results: List[int],
                             relevant: Set[int]) -> float:
    """
    Compute Average Precision for a single query.

    Args:
        ranked_results: List of document IDs in ranking order
        relevant: Set of relevant document IDs

    Returns:
        AP score in [0, 1]

    Formula:
        AP = (Σ P@k × rel(k)) / |relevant|

    Example:
        >>> ranked_results = [1, 3, 5, 2, 7]  # doc IDs in rank order
        >>> relevant = {1, 2, 5}
        >>> compute_average_precision(ranked_results, relevant)
        0.778  # (1.0 + 0.67 + 0.75) / 3
    """
    if not relevant:
        return 0.0

    precision_sum = 0.0
    num_relevant_seen = 0

    for k, doc_id in enumerate(ranked_results, start=1):
        if doc_id in relevant:
            num_relevant_seen += 1
            precision_at_k = num_relevant_seen / k
            precision_sum += precision_at_k

    return precision_sum / len(relevant)


def compute_map(queries_results: List[List[int]],
                queries_relevant: List[Set[int]]) -> float:
    """
    Compute Mean Average Precision across multiple queries.

    Args:
        queries_results: List of ranked results, one per query
        queries_relevant: List of relevant sets, one per query

    Returns:
        MAP score in [0, 1]
    """
    ap_scores = []
    for results, relevant in zip(queries_results, queries_relevant):
        ap = compute_average_precision(results, relevant)
        ap_scores.append(ap)

    return sum(ap_scores) / len(ap_scores) if ap_scores else 0.0
```

### 3.3 nDCG (Normalized Discounted Cumulative Gain)

```python
def compute_dcg(relevance_scores: List[int], k: int = None) -> float:
    """
    Compute Discounted Cumulative Gain.

    Args:
        relevance_scores: List of relevance grades in ranking order (0-3)
        k: Truncation level (None = use all)

    Returns:
        DCG score

    Formula:
        DCG@k = Σ (2^rel_i - 1) / log₂(i + 1)

    Example:
        >>> compute_dcg([3, 2, 3, 0, 1, 2], k=5)
        13.84...
    """
    if k is not None:
        relevance_scores = relevance_scores[:k]

    dcg = 0.0
    for i, rel in enumerate(relevance_scores, start=1):
        dcg += (2 ** rel - 1) / math.log2(i + 1)

    return dcg


def compute_ndcg(relevance_scores: List[int], k: int = None) -> float:
    """
    Compute Normalized DCG.

    Returns:
        nDCG score in [0, 1]
    """
    dcg = compute_dcg(relevance_scores, k)

    # Ideal ranking: sort by relevance descending
    ideal_scores = sorted(relevance_scores, reverse=True)
    idcg = compute_dcg(ideal_scores, k)

    if idcg == 0:
        return 0.0

    return dcg / idcg
```

---

## 4. 查詢擴展 (Rocchio)

### 4.1 Rocchio 演算法

```python
import numpy as np
from typing import List, Dict

def rocchio(query_vector: np.ndarray,
            relevant_docs: List[np.ndarray],
            irrelevant_docs: List[np.ndarray],
            alpha: float = 1.0,
            beta: float = 0.75,
            gamma: float = 0.15) -> np.ndarray:
    """
    Rocchio query expansion algorithm.

    Args:
        query_vector: Original query vector
        relevant_docs: List of relevant document vectors
        irrelevant_docs: List of irrelevant document vectors
        alpha, beta, gamma: Weight parameters

    Returns:
        Modified query vector

    Formula:
        Q_new = α×Q + β×(Σ D_rel / |D_rel|) - γ×(Σ D_irrel / |D_irrel|)

    Typical parameters:
        α = 1.0 (retain original query)
        β = 0.75 (emphasize relevant docs)
        γ = 0.0 or 0.15 (de-emphasize irrelevant)

    Complexity:
        Time: O(|rel| × d + |irrel| × d) where d is vector dimension
        Space: O(d)
    """
    modified_query = alpha * query_vector

    # Add centroid of relevant documents
    if relevant_docs:
        relevant_centroid = np.mean(relevant_docs, axis=0)
        modified_query += beta * relevant_centroid

    # Subtract centroid of irrelevant documents
    if irrelevant_docs:
        irrelevant_centroid = np.mean(irrelevant_docs, axis=0)
        modified_query -= gamma * irrelevant_centroid

    # Ensure non-negative (optional, depends on representation)
    modified_query = np.maximum(modified_query, 0)

    return modified_query


def pseudo_relevance_feedback(query: List[str],
                              initial_results: List[int],
                              documents: List[List[str]],
                              top_k: int = 10,
                              alpha: float = 1.0,
                              beta: float = 0.75) -> List[str]:
    """
    Pseudo-relevance feedback: assume top-k results are relevant.

    Args:
        query: Original query tokens
        initial_results: Document IDs from initial retrieval
        documents: All documents
        top_k: Number of top results to assume relevant

    Returns:
        Expanded query tokens

    Workflow:
        1. Retrieve initial results
        2. Assume top-k are relevant
        3. Apply Rocchio with γ=0
        4. Extract top-n terms from modified query vector
        5. Append to original query
    """
    # Assume top-k are relevant
    relevant_doc_ids = initial_results[:top_k]

    # Compute TF-IDF vectors
    tfidf_docs = compute_tfidf(documents)
    idf = compute_idf(documents)

    # Convert query to TF-IDF vector
    query_tf = compute_tf(query)
    query_tfidf_dict = {term: tf * idf.get(term, 0) for term, tf in query_tf.items()}

    # Convert to numpy arrays (assume fixed vocabulary)
    vocab = sorted(idf.keys())
    vocab_idx = {term: i for i, term in enumerate(vocab)}

    query_vec = dict_to_vector(query_tfidf_dict, vocab_idx)
    relevant_vecs = [dict_to_vector(tfidf_docs[doc_id], vocab_idx)
                     for doc_id in relevant_doc_ids]

    # Apply Rocchio (no irrelevant docs)
    modified_query_vec = rocchio(query_vec, relevant_vecs, [], alpha, beta, gamma=0)

    # Extract top-n terms from modified query
    expansion_terms = extract_top_terms(modified_query_vec, vocab, n=5)

    # Combine original query with expansion terms
    expanded_query = list(query) + [term for term in expansion_terms if term not in query]

    return expanded_query


def dict_to_vector(tfidf_dict: Dict[str, float], vocab_idx: Dict[str, int]) -> np.ndarray:
    """
    Convert sparse TF-IDF dictionary to dense numpy array.
    """
    vector = np.zeros(len(vocab_idx))
    for term, weight in tfidf_dict.items():
        if term in vocab_idx:
            vector[vocab_idx[term]] = weight
    return vector


def extract_top_terms(vector: np.ndarray, vocab: List[str], n: int = 5) -> List[str]:
    """
    Extract top-n terms with highest weights from a vector.
    """
    top_indices = np.argsort(vector)[-n:][::-1]
    return [vocab[i] for i in top_indices]
```

### 4.2 查詢擴展範例

```python
# Example usage
documents = [
    ["machine", "learning", "algorithm", "data"],
    ["deep", "learning", "neural", "network"],
    ["supervised", "learning", "classification"],
    # ... more documents
]

query = ["machine", "learning"]

# Step 1: Initial retrieval
initial_results = retrieve_top_k(query, documents, k=100)
top_doc_ids = [doc_id for doc_id, score in initial_results]

# Step 2: Pseudo-relevance feedback
expanded_query = pseudo_relevance_feedback(
    query,
    top_doc_ids,
    documents,
    top_k=10,
    alpha=1.0,
    beta=0.75
)

print(f"Original query: {query}")
print(f"Expanded query: {expanded_query}")
# Output: Expanded query: ['machine', 'learning', 'algorithm', 'supervised', 'deep']

# Step 3: Re-retrieve with expanded query
final_results = retrieve_top_k(expanded_query, documents, k=10)
```

---

## 5. 分群演算法

### 5.1 K-means 實作

```python
import numpy as np
from typing import List, Tuple

def kmeans(data: np.ndarray,
           k: int,
           max_iters: int = 100,
           tol: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
    """
    K-means clustering algorithm.

    Args:
        data: Data matrix (n_samples × n_features)
        k: Number of clusters
        max_iters: Maximum iterations
        tol: Convergence tolerance

    Returns:
        centers: Cluster centroids (k × n_features)
        labels: Cluster assignments (n_samples,)

    Complexity:
        Time: O(n × k × d × I) where I is iterations (typically < 100)
        Space: O(n + k × d)

    Algorithm:
        1. Initialize k centroids (random or k-means++)
        2. Repeat until convergence:
           a. Assign each point to nearest centroid
           b. Recompute centroids as mean of assigned points
        3. Return centroids and assignments
    """
    n_samples, n_features = data.shape

    # Initialize centroids using k-means++
    centers = kmeans_plus_plus_init(data, k)

    for iteration in range(max_iters):
        # Assignment step
        labels = assign_to_nearest_center(data, centers)

        # Update step
        new_centers = np.zeros((k, n_features))
        for cluster_id in range(k):
            cluster_points = data[labels == cluster_id]
            if len(cluster_points) > 0:
                new_centers[cluster_id] = cluster_points.mean(axis=0)
            else:
                # Handle empty cluster: reinitialize randomly
                new_centers[cluster_id] = data[np.random.randint(n_samples)]

        # Check convergence
        center_shift = np.linalg.norm(new_centers - centers)
        centers = new_centers

        if center_shift < tol:
            break

    return centers, labels


def kmeans_plus_plus_init(data: np.ndarray, k: int) -> np.ndarray:
    """
    K-means++ initialization: choose centroids to maximize initial spread.

    Algorithm:
        1. Choose first centroid uniformly at random
        2. For each subsequent centroid:
           - Choose point with probability proportional to D(x)²
           - D(x) = distance to nearest existing centroid
    """
    n_samples, n_features = data.shape
    centers = np.zeros((k, n_features))

    # First centroid: random
    centers[0] = data[np.random.randint(n_samples)]

    for i in range(1, k):
        # Compute distance to nearest existing centroid
        distances = np.array([min(np.linalg.norm(x - c) for c in centers[:i])
                             for x in data])

        # Choose next centroid with probability ∝ distance²
        probabilities = distances ** 2
        probabilities /= probabilities.sum()

        next_center_idx = np.random.choice(n_samples, p=probabilities)
        centers[i] = data[next_center_idx]

    return centers


def assign_to_nearest_center(data: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Assign each data point to the nearest centroid.
    """
    n_samples = data.shape[0]
    labels = np.zeros(n_samples, dtype=int)

    for i, point in enumerate(data):
        distances = [np.linalg.norm(point - center) for center in centers]
        labels[i] = np.argmin(distances)

    return labels
```

### 5.2 階層式分群

```python
def hierarchical_clustering(data: np.ndarray,
                            linkage: str = 'complete') -> List:
    """
    Hierarchical agglomerative clustering.

    Args:
        data: Data matrix (n_samples × n_features)
        linkage: 'single', 'complete', or 'average'

    Returns:
        Dendrogram as list of merge operations

    Complexity:
        Time: O(n³) naive, O(n² log n) with priority queue
        Space: O(n²) for distance matrix

    Algorithm:
        1. Start with each point as a cluster
        2. Repeat until one cluster remains:
           a. Find closest pair of clusters
           b. Merge them into new cluster
           c. Update distance matrix
        3. Return merge history (dendrogram)
    """
    n_samples = data.shape[0]

    # Initialize: each point is a cluster
    clusters = [{i} for i in range(n_samples)]

    # Compute pairwise distance matrix
    dist_matrix = compute_pairwise_distances(data)

    merges = []

    while len(clusters) > 1:
        # Find closest pair
        min_dist = float('inf')
        merge_i, merge_j = -1, -1

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                dist = cluster_distance(clusters[i], clusters[j], dist_matrix, linkage)
                if dist < min_dist:
                    min_dist = dist
                    merge_i, merge_j = i, j

        # Merge clusters
        new_cluster = clusters[merge_i] | clusters[merge_j]
        merges.append((clusters[merge_i], clusters[merge_j], min_dist))

        # Remove old clusters and add new
        clusters = [c for idx, c in enumerate(clusters) if idx not in {merge_i, merge_j}]
        clusters.append(new_cluster)

    return merges


def cluster_distance(cluster1: set, cluster2: set,
                    dist_matrix: np.ndarray, linkage: str) -> float:
    """
    Compute distance between two clusters.

    Linkage methods:
        - single: min distance between any two points
        - complete: max distance between any two points
        - average: average distance between all pairs
    """
    distances = [dist_matrix[i][j] for i in cluster1 for j in cluster2]

    if linkage == 'single':
        return min(distances)
    elif linkage == 'complete':
        return max(distances)
    elif linkage == 'average':
        return sum(distances) / len(distances)
    else:
        raise ValueError(f"Unknown linkage: {linkage}")


def compute_pairwise_distances(data: np.ndarray) -> np.ndarray:
    """
    Compute n×n pairwise Euclidean distance matrix.
    """
    n = data.shape[0]
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(data[i] - data[j])
            dist_matrix[i][j] = dist
            dist_matrix[j][i] = dist

    return dist_matrix
```

---

## 6. 自動摘要

### 6.1 Lead-K 摘要

```python
def lead_k_summary(document: str, k: int = 3) -> str:
    """
    Generate summary by taking first k sentences.

    Args:
        document: Raw text
        k: Number of sentences to include

    Returns:
        Summary text

    Justification:
        News articles often follow inverted pyramid structure,
        with most important information in opening sentences.

    Complexity:
        Time: O(n) where n is document length
        Space: O(n)
    """
    sentences = split_sentences(document)
    return ' '.join(sentences[:k])


def split_sentences(text: str) -> List[str]:
    """
    Simple sentence splitter (production code should use NLTK or spaCy).
    """
    import re
    # Split on period, question mark, exclamation mark
    sentences = re.split(r'[.!?]+', text)
    # Remove empty strings and strip whitespace
    return [s.strip() for s in sentences if s.strip()]
```

### 6.2 關鍵句萃取

```python
def key_sentence_extraction(document: str, k: int = 3) -> str:
    """
    Extract k most important sentences based on TF-IDF scores.

    Algorithm:
        1. Split document into sentences
        2. Compute TF-IDF for each sentence (treat as mini-document)
        3. Score each sentence by sum of TF-IDF weights
        4. Return top-k sentences in original order

    Complexity:
        Time: O(n × m) where n is sentences, m is avg sentence length
        Space: O(n × V) where V is vocabulary
    """
    sentences = split_sentences(document)

    # Tokenize sentences
    tokenized = [sentence.lower().split() for sentence in sentences]

    # Compute IDF across sentences (treat each sentence as a document)
    idf = compute_idf(tokenized)

    # Score each sentence
    scores = []
    for i, tokens in enumerate(tokenized):
        tf = compute_tf(tokens)
        tfidf_sum = sum(tf[term] * idf.get(term, 0) for term in tf)
        scores.append((i, tfidf_sum))

    # Select top-k sentences
    top_k_indices = sorted(scores, key=lambda x: x[1], reverse=True)[:k]

    # Return in original order
    top_k_indices = sorted([idx for idx, score in top_k_indices])

    summary_sentences = [sentences[i] for i in top_k_indices]
    return ' '.join(summary_sentences)
```

### 6.3 KWIC (KeyWord In Context)

```python
def kwic_summary(document: str, keywords: List[str], window: int = 50) -> str:
    """
    Generate KeyWord In Context summary.

    Args:
        document: Raw text
        keywords: Query terms to highlight
        window: Number of characters on each side of keyword

    Returns:
        Summary with keywords in context

    Example:
        >>> doc = "Machine learning is a subset of AI. Deep learning uses neural networks."
        >>> kwic_summary(doc, ["learning"], window=20)
        "...Machine learning is a subset o...... of AI. Deep learning uses neural ne..."

    Complexity:
        Time: O(n × k) where n is doc length, k is number of keywords
        Space: O(n)
    """
    snippets = []
    doc_lower = document.lower()

    for keyword in keywords:
        keyword_lower = keyword.lower()
        start = 0

        while True:
            # Find next occurrence
            pos = doc_lower.find(keyword_lower, start)
            if pos == -1:
                break

            # Extract context window
            snippet_start = max(0, pos - window)
            snippet_end = min(len(document), pos + len(keyword) + window)

            snippet = document[snippet_start:snippet_end]

            # Add ellipsis if not at document boundaries
            if snippet_start > 0:
                snippet = "..." + snippet
            if snippet_end < len(document):
                snippet = snippet + "..."

            snippets.append(snippet)
            start = pos + 1

    return ' '.join(snippets)
```

---

## 7. 開發最佳實踐

### 7.1 測試策略

```python
# tests/test_boolean.py
def test_inverted_index_basic():
    """Test basic inverted index construction."""
    docs = ["hello world", "world peace", "hello peace"]
    index = build_inverted_index(docs)

    assert index["hello"] == [0, 2]
    assert index["world"] == [0, 1]
    assert index["peace"] == [1, 2]


def test_inverted_index_empty():
    """Test edge case: empty document list."""
    index = build_inverted_index([])
    assert index == {}


def test_boolean_and():
    """Test AND operation."""
    list1 = [1, 3, 5, 7]
    list2 = [2, 3, 6, 7, 9]
    result = merge_and(list1, list2)
    assert result == [3, 7]
```

### 7.2 效能分析

```python
import time
import cProfile

def benchmark_retrieval(documents: List[str], queries: List[str]):
    """
    Benchmark retrieval performance.
    """
    # Build index
    start = time.time()
    index = build_inverted_index(documents)
    index_time = time.time() - start

    # Query processing
    start = time.time()
    for query in queries:
        retrieve_top_k(query.split(), documents, k=10)
    query_time = time.time() - start

    print(f"Indexing: {index_time:.3f}s")
    print(f"Querying ({len(queries)} queries): {query_time:.3f}s")
    print(f"Avg query latency: {query_time / len(queries) * 1000:.1f}ms")


# Profile with cProfile
cProfile.run('benchmark_retrieval(docs, queries)', sort='cumtime')
```

---

**最後更新**：2025-11-12
**維護者**：[您的姓名/學號]

**下一步**：參見 [CSoundex 詳細指南](CSOUNDEX.md) 了解中文文字處理實作細節。
