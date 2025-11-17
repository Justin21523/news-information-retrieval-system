"""
Quick Test Suite for IR System
快速測試套件 - 使用 jieba 分詞，不依賴 CKIP

Usage:
    source activate ai_env && python scripts/quick_test.py
"""

import sys
import time
from pathlib import Path

# Add parent directory to path
_parent_dir = Path(__file__).parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

# Import IR modules
from src.ir.text.chinese_tokenizer import ChineseTokenizer
from src.ir.index.inverted_index import InvertedIndex
from src.ir.index.positional_index import PositionalIndex
from src.ir.index.field_indexer import FieldIndexer
from src.ir.retrieval.vsm import VectorSpaceModel
from src.ir.retrieval.bm25 import BM25Ranker
from src.ir.retrieval.language_model_retrieval import LanguageModelRetrieval
from src.ir.retrieval.bim import BinaryIndependenceModel
from src.ir.retrieval.boolean import BooleanRetrieval
from src.ir.ranking.hybrid import HybridRanker
from src.ir.langmodel.ngram import NGramModel
from src.ir.langmodel.collocation import CollocationExtractor
from src.ir.ranking.rocchio import RocchioExpander
from src.ir.cluster.doc_cluster import DocumentClusterer
from src.ir.summarize.static import StaticSummarizer
from src.ir.summarize.dynamic import KWICGenerator
from src.ir.index.compression import VByteEncoder, GammaEncoder, DeltaEncoder

print("=" * 80)
print("IR 系統快速測試套件 (Quick Test Suite)")
print("=" * 80)
print()

# Test documents
test_docs = [
    "資訊檢索是計算機科學的重要分支",
    "向量空間模型使用TF-IDF權重計算相似度",
    "BM25是一種機率排序函數廣泛應用於搜尋引擎",
    "語言模型可以用於文本檢索和生成",
    "布林檢索支援AND OR NOT等邏輯運算子",
    "倒排索引是資訊檢索系統的核心數據結構",
    "詞彙共現分析可以發現有意義的詞組",
    "文檔分群可以組織大量文檔集合",
    "查詢擴展使用Rocchio演算法改善檢索效果",
    "N-gram語言模型估計詞序列的機率"
]

# Initialize tokenizer (using jieba)
print("【1. 初始化分詞器】")
tokenizer = ChineseTokenizer(engine='jieba')
print("✅ PASS - Jieba 分詞器初始化成功")
print()

# Test 1: Tokenization
print("【2. 測試分詞功能】")
test_text = "資訊檢索系統"
tokens = tokenizer.tokenize(test_text)
print(f"  輸入: '{test_text}'")
print(f"  結果: {tokens}")
if len(tokens) > 0:
    print("✅ PASS - 分詞功能正常")
else:
    print("❌ FAIL - 分詞結果為空")
print()

# Test 2: Inverted Index
print("【3. 測試倒排索引】")
inv_idx = InvertedIndex(tokenizer=tokenizer.tokenize)
inv_idx.build(test_docs)
term_count = len(inv_idx.index)
print(f"  文檔數: {len(test_docs)}")
print(f"  詞彙數: {term_count}")
if term_count > 0:
    print("✅ PASS - 倒排索引建立成功")
else:
    print("❌ FAIL - 倒排索引為空")
print()

# Test 3: Positional Index
print("【4. 測試位置索引】")
pos_idx = PositionalIndex(tokenizer=tokenizer.tokenize)
pos_idx.build(test_docs)
positions = pos_idx.get_positions("檢索", 0)
print(f"  查詢: '檢索' 在文檔0的位置")
print(f"  結果: {positions}")
if positions is not None:
    print("✅ PASS - 位置索引功能正常")
else:
    print("❌ FAIL - 無法獲取位置信息")
print()

# Test 4: VSM Search
print("【5. 測試向量空間模型 (VSM)】")
vsm = VectorSpaceModel(tokenizer=tokenizer.tokenize)
vsm.build_index(test_docs)
query = "資訊檢索"
result = vsm.search(query, topk=3)
print(f"  查詢: '{query}'")
print(f"  結果數: {len(result)}")
if len(result) > 0:
    for i, (doc_id, score) in enumerate(result[:3], 1):
        print(f"    {i}. Doc {doc_id}: {score:.4f}")
    print("✅ PASS - VSM 檢索成功")
else:
    print("❌ FAIL - VSM 無結果")
print()

# Test 5: BM25 Search
print("【6. 測試 BM25 排序】")
bm25 = BM25Ranker(tokenizer=tokenizer.tokenize, k1=1.5, b=0.75)
bm25.build_index(test_docs)
result = bm25.search(query, topk=3)
print(f"  查詢: '{query}'")
print(f"  結果數: {len(result)}")
if len(result) > 0:
    for i, (doc_id, score) in enumerate(result[:3], 1):
        print(f"    {i}. Doc {doc_id}: {score:.4f}")
    print("✅ PASS - BM25 檢索成功")
else:
    print("❌ FAIL - BM25 無結果")
print()

# Test 6: Language Model
print("【7. 測試語言模型檢索】")
lm = LanguageModelRetrieval(tokenizer=tokenizer.tokenize, smoothing='dirichlet', mu_param=2000)
lm.build_index(test_docs)
result = lm.search(query, topk=3)
print(f"  查詢: '{query}'")
print(f"  結果數: {len(result)}")
if len(result) > 0:
    for i, (doc_id, score) in enumerate(result[:3], 1):
        print(f"    {i}. Doc {doc_id}: {score:.4f}")
    print("✅ PASS - Language Model 檢索成功")
else:
    print("❌ FAIL - Language Model 無結果")
print()

# Test 7: Boolean Retrieval
print("【8. 測試布林檢索】")
boolean_engine = BooleanRetrieval(tokenizer=tokenizer.tokenize)
boolean_engine.build_index(test_docs)
bool_query = "資訊 AND 檢索"
result = boolean_engine.search(bool_query)
print(f"  查詢: '{bool_query}'")
print(f"  結果數: {len(result)}")
if len(result) > 0:
    print(f"    匹配文檔: {result}")
    print("✅ PASS - Boolean 檢索成功")
else:
    print("❌ FAIL - Boolean 無結果")
print()

# Test 8: Hybrid Ranking
print("【9. 測試混合排序】")
rankers = {
    'bm25': bm25,
    'vsm': vsm,
    'lm': lm
}
hybrid = HybridRanker(rankers=rankers, fusion_method='rrf', weights=None, normalization='minmax')
result = hybrid.search(query, topk=3, ranker_topk=10)
print(f"  查詢: '{query}'")
print(f"  結果數: {len(result)}")
if len(result) > 0:
    for i, (doc_id, score) in enumerate(result[:3], 1):
        print(f"    {i}. Doc {doc_id}: {score:.4f}")
    print("✅ PASS - Hybrid 檢索成功")
else:
    print("❌ FAIL - Hybrid 無結果")
print()

# Test 9: N-gram Model
print("【10. 測試 N-gram 語言模型】")
ngram = NGramModel(n=2, smoothing='jm', lambda_param=0.7)
ngram.train(test_docs)
test_sentence = "資訊檢索"
prob = ngram.probability(test_sentence)
perplexity = ngram.perplexity(test_sentence)
print(f"  句子: '{test_sentence}'")
print(f"  機率: {prob:.2e}")
print(f"  困惑度: {perplexity:.2f}")
if prob > 0:
    print("✅ PASS - N-gram 模型正常")
else:
    print("❌ FAIL - N-gram 機率為0")
print()

# Test 10: Collocation
print("【11. 測試詞彙共現分析】")
colloc = CollocationExtractor(tokenizer=tokenizer.tokenize, min_freq=1, window_size=2)
colloc.build(test_docs)
collocations = colloc.extract_collocations(measure='pmi', topk=5)
print(f"  測量方法: PMI")
print(f"  結果數: {len(collocations)}")
if len(collocations) > 0:
    for i, col in enumerate(collocations[:3], 1):
        print(f"    {i}. {col['bigram']}: PMI={col['pmi']:.2f}, freq={col['freq']}")
    print("✅ PASS - Collocation 分析成功")
else:
    print("❌ FAIL - 無共現詞組")
print()

# Test 11: Query Expansion
print("【12. 測試查詢擴展 (Rocchio)】")
rocchio = RocchioExpander(alpha=1.0, beta=0.75, gamma=0.15, max_expansion_terms=3)
query_vec = {"資訊": 0.8, "檢索": 0.6}
rel_vecs = [
    {"資訊": 0.5, "檢索": 0.7, "系統": 0.3, "搜尋": 0.2},
    {"資訊": 0.6, "文檔": 0.4, "索引": 0.3}
]
expanded = rocchio.expand_query(query_vec, rel_vecs)
print(f"  原始詞彙: {expanded.original_terms}")
print(f"  擴展詞彙: {expanded.expanded_terms}")
if len(expanded.expanded_terms) > 0:
    print("✅ PASS - Query Expansion 成功")
else:
    print("⚠️  WARN - 無擴展詞彙（可能是正常情況）")
print()

# Test 12: Document Clustering
print("【13. 測試文檔分群】")
clusterer = DocumentClusterer(tokenizer=tokenizer.tokenize, method='kmeans')
clusters = clusterer.cluster(test_docs, n_clusters=3)
print(f"  分群數: {len(clusters)}")
if len(clusters) > 0:
    for i, cluster in enumerate(clusters):
        print(f"    Cluster {i}: {cluster['size']} 個文檔")
    print("✅ PASS - Clustering 成功")
else:
    print("❌ FAIL - 分群失敗")
print()

# Test 13: Summarization
print("【14. 測試文檔摘要】")
summarizer = StaticSummarizer()
doc = test_docs[0]
summary = summarizer.lead_k_summarization(doc, k=1)
print(f"  原文: {doc}")
print(f"  摘要: {summary}")
if len(summary) > 0:
    print("✅ PASS - Summarization 成功")
else:
    print("❌ FAIL - 摘要為空")
print()

# Test 14: KWIC
print("【15. 測試 KWIC (關鍵詞上下文)】")
kwic = KWICGenerator(window_size=5)
doc = "資訊檢索是計算機科學的重要分支，檢索系統用於查找資訊"
keyword = "檢索"
result = kwic.generate(doc, keyword)
print(f"  關鍵詞: '{keyword}'")
print(f"  上下文數: {len(result['contexts'])}")
if len(result['contexts']) > 0:
    for ctx in result['contexts'][:2]:
        print(f"    ...{ctx['left']}【{ctx['keyword']}】{ctx['right']}...")
    print("✅ PASS - KWIC 生成成功")
else:
    print("❌ FAIL - 無上下文")
print()

# Test 15: Compression
print("【16. 測試索引壓縮】")
test_doc_ids = [3, 7, 10, 15, 22, 30]

# VByte
vbyte = VByteEncoder()
encoded = vbyte.encode_gaps(test_doc_ids)
decoded = vbyte.decode_gaps(encoded)
print(f"  VByte: {test_doc_ids} → {len(encoded)} bytes → {decoded}")
if decoded == test_doc_ids:
    print("✅ PASS - VByte 壓縮/解壓正常")
else:
    print("❌ FAIL - VByte 解壓錯誤")

# Gamma
gamma = GammaEncoder()
encoded = gamma.encode_gaps(test_doc_ids)
decoded = gamma.decode_gaps(encoded)
print(f"  Gamma: {test_doc_ids} → {encoded} → {decoded}")
if decoded == test_doc_ids:
    print("✅ PASS - Gamma 壓縮/解壓正常")
else:
    print("❌ FAIL - Gamma 解壓錯誤")

# Delta
delta = DeltaEncoder()
encoded = delta.encode_gaps(test_doc_ids)
decoded = delta.decode_gaps(encoded)
print(f"  Delta: {test_doc_ids} → {encoded} → {decoded}")
if decoded == test_doc_ids:
    print("✅ PASS - Delta 壓縮/解壓正常")
else:
    print("❌ FAIL - Delta 解壓錯誤")
print()

# Summary
print("=" * 80)
print("測試完成！")
print("=" * 80)
print()
print("測試覆蓋範圍:")
print("  ✅ 分詞與文本處理")
print("  ✅ 索引 (倒排、位置、壓縮)")
print("  ✅ 檢索模型 (VSM, BM25, LM, Boolean)")
print("  ✅ 混合排序 (Hybrid Ranking)")
print("  ✅ 語言模型 (N-gram, Collocation)")
print("  ✅ 進階功能 (Query Expansion, Clustering, Summarization, KWIC)")
print()
print("所有核心模組測試完成！✅")
