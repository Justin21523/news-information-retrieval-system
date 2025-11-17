"""
Quick Test Suite for IR System v2
快速測試套件 - 修正版，使用正確的API接口
"""

import sys
from pathlib import Path

# Add parent directory to path
_parent_dir = Path(__file__).parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))

# Import IR modules
from src.ir.text.chinese_tokenizer import ChineseTokenizer
from src.ir.index.inverted_index import InvertedIndex
from src.ir.index.positional_index import PositionalIndex
from src.ir.retrieval.vsm import VectorSpaceModel
from src.ir.retrieval.bm25 import BM25Ranker
from src.ir.retrieval.language_model_retrieval import LanguageModelRetrieval
from src.ir.retrieval.bim import BinaryIndependenceModel
from src.ir.retrieval.boolean import BooleanQueryEngine
from src.ir.ranking.hybrid import HybridRanker
from src.ir.langmodel.ngram import NGramModel
from src.ir.langmodel.collocation import CollocationExtractor
from src.ir.ranking.rocchio import RocchioExpander
from src.ir.cluster.doc_cluster import DocumentClusterer
from src.ir.summarize.static import StaticSummarizer
from src.ir.summarize.dynamic import KWICGenerator
from src.ir.index.compression import VByteEncoder, GammaEncoder, DeltaEncoder

print("=" * 80)
print("IR 系統快速測試套件 v2 (Quick Test Suite)")
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

# Results tracking
tests_passed = 0
tests_failed = 0

def test_result(name, passed, details=""):
    global tests_passed, tests_failed
    if passed:
        tests_passed += 1
        print(f"✅ PASS - {name}")
    else:
        tests_failed += 1
        print(f"❌ FAIL - {name}")
    if details:
        print(f"       {details}")

# Test 1: Tokenizer
print("【1. 分詞器】")
try:
    tokenizer = ChineseTokenizer(engine='jieba')
    test_text = "資訊檢索系統"
    tokens = tokenizer.tokenize(test_text)
    test_result("Jieba 分詞", len(tokens) > 0, f"'{test_text}' → {tokens}")
except Exception as e:
    test_result("Jieba 分詞", False, str(e))
print()

# Test 2: Inverted Index
print("【2. 倒排索引】")
try:
    inv_idx = InvertedIndex(tokenizer=tokenizer.tokenize)
    inv_idx.build(test_docs)
    term_count = len(inv_idx.index)
    test_result("倒排索引建立", term_count > 0, f"{len(test_docs)} docs, {term_count} terms")
except Exception as e:
    test_result("倒排索引建立", False, str(e))
print()

# Test 3: Positional Index
print("【3. 位置索引】")
try:
    pos_idx = PositionalIndex(tokenizer=tokenizer.tokenize)
    pos_idx.build(test_docs)
    positions = pos_idx.get_positions("檢索", 0)
    test_result("位置索引", positions is not None, f"'檢索' 在 doc 0: positions={positions}")
except Exception as e:
    test_result("位置索引", False, str(e))
print()

# Test 4: VSM
print("【4. 向量空間模型 (VSM)】")
try:
    vsm = VectorSpaceModel()
    vsm.build_index(test_docs)
    query = "資訊檢索"
    result = vsm.search(query, topk=3)
    test_result("VSM 檢索", len(result) > 0, f"查詢 '{query}': {len(result)} 結果")
    for i, (doc_id, score) in enumerate(result[:2], 1):
        print(f"       #{i}. Doc {doc_id}: score={score:.4f}")
except Exception as e:
    test_result("VSM 檢索", False, str(e))
print()

# Test 5: BM25
print("【5. BM25 排序】")
try:
    bm25 = BM25Ranker(tokenizer=tokenizer.tokenize, k1=1.5, b=0.75)
    bm25.build_index(test_docs)
    result = bm25.search(query, topk=3)
    test_result("BM25 檢索", len(result) > 0, f"查詢 '{query}': {len(result)} 結果")
    for i, (doc_id, score) in enumerate(result[:2], 1):
        print(f"       #{i}. Doc {doc_id}: score={score:.4f}")
except Exception as e:
    test_result("BM25 檢索", False, str(e))
print()

# Test 6: Language Model
print("【6. 語言模型檢索】")
try:
    lm = LanguageModelRetrieval(tokenizer=tokenizer.tokenize, smoothing='dirichlet', mu_param=2000)
    lm.build_index(test_docs)
    result = lm.search(query, topk=3)
    test_result("LM 檢索", len(result) > 0, f"查詢 '{query}': {len(result)} 結果")
    for i, (doc_id, score) in enumerate(result[:2], 1):
        print(f"       #{i}. Doc {doc_id}: score={score:.4f}")
except Exception as e:
    test_result("LM 檢索", False, str(e))
print()

# Test 7: BIM
print("【7. 二元獨立模型 (BIM)】")
try:
    bim = BinaryIndependenceModel(tokenizer=tokenizer.tokenize, use_idf=True)
    bim.build_index(test_docs)
    result = bim.search(query, topk=3)
    test_result("BIM 檢索", len(result) > 0, f"查詢 '{query}': {len(result)} 結果")
except Exception as e:
    test_result("BIM 檢索", False, str(e))
print()

# Test 8: Boolean Query
print("【8. 布林檢索】")
try:
    boolean_engine = BooleanQueryEngine(inverted_index=inv_idx, positional_index=pos_idx)
    bool_query = "資訊 AND 檢索"
    result = boolean_engine.search(bool_query)
    test_result("Boolean AND", len(result) > 0, f"查詢 '{bool_query}': {result}")
except Exception as e:
    test_result("Boolean AND", False, str(e))
print()

# Test 9: Hybrid Ranking
print("【9. 混合排序】")
try:
    rankers = {'bm25': bm25, 'vsm': vsm, 'lm': lm}
    hybrid = HybridRanker(rankers=rankers, fusion_method='rrf')
    result = hybrid.search(query, topk=3, ranker_topk=10)
    test_result("Hybrid RRF", len(result) > 0, f"融合3個rankers: {len(result)} 結果")
    for i, (doc_id, score) in enumerate(result[:2], 1):
        print(f"       #{i}. Doc {doc_id}: score={score:.4f}")
except Exception as e:
    test_result("Hybrid RRF", False, str(e))
print()

# Test 10: N-gram Model
print("【10. N-gram 語言模型】")
try:
    ngram = NGramModel(n=2, smoothing='jm', lambda_param=0.7)
    ngram.train(test_docs)
    test_sentence = "資訊檢索"
    prob = ngram.probability(test_sentence)
    perplexity = ngram.perplexity(test_sentence)
    test_result("N-gram", prob > 0, f"'{test_sentence}': prob={prob:.2e}, perplexity={perplexity:.2f}")
except Exception as e:
    test_result("N-gram", False, str(e))
print()

# Test 11: Collocation
print("【11. 詞彙共現分析】")
try:
    colloc = CollocationExtractor(tokenizer=tokenizer.tokenize, min_freq=1, window_size=2)
    colloc.build(test_docs)
    collocations = colloc.extract_collocations(measure='pmi', topk=5)
    test_result("Collocation PMI", len(collocations) > 0, f"發現 {len(collocations)} 個共現詞組")
    for i, col in enumerate(collocations[:2], 1):
        print(f"       #{i}. {col['bigram']}: PMI={col['pmi']:.2f}, freq={col['freq']}")
except Exception as e:
    test_result("Collocation PMI", False, str(e))
print()

# Test 12: Rocchio Query Expansion
print("【12. 查詢擴展 (Rocchio)】")
try:
    rocchio = RocchioExpander(alpha=1.0, beta=0.75, gamma=0.15, max_expansion_terms=3)
    query_vec = {"資訊": 0.8, "檢索": 0.6}
    rel_vecs = [
        {"資訊": 0.5, "檢索": 0.7, "系統": 0.3},
        {"資訊": 0.6, "文檔": 0.4}
    ]
    expanded = rocchio.expand_query(query_vec, rel_vecs)
    test_result("Rocchio 擴展", len(expanded.all_terms) >= len(expanded.original_terms),
                f"{len(expanded.original_terms)} → {len(expanded.all_terms)} terms")
    print(f"       原始: {expanded.original_terms}")
    print(f"       擴展: {expanded.expanded_terms}")
except Exception as e:
    test_result("Rocchio 擴展", False, str(e))
print()

# Test 13: Document Clustering
print("【13. 文檔分群】")
try:
    clusterer = DocumentClusterer(tokenizer=tokenizer.tokenize, method='kmeans')
    clusters = clusterer.cluster(test_docs, n_clusters=3)
    test_result("K-means 分群", len(clusters) > 0, f"分成 {len(clusters)} 群")
    for i, cluster in enumerate(clusters):
        print(f"       Cluster {i}: {cluster['size']} docs")
except Exception as e:
    test_result("K-means 分群", False, str(e))
print()

# Test 14: Summarization
print("【14. 文檔摘要】")
try:
    summarizer = StaticSummarizer()
    doc = test_docs[0]
    summary = summarizer.lead_k_summarization(doc, k=1)
    test_result("Lead-K 摘要", len(summary) > 0, f"摘要長度: {len(summary)}")
except Exception as e:
    test_result("Lead-K 摘要", False, str(e))
print()

# Test 15: KWIC
print("【15. KWIC (關鍵詞上下文)】")
try:
    kwic = KWICGenerator(window_size=5)
    doc = "資訊檢索是計算機科學的重要分支，檢索系統用於查找資訊"
    keyword = "檢索"
    result = kwic.generate(doc, keyword)
    test_result("KWIC 生成", len(result['contexts']) > 0,
                f"找到 {len(result['contexts'])} 個 '{keyword}' 上下文")
    for ctx in result['contexts'][:1]:
        print(f"       ...{ctx['left']}【{ctx['keyword']}】{ctx['right']}...")
except Exception as e:
    test_result("KWIC 生成", False, str(e))
print()

# Test 16: Index Compression
print("【16. 索引壓縮】")
test_doc_ids = [3, 7, 10, 15, 22, 30]

try:
    vbyte = VByteEncoder()
    encoded = vbyte.encode_gaps(test_doc_ids)
    decoded = vbyte.decode_gaps(encoded)
    test_result("VByte 壓縮", decoded == test_doc_ids,
                f"{len(test_doc_ids)} ids → {len(encoded)} bytes")
except Exception as e:
    test_result("VByte 壓縮", False, str(e))

try:
    gamma = GammaEncoder()
    encoded = gamma.encode_gaps(test_doc_ids)
    decoded = gamma.decode_gaps(encoded)
    test_result("Gamma 壓縮", decoded == test_doc_ids, f"編碼/解碼正確")
except Exception as e:
    test_result("Gamma 壓縮", False, str(e))

try:
    delta = DeltaEncoder()
    encoded = delta.encode_gaps(test_doc_ids)
    decoded = delta.decode_gaps(encoded)
    test_result("Delta 壓縮", decoded == test_doc_ids, f"編碼/解碼正確")
except Exception as e:
    test_result("Delta 壓縮", False, str(e))
print()

# Summary
print("=" * 80)
print("測試總結")
print("=" * 80)
total_tests = tests_passed + tests_failed
pass_rate = (tests_passed / total_tests * 100) if total_tests > 0 else 0
print(f"總測試數: {total_tests}")
print(f"通過: {tests_passed} ✅")
print(f"失敗: {tests_failed} ❌")
print(f"成功率: {pass_rate:.1f}%")
print()

if tests_failed == 0:
    print("🎉 所有測試通過！系統運作正常！")
else:
    print(f"⚠️  有 {tests_failed} 個測試失敗，請檢查上述錯誤訊息")
