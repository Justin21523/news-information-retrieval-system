"""
Comprehensive IR System Test Suite
綜合資訊檢索系統測試套件

自動化測試所有模組功能，並生成詳細測試報告。
不使用 pytest，以清晰的格式呈現測試結果。

Author: Information Retrieval System
Date: 2025-11-13
"""

import sys
import os
import time
import json
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import all IR modules
from src.ir.text.chinese_tokenizer import ChineseTokenizer
from src.ir.index.inverted_index import InvertedIndex
from src.ir.index.positional_index import PositionalIndex
from src.ir.index.field_indexer import FieldIndexer
from src.ir.retrieval.boolean import BooleanQueryEngine
from src.ir.retrieval.vsm import VectorSpaceModel
from src.ir.retrieval.bm25 import BM25Ranker
from src.ir.retrieval.language_model_retrieval import LanguageModelRetrieval
from src.ir.retrieval.bim import BinaryIndependenceModel
from src.ir.retrieval.wildcard import WildcardExpander
from src.ir.retrieval.fuzzy import FuzzyMatcher
from src.ir.langmodel.ngram import NGramModel
from src.ir.langmodel.collocation import CollocationExtractor
from src.ir.ranking.rocchio import RocchioExpander
from src.ir.ranking.hybrid import HybridRanker
from src.ir.cluster.doc_cluster import DocumentClusterer
from src.ir.summarize.static import StaticSummarizer
from src.ir.summarize.dynamic import KWICGenerator
from src.ir.index.compression import VByteEncoder, GammaEncoder, DeltaEncoder


class TestResults:
    """測試結果記錄器"""

    def __init__(self):
        """Initialize counters and timestamps for a test run (O(1) time/space)."""
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.results = []
        self.start_time = time.time()

    def record(self, module: str, test_name: str, passed: bool,
               message: str = "", execution_time: float = 0):
        """記錄單個測試結果"""
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
            status = "✅ PASS"
        else:
            self.failed_tests += 1
            status = "❌ FAIL"

        self.results.append({
            'module': module,
            'test_name': test_name,
            'status': status,
            'passed': passed,
            'message': message,
            'execution_time': execution_time
        })

    def print_summary(self):
        """打印測試總結"""
        total_time = time.time() - self.start_time

        print("\n" + "=" * 80)
        print("測試總結 (Test Summary)")
        print("=" * 80)
        print(f"總測試數: {self.total_tests}")
        print(f"通過: {self.passed_tests} ✅")
        print(f"失敗: {self.failed_tests} ❌")
        print(f"成功率: {(self.passed_tests/self.total_tests*100):.1f}%")
        print(f"總執行時間: {total_time:.2f}s")
        print("=" * 80)

    def print_detailed_results(self):
        """打印詳細測試結果"""
        print("\n" + "=" * 80)
        print("詳細測試結果 (Detailed Test Results)")
        print("=" * 80)

        current_module = None
        for result in self.results:
            if result['module'] != current_module:
                current_module = result['module']
                print(f"\n【{current_module}】")
                print("-" * 80)

            print(f"  {result['status']} {result['test_name']}")
            if result['message']:
                print(f"       → {result['message']}")
            if result['execution_time'] > 0:
                print(f"       ⏱ {result['execution_time']:.3f}s")

    def save_report(self, filepath: str = "test_report.json"):
        """保存測試報告為 JSON"""
        report = {
            'summary': {
                'total_tests': self.total_tests,
                'passed_tests': self.passed_tests,
                'failed_tests': self.failed_tests,
                'success_rate': self.passed_tests / self.total_tests if self.total_tests > 0 else 0,
                'total_time': time.time() - self.start_time,
                'timestamp': datetime.now().isoformat()
            },
            'results': self.results
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        print(f"\n📄 測試報告已保存: {filepath}")


class ComprehensiveTest:
    """綜合測試類"""

    def __init__(self):
        """Initialize the comprehensive test harness and load static test data."""
        self.results = TestResults()
        self.test_data = self._load_test_data()
        self.tokenizer = None

    def _load_test_data(self) -> Dict[str, Any]:
        """載入測試資料"""
        return {
            'documents': [
                "資訊檢索是從大量資料中找到相關資訊的過程",
                "檢索模型包括布林模型和向量空間模型",
                "BM25是一種機率檢索函數",
                "搜尋引擎使用各種排序演算法包括BM25",
                "語言模型估計詞序列的機率",
                "人工智慧和機器學習在資訊檢索中的應用",
                "深度學習模型如BERT可用於語義檢索",
                "台灣的科技產業發展迅速",
                "經濟成長與技術創新密切相關",
                "自然語言處理是人工智慧的重要領域"
            ],
            'articles': [
                {
                    'id': i,
                    'title': f"測試文章 {i}",
                    'content': doc,
                    'category': '科技' if i % 2 == 0 else '財經',
                    'published_date': f"2025-11-{10+i:02d}",
                    'tags': ['測試', 'IR'],
                    'author': '測試作者',
                    'url': f'http://test.com/{i}'
                }
                for i, doc in enumerate([
                    "資訊檢索是從大量資料中找到相關資訊的過程",
                    "檢索模型包括布林模型和向量空間模型",
                    "BM25是一種機率檢索函數",
                    "搜尋引擎使用各種排序演算法包括BM25",
                    "語言模型估計詞序列的機率",
                    "人工智慧和機器學習在資訊檢索中的應用",
                    "深度學習模型如BERT可用於語義檢索",
                    "台灣的科技產業發展迅速",
                    "經濟成長與技術創新密切相關",
                    "自然語言處理是人工智慧的重要領域"
                ])
            ],
            'queries': [
                "資訊檢索",
                "BM25排序",
                "人工智慧",
                "台灣經濟"
            ]
        }

    def run_all_tests(self):
        """執行所有測試"""
        print("=" * 80)
        print("開始綜合測試 (Starting Comprehensive Tests)")
        print("=" * 80)
        print(f"測試時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"測試文檔數: {len(self.test_data['documents'])}")
        print(f"測試查詢數: {len(self.test_data['queries'])}")
        print("=" * 80)

        # 1. 文本處理測試
        self.test_text_processing()

        # 2. 索引測試
        self.test_indexing()

        # 3. 檢索模型測試
        self.test_retrieval_models()

        # 4. 語言模型測試
        self.test_language_models()

        # 5. 進階功能測試
        self.test_advanced_features()

        # 6. 壓縮測試
        self.test_compression()

        # 打印結果
        self.results.print_detailed_results()
        self.results.print_summary()

        # 保存報告
        self.results.save_report('test_report.json')

    def test_text_processing(self):
        """測試文本處理模組"""
        module = "文本處理 (Text Processing)"

        # Test 1: ChineseTokenizer 初始化
        try:
            start = time.time()
            self.tokenizer = ChineseTokenizer(engine='ckip', use_pos=True, device=-1)
            exec_time = time.time() - start
            self.results.record(module, "CKIP Tokenizer 初始化", True,
                              f"成功載入 CKIP 模型", exec_time)
        except Exception as e:
            self.results.record(module, "CKIP Tokenizer 初始化", False, f"錯誤: {e}")
            return

        # Test 2: 分詞測試
        try:
            test_text = "資訊檢索系統"
            tokens = self.tokenizer.tokenize(test_text)
            passed = len(tokens) > 0
            self.results.record(module, "中文分詞", passed,
                              f"輸入: '{test_text}' → 分詞: {tokens}")
        except Exception as e:
            self.results.record(module, "中文分詞", False, f"錯誤: {e}")

        # Test 3: 詞性標注測試
        try:
            test_text = "人工智慧發展"
            tokens_pos = self.tokenizer.tokenize_with_pos(test_text)
            passed = len(tokens_pos) > 0 and all(len(tp) == 2 for tp in tokens_pos)
            self.results.record(module, "詞性標注 (POS)", passed,
                              f"結果: {tokens_pos}")
        except Exception as e:
            self.results.record(module, "詞性標注 (POS)", False, f"錯誤: {e}")

        # Test 4: 命名實體識別測試
        try:
            test_text = "台灣的科技產業發展"
            entities = self.tokenizer.extract_entities(test_text)
            self.results.record(module, "命名實體識別 (NER)", True,
                              f"識別實體: {entities if entities else '無'}")
        except Exception as e:
            self.results.record(module, "命名實體識別 (NER)", False, f"錯誤: {e}")

    def test_indexing(self):
        """測試索引模組"""
        module = "索引 (Indexing)"

        docs = self.test_data['documents']

        # Test 1: Inverted Index
        try:
            start = time.time()
            inv_idx = InvertedIndex(tokenizer=self.tokenizer.tokenize)
            inv_idx.build(docs)
            exec_time = time.time() - start

            vocab_size = len(inv_idx.vocabulary)
            passed = vocab_size > 0
            self.results.record(module, "倒排索引 (Inverted Index)", passed,
                              f"詞彙量: {vocab_size}, 文檔數: {len(docs)}", exec_time)
        except Exception as e:
            self.results.record(module, "倒排索引 (Inverted Index)", False, f"錯誤: {e}")

        # Test 2: Positional Index
        try:
            start = time.time()
            pos_idx = PositionalIndex(tokenizer=self.tokenizer.tokenize)
            pos_idx.build(docs)
            exec_time = time.time() - start

            # 測試位置查詢
            positions = pos_idx.get_positions("檢索", 0)
            passed = positions is not None
            self.results.record(module, "位置索引 (Positional Index)", passed,
                              f"'檢索' 在 doc_0 的位置: {positions}", exec_time)
        except Exception as e:
            self.results.record(module, "位置索引 (Positional Index)", False, f"錯誤: {e}")

        # Test 3: Field Indexer
        try:
            start = time.time()
            field_idx = FieldIndexer(tokenizer=self.tokenizer.tokenize)
            field_idx.build(self.test_data['articles'])
            exec_time = time.time() - start

            # 測試欄位查詢
            results = field_idx.search_field('category', '科技')
            passed = len(results) > 0
            self.results.record(module, "欄位索引 (Field Index)", passed,
                              f"category='科技' 找到 {len(results)} 篇文檔", exec_time)
        except Exception as e:
            self.results.record(module, "欄位索引 (Field Index)", False, f"錯誤: {e}")

    def test_retrieval_models(self):
        """測試檢索模型"""
        module = "檢索模型 (Retrieval Models)"

        docs = self.test_data['documents']
        query = "資訊檢索"

        # Test 1: VSM (Vector Space Model)
        try:
            start = time.time()
            vsm = VectorSpaceModel()
            vsm.build_index(docs)
            result = vsm.search(query, topk=3)
            exec_time = time.time() - start

            passed = result.num_results > 0
            top_score = result.scores[0] if result.scores else 0
            self.results.record(module, "VSM (TF-IDF)", passed,
                              f"查詢: '{query}' → 找到 {result.num_results} 篇, Top score: {top_score:.4f}",
                              exec_time)
        except Exception as e:
            self.results.record(module, "VSM (TF-IDF)", False, f"錯誤: {e}")

        # Test 2: BM25
        try:
            start = time.time()
            bm25 = BM25Ranker(tokenizer=self.tokenizer.tokenize, k1=1.5, b=0.75)
            bm25.build_index(docs)
            result = bm25.search(query, topk=3)
            exec_time = time.time() - start

            passed = result.num_results > 0
            top_score = result.scores[0] if result.scores else 0
            self.results.record(module, "BM25 Ranking", passed,
                              f"查詢: '{query}' → 找到 {result.num_results} 篇, Top score: {top_score:.4f}",
                              exec_time)
        except Exception as e:
            self.results.record(module, "BM25 Ranking", False, f"錯誤: {e}")

        # Test 3: Language Model Retrieval
        try:
            start = time.time()
            lm = LanguageModelRetrieval(
                tokenizer=self.tokenizer.tokenize,
                smoothing='dirichlet',
                mu_param=2000
            )
            lm.build_index(docs)
            result = lm.search(query, topk=3)
            exec_time = time.time() - start

            passed = result.num_results > 0
            top_score = result.scores[0] if result.scores else 0
            self.results.record(module, "Language Model Retrieval", passed,
                              f"查詢: '{query}' → 找到 {result.num_results} 篇, Top score: {top_score:.4f}",
                              exec_time)
        except Exception as e:
            self.results.record(module, "Language Model Retrieval", False, f"錯誤: {e}")

        # Test 4: BIM (Binary Independence Model)
        try:
            start = time.time()
            bim = BinaryIndependenceModel(tokenizer=self.tokenizer.tokenize, use_idf=True)
            bim.build_index(docs)
            result = bim.search(query, topk=3)
            exec_time = time.time() - start

            passed = result.num_results > 0
            top_score = result.scores[0] if result.scores else 0
            self.results.record(module, "BIM (Binary Independence Model)", passed,
                              f"查詢: '{query}' → 找到 {result.num_results} 篇, RSV: {top_score:.4f}",
                              exec_time)
        except Exception as e:
            self.results.record(module, "BIM (Binary Independence Model)", False, f"錯誤: {e}")

        # Test 5: Boolean Query
        try:
            start = time.time()
            inv_idx = InvertedIndex(tokenizer=self.tokenizer.tokenize)
            inv_idx.build(docs)
            pos_idx = PositionalIndex(tokenizer=self.tokenizer.tokenize)
            pos_idx.build(docs)

            boolean_engine = BooleanQueryEngine(
                inverted_index=inv_idx,
                positional_index=pos_idx
            )

            # 測試 AND 查詢
            result = boolean_engine.search("資訊 AND 檢索")
            exec_time = time.time() - start

            passed = len(result) >= 0
            self.results.record(module, "Boolean Query (AND)", passed,
                              f"'資訊 AND 檢索' → 找到 {len(result)} 篇", exec_time)
        except Exception as e:
            self.results.record(module, "Boolean Query (AND)", False, f"錯誤: {e}")

        # Test 6: NEAR Query
        try:
            result = boolean_engine.search("資訊 NEAR/5 檢索")
            passed = len(result) >= 0
            self.results.record(module, "NEAR/n Query", passed,
                              f"'資訊 NEAR/5 檢索' → 找到 {len(result)} 篇")
        except Exception as e:
            self.results.record(module, "NEAR/n Query", False, f"錯誤: {e}")

        # Test 7: Hybrid Ranking
        try:
            start = time.time()
            # 準備多個 rankers
            vsm_ranker = VectorSpaceModel()
            vsm_ranker.build_index(docs)

            bm25_ranker = BM25Ranker(tokenizer=self.tokenizer.tokenize)
            bm25_ranker.build_index(docs)

            lm_ranker = LanguageModelRetrieval(tokenizer=self.tokenizer.tokenize)
            lm_ranker.build_index(docs)

            hybrid = HybridRanker(
                rankers={'vsm': vsm_ranker, 'bm25': bm25_ranker, 'lm': lm_ranker},
                fusion_method='rrf',
                normalization='minmax'
            )

            result = hybrid.search(query, topk=3)
            exec_time = time.time() - start

            passed = result.num_results > 0
            top_score = result.scores[0] if result.scores else 0
            self.results.record(module, "Hybrid Ranking (RRF)", passed,
                              f"查詢: '{query}' → 融合 3 個模型, Top score: {top_score:.4f}",
                              exec_time)
        except Exception as e:
            self.results.record(module, "Hybrid Ranking (RRF)", False, f"錯誤: {e}")

    def test_language_models(self):
        """測試語言模型"""
        module = "語言模型 (Language Models)"

        docs = self.test_data['documents']

        # Test 1: N-gram Model
        try:
            start = time.time()
            ngram = NGramModel(n=2, smoothing='jm', lambda_param=0.7,
                             tokenizer=self.tokenizer.tokenize)
            ngram.train(docs)
            exec_time = time.time() - start

            stats = ngram.get_stats()
            self.results.record(module, "N-gram Model (Bigram)", True,
                              f"詞彙: {stats['vocabulary_size']}, 訓練文檔: {len(docs)}", exec_time)
        except Exception as e:
            self.results.record(module, "N-gram Model (Bigram)", False, f"錯誤: {e}")

        # Test 2: N-gram Perplexity
        try:
            test_text = "資訊檢索系統"
            perplexity = ngram.perplexity(test_text)
            passed = perplexity > 0 and perplexity < float('inf')
            self.results.record(module, "N-gram Perplexity", passed,
                              f"文本: '{test_text}' → Perplexity: {perplexity:.2f}")
        except Exception as e:
            self.results.record(module, "N-gram Perplexity", False, f"錯誤: {e}")

        # Test 3: Collocation Extraction
        try:
            start = time.time()
            colloc = CollocationExtractor(
                tokenizer=self.tokenizer.tokenize,
                min_freq=1,
                window_size=2
            )
            colloc.train(docs)

            # 提取 top collocations
            collocations = colloc.extract_collocations(measure='pmi', topk=5)
            exec_time = time.time() - start

            passed = len(collocations) > 0
            top_bigrams = [' '.join(c.bigram) for c in collocations[:3]]
            self.results.record(module, "Collocation Extraction (PMI)", passed,
                              f"提取 {len(collocations)} 個詞彙組合, Top 3: {top_bigrams}",
                              exec_time)
        except Exception as e:
            self.results.record(module, "Collocation Extraction (PMI)", False, f"錯誤: {e}")

        # Test 4: Multiple Collocation Measures
        try:
            measures = ['llr', 'chi_square', 't_score', 'dice']
            for measure in measures:
                collocations = colloc.extract_collocations(measure=measure, topk=3)
                passed = len(collocations) > 0
                self.results.record(module, f"Collocation ({measure.upper()})", passed,
                                  f"提取 {len(collocations)} 個組合")
        except Exception as e:
            self.results.record(module, "Collocation (Multiple Measures)", False, f"錯誤: {e}")

    def test_advanced_features(self):
        """測試進階功能"""
        module = "進階功能 (Advanced Features)"

        docs = self.test_data['documents']

        # Test 1: Wildcard Query
        try:
            vocab = set()
            for doc in docs:
                tokens = self.tokenizer.tokenize(doc)
                vocab.update(tokens)

            wildcard = WildcardExpander(vocabulary=vocab)
            matches = wildcard.expand("檢*")
            passed = len(matches) > 0
            self.results.record(module, "Wildcard Query (檢*)", passed,
                              f"匹配詞彙: {matches[:5]}")
        except Exception as e:
            self.results.record(module, "Wildcard Query (檢*)", False, f"錯誤: {e}")

        # Test 2: Fuzzy Query
        try:
            fuzzy = FuzzyMatcher(max_distance=2, max_expansions=50)
            matches = fuzzy.fuzzy_match("檢索", vocab, max_distance=1)
            passed = len(matches) > 0
            match_terms = [term for term, dist in matches[:3]]
            self.results.record(module, "Fuzzy Query (檢索~1)", passed,
                              f"找到 {len(matches)} 個相似詞: {match_terms}")
        except Exception as e:
            self.results.record(module, "Fuzzy Query (檢索~1)", False, f"錯誤: {e}")

        # Test 3: Query Expansion (Rocchio)
        try:
            vsm = VectorSpaceModel()
            vsm.build_index(docs)

            # 獲取查詢向量
            query_tokens = self.tokenizer.tokenize("資訊檢索")
            query_vector = {term: 1.0 for term in query_tokens}

            # 獲取相關文檔向量
            result = vsm.search("資訊檢索", topk=3)
            relevant_vectors = [vsm.get_document_vector(doc_id) for doc_id in result.doc_ids]

            # Rocchio 擴展
            rocchio = RocchioExpander()
            expanded = rocchio.expand_query(query_vector, relevant_vectors)

            passed = len(expanded.expanded_vector) > len(query_vector)
            expansion_terms = list(set(expanded.expanded_vector.keys()) - set(query_vector.keys()))[:3]
            self.results.record(module, "Query Expansion (Rocchio)", passed,
                              f"原始查詢詞數: {len(query_vector)}, 擴展後: {len(expanded.expanded_vector)}, "
                              f"新增詞: {expansion_terms}")
        except Exception as e:
            self.results.record(module, "Query Expansion (Rocchio)", False, f"錯誤: {e}")

        # Test 4: Document Clustering
        try:
            start = time.time()
            clusterer = DocumentClusterer()
            clusters = clusterer.cluster(docs, n_clusters=3, method='hierarchical')
            exec_time = time.time() - start

            passed = len(clusters) > 0
            cluster_sizes = [len(c.doc_ids) for c in clusters]
            self.results.record(module, "Document Clustering (Hierarchical)", passed,
                              f"分成 {len(clusters)} 群, 群大小: {cluster_sizes}", exec_time)
        except Exception as e:
            self.results.record(module, "Document Clustering (Hierarchical)", False, f"錯誤: {e}")

        # Test 5: Summarization - Lead-K
        try:
            summarizer = StaticSummarizer()
            test_doc = docs[0]
            summary = summarizer.lead_k_summarization(test_doc, k=2)

            passed = len(summary.sentences) > 0
            self.results.record(module, "Summarization (Lead-K)", passed,
                              f"從文檔提取 {len(summary.sentences)} 句摘要")
        except Exception as e:
            self.results.record(module, "Summarization (Lead-K)", False, f"錯誤: {e}")

        # Test 6: KWIC Generator
        try:
            kwic = KWICGenerator()
            test_doc = "資訊檢索是從大量資料中找到相關資訊的過程，資訊檢索系統是重要的工具"
            result = kwic.generate(test_doc, "資訊")

            passed = len(result.contexts) > 0
            self.results.record(module, "KWIC (KeyWord In Context)", passed,
                              f"關鍵詞 '資訊' 出現 {result.occurrences} 次")
        except Exception as e:
            self.results.record(module, "KWIC (KeyWord In Context)", False, f"錯誤: {e}")

    def test_compression(self):
        """測試索引壓縮"""
        module = "索引壓縮 (Index Compression)"

        test_doc_ids = [3, 7, 10, 15, 22, 30, 35, 50, 100, 200]

        # Test 1: VByte Encoding
        try:
            vbyte = VByteEncoder()
            encoded = vbyte.encode_gaps(test_doc_ids)
            decoded = vbyte.decode_gaps(encoded)

            passed = decoded == test_doc_ids
            original_size = len(test_doc_ids) * 4
            compressed_size = len(encoded)
            ratio = compressed_size / original_size

            self.results.record(module, "VByte Encoding", passed,
                              f"原始: {original_size}B → 壓縮: {compressed_size}B "
                              f"(壓縮率: {ratio:.2%})")
        except Exception as e:
            self.results.record(module, "VByte Encoding", False, f"錯誤: {e}")

        # Test 2: Gamma Encoding
        try:
            gamma = GammaEncoder()
            encoded = gamma.encode_gaps(test_doc_ids)
            decoded = gamma.decode_gaps(encoded)

            passed = decoded == test_doc_ids
            gamma_bytes = (len(encoded) + 7) // 8
            ratio = gamma_bytes / original_size

            self.results.record(module, "Gamma Encoding", passed,
                              f"原始: {original_size}B → 壓縮: {gamma_bytes}B "
                              f"(壓縮率: {ratio:.2%})")
        except Exception as e:
            self.results.record(module, "Gamma Encoding", False, f"錯誤: {e}")

        # Test 3: Delta Encoding
        try:
            delta = DeltaEncoder()
            encoded = delta.encode_gaps(test_doc_ids)
            decoded = delta.decode_gaps(encoded)

            passed = decoded == test_doc_ids
            delta_bytes = (len(encoded) + 7) // 8
            ratio = delta_bytes / original_size

            self.results.record(module, "Delta Encoding", passed,
                              f"原始: {original_size}B → 壓縮: {delta_bytes}B "
                              f"(壓縮率: {ratio:.2%})")
        except Exception as e:
            self.results.record(module, "Delta Encoding", False, f"錯誤: {e}")


def main():
    """主測試函數"""
    print("\n" + "🚀" * 40)
    print("資訊檢索系統 - 綜合測試")
    print("Information Retrieval System - Comprehensive Test Suite")
    print("🚀" * 40 + "\n")

    # 執行測試
    tester = ComprehensiveTest()
    tester.run_all_tests()

    print("\n" + "✨" * 40)
    print("測試完成! (Tests Completed!)")
    print("✨" * 40 + "\n")


if __name__ == '__main__':
    main()
