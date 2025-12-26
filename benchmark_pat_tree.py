#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PAT-tree Performance Benchmark Script
測試並分析PAT-tree的性能瓶頸
"""

import time
import psutil
import sys
from pathlib import Path
from typing import List, Dict
import json
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.ir.index.pat_tree import PatriciaTree
from src.ir.text.chinese_tokenizer import ChineseTokenizer


class PerformanceBenchmark:
    """PAT-tree性能基準測試"""

    def __init__(self):
        """Initialize the benchmark runner and capture the current process handle."""
        self.results = {}
        self.process = psutil.Process()

    def load_test_documents(self, limit: int = 121) -> List[Dict]:
        """載入測試文檔"""
        print(f"📄 載入測試文檔 (limit={limit})...")
        preprocessed_file = project_root / 'data' / 'preprocessed' / 'cna_mvp_preprocessed.jsonl'

        documents = []
        with open(preprocessed_file, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= limit:
                    break
                if line.strip():
                    doc = json.loads(line)
                    documents.append({
                        'doc_id': doc.get('article_id', ''),
                        'content': doc.get('content', '')
                    })

        print(f"✓ 載入 {len(documents)} 篇文檔")
        return documents

    def measure_build_time(self, documents: List[Dict]) -> Dict:
        """測量建構時間"""
        print("\n🔨 測試PAT-tree建構性能...")

        tokenizer = ChineseTokenizer(engine='jieba')
        tree = PatriciaTree()

        # 記錄初始記憶體
        mem_before = self.process.memory_info().rss / 1024 / 1024

        # 測量建構時間
        start_time = time.time()
        total_tokens = 0

        for doc in documents:
            tokens = tokenizer.tokenize(doc['content'])
            total_tokens += len(tokens)

            for token in tokens:
                tree.insert(token, doc_id=doc['doc_id'])

        build_time = time.time() - start_time

        # 記錄最終記憶體
        mem_after = self.process.memory_info().rss / 1024 / 1024
        mem_used = mem_after - mem_before

        stats = tree.get_statistics()

        result = {
            'build_time_sec': round(build_time, 2),
            'tokens_per_sec': int(total_tokens / build_time),
            'memory_mb': round(mem_used, 2),
            'total_terms': stats['total_terms'],
            'unique_terms': stats['unique_terms'],
            'compression_ratio': round(stats['compression_ratio'], 2)
        }

        print(f"✓ 建構時間: {result['build_time_sec']}秒")
        print(f"  處理速度: {result['tokens_per_sec']:,} tokens/sec")
        print(f"  記憶體使用: {result['memory_mb']} MB")
        print(f"  壓縮率: {result['compression_ratio']}x")

        return tree, result

    def measure_query_performance(self, tree: PatriciaTree) -> Dict:
        """測量查詢性能"""
        print("\n🔍 測試查詢性能...")

        # 測試prefix queries
        test_prefixes = ["台", "中", "美", "經", "政", "科", "教", "文", "資", "新"]

        prefix_times = []
        for prefix in test_prefixes:
            start = time.time()
            results = tree.starts_with(prefix)
            elapsed = (time.time() - start) * 1000  # ms
            prefix_times.append(elapsed)
            print(f"  prefix='{prefix}': {elapsed:.2f}ms ({len(results)} matches)")

        # 測試keyword extraction
        methods = ['tfidf', 'frequency', 'combined']
        extraction_times = {}

        for method in methods:
            start = time.time()
            keywords = tree.extract_keywords(top_k=20, method=method)
            elapsed = (time.time() - start) * 1000
            extraction_times[method] = round(elapsed, 2)
            print(f"  {method}: {elapsed:.2f}ms")

        result = {
            'prefix_search': {
                'mean_ms': round(np.mean(prefix_times), 2),
                'p50_ms': round(np.percentile(prefix_times, 50), 2),
                'p95_ms': round(np.percentile(prefix_times, 95), 2),
                'p99_ms': round(np.percentile(prefix_times, 99), 2),
            },
            'keyword_extraction': extraction_times
        }

        return result

    def measure_visualization(self, tree: PatriciaTree) -> Dict:
        """測量視覺化性能"""
        print("\n🎨 測試視覺化性能...")

        # Test full tree
        start = time.time()
        viz_data = tree.visualize_tree(max_nodes=100, prefix="")
        time_full = (time.time() - start) * 1000

        # Test with prefix
        start = time.time()
        viz_data = tree.visualize_tree(max_nodes=50, prefix="台")
        time_prefix = (time.time() - start) * 1000

        print(f"  Full tree (100 nodes): {time_full:.2f}ms")
        print(f"  With prefix (50 nodes): {time_prefix:.2f}ms")

        return {
            'full_tree_ms': round(time_full, 2),
            'prefix_tree_ms': round(time_prefix, 2)
        }

    def generate_report(self, results: Dict):
        """生成性能報告"""
        print("\n" + "=" * 60)
        print("📊 PAT-tree 性能測試報告")
        print("=" * 60)

        print("\n### 建構性能 (Build Performance)")
        build = results['build']
        print(f"  建構時間: {build['build_time_sec']} 秒")
        print(f"  處理速度: {build['tokens_per_sec']:,} tokens/sec")
        print(f"  記憶體使用: {build['memory_mb']} MB")
        print(f"  壓縮率: {build['compression_ratio']}x")

        print("\n### 查詢性能 (Query Performance)")
        query = results['query']
        print(f"  Prefix Search (平均): {query['prefix_search']['mean_ms']} ms")
        print(f"  Prefix Search (P95): {query['prefix_search']['p95_ms']} ms")

        print("\n  Keyword Extraction:")
        for method, time_ms in query['keyword_extraction'].items():
            print(f"    {method}: {time_ms} ms")

        print("\n### 視覺化性能 (Visualization)")
        viz = results['visualization']
        print(f"  Full tree: {viz['full_tree_ms']} ms")
        print(f"  Prefix tree: {viz['prefix_tree_ms']} ms")

        print("\n### 性能評分 (Performance Score)")
        # 計算綜合分數 (越低越好)
        build_score = build['build_time_sec'] / 30  # 目標30秒
        query_score = query['prefix_search']['mean_ms'] / 10  # 目標10ms
        memory_score = build['memory_mb'] / 100  # 目標100MB

        total_score = (build_score + query_score + memory_score) / 3

        if total_score < 0.8:
            grade = "A (優秀)"
        elif total_score < 1.2:
            grade = "B (良好)"
        elif total_score < 1.5:
            grade = "C (可接受)"
        else:
            grade = "D (需優化)"

        print(f"  綜合評分: {grade}")

        print("\n" + "=" * 60)

        # Save report
        report_file = project_root / 'performance_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n📁 詳細報告已儲存至: {report_file}")

    def run(self):
        """執行完整性能測試"""
        print("🚀 PAT-tree 性能基準測試")
        print("=" * 60)

        # Load documents
        documents = self.load_test_documents(limit=121)

        # Test build performance
        tree, build_results = self.measure_build_time(documents)
        self.results['build'] = build_results

        # Test query performance
        query_results = self.measure_query_performance(tree)
        self.results['query'] = query_results

        # Test visualization
        viz_results = self.measure_visualization(tree)
        self.results['visualization'] = viz_results

        # Generate report
        self.generate_report(self.results)


if __name__ == '__main__':
    benchmark = PerformanceBenchmark()
    benchmark.run()
