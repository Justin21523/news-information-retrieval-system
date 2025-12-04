#!/usr/bin/env python
"""
Information Retrieval System Comprehensive Test

Tests all IR system components:
- CKIP Tokenization Performance
- BM25/VSM/Boolean Search
- Field Queries
- Content Summarization

Usage:
    python scripts/test_ir_system.py --index-dir data/index_50k

Author: Information Retrieval System
"""

import sys
import time
import logging
from pathlib import Path
from typing import List, Dict
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ir.search import (
    UnifiedSearchEngine,
    QueryMode,
    RankingModel
)


class IRSystemTester:
    """Comprehensive IR System Tester."""

    def __init__(self, index_dir: str = 'data/index_50k'):
        """
        Initialize tester.

        Args:
            index_dir: Path to index directory
        """
        self.logger = logging.getLogger(__name__)
        self.index_dir = index_dir
        self.engine = None
        self.results = {
            'ckip_performance': {},
            'search_tests': [],
            'errors': []
        }

    def setup(self):
        """Initialize search engine."""
        print("=" * 80)
        print("IR System Comprehensive Test")
        print("=" * 80)

        print(f"\n[1/7] Loading search engine from {self.index_dir}...")
        try:
            self.engine = UnifiedSearchEngine(index_dir=self.index_dir)

            # Check if indexed
            if not self.engine.is_indexed:
                print("   ✗ Index not found!")
                print(f"   Please build index first:")
                print(f"   python scripts/search_news.py --build --limit 50000 --index-dir {self.index_dir}")
                return False

            # Get stats
            stats = self.engine.get_stats()
            print(f"   ✓ Search engine loaded successfully")
            print(f"   Documents: {stats.get('total_documents', 0):,}")
            print(f"   Unique terms: {stats.get('unique_terms', 0):,}")
            print(f"   Index size: {stats.get('index_size_mb', 0):.2f} MB")

            return True

        except Exception as e:
            self.logger.error(f"Setup failed: {e}", exc_info=True)
            self.results['errors'].append(f"Setup: {str(e)}")
            return False

    def test_ckip_performance(self):
        """Test CKIP tokenization performance."""
        print(f"\n[2/7] Testing CKIP Tokenization Performance...")
        print("-" * 80)

        test_texts = [
            "台灣是一個美麗的島嶼，擁有豐富的自然資源和多元文化。",
            "人工智慧技術在醫療、金融、教育等領域都有廣泛的應用。",
            "新聞報導指出，經濟成長率超出預期，失業率持續下降。",
            "政府宣布將投入大量資源發展綠色能源產業，減少碳排放。",
            "科技公司推出新一代智慧型手機，搭載先進的AI晶片和5G技術。"
        ]

        # Get tokenizer
        from src.ir.text.ckip_tokenizer import get_tokenizer
        tokenizer = get_tokenizer()

        # Single tokenization
        print("\n   Single Tokenization Tests:")
        single_times = []
        for i, text in enumerate(test_texts, 1):
            start = time.time()
            tokens = tokenizer.tokenize(text)
            elapsed = (time.time() - start) * 1000  # milliseconds
            single_times.append(elapsed)
            print(f"   Text {i}: {len(tokens)} tokens in {elapsed:.2f}ms")

        avg_single = sum(single_times) / len(single_times)
        print(f"\n   Average: {avg_single:.2f}ms per text")

        # Batch tokenization
        print("\n   Batch Tokenization Test:")
        start = time.time()
        batch_results = tokenizer.tokenize_batch(test_texts)
        batch_elapsed = (time.time() - start) * 1000
        print(f"   {len(test_texts)} texts in {batch_elapsed:.2f}ms")
        print(f"   Average: {batch_elapsed/len(test_texts):.2f}ms per text")
        print(f"   Speedup: {avg_single/(batch_elapsed/len(test_texts)):.2f}x")

        self.results['ckip_performance'] = {
            'avg_single_ms': avg_single,
            'batch_total_ms': batch_elapsed,
            'batch_avg_ms': batch_elapsed / len(test_texts),
            'speedup': avg_single / (batch_elapsed / len(test_texts))
        }

        print("   ✓ CKIP performance test completed")

    def test_search_modes(self):
        """Test all search modes and ranking models."""
        print(f"\n[3/7] Testing Search Modes and Ranking Models...")
        print("-" * 80)

        test_queries = [
            # Simple queries
            {"query": "台灣 經濟", "mode": QueryMode.SIMPLE, "model": RankingModel.BM25, "desc": "Simple BM25"},
            {"query": "人工智慧", "mode": QueryMode.SIMPLE, "model": RankingModel.VSM, "desc": "Simple VSM"},

            # Boolean queries
            {"query": "台灣 AND 政治", "mode": QueryMode.BOOLEAN, "model": RankingModel.BM25, "desc": "Boolean AND"},
            {"query": "經濟 OR 金融", "mode": QueryMode.BOOLEAN, "model": RankingModel.BM25, "desc": "Boolean OR"},

            # Field queries
            {"query": "title:台灣", "mode": QueryMode.FIELD, "model": RankingModel.BM25, "desc": "Field search (title)"},

            # Hybrid ranking
            {"query": "科技 創新", "mode": QueryMode.SIMPLE, "model": RankingModel.HYBRID, "desc": "Hybrid ranking"},
        ]

        for test in test_queries:
            print(f"\n   Testing: {test['desc']}")
            print(f"   Query: \"{test['query']}\"")

            try:
                start = time.time()
                results = self.engine.search(
                    query=test['query'],
                    mode=test['mode'],
                    ranking_model=test['model'],
                    top_k=5
                )
                elapsed = (time.time() - start) * 1000

                print(f"   Results: {len(results)} documents in {elapsed:.2f}ms")
                if results:
                    print(f"   Top result: {results[0].title[:60]}...")
                    print(f"   Score: {results[0].score:.4f}")

                self.results['search_tests'].append({
                    'query': test['query'],
                    'mode': test['mode'].value,
                    'model': test['model'].value,
                    'num_results': len(results),
                    'time_ms': elapsed,
                    'top_score': results[0].score if results else 0
                })

            except Exception as e:
                print(f"   ✗ Error: {e}")
                self.results['errors'].append(f"{test['desc']}: {str(e)}")

        print("\n   ✓ Search mode tests completed")

    def test_field_queries(self):
        """Test field-specific queries."""
        print(f"\n[4/7] Testing Field Queries...")
        print("-" * 80)

        field_tests = [
            {"query": "title:經濟", "desc": "Title field"},
            {"query": "category:政治", "desc": "Category field"},
            {"query": "source:ltn", "desc": "Source field"},
            {"query": "title:台灣 AND category:政治", "desc": "Multi-field combination"},
        ]

        for test in field_tests:
            print(f"\n   Testing: {test['desc']}")
            print(f"   Query: \"{test['query']}\"")

            try:
                results = self.engine.search(
                    query=test['query'],
                    mode=QueryMode.FIELD,
                    top_k=3
                )

                print(f"   Results: {len(results)} documents")
                for i, result in enumerate(results[:3], 1):
                    print(f"   [{i}] {result.title[:50]}...")
                    print(f"       Fields: {result.matched_fields}")

            except Exception as e:
                print(f"   ✗ Error: {e}")
                self.results['errors'].append(f"Field query {test['desc']}: {str(e)}")

        print("\n   ✓ Field query tests completed")

    def test_content_summarization(self):
        """Test content summarization."""
        print(f"\n[5/7] Testing Content Summarization...")
        print("-" * 80)

        # Search for a document
        query = "台灣 經濟"
        print(f"   Searching for: \"{query}\"")

        try:
            results = self.engine.search(query, top_k=1)

            if results:
                doc = results[0]
                print(f"\n   Document: {doc.title}")
                print(f"   Source: {doc.source}")

                # Show content snippet
                if doc.content:
                    snippet = doc.content[:200]
                    print(f"\n   Content snippet (first 200 chars):")
                    print(f"   {snippet}...")

                # Show matched fields
                print(f"\n   Matched fields: {doc.matched_fields}")
                print(f"   Relevance score: {doc.score:.4f}")
            else:
                print("   No results found")

        except Exception as e:
            print(f"   ✗ Error: {e}")
            self.results['errors'].append(f"Summarization: {str(e)}")

        print("\n   ✓ Content summarization test completed")

    def test_advanced_queries(self):
        """Test advanced query features."""
        print(f"\n[6/7] Testing Advanced Queries...")
        print("-" * 80)

        advanced_tests = [
            {"query": "台灣 AND (經濟 OR 金融)", "desc": "Complex Boolean"},
            {"query": "NOT 政治", "desc": "Boolean NOT"},
            {"query": "title:台灣 AND category:經濟", "desc": "Field combination"},
        ]

        for test in advanced_tests:
            print(f"\n   Testing: {test['desc']}")
            print(f"   Query: \"{test['query']}\"")

            try:
                results = self.engine.search(
                    query=test['query'],
                    mode=QueryMode.AUTO,  # Auto-detect mode
                    top_k=3
                )

                print(f"   Results: {len(results)} documents")
                if results:
                    print(f"   Top: {results[0].title[:60]}...")

            except Exception as e:
                print(f"   ✗ Error: {e}")
                self.results['errors'].append(f"Advanced query {test['desc']}: {str(e)}")

        print("\n   ✓ Advanced query tests completed")

    def print_summary(self):
        """Print test summary."""
        print(f"\n[7/7] Test Summary")
        print("=" * 80)

        # CKIP Performance
        if self.results['ckip_performance']:
            perf = self.results['ckip_performance']
            print(f"\nCKIP Tokenization Performance:")
            print(f"  Single: {perf['avg_single_ms']:.2f}ms")
            print(f"  Batch: {perf['batch_avg_ms']:.2f}ms (speedup: {perf['speedup']:.2f}x)")

        # Search Tests
        if self.results['search_tests']:
            print(f"\nSearch Tests: {len(self.results['search_tests'])} completed")
            avg_time = sum(t['time_ms'] for t in self.results['search_tests']) / len(self.results['search_tests'])
            print(f"  Average query time: {avg_time:.2f}ms")

        # Errors
        if self.results['errors']:
            print(f"\n⚠ Errors: {len(self.results['errors'])}")
            for error in self.results['errors']:
                print(f"  - {error}")
        else:
            print(f"\n✓ All tests passed successfully!")

        print("\n" + "=" * 80)

    def run_all_tests(self):
        """Run all tests."""
        if not self.setup():
            return 1

        try:
            self.test_ckip_performance()
            self.test_search_modes()
            self.test_field_queries()
            self.test_content_summarization()
            self.test_advanced_queries()
            self.print_summary()

            return 0 if not self.results['errors'] else 1

        except Exception as e:
            self.logger.error(f"Test execution failed: {e}", exc_info=True)
            return 1


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Test Information Retrieval System'
    )
    parser.add_argument(
        '--index-dir',
        type=str,
        default='data/index_50k',
        help='Index directory (default: data/index_50k)'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Verbose output (DEBUG level)'
    )

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run tests
    tester = IRSystemTester(index_dir=args.index_dir)
    return tester.run_all_tests()


if __name__ == '__main__':
    sys.exit(main())
