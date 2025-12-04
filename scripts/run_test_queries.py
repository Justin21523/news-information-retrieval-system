#!/usr/bin/env python
"""
Run Test Queries from YAML File

Executes comprehensive test queries against the IR system and generates
performance reports.

Usage:
    python scripts/run_test_queries.py --index-dir data/index_50k
    python scripts/run_test_queries.py --test-file tests/test_queries.yaml --output results.json

Author: Information Retrieval System
"""

import sys
import yaml
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ir.search import (
    UnifiedSearchEngine,
    QueryMode,
    RankingModel
)


class QueryTestRunner:
    """Run and evaluate test queries."""

    def __init__(self, index_dir: str, test_file: str):
        """
        Initialize test runner.

        Args:
            index_dir: Path to index directory
            test_file: Path to YAML test file
        """
        self.logger = logging.getLogger(__name__)
        self.index_dir = index_dir
        self.test_file = test_file

        # Load test cases
        with open(test_file, 'r', encoding='utf-8') as f:
            self.test_cases = yaml.safe_load(f)

        # Results storage
        self.results = {
            'test_run_info': {
                'timestamp': datetime.now().isoformat(),
                'index_dir': index_dir,
                'test_file': test_file
            },
            'summary': defaultdict(int),
            'test_results': []
        }

        # Initialize search engine
        self.logger.info(f"Loading search engine from {index_dir}...")
        self.engine = UnifiedSearchEngine(index_dir=index_dir)

        if not self.engine.is_indexed:
            raise ValueError(f"Index not found at {index_dir}")

        stats = self.engine.get_stats()
        self.results['test_run_info']['index_stats'] = stats

    def run_query(self, query: str, mode: str = 'simple',
                  model: str = 'BM25', top_k: int = 10) -> Dict:
        """
        Run a single query and measure performance.

        Args:
            query: Query string
            mode: Query mode (simple, boolean, field)
            model: Ranking model (BM25, VSM, HYBRID)
            top_k: Number of results to return

        Returns:
            Dictionary with results and metrics
        """
        # Map string mode/model to enums
        mode_map = {
            'simple': QueryMode.SIMPLE,
            'boolean': QueryMode.BOOLEAN,
            'field': QueryMode.FIELD,
            'auto': QueryMode.AUTO
        }

        model_map = {
            'BM25': RankingModel.BM25,
            'VSM': RankingModel.VSM,
            'HYBRID': RankingModel.HYBRID,
            'BOOLEAN': RankingModel.BOOLEAN
        }

        query_mode = mode_map.get(mode, QueryMode.AUTO)
        ranking_model = model_map.get(model, RankingModel.BM25)

        # Execute query with timing
        start_time = time.time()
        try:
            results = self.engine.search(
                query=query,
                mode=query_mode,
                ranking_model=ranking_model,
                top_k=top_k
            )
            elapsed_ms = (time.time() - start_time) * 1000

            return {
                'success': True,
                'num_results': len(results),
                'time_ms': elapsed_ms,
                'results': [
                    {
                        'doc_id': r.doc_id,
                        'title': r.title[:100],
                        'score': r.score,
                        'source': r.source
                    }
                    for r in results[:5]  # Top 5 for review
                ],
                'error': None
            }
        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            self.logger.error(f"Query failed: {e}")
            return {
                'success': False,
                'num_results': 0,
                'time_ms': elapsed_ms,
                'results': [],
                'error': str(e)
            }

    def run_simple_queries(self):
        """Run simple keyword query tests."""
        print("\n" + "="*80)
        print("1. SIMPLE QUERIES TEST")
        print("="*80)

        test_group = self.test_cases.get('simple_queries', [])
        passed = 0
        failed = 0

        for i, test in enumerate(test_group, 1):
            query = test['query']
            desc = test['description']
            min_results = test.get('min_results', 0)

            print(f"\n[{i}/{len(test_group)}] {desc}")
            print(f"   Query: \"{query}\"")

            result = self.run_query(query, mode='simple', model='BM25')

            # Check expectations
            test_passed = (
                result['success'] and
                result['num_results'] >= min_results
            )

            if test_passed:
                print(f"   ✓ PASSED: {result['num_results']} results in {result['time_ms']:.2f}ms")
                passed += 1
            else:
                print(f"   ✗ FAILED: {result['num_results']} results (min: {min_results})")
                failed += 1

            # Store result
            self.results['test_results'].append({
                'category': 'simple_queries',
                'query': query,
                'description': desc,
                'passed': test_passed,
                **result
            })

        self.results['summary']['simple_passed'] = passed
        self.results['summary']['simple_failed'] = failed
        print(f"\nSummary: {passed} passed, {failed} failed")

    def run_boolean_queries(self):
        """Run Boolean query tests."""
        print("\n" + "="*80)
        print("2. BOOLEAN QUERIES TEST")
        print("="*80)

        test_group = self.test_cases.get('boolean_queries', [])
        passed = 0
        failed = 0

        for i, test in enumerate(test_group, 1):
            query = test['query']
            desc = test['description']
            min_results = test.get('min_results', 0)

            print(f"\n[{i}/{len(test_group)}] {desc}")
            print(f"   Query: \"{query}\"")

            result = self.run_query(query, mode='boolean', model='BM25')

            test_passed = (
                result['success'] and
                result['num_results'] >= min_results
            )

            if test_passed:
                print(f"   ✓ PASSED: {result['num_results']} results in {result['time_ms']:.2f}ms")
                passed += 1
            else:
                print(f"   ✗ FAILED: {result['num_results']} results (min: {min_results})")
                failed += 1

            self.results['test_results'].append({
                'category': 'boolean_queries',
                'query': query,
                'description': desc,
                'passed': test_passed,
                **result
            })

        self.results['summary']['boolean_passed'] = passed
        self.results['summary']['boolean_failed'] = failed
        print(f"\nSummary: {passed} passed, {failed} failed")

    def run_field_queries(self):
        """Run field-specific query tests."""
        print("\n" + "="*80)
        print("3. FIELD QUERIES TEST")
        print("="*80)

        test_group = self.test_cases.get('field_queries', [])
        passed = 0
        failed = 0

        for i, test in enumerate(test_group, 1):
            query = test['query']
            desc = test['description']
            min_results = test.get('min_results', 0)

            print(f"\n[{i}/{len(test_group)}] {desc}")
            print(f"   Query: \"{query}\"")

            result = self.run_query(query, mode='field', model='BM25')

            test_passed = (
                result['success'] and
                result['num_results'] >= min_results
            )

            if test_passed:
                print(f"   ✓ PASSED: {result['num_results']} results in {result['time_ms']:.2f}ms")
                passed += 1
            else:
                print(f"   ✗ FAILED: {result['num_results']} results (min: {min_results})")
                failed += 1

            self.results['test_results'].append({
                'category': 'field_queries',
                'query': query,
                'description': desc,
                'passed': test_passed,
                **result
            })

        self.results['summary']['field_passed'] = passed
        self.results['summary']['field_failed'] = failed
        print(f"\nSummary: {passed} passed, {failed} failed")

    def run_performance_tests(self):
        """Run performance benchmark tests."""
        print("\n" + "="*80)
        print("4. PERFORMANCE TESTS")
        print("="*80)

        test_group = self.test_cases.get('performance_tests', [])
        passed = 0
        failed = 0

        for i, test in enumerate(test_group, 1):
            query = test['query']
            desc = test['description']
            max_time_ms = test.get('max_time_ms', 2000)

            print(f"\n[{i}/{len(test_group)}] {desc}")
            print(f"   Query: \"{query}\"")
            print(f"   Max time: {max_time_ms}ms")

            # Run query 3 times and take average
            times = []
            for _ in range(3):
                result = self.run_query(query, mode='auto', model='BM25')
                times.append(result['time_ms'])

            avg_time = sum(times) / len(times)
            test_passed = avg_time <= max_time_ms

            if test_passed:
                print(f"   ✓ PASSED: {avg_time:.2f}ms (max: {max_time_ms}ms)")
                passed += 1
            else:
                print(f"   ✗ FAILED: {avg_time:.2f}ms exceeds {max_time_ms}ms")
                failed += 1

            self.results['test_results'].append({
                'category': 'performance',
                'query': query,
                'description': desc,
                'passed': test_passed,
                'avg_time_ms': avg_time,
                'times': times
            })

        self.results['summary']['performance_passed'] = passed
        self.results['summary']['performance_failed'] = failed
        print(f"\nSummary: {passed} passed, {failed} failed")

    def print_final_summary(self):
        """Print final test summary."""
        print("\n" + "="*80)
        print("FINAL TEST SUMMARY")
        print("="*80)

        summary = self.results['summary']
        total_passed = sum(v for k, v in summary.items() if k.endswith('_passed'))
        total_failed = sum(v for k, v in summary.items() if k.endswith('_failed'))
        total_tests = total_passed + total_failed

        pass_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0

        print(f"\nSimple Queries:      {summary.get('simple_passed', 0)} passed, "
              f"{summary.get('simple_failed', 0)} failed")
        print(f"Boolean Queries:     {summary.get('boolean_passed', 0)} passed, "
              f"{summary.get('boolean_failed', 0)} failed")
        print(f"Field Queries:       {summary.get('field_passed', 0)} passed, "
              f"{summary.get('field_failed', 0)} failed")
        print(f"Performance Tests:   {summary.get('performance_passed', 0)} passed, "
              f"{summary.get('performance_failed', 0)} failed")

        print(f"\n{'='*80}")
        print(f"TOTAL: {total_passed}/{total_tests} passed ({pass_rate:.1f}%)")
        print(f"{'='*80}\n")

        if total_failed == 0:
            print("✓ ALL TESTS PASSED!")
        else:
            print(f"⚠ {total_failed} tests failed")

    def save_results(self, output_file: str):
        """Save test results to JSON file."""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        print(f"\n✓ Results saved to: {output_file}")

    def run_all_tests(self, output_file: str = None):
        """Run all test suites."""
        print("="*80)
        print("IR SYSTEM COMPREHENSIVE QUERY TESTS")
        print("="*80)
        print(f"Index: {self.index_dir}")
        print(f"Test file: {self.test_file}")
        print(f"Documents: {self.results['test_run_info']['index_stats'].get('total_documents', 0):,}")

        try:
            self.run_simple_queries()
            self.run_boolean_queries()
            self.run_field_queries()
            self.run_performance_tests()

            self.print_final_summary()

            if output_file:
                self.save_results(output_file)

            return 0 if self.results['summary'].get('total_failed', 0) == 0 else 1

        except Exception as e:
            self.logger.error(f"Test execution failed: {e}", exc_info=True)
            return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run comprehensive query tests on IR system'
    )
    parser.add_argument(
        '--index-dir',
        type=str,
        default='data/index_50k',
        help='Index directory (default: data/index_50k)'
    )
    parser.add_argument(
        '--test-file',
        type=str,
        default='tests/test_queries.yaml',
        help='Test queries YAML file (default: tests/test_queries.yaml)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='test_results.json',
        help='Output results file (default: test_results.json)'
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
    runner = QueryTestRunner(
        index_dir=args.index_dir,
        test_file=args.test_file
    )
    return runner.run_all_tests(output_file=args.output)


if __name__ == '__main__':
    sys.exit(main())
