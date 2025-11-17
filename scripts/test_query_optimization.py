#!/usr/bin/env python3
"""
Test script for Query Optimization APIs (WAND and MaxScore)

This script tests the WAND and MaxScore query optimization algorithms
and compares their performance with traditional search methods.

Usage:
    python scripts/test_query_optimization.py
"""

import requests
import json
import time
from typing import Dict, Any, List


BASE_URL = "http://localhost:5001"


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_results(response: requests.Response, show_full: bool = False):
    """Print formatted search results."""
    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        if show_full:
            print(json.dumps(data, indent=2, ensure_ascii=False))
        else:
            print(f"Query: {data.get('query', 'N/A')}")
            print(f"Algorithm: {data.get('algorithm', 'N/A')}")
            print(f"Total results: {data.get('total', 0)}")
            print(f"Execution time: {data.get('execution_time', 0):.4f}s")

            if 'statistics' in data:
                stats = data['statistics']
                print(f"\nOptimization Statistics:")
                print(f"  Candidate documents: {stats.get('num_candidate_docs', 0)}")
                print(f"  Scored documents: {stats.get('num_scored_docs', 0)}")
                print(f"  Speedup ratio: {stats.get('speedup_ratio', 1.0)}x")

            if 'results' in data and data['results']:
                print(f"\nTop 5 results:")
                for i, result in enumerate(data['results'][:5], 1):
                    print(f"  {i}. [{result['doc_id']}] {result['title'][:50]}...")
                    print(f"     Score: {result['score']:.4f}, Category: {result.get('category', 'N/A')}")
    else:
        print(f"Error: {response.text}")


def test_wand_search():
    """Test WAND optimized search."""
    print_section("1. WAND Search - Single Term Query")

    query = "人工智慧"
    payload = {
        "query": query,
        "limit": 10
    }

    print(f"Query: {query}")
    print(f"Request: POST {BASE_URL}/api/search/wand\n")

    response = requests.post(f"{BASE_URL}/api/search/wand", json=payload)
    print_results(response)


def test_wand_multi_term():
    """Test WAND with multi-term query."""
    print_section("2. WAND Search - Multi-Term Query")

    query = "台灣經濟發展趨勢"
    payload = {
        "query": query,
        "limit": 10
    }

    print(f"Query: {query}")
    print(f"Request: POST {BASE_URL}/api/search/wand\n")

    response = requests.post(f"{BASE_URL}/api/search/wand", json=payload)
    print_results(response)


def test_maxscore_search():
    """Test MaxScore optimized search."""
    print_section("3. MaxScore Search - Single Term")

    query = "科技"
    payload = {
        "query": query,
        "limit": 10
    }

    print(f"Query: {query}")
    print(f"Request: POST {BASE_URL}/api/search/maxscore\n")

    response = requests.post(f"{BASE_URL}/api/search/maxscore", json=payload)
    print_results(response)


def test_maxscore_multi_term():
    """Test MaxScore with complex query."""
    print_section("4. MaxScore Search - Complex Query")

    query = "深度學習神經網路應用"
    payload = {
        "query": query,
        "limit": 10
    }

    print(f"Query: {query}")
    print(f"Request: POST {BASE_URL}/api/search/maxscore\n")

    response = requests.post(f"{BASE_URL}/api/search/maxscore", json=payload)
    print_results(response)


def compare_algorithms():
    """Compare WAND, MaxScore, and traditional search."""
    print_section("5. Algorithm Comparison")

    query = "資訊檢索系統"
    limit = 10

    print(f"Query: {query}")
    print(f"Top-k: {limit}\n")

    # Test traditional BM25
    print("--- BM25 (Baseline) ---")
    start = time.time()
    bm25_response = requests.post(f"{BASE_URL}/api/search/bm25", json={
        "query": query,
        "limit": limit
    })
    bm25_time = time.time() - start

    if bm25_response.status_code == 200:
        bm25_data = bm25_response.json()
        print(f"Results: {len(bm25_data.get('results', []))}")
        print(f"Time: {bm25_time:.4f}s")
        print(f"Candidates: All documents (baseline)")
    else:
        print(f"Error: {bm25_response.text}")

    # Test WAND
    print("\n--- WAND ---")
    start = time.time()
    wand_response = requests.post(f"{BASE_URL}/api/search/wand", json={
        "query": query,
        "limit": limit
    })
    wand_time = time.time() - start

    if wand_response.status_code == 200:
        wand_data = wand_response.json()
        stats = wand_data.get('statistics', {})
        print(f"Results: {len(wand_data.get('results', []))}")
        print(f"Time: {wand_time:.4f}s")
        print(f"Candidates: {stats.get('num_candidate_docs', 0)}")
        print(f"Scored: {stats.get('num_scored_docs', 0)}")
        print(f"Speedup: {stats.get('speedup_ratio', 1.0)}x")
    else:
        print(f"Error: {wand_response.text}")

    # Test MaxScore
    print("\n--- MaxScore ---")
    start = time.time()
    maxscore_response = requests.post(f"{BASE_URL}/api/search/maxscore", json={
        "query": query,
        "limit": limit
    })
    maxscore_time = time.time() - start

    if maxscore_response.status_code == 200:
        maxscore_data = maxscore_response.json()
        stats = maxscore_data.get('statistics', {})
        print(f"Results: {len(maxscore_data.get('results', []))}")
        print(f"Time: {maxscore_time:.4f}s")
        print(f"Candidates: {stats.get('num_candidate_docs', 0)}")
        print(f"Scored: {stats.get('num_scored_docs', 0)}")
        print(f"Speedup: {stats.get('speedup_ratio', 1.0)}x")
    else:
        print(f"Error: {maxscore_response.text}")

    # Summary
    print("\n--- Summary ---")
    print(f"BM25 time:      {bm25_time:.4f}s (baseline)")
    print(f"WAND time:      {wand_time:.4f}s ({(bm25_time/wand_time if wand_time > 0 else 0):.2f}x faster)")
    print(f"MaxScore time:  {maxscore_time:.4f}s ({(bm25_time/maxscore_time if maxscore_time > 0 else 0):.2f}x faster)")


def test_various_queries():
    """Test optimization with various query types."""
    print_section("6. Various Query Types")

    queries = [
        ("單字", "氣候"),
        ("雙字", "氣候變遷"),
        ("多字", "氣候變遷環境保護"),
        ("長查詢", "全球暖化氣候變遷對經濟影響分析")
    ]

    results_comparison = []

    for query_type, query in queries:
        print(f"\n{query_type}: {query}")
        print("-" * 60)

        # WAND
        wand_resp = requests.post(f"{BASE_URL}/api/search/wand", json={
            "query": query,
            "limit": 10
        })

        if wand_resp.status_code == 200:
            wand_data = wand_resp.json()
            wand_stats = wand_data.get('statistics', {})
            wand_speedup = wand_stats.get('speedup_ratio', 1.0)
            wand_scored = wand_stats.get('num_scored_docs', 0)
        else:
            wand_speedup = 0
            wand_scored = 0

        # MaxScore
        maxscore_resp = requests.post(f"{BASE_URL}/api/search/maxscore", json={
            "query": query,
            "limit": 10
        })

        if maxscore_resp.status_code == 200:
            maxscore_data = maxscore_resp.json()
            maxscore_stats = maxscore_data.get('statistics', {})
            maxscore_speedup = maxscore_stats.get('speedup_ratio', 1.0)
            maxscore_scored = maxscore_stats.get('num_scored_docs', 0)
        else:
            maxscore_speedup = 0
            maxscore_scored = 0

        print(f"  WAND:     Speedup={wand_speedup:.2f}x, Scored={wand_scored}")
        print(f"  MaxScore: Speedup={maxscore_speedup:.2f}x, Scored={maxscore_scored}")

        results_comparison.append({
            'query_type': query_type,
            'query': query,
            'wand_speedup': wand_speedup,
            'maxscore_speedup': maxscore_speedup
        })

    # Print comparison table
    print("\n" + "=" * 80)
    print(f"{'Query Type':<15} {'WAND Speedup':<20} {'MaxScore Speedup':<20}")
    print("-" * 80)
    for result in results_comparison:
        print(f"{result['query_type']:<15} {result['wand_speedup']:<20.2f} {result['maxscore_speedup']:<20.2f}")


def test_top_k_sensitivity():
    """Test performance with different top-k values."""
    print_section("7. Top-K Sensitivity Analysis")

    query = "人工智慧機器學習"
    top_k_values = [5, 10, 20, 50, 100]

    print(f"Query: {query}")
    print(f"Testing top-k values: {top_k_values}\n")

    print(f"{'Top-K':<10} {'WAND Speedup':<20} {'MaxScore Speedup':<20}")
    print("-" * 60)

    for k in top_k_values:
        # WAND
        wand_resp = requests.post(f"{BASE_URL}/api/search/wand", json={
            "query": query,
            "limit": k
        })

        if wand_resp.status_code == 200:
            wand_stats = wand_resp.json().get('statistics', {})
            wand_speedup = wand_stats.get('speedup_ratio', 1.0)
        else:
            wand_speedup = 0

        # MaxScore
        maxscore_resp = requests.post(f"{BASE_URL}/api/search/maxscore", json={
            "query": query,
            "limit": k
        })

        if maxscore_resp.status_code == 200:
            maxscore_stats = maxscore_resp.json().get('statistics', {})
            maxscore_speedup = maxscore_stats.get('speedup_ratio', 1.0)
        else:
            maxscore_speedup = 0

        print(f"{k:<10} {wand_speedup:<20.2f} {maxscore_speedup:<20.2f}")


def run_all_tests():
    """Run all query optimization tests."""
    print("\n" + "=" * 80)
    print("  QUERY OPTIMIZATION API TESTS (WAND & MaxScore)")
    print("=" * 80)
    print(f"\nTesting API at: {BASE_URL}")
    print("Ensure the Flask server is running: python app.py\n")

    tests = [
        ("WAND - Single Term", test_wand_search),
        ("WAND - Multi-Term", test_wand_multi_term),
        ("MaxScore - Single Term", test_maxscore_search),
        ("MaxScore - Complex", test_maxscore_multi_term),
        ("Algorithm Comparison", compare_algorithms),
        ("Various Query Types", test_various_queries),
        ("Top-K Sensitivity", test_top_k_sensitivity),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            start_time = time.time()
            test_func()
            elapsed = time.time() - start_time
            results.append((test_name, "PASS", elapsed))
        except Exception as e:
            results.append((test_name, f"FAIL: {str(e)}", 0))

    # Print summary
    print_section("TEST SUMMARY")

    print(f"{'Test Name':<30} {'Status':<15} {'Time (s)':<10}")
    print("-" * 55)

    for test_name, status, elapsed in results:
        status_str = "✓ PASS" if status == "PASS" else f"✗ {status}"
        time_str = f"{elapsed:.3f}" if elapsed > 0 else "N/A"
        print(f"{test_name:<30} {status_str:<15} {time_str:<10}")

    passed = sum(1 for _, status, _ in results if status == "PASS")
    total = len(results)

    print("-" * 55)
    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        # Quick comparison mode
        print_section("Quick Algorithm Comparison")
        compare_algorithms()
    else:
        # Full test suite
        run_all_tests()
