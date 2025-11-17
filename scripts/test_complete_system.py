#!/usr/bin/env python3
"""
Complete System Integration Test
完整系統整合測試

This script tests ALL 29 API endpoints to ensure the entire IR system
is functioning correctly after integration.

測試所有 29 個 API 端點,確保整個資訊檢索系統整合後正常運作。

Usage:
    python scripts/test_complete_system.py
    python scripts/test_complete_system.py --quick  # Quick test (essential APIs only)
    python scripts/test_complete_system.py --verbose  # Detailed output
"""

import requests
import json
import time
import sys
from typing import Dict, Any, List, Tuple
from datetime import datetime


BASE_URL = "http://localhost:5001"
VERBOSE = False


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(title: str):
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  {title}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 80}{Colors.RESET}\n")


def print_section(title: str):
    """Print a section title."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{title}{Colors.RESET}")
    print(f"{Colors.BLUE}{'-' * 60}{Colors.RESET}")


def print_success(message: str):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {message}{Colors.RESET}")


def print_error(message: str):
    """Print error message."""
    print(f"{Colors.RED}✗ {message}{Colors.RESET}")


def print_warning(message: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {message}{Colors.RESET}")


def test_api(endpoint: str, method: str = "GET", data: Dict = None,
             params: Dict = None, expected_keys: List[str] = None) -> Tuple[bool, str, float]:
    """
    Test a single API endpoint.

    Args:
        endpoint: API endpoint path
        method: HTTP method (GET or POST)
        data: JSON data for POST requests
        params: Query parameters for GET requests
        expected_keys: Expected keys in response JSON

    Returns:
        Tuple of (success, message, response_time)
    """
    url = f"{BASE_URL}{endpoint}"

    try:
        start_time = time.time()

        if method == "GET":
            response = requests.get(url, params=params, timeout=10)
        else:  # POST
            response = requests.post(url, json=data, timeout=10)

        response_time = time.time() - start_time

        if response.status_code != 200:
            return False, f"HTTP {response.status_code}: {response.text[:100]}", response_time

        # Check JSON response
        try:
            json_data = response.json()
        except json.JSONDecodeError:
            return False, "Invalid JSON response", response_time

        # Check expected keys
        if expected_keys:
            missing_keys = [key for key in expected_keys if key not in json_data]
            if missing_keys:
                return False, f"Missing keys: {missing_keys}", response_time

        # Check for error in response
        if 'error' in json_data:
            return False, f"API returned error: {json_data['error']}", response_time

        if VERBOSE:
            print(f"  Response preview: {json.dumps(json_data, ensure_ascii=False)[:200]}...")

        return True, "OK", response_time

    except requests.exceptions.Timeout:
        return False, "Request timeout (>10s)", 0
    except requests.exceptions.ConnectionError:
        return False, "Connection failed - is server running?", 0
    except Exception as e:
        return False, f"Exception: {str(e)}", 0


def test_system_apis() -> List[Tuple[str, bool, str, float]]:
    """Test system APIs."""
    print_section("1. System APIs (1 endpoint)")

    results = []

    # Test stats
    success, msg, time_taken = test_api(
        "/api/stats",
        method="GET",
        expected_keys=['documents', 'vocabulary_size']
    )
    results.append(("GET /api/stats", success, msg, time_taken))

    if success:
        print_success(f"/api/stats - {time_taken:.3f}s")
    else:
        print_error(f"/api/stats - {msg}")

    return results


def test_retrieval_apis() -> List[Tuple[str, bool, str, float]]:
    """Test all retrieval APIs (7 endpoints)."""
    print_section("2. Retrieval APIs (7 endpoints)")

    results = []
    test_query = "人工智慧"

    retrieval_tests = [
        ("POST /api/search/boolean", "/api/search/boolean", {
            "query": f"{test_query}",
            "limit": 5
        }, ['query', 'results', 'total']),

        ("POST /api/search/vsm", "/api/search/vsm", {
            "query": test_query,
            "limit": 5
        }, ['query', 'results', 'total', 'execution_time']),

        ("POST /api/search/bm25", "/api/search/bm25", {
            "query": test_query,
            "limit": 5
        }, ['query', 'results', 'total']),

        ("POST /api/search/lm", "/api/search/lm", {
            "query": test_query,
            "limit": 5
        }, ['query', 'results', 'total']),

        ("POST /api/search/hybrid", "/api/search/hybrid", {
            "query": test_query,
            "limit": 5,
            "fusion_method": "rrf"
        }, ['query', 'results', 'total', 'fusion_method']),

        ("POST /api/search/wand", "/api/search/wand", {
            "query": test_query,
            "limit": 5
        }, ['query', 'results', 'statistics']),

        ("POST /api/search/maxscore", "/api/search/maxscore", {
            "query": test_query,
            "limit": 5
        }, ['query', 'results', 'statistics']),
    ]

    for name, endpoint, data, expected_keys in retrieval_tests:
        success, msg, time_taken = test_api(endpoint, "POST", data, expected_keys=expected_keys)
        results.append((name, success, msg, time_taken))

        if success:
            print_success(f"{name} - {time_taken:.3f}s")
        else:
            print_error(f"{name} - {msg}")

    return results


def test_recommendation_apis() -> List[Tuple[str, bool, str, float]]:
    """Test all recommendation APIs (9 endpoints)."""
    print_section("3. Recommendation APIs (9 endpoints)")

    results = []

    recommendation_tests = [
        ("POST /api/recommend/similar", "/api/recommend/similar", {
            "doc_id": 0,
            "top_k": 5,
            "apply_diversity": True
        }, ['recommendations', 'method']),

        ("POST /api/recommend/personalized", "/api/recommend/personalized", {
            "reading_history": [0, 1, 2],
            "top_k": 5
        }, ['recommendations', 'method']),

        ("GET /api/recommend/trending", "/api/recommend/trending", None,
         {'top_k': 5, 'time_window_hours': 168}, ['recommendations']),

        ("POST /api/recommend/cf/user-based", "/api/recommend/cf/user-based", {
            "user_id": 0,
            "top_k": 5
        }, ['recommendations', 'method']),

        ("POST /api/recommend/cf/item-based", "/api/recommend/cf/item-based", {
            "user_id": 0,
            "top_k": 5
        }, ['recommendations', 'method']),

        ("POST /api/recommend/cf/matrix-factorization", "/api/recommend/cf/matrix-factorization", {
            "user_id": 0,
            "top_k": 5,
            "method": "svd"
        }, ['recommendations', 'method']),

        ("POST /api/recommend/hybrid", "/api/recommend/hybrid", {
            "user_id": 0,
            "top_k": 5,
            "fusion_method": "weighted"
        }, ['recommendations', 'method']),

        ("POST /api/interaction/record", "/api/interaction/record", {
            "user_id": 0,
            "doc_id": 5,
            "interaction_type": "read",
            "duration": 30.5
        }, ['status']),

        ("GET /api/interaction/history", "/api/interaction/history", None,
         {'user_id': 0, 'limit': 10}, ['interactions']),
    ]

    for test_data in recommendation_tests:
        if len(test_data) == 4:
            name, endpoint, data, expected_keys = test_data
            success, msg, time_taken = test_api(endpoint, "POST", data, expected_keys=expected_keys)
        else:
            name, endpoint, data, params, expected_keys = test_data
            success, msg, time_taken = test_api(endpoint, "GET", params=params, expected_keys=expected_keys)

        results.append((name, success, msg, time_taken))

        if success:
            print_success(f"{name} - {time_taken:.3f}s")
        else:
            print_error(f"{name} - {msg}")

    return results


def test_nlp_apis() -> List[Tuple[str, bool, str, float]]:
    """Test all NLP APIs (5 endpoints)."""
    print_section("4. NLP & Text Analysis APIs (5 endpoints)")

    results = []
    test_text = "人工智慧和機器學習是現代科技的重要發展領域"

    nlp_tests = [
        ("POST /api/extract/keywords", "/api/extract/keywords", {
            "text": test_text,
            "method": "textrank",
            "topk": 5
        }, ['keywords', 'method']),

        ("POST /api/extract/topics", "/api/extract/topics", {
            "texts": [test_text, "深度學習神經網路"],
            "method": "lda",
            "n_topics": 2
        }, ['topics', 'method']),

        ("POST /api/extract/patterns", "/api/extract/patterns", {
            "text": test_text,
            "min_freq": 1,
            "topk": 5
        }, ['patterns']),

        ("POST /api/analyze/ner", "/api/analyze/ner", {
            "text": "台灣位於東亞,首都是台北"
        }, ['entities']),

        ("POST /api/analyze/syntax", "/api/analyze/syntax", {
            "text": test_text,
            "parse_type": "svo"
        }, ['triples']),
    ]

    for name, endpoint, data, expected_keys in nlp_tests:
        success, msg, time_taken = test_api(endpoint, "POST", data, expected_keys=expected_keys)
        results.append((name, success, msg, time_taken))

        if success:
            print_success(f"{name} - {time_taken:.3f}s")
        else:
            print_error(f"{name} - {msg}")

    return results


def test_document_apis() -> List[Tuple[str, bool, str, float]]:
    """Test document-related APIs."""
    print_section("5. Document APIs (4 endpoints)")

    results = []

    # Test get document
    success, msg, time_taken = test_api(
        "/api/document/0",
        method="GET",
        expected_keys=['title', 'content']
    )
    results.append(("GET /api/document/:id", success, msg, time_taken))

    if success:
        print_success(f"GET /api/document/:id - {time_taken:.3f}s")
    else:
        print_error(f"GET /api/document/:id - {msg}")

    # Test document analysis
    success, msg, time_taken = test_api(
        "/api/document/0/analysis",
        method="GET",
        expected_keys=['doc_id']
    )
    results.append(("GET /api/document/:id/analysis", success, msg, time_taken))

    if success:
        print_success(f"GET /api/document/:id/analysis - {time_taken:.3f}s")
    else:
        print_error(f"GET /api/document/:id/analysis - {msg}")

    # Test summarize
    success, msg, time_taken = test_api(
        "/api/summarize/0",
        method="POST",
        data={"method": "lead_k", "k": 2},
        expected_keys=['summary']
    )
    results.append(("POST /api/summarize/:id", success, msg, time_taken))

    if success:
        print_success(f"POST /api/summarize/:id - {time_taken:.3f}s")
    else:
        print_error(f"POST /api/summarize/:id - {msg}")

    # Test query expansion
    success, msg, time_taken = test_api(
        "/api/expand_query",
        method="POST",
        data={
            "query": "人工智慧",
            "mode": "pseudo",
            "topk": 3
        },
        expected_keys=['original_query', 'expanded_query']
    )
    results.append(("POST /api/expand_query", success, msg, time_taken))

    if success:
        print_success(f"POST /api/expand_query - {time_taken:.3f}s")
    else:
        print_error(f"POST /api/expand_query - {msg}")

    return results


def test_language_model_apis() -> List[Tuple[str, bool, str, float]]:
    """Test language model APIs."""
    print_section("6. Language Model APIs (2 endpoints)")

    results = []

    # Test collocation
    success, msg, time_taken = test_api(
        "/api/analyze/collocation",
        method="POST",
        data={"measure": "pmi", "topk": 10},
        expected_keys=['collocations']
    )
    results.append(("POST /api/analyze/collocation", success, msg, time_taken))

    if success:
        print_success(f"POST /api/analyze/collocation - {time_taken:.3f}s")
    else:
        print_error(f"POST /api/analyze/collocation - {msg}")

    # Test n-gram
    success, msg, time_taken = test_api(
        "/api/analyze/ngram",
        method="POST",
        data={"text": "人工智慧機器學習", "n": 2},
        expected_keys=['ngrams']
    )
    results.append(("POST /api/analyze/ngram", success, msg, time_taken))

    if success:
        print_success(f"POST /api/analyze/ngram - {time_taken:.3f}s")
    else:
        print_error(f"POST /api/analyze/ngram - {msg}")

    return results


def generate_test_report(all_results: List[Tuple[str, bool, str, float]]):
    """Generate comprehensive test report."""
    print_header("TEST REPORT - 完整測試報告")

    # Summary statistics
    total_tests = len(all_results)
    passed_tests = sum(1 for _, success, _, _ in all_results if success)
    failed_tests = total_tests - passed_tests

    total_time = sum(time_taken for _, _, _, time_taken in all_results)
    avg_time = total_time / total_tests if total_tests > 0 else 0

    # Print summary
    print(f"\n{Colors.BOLD}Summary:{Colors.RESET}")
    print(f"  Total APIs tested: {total_tests}")
    print(f"  {Colors.GREEN}Passed: {passed_tests}{Colors.RESET}")
    print(f"  {Colors.RED}Failed: {failed_tests}{Colors.RESET}")
    print(f"  Success rate: {(passed_tests/total_tests*100):.1f}%")
    print(f"  Total execution time: {total_time:.2f}s")
    print(f"  Average response time: {avg_time:.3f}s")

    # Detailed results table
    print(f"\n{Colors.BOLD}Detailed Results:{Colors.RESET}\n")
    print(f"{'API Endpoint':<45} {'Status':<10} {'Time (s)':<10} {'Message'}")
    print("-" * 100)

    for name, success, msg, time_taken in all_results:
        status = f"{Colors.GREEN}✓ PASS{Colors.RESET}" if success else f"{Colors.RED}✗ FAIL{Colors.RESET}"
        time_str = f"{time_taken:.3f}" if time_taken > 0 else "N/A"
        msg_short = msg[:40] if not success else "OK"
        print(f"{name:<45} {status:<20} {time_str:<10} {msg_short}")

    print("-" * 100)

    # Performance analysis
    if passed_tests > 0:
        print(f"\n{Colors.BOLD}Performance Analysis:{Colors.RESET}")

        # Fastest APIs
        sorted_by_time = sorted([r for r in all_results if r[1]], key=lambda x: x[3])
        print(f"\n  Fastest APIs (top 5):")
        for name, _, _, time_taken in sorted_by_time[:5]:
            print(f"    {name:<45} {time_taken:.3f}s")

        # Slowest APIs
        print(f"\n  Slowest APIs (top 5):")
        for name, _, _, time_taken in sorted_by_time[-5:][::-1]:
            print(f"    {name:<45} {time_taken:.3f}s")

    # Failed tests detail
    if failed_tests > 0:
        print(f"\n{Colors.BOLD}{Colors.RED}Failed Tests Detail:{Colors.RESET}")
        for name, success, msg, _ in all_results:
            if not success:
                print(f"  {Colors.RED}✗{Colors.RESET} {name}")
                print(f"    Reason: {msg}\n")

    # Final verdict
    print(f"\n{Colors.BOLD}{'=' * 80}{Colors.RESET}")
    if failed_tests == 0:
        print(f"{Colors.GREEN}{Colors.BOLD}🎉 ALL TESTS PASSED! System is fully operational.{Colors.RESET}")
    elif passed_tests > total_tests * 0.9:
        print(f"{Colors.YELLOW}{Colors.BOLD}⚠️  MOSTLY PASSED - {failed_tests} test(s) failed. Review failures above.{Colors.RESET}")
    else:
        print(f"{Colors.RED}{Colors.BOLD}❌ MULTIPLE FAILURES - System may not be fully operational.{Colors.RESET}")
    print(f"{Colors.BOLD}{'=' * 80}{Colors.RESET}\n")

    return passed_tests == total_tests


def run_quick_test():
    """Run quick test of essential APIs only."""
    print_header("QUICK SYSTEM TEST - 快速系統測試")
    print("Testing essential APIs only...\n")

    essential_tests = [
        ("System Stats", "/api/stats", "GET", None, None),
        ("Boolean Search", "/api/search/boolean", "POST", {"query": "台灣", "limit": 5}, None),
        ("VSM Search", "/api/search/vsm", "POST", {"query": "經濟", "limit": 5}, None),
        ("WAND Search", "/api/search/wand", "POST", {"query": "科技", "limit": 5}, None),
        ("Similar Docs", "/api/recommend/similar", "POST", {"doc_id": 0, "top_k": 5}, None),
        ("CF Recommendation", "/api/recommend/cf/item-based", "POST", {"user_id": 0, "top_k": 5}, None),
        ("Keyword Extract", "/api/extract/keywords", "POST", {"text": "人工智慧", "method": "textrank"}, None),
    ]

    results = []
    for name, endpoint, method, data, params in essential_tests:
        if method == "GET":
            success, msg, time_taken = test_api(endpoint, method, params=params)
        else:
            success, msg, time_taken = test_api(endpoint, method, data)

        results.append((name, success, msg, time_taken))

        if success:
            print_success(f"{name:<25} {time_taken:.3f}s")
        else:
            print_error(f"{name:<25} {msg}")

    # Quick summary
    passed = sum(1 for _, s, _, _ in results if s)
    total = len(results)

    print(f"\n{Colors.BOLD}Quick Test Result:{Colors.RESET}")
    print(f"  {passed}/{total} essential APIs working")

    if passed == total:
        print(f"  {Colors.GREEN}✓ System appears operational{Colors.RESET}\n")
        return True
    else:
        print(f"  {Colors.RED}✗ Some essential APIs failed{Colors.RESET}\n")
        return False


def main():
    """Main test execution."""
    global VERBOSE

    # Parse arguments
    quick_mode = "--quick" in sys.argv
    VERBOSE = "--verbose" in sys.argv

    if quick_mode:
        success = run_quick_test()
        sys.exit(0 if success else 1)

    # Full system test
    print_header("COMPLETE SYSTEM INTEGRATION TEST")
    print(f"Testing all 29 API endpoints...")
    print(f"Server: {BASE_URL}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Check if server is running
    try:
        response = requests.get(f"{BASE_URL}/api/stats", timeout=5)
        if response.status_code != 200:
            print_error("Server is not responding correctly!")
            print_warning("Please start the server with: python app.py")
            sys.exit(1)
    except requests.exceptions.ConnectionError:
        print_error("Cannot connect to server!")
        print_warning("Please start the server with: python app.py")
        sys.exit(1)

    print_success("Server is running\n")

    # Run all tests
    all_results = []

    all_results.extend(test_system_apis())
    all_results.extend(test_retrieval_apis())
    all_results.extend(test_recommendation_apis())
    all_results.extend(test_nlp_apis())
    all_results.extend(test_document_apis())
    all_results.extend(test_language_model_apis())

    # Generate report
    success = generate_test_report(all_results)

    # Save report to file
    report_file = "test_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("COMPLETE SYSTEM TEST REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        for name, success_flag, msg, time_taken in all_results:
            status = "PASS" if success_flag else "FAIL"
            f.write(f"{name:<45} {status:<10} {time_taken:.3f}s  {msg}\n")

    print(f"\nDetailed report saved to: {report_file}")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
