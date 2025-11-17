#!/usr/bin/env python3
"""
Test script for Recommendation System APIs
Tests content-based, collaborative filtering, hybrid, and interaction tracking APIs.

Usage:
    python scripts/test_recommendation_apis.py
"""

import requests
import json
import time
from typing import Dict, Any


BASE_URL = "http://localhost:5001"


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def print_response(response: requests.Response, show_full: bool = False):
    """Print formatted API response."""
    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        if show_full:
            print(json.dumps(data, indent=2, ensure_ascii=False))
        else:
            # Print summary
            if 'recommendations' in data:
                print(f"Method: {data.get('method', 'N/A')}")
                print(f"Recommendations returned: {len(data['recommendations'])}")
                print(f"Computation time: {data.get('computation_time', 'N/A')}s")

                if data['recommendations']:
                    print("\nTop 3 recommendations:")
                    for i, rec in enumerate(data['recommendations'][:3], 1):
                        print(f"  {i}. Doc {rec['doc_id']}: {rec.get('title', 'N/A')[:50]}...")
                        print(f"     Score: {rec['score']}, Reason: {rec.get('reason', 'N/A')}")
            elif 'status' in data:
                print(f"Status: {data['status']}")
                print(f"Message: {data.get('message', 'N/A')}")
            elif 'interactions' in data:
                print(f"User ID: {data.get('user_id', 'N/A')}")
                print(f"Total interactions: {data.get('total', 0)}")
                print(f"Returned: {data.get('returned', 0)}")
            else:
                print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        print(f"Error: {response.text}")


def test_content_based_similar():
    """Test content-based similar document recommendations."""
    print_section("1. Content-Based Similar Documents")

    payload = {
        "doc_id": 0,
        "top_k": 10,
        "use_embeddings": False,
        "apply_diversity": True
    }

    print(f"Request: POST {BASE_URL}/api/recommend/similar")
    print(f"Payload: {json.dumps(payload, indent=2)}\n")

    response = requests.post(
        f"{BASE_URL}/api/recommend/similar",
        json=payload
    )

    print_response(response)


def test_content_based_personalized():
    """Test personalized content-based recommendations."""
    print_section("2. Personalized Content-Based Recommendations")

    payload = {
        "reading_history": [0, 1, 5, 10],
        "top_k": 10,
        "use_embeddings": False
    }

    print(f"Request: POST {BASE_URL}/api/recommend/personalized")
    print(f"Payload: {json.dumps(payload, indent=2)}\n")

    response = requests.post(
        f"{BASE_URL}/api/recommend/personalized",
        json=payload
    )

    print_response(response)


def test_trending_recommendations():
    """Test trending document recommendations."""
    print_section("3. Trending Documents")

    params = {
        "top_k": 10,
        "time_window_hours": 168  # Last 7 days
    }

    print(f"Request: GET {BASE_URL}/api/recommend/trending")
    print(f"Params: {params}\n")

    response = requests.get(
        f"{BASE_URL}/api/recommend/trending",
        params=params
    )

    print_response(response)


def test_cf_user_based():
    """Test user-based collaborative filtering."""
    print_section("4. User-Based Collaborative Filtering")

    payload = {
        "user_id": 0,
        "top_k": 10,
        "n_neighbors": 20,
        "similarity_metric": "cosine"
    }

    print(f"Request: POST {BASE_URL}/api/recommend/cf/user-based")
    print(f"Payload: {json.dumps(payload, indent=2)}\n")

    response = requests.post(
        f"{BASE_URL}/api/recommend/cf/user-based",
        json=payload
    )

    print_response(response)


def test_cf_item_based():
    """Test item-based collaborative filtering."""
    print_section("5. Item-Based Collaborative Filtering")

    payload = {
        "user_id": 0,
        "top_k": 10,
        "n_neighbors": 50,
        "similarity_metric": "cosine"
    }

    print(f"Request: POST {BASE_URL}/api/recommend/cf/item-based")
    print(f"Payload: {json.dumps(payload, indent=2)}\n")

    response = requests.post(
        f"{BASE_URL}/api/recommend/cf/item-based",
        json=payload
    )

    print_response(response)


def test_cf_matrix_factorization():
    """Test matrix factorization recommendations."""
    print_section("6. Matrix Factorization (SVD)")

    payload = {
        "user_id": 0,
        "top_k": 10,
        "n_factors": 50,
        "method": "svd"
    }

    print(f"Request: POST {BASE_URL}/api/recommend/cf/matrix-factorization")
    print(f"Payload: {json.dumps(payload, indent=2)}\n")

    response = requests.post(
        f"{BASE_URL}/api/recommend/cf/matrix-factorization",
        json=payload
    )

    print_response(response)

    # Also test ALS
    print("\n--- Testing ALS method ---\n")
    payload['method'] = 'als'

    response = requests.post(
        f"{BASE_URL}/api/recommend/cf/matrix-factorization",
        json=payload
    )

    print_response(response)


def test_hybrid_weighted():
    """Test weighted hybrid recommendations."""
    print_section("7. Hybrid Recommendations - Weighted Fusion")

    payload = {
        "user_id": 0,
        "doc_id": 5,
        "top_k": 10,
        "fusion_method": "weighted",
        "content_weight": 0.5,
        "cf_weight": 0.4,
        "popularity_weight": 0.1,
        "use_embeddings": False
    }

    print(f"Request: POST {BASE_URL}/api/recommend/hybrid")
    print(f"Payload: {json.dumps(payload, indent=2)}\n")

    response = requests.post(
        f"{BASE_URL}/api/recommend/hybrid",
        json=payload
    )

    print_response(response)


def test_hybrid_cascade():
    """Test cascade hybrid recommendations."""
    print_section("8. Hybrid Recommendations - Cascade Fusion")

    payload = {
        "user_id": 0,
        "doc_id": 5,
        "top_k": 10,
        "fusion_method": "cascade",
        "use_embeddings": False
    }

    print(f"Request: POST {BASE_URL}/api/recommend/hybrid")
    print(f"Payload: {json.dumps(payload, indent=2)}\n")

    response = requests.post(
        f"{BASE_URL}/api/recommend/hybrid",
        json=payload
    )

    print_response(response)


def test_hybrid_switching():
    """Test switching hybrid recommendations."""
    print_section("9. Hybrid Recommendations - Switching Strategy")

    payload = {
        "user_id": 0,
        "doc_id": 5,
        "top_k": 10,
        "fusion_method": "switching",
        "use_embeddings": False
    }

    print(f"Request: POST {BASE_URL}/api/recommend/hybrid")
    print(f"Payload: {json.dumps(payload, indent=2)}\n")

    response = requests.post(
        f"{BASE_URL}/api/recommend/hybrid",
        json=payload
    )

    print_response(response)


def test_interaction_recording():
    """Test recording user interactions."""
    print_section("10. User Interaction Recording")

    # Record multiple interactions
    interactions = [
        {"user_id": 0, "doc_id": 5, "interaction_type": "click"},
        {"user_id": 0, "doc_id": 5, "interaction_type": "read", "duration": 45.5},
        {"user_id": 0, "doc_id": 10, "interaction_type": "like"},
        {"user_id": 0, "doc_id": 15, "interaction_type": "share"},
        {"user_id": 1, "doc_id": 5, "interaction_type": "read", "duration": 120.0},
    ]

    print(f"Recording {len(interactions)} interactions...\n")

    for i, interaction in enumerate(interactions, 1):
        print(f"Recording interaction {i}/{len(interactions)}: {interaction}")
        response = requests.post(
            f"{BASE_URL}/api/interaction/record",
            json=interaction
        )

        if response.status_code == 200:
            data = response.json()
            print(f"  ✓ Recorded with ID: {data.get('interaction_id')}")
        else:
            print(f"  ✗ Error: {response.text}")

        time.sleep(0.1)  # Small delay between requests


def test_interaction_history():
    """Test retrieving user interaction history."""
    print_section("11. User Interaction History")

    params = {
        "user_id": 0,
        "limit": 20
    }

    print(f"Request: GET {BASE_URL}/api/interaction/history")
    print(f"Params: {params}\n")

    response = requests.get(
        f"{BASE_URL}/api/interaction/history",
        params=params
    )

    print_response(response, show_full=True)


def test_error_handling():
    """Test API error handling."""
    print_section("12. Error Handling Tests")

    # Test missing required parameter
    print("Test 1: Missing user_id in CF recommendation")
    response = requests.post(
        f"{BASE_URL}/api/recommend/cf/user-based",
        json={"top_k": 10}
    )
    print(f"Status: {response.status_code}, Response: {response.text}\n")

    # Test invalid doc_id
    print("Test 2: Invalid doc_id in similar documents")
    response = requests.post(
        f"{BASE_URL}/api/recommend/similar",
        json={"doc_id": 999999, "top_k": 10}
    )
    print(f"Status: {response.status_code}, Response: {response.text}\n")

    # Test invalid fusion method
    print("Test 3: Invalid fusion method in hybrid")
    response = requests.post(
        f"{BASE_URL}/api/recommend/hybrid",
        json={"user_id": 0, "fusion_method": "invalid_method"}
    )
    print(f"Status: {response.status_code}, Response: {response.text}\n")

    # Test invalid MF method
    print("Test 4: Invalid matrix factorization method")
    response = requests.post(
        f"{BASE_URL}/api/recommend/cf/matrix-factorization",
        json={"user_id": 0, "method": "invalid"}
    )
    print(f"Status: {response.status_code}, Response: {response.text}\n")


def run_all_tests():
    """Run all recommendation API tests."""
    print("\n" + "=" * 80)
    print("  RECOMMENDATION SYSTEM API TESTS")
    print("=" * 80)
    print(f"\nTesting API at: {BASE_URL}")
    print("Ensure the Flask server is running: python app.py\n")

    tests = [
        ("Content-Based Similar", test_content_based_similar),
        ("Personalized Recommendations", test_content_based_personalized),
        ("Trending Documents", test_trending_recommendations),
        ("User-Based CF", test_cf_user_based),
        ("Item-Based CF", test_cf_item_based),
        ("Matrix Factorization", test_cf_matrix_factorization),
        ("Hybrid Weighted", test_hybrid_weighted),
        ("Hybrid Cascade", test_hybrid_cascade),
        ("Hybrid Switching", test_hybrid_switching),
        ("Interaction Recording", test_interaction_recording),
        ("Interaction History", test_interaction_history),
        ("Error Handling", test_error_handling),
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

    print(f"{'Test Name':<35} {'Status':<15} {'Time (s)':<10}")
    print("-" * 60)

    for test_name, status, elapsed in results:
        status_str = "✓ PASS" if status == "PASS" else f"✗ {status}"
        time_str = f"{elapsed:.3f}" if elapsed > 0 else "N/A"
        print(f"{test_name:<35} {status_str:<15} {time_str:<10}")

    passed = sum(1 for _, status, _ in results if status == "PASS")
    total = len(results)

    print("-" * 60)
    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")


def run_quick_demo():
    """Run a quick demonstration of key features."""
    print_section("QUICK DEMO - Key Recommendation Features")

    print("1. Similar Documents (Content-Based)")
    test_content_based_similar()

    print("\n2. Collaborative Filtering (Item-Based)")
    test_cf_item_based()

    print("\n3. Hybrid Recommendations (Weighted)")
    test_hybrid_weighted()

    print("\n4. Record & Retrieve Interactions")
    test_interaction_recording()
    time.sleep(0.5)
    test_interaction_history()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        run_quick_demo()
    else:
        run_all_tests()
