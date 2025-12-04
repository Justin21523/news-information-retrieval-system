#!/usr/bin/env python3
"""
Test script for Faceted Search API endpoints.

This script tests the /api/facets and /api/search/faceted endpoints.
"""

import requests
import json


BASE_URL = "http://localhost:5000"


def test_facets_endpoint():
    """Test the /api/facets endpoint."""
    print("=" * 60)
    print("Testing /api/facets endpoint")
    print("=" * 60)

    payload = {
        "query": "台灣",
        "model": "bm25",
        "top_k": 50
    }

    try:
        response = requests.post(f"{BASE_URL}/api/facets", json=payload, timeout=30)

        if response.status_code == 200:
            data = response.json()

            if data.get('success'):
                print("\n✅ /api/facets endpoint working!")
                print(f"Total results: {data.get('total_results')}")
                print(f"\nFacets available: {list(data.get('facets', {}).keys())}")

                # Display source facet details
                if 'source' in data.get('facets', {}):
                    source_facet = data['facets']['source']
                    print(f"\n{source_facet['display_name']} facet:")
                    for fv in source_facet['values'][:5]:
                        print(f"  {fv['value']}: {fv['count']} 篇")

                # Display category facet details
                if 'category' in data.get('facets', {}):
                    category_facet = data['facets']['category']
                    print(f"\n{category_facet['display_name']} facet:")
                    for fv in category_facet['values'][:5]:
                        print(f"  {fv['value']}: {fv['count']} 篇")

                return True
            else:
                print(f"\n❌ API returned error: {data.get('error')}")
                return False
        else:
            print(f"\n❌ HTTP error {response.status_code}")
            print(response.text)
            return False

    except Exception as e:
        print(f"\n❌ Exception: {e}")
        return False


def test_faceted_search_endpoint():
    """Test the /api/search/faceted endpoint."""
    print("\n" + "=" * 60)
    print("Testing /api/search/faceted endpoint")
    print("=" * 60)

    # Test 1: Search with source filter
    print("\n--- Test 1: Filter by source (CNA) ---")
    payload = {
        "query": "台灣",
        "model": "bm25",
        "top_k": 10,
        "filters": {
            "source": ["中央社"]
        }
    }

    try:
        response = requests.post(f"{BASE_URL}/api/search/faceted", json=payload, timeout=30)

        if response.status_code == 200:
            data = response.json()

            if data.get('success'):
                print(f"✅ Filter test passed!")
                print(f"Total results before filter: {data.get('total_results')}")
                print(f"Filtered results: {data.get('filtered_results')}")
                print(f"Active filters: {data.get('active_filters', {}).get('count')}")

                # Show first few results
                results = data.get('results', [])
                if results:
                    print(f"\nFirst 3 filtered results:")
                    for i, r in enumerate(results[:3], 1):
                        print(f"  {i}. {r.get('title', 'N/A')[:50]}...")
                        print(f"     Source: {r.get('source', 'N/A')}, Score: {r.get('score', 0):.3f}")
            else:
                print(f"❌ API returned error: {data.get('error')}")
                return False
        else:
            print(f"❌ HTTP error {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

    # Test 2: Search with multiple filters
    print("\n--- Test 2: Filter by category ---")
    payload = {
        "query": "經濟",
        "model": "bm25",
        "top_k": 10,
        "filters": {
            "category": "afe"  # Finance category
        }
    }

    try:
        response = requests.post(f"{BASE_URL}/api/search/faceted", json=payload, timeout=30)

        if response.status_code == 200:
            data = response.json()

            if data.get('success'):
                print(f"✅ Category filter test passed!")
                print(f"Filtered results: {data.get('filtered_results')}")

                results = data.get('results', [])
                if results:
                    print(f"\nSample result:")
                    r = results[0]
                    print(f"  Title: {r.get('title', 'N/A')[:60]}...")
                    print(f"  Category: {r.get('category', 'N/A')}")
                return True
            else:
                print(f"❌ API returned error: {data.get('error')}")
                return False
        else:
            print(f"❌ HTTP error {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Faceted Search API Test Suite")
    print("=" * 60)
    print(f"Testing against: {BASE_URL}")
    print("Make sure Flask app is running: python app_simple.py")
    print("=" * 60)

    # Test endpoints
    facets_ok = test_facets_endpoint()
    faceted_search_ok = test_faceted_search_endpoint()

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"/api/facets: {'✅ PASS' if facets_ok else '❌ FAIL'}")
    print(f"/api/search/faceted: {'✅ PASS' if faceted_search_ok else '❌ FAIL'}")

    if facets_ok and faceted_search_ok:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the output above.")
        return 1


if __name__ == "__main__":
    exit(main())
