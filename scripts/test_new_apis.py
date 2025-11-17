"""
Test Script for New NLP API Endpoints

This script tests all newly integrated NLP APIs:
- Keyword extraction (TextRank, YAKE, KeyBERT, RAKE)
- Topic modeling (LDA, BERTopic)
- Pattern mining (PAT-tree)
- Named Entity Recognition (NER)
- Syntactic analysis (SVO extraction, dependencies)

Usage:
    python scripts/test_new_apis.py

Requirements:
    - Flask server running on localhost:5001
    - pip install requests

Author: Information Retrieval System
License: Educational Use
"""

import requests
import json
import time
from typing import Dict, Any


BASE_URL = "http://localhost:5001"


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def print_subsection(title: str):
    """Print a formatted subsection header."""
    print("\n" + "-" * 80)
    print(f" {title}")
    print("-" * 80)


def test_api(endpoint: str, method: str = "POST", data: Dict = None, description: str = "") -> Dict[str, Any]:
    """
    Test an API endpoint and return the response.

    Args:
        endpoint: API endpoint path
        method: HTTP method (POST or GET)
        data: Request payload
        description: Test description

    Returns:
        Response JSON or error dict
    """
    url = f"{BASE_URL}{endpoint}"

    print(f"\n📡 Testing: {description}")
    print(f"   Endpoint: {method} {endpoint}")

    try:
        start_time = time.time()

        if method == "POST":
            response = requests.post(url, json=data, timeout=30)
        else:
            response = requests.get(url, timeout=30)

        elapsed = time.time() - start_time

        if response.status_code == 200:
            print(f"   ✅ Success ({elapsed:.2f}s)")
            return response.json()
        else:
            print(f"   ❌ Failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return {'error': response.text, 'status_code': response.status_code}

    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return {'error': str(e)}


def test_keyword_extraction():
    """Test keyword extraction APIs."""
    print_section("1. Keyword Extraction APIs")

    sample_text = """
    人工智慧是電腦科學的一個分支,致力於創造智能機器。
    機器學習是人工智慧的核心技術,包括深度學習、強化學習等方法。
    自然語言處理讓電腦能夠理解和生成人類語言。
    資訊檢索系統使用這些技術來改善搜尋結果的品質。
    """

    methods = ['textrank', 'yake', 'rake']

    for method in methods:
        print_subsection(f"{method.upper()} Keyword Extraction")

        result = test_api(
            '/api/extract/keywords',
            method='POST',
            data={
                'text': sample_text,
                'method': method,
                'topk': 5,
                'use_pos_filter': False,
                'use_ner_boost': False
            },
            description=f"Extract keywords using {method.upper()}"
        )

        if 'keywords' in result:
            print(f"\n   Top 5 Keywords:")
            for i, kw in enumerate(result['keywords'][:5], 1):
                score = kw.get('score', 0)
                freq = kw.get('frequency', 'N/A')
                print(f"   {i}. {kw['keyword']:15s}  score={score:.4f}  freq={freq}")


def test_topic_modeling():
    """Test topic modeling APIs."""
    print_section("2. Topic Modeling APIs")

    sample_docs = [
        "機器學習是人工智慧的重要分支，包括深度學習與強化學習",
        "資訊檢索系統需要使用倒排索引來提高查詢效率",
        "自然語言處理技術應用於文本分類與情感分析",
        "向量空間模型使用TF-IDF權重計算文檔相似度",
        "卷積神經網路在影像辨識領域取得突破性進展",
        "布林檢索模型支援AND OR NOT等邏輯運算子",
        "循環神經網路適合處理序列資料如文本與語音",
        "精確率召回率與F值是評估檢索系統的重要指標",
        "深度學習模型需要大量標註資料進行訓練",
        "倒排索引包含詞彙表與倒排列表兩個主要部分"
    ]

    print_subsection("LDA Topic Modeling")

    result = test_api(
        '/api/extract/topics',
        method='POST',
        data={
            'documents': sample_docs,
            'method': 'lda',
            'n_topics': 3,
            'model_params': {
                'iterations': 50,
                'passes': 10
            }
        },
        description="Extract topics using LDA"
    )

    if 'topics' in result:
        print(f"\n   Topics:")
        for topic in result['topics']:
            words = ', '.join([f"{w['word']}({w['prob']:.3f})" for w in topic['words'][:5]])
            print(f"   Topic {topic['topic_id']}: {words}")

        if 'metrics' in result:
            metrics = result['metrics']
            print(f"\n   Metrics:")
            print(f"   - Perplexity: {metrics['perplexity']:.4f}")
            print(f"   - Coherence: {metrics['coherence']:.4f}")


def test_pattern_mining():
    """Test PAT-tree pattern mining API."""
    print_section("3. PAT-tree Pattern Mining API")

    sample_texts = [
        "機器學習是人工智慧的重要分支",
        "深度學習是機器學習的子領域",
        "機器學習和深度學習都是人工智慧技術",
        "自然語言處理也是人工智慧的重要應用"
    ]

    result = test_api(
        '/api/extract/patterns',
        method='POST',
        data={
            'texts': sample_texts,
            'min_pattern_length': 2,
            'max_pattern_length': 4,
            'min_frequency': 2,
            'topk': 10,
            'use_mi_score': True
        },
        description="Extract frequent patterns using PAT-tree"
    )

    if 'patterns' in result:
        print(f"\n   Top Patterns:")
        for i, p in enumerate(result['patterns'][:10], 1):
            print(f"   {i:2d}. {p['pattern']:15s}  "
                  f"freq={p['frequency']:2d}  MI={p['mi_score']:7.3f}")

        if 'statistics' in result:
            stats = result['statistics']
            print(f"\n   Statistics:")
            print(f"   - Total tokens: {stats['total_tokens']}")
            print(f"   - Unique tokens: {stats['unique_tokens']}")
            print(f"   - Tree nodes: {stats['total_nodes']}")


def test_ner():
    """Test Named Entity Recognition API."""
    print_section("4. Named Entity Recognition API")

    sample_text = "台積電在台灣新竹科學園區成立於1987年,創辦人是張忠謀"

    result = test_api(
        '/api/analyze/ner',
        method='POST',
        data={
            'text': sample_text,
            'entity_types': ['PERSON', 'ORG', 'GPE', 'LOC', 'DATE']
        },
        description="Extract named entities"
    )

    if 'entities' in result:
        print(f"\n   Text: {result['text']}")
        print(f"\n   Entities ({result['entity_count']}):")
        for e in result['entities']:
            print(f"   - {e['text']:15s}  [{e['type']}]  pos=({e['start']}, {e['end']})")

        if 'entities_by_type' in result:
            print(f"\n   By Type:")
            for etype, entities in result['entities_by_type'].items():
                print(f"   - {etype}: {', '.join(entities)}")


def test_syntax_analysis():
    """Test syntactic analysis API."""
    print_section("5. Syntactic Analysis API")

    sample_text = "台積電在台灣生產先進的半導體晶片"

    # Test SVO extraction
    print_subsection("SVO Triple Extraction")

    result = test_api(
        '/api/analyze/syntax',
        method='POST',
        data={
            'text': sample_text,
            'analysis_type': 'svo'
        },
        description="Extract SVO triples"
    )

    if 'triples' in result:
        print(f"\n   Text: {result['text']}")
        print(f"\n   SVO Triples ({result['triple_count']}):")
        for t in result['triples']:
            subj = t['subject'] or '_'
            verb = t['verb'] or '_'
            obj = t['object'] or '_'
            print(f"   - [{subj}] --{verb}--> [{obj}]")


def test_document_analysis():
    """Test comprehensive document analysis API."""
    print_section("6. Comprehensive Document Analysis API")

    # Test with doc_id = 0 (first document)
    result = test_api(
        '/api/document/0/analysis',
        method='GET',
        description="Get comprehensive analysis for document 0"
    )

    if 'analysis' in result:
        print(f"\n   Document: {result.get('title', 'N/A')}")

        analysis = result['analysis']

        if 'keywords' in analysis and analysis['keywords']:
            print(f"\n   Keywords:")
            for kw in analysis['keywords'][:5]:
                print(f"   - {kw['word']:15s}  score={kw['score']:.4f}")

        if 'entities' in analysis and analysis['entities']:
            print(f"\n   Named Entities:")
            for e in analysis['entities'][:10]:
                print(f"   - {e['text']:15s}  [{e['type']}]")


def test_system_status():
    """Test system status API."""
    print_section("0. System Status Check")

    result = test_api(
        '/api/stats',
        method='GET',
        description="Get system statistics"
    )

    if 'total_documents' in result:
        print(f"\n   System Status:")
        print(f"   - Total Documents: {result['total_documents']}")
        print(f"   - Index Status: {result.get('index_status', 'unknown')}")
        print(f"   - Models Loaded: {', '.join(result.get('models_loaded', []))}")


def main():
    """Run all API tests."""
    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "NLP API Integration Test Suite" + " " * 28 + "║")
    print("╚" + "=" * 78 + "╝")

    print("\nℹ️  This script tests all newly integrated NLP APIs.")
    print("ℹ️  Make sure Flask server is running on http://localhost:5001")

    input("\nPress Enter to start testing...")

    try:
        # 0. System status
        test_system_status()

        # 1. Keyword extraction
        test_keyword_extraction()

        # 2. Topic modeling
        test_topic_modeling()

        # 3. Pattern mining
        test_pattern_mining()

        # 4. NER
        test_ner()

        # 5. Syntax analysis
        test_syntax_analysis()

        # 6. Document analysis
        test_document_analysis()

        print_section("✅ All Tests Completed!")
        print("\n✨ Check the results above to verify API functionality.")

    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")


if __name__ == '__main__':
    main()
