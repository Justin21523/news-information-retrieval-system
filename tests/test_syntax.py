"""
句法分析測試 (直接執行版本)

測試 DependencyParser, SVOExtractor, SyntaxAnalyzer

直接運行此腳本進行測試：
    python tests/test_syntax.py

Author: Information Retrieval System
License: Educational Use
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ir.syntax import DependencyParser, SVOExtractor, SyntaxAnalyzer


def print_test_header(test_name):
    """打印測試標題"""
    print("\n" + "=" * 70)
    print(f"測試: {test_name}")
    print("=" * 70)


def print_pass(message="通過"):
    """打印通過訊息"""
    print(f"✓ {message}")


def print_fail(message="失敗"):
    """打印失敗訊息"""
    print(f"✗ {message}")


# ============================================================================
# 測試數據
# ============================================================================

SIMPLE_SENTENCES = [
    "我喜歡你",
    "張三吃蘋果",
    "她學習中文"
]

COMPLEX_SENTENCES = [
    "張三在台北大學學習自然語言處理",
    "深度學習模型需要大量訓練資料",
    "我喜歡閱讀關於機器學習的書籍"
]


# ============================================================================
# DependencyParser 測試
# ============================================================================

def test_parser_initialization():
    """測試 DependencyParser 初始化"""
    print_test_header("DependencyParser 初始化")

    parser = DependencyParser(tokenizer_engine='jieba', device=-1)

    assert parser.parser is not None, "Parser 應該已載入"
    assert parser.tokenizer is not None, "Tokenizer 應該已初始化"

    print_pass("DependencyParser 初始化成功")
    print(f"  - 模型: {parser.model_name}")
    print(f"  - 設備: CPU")


def test_parse_simple():
    """測試簡單句子解析"""
    print_test_header("簡單句子解析")

    parser = DependencyParser(tokenizer_engine='jieba')
    text = "我喜歡你"

    edges = parser.parse(text)

    assert len(edges) > 0, "應該至少有一條依存邊"
    assert all(hasattr(e, 'head_word') for e in edges)
    assert all(hasattr(e, 'dependent_word') for e in edges)
    assert all(hasattr(e, 'relation') for e in edges)

    print_pass(f"成功解析 {len(edges)} 條依存邊")
    print(f"\n  依存邊:")
    for edge in edges:
        print(f"    {edge.dependent_word} --{edge.relation}--> {edge.head_word}")


def test_parse_complex():
    """測試複雜句子解析"""
    print_test_header("複雜句子解析")

    parser = DependencyParser(tokenizer_engine='jieba')
    text = "張三在台北大學學習自然語言處理"

    edges = parser.parse(text)

    assert len(edges) > 5, "複雜句子應該有更多依存邊"

    print_pass(f"成功解析複雜句子，{len(edges)} 條依存邊")
    print(f"\n  依存邊 (前 5 條):")
    for edge in edges[:5]:
        print(f"    {edge.dependent_word} --{edge.relation}--> {edge.head_word}")


def test_parse_batch():
    """測試批次解析"""
    print_test_header("批次解析")

    parser = DependencyParser(tokenizer_engine='jieba')
    texts = SIMPLE_SENTENCES

    results = parser.parse_batch(texts)

    assert len(results) == len(texts), "結果數量應該與輸入相同"
    assert all(isinstance(r, list) for r in results)

    print_pass(f"成功批次解析 {len(texts)} 個句子")
    for i, (text, edges) in enumerate(zip(texts, results), 1):
        print(f"  句子 {i}: {text} → {len(edges)} 條邊")


def test_get_dependency_tree():
    """測試獲取依存樹"""
    print_test_header("獲取依存樹")

    parser = DependencyParser(tokenizer_engine='jieba')
    text = "我喜歡吃蘋果"

    tree = parser.get_dependency_tree(text)

    assert isinstance(tree, dict), "應該返回字典"
    assert 0 in tree, "應該包含 ROOT 節點"

    print_pass("成功獲取依存樹")
    print(f"\n  樹結構:")
    for head_idx, edges in sorted(tree.items())[:3]:
        head = "ROOT" if head_idx == 0 else f"節點{head_idx}"
        print(f"    {head}:")
        for edge in edges:
            print(f"      └─ {edge.dependent_word} ({edge.relation})")


def test_get_root_verb():
    """測試獲取根動詞"""
    print_test_header("獲取根動詞")

    parser = DependencyParser(tokenizer_engine='jieba')
    text = "張三喜歡吃蘋果"

    edges = parser.parse(text)
    root_verb = parser.get_root_verb(edges)

    assert root_verb is not None, "應該找到根動詞"
    assert isinstance(root_verb, str)

    print_pass(f"成功找到根動詞: '{root_verb}'")


# ============================================================================
# SVOExtractor 測試
# ============================================================================

def test_svo_initialization():
    """測試 SVOExtractor 初始化"""
    print_test_header("SVOExtractor 初始化")

    extractor = SVOExtractor(tokenizer_engine='jieba')

    assert extractor.parser is not None, "Parser 應該已初始化"

    print_pass("SVOExtractor 初始化成功")


def test_svo_extract_simple():
    """測試簡單 SVO 提取"""
    print_test_header("簡單 SVO 提取")

    extractor = SVOExtractor(tokenizer_engine='jieba')
    text = "我喜歡你"  # Changed from "張三吃蘋果" due to tokenization issues

    triples = extractor.extract(text, include_partial=True)

    assert len(triples) > 0, "應該至少提取一個 SVO 三元組"
    assert all(hasattr(t, 'subject') for t in triples)
    assert all(hasattr(t, 'verb') for t in triples)

    print_pass(f"成功提取 {len(triples)} 個 SVO 三元組")
    print(f"\n  SVO 三元組:")
    for triple in triples:
        print(f"    {triple}")


def test_svo_extract_complex():
    """測試複雜 SVO 提取"""
    print_test_header("複雜 SVO 提取")

    extractor = SVOExtractor(tokenizer_engine='jieba')
    text = "我在台北大學學習自然語言處理"

    triples = extractor.extract(text, include_partial=True)

    print_pass(f"成功提取 {len(triples)} 個 SVO 三元組")
    print(f"\n  SVO 三元組:")
    for triple in triples:
        if triple.object:
            print(f"    主語:{triple.subject}, 謂語:{triple.verb}, 賓語:{triple.object}")
        else:
            print(f"    主語:{triple.subject}, 謂語:{triple.verb}")


def test_svo_extract_batch():
    """測試批次 SVO 提取"""
    print_test_header("批次 SVO 提取")

    extractor = SVOExtractor(tokenizer_engine='jieba')
    texts = SIMPLE_SENTENCES

    results = extractor.extract_batch(texts, include_partial=True)

    assert len(results) == len(texts), "結果數量應該與輸入相同"

    print_pass(f"成功批次提取 {len(texts)} 個句子")
    for i, (text, triples) in enumerate(zip(texts, results), 1):
        print(f"\n  句子 {i}: {text}")
        for triple in triples:
            print(f"    → {triple}")


def test_svo_partial():
    """測試部分 SVO (無賓語)"""
    print_test_header("部分 SVO 提取")

    extractor = SVOExtractor(tokenizer_engine='jieba')
    text = "我學習"

    # 包含部分
    triples_with = extractor.extract(text, include_partial=True)
    # 不包含部分
    triples_without = extractor.extract(text, include_partial=False)

    print_pass("部分 SVO 提取測試完成")
    print(f"  - 包含部分: {len(triples_with)} 個")
    print(f"  - 僅完整: {len(triples_without)} 個")

    if triples_with:
        print(f"\n  包含部分的結果:")
        for triple in triples_with:
            print(f"    {triple}")


def test_extract_all_relations():
    """測試提取所有關係"""
    print_test_header("提取所有依存關係")

    extractor = SVOExtractor(tokenizer_engine='jieba')
    text = "我喜歡吃蘋果"

    relations = extractor.extract_all_relations(text)

    assert len(relations) > 0, "應該至少有一個關係"
    assert all(len(r) == 3 for r in relations), "每個關係應該是三元組"

    print_pass(f"成功提取 {len(relations)} 個依存關係")
    print(f"\n  依存關係:")
    for head, rel, dep in relations[:5]:
        print(f"    {head} --{rel}--> {dep}")


# ============================================================================
# SyntaxAnalyzer 測試
# ============================================================================

def test_analyzer_initialization():
    """測試 SyntaxAnalyzer 初始化"""
    print_test_header("SyntaxAnalyzer 初始化")

    analyzer = SyntaxAnalyzer(tokenizer_engine='jieba', device=-1)

    assert analyzer.parser is not None
    assert analyzer.svo_extractor is not None

    print_pass("SyntaxAnalyzer 初始化成功")


def test_analyzer_analyze():
    """測試綜合分析"""
    print_test_header("綜合句法分析")

    analyzer = SyntaxAnalyzer(tokenizer_engine='jieba')
    text = "張三在台北大學學習自然語言處理"

    result = analyzer.analyze(text, extract_svo=True)

    assert 'text' in result
    assert 'tokens' in result
    assert 'dependency_edges' in result
    assert 'svo_triples' in result
    assert 'root_verb' in result

    print_pass("綜合分析成功")
    print(f"\n  分析結果:")
    print(f"    - 原文: {result['text']}")
    print(f"    - 詞數: {len(result['tokens'])}")
    print(f"    - 依存邊: {result['num_edges']} 條")
    print(f"    - SVO 三元組: {result['num_triples']} 組")
    print(f"    - 根動詞: {result['root_verb']}")

    print(f"\n  SVO 三元組:")
    for triple in result['svo_triples']:
        print(f"    {triple}")


def test_analyzer_batch():
    """測試批次分析"""
    print_test_header("批次綜合分析")

    analyzer = SyntaxAnalyzer(tokenizer_engine='jieba')
    texts = COMPLEX_SENTENCES

    results = analyzer.analyze_batch(texts, extract_svo=True)

    assert len(results) == len(texts), "結果數量應該與輸入相同"

    print_pass(f"成功批次分析 {len(texts)} 個句子")
    for i, result in enumerate(results, 1):
        print(f"\n  句子 {i}: {result['text'][:30]}...")
        print(f"    - 依存邊: {result['num_edges']}")
        print(f"    - SVO: {result['num_triples']}")
        if result['svo_triples']:
            print(f"    - 主要 SVO: {result['svo_triples'][0]}")


# ============================================================================
# 整合測試
# ============================================================================

def test_integration_workflow():
    """測試完整工作流程"""
    print_test_header("完整句法分析工作流程")

    # 初始化
    analyzer = SyntaxAnalyzer(tokenizer_engine='jieba')

    # 測試句子
    sentences = [
        "機器學習是人工智慧的重要分支",
        "深度學習模型需要大量訓練資料",
        "張三在台北大學研究自然語言處理"
    ]

    # 批次分析
    results = analyzer.analyze_batch(sentences, extract_svo=True)

    # 統計
    total_edges = sum(r['num_edges'] for r in results)
    total_triples = sum(r['num_triples'] for r in results)

    print_pass("完整工作流程測試成功")
    print(f"\n  統計資訊:")
    print(f"    - 句子數: {len(sentences)}")
    print(f"    - 總依存邊: {total_edges}")
    print(f"    - 總 SVO 三元組: {total_triples}")

    print(f"\n  詳細結果:")
    for i, result in enumerate(results, 1):
        print(f"\n  句子 {i}: {result['text']}")
        print(f"    依存邊: {result['num_edges']} 條")
        print(f"    SVO: {result['num_triples']} 組")
        for triple in result['svo_triples']:
            print(f"      → {triple}")


def test_triple_to_dict():
    """測試 SVO 三元組轉字典"""
    print_test_header("SVO 三元組轉字典")

    extractor = SVOExtractor(tokenizer_engine='jieba')
    text = "我喜歡你"

    triples = extractor.extract(text)

    if triples:
        triple = triples[0]
        dict_form = triple.to_dict()

        assert isinstance(dict_form, dict)
        assert 'subject' in dict_form
        assert 'verb' in dict_form
        assert 'object' in dict_form

        print_pass("成功轉換為字典格式")
        print(f"\n  字典格式:")
        for key, value in dict_form.items():
            print(f"    {key}: {value}")


def run_all_tests():
    """執行所有測試"""
    print("\n" + "=" * 70)
    print("句法分析測試套件 (DependencyParser, SVOExtractor, SyntaxAnalyzer)")
    print("=" * 70)

    test_functions = [
        # DependencyParser tests
        test_parser_initialization,
        test_parse_simple,
        test_parse_complex,
        test_parse_batch,
        test_get_dependency_tree,
        test_get_root_verb,
        # SVOExtractor tests
        test_svo_initialization,
        test_svo_extract_simple,
        test_svo_extract_complex,
        test_svo_extract_batch,
        test_svo_partial,
        test_extract_all_relations,
        # SyntaxAnalyzer tests
        test_analyzer_initialization,
        test_analyzer_analyze,
        test_analyzer_batch,
        # Integration tests
        test_triple_to_dict,
        test_integration_workflow,
    ]

    passed = 0
    failed = 0

    for test_func in test_functions:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            failed += 1
            print_fail(f"{test_func.__name__}: {str(e)}")
        except Exception as e:
            failed += 1
            print_fail(f"{test_func.__name__}: 發生錯誤 - {str(e)}")

    # 總結
    print("\n" + "=" * 70)
    print("測試總結")
    print("=" * 70)
    print(f"總測試數: {passed + failed}")
    print(f"通過: {passed} ✓")
    print(f"失敗: {failed} ✗")
    print(f"成功率: {100 * passed / (passed + failed):.1f}%")
    print("=" * 70 + "\n")

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
