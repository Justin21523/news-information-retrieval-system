"""
PAT-tree Pattern Mining Tests (直接執行版本)

直接運行此腳本進行測試：
    python tests/test_pat_tree_simple.py

Author: Information Retrieval System
License: Educational Use
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ir.patterns import PATTree, Pattern, PATNode
from src.ir.text.chinese_tokenizer import ChineseTokenizer


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


def test_node_initialization():
    """測試 PATNode 初始化"""
    print_test_header("PATNode 初始化")

    node = PATNode()

    assert len(node.children) == 0, "children 應該為空"
    assert node.frequency == 0, "frequency 應該為 0"
    assert node.is_end == False, "is_end 應該為 False"
    assert len(node.positions) == 0, "positions 應該為空"

    print_pass("PATNode 初始化成功")
    print(f"  - children: {len(node.children)}")
    print(f"  - frequency: {node.frequency}")
    print(f"  - is_end: {node.is_end}")


def test_tree_initialization():
    """測試 PAT-tree 初始化"""
    print_test_header("PAT-tree 初始化")

    tree = PATTree(
        min_pattern_length=2,
        max_pattern_length=5,
        min_frequency=2
    )

    assert tree.min_pattern_length == 2
    assert tree.max_pattern_length == 5
    assert tree.min_frequency == 2
    assert tree.total_tokens == 0

    print_pass("PAT-tree 初始化成功")
    print(f"  - 最小樣式長度: {tree.min_pattern_length}")
    print(f"  - 最大樣式長度: {tree.max_pattern_length}")
    print(f"  - 最小頻率: {tree.min_frequency}")
    print(f"  - 總詞數: {tree.total_tokens}")


def test_insert_sequences():
    """測試插入詞序列"""
    print_test_header("插入詞序列")

    tree = PATTree(min_pattern_length=2)

    sequences = [
        ['機器', '學習', '是', '重要', '技術'],
        ['機器', '學習', '和', '深度', '學習'],
        ['深度', '學習', '是', '機器', '學習']
    ]

    for seq in sequences:
        tree.insert_sequence(seq)

    total_tokens = sum(len(s) for s in sequences)
    assert tree.total_tokens == total_tokens

    print_pass("成功插入多個序列")
    print(f"  - 插入序列數: {len(sequences)}")
    print(f"  - 總詞數: {tree.total_tokens}")
    print(f"  - '機器' 出現次數: {tree.token_freq['機器']}")
    print(f"  - '學習' 出現次數: {tree.token_freq['學習']}")


def test_pattern_extraction():
    """測試樣式提取"""
    print_test_header("樣式提取")

    tree = PATTree(min_pattern_length=2, min_frequency=2)

    sequences = [
        ['機器', '學習', '是', '重要', '技術'],
        ['機器', '學習', '和', '深度', '學習'],
        ['深度', '學習', '是', '機器', '學習']
    ]

    for seq in sequences:
        tree.insert_sequence(seq)

    patterns = tree.extract_patterns(use_mi_score=False)

    assert len(patterns) > 0, "應該找到至少一個樣式"
    assert all(p.frequency >= 2 for p in patterns), "所有樣式頻率應該 >= 2"

    print_pass(f"成功提取 {len(patterns)} 個樣式")
    print(f"\n  前 5 個樣式 (按頻率):")
    for i, p in enumerate(patterns[:5], 1):
        print(f"    {i}. '{p.text}' - 頻率: {p.frequency}")


def test_mi_calculation():
    """測試 Mutual Information 計算"""
    print_test_header("Mutual Information 計算")

    tree = PATTree(min_pattern_length=2, min_frequency=2)

    sequences = [
        ['機器', '學習', '是', '重要', '技術'],
        ['機器', '學習', '和', '深度', '學習'],
        ['深度', '學習', '是', '機器', '學習']
    ]

    for seq in sequences:
        tree.insert_sequence(seq)

    patterns = tree.extract_patterns(use_mi_score=True)

    assert len(patterns) > 0
    assert all(p.mi_score > 0 for p in patterns), "所有 MI 分數應該 > 0"

    # 檢查是否按 MI 分數排序
    for i in range(len(patterns) - 1):
        assert patterns[i].mi_score >= patterns[i + 1].mi_score, "應該按 MI 分數降序排列"

    print_pass(f"成功計算 {len(patterns)} 個樣式的 MI 分數")
    print(f"\n  前 5 個樣式 (按 MI 分數):")
    for i, p in enumerate(patterns[:5], 1):
        print(f"    {i}. '{p.text}' - 頻率: {p.frequency}, MI: {p.mi_score:.3f}")


def test_search_query():
    """測試搜尋功能"""
    print_test_header("搜尋功能")

    tree = PATTree()
    tree.insert_sequence(['A', 'B', 'C', 'D'])
    tree.insert_sequence(['A', 'B', 'E'])
    tree.insert_sequence(['A', 'B', 'C', 'F'])

    # 測試存在的樣式
    node = tree.search(['A', 'B'])
    assert node is not None, "應該找到 'AB'"
    assert node.frequency == 3, "AB 應該出現 3 次"

    # 測試不存在的樣式
    node = tree.search(['X', 'Y'])
    assert node is None, "不應該找到 'XY'"

    # 測試 get_frequency
    freq = tree.get_frequency(['A', 'B', 'C'])
    assert freq == 2, "ABC 應該出現 2 次"

    print_pass("搜尋功能正常")
    print(f"  - 'AB' 頻率: {tree.get_frequency(['A', 'B'])}")
    print(f"  - 'ABC' 頻率: {tree.get_frequency(['A', 'B', 'C'])}")
    print(f"  - 'XY' 頻率: {tree.get_frequency(['X', 'Y'])}")


def test_chinese_integration():
    """測試中文整合"""
    print_test_header("中文文本整合測試")

    tokenizer = ChineseTokenizer(engine='jieba')
    tree = PATTree(min_pattern_length=2, min_frequency=2)

    texts = [
        "機器學習和深度學習都很重要",
        "深度學習是機器學習的子領域",
        "機器學習技術發展迅速"
    ]

    print(f"處理 {len(texts)} 篇文本...")
    for text in texts:
        tree.insert_text(text, tokenizer)

    patterns = tree.extract_patterns(top_k=10, use_mi_score=True)

    assert len(patterns) > 0, "應該找到至少一個樣式"

    print_pass(f"成功提取 {len(patterns)} 個中文樣式")
    print(f"\n  統計資訊:")
    stats = tree.get_statistics()
    print(f"    - 總詞數: {stats['total_tokens']}")
    print(f"    - 唯一詞: {stats['unique_tokens']}")
    print(f"    - 樹節點: {stats['total_nodes']}")

    print(f"\n  前 10 個樣式:")
    for i, p in enumerate(patterns, 1):
        print(f"    {i}. '{p.text}' - 頻率: {p.frequency}, MI: {p.mi_score:.3f}")


def test_pattern_class():
    """測試 Pattern 類別"""
    print_test_header("Pattern 類別")

    pattern = Pattern(
        tokens=('機器', '學習'),
        frequency=5,
        mi_score=2.5,
        positions=[0, 10, 20]
    )

    assert pattern.text == '機器學習', "text 屬性應該正確連接"
    assert pattern.frequency == 5
    assert pattern.mi_score == 2.5
    assert len(pattern.positions) == 3

    print_pass("Pattern 類別運作正常")
    print(f"  - 樣式文本: {pattern.text}")
    print(f"  - 頻率: {pattern.frequency}")
    print(f"  - MI 分數: {pattern.mi_score}")
    print(f"  - 位置數: {len(pattern.positions)}")
    print(f"  - 表示法: {repr(pattern)}")


def test_statistics():
    """測試統計功能"""
    print_test_header("統計功能")

    tree = PATTree(min_pattern_length=2, max_pattern_length=4)

    sequences = [
        ['A', 'B', 'C'],
        ['D', 'E', 'F'],
        ['A', 'B', 'D']
    ]

    for seq in sequences:
        tree.insert_sequence(seq)

    stats = tree.get_statistics()

    assert 'total_tokens' in stats
    assert 'unique_tokens' in stats
    assert 'total_nodes' in stats
    assert stats['total_tokens'] == 9

    print_pass("統計功能正常")
    print(f"  統計資訊:")
    for key, value in stats.items():
        print(f"    - {key}: {value}")
    print(f"\n  樹表示法: {repr(tree)}")


def run_all_tests():
    """執行所有測試"""
    print("\n" + "=" * 70)
    print("PAT-tree 測試套件 (直接執行版本)")
    print("=" * 70)

    test_functions = [
        test_node_initialization,
        test_tree_initialization,
        test_insert_sequences,
        test_pattern_extraction,
        test_mi_calculation,
        test_search_query,
        test_pattern_class,
        test_statistics,
        test_chinese_integration,
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
