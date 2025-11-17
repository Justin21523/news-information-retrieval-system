"""
關鍵詞提取測試 (直接執行版本)

測試 TextRank, YAKE, RAKE 關鍵詞提取算法

直接運行此腳本進行測試：
    python tests/test_keyextract.py

Author: Information Retrieval System
License: Educational Use
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ir.keyextract import TextRankExtractor, YAKEExtractor, RAKEExtractor


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

SAMPLE_TEXT_ZH = """
機器學習是人工智慧的重要分支，它讓電腦能夠從資料中學習模式。
深度學習是機器學習的子領域，使用神經網路來建立複雜的模型。
自然語言處理是人工智慧的另一個重要應用，涉及文字分析和理解。
"""

SAMPLE_TEXT_EN = """
Machine learning is a branch of artificial intelligence that enables
computers to learn from data. Deep learning uses neural networks to
build complex models. Natural language processing analyzes text.
"""


# ============================================================================
# TextRank 測試
# ============================================================================

def test_textrank_initialization():
    """測試 TextRank 初始化"""
    print_test_header("TextRank 初始化")

    extractor = TextRankExtractor(
        window_size=5,
        use_position_weight=True,
        tokenizer_engine='jieba'
    )

    assert extractor.window_size == 5
    assert extractor.use_position_weight == True
    assert extractor.damping_factor == 0.85

    print_pass("TextRank 初始化成功")
    print(f"  - 窗口大小: {extractor.window_size}")
    print(f"  - 位置權重: {extractor.use_position_weight}")
    print(f"  - 阻尼因子: {extractor.damping_factor}")


def test_textrank_extract_basic():
    """測試基本關鍵詞提取"""
    print_test_header("TextRank 基本提取")

    extractor = TextRankExtractor(tokenizer_engine='jieba')
    keywords = extractor.extract(SAMPLE_TEXT_ZH, top_k=5)

    assert len(keywords) <= 5
    assert all(hasattr(kw, 'word') for kw in keywords)
    assert all(hasattr(kw, 'score') for kw in keywords)
    assert all(0.0 <= kw.score <= 1.0 for kw in keywords)

    print_pass(f"成功提取 {len(keywords)} 個關鍵詞")
    print(f"\n  前 5 個關鍵詞:")
    for i, kw in enumerate(keywords, 1):
        print(f"    {i}. {kw.word:15s}  score={kw.score:.4f}  freq={kw.frequency}")


def test_textrank_with_pos_filter():
    """測試 POS 過濾"""
    print_test_header("TextRank POS 過濾")

    extractor = TextRankExtractor(
        pos_filter=['N', 'V'],  # 只要名詞和動詞
        tokenizer_engine='jieba'
    )

    keywords = extractor.extract(SAMPLE_TEXT_ZH, top_k=5)

    assert len(keywords) > 0
    assert len(keywords) <= 5

    print_pass(f"POS 過濾後提取 {len(keywords)} 個關鍵詞")
    for i, kw in enumerate(keywords, 1):
        print(f"  {i}. {kw.word}")


def test_textrank_keyphrases():
    """測試多詞關鍵短語提取"""
    print_test_header("TextRank 關鍵短語提取")

    extractor = TextRankExtractor(tokenizer_engine='jieba')
    keyphrases = extractor.extract_keyphrases(
        SAMPLE_TEXT_ZH,
        top_k=3,
        max_phrase_length=3
    )

    assert len(keyphrases) <= 3

    print_pass(f"成功提取 {len(keyphrases)} 個關鍵短語")
    for i, kp in enumerate(keyphrases, 1):
        print(f"  {i}. {kp.word:20s}  score={kp.score:.4f}")


def test_textrank_position_weighting():
    """測試位置權重效果"""
    print_test_header("TextRank 位置權重")

    extractor_with = TextRankExtractor(
        use_position_weight=True,
        tokenizer_engine='jieba'
    )
    extractor_without = TextRankExtractor(
        use_position_weight=False,
        tokenizer_engine='jieba'
    )

    kw_with = extractor_with.extract(SAMPLE_TEXT_ZH, top_k=5)
    kw_without = extractor_without.extract(SAMPLE_TEXT_ZH, top_k=5)

    assert len(kw_with) > 0
    assert len(kw_without) > 0

    print_pass("位置權重測試完成")
    print(f"\n  有位置權重 (前3個):")
    for kw in kw_with[:3]:
        print(f"    {kw.word:12s}  {kw.score:.4f}")
    print(f"\n  無位置權重 (前3個):")
    for kw in kw_without[:3]:
        print(f"    {kw.word:12s}  {kw.score:.4f}")


def test_textrank_ckip():
    """測試 CKIP 分詞器"""
    print_test_header("TextRank 使用 CKIP 分詞")

    extractor = TextRankExtractor(tokenizer_engine='ckip')
    keywords = extractor.extract(SAMPLE_TEXT_ZH, top_k=5)

    assert len(keywords) > 0
    assert len(keywords) <= 5

    print_pass(f"CKIP 分詞成功提取 {len(keywords)} 個關鍵詞")
    for i, kw in enumerate(keywords, 1):
        print(f"  {i}. {kw.word}")


def test_textrank_ckip_pos():
    """測試 CKIP POS 過濾"""
    print_test_header("TextRank CKIP POS 過濾")

    extractor = TextRankExtractor(
        pos_filter=['N', 'V'],  # CKIP 大寫標籤
        tokenizer_engine='ckip'
    )

    keywords = extractor.extract(SAMPLE_TEXT_ZH, top_k=5)

    assert len(keywords) > 0

    print_pass(f"CKIP POS 過濾成功")
    for i, kw in enumerate(keywords[:3], 1):
        print(f"  {i}. {kw.word}")


def test_textrank_ner_boost():
    """測試 NER 實體權重提升"""
    print_test_header("TextRank NER 實體權重提升")

    text = """
    台灣大學位於台北市。張三教授在這裡教授機器學習。
    Google和微軟都是重要的科技公司。
    """

    extractor_with_ner = TextRankExtractor(
        tokenizer_engine='ckip',
        use_ner_boost=True,
        ner_boost_weight=0.5
    )

    extractor_no_ner = TextRankExtractor(
        tokenizer_engine='ckip',
        use_ner_boost=False
    )

    kw_with_ner = extractor_with_ner.extract(text, top_k=5)
    kw_no_ner = extractor_no_ner.extract(text, top_k=5)

    assert len(kw_with_ner) > 0
    assert len(kw_no_ner) > 0

    print_pass("NER 權重提升測試完成")
    print(f"\n  有 NER 權重 (前5個):")
    for kw in kw_with_ner:
        print(f"    {kw.word:12s}  score={kw.score:.4f}")
    print(f"\n  無 NER 權重 (前5個):")
    for kw in kw_no_ner:
        print(f"    {kw.word:12s}  score={kw.score:.4f}")


# ============================================================================
# YAKE 測試
# ============================================================================

def test_yake_initialization():
    """測試 YAKE 初始化"""
    print_test_header("YAKE 初始化")

    extractor = YAKEExtractor(
        language='zh',
        max_ngram_size=3,
        tokenizer_engine='jieba'
    )

    assert extractor.language == 'zh'
    assert extractor.max_ngram_size == 3

    print_pass("YAKE 初始化成功")
    print(f"  - 語言: {extractor.language}")
    print(f"  - 最大 n-gram: {extractor.max_ngram_size}")


def test_yake_extract():
    """測試 YAKE 提取"""
    print_test_header("YAKE 關鍵詞提取")

    extractor = YAKEExtractor(language='zh', tokenizer_engine='jieba')
    keywords = extractor.extract(SAMPLE_TEXT_ZH, top_k=5)

    assert len(keywords) <= 5

    print_pass(f"YAKE 成功提取 {len(keywords)} 個關鍵詞")
    for i, kw in enumerate(keywords, 1):
        print(f"  {i}. {kw.word:15s}  score={kw.score:.4f}")


def test_yake_english():
    """測試 YAKE 英文"""
    print_test_header("YAKE 英文文本")

    extractor = YAKEExtractor(language='en', max_ngram_size=2)
    keywords = extractor.extract(SAMPLE_TEXT_EN, top_k=5, preprocess=False)

    assert len(keywords) <= 5

    print_pass(f"YAKE 英文提取成功")
    for i, kw in enumerate(keywords, 1):
        print(f"  {i}. {kw.word}")


# ============================================================================
# RAKE 測試
# ============================================================================

def test_rake_initialization():
    """測試 RAKE 初始化"""
    print_test_header("RAKE 初始化")

    extractor = RAKEExtractor(
        max_length=4,
        ranking_metric='degree_to_frequency',
        tokenizer_engine='jieba'
    )

    assert extractor.max_length == 4
    assert extractor.ranking_metric == 'degree_to_frequency'

    print_pass("RAKE 初始化成功")
    print(f"  - 最大長度: {extractor.max_length}")
    print(f"  - 排序指標: {extractor.ranking_metric}")


def test_rake_extract():
    """測試 RAKE 提取"""
    print_test_header("RAKE 關鍵詞提取")

    extractor = RAKEExtractor(tokenizer_engine='jieba')
    keywords = extractor.extract(SAMPLE_TEXT_ZH, top_k=5)

    assert len(keywords) <= 5

    print_pass(f"RAKE 成功提取 {len(keywords)} 個關鍵詞")
    for i, kw in enumerate(keywords, 1):
        print(f"  {i}. {kw.word:15s}  score={kw.score:.4f}")


# ============================================================================
# 整合測試
# ============================================================================

def test_multiple_extractors():
    """測試多種提取器比較"""
    print_test_header("多種提取器比較")

    extractors = {
        'TextRank': TextRankExtractor(tokenizer_engine='jieba'),
        'YAKE': YAKEExtractor(language='zh', tokenizer_engine='jieba'),
        'RAKE': RAKEExtractor(tokenizer_engine='jieba')
    }

    results = {}
    for name, extractor in extractors.items():
        keywords = extractor.extract(SAMPLE_TEXT_ZH, top_k=5)
        results[name] = [kw.word for kw in keywords]

    assert len(results) == 3

    print_pass("成功比較3種提取器")
    for name, keywords in results.items():
        print(f"\n  {name}:")
        for kw in keywords[:3]:
            print(f"    - {kw}")


def test_extraction_workflow():
    """測試完整提取工作流程"""
    print_test_header("完整關鍵詞提取工作流程")

    # 初始化
    extractor = TextRankExtractor(
        window_size=5,
        use_position_weight=True,
        pos_filter=['N', 'V'],
        tokenizer_engine='jieba'
    )

    # 提取關鍵詞
    keywords = extractor.extract(SAMPLE_TEXT_ZH, top_k=10)

    # 提取關鍵短語
    keyphrases = extractor.extract_keyphrases(SAMPLE_TEXT_ZH, top_k=5)

    # 獲取配置
    config = extractor.get_config()

    assert len(keywords) > 0
    assert len(keyphrases) >= 0
    assert 'window_size' in config

    print_pass("完整工作流程測試成功")
    print(f"  - 關鍵詞數: {len(keywords)}")
    print(f"  - 關鍵短語數: {len(keyphrases)}")
    print(f"\n  配置:")
    for key, value in config.items():
        print(f"    - {key}: {value}")


def run_all_tests():
    """執行所有測試"""
    print("\n" + "=" * 70)
    print("關鍵詞提取測試套件 (TextRank, YAKE, RAKE)")
    print("=" * 70)

    test_functions = [
        # TextRank tests
        test_textrank_initialization,
        test_textrank_extract_basic,
        test_textrank_with_pos_filter,
        test_textrank_keyphrases,
        test_textrank_position_weighting,
        test_textrank_ckip,
        test_textrank_ckip_pos,
        test_textrank_ner_boost,
        # YAKE tests
        test_yake_initialization,
        test_yake_extract,
        test_yake_english,
        # RAKE tests
        test_rake_initialization,
        test_rake_extract,
        # Integration tests
        test_multiple_extractors,
        test_extraction_workflow,
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
