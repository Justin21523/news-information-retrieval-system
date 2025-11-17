"""
主題模型測試 (直接執行版本)

測試 BERTopic 和 LDA 主題模型

直接運行此腳本進行測試：
    python tests/test_topic.py

Author: Information Retrieval System
License: Educational Use
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ir.topic import BERTopicModel, LDAModel
import numpy as np


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

SMALL_DOCUMENTS = [
    "機器學習與深度學習",
    "資訊檢索與文本處理",
    "自然語言處理技術"
]

SAMPLE_DOCUMENTS = [
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


# ============================================================================
# LDA 測試
# ============================================================================

def test_lda_initialization():
    """測試 LDA 初始化"""
    print_test_header("LDA 模型初始化")

    lda = LDAModel(n_topics=3, tokenizer_engine='jieba')

    assert lda.n_topics == 3
    assert lda.is_fitted == False
    assert lda.model is None

    print_pass("LDA 初始化成功")
    print(f"  - 主題數: {lda.n_topics}")
    print(f"  - 是否訓練: {lda.is_fitted}")


def test_lda_fit():
    """測試 LDA 訓練"""
    print_test_header("LDA 模型訓練")

    lda = LDAModel(
        n_topics=2,
        iterations=10,
        passes=5,
        min_word_freq=1,
        use_stopwords=False,
        tokenizer_engine='jieba'
    )

    lda.fit(SMALL_DOCUMENTS)

    assert lda.is_fitted == True
    assert lda.model is not None
    assert lda.dictionary is not None
    assert len(lda.dictionary) > 0

    print_pass("LDA 訓練成功")
    print(f"  - 詞彙數: {len(lda.dictionary)}")
    print(f"  - 文檔數: {len(SMALL_DOCUMENTS)}")


def test_lda_get_topics():
    """測試獲取主題"""
    print_test_header("LDA 獲取主題")

    lda = LDAModel(
        n_topics=2,
        min_word_freq=1,
        use_stopwords=False,
        tokenizer_engine='jieba'
    )
    lda.fit(SMALL_DOCUMENTS)

    topics = lda.get_topics()

    assert isinstance(topics, dict)
    assert len(topics) == 2

    print_pass("成功獲取主題")
    print(f"\n  主題分布:")
    for topic_id, words in topics.items():
        print(f"    主題 {topic_id}:")
        for word, prob in words[:3]:  # 只顯示前3個詞
            print(f"      - {word}: {prob:.4f}")


def test_lda_get_topic_words():
    """測試獲取特定主題的詞"""
    print_test_header("LDA 獲取特定主題詞")

    lda = LDAModel(n_topics=2, min_word_freq=1, use_stopwords=False, tokenizer_engine='jieba')
    lda.fit(SMALL_DOCUMENTS)

    words = lda.get_topic_words(topic_id=0, top_n=5)

    assert isinstance(words, list)
    assert len(words) <= 5
    for word, prob in words:
        assert isinstance(word, str)
        assert isinstance(prob, (float, np.floating))
        assert 0 <= prob <= 1

    print_pass(f"成功獲取主題 0 的前 {len(words)} 個詞")
    for word, prob in words:
        print(f"  - {word}: {prob:.4f}")


def test_lda_transform():
    """測試轉換新文檔"""
    print_test_header("LDA 轉換新文檔")

    lda = LDAModel(n_topics=2, min_word_freq=1, use_stopwords=False, tokenizer_engine='jieba')
    lda.fit(SMALL_DOCUMENTS)

    new_docs = ["機器學習技術", "文本檢索系統"]
    topic_dists = lda.transform(new_docs)

    assert len(topic_dists) == len(new_docs)

    print_pass(f"成功轉換 {len(new_docs)} 篇文檔")
    for i, dist in enumerate(topic_dists, 1):
        print(f"\n  文檔 {i} 的主題分布:")
        for topic_id, prob in dist:
            print(f"    主題 {topic_id}: {prob:.4f}")


def test_lda_perplexity():
    """測試困惑度計算"""
    print_test_header("LDA 困惑度計算")

    lda = LDAModel(n_topics=2, min_word_freq=1, use_stopwords=False, tokenizer_engine='jieba')
    lda.fit(SMALL_DOCUMENTS)

    perplexity = lda.calculate_perplexity()

    assert isinstance(perplexity, float)
    assert perplexity > 0

    print_pass("困惑度計算成功")
    print(f"  - Perplexity: {perplexity:.4f}")


def test_lda_coherence():
    """測試一致性計算"""
    print_test_header("LDA 一致性計算")

    lda = LDAModel(n_topics=2, min_word_freq=1, use_stopwords=False, tokenizer_engine='jieba')
    lda.fit(SMALL_DOCUMENTS)

    coherence = lda.calculate_coherence('c_v')

    assert isinstance(coherence, float)
    assert -1 <= coherence <= 1

    print_pass("一致性計算成功")
    print(f"  - Coherence (c_v): {coherence:.4f}")


# ============================================================================
# BERTopic 測試
# ============================================================================

def test_bertopic_initialization():
    """測試 BERTopic 初始化"""
    print_test_header("BERTopic 模型初始化")

    model = BERTopicModel(
        n_topics=2,
        min_cluster_size=2,
        verbose=False
    )

    assert model.is_fitted == False
    assert model.embedding_model is not None

    print_pass("BERTopic 初始化成功")
    print(f"  - 是否訓練: {model.is_fitted}")


def test_bertopic_fit():
    """測試 BERTopic 訓練"""
    print_test_header("BERTopic 模型訓練")

    model = BERTopicModel(
        min_cluster_size=2,
        min_topic_size=2,
        verbose=False
    )

    topics, probs = model.fit_transform(SMALL_DOCUMENTS)

    assert len(topics) == len(SMALL_DOCUMENTS)
    assert len(probs) == len(SMALL_DOCUMENTS)
    assert model.is_fitted == True

    print_pass("BERTopic 訓練成功")
    print(f"  - 文檔數: {len(SMALL_DOCUMENTS)}")
    print(f"  - 發現主題: {len(set(topics))}")


def test_bertopic_get_topics():
    """測試獲取 BERTopic 主題"""
    print_test_header("BERTopic 獲取主題")

    model = BERTopicModel(
        min_cluster_size=2,
        verbose=False
    )
    model.fit(SMALL_DOCUMENTS)

    topics = model.get_topics()

    assert isinstance(topics, dict)

    print_pass(f"成功獲取 {len(topics)} 個主題")
    for topic_id, words in list(topics.items())[:2]:  # 只顯示前2個主題
        if topic_id != -1:  # 跳過 outlier topic
            print(f"\n  主題 {topic_id}:")
            for word, score in words[:3]:
                print(f"    - {word}: {score:.4f}")


def test_bertopic_transform():
    """測試 BERTopic 轉換"""
    print_test_header("BERTopic 轉換新文檔")

    model = BERTopicModel(
        min_cluster_size=2,
        verbose=False
    )
    model.fit(SMALL_DOCUMENTS)

    new_docs = ["機器學習", "文本處理"]
    topics, probs = model.transform(new_docs)

    assert len(topics) == len(new_docs)
    assert len(probs) == len(new_docs)

    print_pass(f"成功轉換 {len(new_docs)} 篇文檔")
    for i, (topic, prob) in enumerate(zip(topics, probs), 1):
        print(f"  文檔 {i}: 主題 {topic}, 概率 {prob:.4f}")


# ============================================================================
# 整合測試
# ============================================================================

def test_lda_bertopic_comparison():
    """測試 LDA 和 BERTopic 比較"""
    print_test_header("LDA vs BERTopic 比較")

    # LDA
    lda = LDAModel(n_topics=2, min_word_freq=1, use_stopwords=False, tokenizer_engine='jieba')
    lda.fit(SMALL_DOCUMENTS)
    lda_topics = lda.get_topics()

    # BERTopic
    bertopic = BERTopicModel(min_cluster_size=2, verbose=False)
    bertopic.fit(SMALL_DOCUMENTS)
    bertopic_topics = bertopic.get_topics()

    assert len(lda_topics) > 0
    assert len(bertopic_topics) > 0

    print_pass("兩種模型都能成功生成主題")
    print(f"  - LDA 主題數: {len(lda_topics)}")
    print(f"  - BERTopic 主題數: {len(bertopic_topics)}")


def test_topic_workflow():
    """測試完整主題建模工作流程"""
    print_test_header("完整主題建模工作流程")

    # 初始化
    lda = LDAModel(n_topics=2, min_word_freq=1, use_stopwords=False, tokenizer_engine='jieba')

    # 訓練
    train_docs = SMALL_DOCUMENTS[:2]
    lda.fit(train_docs)

    # 獲取主題
    topics = lda.get_topics()
    assert len(topics) > 0

    # 轉換新文檔
    test_docs = SMALL_DOCUMENTS[2:]
    topic_dists = lda.transform(test_docs)
    assert len(topic_dists) == len(test_docs)

    # 評估
    perplexity = lda.calculate_perplexity()
    assert perplexity > 0

    print_pass("完整工作流程測試成功")
    print(f"  - 訓練文檔: {len(train_docs)}")
    print(f"  - 測試文檔: {len(test_docs)}")
    print(f"  - 主題數: {len(topics)}")
    print(f"  - Perplexity: {perplexity:.4f}")


def run_all_tests():
    """執行所有測試"""
    print("\n" + "=" * 70)
    print("主題模型測試套件 (LDA & BERTopic)")
    print("=" * 70)

    test_functions = [
        # LDA tests
        test_lda_initialization,
        test_lda_fit,
        test_lda_get_topics,
        test_lda_get_topic_words,
        test_lda_transform,
        test_lda_perplexity,
        test_lda_coherence,
        # BERTopic tests
        test_bertopic_initialization,
        test_bertopic_fit,
        test_bertopic_get_topics,
        test_bertopic_transform,
        # Integration tests
        test_lda_bertopic_comparison,
        test_topic_workflow,
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
