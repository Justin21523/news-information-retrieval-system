"""
命名實體識別 (NER) 測試 (直接執行版本)

直接運行此腳本進行測試：
    python tests/test_ner.py

Author: Information Retrieval System
License: Educational Use
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ir.text.chinese_tokenizer import ChineseTokenizer
from src.ir.text.ner_extractor import NERExtractor, Entity


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


def test_tokenizer_ner_basic():
    """測試 ChineseTokenizer 的 NER 功能"""
    print_test_header("ChineseTokenizer NER 基本功能")

    tokenizer = ChineseTokenizer(engine='ckip')
    text = "張三在國立臺灣大學圖書資訊學系讀書"

    entities = tokenizer.extract_entities(text)

    assert len(entities) > 0, "應該找到至少一個實體"
    assert all(isinstance(e, tuple) for e in entities), "實體應該是 tuple"
    assert all(len(e) == 4 for e in entities), "每個實體應該有 4 個元素"

    print_pass(f"成功提取 {len(entities)} 個實體")
    print(f"\n  提取的實體:")
    for text, etype, start, end in entities:
        print(f"    - {text} ({etype}) [{start}:{end}]")


def test_tokenizer_ner_batch():
    """測試批次 NER 提取"""
    print_test_header("ChineseTokenizer 批次 NER 提取")

    tokenizer = ChineseTokenizer(engine='ckip')
    texts = [
        "張三在國立臺灣大學讀書",
        "2025年一月台北將舉辦研討會",
        "Google和Meta是知名科技公司"
    ]

    entities_batch = tokenizer.extract_entities_batch(texts)

    assert len(entities_batch) == len(texts), "應該返回相同數量的結果"
    assert all(isinstance(e_list, list) for e_list in entities_batch)

    print_pass(f"成功批次處理 {len(texts)} 篇文本")
    for i, entities in enumerate(entities_batch, 1):
        print(f"\n  文本 {i}: 找到 {len(entities)} 個實體")
        for text, etype, start, end in entities[:3]:  # 只顯示前 3 個
            print(f"    - {text} ({etype})")


def test_ner_extractor_initialization():
    """測試 NERExtractor 初始化"""
    print_test_header("NERExtractor 初始化")

    extractor = NERExtractor(
        entity_types={'PERSON', 'ORG', 'GPE'},
        device=-1
    )

    assert extractor.entity_types == {'PERSON', 'ORG', 'GPE'}
    assert extractor.device == -1
    assert extractor.tokenizer is not None

    print_pass("NERExtractor 初始化成功")
    print(f"  - 實體類型: {len(extractor.entity_types)}")
    print(f"  - 設備: CPU")


def test_ner_extractor_extract():
    """測試 NERExtractor 提取功能"""
    print_test_header("NERExtractor 實體提取")

    extractor = NERExtractor()
    text = "張三和李四在台灣大學學習自然語言處理"

    entities = extractor.extract(text)

    assert len(entities) > 0, "應該找到實體"
    assert all(isinstance(e, Entity) for e in entities), "應該返回 Entity 對象"
    assert all(hasattr(e, 'text') for e in entities)
    assert all(hasattr(e, 'type') for e in entities)

    print_pass(f"成功提取 {len(entities)} 個實體")
    print(f"\n  提取的實體:")
    for entity in entities:
        print(f"    - {entity}")


def test_ner_filter_by_type():
    """測試按類型過濾實體"""
    print_test_header("按類型過濾實體")

    extractor = NERExtractor()
    text = "張三在Google工作，李四在Meta工作"

    entities = extractor.extract(text)
    persons = extractor.filter_by_type(entities, ['PERSON'])
    orgs = extractor.filter_by_type(entities, ['ORG'])

    assert all(e.type == 'PERSON' for e in persons), "應該只有人名"
    assert all(e.type == 'ORG' for e in orgs), "應該只有組織"

    print_pass("過濾功能正常")
    print(f"  - 總實體數: {len(entities)}")
    print(f"  - 人名 (PERSON): {len(persons)}")
    for p in persons:
        print(f"      {p.text}")
    print(f"  - 組織 (ORG): {len(orgs)}")
    for o in orgs:
        print(f"      {o.text}")


def test_ner_filter_by_text():
    """測試按文本過濾實體"""
    print_test_header("按文本過濾實體")

    extractor = NERExtractor()
    text = "台灣大學、台灣師範大學、台灣科技大學都是知名學府"

    entities = extractor.extract(text)
    taiwan_entities = extractor.filter_by_text(entities, lambda t: '台灣' in t)

    assert all('台灣' in e.text for e in taiwan_entities), "應該都包含'台灣'"

    print_pass("文本過濾功能正常")
    print(f"  - 總實體數: {len(entities)}")
    print(f"  - 包含'台灣': {len(taiwan_entities)}")
    for e in taiwan_entities:
        print(f"      {e.text}")


def test_ner_statistics():
    """測試實體統計"""
    print_test_header("實體統計功能")

    extractor = NERExtractor()
    text = "張三和李四在台灣大學讀書，王五在Google工作"

    entities = extractor.extract(text)
    stats = extractor.entity_statistics(entities)

    assert 'total' in stats
    assert 'by_type' in stats
    assert stats['total'] == len(entities)

    print_pass("統計功能正常")
    print(f"\n  統計資訊:")
    print(f"    - 總實體數: {stats['total']}")
    print(f"    - 唯一實體數: {stats['unique']}")
    print(f"\n  按類型分布:")
    for etype, count in stats['by_type'].items():
        print(f"    - {etype}: {count}")


def test_ner_most_common():
    """測試最常見實體"""
    print_test_header("最常見實體")

    extractor = NERExtractor()
    text = "張三和李四在台北見面。張三說台北天氣很好。李四也覺得台北不錯。"

    entities = extractor.extract(text)
    most_common = extractor.most_common_entities(entities, top_n=3)

    assert len(most_common) <= 3
    assert all(isinstance(item, tuple) for item in most_common)

    print_pass(f"找到 {len(most_common)} 個最常見實體")
    print(f"\n  最常見實體:")
    for entity_text, count in most_common:
        print(f"    - {entity_text}: {count} 次")


def test_ner_group_by_type():
    """測試按類型分組"""
    print_test_header("按類型分組實體")

    extractor = NERExtractor()
    text = "張三和李四在Google和Meta討論AI技術"

    entities = extractor.extract(text)
    grouped = extractor.group_by_type(entities)

    assert isinstance(grouped, dict)
    assert all(isinstance(k, str) for k in grouped.keys())
    assert all(isinstance(v, list) for v in grouped.values())

    print_pass("分組功能正常")
    print(f"\n  按類型分組:")
    for etype, entity_list in grouped.items():
        print(f"    - {etype} ({len(entity_list)}):")
        for e in entity_list:
            print(f"        {e.text}")


def test_ner_batch_processing():
    """測試批次處理"""
    print_test_header("批次處理多篇文本")

    extractor = NERExtractor()
    texts = [
        "張三在台灣大學讀書",
        "李四在Google工作",
        "2025年1月在台北舉辦會議"
    ]

    entities_batch = extractor.extract_batch(texts)

    assert len(entities_batch) == len(texts)
    assert all(isinstance(e_list, list) for e_list in entities_batch)

    print_pass(f"成功批次處理 {len(texts)} 篇文本")
    total_entities = sum(len(e) for e in entities_batch)
    print(f"  - 總共找到 {total_entities} 個實體")
    for i, entities in enumerate(entities_batch, 1):
        print(f"\n  文本 {i}: {len(entities)} 個實體")
        for e in entities[:2]:  # 只顯示前 2 個
            print(f"    - {e}")


def test_entity_dataclass():
    """測試 Entity 數據類"""
    print_test_header("Entity 數據類")

    entity = Entity(
        text="台灣大學",
        type="ORG",
        start_pos=5,
        end_pos=9,
        source_text="張三在台灣大學讀書"
    )

    assert entity.text == "台灣大學"
    assert entity.type == "ORG"
    assert entity.start_pos == 5
    assert entity.end_pos == 9

    print_pass("Entity 類別運作正常")
    print(f"  - 實體文本: {entity.text}")
    print(f"  - 類型: {entity.type}")
    print(f"  - 位置: [{entity.start_pos}:{entity.end_pos}]")
    print(f"  - 字串表示: {str(entity)}")
    print(f"  - repr: {repr(entity)}")


def test_ner_empty_text():
    """測試空文本處理"""
    print_test_header("空文本處理")

    extractor = NERExtractor()

    # 空字串
    entities = extractor.extract("")
    assert len(entities) == 0, "空字串應該返回空列表"

    # 只有空白
    entities = extractor.extract("   ")
    assert len(entities) == 0, "只有空白應該返回空列表"

    print_pass("空文本處理正常")


def test_ner_integration():
    """測試完整工作流程"""
    print_test_header("NER 完整工作流程整合測試")

    # 初始化
    extractor = NERExtractor(
        entity_types={'PERSON', 'ORG', 'GPE', 'LOC'},
        device=-1
    )

    # 處理文本
    text = """
    張三是台灣大學的教授，專門研究人工智慧。
    他曾在Google和微軟工作過。
    2025年1月，他將在台北主持一場研討會。
    """

    entities = extractor.extract(text)

    # 過濾和統計
    persons = extractor.filter_by_type(entities, ['PERSON'])
    orgs = extractor.filter_by_type(entities, ['ORG'])
    stats = extractor.entity_statistics(entities)
    grouped = extractor.group_by_type(entities)

    print_pass("完整工作流程測試成功")
    print(f"\n  處理結果:")
    print(f"    - 總實體數: {len(entities)}")
    print(f"    - 人名: {len(persons)}")
    print(f"    - 組織: {len(orgs)}")
    print(f"\n  統計:")
    for key, value in stats.items():
        if key != 'by_type':
            print(f"    - {key}: {value}")
    print(f"\n  分組:")
    for etype, entity_list in grouped.items():
        print(f"    - {etype}: {len(entity_list)} 個")


def run_all_tests():
    """執行所有測試"""
    print("\n" + "=" * 70)
    print("命名實體識別 (NER) 測試套件")
    print("=" * 70)

    test_functions = [
        test_tokenizer_ner_basic,
        test_tokenizer_ner_batch,
        test_ner_extractor_initialization,
        test_ner_extractor_extract,
        test_ner_filter_by_type,
        test_ner_filter_by_text,
        test_ner_statistics,
        test_ner_most_common,
        test_ner_group_by_type,
        test_ner_batch_processing,
        test_entity_dataclass,
        test_ner_empty_text,
        test_ner_integration,
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
