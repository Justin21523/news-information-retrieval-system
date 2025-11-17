"""
Unit Tests for CSoundex Module

Comprehensive test suite for Chinese Soundex phonetic encoding system.

Test Categories:
    - Basic encoding functionality
    - Homophone matching
    - Variant character handling
    - Mixed text processing
    - Similarity calculation
    - Edge cases and error handling
    - Performance and caching

Author: Information Retrieval System
License: Educational Use
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ir.text.csoundex import CSoundex


# ============================================
# Fixtures
# ============================================

@pytest.fixture
def csoundex():
    """Create CSoundex instance for testing."""
    return CSoundex()


@pytest.fixture
def csoundex_with_tone():
    """Create CSoundex instance with tone enabled by default."""
    encoder = CSoundex()
    encoder.default_include_tone = True
    return encoder


# ============================================
# Basic Encoding Tests
# ============================================

@pytest.mark.unit
@pytest.mark.csoundex
class TestBasicEncoding:
    """Test basic character encoding functionality."""

    def test_encode_single_character(self, csoundex):
        """Test encoding a single Chinese character."""
        code = csoundex.encode_character("張")
        # zhang1: zh(8) + ang(9) = Z89
        assert code == "Z89", f"Expected 'Z89', got '{code}'"

    def test_encode_character_with_tone(self, csoundex):
        """Test encoding with tone information."""
        code = csoundex.encode_character("張", include_tone=True)
        # zhang1: zh(8) + ang(9) + tone(1) = Z891
        assert code == "Z891", f"Expected 'Z891', got '{code}'"

    def test_encode_multiple_characters(self, csoundex):
        """Test encoding multiple characters."""
        code = csoundex.encode("張三")
        codes = code.split()
        assert len(codes) == 2
        # zhang1: Z89, san1: S99
        assert codes[0] == "Z89"
        assert codes[1] == "S99"

    def test_encode_common_surname(self, csoundex):
        """Test encoding common Chinese surnames."""
        # Updated with correct encoding based on actual pinyin groupings
        surnames = {
            "王": "W09",    # wang2: 0(w-zero initial) + 9(ang-nasal)
            "李": "L54",    # li3: 5(l-alveolar nasal) + 4(i-vowel)
            "趙": "Z88",    # zhao4: 8(zh-retroflex) + 8(ao-diphthong ending in u)
            "錢": "Q79",    # qian2: 7(q-palatal) + 9(ian-nasal)
            "孫": "S99",    # sun1: 9(s-dental) + 9(un-nasal)
            "周": "Z88",    # zhou1: 8(zh-retroflex) + 8(ou-diphthong)
            "吳": "W05",    # wu2: 0(w-zero) + 5(u-vowel)
            "鄭": "Z89"     # zheng4: 8(zh-retroflex) + 9(eng-nasal)
        }

        for char, expected in surnames.items():
            code = csoundex.encode_character(char)
            assert code == expected, f"{char}: expected '{expected}', got '{code}'"


# ============================================
# Homophone Matching Tests
# ============================================

@pytest.mark.unit
@pytest.mark.csoundex
class TestHomophoneMatching:
    """Test homophone (same pronunciation) matching."""

    def test_exact_homophones(self, csoundex):
        """Test characters with identical pronunciation."""
        # zhang1 group
        homophones = ["張", "章", "彰"]
        codes = [csoundex.encode_character(c) for c in homophones]

        # All should have same code
        assert len(set(codes)) == 1, f"Homophones have different codes: {codes}"
        assert codes[0] == "Z89"  # zhang1: zh(8) + ang(9)

    def test_tone_variation_homophones(self, csoundex):
        """Test homophones with different tones (without tone encoding)."""
        # shi with different tones
        chars = ["詩", "時", "史", "試"]  # shi1, shi2, shi3, shi4
        codes = [csoundex.encode_character(c, include_tone=False) for c in chars]

        # All should have similar pattern (S8xx)
        for code in codes:
            assert code.startswith("S8"), f"Expected S8xx pattern, got {code}"

    def test_li_homophones(self, csoundex):
        """Test homophones for 'li' sound."""
        # li3 group
        chars = ["李", "理", "裡"]
        codes = [csoundex.encode_character(c) for c in chars]

        # All should map to L5xx (l + i/li final)
        for code in codes:
            assert code.startswith("L5"), f"Expected L5xx, got {code}"


# ============================================
# Variant Character Tests
# ============================================

@pytest.mark.unit
@pytest.mark.csoundex
class TestVariantCharacters:
    """Test handling of variant characters (異體字)."""

    def test_traditional_simplified_variants(self, csoundex):
        """Test traditional and simplified character variants."""
        pairs = [
            ("裏", "裡"),  # li3
            ("台", "臺"),  # tai2
        ]

        for var1, var2 in pairs:
            code1 = csoundex.encode_character(var1)
            code2 = csoundex.encode_character(var2)
            # Should have same or very similar codes
            assert code1[:2] == code2[:2], f"{var1} ({code1}) vs {var2} ({code2})"


# ============================================
# Mixed Text Processing Tests
# ============================================

@pytest.mark.unit
@pytest.mark.csoundex
class TestMixedText:
    """Test processing of mixed Chinese/English/punctuation text."""

    def test_pure_chinese(self, csoundex):
        """Test pure Chinese text."""
        code = csoundex.encode("信息檢索")
        assert code.count(" ") == 3, "Should have 4 characters"

    def test_chinese_with_english(self, csoundex):
        """Test mixed Chinese and English."""
        code = csoundex.encode("hello世界")
        codes = code.split()
        # Should have: H E L L O + 2 Chinese
        assert len(codes) == 7

    def test_punctuation_handling(self, csoundex):
        """Test that punctuation is ignored."""
        text1 = "三聚氰胺"
        text2 = "三、聚、氰、胺！"

        code1 = csoundex.encode(text1)
        code2 = csoundex.encode(text2)

        # Remove spaces and compare
        assert code1.replace(" ", "") == code2.replace(" ", "")

    def test_numbers_in_text(self, csoundex):
        """Test handling of numbers."""
        code = csoundex.encode("第123章")
        codes = code.split()
        # Numbers should be ignored, only 第 and 章
        assert len(codes) == 2


# ============================================
# Real-World Examples Tests
# ============================================

@pytest.mark.integration
@pytest.mark.csoundex
class TestRealWorldExamples:
    """Test real-world use cases."""

    def test_melamine_scandal(self, csoundex):
        """Test encoding of '三聚氰胺' (melamine)."""
        code = csoundex.encode("三聚氰胺")
        codes = code.split()

        assert len(codes) == 4
        assert codes[0] == "S99"  # 三 san1: s(9) + an(9)
        assert codes[1] == "J75"  # 聚 ju4: j(7) + u(5)
        # 氰 qing2, 胺 an4

    def test_information_retrieval(self, csoundex):
        """Test IR-related terms."""
        terms = ["信息檢索", "資訊檢索", "搜尋引擎"]

        for term in terms:
            code = csoundex.encode(term)
            assert len(code.split()) == 4, f"{term} should encode to 4 codes"

    def test_chinese_names(self, csoundex):
        """Test encoding of common Chinese names."""
        names = ["張偉", "王芳", "李娜", "劉洋"]

        for name in names:
            code = csoundex.encode(name)
            codes = code.split()
            assert len(codes) == 2, f"Name {name} should have 2 parts"


# ============================================
# Similarity Calculation Tests
# ============================================

@pytest.mark.unit
@pytest.mark.csoundex
class TestSimilarity:
    """Test similarity calculation between texts."""

    def test_exact_match(self, csoundex):
        """Test similarity of identical texts."""
        sim = csoundex.similarity("張三", "張三", mode='exact')
        assert sim == 1.0

    def test_homophone_similarity(self, csoundex):
        """Test similarity of homophones."""
        sim = csoundex.similarity("張三", "章三", mode='fuzzy')
        # 張≈章 (1.0), 三=三 (1.0), average = 1.0
        assert sim == 1.0, f"Expected 1.0, got {sim}"

    def test_partial_similarity(self, csoundex):
        """Test partial similarity."""
        sim = csoundex.similarity("張三", "張四", mode='fuzzy')
        # 張=張 (1.0), 三≠四 (0.0), so 0.5
        assert sim == 0.5, f"Expected 0.5, got {sim}"

    def test_no_similarity(self, csoundex):
        """Test completely different texts."""
        sim = csoundex.similarity("張三", "李四", mode='fuzzy')
        assert sim == 0.0

    def test_weighted_similarity(self, csoundex):
        """Test position-weighted similarity."""
        # Same first character, different second
        sim = csoundex.similarity("張三", "張四", mode='weighted')
        # First position has higher weight
        assert 0.5 < sim < 1.0


# ============================================
# Batch Processing Tests
# ============================================

@pytest.mark.unit
@pytest.mark.csoundex
class TestBatchProcessing:
    """Test batch encoding functionality."""

    def test_encode_batch(self, csoundex):
        """Test encoding multiple texts."""
        texts = ["張三", "李四", "王五"]
        codes = csoundex.encode_batch(texts)

        assert len(codes) == 3
        assert all(isinstance(code, str) for code in codes)

    def test_find_similar(self, csoundex):
        """Test finding similar texts."""
        query = "張偉"
        candidates = ["張偉", "章偉", "張維", "李偉", "王偉"]

        results = csoundex.find_similar(query, candidates, threshold=0.5)

        # Should find at least: 張偉 (1.0), 章偉 (0.5+), 張維 (0.5+)
        assert len(results) >= 3
        # Results should be sorted by score
        assert all(results[i][1] >= results[i+1][1] for i in range(len(results)-1))

    def test_find_similar_with_topk(self, csoundex):
        """Test finding similar texts with topk limit."""
        query = "張三"
        candidates = ["張三", "章三", "張四", "李三", "王三"]

        results = csoundex.find_similar(query, candidates, threshold=0.0, topk=2)

        assert len(results) == 2
        assert results[0][1] >= results[1][1]  # Sorted


# ============================================
# Edge Cases and Error Handling
# ============================================

@pytest.mark.unit
@pytest.mark.csoundex
class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_string(self, csoundex):
        """Test encoding empty string."""
        code = csoundex.encode("")
        assert code == ""

    def test_single_character(self, csoundex):
        """Test encoding single character."""
        code = csoundex.encode("我")
        # wo3: w(0) + o(2) = W02
        assert code == "W02"

    def test_whitespace_only(self, csoundex):
        """Test encoding whitespace."""
        code = csoundex.encode("   ")
        assert code == ""

    def test_pure_punctuation(self, csoundex):
        """Test encoding only punctuation."""
        code = csoundex.encode("！？。，")
        assert code == ""

    def test_pure_english(self, csoundex):
        """Test encoding pure English."""
        code = csoundex.encode("hello")
        assert code == "H E L L O"

    def test_unknown_character(self, csoundex):
        """Test handling of unknown/rare characters."""
        # Use a rare character that might not be in lexicon
        code = csoundex.encode_character("𠮷")  # Rare variant of 吉
        # Should return something (either code or original char)
        assert code is not None
        assert len(code) > 0


# ============================================
# Configuration Tests
# ============================================

@pytest.mark.unit
@pytest.mark.csoundex
class TestConfiguration:
    """Test configuration and initialization."""

    def test_default_initialization(self):
        """Test CSoundex with default configuration."""
        csoundex = CSoundex()
        assert csoundex.initial_groups is not None
        assert csoundex.final_groups is not None

    def test_custom_config_path(self):
        """Test CSoundex with custom config path."""
        config_path = Path(__file__).parent.parent / 'configs' / 'csoundex.yaml'
        csoundex = CSoundex(config_path=str(config_path))
        assert csoundex.config is not None

    def test_lexicon_loading(self, csoundex):
        """Test that lexicon is loaded."""
        # Should have loaded basic_pinyin.tsv
        assert len(csoundex.lexicon) > 0
        # Check for some common characters
        assert "張" in csoundex.lexicon
        assert "三" in csoundex.lexicon


# ============================================
# Normalization Tests
# ============================================

@pytest.mark.unit
@pytest.mark.csoundex
class TestNormalization:
    """Test pinyin normalization."""

    def test_normalize_with_tone(self, csoundex):
        """Test normalizing pinyin with tone number."""
        initial, final, tone = csoundex.normalize_pinyin("zhang1")
        assert initial == "zh"
        assert final == "ang"
        assert tone == "1"

    def test_normalize_without_tone(self, csoundex):
        """Test normalizing pinyin without tone."""
        initial, final, tone = csoundex.normalize_pinyin("zhang")
        assert initial == "zh"
        assert final == "ang"
        assert tone == "0"

    def test_normalize_uppercase(self, csoundex):
        """Test normalizing uppercase pinyin."""
        initial, final, tone = csoundex.normalize_pinyin("ZHANG1")
        assert initial == "zh"
        assert final == "ang"
        assert tone == "1"

    def test_normalize_zero_initial(self, csoundex):
        """Test normalizing pinyin with zero initial (pure vowel)."""
        initial, final, tone = csoundex.normalize_pinyin("an1")
        assert initial == ""
        assert final == "an"
        assert tone == "1"


# ============================================
# Performance and Caching Tests
# ============================================

@pytest.mark.performance
@pytest.mark.csoundex
class TestPerformance:
    """Test performance and caching behavior."""

    def test_cache_enabled(self, csoundex):
        """Test that caching is working."""
        # Clear cache first
        csoundex.clear_cache()

        # First call - cache miss
        code1 = csoundex.encode_character("張")

        # Second call - cache hit
        code2 = csoundex.encode_character("張")

        assert code1 == code2

        # Check cache info
        info = csoundex.get_cache_info()
        assert info['hits'] >= 1
        assert info['size'] > 0

    def test_large_batch_encoding(self, csoundex):
        """Test encoding large batch of texts."""
        texts = ["張三"] * 100
        codes = csoundex.encode_batch(texts)

        assert len(codes) == 100
        # All should be identical
        assert len(set(codes)) == 1

    @pytest.mark.slow
    def test_encoding_speed(self, csoundex):
        """Test encoding speed for large text."""
        import time

        text = "信息檢索系統是現代資訊科學的重要組成部分" * 100

        start = time.time()
        code = csoundex.encode(text)
        elapsed = time.time() - start

        # Should complete in reasonable time (< 1 second)
        assert elapsed < 1.0, f"Encoding took {elapsed:.3f}s, expected < 1.0s"


# ============================================
# Integration Tests
# ============================================

@pytest.mark.integration
@pytest.mark.csoundex
class TestIntegration:
    """Integration tests for complete workflows."""

    def test_name_matching_workflow(self, csoundex):
        """Test complete name matching workflow."""
        # Step 1: Encode a database of names
        database = ["張偉", "王芳", "李娜", "劉洋", "陳靜", "楊軍"]
        db_codes = [(name, csoundex.encode(name)) for name in database]

        # Step 2: Search for similar name
        query = "張偉"
        query_code = csoundex.encode(query)

        # Step 3: Find matches
        matches = []
        for name, code in db_codes:
            if code == query_code:
                matches.append(name)

        # Should find exact match
        assert "張偉" in matches

    def test_deduplication_workflow(self, csoundex):
        """Test deduplication of homophone names."""
        names = ["張三", "章三", "張三", "彰三"]

        # Group by phonetic code
        groups = {}
        for name in names:
            code = csoundex.encode(name)
            if code not in groups:
                groups[code] = []
            groups[code].append(name)

        # All should map to same group
        assert len(groups) == 1

    def test_query_expansion_workflow(self, csoundex):
        """Test query expansion using homophones."""
        query = "信息"
        query_code = csoundex.encode(query)

        # Potential expansions
        candidates = ["信息", "資訊", "消息", "訊息"]

        # Find phonetically similar
        similar = csoundex.find_similar(query, candidates, threshold=0.3)

        # Should find at least original and some variations
        assert len(similar) >= 1


# ============================================
# Special Test Markers
# ============================================

def test_suite_completeness():
    """Meta-test to ensure we have enough tests."""
    # This test file should have at least 30 test functions
    import inspect

    test_functions = [
        name for name, obj in globals().items()
        if name.startswith('test_') and callable(obj)
    ]

    # Count test methods in test classes
    test_classes = [
        obj for name, obj in globals().items()
        if inspect.isclass(obj) and name.startswith('Test')
    ]

    test_method_count = 0
    for cls in test_classes:
        methods = [name for name in dir(cls) if name.startswith('test_')]
        test_method_count += len(methods)

    total_tests = len(test_functions) + test_method_count

    assert total_tests >= 30, f"Only {total_tests} tests found, expected >= 30"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
