"""
Tests for PAT-tree Pattern Mining Module

This test suite covers:
    - PAT-tree construction
    - Pattern extraction
    - Mutual Information calculation
    - Search and query operations
    - Integration with Chinese tokenization

Author: Information Retrieval System
License: Educational Use
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ir.patterns import PATTree, Pattern, PATNode
from src.ir.text.chinese_tokenizer import ChineseTokenizer


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_tokens():
    """Simple token sequences for testing."""
    return [
        ['機器', '學習', '是', '重要', '技術'],
        ['機器', '學習', '和', '深度', '學習'],
        ['深度', '學習', '是', '機器', '學習']
    ]


@pytest.fixture
def chinese_tokenizer():
    """Chinese tokenizer for integration tests."""
    return ChineseTokenizer(engine='jieba')


# ============================================================================
# PATNode Tests
# ============================================================================

class TestPATNode:
    """Tests for PATNode class."""

    def test_node_initialization(self):
        """Test PATNode initialization."""
        node = PATNode()

        assert len(node.children) == 0
        assert node.frequency == 0
        assert node.is_end == False
        assert len(node.positions) == 0

    def test_node_add_child(self):
        """Test adding child nodes."""
        node = PATNode()
        child = PATNode()

        node.children['token'] = child

        assert 'token' in node.children
        assert node.children['token'] == child


# ============================================================================
# PATTree Construction Tests
# ============================================================================

class TestPATTreeConstruction:
    """Tests for PAT-tree construction."""

    def test_tree_initialization(self):
        """Test PAT-tree initialization."""
        tree = PATTree(
            min_pattern_length=2,
            max_pattern_length=5,
            min_frequency=2
        )

        assert tree.min_pattern_length == 2
        assert tree.max_pattern_length == 5
        assert tree.min_frequency == 2
        assert tree.total_tokens == 0
        assert len(tree.token_freq) == 0

    def test_insert_single_sequence(self):
        """Test inserting a single token sequence."""
        tree = PATTree(min_pattern_length=2)
        tokens = ['A', 'B', 'C']

        tree.insert_sequence(tokens)

        assert tree.total_tokens == 3
        assert tree.token_freq['A'] == 1
        assert tree.token_freq['B'] == 1
        assert tree.token_freq['C'] == 1

    def test_insert_multiple_sequences(self, simple_tokens):
        """Test inserting multiple sequences."""
        tree = PATTree(min_pattern_length=2)

        for tokens in simple_tokens:
            tree.insert_sequence(tokens)

        assert tree.total_tokens == sum(len(t) for t in simple_tokens)
        assert tree.token_freq['機器'] == 3
        assert tree.token_freq['學習'] == 5  # Appears in all 3 sequences

    def test_insert_empty_sequence(self):
        """Test inserting empty sequence."""
        tree = PATTree()
        tree.insert_sequence([])

        assert tree.total_tokens == 0
        assert len(tree.token_freq) == 0


# ============================================================================
# Pattern Extraction Tests
# ============================================================================

class TestPatternExtraction:
    """Tests for pattern extraction."""

    def test_extract_patterns_basic(self, simple_tokens):
        """Test basic pattern extraction."""
        tree = PATTree(min_pattern_length=2, min_frequency=2)

        for tokens in simple_tokens:
            tree.insert_sequence(tokens)

        patterns = tree.extract_patterns(use_mi_score=False)

        # Should find patterns that appear at least twice
        assert len(patterns) > 0
        assert all(isinstance(p, Pattern) for p in patterns)
        assert all(p.frequency >= 2 for p in patterns)

    def test_extract_with_mi_scores(self, simple_tokens):
        """Test pattern extraction with MI scoring."""
        tree = PATTree(min_pattern_length=2, min_frequency=2)

        for tokens in simple_tokens:
            tree.insert_sequence(tokens)

        patterns = tree.extract_patterns(use_mi_score=True)

        assert len(patterns) > 0
        # Patterns should be sorted by MI score (descending)
        for i in range(len(patterns) - 1):
            assert patterns[i].mi_score >= patterns[i + 1].mi_score

    def test_extract_top_k(self, simple_tokens):
        """Test extracting top-k patterns."""
        tree = PATTree(min_pattern_length=2, min_frequency=1)

        for tokens in simple_tokens:
            tree.insert_sequence(tokens)

        patterns = tree.extract_patterns(top_k=5)

        assert len(patterns) <= 5

    def test_extract_min_frequency(self):
        """Test minimum frequency filtering."""
        tree = PATTree(min_pattern_length=2, min_frequency=3)

        # Insert sequences where some patterns appear less than 3 times
        for _ in range(2):
            tree.insert_sequence(['A', 'B', 'C'])

        for _ in range(3):
            tree.insert_sequence(['D', 'E', 'F'])

        patterns = tree.extract_patterns()

        # Only patterns from 'D', 'E', 'F' should appear (freq=3)
        assert len(patterns) > 0
        assert all(p.frequency >= 3 for p in patterns)

    def test_extract_pattern_length_constraints(self):
        """Test min/max pattern length constraints."""
        tree = PATTree(
            min_pattern_length=3,
            max_pattern_length=3,
            min_frequency=1
        )

        tree.insert_sequence(['A', 'B', 'C', 'D', 'E'])

        patterns = tree.extract_patterns()

        # All patterns should be exactly length 3
        assert all(len(p.tokens) == 3 for p in patterns)


# ============================================================================
# Mutual Information Tests
# ============================================================================

class TestMutualInformation:
    """Tests for Mutual Information calculation."""

    def test_mi_calculation(self):
        """Test MI score calculation."""
        tree = PATTree(min_pattern_length=2)

        # Create pattern with known frequencies
        for _ in range(10):
            tree.insert_sequence(['A', 'B'])

        for _ in range(5):
            tree.insert_sequence(['A', 'X'])

        for _ in range(5):
            tree.insert_sequence(['Y', 'B'])

        patterns = tree.extract_patterns(use_mi_score=True)

        # Pattern 'AB' should have high MI (appears together more than expected)
        ab_pattern = [p for p in patterns if list(p.tokens) == ['A', 'B']]
        assert len(ab_pattern) > 0
        assert ab_pattern[0].mi_score > 0

    def test_pmi_method(self):
        """Test PMI calculation method."""
        tree = PATTree(min_pattern_length=2)

        tree.insert_sequence(['A', 'B', 'C'])

        patterns = tree.extract_patterns(top_k=1, use_mi_score=True)

        if patterns:
            pmi = tree.calculate_pmi(patterns[0])
            assert isinstance(pmi, float)

    def test_mi_zero_frequency(self):
        """Test MI with zero frequency (edge case)."""
        tree = PATTree()

        # Create a pattern manually
        pattern = Pattern(
            tokens=('A', 'B'),
            frequency=0,
            mi_score=0.0,
            positions=[]
        )

        mi = tree._calculate_mi(pattern)

        # Should return 0 for zero frequency
        assert mi == 0.0


# ============================================================================
# Search and Query Tests
# ============================================================================

class TestSearchQuery:
    """Tests for search and query operations."""

    def test_search_existing_pattern(self):
        """Test searching for existing pattern."""
        tree = PATTree()

        tree.insert_sequence(['A', 'B', 'C'])

        node = tree.search(['A', 'B'])

        assert node is not None
        assert node.frequency == 1

    def test_search_nonexistent_pattern(self):
        """Test searching for non-existent pattern."""
        tree = PATTree()

        tree.insert_sequence(['A', 'B', 'C'])

        node = tree.search(['X', 'Y'])

        assert node is None

    def test_get_frequency(self):
        """Test getting pattern frequency."""
        tree = PATTree()

        for _ in range(3):
            tree.insert_sequence(['A', 'B', 'C'])

        freq = tree.get_frequency(['A', 'B'])

        assert freq == 3

    def test_get_frequency_zero(self):
        """Test getting frequency for non-existent pattern."""
        tree = PATTree()

        tree.insert_sequence(['A', 'B'])

        freq = tree.get_frequency(['X', 'Y'])

        assert freq == 0


# ============================================================================
# Statistics Tests
# ============================================================================

class TestStatistics:
    """Tests for tree statistics."""

    def test_get_statistics(self, simple_tokens):
        """Test getting tree statistics."""
        tree = PATTree(min_pattern_length=2, max_pattern_length=4)

        for tokens in simple_tokens:
            tree.insert_sequence(tokens)

        stats = tree.get_statistics()

        assert 'total_tokens' in stats
        assert 'unique_tokens' in stats
        assert 'total_nodes' in stats
        assert stats['total_tokens'] == sum(len(t) for t in simple_tokens)
        assert stats['unique_tokens'] > 0
        assert stats['total_nodes'] >= 1  # At least root node

    def test_repr(self):
        """Test string representation."""
        tree = PATTree()

        tree.insert_sequence(['A', 'B', 'C'])

        repr_str = repr(tree)

        assert 'PATTree' in repr_str
        assert 'tokens=' in repr_str


# ============================================================================
# Pattern Class Tests
# ============================================================================

class TestPattern:
    """Tests for Pattern class."""

    def test_pattern_initialization(self):
        """Test Pattern initialization."""
        pattern = Pattern(
            tokens=('機器', '學習'),
            frequency=5,
            mi_score=2.5,
            positions=[0, 10, 20]
        )

        assert pattern.tokens == ('機器', '學習')
        assert pattern.frequency == 5
        assert pattern.mi_score == 2.5
        assert len(pattern.positions) == 3

    def test_pattern_text_property(self):
        """Test Pattern text property."""
        pattern = Pattern(
            tokens=('機器', '學習'),
            frequency=1,
            mi_score=0.0,
            positions=[0]
        )

        assert pattern.text == '機器學習'

    def test_pattern_repr(self):
        """Test Pattern string representation."""
        pattern = Pattern(
            tokens=('A', 'B'),
            frequency=3,
            mi_score=1.5,
            positions=[0]
        )

        repr_str = repr(pattern)

        assert 'Pattern' in repr_str
        assert 'freq=3' in repr_str
        assert 'MI=' in repr_str


# ============================================================================
# Integration Tests
# ============================================================================

class TestPATTreeIntegration:
    """Integration tests with Chinese tokenizer."""

    def test_insert_text_with_tokenizer(self, chinese_tokenizer):
        """Test inserting text using Chinese tokenizer."""
        tree = PATTree(min_pattern_length=2, min_frequency=1)

        text = "機器學習是人工智慧的重要技術"

        tree.insert_text(text, chinese_tokenizer)

        assert tree.total_tokens > 0
        patterns = tree.extract_patterns()
        assert len(patterns) > 0

    def test_multiple_texts_chinese(self, chinese_tokenizer):
        """Test processing multiple Chinese texts."""
        tree = PATTree(min_pattern_length=2, min_frequency=2)

        texts = [
            "機器學習和深度學習都很重要",
            "深度學習是機器學習的子領域",
            "機器學習技術發展迅速"
        ]

        for text in texts:
            tree.insert_text(text, chinese_tokenizer)

        patterns = tree.extract_patterns(top_k=10, use_mi_score=True)

        assert len(patterns) > 0

        # Check that '機器學習' is extracted (appears in all texts)
        ml_patterns = [p for p in patterns if '機器' in p.text and '學習' in p.text]
        assert len(ml_patterns) > 0

    def test_pattern_mining_workflow(self, chinese_tokenizer):
        """Test complete pattern mining workflow."""
        # Initialize tree
        tree = PATTree(
            min_pattern_length=2,
            max_pattern_length=4,
            min_frequency=2
        )

        # Process corpus
        texts = [
            "自然語言處理是重要技術",
            "機器學習和自然語言處理都很重要",
            "深度學習用於自然語言處理"
        ]

        for text in texts:
            tree.insert_text(text, chinese_tokenizer)

        # Extract patterns
        patterns = tree.extract_patterns(top_k=5, use_mi_score=True)

        # Verify results
        assert len(patterns) > 0
        assert all(p.frequency >= 2 for p in patterns)
        assert all(p.mi_score > 0 for p in patterns)

        # Statistics
        stats = tree.get_statistics()
        assert stats['total_tokens'] > 0


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v', '--tb=short'])
