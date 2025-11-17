"""Tests for Summarization Algorithms"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ir.summarize.static import StaticSummarizer, Sentence, Summary
from src.ir.summarize.dynamic import KWICGenerator, KWICMatch, KWICResult


# ============================================================================
# Static Summarization Tests
# ============================================================================

@pytest.fixture
def static_summarizer():
    return StaticSummarizer()


@pytest.fixture
def sample_text():
    return """
    First sentence is here with some extra words. Second sentence follows with additional content. Third sentence appears with more text.
    Fourth sentence continues with extra information. Fifth sentence ends the document with final words.
    """


@pytest.fixture
def multi_doc_texts():
    return [
        "Python is a programming language. It is easy to learn.",
        "Java is widely used. It is platform independent.",
        "JavaScript runs in browsers. It is essential for web development."
    ]


@pytest.mark.unit
class TestStaticSummarization:
    def test_sentence_segmentation(self, static_summarizer, sample_text):
        """Test sentence segmentation."""
        sentences = static_summarizer.segment_sentences(sample_text)
        assert len(sentences) == 5
        assert all(isinstance(s, Sentence) for s in sentences)
        assert sentences[0].position == 0

    def test_lead_k_summarization(self, static_summarizer, sample_text):
        """Test Lead-k summarization."""
        summary = static_summarizer.lead_k_summarization(sample_text, k=3)
        assert isinstance(summary, Summary)
        assert summary.length == 3
        assert summary.method == 'lead-k'
        assert summary.compression_ratio == pytest.approx(0.6)
        assert "First" in summary.text

    def test_key_sentence_extraction(self, static_summarizer, sample_text):
        """Test key sentence extraction."""
        summary = static_summarizer.key_sentence_extraction(sample_text, k=2)
        assert summary.length == 2
        assert summary.method == 'key-sentence-tfidf'
        assert all(hasattr(s, 'score') for s in summary.sentences)

    def test_query_focused_summarization(self, static_summarizer):
        """Test query-focused summarization."""
        text = "Dogs are very loyal animals indeed. Cats are independent creatures by nature. Birds can fly high in the sky."
        summary = static_summarizer.query_focused_summarization(text, "dogs cats", k=2)
        assert summary.length == 2
        assert summary.method == 'query-focused'

    def test_multi_document_summarization(self, static_summarizer, multi_doc_texts):
        """Test multi-document summarization."""
        summary = static_summarizer.multi_document_summarization(multi_doc_texts, k=3)
        assert summary.length == 3
        assert summary.method == 'multi-document'
        # Check that sentences come from different documents
        doc_ids = set(s.doc_id for s in summary.sentences)
        assert len(doc_ids) >= 2

    def test_empty_text(self, static_summarizer):
        """Test empty text handling."""
        summary = static_summarizer.lead_k_summarization("", k=3)
        assert summary.length == 0
        assert summary.original_length == 0

    def test_compute_term_frequencies(self, static_summarizer):
        """Test term frequency computation."""
        sentences = static_summarizer.segment_sentences("Machine learning is a field of study. Machine intelligence continues to grow.")
        tf = static_summarizer.compute_term_frequencies(sentences)
        assert 'machine' in tf
        assert tf['machine'] == 2

    def test_compute_idf(self, static_summarizer):
        """Test IDF computation."""
        sentences = static_summarizer.segment_sentences("First sentence with enough words here. Second sentence with more words there. Third text with different content.")
        idf = static_summarizer.compute_idf(sentences)
        assert 'sentence' in idf or 'with' in idf
        if 'sentence' in idf:
            assert idf['sentence'] > 0


# ============================================================================
# Dynamic Summarization (KWIC) Tests
# ============================================================================

@pytest.fixture
def kwic_generator():
    return KWICGenerator(window_size=30, window_type='fixed')


@pytest.fixture
def kwic_text():
    return "Machine learning is a subset of artificial intelligence. Deep learning uses neural networks."


@pytest.mark.unit
class TestKWICGeneration:
    def test_basic_kwic_generation(self, kwic_generator, kwic_text):
        """Test basic KWIC generation."""
        result = kwic_generator.generate(kwic_text, "machine learning")
        assert isinstance(result, KWICResult)
        assert result.num_matches >= 1
        assert result.query == "machine learning"

    def test_kwic_match_structure(self, kwic_generator, kwic_text):
        """Test KWIC match structure."""
        result = kwic_generator.generate(kwic_text, "learning")
        assert len(result.matches) >= 1
        match = result.matches[0]
        assert isinstance(match, KWICMatch)
        assert match.keyword == "learning"
        assert isinstance(match.position, int)
        assert isinstance(match.left_context, str)
        assert isinstance(match.right_context, str)

    def test_kwic_case_insensitive(self):
        """Test case-insensitive KWIC."""
        generator = KWICGenerator(case_sensitive=False)
        text = "Machine LEARNING is powerful"
        result = generator.generate(text, "learning")
        assert result.num_matches >= 1

    def test_kwic_case_sensitive(self):
        """Test case-sensitive KWIC."""
        generator = KWICGenerator(case_sensitive=True)
        text = "Machine LEARNING is powerful"
        result = generator.generate(text, "learning")
        assert result.num_matches == 0  # Should not match "LEARNING"

    def test_kwic_sentence_window(self):
        """Test sentence window extraction."""
        generator = KWICGenerator(window_type='sentence')
        text = "First sentence. Second sentence with keyword. Third sentence."
        result = generator.generate(text, "keyword")
        assert result.num_matches == 1
        match = result.matches[0]
        # Should extract the full sentence
        assert "Second sentence" in match.left_context or "Second sentence" in match.snippet

    def test_kwic_adaptive_window(self):
        """Test adaptive window extraction."""
        generator = KWICGenerator(window_size=50, window_type='adaptive')
        text = "This is a test, with some keyword, and more text."
        result = generator.generate(text, "keyword")
        assert result.num_matches == 1

    def test_kwic_multiple_matches(self, kwic_generator):
        """Test multiple keyword matches."""
        text = "First match here. Second match there. Third match everywhere."
        result = kwic_generator.generate(text, "match")
        assert result.num_matches == 3

    def test_kwic_max_matches(self, kwic_generator):
        """Test max matches limit."""
        text = "match match match match match"
        result = kwic_generator.generate(text, "match", max_matches=2)
        assert result.num_matches == 2

    def test_kwic_multi_document(self, kwic_generator):
        """Test multi-document KWIC."""
        docs = [
            "First document with keyword.",
            "Second document with keyword.",
            "Third document without it."
        ]
        result = kwic_generator.generate_multi(docs, "keyword", max_matches_per_doc=1)
        assert result.num_matches >= 2
        assert result.num_documents == 3

    def test_kwic_no_match(self, kwic_generator, kwic_text):
        """Test no match case."""
        result = kwic_generator.generate(kwic_text, "nonexistent")
        assert result.num_matches == 0
        assert len(result.matches) == 0

    def test_kwic_cache(self):
        """Test KWIC caching."""
        generator = KWICGenerator(enable_cache=True)
        text = "Test text for caching"

        # First call - cache miss
        result1 = generator.generate(text, "test")
        assert result1.cache_hit == False

        # Second call - cache hit
        result2 = generator.generate(text, "test")
        assert result2.cache_hit == True

        # Verify cache stats
        stats = generator.get_cache_stats()
        assert stats['size'] >= 1

    def test_kwic_cache_clear(self):
        """Test cache clearing."""
        generator = KWICGenerator(enable_cache=True)
        text = "Test text"
        generator.generate(text, "test")

        # Clear cache
        generator.clear_cache()
        stats = generator.get_cache_stats()
        assert stats['size'] == 0

    def test_kwic_formatting(self, kwic_generator, kwic_text):
        """Test result formatting."""
        result = kwic_generator.generate(kwic_text, "learning")
        output = kwic_generator.format_results(result, highlight_style='markdown')
        assert isinstance(output, str)
        assert '**learning**' in output or result.num_matches == 0

    def test_kwic_snippet_property(self, kwic_generator, kwic_text):
        """Test snippet property."""
        result = kwic_generator.generate(kwic_text, "machine")
        if result.num_matches > 0:
            match = result.matches[0]
            snippet = match.snippet
            assert '**' in snippet  # Markdown highlighting
            assert match.keyword in snippet

    def test_get_snippets(self, kwic_generator, kwic_text):
        """Test get_snippets method."""
        result = kwic_generator.generate(kwic_text, "learning")
        snippets = result.get_snippets(max_snippets=2)
        assert isinstance(snippets, list)
        assert len(snippets) <= 2


# ============================================================================
# Edge Cases and Integration
# ============================================================================

@pytest.mark.unit
class TestEdgeCases:
    def test_single_word_document(self, static_summarizer):
        """Test single word document."""
        summary = static_summarizer.lead_k_summarization("Word", k=1)
        assert summary.length == 0  # Too short, filtered out

    def test_very_long_sentence(self):
        """Test very long sentence filtering."""
        summarizer = StaticSummarizer(max_sentence_length=10)
        # Create sentence with 15 words
        text = " ".join(["word"] * 15) + "."
        sentences = summarizer.segment_sentences(text)
        assert len(sentences) == 0  # Filtered by length

    def test_special_characters(self, static_summarizer):
        """Test text with special characters."""
        text = "First sentence with enough words! Second sentence has more content? Third is a longer sentence indeed... Fourth sentence completes the test."
        sentences = static_summarizer.segment_sentences(text)
        assert len(sentences) >= 3

    def test_unicode_text(self, static_summarizer):
        """Test Unicode text."""
        text = "中文句子需要有足夠的長度才能通過測試。English sentence with enough words to pass. Mixed sentence需要更多的文字內容。"
        summary = static_summarizer.lead_k_summarization(text, k=2)
        assert summary.length >= 1

    def test_kwic_empty_query(self, kwic_generator):
        """Test KWIC with empty query."""
        result = kwic_generator.generate("Some text", "")
        assert result.num_matches == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
