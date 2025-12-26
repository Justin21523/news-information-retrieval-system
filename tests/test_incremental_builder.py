"""
Unit Tests for IncrementalIndexBuilder

These tests focus on batch indexing behavior that should be deterministic even
without CKIP models installed (we stub tokenizers via monkeypatch).

Author: Information Retrieval System
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ir.index.doc_reader import NewsDocument
from src.ir.index.incremental_builder import IncrementalIndexBuilder


class _DummyTokenizer:
    """Simple whitespace tokenizer stub for unit tests."""

    def tokenize(self, text: str, filter_stopwords: bool = True, min_length: int = 2):
        """Tokenize by whitespace and apply a minimal length filter (test stub)."""
        return [t for t in text.lower().split() if len(t) >= min_length]


class _DummyOptimizedTokenizer:
    """Batch tokenizer stub returning whitespace-split tokens."""

    def tokenize_batch(self, texts, batch_size: int = 512, filter_stopwords: bool = True, min_length: int = 2):
        """Tokenize a batch of texts with the same stub logic (test helper)."""
        return [[t for t in text.lower().split() if len(t) >= min_length] for text in texts]


@pytest.mark.unit
@pytest.mark.index
class TestIncrementalIndexBuilderBatch:
    """Unit tests for IncrementalIndexBuilder.add_documents_batch()."""

    def test_add_documents_batch_preserves_input_order(self, tmp_path, monkeypatch):
        """Batch results align with input docs and doc_id assignment is stable."""
        import src.ir.index.incremental_builder as incremental_builder_module
        import src.ir.text.ckip_tokenizer_optimized as ckip_opt_module

        monkeypatch.setattr(
            incremental_builder_module,
            "get_tokenizer",
            lambda *args, **kwargs: _DummyTokenizer(),
        )
        monkeypatch.setattr(
            ckip_opt_module,
            "get_optimized_tokenizer",
            lambda *args, **kwargs: _DummyOptimizedTokenizer(),
        )

        builder = IncrementalIndexBuilder(
            index_dir=str(tmp_path / "index"),
            checkpoint_interval=1000,  # avoid checkpoint writes in this test
            use_dedup=True,
            fuzzy_threshold=3,
            ckip_model="bert-base",
        )

        doc_a = NewsDocument(
            title="DocA",
            content="hello world",
            url="u1",
            published_at="2025-11-10",
            source="CNA",
            category="politics",
            author="a",
        )
        doc_a_dup = NewsDocument(
            title="DocA",
            content="hello world",
            url="u2",
            published_at="2025-11-10",
            source="CNA",
            category="politics",
            author="a",
        )
        doc_b = NewsDocument(
            title="DocB",
            content="goodbye world",
            url="u3",
            published_at="2025-11-11",
            source="CNA",
            category="economy",
            author="b",
        )

        results = builder.add_documents_batch([doc_a, doc_a_dup, doc_b], ckip_batch_size=512)

        assert len(results) == 3
        assert results[0][0] is True
        assert results[1][0] is False and "Duplicate" in results[1][1]
        assert results[2][0] is True

        assert doc_a.doc_id == 0
        assert doc_a_dup.doc_id is None
        assert doc_b.doc_id == 1

        assert builder.docs_processed == 3
        assert builder.docs_indexed == 2
        assert builder.docs_duplicates == 1
        assert builder.index.doc_count == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
