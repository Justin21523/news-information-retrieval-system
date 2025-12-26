"""
Unit Tests for FieldIndexer

Author: Information Retrieval System
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ir.index.field_indexer import FieldIndexer


@pytest.mark.unit
@pytest.mark.index
class TestFieldIndexerDocId:
    """Unit tests for FieldIndexer doc_id handling."""

    def test_build_respects_explicit_doc_id(self):
        """Indexing uses explicit doc_id values when provided."""
        docs = [
            {"doc_id": 1, "category": "economy", "title": "台灣 經濟"},
            {"doc_id": 0, "category": "politics", "title": "台灣 政治"},
        ]
        indexer = FieldIndexer()
        indexer.build(docs)

        assert indexer.doc_count == 2
        assert indexer.search_field("category", "economy") == {1}
        assert indexer.search_field("category", "politics") == {0}

    def test_build_defaults_to_enumeration_when_no_doc_id(self):
        """Indexing falls back to enumeration order when doc_id is missing."""
        docs = [
            {"category": "politics", "title": "台灣 政治"},
            {"category": "economy", "title": "台灣 經濟"},
        ]
        indexer = FieldIndexer()
        indexer.build(docs)

        assert indexer.doc_count == 2
        assert indexer.search_field("category", "politics") == {0}
        assert indexer.search_field("category", "economy") == {1}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
