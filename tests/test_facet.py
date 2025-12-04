"""
Unit tests for Faceted Search module.

Tests cover:
- Term facets (discrete values)
- Date range facets (bucketing)
- Filter conditions (single, multi-select, range)
- Filter combinations (AND logic)

Author: Information Retrieval System
"""

import pytest
from datetime import datetime
from src.ir.facet import (
    FacetEngine,
    FacetResult,
    FacetValue,
    FacetFilter,
    FilterCondition,
    RangeFilter,
    FilterOperator,
    create_term_filter,
    create_date_range_filter
)


# Sample documents for testing
SAMPLE_DOCS = [
    {
        "id": "1",
        "title": "台灣經濟成長超預期",
        "source": "CNA",
        "category": "finance",
        "pub_date": "2024-11-15",
        "author": "張三"
    },
    {
        "id": "2",
        "title": "總統出訪東南亞",
        "source": "UDN",
        "category": "politics",
        "pub_date": "2024-11-16",
        "author": "李四"
    },
    {
        "id": "3",
        "title": "央行宣布升息",
        "source": "CNA",
        "category": "finance",
        "pub_date": "2024-11-15",
        "author": "王五"
    },
    {
        "id": "4",
        "title": "立法院通過預算案",
        "source": "LTN",
        "category": "politics",
        "pub_date": "2024-11-14",
        "author": "趙六"
    },
    {
        "id": "5",
        "title": "科技股大漲",
        "source": "UDN",
        "category": "finance",
        "pub_date": "2024-11-17",
        "author": "張三"
    },
    {
        "id": "6",
        "title": "國際峰會召開",
        "source": "CNA",
        "category": "international",
        "pub_date": "2024-10-20",
        "author": "李四"
    },
]


class TestFacetEngine:
    """Test FacetEngine functionality."""

    def test_configure_facet(self):
        """Test facet configuration."""
        engine = FacetEngine()

        engine.configure_facet("source", "新聞來源", "term")
        engine.configure_facet("pub_date", "發布日期", "date_range", date_format="%Y-%m")

        assert "source" in engine.facet_configs
        assert "pub_date" in engine.facet_configs
        assert engine.facet_configs["source"]["facet_type"] == "term"
        assert engine.facet_configs["pub_date"]["facet_type"] == "date_range"

    def test_build_term_facet(self):
        """Test building term facets from documents."""
        engine = FacetEngine()
        engine.configure_facet("source", "新聞來源", "term")
        engine.configure_facet("category", "分類", "term")

        facets = engine.build_facets(SAMPLE_DOCS)

        # Check source facet
        source_facet = facets["source"]
        assert source_facet.field_name == "source"
        assert source_facet.facet_type == "term"
        assert source_facet.total_docs == 6

        # CNA appears 3 times, should be first
        assert source_facet.values[0].value == "CNA"
        assert source_facet.values[0].count == 3

        # UDN appears 2 times
        assert source_facet.values[1].value == "UDN"
        assert source_facet.values[1].count == 2

    def test_build_date_range_facet(self):
        """Test building date range facets with monthly bucketing."""
        engine = FacetEngine()
        engine.configure_facet("pub_date", "發布月份", "date_range", date_format="%Y-%m")

        facets = engine.build_facets(SAMPLE_DOCS)

        date_facet = facets["pub_date"]
        assert date_facet.field_name == "pub_date"
        assert date_facet.facet_type == "date_range"

        # Should have buckets for 2024-11 and 2024-10
        values_dict = {fv.value: fv.count for fv in date_facet.values}
        assert "2024-11" in values_dict
        assert "2024-10" in values_dict

        # 5 docs in 2024-11
        assert values_dict["2024-11"] == 5
        # 1 doc in 2024-10
        assert values_dict["2024-10"] == 1

    def test_facet_top_k(self):
        """Test getting top-k facet values."""
        engine = FacetEngine()
        engine.configure_facet("category", "分類", "term")

        facets = engine.build_facets(SAMPLE_DOCS)
        category_facet = facets["category"]

        # Get top 2
        top_2 = category_facet.get_top_k(2)
        assert len(top_2) == 2
        assert top_2[0].value == "finance"  # 3 docs
        assert top_2[0].count == 3

    def test_filter_documents_single_value(self):
        """Test filtering with single value condition."""
        engine = FacetEngine()

        filters = {"source": "CNA"}
        filtered = engine.filter_documents(SAMPLE_DOCS, filters)

        assert len(filtered) == 3
        assert all(doc["source"] == "CNA" for doc in filtered)

    def test_filter_documents_multi_select(self):
        """Test filtering with multi-select (list of values)."""
        engine = FacetEngine()

        filters = {"source": ["CNA", "UDN"]}
        filtered = engine.filter_documents(SAMPLE_DOCS, filters)

        assert len(filtered) == 5  # 3 CNA + 2 UDN
        assert all(doc["source"] in ["CNA", "UDN"] for doc in filtered)

    def test_filter_documents_range(self):
        """Test filtering with date range."""
        engine = FacetEngine()

        filters = {"pub_date": ("2024-11-15", "2024-11-17")}
        filtered = engine.filter_documents(SAMPLE_DOCS, filters)

        assert len(filtered) == 4
        for doc in filtered:
            assert "2024-11-15" <= doc["pub_date"] <= "2024-11-17"

    def test_filter_documents_combined(self):
        """Test filtering with multiple conditions (AND logic)."""
        engine = FacetEngine()

        filters = {
            "source": ["CNA", "UDN"],
            "category": "finance"
        }
        filtered = engine.filter_documents(SAMPLE_DOCS, filters)

        # Should get docs 1, 3, 5 (CNA/UDN + finance)
        assert len(filtered) == 3
        assert all(
            doc["source"] in ["CNA", "UDN"] and doc["category"] == "finance"
            for doc in filtered
        )


class TestFilterCondition:
    """Test FilterCondition functionality."""

    def test_equals_operator(self):
        """Test EQUALS operator."""
        condition = FilterCondition("source", FilterOperator.EQUALS, "CNA")

        assert condition.matches("CNA") is True
        assert condition.matches("UDN") is False
        assert condition.matches(None) is False

    def test_in_operator(self):
        """Test IN operator (multi-select)."""
        condition = FilterCondition("source", FilterOperator.IN, ["CNA", "UDN"])

        assert condition.matches("CNA") is True
        assert condition.matches("UDN") is True
        assert condition.matches("LTN") is False

    def test_range_operator(self):
        """Test RANGE operator."""
        condition = FilterCondition(
            "pub_date",
            FilterOperator.RANGE,
            ("2024-11-01", "2024-11-30")
        )

        assert condition.matches("2024-11-15") is True
        assert condition.matches("2024-11-01") is True
        assert condition.matches("2024-11-30") is True
        assert condition.matches("2024-10-31") is False
        assert condition.matches("2024-12-01") is False

    def test_greater_than_operator(self):
        """Test GREATER_THAN operator."""
        condition = FilterCondition("score", FilterOperator.GREATER_THAN, 0.5)

        assert condition.matches(0.6) is True
        assert condition.matches(0.5) is False
        assert condition.matches(0.4) is False

    def test_condition_to_dict(self):
        """Test serialization to dictionary."""
        condition = FilterCondition(
            "source",
            FilterOperator.IN,
            ["CNA", "UDN"],
            label="中央社 + 聯合"
        )

        data = condition.to_dict()
        assert data["field"] == "source"
        assert data["operator"] == "in"
        assert data["value"] == ["CNA", "UDN"]
        assert data["label"] == "中央社 + 聯合"

    def test_condition_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "field": "category",
            "operator": "equals",
            "value": "politics",
            "label": "政治"
        }

        condition = FilterCondition.from_dict(data)
        assert condition.field == "category"
        assert condition.operator == FilterOperator.EQUALS
        assert condition.value == "politics"
        assert condition.label == "政治"


class TestRangeFilter:
    """Test RangeFilter functionality."""

    def test_date_range_filter(self):
        """Test date range filter creation."""
        filter_obj = RangeFilter(
            "pub_date",
            "2024-11-01",
            "2024-11-30",
            label="2024年11月"
        )

        assert filter_obj.field == "pub_date"
        assert filter_obj.operator == FilterOperator.RANGE
        assert filter_obj.min_value == "2024-11-01"
        assert filter_obj.max_value == "2024-11-30"
        assert filter_obj.matches("2024-11-15") is True
        assert filter_obj.matches("2024-10-31") is False

    def test_numeric_range_filter(self):
        """Test numeric range filter."""
        filter_obj = RangeFilter("score", 0.5, 1.0, label="高分")

        assert filter_obj.matches(0.7) is True
        assert filter_obj.matches(0.5) is True
        assert filter_obj.matches(1.0) is True
        assert filter_obj.matches(0.4) is False


class TestFacetFilter:
    """Test FacetFilter (filter manager) functionality."""

    def test_add_condition(self):
        """Test adding filter conditions."""
        filter_mgr = FacetFilter()

        condition1 = FilterCondition("source", FilterOperator.EQUALS, "CNA")
        condition2 = FilterCondition("category", FilterOperator.IN, ["politics", "finance"])

        filter_mgr.add_condition(condition1)
        filter_mgr.add_condition(condition2)

        assert filter_mgr.get_filter_count() == 2
        assert filter_mgr.has_filter("source") is True
        assert filter_mgr.has_filter("category") is True

    def test_remove_condition(self):
        """Test removing filter conditions."""
        filter_mgr = FacetFilter()

        filter_mgr.add_condition(
            FilterCondition("source", FilterOperator.EQUALS, "CNA")
        )
        filter_mgr.add_condition(
            FilterCondition("category", FilterOperator.IN, ["politics"])
        )

        assert filter_mgr.get_filter_count() == 2

        filter_mgr.remove_condition("source")
        assert filter_mgr.get_filter_count() == 1
        assert filter_mgr.has_filter("source") is False
        assert filter_mgr.has_filter("category") is True

    def test_filter_single_condition(self):
        """Test filtering with single condition."""
        filter_mgr = FacetFilter()
        filter_mgr.add_condition(
            FilterCondition("source", FilterOperator.EQUALS, "CNA")
        )

        filtered = filter_mgr.filter(SAMPLE_DOCS)

        assert len(filtered) == 3
        assert all(doc["source"] == "CNA" for doc in filtered)

    def test_filter_multiple_conditions_and_logic(self):
        """Test filtering with multiple conditions (AND logic)."""
        filter_mgr = FacetFilter()

        # Source must be CNA OR UDN
        filter_mgr.add_condition(
            FilterCondition("source", FilterOperator.IN, ["CNA", "UDN"])
        )

        # AND category must be finance
        filter_mgr.add_condition(
            FilterCondition("category", FilterOperator.EQUALS, "finance")
        )

        filtered = filter_mgr.filter(SAMPLE_DOCS)

        # Should get docs 1, 3, 5
        assert len(filtered) == 3
        assert all(
            doc["source"] in ["CNA", "UDN"] and doc["category"] == "finance"
            for doc in filtered
        )

    def test_filter_with_range_condition(self):
        """Test filtering with range condition."""
        filter_mgr = FacetFilter()

        filter_mgr.add_condition(
            RangeFilter("pub_date", "2024-11-15", "2024-11-17")
        )

        filtered = filter_mgr.filter(SAMPLE_DOCS)

        assert len(filtered) == 4
        for doc in filtered:
            assert "2024-11-15" <= doc["pub_date"] <= "2024-11-17"

    def test_filter_complex_combination(self):
        """Test complex filter combination."""
        filter_mgr = FacetFilter()

        # Multi-source
        filter_mgr.add_condition(
            FilterCondition("source", FilterOperator.IN, ["CNA", "UDN"])
        )

        # Single category
        filter_mgr.add_condition(
            FilterCondition("category", FilterOperator.EQUALS, "finance")
        )

        # Date range
        filter_mgr.add_condition(
            RangeFilter("pub_date", "2024-11-15", "2024-11-20")
        )

        filtered = filter_mgr.filter(SAMPLE_DOCS)

        # Should get docs 1, 3, 5
        assert len(filtered) == 3
        for doc in filtered:
            assert doc["source"] in ["CNA", "UDN"]
            assert doc["category"] == "finance"
            assert "2024-11-15" <= doc["pub_date"] <= "2024-11-20"

    def test_clear_filters(self):
        """Test clearing all filters."""
        filter_mgr = FacetFilter()

        filter_mgr.add_condition(FilterCondition("source", FilterOperator.EQUALS, "CNA"))
        filter_mgr.add_condition(FilterCondition("category", FilterOperator.IN, ["politics"]))

        assert filter_mgr.get_filter_count() == 2

        filter_mgr.clear()
        assert filter_mgr.get_filter_count() == 0
        assert filter_mgr.has_filter("source") is False

    def test_to_dict_and_from_dict(self):
        """Test serialization and deserialization."""
        filter_mgr = FacetFilter()

        filter_mgr.add_condition(
            FilterCondition("source", FilterOperator.IN, ["CNA", "UDN"], label="中央社+聯合")
        )
        filter_mgr.add_condition(
            RangeFilter("pub_date", "2024-11-01", "2024-11-30", label="11月")
        )

        # Serialize
        data = filter_mgr.to_dict()
        assert data["count"] == 2
        assert len(data["conditions"]) == 2

        # Deserialize
        new_filter_mgr = FacetFilter.from_dict(data)
        assert new_filter_mgr.get_filter_count() == 2

        # Should filter the same way
        filtered1 = filter_mgr.filter(SAMPLE_DOCS)
        filtered2 = new_filter_mgr.filter(SAMPLE_DOCS)
        assert len(filtered1) == len(filtered2)


class TestConvenienceFunctions:
    """Test convenience functions for creating filters."""

    def test_create_term_filter_single(self):
        """Test creating single-value term filter."""
        filter_obj = create_term_filter("category", "politics", label="政治")

        assert filter_obj.field == "category"
        assert filter_obj.operator == FilterOperator.EQUALS
        assert filter_obj.value == "politics"
        assert filter_obj.label == "政治"

    def test_create_term_filter_multi(self):
        """Test creating multi-select term filter."""
        filter_obj = create_term_filter(
            "source",
            ["CNA", "UDN"],
            label="中央社 + 聯合"
        )

        assert filter_obj.field == "source"
        assert filter_obj.operator == FilterOperator.IN
        assert filter_obj.value == ["CNA", "UDN"]

    def test_create_date_range_filter(self):
        """Test creating date range filter."""
        filter_obj = create_date_range_filter(
            "pub_date",
            "2024-11-01",
            "2024-11-30",
            label="2024年11月"
        )

        assert isinstance(filter_obj, RangeFilter)
        assert filter_obj.field == "pub_date"
        assert filter_obj.min_value == "2024-11-01"
        assert filter_obj.max_value == "2024-11-30"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_documents(self):
        """Test handling empty document list."""
        engine = FacetEngine()
        engine.configure_facet("source", "來源", "term")

        facets = engine.build_facets([])
        assert facets == {}

    def test_missing_field_values(self):
        """Test handling documents with missing field values."""
        docs = [
            {"source": "CNA", "category": "politics"},
            {"category": "finance"},  # No source
            {"source": "UDN"},  # No category
        ]

        engine = FacetEngine()
        engine.configure_facet("source", "來源", "term")

        facets = engine.build_facets(docs)
        source_facet = facets["source"]

        # Should only count docs with source field
        assert source_facet.total_docs == 3
        values_dict = {fv.value: fv.count for fv in source_facet.values}
        assert values_dict["CNA"] == 1
        assert values_dict["UDN"] == 1

    def test_filter_with_no_matches(self):
        """Test filtering that returns no matches."""
        filter_mgr = FacetFilter()
        filter_mgr.add_condition(
            FilterCondition("source", FilterOperator.EQUALS, "NONEXISTENT")
        )

        filtered = filter_mgr.filter(SAMPLE_DOCS)
        assert len(filtered) == 0

    def test_filter_with_all_matches(self):
        """Test filtering that matches all documents."""
        filter_mgr = FacetFilter()
        # Empty filter should return all

        filtered = filter_mgr.filter(SAMPLE_DOCS)
        assert len(filtered) == len(SAMPLE_DOCS)
