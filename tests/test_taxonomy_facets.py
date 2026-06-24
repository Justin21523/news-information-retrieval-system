"""Tests for taxonomy normalization and indexed facets."""

from src.ir_app.services.facet_service import FacetService
from src.ir_app.services.taxonomy import classify_category, normalize_taxonomy


def test_taxonomy_maps_raw_categories_to_topics():
    """Common raw category aliases map to stable top-level topics."""
    assert classify_category("aipl", "政治") == ("politics", "politics")
    assert classify_category("財經", "finance") == ("business", "finance")
    assert classify_category("AI科技", "aitech") == ("tech", "ai")
    assert classify_category("unknown", "未分類") == ("other", "unknown")


def test_taxonomy_infers_yahoo_category_from_origin_path():
    """Yahoo raw files get useful topics even when record category is empty."""
    info = normalize_taxonomy(
        {"source": "yahoo", "source_name": "Yahoo", "category": None},
        "data/raw/yahoo_finance_14days.jsonl",
    )

    assert info.source == "Yahoo"
    assert info.source_label == "Yahoo 新聞"
    assert info.taxonomy_topic == "business"
    assert info.taxonomy_path == "news/business/finance"


def test_facet_service_filters_with_and_or_semantics():
    """Facet filters OR within a field and AND across fields."""
    docs = [
        {
            "doc_id": 0,
            "source": "LTN",
            "category": "財經",
            "taxonomy_topic": "business",
            "taxonomy_path": "news/business/finance",
            "content_type": "news_article",
            "published_date": "2026-01-05",
            "tags": ["台灣", "半導體"],
        },
        {
            "doc_id": 1,
            "source": "Yahoo",
            "category": "AI科技",
            "taxonomy_topic": "tech",
            "taxonomy_path": "news/tech/ai",
            "content_type": "news_article",
            "published_date": "2026-02-01",
            "tags": ["AI"],
        },
        {
            "doc_id": 2,
            "source": "CNA",
            "category": "政治",
            "taxonomy_topic": "politics",
            "taxonomy_path": "news/politics/politics",
            "content_type": "news_article",
            "published_date": "2025-12-12",
            "tags": ["台灣"],
        },
    ]
    service = FacetService(docs)

    assert service.matching_doc_ids({"source": ["LTN", "Yahoo"]}) == {"0", "1"}
    assert service.matching_doc_ids({"source": ["LTN", "Yahoo"], "taxonomy_topic": ["tech"]}) == {"1"}
    assert service.matching_doc_ids({"tags": ["台灣"], "published_year": ["2026"]}) == {"0"}

    facets = service.build_facets({"0", "1"}, {"taxonomy_topic": ["tech"]})
    assert facets["taxonomy_topic"]["field_type"] == "taxonomy"
    assert any(value["label"] == "科技 Tech" for value in facets["taxonomy_topic"]["values"])
    assert facets["taxonomy_topic"]["result_count"] == 2
