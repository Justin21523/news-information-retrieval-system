"""Tests for app-level Chinese/news text quality helpers."""

from src.ir_app.services.text_quality import TextQualityService


def make_quality_service():
    """Build a lightweight text quality service for unit tests."""
    def normalize(text: str) -> str:
        return (text or "").replace("臺", "台").strip().lower()

    def tokenize(text: str) -> list[str]:
        return normalize(text).split()

    return TextQualityService(tokenize, normalize)


def test_text_quality_normalizes_and_tracks_stopwords():
    """News boilerplate terms are removed from quality signals."""
    quality = make_quality_service()

    analysis = quality.analysis("中央社 AI 臺灣 報導")

    assert analysis["normalized_query"] == "中央社 ai 台灣 報導"
    assert analysis["query_terms"] == ["中央社", "ai", "台灣", "報導"]
    assert analysis["removed_stopwords"] == ["中央社", "報導"]
    assert analysis["significant_terms"] == ["ai", "台灣"]


def test_text_quality_keeps_protected_news_entities():
    """Protected proper nouns are not removed as stopwords."""
    quality = make_quality_service()

    terms = ["ai", "台灣", "美國", "中國", "半導體"]

    assert quality.significant_terms(terms) == terms


def test_text_quality_adds_conservative_synonyms():
    """Known query variants are exposed as synonym terms."""
    quality = make_quality_service()

    analysis = quality.analysis("ai 台灣 半導體")

    assert "人工智慧" in analysis["synonym_terms"]
    assert "晶片" in analysis["synonym_terms"]


def test_text_quality_adds_phrase_synonyms_from_raw_query():
    """Phrase synonyms can be detected even if tokenizer splits the phrase."""
    quality = make_quality_service()

    analysis = quality.analysis("氣候變遷")

    assert "氣候變化" in analysis["synonym_terms"]
