"""Text quality helpers for Chinese news retrieval."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.ir.text.stopwords import StopwordsFilter


NEWS_LOW_INFORMATION_TERMS = {
    "中央社",
    "記者",
    "報導",
    "綜合報導",
    "編輯",
    "新聞稿",
    "快訊",
    "圖",
    "文",
    "指出",
    "表示",
    "今天",
    "昨天",
    "明天",
}


PROTECTED_TERMS = {
    "ai",
    "人工智慧",
    "台灣",
    "臺灣",
    "美國",
    "中國",
    "台積電",
    "半導體",
}


SYNONYM_MAP = {
    "ai": ["人工智慧"],
    "人工智慧": ["ai"],
    "台灣": ["臺灣"],
    "臺灣": ["台灣"],
    "半導體": ["晶片", "芯片"],
    "晶片": ["半導體"],
    "美國": ["美方"],
    "中國": ["大陸", "中方"],
    "氣候變遷": ["氣候變化"],
}


class TextQualityService:
    """Normalize and analyze query terms for the app retrieval layer.

    Complexity:
        Time: O(q) for query analysis
        Space: O(q)
    """

    def __init__(
        self,
        tokenizer: Callable[[str], list[str]],
        normalizer: Callable[[str], str],
    ):
        self.tokenizer = tokenizer
        self.normalizer = normalizer
        self.stopwords = StopwordsFilter(additional_stopwords=list(NEWS_LOW_INFORMATION_TERMS))
        self._protected_terms = {self.normalize(item) for item in PROTECTED_TERMS}

    def normalize(self, text: str) -> str:
        """Normalize user-visible text consistently with the index.

        Complexity:
            Time: O(n)
            Space: O(n)
        """
        return self.normalizer(text or "")

    def query_terms(self, query: str) -> list[str]:
        """Tokenize a query with normalized lexical processing.

        Complexity:
            Time: O(n)
            Space: O(q)
        """
        return self.tokenizer(query or "")

    def significant_terms(self, terms: list[str]) -> list[str]:
        """Return terms that should drive ranking and snippets.

        Complexity:
            Time: O(q)
            Space: O(q)
        """
        return [
            term for term in terms
            if term and not self.is_low_information(term)
        ]

    def removed_stopwords(self, terms: list[str]) -> list[str]:
        """Return low-information query terms removed from quality signals.

        Complexity:
            Time: O(q)
            Space: O(q)
        """
        return [
            term for term in terms
            if term and self.is_low_information(term)
        ]

    def synonym_terms(self, terms: list[str], raw_text: str = "") -> list[str]:
        """Return conservative synonym expansions for display and matching.

        Complexity:
            Time: O(q)
            Space: O(s)
        """
        expanded: list[str] = []
        normalized_originals = {self.normalize(term) for term in terms}
        for term in terms:
            for synonym in SYNONYM_MAP.get(term, []):
                if self.normalize(synonym) not in normalized_originals:
                    expanded.append(synonym)
        normalized_text = self.normalize(raw_text)
        for phrase, synonyms in SYNONYM_MAP.items():
            if phrase in terms:
                continue
            if self.normalize(phrase) and self.normalize(phrase) in normalized_text:
                for synonym in synonyms:
                    if self.normalize(synonym) not in normalized_originals:
                        expanded.append(synonym)
        return list(dict.fromkeys(expanded))

    def analysis(self, query: str, filters: dict[str, Any] | None = None) -> dict[str, Any]:
        """Build query analysis used by API responses.

        Complexity:
            Time: O(n + q)
            Space: O(q)
        """
        terms = self.query_terms(query)
        significant = self.significant_terms(terms)
        return {
            "raw_query": query or "",
            "normalized_query": self.normalize(query),
            "query_terms": terms,
            "term_count": len(terms),
            "significant_terms": significant,
            "removed_stopwords": self.removed_stopwords(terms),
            "synonym_terms": self.synonym_terms(significant, query),
            "filters_applied": filters or {},
        }

    def is_low_information(self, term: str) -> bool:
        """Return whether a term should be ignored for quality signals.

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        normalized = self.normalize(term)
        if normalized in self._protected_terms:
            return False
        return self.stopwords.is_stopword(normalized)

    def matching_terms(self, query_terms: list[str], raw_text: str = "") -> list[str]:
        """Return query terms plus conservative synonyms for matching.

        Complexity:
            Time: O(q)
            Space: O(q)
        """
        significant = self.significant_terms(query_terms)
        return list(dict.fromkeys(significant + self.synonym_terms(significant, raw_text)))
