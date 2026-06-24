"""Facet indexing and metadata filtering for the IR app."""

from __future__ import annotations

from collections import Counter, defaultdict
import re
from typing import Any

from src.ir_app.services.taxonomy import SOURCE_LABELS, facet_label


FACET_CONFIG = {
    "source": {"display_name": "來源 Source", "field_type": "term", "order": 10},
    "taxonomy_topic": {"display_name": "主題 Taxonomy", "field_type": "taxonomy", "order": 20},
    "taxonomy_path": {"display_name": "分類路徑 Taxonomy Path", "field_type": "taxonomy", "order": 30},
    "published_year": {"display_name": "年份 Year", "field_type": "date", "order": 40},
    "published_month": {"display_name": "月份 Month", "field_type": "date", "order": 50},
    "tags": {"display_name": "標籤 Tags", "field_type": "multi_term", "order": 60},
    "source_name": {"display_name": "出版者 Publisher", "field_type": "term", "order": 70},
    "content_type": {"display_name": "內容類型 Content Type", "field_type": "term", "order": 80},
    "category": {"display_name": "原始分類 Raw Category", "field_type": "term", "order": 90},
    "category_name": {"display_name": "原始分類名稱 Raw Category Name", "field_type": "term", "order": 100},
    "author": {"display_name": "作者 Author", "field_type": "term", "order": 110},
}

NOISY_VALUES = {"", "unknown", "none", "null", "missing", "未分類", "其他", "nan"}
DATE_RE = re.compile(r"^(19|20)\d{2}-\d{2}-\d{2}")
PUBLISHER_HINTS = (
    "新聞",
    "通訊社",
    "中央社",
    "時報",
    "日報",
    "週刊",
    "電視",
    "華視",
    "民視",
    "東森",
    "三立",
    "公視",
    "Yahoo",
    "FTNN",
)


class FacetService:
    """Build facet counts and filter document IDs.

    Complexity:
        Time: O(n * f) initialization for n documents and f fields
        Space: O(v + p) for values and postings
    """

    def __init__(self, documents: list[dict[str, Any]]):
        self.documents = documents
        self.documents_by_id = {str(doc.get("doc_id")): doc for doc in documents}
        self.index: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        self.all_doc_ids = {str(doc.get("doc_id")) for doc in documents}
        self.quality = self._field_quality()
        self._build_index()

    def _build_index(self) -> None:
        """Build facet postings.

        Complexity:
            Time: O(n * f)
            Space: O(v + p)
        """
        for doc in self.documents:
            doc_id = str(doc.get("doc_id"))
            for field in FACET_CONFIG:
                for value in self.values_for_doc(doc, field):
                    self.index[field][value].add(doc_id)

    def values_for_doc(self, doc: dict[str, Any], field: str) -> list[str]:
        """Return normalized facet values for one document field.

        Complexity:
            Time: O(t) for tags, otherwise O(1)
            Space: O(t)
        """
        if field == "tags":
            return self._clean_multi_values(doc.get("tags") or [])
        if field == "published_year":
            date = self._valid_date(doc.get("published_date"))
            return [date[:4]] if date else []
        if field == "published_month":
            date = self._valid_date(doc.get("published_date"))
            return [date[:7]] if date else []
        if field == "author":
            return self._author_values(doc)
        value = doc.get(field)
        value = self._clean_value(value)
        if value is None:
            return []
        return [value]

    def matching_doc_ids(self, filters: dict[str, Any] | None) -> set[str]:
        """Return documents matching facet filters.

        Filters use OR within one field and AND across fields.

        Complexity:
            Time: O(f * v + r)
            Space: O(r)
        """
        if not filters:
            return set(self.all_doc_ids)
        current: set[str] | None = None
        for field, raw_values in filters.items():
            if field not in FACET_CONFIG:
                continue
            values = raw_values if isinstance(raw_values, list) else [raw_values]
            values = [str(value) for value in values if value is not None and str(value) != ""]
            if not values:
                continue
            field_matches: set[str] = set()
            for value in values:
                field_matches.update(self.index.get(field, {}).get(value, set()))
            current = field_matches if current is None else current.intersection(field_matches)
        return current if current is not None else set(self.all_doc_ids)

    def build_facets(
        self,
        doc_ids: set[str] | list[str] | None = None,
        selected_filters: dict[str, Any] | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Build facet counts for a document ID set.

        Complexity:
            Time: O(f * v)
            Space: O(v)
        """
        allowed = set(doc_ids) if doc_ids is not None else set(self.all_doc_ids)
        selected_filters = selected_filters or {}
        facets: dict[str, Any] = {}
        for field, config in sorted(FACET_CONFIG.items(), key=lambda item: item[1]["order"]):
            values = []
            for value, postings in self.index.get(field, {}).items():
                count = len(postings.intersection(allowed))
                if count <= 0:
                    continue
                selected_values = selected_filters.get(field) or []
                if not isinstance(selected_values, list):
                    selected_values = [selected_values]
                values.append({
                    "value": value,
                    "label": facet_label(field, value),
                    "count": count,
                    "selected": value in {str(item) for item in selected_values},
                })
            values.sort(key=lambda item: (self._is_low_quality_value(item["value"]), -item["count"], item["label"]))
            if values:
                quality = self.quality.get(field, {})
                facets[field] = {
                    "display_name": config["display_name"],
                    "field_type": config["field_type"],
                    "total_count": sum(item["count"] for item in values),
                    "result_count": len(allowed),
                    "coverage": quality.get("coverage", 0.0),
                    "quality": {
                        **quality,
                        "is_low_information": len(values) <= 1,
                        "collapsed_by_default": self._collapsed_by_default(field, quality, values),
                    },
                    "values": values[:limit],
                }
        return facets

    def corpus_distribution(self) -> dict[str, list[dict[str, Any]]]:
        """Return compact corpus-level distributions.

        Complexity:
            Time: O(n)
            Space: O(v)
        """
        distributions: dict[str, list[dict[str, Any]]] = {}
        for field in ("source", "taxonomy_topic", "content_type"):
            counts: Counter[str] = Counter()
            for doc in self.documents:
                for value in self.values_for_doc(doc, field):
                    counts[value] += 1
            distributions[field] = [
                {"value": value, "label": facet_label(field, value), "count": count}
                for value, count in counts.most_common()
            ]
        return distributions

    def _field_quality(self) -> dict[str, dict[str, Any]]:
        """Return usable metadata coverage for facet fields.

        Complexity:
            Time: O(n * f)
            Space: O(f)
        """
        total = len(self.documents)
        quality: dict[str, dict[str, Any]] = {}
        for field in FACET_CONFIG:
            usable_docs = 0
            total_values = 0
            for doc in self.documents:
                values = self.values_for_doc(doc, field)
                if values:
                    usable_docs += 1
                    total_values += len(values)
            coverage = usable_docs / total if total else 0.0
            quality[field] = {
                "usable_documents": usable_docs,
                "missing_or_hidden_documents": max(total - usable_docs, 0),
                "coverage": coverage,
                "average_values_per_document": total_values / total if total else 0.0,
            }
        return quality

    def _clean_multi_values(self, values: list[Any]) -> list[str]:
        """Clean multi-value facet fields.

        Complexity:
            Time: O(v)
            Space: O(v)
        """
        cleaned = []
        for value in values:
            item = self._clean_value(value)
            if item:
                cleaned.append(item)
        return sorted(set(cleaned))

    def _clean_value(self, value: Any) -> str | None:
        """Normalize one displayable facet value.

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        text = str(value or "").strip()
        if not text or text.lower() in NOISY_VALUES or text in NOISY_VALUES:
            return None
        return text

    def _valid_date(self, value: Any) -> str | None:
        """Return a valid YYYY-MM-DD date prefix for facet use.

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        text = str(value or "").strip()
        if not DATE_RE.match(text):
            return None
        year = int(text[:4])
        if year < 1990 or year > 2030:
            return None
        return text[:10]

    def _author_values(self, doc: dict[str, Any]) -> list[str]:
        """Return author facet values only when they are not publisher aliases.

        Complexity:
            Time: O(s)
            Space: O(s)
        """
        author = self._clean_value(doc.get("author"))
        if not author:
            return []
        publisher_aliases = {
            self._normalize_alias(doc.get("source")),
            self._normalize_alias(doc.get("source_name")),
            self._normalize_alias(doc.get("source_label")),
            *{self._normalize_alias(value) for value in SOURCE_LABELS.values()},
        }
        if self._normalize_alias(author) in publisher_aliases:
            return []
        if self._looks_like_publisher(author):
            return []
        return [author]

    def _normalize_alias(self, value: Any) -> str:
        """Normalize publisher aliases for author reliability checks.

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        return str(value or "").strip().lower().replace(" ", "")

    def _looks_like_publisher(self, value: str) -> bool:
        """Return True when an author value appears to be a publisher name.

        Complexity:
            Time: O(h)
            Space: O(1)
        """
        return any(hint.lower() in value.lower() for hint in PUBLISHER_HINTS)

    def _is_low_quality_value(self, value: Any) -> bool:
        """Return True when a value should be sorted after useful values.

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        text = str(value or "").strip()
        return text.lower() in NOISY_VALUES or text in {"other", "news/other/unknown"}

    def _collapsed_by_default(
        self, field: str, quality: dict[str, Any], values: list[dict[str, Any]]
    ) -> bool:
        """Return whether the UI should collapse a lower-priority facet.

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        if field in {"source", "taxonomy_topic", "published_year", "tags"}:
            return False
        if len(values) <= 1:
            return True
        return float(quality.get("coverage", 0.0)) < 0.15
