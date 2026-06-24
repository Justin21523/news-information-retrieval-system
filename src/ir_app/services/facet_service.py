"""Facet indexing and metadata filtering for the IR app."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from src.ir_app.services.taxonomy import facet_label


FACET_CONFIG = {
    "source": {"display_name": "來源 Source", "field_type": "term"},
    "source_name": {"display_name": "來源名稱 Source Name", "field_type": "term"},
    "content_type": {"display_name": "內容類型 Content Type", "field_type": "term"},
    "taxonomy_topic": {"display_name": "主題 Taxonomy", "field_type": "taxonomy"},
    "taxonomy_path": {"display_name": "分類路徑 Taxonomy Path", "field_type": "taxonomy"},
    "category": {"display_name": "原始分類 Raw Category", "field_type": "term"},
    "category_name": {"display_name": "原始分類名稱 Raw Category Name", "field_type": "term"},
    "published_year": {"display_name": "年份 Year", "field_type": "date"},
    "published_month": {"display_name": "月份 Month", "field_type": "date"},
    "author": {"display_name": "作者 Author", "field_type": "term"},
    "tags": {"display_name": "標籤 Tags", "field_type": "multi_term"},
}


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
            return [str(tag) for tag in (doc.get("tags") or []) if str(tag).strip()]
        if field == "published_year":
            date = str(doc.get("published_date") or "")
            return [date[:4]] if len(date) >= 4 and date[:4].isdigit() else []
        if field == "published_month":
            date = str(doc.get("published_date") or "")
            return [date[:7]] if len(date) >= 7 else []
        value = doc.get(field)
        if value is None or value == "":
            return []
        return [str(value)]

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
        for field, config in FACET_CONFIG.items():
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
            values.sort(key=lambda item: (-item["count"], item["label"]))
            if values:
                facets[field] = {
                    "display_name": config["display_name"],
                    "field_type": config["field_type"],
                    "total_count": sum(item["count"] for item in values),
                    "result_count": len(allowed),
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
