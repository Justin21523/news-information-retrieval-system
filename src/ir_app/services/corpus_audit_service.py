"""Corpus audit and demo readiness diagnostics."""

from __future__ import annotations

import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

from src.ir_app.services.document_service import DocumentService
from src.ir_app.services.search_service import SearchService
from src.ir_app.services.taxonomy import facet_label


AUDIT_FIELDS = [
    "title",
    "content",
    "url",
    "published_date",
    "category",
    "category_name",
    "source",
    "source_label",
    "content_type",
    "taxonomy_topic",
    "taxonomy_label",
    "taxonomy_path",
    "tags",
]


class CorpusAuditService:
    """Summarize corpus quality, facets, and index readiness.

    Complexity:
        Time: O(n * f) for n documents and f audited fields
        Space: O(v) for distribution values
    """

    def __init__(self, document_service: DocumentService, search_service: SearchService):
        self.document_service = document_service
        self.search_service = search_service

    def audit(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """Build a corpus audit payload.

        Complexity:
            Time: O(n * f)
            Space: O(v)
        """
        started = time.perf_counter()
        documents = self.document_service.documents
        total = len(documents)
        missing = self._missing_fields(documents)
        content_lengths = [len(str(doc.get("content") or "")) for doc in documents]
        token_lengths = [
            int(length) for length in self.search_service.index.doc_lengths.values()
        ]

        data = {
            "summary": {
                "total_documents": total,
                "dataset_source": self.document_service.dataset_source,
                "dataset_hash": self.document_service.dataset_hash,
                "dataset_size": self.document_service.dataset_size,
                "dataset_mtime": self.document_service.dataset_mtime,
                "dataset_limit": self.document_service.dataset_limit,
                "validation": self.document_service.validation_stats.to_dict(),
            },
            "metadata_completeness": self._completeness(missing, total),
            "facet_quality": self._facet_quality(),
            "distributions": {
                "source": self._distribution(documents, "source"),
                "source_label": self._distribution(documents, "source_label"),
                "content_type": self._distribution(documents, "content_type"),
                "taxonomy_topic": self._distribution(documents, "taxonomy_topic"),
                "taxonomy_path": self._distribution(documents, "taxonomy_path", limit=20),
                "category": self._distribution(documents, "category", limit=20),
                "published_year": self._year_distribution(documents),
            },
            "lengths": {
                "content_chars": self._numeric_stats(content_lengths),
                "tokens": self._numeric_stats(token_lengths),
            },
            "deduplication": self._dedup_stats(documents),
            "index": {
                **self.search_service.index.stats(),
                "index_dir": str(self.search_service.settings.index_dir),
                "cache_files": self._cache_files(self.search_service.settings.index_dir),
            },
            "readiness": self._readiness(total, missing),
            "external_sources": self._external_source_notes(),
            "flow": self._flow_steps(),
        }
        return data, {"execution_time": time.perf_counter() - started}

    def _missing_fields(self, documents: list[dict[str, Any]]) -> dict[str, int]:
        """Count missing values for audited fields.

        Complexity:
            Time: O(n * f)
            Space: O(f)
        """
        missing = {field: 0 for field in AUDIT_FIELDS}
        for doc in documents:
            for field in AUDIT_FIELDS:
                value = doc.get(field)
                if value is None or value == "" or value == []:
                    missing[field] += 1
        return missing

    def _completeness(self, missing: dict[str, int], total: int) -> dict[str, Any]:
        """Return per-field completeness ratios.

        Complexity:
            Time: O(f)
            Space: O(f)
        """
        fields = []
        for field in AUDIT_FIELDS:
            missing_count = missing.get(field, 0)
            present = max(total - missing_count, 0)
            ratio = (present / total) if total else 0.0
            fields.append(
                {
                    "field": field,
                    "present": present,
                    "missing": missing_count,
                    "coverage": ratio,
                    "status": self._coverage_status(ratio),
                }
            )
        return {"fields": fields, "required_fields": ["title", "content"]}

    def _coverage_status(self, ratio: float) -> str:
        """Map coverage ratio to a display status.

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        if ratio >= 0.95:
            return "good"
        if ratio >= 0.75:
            return "watch"
        return "needs_work"

    def _distribution(
        self, documents: list[dict[str, Any]], field: str, limit: int = 12
    ) -> list[dict[str, Any]]:
        """Count field values with display labels.

        Complexity:
            Time: O(n)
            Space: O(v)
        """
        counts: Counter[str] = Counter()
        for doc in documents:
            value = doc.get(field)
            if value is None or value == "":
                value = "missing"
            counts[str(value)] += 1
        return [
            {
                "value": value,
                "label": facet_label(field, value),
                "count": count,
                "ratio": count / len(documents) if documents else 0.0,
            }
            for value, count in counts.most_common(limit)
        ]

    def _facet_quality(self) -> dict[str, Any]:
        """Return facet-ready metadata quality from the search facet index.

        Complexity:
            Time: O(f)
            Space: O(f)
        """
        facet_service = getattr(self.search_service, "facet_service", None)
        quality = getattr(facet_service, "quality", {}) or {}
        fields = []
        for field, values in quality.items():
            fields.append(
                {
                    "field": field,
                    "coverage": values.get("coverage", 0.0),
                    "usable_documents": values.get("usable_documents", 0),
                    "missing_or_hidden_documents": values.get(
                        "missing_or_hidden_documents", 0
                    ),
                    "average_values_per_document": values.get(
                        "average_values_per_document", 0.0
                    ),
                    "status": self._coverage_status(values.get("coverage", 0.0)),
                }
            )
        fields.sort(key=lambda item: (-item["coverage"], item["field"]))
        return {
            "fields": fields,
            "notes": [
                "Raw unknown categories and invalid dates are hidden from search facets.",
                "Author facets exclude publisher names when author metadata is not reliable.",
            ],
        }

    def _year_distribution(self, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Count documents by published year.

        Complexity:
            Time: O(n)
            Space: O(y)
        """
        counts: Counter[str] = Counter()
        for doc in documents:
            date = str(doc.get("published_date") or "")
            year = date[:4] if len(date) >= 4 and date[:4].isdigit() else "missing"
            counts[year] += 1
        return [
            {"value": year, "label": year, "count": count, "ratio": count / len(documents)}
            for year, count in sorted(counts.items(), key=lambda item: item[0])
        ] if documents else []

    def _numeric_stats(self, values: list[int]) -> dict[str, float | int]:
        """Return compact numeric distribution stats.

        Complexity:
            Time: O(n log n)
            Space: O(n)
        """
        if not values:
            return {"min": 0, "max": 0, "mean": 0.0, "median": 0.0}
        return {
            "min": min(values),
            "max": max(values),
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
        }

    def _dedup_stats(self, documents: list[dict[str, Any]]) -> dict[str, Any]:
        """Report duplicate hash status after load-time deduplication.

        Complexity:
            Time: O(n)
            Space: O(d)
        """
        hashes = [str(doc.get("dedup_hash") or "") for doc in documents if doc.get("dedup_hash")]
        unique = len(set(hashes))
        return {
            "dedup_hashes": len(hashes),
            "unique_hashes": unique,
            "duplicate_hashes_after_load": max(len(hashes) - unique, 0),
            "loader_duplicates_removed": self.document_service.validation_stats.duplicates,
        }

    def _cache_files(self, index_dir: Path) -> dict[str, Any]:
        """Return index cache file presence.

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        lexical_dir = Path(index_dir) / "lexical"
        files = {
            "manifest": lexical_dir / "manifest.json",
            "cache": lexical_dir / "lexical_cache.pkl",
        }
        return {
            name: {
                "path": str(path),
                "exists": path.exists(),
                "size": path.stat().st_size if path.exists() else 0,
            }
            for name, path in files.items()
        }

    def _readiness(self, total: int, missing: dict[str, int]) -> dict[str, Any]:
        """Summarize portfolio demo readiness and remaining gaps.

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        tags_coverage = 1.0 - (missing.get("tags", 0) / total if total else 1.0)
        category_coverage = 1.0 - (
            missing.get("taxonomy_topic", 0) / total if total else 1.0
        )
        date_coverage = 1.0 - (
            missing.get("published_date", 0) / total if total else 1.0
        )
        checks = [
            {
                "id": "large_corpus",
                "label": "Large corpus",
                "passed": total >= 10000,
                "detail": f"{total:,} searchable documents",
            },
            {
                "id": "taxonomy_facets",
                "label": "Taxonomy facets",
                "passed": category_coverage >= 0.95,
                "detail": f"{category_coverage:.1%} taxonomy coverage",
            },
            {
                "id": "date_facets",
                "label": "Date facets",
                "passed": date_coverage >= 0.75,
                "detail": f"{date_coverage:.1%} published date coverage",
            },
            {
                "id": "tags",
                "label": "Tags",
                "passed": tags_coverage >= 0.5,
                "detail": f"{tags_coverage:.1%} tag coverage",
            },
            {
                "id": "index_cache",
                "label": "Index cache",
                "passed": bool(self.search_service.index.stats().get("manifest")),
                "detail": "Lexical index manifest is available",
            },
        ]
        return {
            "overall_status": "demo_ready_with_metadata_gaps"
            if checks[0]["passed"]
            else "needs_more_documents",
            "checks": checks,
            "known_gaps": [
                "Tags are sparse in the active news corpus.",
                "Some source records still lack published_date.",
                "LDA and BERTopic remain optional heavy topic modeling features.",
            ],
        }

    def _external_source_notes(self) -> list[dict[str, Any]]:
        """Describe adjacent projects used for data and architecture reference.

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        return [
            {
                "project": "news-web-crawler",
                "path": "/home/justin/web-projects/news-web-crawler",
                "role": "Schema, validation, cleaning, and crawler pipeline reference.",
                "import_policy": "Use article schema/validator rules; import data only when concrete JSONL/DB exports exist.",
            },
            {
                "project": "dcard-trending-crawler",
                "path": "/home/justin/web-projects/dcard-trending-crawler",
                "role": "Social/trend corpus and analytics architecture reference.",
                "import_policy": "Keep as separate social/trend content_type facet, not mixed into default news search.",
            },
        ]

    def _flow_steps(self) -> list[dict[str, str]]:
        """Return the demo system flow shown in the corpus dashboard.

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        return [
            {"id": "crawl", "label": "Crawler outputs", "detail": "CNA, Yahoo, LTN, SETN, UDN, PTS, NextApple"},
            {"id": "validate", "label": "Validation + dedup", "detail": "Required fields, content length, dedup hash"},
            {"id": "taxonomy", "label": "Taxonomy + facets", "detail": "source, content_type, category, taxonomy_path"},
            {"id": "tokenize", "label": "Chinese NLP", "detail": "normalization, Jieba fallback, stopwords"},
            {"id": "index", "label": "Indexes", "detail": "inverted, positional, TF-IDF, BM25, LM cache"},
            {"id": "retrieve", "label": "Retrieval", "detail": "Boolean, BM25, TF-IDF, Hybrid, LM, fuzzy"},
            {"id": "explain", "label": "Explain + evaluate", "detail": "snippets, ranking diagnostics, qrels, feedback, LTR"},
        ]
