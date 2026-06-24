"""Document loading and metadata normalization service."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

from src.ir_app.config import Settings
from src.ir_app.services.data_contract import (
    DatasetValidationStats,
    compute_dedup_hash,
    normalize_tags,
    validate_article,
)
from src.ir_app.services.taxonomy import normalize_taxonomy


class DocumentService:
    """Load news/demo documents and provide stable document lookup.

    Complexity:
        Time: O(n) for loading n documents
        Space: O(n)
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.dataset_source = "none"
        self.dataset_hash = ""
        self.dataset_mtime: float | None = None
        self.dataset_size: int | None = None
        self.dataset_limit = settings.max_documents
        self.validation_stats = DatasetValidationStats()
        self.documents = self._load_documents()
        self._by_doc_id = {str(doc["doc_id"]): doc for doc in self.documents}
        self._by_article_id = {
            str(doc["article_id"]): doc
            for doc in self.documents
            if doc.get("article_id") is not None
        }

    def _load_documents(self) -> list[dict[str, Any]]:
        """Load configured CNA JSONL or fallback mini JSON.

        Complexity:
            Time: O(n)
            Space: O(n)
        """
        if self.settings.dataset_path.exists():
            self.dataset_source = str(self.settings.dataset_path)
            return self._load_jsonl(self.settings.dataset_path)

        if self.settings.fallback_dataset_path.exists():
            self.dataset_source = str(self.settings.fallback_dataset_path)
            return self._load_json(self.settings.fallback_dataset_path)

        self.dataset_source = "empty"
        return []

    def _load_jsonl(self, path: Path) -> list[dict[str, Any]]:
        """Load JSONL documents.

        Complexity:
            Time: O(n)
            Space: O(n)
        """
        self._record_dataset_file(path)
        raw_docs: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw_docs.append(json.loads(line))
                if self.dataset_limit is not None and len(raw_docs) >= self.dataset_limit:
                    break
        return self._normalize_records(raw_docs)

    def _load_json(self, path: Path) -> list[dict[str, Any]]:
        """Load JSON array documents.

        Complexity:
            Time: O(n)
            Space: O(n)
        """
        self._record_dataset_file(path)
        with path.open("r", encoding="utf-8") as handle:
            raw_docs = json.load(handle)
        if self.dataset_limit is not None:
            raw_docs = raw_docs[: self.dataset_limit]
        return self._normalize_records(raw_docs)

    def _record_dataset_file(self, path: Path) -> None:
        """Record dataset file metadata for index invalidation.

        Complexity:
            Time: O(n) where n is file size
            Space: O(1)
        """
        stat = path.stat()
        self.dataset_mtime = stat.st_mtime
        self.dataset_size = stat.st_size

        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        self.dataset_hash = digest.hexdigest()

    def _normalize_records(self, raw_docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Validate, deduplicate, and normalize raw article records.

        Complexity:
            Time: O(n)
            Space: O(n)
        """
        docs: list[dict[str, Any]] = []
        seen_hashes: set[str] = set()
        self.validation_stats = DatasetValidationStats(total=len(raw_docs))

        for row, raw in enumerate(raw_docs):
            issues = validate_article(raw, row)
            for issue in issues:
                if issue.code == "MISSING_FIELD":
                    field_name = issue.message.rsplit(": ", 1)[-1]
                    self.validation_stats.add_missing(field_name)
            if issues:
                self.validation_stats.invalid += 1
                self.validation_stats.issues.extend(issues)
                continue

            dedup_hash = compute_dedup_hash(str(raw.get("title") or ""), str(raw.get("url") or ""))
            if dedup_hash in seen_hashes:
                self.validation_stats.duplicates += 1
                continue
            seen_hashes.add(dedup_hash)

            doc = self._normalize_document(raw, len(docs), dedup_hash)
            docs.append(doc)
            self.validation_stats.valid += 1

        return docs

    def _normalize_document(
        self,
        raw: dict[str, Any],
        fallback_id: int,
        dedup_hash: str | None = None,
    ) -> dict[str, Any]:
        """Normalize crawler/demo fields into a single document shape.

        Complexity:
            Time: O(t) where t is number of tags
            Space: O(t)
        """
        doc_id = fallback_id
        article_id = raw.get("article_id") or dedup_hash or str(fallback_id)
        content = (
            raw.get("content")
            or raw.get("content_clean")
            or raw.get("text")
            or raw.get("body")
            or ""
        )
        title = raw.get("title") or raw.get("title_clean") or f"Document {doc_id}"
        tags = normalize_tags(raw.get("tags"))

        published_date = (
            raw.get("published_date")
            or raw.get("publish_date")
            or raw.get("pub_date")
            or raw.get("date")
        )
        taxonomy = normalize_taxonomy(raw)

        normalized = {
            "doc_id": doc_id,
            "article_id": article_id,
            "title": str(title),
            "content": str(content),
            "url": raw.get("url"),
            "published_date": published_date,
            "category": raw.get("category"),
            "category_name": raw.get("category_name") or raw.get("category"),
            "source": raw.get("source") or taxonomy.source,
            "source_name": raw.get("source_name") or taxonomy.source_name,
            "source_label": raw.get("source_label") or taxonomy.source_label,
            "author": raw.get("author"),
            "tags": tags,
            "dedup_hash": dedup_hash or compute_dedup_hash(str(title), str(raw.get("url") or "")),
            "content_type": raw.get("content_type") or "news_article",
            "taxonomy_topic": raw.get("taxonomy_topic") or taxonomy.taxonomy_topic,
            "taxonomy_label": raw.get("taxonomy_label") or taxonomy.taxonomy_label,
            "taxonomy_path": raw.get("taxonomy_path") or taxonomy.taxonomy_path,
            "origin_path": raw.get("origin_path"),
        }
        normalized["text"] = normalized["content"]
        return normalized

    def get_document(self, doc_id: int | str) -> dict[str, Any] | None:
        """Get a document by numeric doc_id or article_id.

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        key = str(doc_id)
        return self._by_doc_id.get(key) or self._by_article_id.get(key)

    def stats(self) -> dict[str, Any]:
        """Return document collection statistics.

        Complexity:
            Time: O(n)
            Space: O(1)
        """
        categories = {doc.get("category") for doc in self.documents if doc.get("category")}
        sources = {doc.get("source") for doc in self.documents if doc.get("source")}
        return {
            "total_documents": len(self.documents),
            "categories": len(categories),
            "sources": len(sources),
            "dataset_source": self.dataset_source,
            "dataset_hash": self.dataset_hash,
            "dataset_mtime": self.dataset_mtime,
            "dataset_size": self.dataset_size,
            "dataset_limit": self.dataset_limit,
            "validation": self.validation_stats.to_dict(),
        }

    def to_api_document(self, doc: dict[str, Any]) -> dict[str, Any]:
        """Convert an internal document to API shape.

        Complexity:
            Time: O(t) where t is number of tags
            Space: O(t)
        """
        metadata = {
            "article_id": doc.get("article_id"),
            "url": doc.get("url"),
            "published_date": doc.get("published_date"),
            "date": doc.get("published_date"),
            "category": doc.get("category"),
            "category_name": doc.get("category_name"),
            "source": doc.get("source"),
            "source_name": doc.get("source_name"),
            "source_label": doc.get("source_label"),
            "author": doc.get("author"),
            "tags": doc.get("tags") or [],
            "dedup_hash": doc.get("dedup_hash"),
            "content_type": doc.get("content_type"),
            "taxonomy_topic": doc.get("taxonomy_topic"),
            "taxonomy_label": doc.get("taxonomy_label"),
            "taxonomy_path": doc.get("taxonomy_path"),
            "origin_path": doc.get("origin_path"),
            "content": doc.get("content") or "",
        }
        return {
            "doc_id": doc.get("doc_id"),
            "article_id": doc.get("article_id"),
            "title": doc.get("title") or "",
            "content": doc.get("content") or "",
            "metadata": metadata,
        }
