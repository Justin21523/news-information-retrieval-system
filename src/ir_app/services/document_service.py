"""Document loading and metadata normalization service."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.ir_app.config import Settings


class DocumentService:
    """Load news/demo documents and provide stable document lookup.

    Complexity:
        Time: O(n) for loading n documents
        Space: O(n)
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.dataset_source = "none"
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
        docs: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                if not line.strip():
                    continue
                docs.append(self._normalize_document(json.loads(line), idx))
        return docs

    def _load_json(self, path: Path) -> list[dict[str, Any]]:
        """Load JSON array documents.

        Complexity:
            Time: O(n)
            Space: O(n)
        """
        with path.open("r", encoding="utf-8") as handle:
            raw_docs = json.load(handle)
        return [self._normalize_document(doc, idx) for idx, doc in enumerate(raw_docs)]

    def _normalize_document(self, raw: dict[str, Any], fallback_id: int) -> dict[str, Any]:
        """Normalize crawler/demo fields into a single document shape.

        Complexity:
            Time: O(t) where t is number of tags
            Space: O(t)
        """
        doc_id = raw.get("doc_id", fallback_id)
        article_id = raw.get("article_id")
        content = raw.get("content") or raw.get("text") or raw.get("body") or ""
        title = raw.get("title") or f"Document {doc_id}"
        tags = raw.get("tags") or []
        if isinstance(tags, str):
            tags = [tag.strip() for tag in tags.split(",") if tag.strip()]

        published_date = (
            raw.get("published_date")
            or raw.get("publish_date")
            or raw.get("pub_date")
            or raw.get("date")
        )

        normalized = {
            "doc_id": doc_id,
            "article_id": article_id,
            "title": str(title),
            "content": str(content),
            "url": raw.get("url"),
            "published_date": published_date,
            "category": raw.get("category"),
            "category_name": raw.get("category_name") or raw.get("category"),
            "source": raw.get("source"),
            "source_name": raw.get("source_name"),
            "author": raw.get("author"),
            "tags": tags,
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
            "author": doc.get("author"),
            "tags": doc.get("tags") or [],
            "content": doc.get("content") or "",
        }
        return {
            "doc_id": doc.get("doc_id"),
            "article_id": doc.get("article_id"),
            "title": doc.get("title") or "",
            "content": doc.get("content") or "",
            "metadata": metadata,
        }
