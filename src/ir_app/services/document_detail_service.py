"""Document detail enrichment for the Flask IR application."""

from __future__ import annotations

import html
import re
import time
from typing import Any

from src.ir.summarize.dynamic import KWICGenerator
from src.ir_app.services.document_service import DocumentService
from src.ir_app.services.search_service import SearchService


class DocumentDetailService:
    """Build enriched document detail payloads for the UI.

    Complexity:
        Time: O(n + r) per document detail request
        Space: O(k + r)
    """

    def __init__(
        self, document_service: DocumentService, search_service: SearchService
    ):
        self.document_service = document_service
        self.search_service = search_service
        self.kwic_generator = KWICGenerator(
            window_size=60, window_type="fixed", enable_cache=True
        )

    def build_detail(
        self,
        doc: dict[str, Any],
        query: str = "",
        top_k: int = 5,
        include_related: bool = True,
        include_kwic: bool | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Build an enriched document detail response.

        Complexity:
            Time: O(n + r log r)
            Space: O(k + r)
        """
        started = time.perf_counter()
        top_k = max(1, min(int(top_k or 5), 20))
        query = (query or "").strip()
        should_include_kwic = bool(query) if include_kwic is None else include_kwic

        api_doc = self.document_service.to_api_document(doc)
        summary = self.summary(doc)
        keywords = self.keywords(doc)
        kwic = (
            self.kwic(doc, query)
            if should_include_kwic
            else self.empty_kwic(query, "query_required")
        )
        related = (
            self.search_service.related_documents(doc, top_k=top_k, model="hybrid")
            if include_related
            else []
        )
        taxonomy = self.taxonomy(doc)
        topic = self.topic(doc)
        explanation = self.explanation(doc, query, summary, keywords, kwic, related)

        data = {
            "document": api_doc,
            "summary": summary,
            "keywords": keywords,
            "kwic": kwic,
            "related_documents": related,
            "similar_documents": related,
            "taxonomy": taxonomy,
            "topic": topic,
            "explanation": explanation,
        }
        meta = {
            "execution_time": time.perf_counter() - started,
            "query": query,
            "include_related": include_related,
            "include_kwic": should_include_kwic,
            "top_k": top_k,
        }
        return data, meta

    def summary(
        self, doc: dict[str, Any], method: str = "key_sentence", k: int = 3
    ) -> dict[str, Any]:
        """Return structured lightweight summary data.

        Complexity:
            Time: O(n)
            Space: O(s)
        """
        text = self.search_service.summarize(doc, method, k)
        sentences = [
            part.strip() for part in re.split(r"\n+", text or "") if part.strip()
        ]
        return {
            "available": bool(text),
            "method": method,
            "text": text,
            "sentences": sentences,
            "sentence_count": len(sentences),
        }

    def keywords(
        self, doc: dict[str, Any], method: str = "tfidf", top_k: int = 12
    ) -> dict[str, Any]:
        """Return structured keyword data.

        Complexity:
            Time: O(t log t)
            Space: O(k)
        """
        raw_keywords = self.search_service.extract_keywords(doc, method, top_k)
        items = [
            {
                "term": item.get("word") or item.get("term") or "",
                "word": item.get("word") or item.get("term") or "",
                "score": float(item.get("score") or 0.0),
                "frequency": int(item.get("frequency") or 0),
                "method": method,
            }
            for item in raw_keywords
        ]
        return {
            "available": True,
            "method": method,
            "items": items,
            "keywords": items,
        }

    def kwic(
        self, doc: dict[str, Any], query: str, max_matches: int = 5
    ) -> dict[str, Any]:
        """Return KWIC matches for the current query.

        Complexity:
            Time: O(n * q)
            Space: O(k)
        """
        if not query:
            return self.empty_kwic(query, "query_required")

        content = doc.get("content") or doc.get("text") or ""
        result = self.kwic_generator.generate(
            content,
            query,
            max_matches=max_matches,
            doc_id=int(doc.get("doc_id", 0) or 0),
        )
        matches = [
            {
                "keyword": match.keyword,
                "position": match.position,
                "left_context": match.left_context,
                "right_context": match.right_context,
                "plain_snippet": match.plain_snippet,
                "highlighted_snippet": self._highlight_kwic(
                    match.left_context, match.keyword, match.right_context
                ),
            }
            for match in result.matches
        ]
        return {
            "available": True,
            "query": query,
            "matches": matches,
            "match_count": len(matches),
            "cache_hit": result.cache_hit,
        }

    def empty_kwic(self, query: str, reason: str) -> dict[str, Any]:
        """Return a stable empty KWIC payload.

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        return {
            "available": False,
            "query": query or "",
            "matches": [],
            "match_count": 0,
            "cache_hit": False,
            "reason": reason,
        }

    def taxonomy(self, doc: dict[str, Any]) -> dict[str, Any]:
        """Return taxonomy metadata for one document.

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        return {
            "topic": doc.get("taxonomy_topic"),
            "label": doc.get("taxonomy_label"),
            "path": doc.get("taxonomy_path"),
            "category": doc.get("category"),
            "category_name": doc.get("category_name"),
            "source": doc.get("source"),
            "source_label": doc.get("source_label"),
            "content_type": doc.get("content_type"),
        }

    def topic(self, doc: dict[str, Any]) -> dict[str, Any]:
        """Return lightweight topic information without heavy models.

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        return {
            "available": bool(doc.get("taxonomy_topic") or doc.get("taxonomy_label")),
            "method": "metadata_taxonomy",
            "topic_id": doc.get("taxonomy_topic"),
            "label": doc.get("taxonomy_label"),
            "path": doc.get("taxonomy_path"),
            "cluster": {
                "available": False,
                "reason": "Cluster artifacts are not loaded in lightweight document detail mode.",
            },
        }

    def explanation(
        self,
        doc: dict[str, Any],
        query: str,
        summary: dict[str, Any],
        keywords: dict[str, Any],
        kwic: dict[str, Any],
        related: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Return document-level explanation metadata.

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        return {
            "doc_id": doc.get("doc_id"),
            "query": query,
            "sections": {
                "summary": summary.get("available", False),
                "keywords": keywords.get("available", False),
                "kwic": kwic.get("available", False),
                "related_documents": bool(related),
                "topic": bool(doc.get("taxonomy_topic") or doc.get("taxonomy_label")),
            },
            "signals": {
                "taxonomy_topic": doc.get("taxonomy_topic"),
                "source": doc.get("source"),
                "category": doc.get("category"),
                "keyword_count": len(keywords.get("items") or []),
                "kwic_match_count": kwic.get("match_count", 0),
                "related_count": len(related),
            },
        }

    def _highlight_kwic(self, left: str, keyword: str, right: str) -> str:
        """Return escaped KWIC snippet with the keyword marked.

        Complexity:
            Time: O(n)
            Space: O(n)
        """
        return f"{html.escape(left)}<mark>{html.escape(keyword)}</mark>{html.escape(right)}"
