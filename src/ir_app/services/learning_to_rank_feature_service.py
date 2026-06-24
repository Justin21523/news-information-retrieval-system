"""Learning-to-rank feature export foundation."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from src.ir_app.services.document_service import DocumentService
from src.ir_app.services.feedback_service import FeedbackService
from src.ir_app.services.ranking_diagnostics_service import RankingDiagnosticsService
from src.ir_app.services.search_service import SearchService


class LearningToRankFeatureService:
    """Create feature rows from real search and feedback events.

    Complexity:
        Time: O(n * q)
        Space: O(n)
    """

    def __init__(
        self,
        feedback_service: FeedbackService,
        document_service: DocumentService,
        search_service: SearchService,
        diagnostics_service: RankingDiagnosticsService,
    ):
        self.feedback_service = feedback_service
        self.document_service = document_service
        self.search_service = search_service
        self.diagnostics_service = diagnostics_service

    def features(self, limit: int = 200) -> dict[str, Any]:
        """Return feature rows for API preview/export.

        Complexity:
            Time: O(n * q)
            Space: O(n)
        """
        limit = max(1, min(int(limit or 200), 1000))
        rows = self._candidate_rows(limit)
        features = [self._feature_row(row) for row in rows]
        return {
            "feature_set": "feedback_ltr_v1",
            "description": "Feature foundation built from real search and feedback logs.",
            "rows": [row for row in features if row],
            "row_count": len([row for row in features if row]),
            "limit": limit,
        }

    def export_jsonl(self, output_path: Path, limit: int = 1000) -> dict[str, Any]:
        """Write feature rows as JSONL.

        Complexity:
            Time: O(n * q)
            Space: O(n)
        """
        payload = self.features(limit)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            for row in payload["rows"]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return {
            "output_path": str(output_path),
            "row_count": payload["row_count"],
            "feature_set": payload["feature_set"],
        }

    def _candidate_rows(self, limit: int) -> list[dict[str, Any]]:
        """Return feedback rows plus search top-result rows.

        Complexity:
            Time: O(n)
            Space: O(n)
        """
        rows: list[dict[str, Any]] = []
        with self.feedback_service._connect() as conn:
            conn.row_factory = sqlite3.Row
            feedback_rows = conn.execute(
                """
                SELECT event_type, query, model, doc_id, article_id, rank, score,
                       relevance_grade, metadata, timestamp
                FROM feedback_events
                WHERE query IS NOT NULL AND query != ''
                ORDER BY timestamp DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows.extend(dict(row) | {"source_event": "feedback"} for row in feedback_rows)
            if len(rows) < limit:
                search_rows = conn.execute(
                    """
                    SELECT query, model, top_results, timestamp
                    FROM search_events
                    WHERE query IS NOT NULL AND query != ''
                      AND top_results IS NOT NULL AND top_results != '[]'
                    ORDER BY timestamp DESC, id DESC
                    LIMIT ?
                    """,
                    (limit,),
                )
                for row in search_rows:
                    rows.extend(self._rows_from_search_event(dict(row)))
        return rows[:limit]

    def _rows_from_search_event(self, row: dict[str, Any]) -> list[dict[str, Any]]:
        """Expand compact top_results into feature candidates.

        Complexity:
            Time: O(k)
            Space: O(k)
        """
        try:
            top_results = json.loads(row.get("top_results") or "[]")
        except json.JSONDecodeError:
            top_results = []
        expanded = []
        for rank, identifier in enumerate(top_results, 1):
            expanded.append(
                {
                    "source_event": "search",
                    "event_type": "impression",
                    "query": row.get("query"),
                    "model": row.get("model"),
                    "doc_id": identifier,
                    "article_id": None,
                    "rank": rank,
                    "score": None,
                    "relevance_grade": None,
                    "metadata": "{}",
                    "timestamp": row.get("timestamp"),
                }
            )
        return expanded

    def _feature_row(self, row: dict[str, Any]) -> dict[str, Any] | None:
        """Build one LTR feature row.

        Complexity:
            Time: O(q)
            Space: O(q)
        """
        query = str(row.get("query") or "").strip()
        identifier = row.get("doc_id") if row.get("doc_id") is not None else row.get("article_id")
        if not query or identifier is None:
            return None
        doc = self.document_service.get_document(identifier)
        if not doc and row.get("article_id"):
            doc = self.document_service.get_document(row["article_id"])
        if not doc:
            return None
        model = str(row.get("model") or "bm25").lower()
        query_terms = self.search_service.index.tokenize(query)
        field_boost = self.search_service._field_boost(str(doc["doc_id"]), query_terms)
        field_matches = self.search_service._field_matches(doc, query_terms)
        diagnostics = self._diagnostic_scores(query, doc["doc_id"])
        relevance_grade = row.get("relevance_grade")
        clicked = row.get("event_type") == "click"
        return {
            "query": query,
            "model": model,
            "doc_id": doc.get("doc_id"),
            "article_id": doc.get("article_id"),
            "rank": self._int_or_none(row.get("rank")),
            "score": self._float_or_none(row.get("score")),
            "clicked": clicked,
            "relevance_grade": self._int_or_none(relevance_grade),
            "label": self._label(clicked, relevance_grade),
            "source_event": row.get("source_event"),
            "timestamp": row.get("timestamp"),
            "features": {
                "query_term_count": len(query_terms),
                "title_match_count": len(field_matches.get("title", [])),
                "content_match_count": len(field_matches.get("content", [])),
                "tags_match_count": len(field_matches.get("tags", [])),
                "category_match_count": len(field_matches.get("category", [])),
                "field_boost": float(field_boost.get("boost", 0.0)),
                "bm25_score": diagnostics.get("bm25_score", 0.0),
                "tfidf_score": diagnostics.get("tfidf_score", 0.0),
                "lm_score": diagnostics.get("lm_score", 0.0),
            },
            "metadata": {
                "title": doc.get("title"),
                "source": doc.get("source"),
                "source_label": doc.get("source_label"),
                "category": doc.get("category"),
                "category_name": doc.get("category_name"),
                "taxonomy_topic": doc.get("taxonomy_topic"),
                "taxonomy_label": doc.get("taxonomy_label"),
                "content_type": doc.get("content_type"),
                "tags": doc.get("tags") or [],
            },
        }

    def _diagnostic_scores(self, query: str, doc_id: int | str) -> dict[str, float]:
        """Return compact model scores from diagnostics.

        Complexity:
            Time: O(q)
            Space: O(1)
        """
        try:
            data = self.diagnostics_service.explain(query, doc_id, ["bm25", "tfidf", "lm"])
        except (LookupError, ValueError):
            return {"bm25_score": 0.0, "tfidf_score": 0.0, "lm_score": 0.0}
        models = data.get("models", {})
        return {
            "bm25_score": float(models.get("bm25", {}).get("total_score") or 0.0),
            "tfidf_score": float(models.get("tfidf", {}).get("total_score") or 0.0),
            "lm_score": float(models.get("lm", {}).get("total_score") or 0.0),
        }

    def _label(self, clicked: bool, relevance_grade: Any) -> float:
        """Convert feedback into a weak supervision label.

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        parsed = self._int_or_none(relevance_grade)
        if parsed is not None:
            return round(parsed / 3.0, 6)
        return 1.0 if clicked else 0.0

    def _int_or_none(self, value: Any) -> int | None:
        """Parse optional integer.

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        if value is None or value == "":
            return None
        return int(value)

    def _float_or_none(self, value: Any) -> float | None:
        """Parse optional float.

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        if value is None or value == "":
            return None
        return float(value)
