"""SQLite-backed search and feedback event storage."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any


class FeedbackService:
    """Store search, click, and relevance feedback events.

    Complexity:
        Time: O(1) per event
        Space: O(events)
    """

    def __init__(self, project_root: Path, db_path: Path | None = None):
        self.db_path = db_path or project_root / "data" / "feedback" / "feedback.sqlite"
        self._ensure_schema()

    def log_search_event(
        self,
        endpoint: str,
        payload: dict[str, Any],
        response_data: dict[str, Any],
        latency: float,
        dataset_hash: str,
        session_id: str | None = None,
    ) -> int | None:
        """Insert one search-like request event.

        Complexity:
            Time: O(k)
            Space: O(k)
        """
        try:
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO search_events (
                        timestamp, endpoint, query, model, models, filters,
                        top_k, top_results, latency, result_count,
                        dataset_hash, session_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        self._timestamp(),
                        endpoint,
                        payload.get("query"),
                        payload.get("model"),
                        self._json(payload.get("models")),
                        self._json(payload.get("filters") or {}),
                        payload.get("top_k"),
                        self._json(self._top_result_ids(response_data)),
                        latency,
                        self._result_count(response_data),
                        dataset_hash,
                        session_id,
                    ),
                )
                return int(cursor.lastrowid)
        except sqlite3.Error:
            return None

    def record_feedback(
        self,
        payload: dict[str, Any],
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Validate and store one feedback event.

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        event_type = str(payload.get("event_type") or "").lower()
        if event_type not in {"click", "relevance"}:
            raise ValueError("event_type must be click or relevance")
        if payload.get("doc_id") is None and payload.get("article_id") is None:
            raise ValueError("doc_id or article_id is required")
        relevance_grade = payload.get("relevance_grade")
        if event_type == "relevance":
            relevance_grade = int(relevance_grade)
            if relevance_grade < 0 or relevance_grade > 3:
                raise ValueError("relevance_grade must be between 0 and 3")

        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO feedback_events (
                    timestamp, event_type, query, model, doc_id, article_id,
                    rank, score, relevance_grade, session_id, search_event_id,
                    metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self._timestamp(),
                    event_type,
                    payload.get("query"),
                    payload.get("model"),
                    payload.get("doc_id"),
                    payload.get("article_id"),
                    payload.get("rank"),
                    payload.get("score"),
                    relevance_grade,
                    session_id or payload.get("session_id"),
                    payload.get("search_event_id"),
                    self._json(payload.get("metadata") or {}),
                ),
            )
            event_id = int(cursor.lastrowid)
        return {"event_id": event_id, "event_type": event_type}

    def stats(self) -> dict[str, Any]:
        """Return aggregate feedback statistics.

        Complexity:
            Time: O(n log n)
            Space: O(k)
        """
        with self._connect() as conn:
            total_searches = self._scalar(conn, "SELECT COUNT(*) FROM search_events")
            total_clicks = self._scalar(
                conn, "SELECT COUNT(*) FROM feedback_events WHERE event_type='click'"
            )
            total_relevance = self._scalar(
                conn,
                "SELECT COUNT(*) FROM feedback_events WHERE event_type='relevance'",
            )
            top_queries = [
                {"query": row[0], "count": row[1]}
                for row in conn.execute(
                    """
                    SELECT query, COUNT(*) FROM search_events
                    WHERE query IS NOT NULL AND query != ''
                    GROUP BY query ORDER BY COUNT(*) DESC LIMIT 10
                    """
                )
            ]
            top_clicked_docs = [
                {"doc_id": row[0], "article_id": row[1], "count": row[2]}
                for row in conn.execute(
                    """
                    SELECT doc_id, article_id, COUNT(*) FROM feedback_events
                    WHERE event_type='click'
                    GROUP BY doc_id, article_id ORDER BY COUNT(*) DESC LIMIT 10
                    """
                )
            ]
        return {
            "total_searches": total_searches,
            "total_clicks": total_clicks,
            "total_relevance_labels": total_relevance,
            "top_queries": top_queries,
            "top_clicked_docs": top_clicked_docs,
        }

    def _ensure_schema(self) -> None:
        """Create feedback tables if missing.

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS search_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    query TEXT,
                    model TEXT,
                    models TEXT,
                    filters TEXT,
                    top_k INTEGER,
                    top_results TEXT,
                    latency REAL,
                    result_count INTEGER,
                    dataset_hash TEXT,
                    session_id TEXT
                );
                CREATE TABLE IF NOT EXISTS feedback_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    query TEXT,
                    model TEXT,
                    doc_id TEXT,
                    article_id TEXT,
                    rank INTEGER,
                    score REAL,
                    relevance_grade INTEGER,
                    session_id TEXT,
                    search_event_id INTEGER,
                    metadata TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_feedback_query ON feedback_events(query);
                CREATE INDEX IF NOT EXISTS idx_feedback_doc ON feedback_events(doc_id, article_id);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        """Open a SQLite connection.

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        return sqlite3.connect(self.db_path)

    def _scalar(self, conn: sqlite3.Connection, sql: str) -> int:
        """Return one integer scalar.

        Complexity:
            Time: O(n)
            Space: O(1)
        """
        return int(conn.execute(sql).fetchone()[0])

    def _timestamp(self) -> str:
        """Return current UTC timestamp.

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def _json(self, value: Any) -> str:
        """Serialize JSON safely.

        Complexity:
            Time: O(n)
            Space: O(n)
        """
        return json.dumps(value, ensure_ascii=False)

    def _result_count(self, response_data: dict[str, Any]) -> int:
        """Return result count from search-like payloads.

        Complexity:
            Time: O(m)
            Space: O(1)
        """
        if isinstance(response_data.get("results"), list):
            return len(response_data["results"])
        if isinstance(response_data.get("models"), dict):
            return sum(
                len(model.get("results", []))
                for model in response_data["models"].values()
                if isinstance(model, dict)
            )
        return 0

    def _top_result_ids(self, response_data: dict[str, Any]) -> list[Any]:
        """Return compact top result identifiers.

        Complexity:
            Time: O(k)
            Space: O(k)
        """
        if isinstance(response_data.get("results"), list):
            return [
                item.get("article_id") or item.get("doc_id")
                for item in response_data["results"][:10]
                if isinstance(item, dict)
            ]
        if isinstance(response_data.get("models"), dict):
            ids = []
            for model in response_data["models"].values():
                if not isinstance(model, dict):
                    continue
                for item in model.get("results", [])[:5]:
                    ids.append(item.get("article_id") or item.get("doc_id"))
            return ids[:20]
        return []
