"""Feedback analytics over SQLite search and feedback events."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from typing import Any

from src.ir_app.services.feedback_service import FeedbackService


class FeedbackAnalyticsService:
    """Build dashboard-ready feedback aggregates.

    Complexity:
        Time: O(n log n)
        Space: O(k)
    """

    def __init__(self, feedback_service: FeedbackService):
        self.feedback_service = feedback_service

    def analytics(self, days: int = 30, limit: int = 20) -> dict[str, Any]:
        """Return feedback analytics for the dashboard.

        Complexity:
            Time: O(n log n)
            Space: O(k)
        """
        days = max(1, min(int(days or 30), 365))
        limit = max(1, min(int(limit or 20), 100))
        window = f"-{days} days"
        with self.feedback_service._connect() as conn:
            conn.row_factory = sqlite3.Row
            total_searches = self._count(
                conn,
                "SELECT COUNT(*) FROM search_events WHERE timestamp >= datetime('now', ?)",
                (window,),
            )
            total_clicks = self._count(
                conn,
                """
                SELECT COUNT(*) FROM feedback_events
                WHERE event_type='click' AND timestamp >= datetime('now', ?)
                """,
                (window,),
            )
            total_labels = self._count(
                conn,
                """
                SELECT COUNT(*) FROM feedback_events
                WHERE event_type='relevance' AND timestamp >= datetime('now', ?)
                """,
                (window,),
            )
            zero_result_queries = self._zero_result_queries(conn, window, limit)
            model_metrics = self._model_metrics(conn, window)
            relevance_distribution = self._relevance_distribution(conn, window)
            top_queries = self._top_queries(conn, window, limit)
            top_clicked_docs = self._top_clicked_docs(conn, window, limit)
            recent_feedback = self._recent_feedback(conn, window, limit)
            session_metrics = self._session_metrics(conn, window)
            quality = self._quality_metrics(conn, window, limit)
            position_bias = self._position_bias(conn, window)

        return {
            "window_days": days,
            "summary": {
                "total_searches": total_searches,
                "total_clicks": total_clicks,
                "total_relevance_labels": total_labels,
                "ctr": self._ratio(total_clicks, total_searches),
                "zero_result_queries": sum(item["count"] for item in zero_result_queries),
                "has_events": bool(total_searches or total_clicks or total_labels),
            },
            "top_queries": top_queries,
            "zero_result_queries": zero_result_queries,
            "top_clicked_docs": top_clicked_docs,
            "model_metrics": model_metrics,
            "relevance_distribution": relevance_distribution,
            "recent_feedback": recent_feedback,
            "session_metrics": session_metrics,
            "quality": quality,
            "position_bias": position_bias,
        }

    def _count(
        self,
        conn: sqlite3.Connection,
        sql: str,
        params: tuple[Any, ...],
    ) -> int:
        """Return an integer count.

        Complexity:
            Time: O(n)
            Space: O(1)
        """
        return int(conn.execute(sql, params).fetchone()[0])

    def _top_queries(
        self,
        conn: sqlite3.Connection,
        window: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Return most common queries.

        Complexity:
            Time: O(n log n)
            Space: O(k)
        """
        rows = conn.execute(
            """
            SELECT query, COUNT(*) AS count, AVG(latency) AS avg_latency
            FROM search_events
            WHERE timestamp >= datetime('now', ?)
              AND query IS NOT NULL AND query != ''
            GROUP BY query
            ORDER BY count DESC, query ASC
            LIMIT ?
            """,
            (window, limit),
        )
        return [
            {
                "query": row["query"],
                "count": int(row["count"]),
                "avg_latency": round(float(row["avg_latency"] or 0.0), 6),
            }
            for row in rows
        ]

    def _zero_result_queries(
        self,
        conn: sqlite3.Connection,
        window: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Return zero-result query aggregates.

        Complexity:
            Time: O(n log n)
            Space: O(k)
        """
        rows = conn.execute(
            """
            SELECT query, model, COUNT(*) AS count, MAX(timestamp) AS last_seen
            FROM search_events
            WHERE timestamp >= datetime('now', ?)
              AND result_count = 0
              AND query IS NOT NULL AND query != ''
            GROUP BY query, model
            ORDER BY count DESC, last_seen DESC
            LIMIT ?
            """,
            (window, limit),
        )
        return [
            {
                "query": row["query"],
                "model": row["model"],
                "count": int(row["count"]),
                "last_seen": row["last_seen"],
            }
            for row in rows
        ]

    def _top_clicked_docs(
        self,
        conn: sqlite3.Connection,
        window: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Return most clicked documents.

        Complexity:
            Time: O(n log n)
            Space: O(k)
        """
        rows = conn.execute(
            """
            SELECT doc_id, article_id, query, model, COUNT(*) AS clicks
            FROM feedback_events
            WHERE timestamp >= datetime('now', ?)
              AND event_type='click'
            GROUP BY doc_id, article_id, query, model
            ORDER BY clicks DESC
            LIMIT ?
            """,
            (window, limit),
        )
        return [
            {
                "doc_id": row["doc_id"],
                "article_id": row["article_id"],
                "query": row["query"],
                "model": row["model"],
                "clicks": int(row["clicks"]),
            }
            for row in rows
        ]

    def _model_metrics(
        self,
        conn: sqlite3.Connection,
        window: str,
    ) -> list[dict[str, Any]]:
        """Return per-model search and click metrics.

        Complexity:
            Time: O(n)
            Space: O(m)
        """
        searches: dict[str, dict[str, Any]] = {}
        for row in conn.execute(
            """
            SELECT COALESCE(model, 'compare') AS model,
                   COUNT(*) AS searches,
                   AVG(latency) AS avg_latency,
                   SUM(CASE WHEN result_count = 0 THEN 1 ELSE 0 END) AS zero_results
            FROM search_events
            WHERE timestamp >= datetime('now', ?)
            GROUP BY COALESCE(model, 'compare')
            """,
            (window,),
        ):
            searches[row["model"]] = {
                "model": row["model"],
                "searches": int(row["searches"]),
                "avg_latency": round(float(row["avg_latency"] or 0.0), 6),
                "zero_results": int(row["zero_results"] or 0),
                "clicks": 0,
                "relevance_labels": 0,
            }

        for row in conn.execute(
            """
            SELECT COALESCE(model, 'unknown') AS model,
                   SUM(CASE WHEN event_type='click' THEN 1 ELSE 0 END) AS clicks,
                   SUM(CASE WHEN event_type='relevance' THEN 1 ELSE 0 END) AS labels
            FROM feedback_events
            WHERE timestamp >= datetime('now', ?)
            GROUP BY COALESCE(model, 'unknown')
            """,
            (window,),
        ):
            item = searches.setdefault(
                row["model"],
                {
                    "model": row["model"],
                    "searches": 0,
                    "avg_latency": 0.0,
                    "zero_results": 0,
                    "clicks": 0,
                    "relevance_labels": 0,
                },
            )
            item["clicks"] = int(row["clicks"] or 0)
            item["relevance_labels"] = int(row["labels"] or 0)

        for item in searches.values():
            item["ctr"] = self._ratio(item["clicks"], item["searches"])
            item["zero_result_rate"] = self._ratio(item["zero_results"], item["searches"])
        return sorted(searches.values(), key=lambda item: item["searches"], reverse=True)

    def _relevance_distribution(
        self,
        conn: sqlite3.Connection,
        window: str,
    ) -> list[dict[str, Any]]:
        """Return relevance label counts.

        Complexity:
            Time: O(n)
            Space: O(1)
        """
        counts = Counter({0: 0, 1: 0, 2: 0, 3: 0})
        rows = conn.execute(
            """
            SELECT relevance_grade, COUNT(*) AS count
            FROM feedback_events
            WHERE timestamp >= datetime('now', ?)
              AND event_type='relevance'
              AND relevance_grade IS NOT NULL
            GROUP BY relevance_grade
            """,
            (window,),
        )
        for row in rows:
            counts[int(row["relevance_grade"])] = int(row["count"])
        return [{"grade": grade, "count": counts[grade]} for grade in range(4)]

    def _recent_feedback(
        self,
        conn: sqlite3.Connection,
        window: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        """Return recent feedback events.

        Complexity:
            Time: O(n log n)
            Space: O(k)
        """
        rows = conn.execute(
            """
            SELECT timestamp, event_type, query, model, doc_id, article_id,
                   rank, score, relevance_grade, metadata
            FROM feedback_events
            WHERE timestamp >= datetime('now', ?)
            ORDER BY timestamp DESC, id DESC
            LIMIT ?
            """,
            (window, limit),
        )
        return [
            {
                "timestamp": row["timestamp"],
                "event_type": row["event_type"],
                "query": row["query"],
                "model": row["model"],
                "doc_id": row["doc_id"],
                "article_id": row["article_id"],
                "rank": row["rank"],
                "score": row["score"],
                "relevance_grade": row["relevance_grade"],
                "metadata": self._loads(row["metadata"]),
            }
            for row in rows
        ]

    def _session_metrics(
        self,
        conn: sqlite3.Connection,
        window: str,
    ) -> dict[str, Any]:
        """Return session-level analytics.

        Complexity:
            Time: O(n)
            Space: O(1)
        """
        search_sessions = self._count(
            conn,
            """
            SELECT COUNT(DISTINCT session_id) FROM search_events
            WHERE timestamp >= datetime('now', ?)
              AND session_id IS NOT NULL AND session_id != ''
            """,
            (window,),
        )
        feedback_sessions = self._count(
            conn,
            """
            SELECT COUNT(DISTINCT session_id) FROM feedback_events
            WHERE timestamp >= datetime('now', ?)
              AND session_id IS NOT NULL AND session_id != ''
            """,
            (window,),
        )
        rows = conn.execute(
            """
            SELECT session_id, COUNT(*) AS events
            FROM feedback_events
            WHERE timestamp >= datetime('now', ?)
              AND session_id IS NOT NULL AND session_id != ''
            GROUP BY session_id
            ORDER BY events DESC
            LIMIT 10
            """,
            (window,),
        )
        top_feedback_sessions = [
            {"session_id": row["session_id"], "events": int(row["events"])}
            for row in rows
        ]
        total_searches = self._count(
            conn,
            "SELECT COUNT(*) FROM search_events WHERE timestamp >= datetime('now', ?)",
            (window,),
        )
        total_feedback = self._count(
            conn,
            "SELECT COUNT(*) FROM feedback_events WHERE timestamp >= datetime('now', ?)",
            (window,),
        )
        return {
            "unique_search_sessions": search_sessions,
            "unique_feedback_sessions": feedback_sessions,
            "searches_per_session": self._ratio(total_searches, search_sessions),
            "feedback_per_session": self._ratio(total_feedback, feedback_sessions),
            "top_feedback_sessions": top_feedback_sessions,
        }

    def _quality_metrics(
        self,
        conn: sqlite3.Connection,
        window: str,
        limit: int,
    ) -> dict[str, Any]:
        """Return raw feedback quality diagnostics.

        Complexity:
            Time: O(n log n)
            Space: O(k)
        """
        duplicate_rows = conn.execute(
            """
            SELECT session_id, event_type, query, model, doc_id, article_id,
                   COUNT(*) AS count
            FROM feedback_events
            WHERE timestamp >= datetime('now', ?)
            GROUP BY session_id, event_type, query, model, doc_id, article_id
            HAVING COUNT(*) > 1
            ORDER BY count DESC
            LIMIT ?
            """,
            (window, limit),
        )
        duplicate_groups = [
            {
                "session_id": row["session_id"],
                "event_type": row["event_type"],
                "query": row["query"],
                "model": row["model"],
                "doc_id": row["doc_id"],
                "article_id": row["article_id"],
                "count": int(row["count"]),
            }
            for row in duplicate_rows
        ]
        duplicate_event_count = sum(item["count"] - 1 for item in duplicate_groups)
        demo_seed_events = self._count(
            conn,
            """
            SELECT COUNT(*) FROM feedback_events
            WHERE timestamp >= datetime('now', ?)
              AND metadata LIKE '%playwright_demo%'
            """,
            (window,),
        )
        return {
            "raw_metrics_count_all_events": True,
            "notice": "Raw metrics count all events, including duplicates and demo seed events.",
            "duplicate_groups": duplicate_groups,
            "duplicate_group_count": len(duplicate_groups),
            "duplicate_event_count": duplicate_event_count,
            "demo_seed_event_count": demo_seed_events,
        }

    def _position_bias(
        self,
        conn: sqlite3.Connection,
        window: str,
    ) -> dict[str, Any]:
        """Return click/relevance counts by rank bucket.

        Complexity:
            Time: O(n)
            Space: O(1)
        """
        buckets = {
            "rank_1": {"clicks": 0, "relevance": 0},
            "rank_2_3": {"clicks": 0, "relevance": 0},
            "rank_4_10": {"clicks": 0, "relevance": 0},
            "rank_11_plus": {"clicks": 0, "relevance": 0},
            "unknown": {"clicks": 0, "relevance": 0},
        }
        ranks = []
        rows = conn.execute(
            """
            SELECT event_type, rank FROM feedback_events
            WHERE timestamp >= datetime('now', ?)
            """,
            (window,),
        )
        for row in rows:
            bucket = self._rank_bucket(row["rank"])
            metric = "clicks" if row["event_type"] == "click" else "relevance"
            buckets[bucket][metric] += 1
            if row["event_type"] == "click" and row["rank"] is not None:
                ranks.append(int(row["rank"]))
        average_clicked_rank = round(sum(ranks) / len(ranks), 4) if ranks else None
        return {
            "buckets": buckets,
            "average_clicked_rank": average_clicked_rank,
            "warning": "Clicks are position-biased; early ranks are more likely to receive feedback.",
        }

    def _rank_bucket(self, rank: Any) -> str:
        """Return a small rank bucket label.

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        if rank is None or rank == "":
            return "unknown"
        rank_value = int(rank)
        if rank_value == 1:
            return "rank_1"
        if rank_value <= 3:
            return "rank_2_3"
        if rank_value <= 10:
            return "rank_4_10"
        return "rank_11_plus"

    def _loads(self, value: str | None) -> Any:
        """Parse JSON text defensively.

        Complexity:
            Time: O(n)
            Space: O(n)
        """
        if not value:
            return {}
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {}

    def _ratio(self, numerator: int | float, denominator: int | float) -> float:
        """Return a rounded ratio.

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        if not denominator:
            return 0.0
        return round(float(numerator) / float(denominator), 6)
