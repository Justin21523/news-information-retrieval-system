"""JSONL search logging service for future feedback workflows."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class SearchLogService:
    """Append lightweight API interaction logs without affecting requests.

    Complexity:
        Time: O(k) per event
        Space: O(1)
    """

    def __init__(self, project_root: Path, log_path: Path | None = None):
        self.log_path = log_path or project_root / "data" / "logs" / "search_logs.jsonl"

    def log_event(
        self,
        endpoint: str,
        payload: dict[str, Any],
        response_data: dict[str, Any],
        latency: float,
    ) -> None:
        """Persist one request event.

        Complexity:
            Time: O(k)
            Space: O(k)
        """
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            event = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "endpoint": endpoint,
                "query": payload.get("query"),
                "model": payload.get("model"),
                "models": payload.get("models"),
                "query_set": payload.get("query_set"),
                "filters": payload.get("filters") or {},
                "top_k": payload.get("top_k"),
                "latency": latency,
                "result_count": self._result_count(response_data),
                "top_results": self._top_result_ids(response_data),
            }
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        except OSError:
            return

    def _result_count(self, response_data: dict[str, Any]) -> int:
        """Return a compact result count for the event.

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        if isinstance(response_data.get("results"), list):
            return len(response_data["results"])
        if isinstance(response_data.get("models"), dict):
            return sum(
                len(model_data.get("results", []))
                for model_data in response_data["models"].values()
                if isinstance(model_data, dict)
            )
        return 0

    def _top_result_ids(self, response_data: dict[str, Any]) -> list[Any]:
        """Return top document identifiers for future feedback features.

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
            for model_data in response_data["models"].values():
                if not isinstance(model_data, dict):
                    continue
                for item in model_data.get("results", [])[:5]:
                    ids.append(item.get("article_id") or item.get("doc_id"))
            return ids[:20]
        return []
