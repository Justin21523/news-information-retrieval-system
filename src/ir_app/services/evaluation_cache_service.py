"""Disk cache for evaluation responses."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any


class EvaluationCacheService:
    """Cache expensive evaluation payloads by dataset and request identity.

    Complexity:
        Time: O(n) for key normalization and file I/O
        Space: O(n)
    """

    def __init__(self, project_root: Path, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or project_root / "data" / "cache" / "evaluation"

    def make_key(
        self,
        payload: dict[str, Any],
        dataset_hash: str,
        qrels_path: Path,
    ) -> str:
        """Return a stable cache key for one evaluation request.

        Complexity:
            Time: O(n)
            Space: O(n)
        """
        qrels_hash = self._file_hash(qrels_path)
        normalized = {
            "payload": self._normalized_payload(payload),
            "dataset_hash": dataset_hash,
            "qrels_hash": qrels_hash,
        }
        encoded = json.dumps(normalized, sort_keys=True, ensure_ascii=False).encode(
            "utf-8"
        )
        return hashlib.sha256(encoded).hexdigest()

    def get(self, cache_key: str) -> dict[str, Any] | None:
        """Return a cached evaluation result if present.

        Complexity:
            Time: O(n)
            Space: O(n)
        """
        path = self._path(cache_key)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None

    def set(self, cache_key: str, data: dict[str, Any], meta: dict[str, Any]) -> None:
        """Persist an evaluation result.

        Complexity:
            Time: O(n)
            Space: O(n)
        """
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "cache_key": cache_key,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "data": data,
                "meta": meta,
            }
            tmp_path = self._path(cache_key).with_suffix(".tmp")
            with tmp_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False)
            tmp_path.replace(self._path(cache_key))
        except OSError:
            return

    def _path(self, cache_key: str) -> Path:
        """Return cache file path.

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        return self.cache_dir / f"{cache_key}.json"

    def _normalized_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Remove non-semantic controls from cache identity.

        Complexity:
            Time: O(n)
            Space: O(n)
        """
        normalized = dict(payload or {})
        normalized.pop("force_refresh", None)
        return normalized

    def _file_hash(self, path: Path) -> str:
        """Return file hash or a missing marker.

        Complexity:
            Time: O(n)
            Space: O(1)
        """
        if not path.exists():
            return "missing"
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
