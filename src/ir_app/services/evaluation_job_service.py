"""In-process background jobs for evaluation requests."""

from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from src.ir_app.services.evaluation_service import EvaluationService


class EvaluationJobService:
    """Run evaluation jobs in one local worker.

    Complexity:
        Time: O(search) per job
        Space: O(jobs + result)
    """

    def __init__(self, evaluation_service: EvaluationService):
        self.evaluation_service = evaluation_service
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.jobs: dict[str, dict[str, Any]] = {}
        self.lock = threading.Lock()

    def submit(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Submit or reuse an evaluation job.

        Complexity:
            Time: O(n)
            Space: O(1)
        """
        cache_key = self.evaluation_service.cache_key(payload)
        if not payload.get("force_refresh"):
            cached = self.evaluation_service.cached_result(cache_key)
            if cached:
                return {
                    "job_id": None,
                    "status": "completed",
                    "cache_key": cache_key,
                    "cached": True,
                    "result": cached["data"],
                    "meta": cached.get("meta", {}),
                }

        with self.lock:
            for job_id, job in self.jobs.items():
                if job["cache_key"] == cache_key and job["status"] in {
                    "queued",
                    "running",
                }:
                    return self.public_job(job_id, job)
            job_id = uuid.uuid4().hex
            job = {
                "job_id": job_id,
                "status": "queued",
                "cache_key": cache_key,
                "cached": False,
                "created_at": time.time(),
                "updated_at": time.time(),
            }
            self.jobs[job_id] = job
            self.executor.submit(self._run, job_id, dict(payload))
            return self.public_job(job_id, job)

    def get(self, job_id: str) -> dict[str, Any] | None:
        """Return one public job payload.

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        with self.lock:
            job = self.jobs.get(job_id)
            return self.public_job(job_id, job) if job else None

    def public_job(self, job_id: str | None, job: dict[str, Any]) -> dict[str, Any]:
        """Return serializable job state.

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        payload = {
            "job_id": job_id,
            "status": job["status"],
            "cache_key": job["cache_key"],
            "cached": job.get("cached", False),
            "created_at": job.get("created_at"),
            "updated_at": job.get("updated_at"),
        }
        if "result" in job:
            payload["result"] = job["result"]
        if "meta" in job:
            payload["meta"] = job["meta"]
        if "error" in job:
            payload["error"] = job["error"]
        return payload

    def _run(self, job_id: str, payload: dict[str, Any]) -> None:
        """Run one job and update shared state.

        Complexity:
            Time: O(search)
            Space: O(result)
        """
        with self.lock:
            self.jobs[job_id]["status"] = "running"
            self.jobs[job_id]["updated_at"] = time.time()
        try:
            data, meta = self.evaluation_service.evaluate(payload)
            with self.lock:
                self.jobs[job_id].update(
                    {
                        "status": "completed",
                        "result": data,
                        "meta": meta,
                        "updated_at": time.time(),
                    }
                )
        except Exception as exc:  # pragma: no cover - defensive worker boundary
            with self.lock:
                self.jobs[job_id].update(
                    {
                        "status": "failed",
                        "error": {
                            "code": "EVALUATION_JOB_FAILED",
                            "message": str(exc),
                        },
                        "updated_at": time.time(),
                    }
                )
