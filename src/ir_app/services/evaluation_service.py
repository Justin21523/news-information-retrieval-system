"""Evaluation service for demo qrels and retrieval model comparison."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from src.ir.eval.metrics import Metrics
from src.ir_app.services.document_service import DocumentService
from src.ir_app.services.evaluation_cache_service import EvaluationCacheService
from src.ir_app.services.retrieval_orchestrator import RetrievalOrchestrator


class EvaluationService:
    """Compute demo IR metrics against curated qrels.

    Complexity:
        Time: O(m * q * search + m * q * k)
        Space: O(m * q * k)
    """

    def __init__(
        self,
        document_service: DocumentService,
        retrieval_orchestrator: RetrievalOrchestrator,
        qrels_path: Path | None = None,
        cache_service: EvaluationCacheService | None = None,
    ):
        self.document_service = document_service
        self.retrieval_orchestrator = retrieval_orchestrator
        self.qrels_path = (
            qrels_path
            or document_service.settings.project_root
            / "data"
            / "evaluation"
            / "demo_qrels.json"
        )
        self.cache_service = cache_service
        self.metrics = Metrics()
        self._qrels_payload = self._load_qrels()

    def evaluate(self, payload: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        """Evaluate selected models on a demo query set.

        Complexity:
            Time: O(m * q * search)
            Space: O(m * q * k)
        """
        payload = payload or {}
        cache_key = self.cache_key(payload)
        if self.cache_service and not payload.get("force_refresh"):
            cached = self.cache_service.get(cache_key)
            if cached:
                data = cached["data"]
                meta = {**cached.get("meta", {}), "cached": True, "cache_key": cache_key}
                data["cached"] = True
                data["cache_key"] = cache_key
                return data, meta

        started = time.perf_counter()
        query_set_id = payload.get("query_set") or self.default_query_set_id()
        query_set = self.query_set(query_set_id)
        query_overrides = payload.get("queries") or None
        queries = self._queries_from_payload(query_set, query_overrides)
        models = self._normalize_models(payload.get("models"))
        top_k = self._bounded_int(payload.get("top_k"), 20, 1, 100)
        k_values = self._normalize_k_values(payload.get("k_values"), top_k)
        filters = payload.get("filters") or None
        operator = payload.get("operator", "OR")

        results: dict[str, Any] = {}
        per_query: dict[str, Any] = {}
        timings: dict[str, float] = {}
        qrels_by_query = self._resolved_qrels(query_set)

        for query_def in queries:
            per_query[query_def["id"]] = {
                "query": query_def["query"],
                "description": query_def.get("description", ""),
                "models": {},
            }

        for model in models:
            model_started = time.perf_counter()
            retrieved_by_query: dict[str, list[int]] = {}
            model_query_payloads: dict[str, Any] = {}

            for query_def in queries:
                query_id = query_def["id"]
                qrels = qrels_by_query.get(query_id, {})
                relevant = {doc_id for doc_id, grade in qrels.items() if grade > 0}
                try:
                    search_results, search_meta = self.retrieval_orchestrator.search(
                        query_def["query"],
                        model,
                        top_k,
                        operator,
                        filters,
                    )
                    retrieved = [
                        int(result["doc_id"])
                        for result in search_results
                        if result.get("doc_id") is not None
                    ]
                    query_metrics = self._query_metrics(
                        retrieved,
                        relevant,
                        qrels,
                        k_values,
                    )
                    top_results = self._top_results(search_results, qrels)
                    model_query_payloads[query_id] = {
                        "available": True,
                        "metrics": query_metrics,
                        "top_results": top_results,
                        "relevant_hits": [
                            doc_id for doc_id in retrieved if doc_id in relevant
                        ],
                        "missed_relevant": [
                            self._doc_reference(doc_id)
                            for doc_id in sorted(relevant - set(retrieved))
                        ],
                        "execution_time": search_meta.get("execution_time", 0.0),
                    }
                    retrieved_by_query[query_id] = retrieved
                except Exception as exc:  # pragma: no cover - defensive API boundary
                    model_query_payloads[query_id] = {
                        "available": False,
                        "metrics": self._empty_query_metrics(k_values),
                        "top_results": [],
                        "relevant_hits": [],
                        "missed_relevant": [
                            self._doc_reference(doc_id) for doc_id in sorted(relevant)
                        ],
                        "execution_time": 0.0,
                        "error": {
                            "code": "MODEL_EVALUATION_FAILED",
                            "message": str(exc),
                        },
                    }
                    retrieved_by_query[query_id] = []

            model_metrics = self._aggregate_model_metrics(
                retrieved_by_query,
                qrels_by_query,
                k_values,
            )
            model_metrics["queries_evaluated"] = len(queries)
            model_metrics["available"] = any(
                item.get("available") for item in model_query_payloads.values()
            )
            results[model] = model_metrics
            timings[model] = time.perf_counter() - model_started
            for query_id, query_payload in model_query_payloads.items():
                per_query[query_id]["models"][model] = query_payload

        data = {
            "evaluation_type": "demo",
            "disclaimer": "Demo evaluation, not full benchmark. Qrels are small curated judgments for portfolio demonstration.",
            "query_set": query_set_id,
            "query_set_info": {
                "id": query_set_id,
                "name": query_set.get("name", query_set_id),
                "description": query_set.get("description", ""),
                "judgment_type": query_set.get("judgment_type", "curated_demo"),
            },
            "queries": queries,
            "models": models,
            "top_k": top_k,
            "k_values": k_values,
            "results": results,
            "per_query": per_query,
            "timings": timings,
            "qrels_coverage": self._coverage(query_set, qrels_by_query),
            "dataset": {
                "source": self.document_service.dataset_source,
                "total_documents": len(self.document_service.documents),
                "dataset_hash": self.document_service.dataset_hash,
                "qrels_path": str(self.qrels_path),
            },
            "cached": False,
            "cache_key": cache_key,
        }
        meta = {
            "execution_time": time.perf_counter() - started,
            "cached": False,
            "cache_key": cache_key,
        }
        if self.cache_service:
            self.cache_service.set(cache_key, data, meta)
        return data, meta

    def cache_key(self, payload: dict[str, Any] | None = None) -> str:
        """Return the cache key for one evaluation payload.

        Complexity:
            Time: O(n)
            Space: O(n)
        """
        if not self.cache_service:
            return ""
        return self.cache_service.make_key(
            payload or {},
            self.document_service.dataset_hash,
            self.qrels_path,
        )

    def cached_result(self, cache_key: str) -> dict[str, Any] | None:
        """Return a cached result by key.

        Complexity:
            Time: O(n)
            Space: O(n)
        """
        if not self.cache_service or not cache_key:
            return None
        return self.cache_service.get(cache_key)

    def default_query_set_id(self) -> str:
        """Return the best default query set for the loaded corpus.

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        if "unified_news_corpus" in self.document_service.dataset_source:
            return "news_demo"
        return "mini_ir"

    def query_sets(self) -> list[dict[str, Any]]:
        """Return available query sets for UI discovery.

        Complexity:
            Time: O(s)
            Space: O(s)
        """
        return [
            {
                "id": query_set_id,
                "name": data.get("name", query_set_id),
                "description": data.get("description", ""),
                "query_count": len(data.get("queries", [])),
                "default": query_set_id == self.default_query_set_id(),
            }
            for query_set_id, data in self._qrels_payload.get("query_sets", {}).items()
        ]

    def query_set(self, query_set_id: str) -> dict[str, Any]:
        """Return one query set or the default when missing.

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        query_sets = self._qrels_payload.get("query_sets", {})
        if query_set_id in query_sets:
            return query_sets[query_set_id]
        return query_sets.get(self.default_query_set_id(), {"queries": []})

    def _load_qrels(self) -> dict[str, Any]:
        """Load demo qrels JSON.

        Complexity:
            Time: O(n)
            Space: O(n)
        """
        if not self.qrels_path.exists():
            return {"query_sets": {}}
        with self.qrels_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _queries_from_payload(
        self,
        query_set: dict[str, Any],
        query_overrides: Any,
    ) -> list[dict[str, str]]:
        """Build query definitions from qrels or request overrides.

        Complexity:
            Time: O(q)
            Space: O(q)
        """
        if isinstance(query_overrides, list) and query_overrides:
            queries = []
            for index, item in enumerate(query_overrides, 1):
                if isinstance(item, str):
                    queries.append({"id": f"custom_{index}", "query": item})
                elif isinstance(item, dict) and item.get("query"):
                    queries.append(
                        {
                            "id": str(item.get("id") or f"custom_{index}"),
                            "query": str(item["query"]),
                            "description": str(item.get("description") or ""),
                        }
                    )
            if queries:
                return queries
        return [
            {
                "id": str(item["id"]),
                "query": str(item["query"]),
                "description": str(item.get("description") or ""),
            }
            for item in query_set.get("queries", [])
            if item.get("id") and item.get("query")
        ]

    def _normalize_models(self, models: Any) -> list[str]:
        """Return canonical model IDs for evaluation.

        Complexity:
            Time: O(m)
            Space: O(m)
        """
        requested = models if isinstance(models, list) and models else ["bm25", "tfidf", "hybrid", "lm"]
        normalized = []
        seen = set()
        for model in requested:
            model_id = self.retrieval_orchestrator.canonical_model(str(model))
            if model_id == "bert":
                continue
            if model_id not in seen:
                seen.add(model_id)
                normalized.append(model_id)
        return normalized or ["bm25"]

    def _normalize_k_values(self, value: Any, top_k: int) -> list[int]:
        """Normalize requested k cutoffs.

        Complexity:
            Time: O(k log k)
            Space: O(k)
        """
        if isinstance(value, list):
            raw_values = value
        else:
            raw_values = [5, 10, 20, top_k]
        k_values = {
            self._bounded_int(item, 10, 1, 100)
            for item in raw_values
            if str(item).strip()
        }
        k_values.add(top_k)
        return sorted(k_values)

    def _bounded_int(self, value: Any, default: int, minimum: int, maximum: int) -> int:
        """Parse a bounded integer.

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = default
        return max(minimum, min(parsed, maximum))

    def _resolved_qrels(self, query_set: dict[str, Any]) -> dict[str, dict[int, float]]:
        """Resolve qrel identifiers to current internal doc IDs.

        Complexity:
            Time: O(q * r)
            Space: O(q * r)
        """
        resolved: dict[str, dict[int, float]] = {}
        qrels = query_set.get("qrels", {})
        for query_id, judgments in qrels.items():
            resolved[str(query_id)] = {}
            for judgment in judgments:
                doc = self._resolve_judgment_document(judgment)
                if not doc:
                    continue
                grade = float(judgment.get("relevance", judgment.get("grade", 1)))
                resolved[str(query_id)][int(doc["doc_id"])] = grade
        return resolved

    def _resolve_judgment_document(self, judgment: dict[str, Any]) -> dict[str, Any] | None:
        """Resolve one judgment by article_id or doc_id.

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        identifier = judgment.get("article_id")
        if identifier is None:
            identifier = judgment.get("doc_id")
        if identifier is None:
            return None
        return self.document_service.get_document(identifier)

    def _query_metrics(
        self,
        retrieved: list[int],
        relevant: set[int],
        relevance_scores: dict[int, float],
        k_values: list[int],
    ) -> dict[str, Any]:
        """Compute metrics for one query.

        Complexity:
            Time: O(k log k)
            Space: O(k)
        """
        base = self.metrics.evaluate_query(
            retrieved,
            relevant,
            relevance_scores,
            k_values,
        )
        precision = base.get("precision", 0.0)
        recall = base.get("recall", 0.0)
        return {
            "precision": precision,
            "recall": recall,
            "f1": base.get("f1", 0.0),
            "ap": base.get("ap", 0.0),
            "rr": base.get("rr", 0.0),
            "r_precision": base.get("r_precision", 0.0),
            "bpref": base.get("bpref", 0.0),
            "precision_at_k": [
                {"k": k, "value": base.get(f"p@{k}", 0.0)} for k in k_values
            ],
            "recall_at_k": [
                {"k": k, "value": base.get(f"r@{k}", 0.0)} for k in k_values
            ],
            "f1_at_k": [
                {
                    "k": k,
                    "value": self.metrics.f_measure(
                        base.get(f"p@{k}", 0.0),
                        base.get(f"r@{k}", 0.0),
                    ),
                }
                for k in k_values
            ],
            "ndcg_at_k": [
                {"k": k, "value": base.get(f"ndcg@{k}", 0.0)} for k in k_values
            ],
            "f_beta_scores": self._f_beta_scores(
                [
                    (k, base.get(f"p@{k}", 0.0), base.get(f"r@{k}", 0.0))
                    for k in k_values
                ]
            ),
            "precision_at_recall": self._precision_at_recall(retrieved, relevant),
            "pr_curve": self._pr_curve(retrieved, relevant),
            "interpolated_precision": self._interpolated_precision(retrieved, relevant),
        }

    def _empty_query_metrics(self, k_values: list[int]) -> dict[str, Any]:
        """Return a zero-filled query metric payload.

        Complexity:
            Time: O(k)
            Space: O(k)
        """
        return self._query_metrics([], set(), {}, k_values)

    def _aggregate_model_metrics(
        self,
        retrieved_by_query: dict[str, list[int]],
        qrels_by_query: dict[str, dict[int, float]],
        k_values: list[int],
    ) -> dict[str, Any]:
        """Aggregate metrics across all evaluated queries.

        Complexity:
            Time: O(q * k log k)
            Space: O(q * k)
        """
        query_metrics = []
        for query_id, retrieved in retrieved_by_query.items():
            qrels = qrels_by_query.get(query_id, {})
            relevant = {doc_id for doc_id, grade in qrels.items() if grade > 0}
            query_metrics.append(
                self._query_metrics(retrieved, relevant, qrels, k_values)
            )
        if not query_metrics:
            return self._empty_query_metrics(k_values) | {"map": 0.0, "mrr": 0.0}

        return {
            "map": self._mean(item["ap"] for item in query_metrics),
            "mrr": self._mean(item["rr"] for item in query_metrics),
            "r_precision": self._mean(item["r_precision"] for item in query_metrics),
            "bpref": self._mean(item["bpref"] for item in query_metrics),
            "precision_at_k": self._mean_series(query_metrics, "precision_at_k", k_values),
            "recall_at_k": self._mean_series(query_metrics, "recall_at_k", k_values),
            "f1_at_k": self._mean_series(query_metrics, "f1_at_k", k_values),
            "ndcg_at_k": self._mean_series(query_metrics, "ndcg_at_k", k_values),
            "f_beta_scores": self._mean_f_beta(query_metrics, k_values),
            "precision_at_recall": self._mean_recall_levels(query_metrics, "precision_at_recall"),
            "pr_curve": self._mean_recall_levels(query_metrics, "interpolated_precision"),
            "interpolated_precision": self._mean_recall_levels(query_metrics, "interpolated_precision"),
        }

    def _top_results(
        self,
        search_results: list[dict[str, Any]],
        qrels: dict[int, float],
    ) -> list[dict[str, Any]]:
        """Return compact judged top results.

        Complexity:
            Time: O(k)
            Space: O(k)
        """
        top_results = []
        for rank, result in enumerate(search_results[:10], 1):
            doc_id = int(result["doc_id"])
            top_results.append(
                {
                    "rank": rank,
                    "doc_id": doc_id,
                    "article_id": result.get("article_id"),
                    "title": result.get("title"),
                    "score": result.get("score"),
                    "category": result.get("category"),
                    "source": result.get("source"),
                    "relevance": qrels.get(doc_id, 0.0),
                    "judged": doc_id in qrels,
                }
            )
        return top_results

    def _doc_reference(self, doc_id: int) -> dict[str, Any]:
        """Return a compact document reference.

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        doc = self.document_service.get_document(doc_id)
        if not doc:
            return {"doc_id": doc_id}
        return {
            "doc_id": doc_id,
            "article_id": doc.get("article_id"),
            "title": doc.get("title"),
            "source": doc.get("source"),
        }

    def _coverage(
        self,
        query_set: dict[str, Any],
        qrels_by_query: dict[str, dict[int, float]],
    ) -> dict[str, Any]:
        """Report how many curated judgments resolve in this corpus.

        Complexity:
            Time: O(q)
            Space: O(q)
        """
        raw_qrels = query_set.get("qrels", {})
        per_query = {}
        resolved_total = 0
        raw_total = 0
        for query in query_set.get("queries", []):
            query_id = str(query["id"])
            raw_count = len(raw_qrels.get(query_id, []))
            resolved_count = len(qrels_by_query.get(query_id, {}))
            raw_total += raw_count
            resolved_total += resolved_count
            per_query[query_id] = {
                "query": query["query"],
                "judgments": raw_count,
                "resolved": resolved_count,
                "coverage": resolved_count / raw_count if raw_count else 0.0,
            }
        return {
            "judgments": raw_total,
            "resolved": resolved_total,
            "coverage": resolved_total / raw_total if raw_total else 0.0,
            "per_query": per_query,
        }

    def _precision_at_recall(self, retrieved: list[int], relevant: set[int]) -> list[dict[str, float]]:
        """Return precision at standard recall levels.

        Complexity:
            Time: O(k * r)
            Space: O(1)
        """
        curve = self._pr_curve(retrieved, relevant)
        levels = [i / 10 for i in range(11)]
        values = []
        for level in levels:
            precision = max(
                [point["precision"] for point in curve if point["recall"] >= level],
                default=0.0,
            )
            values.append({"recall": level, "precision": precision})
        return values

    def _pr_curve(self, retrieved: list[int], relevant: set[int]) -> list[dict[str, float]]:
        """Return raw precision-recall points along the ranking.

        Complexity:
            Time: O(k)
            Space: O(k)
        """
        if not retrieved or not relevant:
            return [{"recall": 0.0, "precision": 0.0}]
        points = []
        hits = 0
        for rank, doc_id in enumerate(retrieved, 1):
            if doc_id in relevant:
                hits += 1
            points.append({"recall": hits / len(relevant), "precision": hits / rank})
        return points

    def _interpolated_precision(
        self,
        retrieved: list[int],
        relevant: set[int],
    ) -> list[dict[str, float]]:
        """Return 11-point interpolated precision.

        Complexity:
            Time: O(k * r)
            Space: O(1)
        """
        return self._precision_at_recall(retrieved, relevant)

    def _f_beta_scores(self, values: list[tuple[int, float, float]]) -> list[dict[str, float]]:
        """Return F-beta scores for beta 0.5 and 2.0.

        Complexity:
            Time: O(k)
            Space: O(k)
        """
        scores = []
        for beta in (0.5, 2.0):
            for k, precision, recall in values:
                scores.append(
                    {
                        "beta": beta,
                        "k": k,
                        "value": self.metrics.f_measure(precision, recall, beta),
                    }
                )
        return scores

    def _mean_series(
        self,
        query_metrics: list[dict[str, Any]],
        key: str,
        k_values: list[int],
    ) -> list[dict[str, float]]:
        """Average a k-based metric series.

        Complexity:
            Time: O(q * k)
            Space: O(k)
        """
        averaged = []
        for k in k_values:
            averaged.append(
                {
                    "k": k,
                    "value": self._mean(
                        self._series_value(metrics[key], "k", k, "value")
                        for metrics in query_metrics
                    ),
                }
            )
        return averaged

    def _mean_f_beta(
        self,
        query_metrics: list[dict[str, Any]],
        k_values: list[int],
    ) -> list[dict[str, float]]:
        """Average F-beta score series.

        Complexity:
            Time: O(q * k)
            Space: O(k)
        """
        averaged = []
        for beta in (0.5, 2.0):
            for k in k_values:
                averaged.append(
                    {
                        "beta": beta,
                        "k": k,
                        "value": self._mean(
                            self._series_value(
                                metrics["f_beta_scores"],
                                "k",
                                k,
                                "value",
                                extra_key="beta",
                                extra_value=beta,
                            )
                            for metrics in query_metrics
                        ),
                    }
                )
        return averaged

    def _mean_recall_levels(
        self,
        query_metrics: list[dict[str, Any]],
        key: str,
    ) -> list[dict[str, float]]:
        """Average recall-level metric series.

        Complexity:
            Time: O(q * 11)
            Space: O(1)
        """
        levels = [i / 10 for i in range(11)]
        return [
            {
                "recall": level,
                "precision": self._mean(
                    self._series_value(metrics[key], "recall", level, "precision")
                    for metrics in query_metrics
                ),
            }
            for level in levels
        ]

    def _series_value(
        self,
        series: list[dict[str, Any]],
        match_key: str,
        match_value: Any,
        value_key: str,
        extra_key: str | None = None,
        extra_value: Any = None,
    ) -> float:
        """Read one value from a metric series.

        Complexity:
            Time: O(k)
            Space: O(1)
        """
        for item in series:
            if item.get(match_key) != match_value:
                continue
            if extra_key is not None and item.get(extra_key) != extra_value:
                continue
            return float(item.get(value_key, 0.0))
        return 0.0

    def _mean(self, values: Any) -> float:
        """Return arithmetic mean for a finite iterable.

        Complexity:
            Time: O(n)
            Space: O(n)
        """
        values_list = [float(value) for value in values]
        return sum(values_list) / len(values_list) if values_list else 0.0
