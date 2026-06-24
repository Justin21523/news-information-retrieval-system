"""Unified retrieval pipeline and model comparison orchestration."""

from __future__ import annotations

import re
import time
from itertools import combinations
from typing import Any

from src.ir_app.services.search_service import FeatureUnavailableError, SearchService


MODEL_ALIASES = {
    "vsm": "tfidf",
    "tf-idf": "tfidf",
}


class RetrievalOrchestrator:
    """Coordinate retrieval models behind one stable API contract.

    Complexity:
        Time: O(m * search) for compare
        Space: O(m * k)
    """

    def __init__(self, search_service: SearchService):
        self.search_service = search_service

    def canonical_model(self, model: str | None) -> str:
        """Return a canonical model identifier.

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        model_id = (model or "bm25").strip().lower()
        return MODEL_ALIASES.get(model_id, model_id)

    def supported_models(self) -> list[dict[str, Any]]:
        """Return public model capabilities for UI/API discovery.

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        return [
            self.model_info("bm25"),
            self.model_info("tfidf"),
            self.model_info("boolean"),
            self.model_info("hybrid"),
            self.model_info("lm"),
            self.model_info("fuzzy"),
            self.model_info("csoundex"),
            {
                "id": "bert",
                "name": "BERT Semantic Search",
                "available": False,
                "default": False,
                "supports_filters": False,
                "supports_explanation": False,
                "description": "Disabled unless optional semantic dependencies are enabled.",
            },
        ]

    def model_info(self, model: str | None) -> dict[str, Any]:
        """Return model metadata shown in API responses.

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        model_id = self.canonical_model(model)
        if model_id == "bert":
            raise FeatureUnavailableError(
                "BERT semantic search is disabled or unavailable in lightweight startup mode."
            )
        info = {
            "bm25": {
                "name": "BM25",
                "description": "Lexical probabilistic ranking with k1=1.5, b=0.75.",
                "parameters": {"k1": 1.5, "b": 0.75},
            },
            "tfidf": {
                "name": "TF-IDF Cosine",
                "description": "Vector Space Model using normalized TF-IDF cosine scores.",
                "parameters": {},
            },
            "boolean": {
                "name": "Boolean Retrieval",
                "description": "Boolean retrieval with AND/OR/NOT, phrase, and field syntax.",
                "parameters": {},
            },
            "hybrid": {
                "name": "Hybrid RRF",
                "description": "Reciprocal-rank fusion over BM25 and TF-IDF.",
                "parameters": {"bm25_weight": 0.65, "tfidf_weight": 0.35, "rrf_k": 60},
            },
            "lm": {
                "name": "Language Model",
                "description": "Query likelihood retrieval with Dirichlet smoothing.",
                "parameters": {"smoothing": "dirichlet", "mu": 2000.0},
            },
            "fuzzy": {
                "name": "Fuzzy BM25",
                "description": "Edit-distance expansion followed by BM25 ranking.",
                "parameters": {"max_distance": 1},
            },
            "csoundex": {
                "name": "CSoundex BM25",
                "description": "Chinese phonetic expansion followed by BM25 ranking.",
                "parameters": {"threshold": 0.72},
            },
        }.get(model_id)
        if info is None:
            raise ValueError(f"Unknown retrieval model: {model_id}")

        return {
            "id": model_id,
            "available": True,
            "default": model_id == "bm25",
            "supports_filters": True,
            "supports_explanation": True,
            **info,
        }

    def analyze_query(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Analyze query text once for all retrieval models.

        Complexity:
            Time: O(n)
            Space: O(q)
        """
        raw_query = query or ""
        normalized_query = self.search_service.index.normalize_text(raw_query)
        query_terms = self.search_service.index.tokenize(raw_query)
        has_boolean_syntax = bool(
            re.search(r"\b(AND|OR|NOT)\b|:|\"|\(|\)", raw_query, re.IGNORECASE)
        )
        detected_type = "boolean" if has_boolean_syntax else "natural_language"
        return {
            "raw_query": raw_query,
            "normalized_query": normalized_query,
            "query_terms": query_terms,
            "term_count": len(query_terms),
            "detected_type": detected_type,
            "has_boolean_syntax": has_boolean_syntax,
            "filters_applied": filters or {},
        }

    def search(
        self,
        query: str,
        model: str = "bm25",
        top_k: int = 20,
        operator: str = "AND",
        filters: dict[str, list[str]] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Run one model through the unified response contract.

        Complexity:
            Time: O(search)
            Space: O(k)
        """
        model_id = self.canonical_model(model)
        model_info = self.model_info(model_id)
        results, meta = self.search_service.search(query, model_id, top_k, operator, filters)
        meta = {
            **meta,
            "query_analysis": self.analyze_query(query, filters),
            "model_info": model_info,
        }
        return results, meta

    def compare(
        self,
        query: str,
        models: list[str] | None = None,
        top_k: int = 10,
        operator: str = "AND",
        filters: dict[str, list[str]] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run multiple models and compute overlap/rank deltas.

        Complexity:
            Time: O(m * search + m^2 * k)
            Space: O(m * k)
        """
        started = time.perf_counter()
        requested = models or ["bm25", "tfidf", "hybrid", "lm"]
        seen: set[str] = set()
        model_ids = []
        for model in requested:
            model_id = self.canonical_model(model)
            if model_id not in seen:
                seen.add(model_id)
                model_ids.append(model_id)

        model_payloads: dict[str, Any] = {}
        timings: dict[str, float] = {}
        comparisons: dict[str, list[dict[str, Any]]] = {}

        for model_id in model_ids:
            model_started = time.perf_counter()
            try:
                results, meta = self.search(query, model_id, top_k, operator, filters)
                execution_time = meta.get("execution_time", time.perf_counter() - model_started)
                model_payloads[model_id] = {
                    "available": True,
                    "model_info": meta["model_info"],
                    "query_analysis": meta["query_analysis"],
                    "results": results,
                    "total_results": len(results),
                    "execution_time": execution_time,
                }
                comparisons[model_id] = results
                timings[model_id] = execution_time
            except FeatureUnavailableError as exc:
                model_payloads[model_id] = self._unavailable_model_payload(model_id, str(exc))
                comparisons[model_id] = []
                timings[model_id] = 0.0
            except ValueError as exc:
                model_payloads[model_id] = self._unavailable_model_payload(model_id, str(exc))
                comparisons[model_id] = []
                timings[model_id] = 0.0

        data = {
            "query": query,
            "query_analysis": self.analyze_query(query, filters),
            "models": model_payloads,
            "comparison": self._comparison_summary(model_payloads),
            "comparisons": comparisons,
            "timings": timings,
        }
        meta = {"execution_time": time.perf_counter() - started}
        return data, meta

    def _unavailable_model_payload(self, model_id: str, message: str) -> dict[str, Any]:
        """Build a per-model failure payload for comparison responses.

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        return {
            "available": False,
            "model_info": {
                "id": model_id,
                "name": model_id.upper(),
                "available": False,
                "supports_filters": False,
                "supports_explanation": False,
            },
            "results": [],
            "total_results": 0,
            "execution_time": 0.0,
            "error": {"code": "MODEL_UNAVAILABLE", "message": message},
        }

    def _comparison_summary(self, model_payloads: dict[str, Any]) -> dict[str, Any]:
        """Compute overlap, unique documents, and rank changes.

        Complexity:
            Time: O(m^2 * k)
            Space: O(m * k)
        """
        doc_ranks: dict[str, dict[str, int]] = {}
        for model_id, payload in model_payloads.items():
            for result in payload.get("results", []):
                doc_id = str(result.get("doc_id"))
                doc_ranks.setdefault(doc_id, {})[model_id] = int(result.get("rank") or 0)

        overlap: dict[str, int] = {}
        for left, right in combinations(model_payloads.keys(), 2):
            left_docs = {str(item.get("doc_id")) for item in model_payloads[left].get("results", [])}
            right_docs = {str(item.get("doc_id")) for item in model_payloads[right].get("results", [])}
            overlap[f"{left}:{right}"] = len(left_docs.intersection(right_docs))

        unique_docs = {}
        for model_id, payload in model_payloads.items():
            current = {str(item.get("doc_id")) for item in payload.get("results", [])}
            others = set()
            for other_id, other_payload in model_payloads.items():
                if other_id == model_id:
                    continue
                others.update(str(item.get("doc_id")) for item in other_payload.get("results", []))
            unique_docs[model_id] = len(current.difference(others))

        rank_changes = [
            {"doc_id": doc_id, "ranks": ranks, "rank_span": max(ranks.values()) - min(ranks.values())}
            for doc_id, ranks in doc_ranks.items()
            if len(ranks) > 1
        ]
        rank_changes.sort(key=lambda item: item["rank_span"], reverse=True)

        return {
            "overlap": overlap,
            "unique_docs": unique_docs,
            "rank_changes": rank_changes[:50],
        }
