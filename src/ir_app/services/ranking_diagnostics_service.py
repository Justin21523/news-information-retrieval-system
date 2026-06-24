"""Ranking diagnostics for explainable retrieval results."""

from __future__ import annotations

from collections import Counter
from typing import Any

from src.ir_app.services.document_service import DocumentService
from src.ir_app.services.search_service import SearchService


class RankingDiagnosticsService:
    """Build model-specific ranking explanations for one query/document pair.

    Complexity:
        Time: O(q + v)
        Space: O(q)
    """

    def __init__(self, document_service: DocumentService, search_service: SearchService):
        self.document_service = document_service
        self.search_service = search_service

    def explain(
        self,
        query: str,
        doc_id: str | int,
        models: list[str] | None = None,
    ) -> dict[str, Any]:
        """Return ranking diagnostics for selected models.

        Complexity:
            Time: O(q)
            Space: O(q)
        """
        query = (query or "").strip()
        if not query:
            raise ValueError("query is required")
        doc = self.document_service.get_document(doc_id)
        if not doc:
            raise LookupError(f"Document not found: {doc_id}")
        model_ids = self._models(models)
        internal_doc_id = int(doc["doc_id"])
        query_terms = self.search_service.index.tokenize(query)
        diagnostics = {}
        for model in model_ids:
            if model == "bm25":
                diagnostics[model] = self._bm25(query, internal_doc_id)
            elif model == "tfidf":
                diagnostics[model] = self._tfidf(query_terms, internal_doc_id)
            elif model == "lm":
                diagnostics[model] = self._lm(query, internal_doc_id)
            elif model == "hybrid":
                diagnostics[model] = self._hybrid(query, query_terms, internal_doc_id)
        return {
            "query": query,
            "query_terms": query_terms,
            "document": self.document_service.to_api_document(doc),
            "models": diagnostics,
        }

    def _models(self, models: list[str] | None) -> list[str]:
        """Normalize model identifiers for diagnostics.

        Complexity:
            Time: O(m)
            Space: O(m)
        """
        requested = models or ["bm25", "tfidf", "lm"]
        aliases = {"vsm": "tfidf", "tf-idf": "tfidf"}
        supported = {"bm25", "tfidf", "lm", "hybrid"}
        normalized = []
        for model in requested:
            model_id = aliases.get(str(model).lower(), str(model).lower())
            if model_id in supported and model_id not in normalized:
                normalized.append(model_id)
        return normalized or ["bm25"]

    def _bm25(self, query: str, doc_id: int) -> dict[str, Any]:
        """Return BM25 diagnostics.

        Complexity:
            Time: O(q)
            Space: O(q)
        """
        raw = self.search_service.index.bm25.explain_score(query, doc_id)
        terms = self._term_array(raw.get("term_contributions", {}))
        return {
            "model": "bm25",
            "total_score": raw.get("total_score", 0.0),
            "doc_length": raw.get("doc_length", 0),
            "avg_doc_length": raw.get("avg_doc_length", 0),
            "parameters": raw.get("parameters", {}),
            "terms": terms,
        }

    def _tfidf(self, query_terms: list[str], doc_id: int) -> dict[str, Any]:
        """Return TF-IDF cosine diagnostics.

        Complexity:
            Time: O(q)
            Space: O(q)
        """
        doc_key = str(doc_id)
        query_vector = self.search_service._query_vector(query_terms)
        doc_vector = self.search_service.index.tfidf_vectors.get(doc_key, {})
        freqs = self.search_service.index.doc_term_freqs.get(doc_key, Counter())
        terms = []
        numerator = 0.0
        for term in query_terms:
            query_weight = float(query_vector.get(term, 0.0))
            doc_weight = float(doc_vector.get(term, 0.0))
            contribution = query_weight * doc_weight
            numerator += contribution
            postings = self.search_service.index.inverted_index.get(term, {})
            terms.append(
                {
                    "term": term,
                    "tf": int(freqs.get(term, 0)),
                    "df": len(postings),
                    "idf": round(float(self.search_service.index.idf.get(term, 0.0)), 6),
                    "query_weight": round(query_weight, 6),
                    "doc_weight": round(doc_weight, 6),
                    "score": round(contribution, 6),
                    "reason": "matched" if doc_weight else "term not in document",
                }
            )
        return {
            "model": "tfidf",
            "total_score": round(numerator, 6),
            "terms": terms,
        }

    def _lm(self, query: str, doc_id: int) -> dict[str, Any]:
        """Return Language Model diagnostics.

        Complexity:
            Time: O(q)
            Space: O(q)
        """
        raw = self.search_service.index.language_model.explain_score(query, doc_id)
        return {
            "model": "lm",
            "total_score": raw.get("total_log_likelihood", 0.0),
            "doc_length": raw.get("doc_length", 0),
            "smoothing": raw.get("smoothing"),
            "parameters": raw.get("parameters", {}),
            "terms": self._term_array(raw.get("term_contributions", {})),
        }

    def _hybrid(self, query: str, query_terms: list[str], doc_id: int) -> dict[str, Any]:
        """Return hybrid diagnostics from BM25 and TF-IDF components.

        Complexity:
            Time: O(q)
            Space: O(q)
        """
        bm25 = self._bm25(query, doc_id)
        tfidf = self._tfidf(query_terms, doc_id)
        return {
            "model": "hybrid",
            "method": "weighted_rrf_components",
            "component_scores": {
                "bm25": bm25.get("total_score", 0.0),
                "tfidf": tfidf.get("total_score", 0.0),
            },
            "components": {"bm25": bm25, "tfidf": tfidf},
        }

    def _term_array(self, contributions: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert term contribution mapping into sorted rows.

        Complexity:
            Time: O(q log q)
            Space: O(q)
        """
        rows = []
        for term, values in contributions.items():
            row = {"term": term, **values}
            if "score" not in row and "log_prob" in row:
                row["score"] = row["log_prob"]
            rows.append(row)
        rows.sort(key=lambda item: abs(float(item.get("score", 0.0))), reverse=True)
        return rows
