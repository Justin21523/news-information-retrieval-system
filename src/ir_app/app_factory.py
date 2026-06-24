"""Flask application factory for the news IR demo."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from flask import Flask, request, render_template

try:
    from flask_cors import CORS
except ImportError:  # pragma: no cover
    CORS = None

from src.ir_app.config import Settings
from src.ir_app.schemas import api_error, api_success
from src.ir_app.services import DocumentService, FeatureUnavailableError, SearchService


def create_app(settings: Settings | None = None) -> Flask:
    """Create and configure the Flask application.

    Complexity:
        Time: O(n) where n is fallback document count
        Space: O(n)
    """
    settings = settings or Settings.from_env()
    app = Flask(
        __name__,
        template_folder=str(settings.project_root / "templates"),
        static_folder=str(settings.project_root / "static"),
    )
    app.config["IR_SETTINGS"] = settings

    if CORS is not None:
        CORS(app)

    document_service = DocumentService(settings)
    search_service = SearchService(settings, document_service)
    app.config["DOCUMENT_SERVICE"] = document_service
    app.config["SEARCH_SERVICE"] = search_service

    register_page_routes(app, settings)
    register_api_routes(app, document_service, search_service)
    return app


def register_page_routes(app: Flask, settings: Settings) -> None:
    """Register template routes.

    Complexity:
        Time: O(1)
        Space: O(1)
    """

    def render_if_exists(template_name: str):
        path = settings.project_root / "templates" / template_name
        if not path.exists():
            return api_error("PAGE_NOT_FOUND", f"Template not found: {template_name}", 404)
        return render_template(template_name)

    @app.route("/")
    def index():
        return render_if_exists("search.html")

    @app.route("/compare")
    def compare_page():
        return render_if_exists("compare.html")

    @app.route("/about")
    def about_page():
        return render_if_exists("about.html")

    @app.route("/expand")
    def expand_page():
        return render_if_exists("expand.html")

    @app.route("/evaluation")
    def evaluation_page():
        return render_if_exists("evaluation.html")

    @app.route("/pat_tree")
    def pat_tree_page():
        return render_if_exists("pat_tree.html")


def register_api_routes(
    app: Flask,
    document_service: DocumentService,
    search_service: SearchService,
) -> None:
    """Register API routes.

    Complexity:
        Time: O(1)
        Space: O(1)
    """

    @app.errorhandler(404)
    def not_found(error):  # noqa: ARG001
        return api_error("NOT_FOUND", "Resource not found", 404)

    @app.errorhandler(500)
    def internal_error(error):  # noqa: ARG001
        return api_error("INTERNAL_ERROR", "Internal server error", 500)

    @app.get("/api/stats")
    def stats():
        data = {"stats": search_service.stats()}
        return api_success(data, stats=data["stats"])

    @app.post("/api/search")
    def search():
        payload = request.get_json(silent=True) or {}
        query = (payload.get("query") or "").strip()
        if not query:
            return api_error("QUERY_REQUIRED", "Query is required", 400)

        model = payload.get("model", "bm25")
        top_k = payload.get("top_k", 20)
        operator = payload.get("operator", "AND")
        filters = payload.get("filters") or None

        try:
            results, meta = search_service.search(query, model, top_k, operator, filters)
        except FeatureUnavailableError as exc:
            return api_error("FEATURE_UNAVAILABLE", str(exc), 503)
        except ValueError as exc:
            return api_error("INVALID_MODEL", str(exc), 400)

        data = {
            "query": query,
            "model": "tfidf" if str(model).lower() == "vsm" else str(model).lower(),
            "results": results,
            "total_results": len(results),
            "response_time": meta["execution_time"],
        }
        return api_success(
            data,
            meta,
            query=data["query"],
            model=data["model"],
            results=results,
            total_results=len(results),
            response_time=meta["execution_time"],
        )

    @app.post("/api/search/faceted")
    def search_faceted():
        payload = request.get_json(silent=True) or {}
        payload.setdefault("filters", {})
        with app.test_request_context(json=payload):
            return search()

    @app.post("/api/facets")
    def facets_for_search():
        payload = request.get_json(silent=True) or {}
        query = (payload.get("query") or "").strip()
        if not query:
            facets = search_service.facets()
            data = {"facets": facets, "total_documents": len(document_service.documents)}
            return api_success(data, facets=facets, total_documents=len(document_service.documents))

        try:
            results, meta = search_service.search(
                query,
                payload.get("model", "bm25"),
                payload.get("top_k", 100),
                payload.get("operator", "AND"),
            )
        except FeatureUnavailableError as exc:
            return api_error("FEATURE_UNAVAILABLE", str(exc), 503)
        doc_ids = [str(result["doc_id"]) for result in results]
        facets = search_service.facets(doc_ids)
        data = {
            "facets": facets,
            "total_documents": len(doc_ids),
            "response_time": meta["execution_time"],
        }
        return api_success(data, meta, facets=facets, total_documents=len(doc_ids))

    @app.get("/api/all_facets")
    def all_facets():
        facets = search_service.facets()
        data = {"facets": facets, "total_documents": len(document_service.documents)}
        return api_success(data, facets=facets, total_documents=len(document_service.documents))

    @app.get("/api/document/<path:doc_id>")
    def document(doc_id: str):
        doc = document_service.get_document(doc_id)
        if not doc:
            return api_error("DOCUMENT_NOT_FOUND", f"Document not found: {doc_id}", 404)

        api_doc = document_service.to_api_document(doc)
        data: dict[str, Any] = {"document": api_doc}
        if request.args.get("include_similar", "false").lower() == "true":
            data["similar_documents"] = []
        return api_success(data, document=api_doc, similar_documents=data.get("similar_documents", []))

    @app.post("/api/summarize")
    def summarize():
        started = time.perf_counter()
        payload = request.get_json(silent=True) or {}
        doc_id = payload.get("doc_id")
        if doc_id is None:
            return api_error("DOC_ID_REQUIRED", "doc_id is required", 400)
        doc = document_service.get_document(doc_id)
        if not doc:
            return api_error("DOCUMENT_NOT_FOUND", f"Document not found: {doc_id}", 404)
        summary = search_service.summarize(doc, payload.get("method", "lead_k"), payload.get("k", 3))
        processing_time = time.perf_counter() - started
        data = {"summary": summary, "processing_time": processing_time}
        return api_success(data, {"execution_time": processing_time}, summary=summary, processing_time=processing_time)

    @app.post("/api/extract_keywords")
    def extract_keywords():
        started = time.perf_counter()
        payload = request.get_json(silent=True) or {}
        doc_id = payload.get("doc_id")
        if doc_id is None:
            return api_error("DOC_ID_REQUIRED", "doc_id is required", 400)
        doc = document_service.get_document(doc_id)
        if not doc:
            return api_error("DOCUMENT_NOT_FOUND", f"Document not found: {doc_id}", 404)
        method = payload.get("method", "tfidf")
        if method in {"yake", "keybert"}:
            return api_error(
                "FEATURE_UNAVAILABLE",
                f"{method} keyword extraction requires optional dependencies.",
                503,
            )
        keywords = search_service.extract_keywords(doc, method, payload.get("top_k", 10))
        processing_time = time.perf_counter() - started
        data = {"keywords": keywords, "method": method, "processing_time": processing_time}
        return api_success(
            data,
            {"execution_time": processing_time},
            keywords=keywords,
            method=method,
            processing_time=processing_time,
        )

    @app.post("/api/expand_query")
    def expand_query():
        payload = request.get_json(silent=True) or {}
        query = (payload.get("query") or "").strip()
        if not query:
            return api_error("QUERY_REQUIRED", "Query is required", 400)
        terms = search_service.index.tokenize(query)
        expanded = list(dict.fromkeys(terms))
        data = {
            "original_query": query,
            "expanded_query": " ".join(expanded),
            "expanded_terms": expanded,
            "method": "tokenizer_terms",
        }
        return api_success(data, **data)

    @app.post("/api/compare")
    def compare_models():
        payload = request.get_json(silent=True) or {}
        query = (payload.get("query") or "").strip()
        if not query:
            return api_error("QUERY_REQUIRED", "Query is required", 400)
        models = payload.get("models") or ["bm25", "tfidf", "boolean"]
        comparisons = {}
        timings = {}
        for model in models:
            try:
                results, meta = search_service.search(query, model, payload.get("top_k", 10))
            except (FeatureUnavailableError, ValueError):
                results, meta = [], {"execution_time": 0.0}
            comparisons[model] = results
            timings[model] = meta["execution_time"]
        data = {"query": query, "comparisons": comparisons, "timings": timings}
        return api_success(data, query=query, comparisons=comparisons, timings=timings)

    @app.post("/api/evaluate")
    def evaluate():
        data = {
            "evaluation_type": "demo",
            "message": "Demo evaluation requires qrels and is not computed in startup fallback mode.",
            "metrics": {},
        }
        return api_success(data, **data)

    @app.get("/api/algorithms")
    def algorithms():
        data = {
            "models": [
                {"id": "bm25", "name": "BM25", "available": True},
                {"id": "tfidf", "name": "TF-IDF", "available": True},
                {"id": "boolean", "name": "Boolean", "available": True},
                {"id": "bert", "name": "BERT", "available": False},
            ]
        }
        return api_success(data, models=data["models"])

    @app.post("/api/export")
    def export_results():
        payload = request.get_json(silent=True) or {}
        data = {"results": payload.get("results") or [], "format": payload.get("format", "json")}
        return api_success(data, **data)


def run() -> None:
    """Run the development server.

    Complexity:
        Time: O(n) startup
        Space: O(n)
    """
    settings = Settings.from_env()
    app = create_app(settings)
    app.run(host=settings.host, port=settings.port, debug=settings.debug)
