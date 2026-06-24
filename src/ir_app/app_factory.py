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
from src.ir_app.services import (
    DocumentDetailService,
    DocumentService,
    EvaluationCacheService,
    EvaluationJobService,
    EvaluationService,
    FeedbackAnalyticsService,
    FeedbackService,
    FeatureUnavailableError,
    LearningToRankFeatureService,
    LearningToRankTrainingService,
    RankingDiagnosticsService,
    RetrievalOrchestrator,
    SearchLogService,
    SearchService,
)


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
    document_detail_service = DocumentDetailService(document_service, search_service)
    retrieval_orchestrator = RetrievalOrchestrator(search_service)
    evaluation_cache_service = EvaluationCacheService(settings.project_root)
    evaluation_service = EvaluationService(
        document_service,
        retrieval_orchestrator,
        cache_service=evaluation_cache_service,
    )
    evaluation_job_service = EvaluationJobService(evaluation_service)
    search_log_service = SearchLogService(settings.project_root)
    feedback_service = FeedbackService(settings.project_root)
    feedback_analytics_service = FeedbackAnalyticsService(feedback_service)
    ranking_diagnostics_service = RankingDiagnosticsService(
        document_service, search_service
    )
    ltr_feature_service = LearningToRankFeatureService(
        feedback_service,
        document_service,
        search_service,
        ranking_diagnostics_service,
    )
    ltr_training_service = LearningToRankTrainingService(ltr_feature_service)
    app.config["DOCUMENT_SERVICE"] = document_service
    app.config["SEARCH_SERVICE"] = search_service
    app.config["DOCUMENT_DETAIL_SERVICE"] = document_detail_service
    app.config["RETRIEVAL_ORCHESTRATOR"] = retrieval_orchestrator
    app.config["EVALUATION_CACHE_SERVICE"] = evaluation_cache_service
    app.config["EVALUATION_SERVICE"] = evaluation_service
    app.config["EVALUATION_JOB_SERVICE"] = evaluation_job_service
    app.config["SEARCH_LOG_SERVICE"] = search_log_service
    app.config["FEEDBACK_SERVICE"] = feedback_service
    app.config["FEEDBACK_ANALYTICS_SERVICE"] = feedback_analytics_service
    app.config["RANKING_DIAGNOSTICS_SERVICE"] = ranking_diagnostics_service
    app.config["LTR_FEATURE_SERVICE"] = ltr_feature_service
    app.config["LTR_TRAINING_SERVICE"] = ltr_training_service

    register_page_routes(app, settings)
    register_api_routes(
        app,
        document_service,
        search_service,
        retrieval_orchestrator,
        document_detail_service,
        evaluation_service,
        evaluation_job_service,
        search_log_service,
        feedback_service,
        feedback_analytics_service,
        ranking_diagnostics_service,
        ltr_feature_service,
        ltr_training_service,
    )
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
            return api_error(
                "PAGE_NOT_FOUND", f"Template not found: {template_name}", 404
            )
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

    @app.route("/diagnostics")
    def diagnostics_page():
        return render_if_exists("diagnostics.html")

    @app.route("/feedback")
    def feedback_page():
        return render_if_exists("feedback.html")

    @app.route("/pat_tree")
    def pat_tree_page():
        return render_if_exists("pat_tree.html")


def register_api_routes(
    app: Flask,
    document_service: DocumentService,
    search_service: SearchService,
    retrieval_orchestrator: RetrievalOrchestrator,
    document_detail_service: DocumentDetailService,
    evaluation_service: EvaluationService,
    evaluation_job_service: EvaluationJobService,
    search_log_service: SearchLogService,
    feedback_service: FeedbackService,
    feedback_analytics_service: FeedbackAnalyticsService,
    ranking_diagnostics_service: RankingDiagnosticsService,
    ltr_feature_service: LearningToRankFeatureService,
    ltr_training_service: LearningToRankTrainingService,
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
            results, meta = retrieval_orchestrator.search(
                query, model, top_k, operator, filters
            )
        except FeatureUnavailableError as exc:
            return api_error("FEATURE_UNAVAILABLE", str(exc), 503)
        except ValueError as exc:
            return api_error("INVALID_MODEL", str(exc), 400)

        model_info = meta.get("model_info", {})
        data = {
            "query": query,
            "model": model_info.get(
                "id", "tfidf" if str(model).lower() == "vsm" else str(model).lower()
            ),
            "results": results,
            "total_results": len(results),
            "response_time": meta["execution_time"],
            "query_analysis": meta.get("query_analysis", {}),
            "model_info": model_info,
            "suggestions": meta.get("suggestions", []),
        }
        search_log_service.log_event(
            "/api/search",
            payload,
            data,
            meta["execution_time"],
        )
        feedback_service.log_search_event(
            "/api/search",
            payload,
            data,
            meta["execution_time"],
            document_service.dataset_hash,
            _session_id(),
        )
        return api_success(
            data,
            meta,
            query=data["query"],
            model=data["model"],
            results=results,
            total_results=len(results),
            response_time=meta["execution_time"],
            query_analysis=data["query_analysis"],
            model_info=model_info,
            suggestions=data["suggestions"],
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
        filters = payload.get("filters") or None
        started = time.perf_counter()
        try:
            doc_ids = search_service.candidate_doc_ids(
                query,
                payload.get("model", "bm25"),
                payload.get("operator", "AND"),
                filters,
            )
        except FeatureUnavailableError as exc:
            return api_error("FEATURE_UNAVAILABLE", str(exc), 503)
        facets = search_service.facets(doc_ids, filters)
        meta = {"execution_time": time.perf_counter() - started}
        data = {
            "facets": facets,
            "total_documents": len(doc_ids),
            "corpus_distribution": search_service.corpus_distribution(),
            "response_time": meta["execution_time"],
        }
        return api_success(
            data,
            meta,
            facets=facets,
            total_documents=len(doc_ids),
            corpus_distribution=data["corpus_distribution"],
        )

    @app.get("/api/all_facets")
    def all_facets():
        facets = search_service.facets()
        distribution = search_service.corpus_distribution()
        data = {
            "facets": facets,
            "total_documents": len(document_service.documents),
            "corpus_distribution": distribution,
        }
        return api_success(
            data,
            facets=facets,
            total_documents=len(document_service.documents),
            corpus_distribution=distribution,
        )

    @app.get("/api/document/<path:doc_id>")
    def document(doc_id: str):
        doc = document_service.get_document(doc_id)
        if not doc:
            return api_error("DOCUMENT_NOT_FOUND", f"Document not found: {doc_id}", 404)

        include_related = _truthy_arg("include_related", True) or _truthy_arg(
            "include_similar", False
        )
        query = (request.args.get("query") or "").strip()
        include_kwic = request.args.get("include_kwic")
        kwic_flag = (
            None
            if include_kwic is None
            else include_kwic.lower() in {"1", "true", "yes", "on"}
        )
        top_k = request.args.get("top_k", 5)
        data, meta = document_detail_service.build_detail(
            doc,
            query=query,
            top_k=top_k,
            include_related=include_related,
            include_kwic=kwic_flag,
        )
        return api_success(
            data,
            meta,
            document=data["document"],
            summary=data["summary"],
            keywords=data["keywords"],
            kwic=data["kwic"],
            related_documents=data["related_documents"],
            similar_documents=data["similar_documents"],
            taxonomy=data["taxonomy"],
            topic=data["topic"],
            explanation=data["explanation"],
        )

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
        summary = search_service.summarize(
            doc, payload.get("method", "lead_k"), payload.get("k", 3)
        )
        processing_time = time.perf_counter() - started
        data = {"summary": summary, "processing_time": processing_time}
        return api_success(
            data,
            {"execution_time": processing_time},
            summary=summary,
            processing_time=processing_time,
        )

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
        keywords = search_service.extract_keywords(
            doc, method, payload.get("top_k", 10)
        )
        processing_time = time.perf_counter() - started
        data = {
            "keywords": keywords,
            "method": method,
            "processing_time": processing_time,
        }
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
        data = search_service.expand_query(query, payload.get("top_k", 5))
        return api_success(data, **data)

    def _compare_payload_response():
        payload = request.get_json(silent=True) or {}
        query = (payload.get("query") or "").strip()
        if not query:
            return api_error("QUERY_REQUIRED", "Query is required", 400)
        data, meta = retrieval_orchestrator.compare(
            query=query,
            models=payload.get("models") or ["bm25", "tfidf", "hybrid", "lm"],
            top_k=payload.get("top_k", 10),
            operator=payload.get("operator", "AND"),
            filters=payload.get("filters") or None,
        )
        search_log_service.log_event(
            request.path,
            payload,
            data,
            meta["execution_time"],
        )
        feedback_service.log_search_event(
            request.path,
            payload,
            data,
            meta["execution_time"],
            document_service.dataset_hash,
            _session_id(),
        )
        return api_success(
            data,
            meta,
            query=data["query"],
            models=data["models"],
            comparisons=data["comparisons"],
            timings=data["timings"],
            comparison=data["comparison"],
            query_analysis=data["query_analysis"],
        )

    @app.post("/api/search/compare")
    def compare_search_models():
        return _compare_payload_response()

    @app.post("/api/compare")
    def compare_models():
        return _compare_payload_response()

    @app.post("/api/evaluate")
    def evaluate():
        payload = request.get_json(silent=True) or {}
        data, meta = evaluation_service.evaluate(payload)
        search_log_service.log_event(
            "/api/evaluate",
            payload,
            data,
            meta["execution_time"],
        )
        feedback_service.log_search_event(
            "/api/evaluate",
            payload,
            data,
            meta["execution_time"],
            document_service.dataset_hash,
            _session_id(),
        )
        return api_success(data, meta, **data)

    @app.post("/api/evaluate/jobs")
    def start_evaluation_job():
        payload = request.get_json(silent=True) or {}
        data = evaluation_job_service.submit(payload)
        status_code = 200 if data["status"] == "completed" else 202
        return api_success(data, {"status": data["status"]}), status_code

    @app.get("/api/evaluate/jobs/<job_id>")
    def evaluation_job(job_id: str):
        data = evaluation_job_service.get(job_id)
        if not data:
            return api_error("JOB_NOT_FOUND", f"Evaluation job not found: {job_id}", 404)
        return api_success(data, {"status": data["status"]})

    @app.get("/api/evaluation/query_sets")
    def evaluation_query_sets():
        data = {
            "query_sets": evaluation_service.query_sets(),
            "default_query_set": evaluation_service.default_query_set_id(),
        }
        return api_success(data, **data)

    @app.post("/api/feedback")
    def feedback():
        payload = request.get_json(silent=True) or {}
        try:
            data = feedback_service.record_feedback(payload, _session_id())
        except (TypeError, ValueError) as exc:
            return api_error("INVALID_FEEDBACK", str(exc), 400)
        return api_success(data, **data)

    @app.get("/api/feedback/stats")
    def feedback_stats():
        data = {"stats": feedback_service.stats()}
        return api_success(data, stats=data["stats"])

    @app.get("/api/feedback/analytics")
    def feedback_analytics():
        days = request.args.get("days", 30)
        limit = request.args.get("limit", 20)
        data = feedback_analytics_service.analytics(int(days), int(limit))
        return api_success(data, **data)

    @app.get("/api/feedback/features")
    def feedback_features():
        limit = request.args.get("limit", 200)
        data = ltr_feature_service.features(int(limit))
        return api_success(data, **data)

    @app.post("/api/ltr/train")
    def train_ltr_demo():
        payload = request.get_json(silent=True) or {}
        data = ltr_training_service.train(payload.get("limit", 500))
        if data.get("code") == "SKLEARN_UNAVAILABLE":
            return api_error(
                "FEATURE_UNAVAILABLE",
                data.get("reason", "scikit-learn is unavailable"),
                503,
            )
        return api_success(data, **data)

    @app.post("/api/diagnostics/ranking")
    def ranking_diagnostics():
        payload = request.get_json(silent=True) or {}
        query = (payload.get("query") or "").strip()
        doc_id = payload.get("doc_id")
        if doc_id is None:
            doc_id = payload.get("article_id")
        if not query:
            return api_error("QUERY_REQUIRED", "Query is required", 400)
        if doc_id is None:
            return api_error("DOC_ID_REQUIRED", "doc_id is required", 400)
        try:
            data = ranking_diagnostics_service.explain(
                query,
                doc_id,
                payload.get("models") or ["bm25", "tfidf", "lm"],
            )
        except LookupError as exc:
            return api_error("DOCUMENT_NOT_FOUND", str(exc), 404)
        except ValueError as exc:
            return api_error("INVALID_DIAGNOSTICS_REQUEST", str(exc), 400)
        return api_success(data, **data)

    @app.get("/api/algorithms")
    def algorithms():
        data = {"models": retrieval_orchestrator.supported_models()}
        return api_success(data, models=data["models"])

    @app.post("/api/export")
    def export_results():
        payload = request.get_json(silent=True) or {}
        data = {
            "results": payload.get("results") or [],
            "format": payload.get("format", "json"),
        }
        return api_success(data, **data)

    def _truthy_arg(name: str, default: bool) -> bool:
        """Parse a boolean query argument.

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        value = request.args.get(name)
        if value is None:
            return default
        return value.lower() in {"1", "true", "yes", "on"}

    def _session_id() -> str | None:
        """Read the optional frontend session identifier.

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        return request.headers.get("X-IR-Session") or request.headers.get("X-Session-ID")


def run() -> None:
    """Run the development server.

    Complexity:
        Time: O(n) startup
        Space: O(n)
    """
    settings = Settings.from_env()
    app = create_app(settings)
    app.run(host=settings.host, port=settings.port, debug=settings.debug)
