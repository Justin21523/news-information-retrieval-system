"""Smoke tests for the Flask IR application layer."""

import time

from src.ir_app import create_app
from src.ir_app.config import Settings
from src.ir_app.services.search_log_service import SearchLogService


def make_test_app(tmp_path):
    """Build a Flask app using the mini fallback dataset.

    Complexity:
        Time: O(n) where n is mini dataset size
        Space: O(n)
    """
    settings = Settings.from_env()
    settings = Settings(
        project_root=settings.project_root,
        dataset_path=tmp_path / "missing.jsonl",
        fallback_dataset_path=settings.project_root
        / "datasets"
        / "mini"
        / "ir_documents.json",
        index_dir=tmp_path / "indexes",
        tokenizer_engine="jieba",
        enable_heavy_models=False,
        max_documents=None,
        host="127.0.0.1",
        port=5001,
        debug=False,
    )
    app = create_app(settings)
    app.config.update(TESTING=True)
    return app


def test_app_stats_uses_structured_schema(tmp_path):
    """Stats endpoint returns the new schema and legacy compatibility fields."""
    client = make_test_app(tmp_path).test_client()

    response = client.get("/api/stats")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["ok"] is True
    assert payload["success"] is True
    assert payload["data"]["stats"]["total_documents"] > 0
    assert (
        payload["stats"]["total_documents"]
        == payload["data"]["stats"]["total_documents"]
    )
    assert "validation" in payload["data"]["stats"]
    assert "index" in payload["data"]["stats"]
    assert "dataset_limit" in payload["data"]["stats"]


def test_guide_page_exposes_demo_walkthrough(tmp_path):
    """Guide page renders the user-facing demo workflow."""
    client = make_test_app(tmp_path).test_client()

    response = client.get("/guide")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "CNIRS Demo Guide" in html
    assert "Operation Flow" in html
    assert "Start Guided Demo" in html
    assert "demo-assistant.js" in html
    assert "run=1" in html


def test_search_requires_query(tmp_path):
    """Search rejects empty queries with structured errors."""
    client = make_test_app(tmp_path).test_client()

    response = client.post("/api/search", json={"query": ""})
    payload = response.get_json()

    assert response.status_code == 400
    assert payload["ok"] is False
    assert payload["error"]["code"] == "QUERY_REQUIRED"


def test_bm25_search_returns_results(tmp_path):
    """BM25 can search the mini fallback corpus."""
    client = make_test_app(tmp_path).test_client()

    response = client.post(
        "/api/search", json={"query": "information retrieval", "model": "bm25"}
    )
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["ok"] is True
    assert payload["data"]["results"]
    first = payload["data"]["results"][0]
    assert {
        "doc_id",
        "title",
        "snippet",
        "highlighted_snippet",
        "score",
        "model",
    } <= set(first)
    assert payload["results"] == payload["data"]["results"]
    assert "ranking_features" in first["explanation"]


def test_supported_search_models_return_stable_schema(tmp_path):
    """Formal lexical, hybrid, LM, fuzzy, and CSoundex models share one result schema."""
    client = make_test_app(tmp_path).test_client()

    for model in [
        "tfidf",
        "boolean",
        "hybrid",
        "lm",
        "bim",
        "wand_bm25",
        "maxscore_bm25",
        "fuzzy",
        "csoundex",
    ]:
        response = client.post(
            "/api/search",
            json={"query": "information retrieval", "model": model, "operator": "OR"},
        )
        payload = response.get_json()

        assert response.status_code == 200
        assert payload["ok"] is True
        assert payload["data"]["model_info"]["id"] == model
        assert "query_analysis" in payload["data"]
        assert "significant_terms" in payload["data"]["query_analysis"]
        assert payload["meta"]["execution_time"] >= 0
        assert "results" in payload["data"]
        if payload["data"]["results"]:
            result = payload["data"]["results"][0]
            assert "explanation" in result
            assert "component_scores" in result["explanation"]
            assert "field_boost" in result["explanation"]["component_scores"]
            assert "field_boost" in result["explanation"]["ranking_features"]
            assert "snippet_source" in result["explanation"]["ranking_features"]
            if model == "bim":
                assert "bim" in result["explanation"]["ranking_features"]
            if model in {"wand_bm25", "maxscore_bm25"}:
                optimization = result["explanation"]["ranking_features"]["optimization"]
                assert optimization["num_candidate_docs"] >= optimization["num_scored_docs"]


def test_document_endpoint_returns_document(tmp_path):
    """Document endpoint supports numeric doc IDs."""
    client = make_test_app(tmp_path).test_client()

    response = client.get("/api/document/0")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["ok"] is True
    assert payload["data"]["document"]["doc_id"] == 0
    assert payload["document"]["title"]
    assert "summary" in payload["data"]
    assert "keywords" in payload["data"]
    assert "related_documents" in payload["data"]
    assert "taxonomy" in payload["data"]
    assert "topic" in payload["data"]
    assert "explanation" in payload["data"]


def test_document_endpoint_returns_kwic_for_query(tmp_path):
    """Document details include stable KWIC payloads when query is provided."""
    client = make_test_app(tmp_path).test_client()

    response = client.get(
        "/api/document/0?query=information%20retrieval&include_kwic=true"
    )
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["ok"] is True
    kwic = payload["data"]["kwic"]
    assert kwic["available"] is True
    assert kwic["query"] == "information retrieval"
    assert "matches" in kwic
    if kwic["matches"]:
        assert {"keyword", "position", "plain_snippet", "highlighted_snippet"} <= set(
            kwic["matches"][0]
        )


def test_document_endpoint_related_documents_exclude_self(tmp_path):
    """Related documents are returned with explanations and never include the source doc."""
    client = make_test_app(tmp_path).test_client()

    response = client.get("/api/document/0?include_related=true&top_k=5")
    payload = response.get_json()

    assert response.status_code == 200
    related = payload["data"]["related_documents"]
    assert related
    assert all(str(item["doc_id"]) != "0" for item in related)
    first = related[0]
    assert "similarity" in first
    assert "relation_reason" in first
    assert "component_scores" in first["explanation"]


def test_document_endpoint_missing_doc_returns_structured_error(tmp_path):
    """Missing document details return the standard structured error."""
    client = make_test_app(tmp_path).test_client()

    response = client.get("/api/document/missing-doc")
    payload = response.get_json()

    assert response.status_code == 404
    assert payload["ok"] is False
    assert payload["error"]["code"] == "DOCUMENT_NOT_FOUND"


def test_bert_search_is_structured_unavailable(tmp_path):
    """Disabled heavy semantic search returns a structured unavailable error."""
    client = make_test_app(tmp_path).test_client()

    response = client.post("/api/search", json={"query": "retrieval", "model": "bert"})
    payload = response.get_json()

    assert response.status_code == 503
    assert payload["ok"] is False
    assert payload["error"]["code"] == "FEATURE_UNAVAILABLE"


def test_expand_query_uses_rocchio_schema(tmp_path):
    """Query expansion endpoint returns Rocchio-style metadata."""
    client = make_test_app(tmp_path).test_client()

    response = client.post("/api/expand_query", json={"query": "information retrieval"})
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["ok"] is True
    assert payload["data"]["method"] == "rocchio_prf"
    assert "expanded_terms" in payload["data"]


def test_index_cache_reused_for_same_dataset(tmp_path):
    """Index cache manifest is reused when dataset settings are unchanged."""
    app1 = make_test_app(tmp_path)
    first_index = app1.config["SEARCH_SERVICE"].index

    app2 = make_test_app(tmp_path)
    second_index = app2.config["SEARCH_SERVICE"].index

    assert first_index.cache_used is False
    assert second_index.cache_used is True
    assert second_index.manifest["document_count"] > 0


def test_all_facets_exposes_taxonomy_metadata(tmp_path):
    """Facet API exposes taxonomy, content type, and corpus distribution."""
    client = make_test_app(tmp_path).test_client()

    response = client.get("/api/all_facets")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["ok"] is True
    facets = payload["data"]["facets"]
    assert "taxonomy_topic" in facets
    assert "content_type" in facets
    assert facets["taxonomy_topic"]["field_type"] == "taxonomy"
    assert "coverage" in facets["taxonomy_topic"]
    assert "quality" in facets["taxonomy_topic"]
    assert facets["taxonomy_topic"]["quality"]["coverage"] > 0
    assert payload["data"]["corpus_distribution"]["content_type"]


def test_all_facets_hides_invalid_dates_and_noisy_values(tmp_path):
    """Facet API keeps UI filters focused on usable metadata values."""
    client = make_test_app(tmp_path).test_client()

    response = client.get("/api/all_facets")
    payload = response.get_json()
    facets = payload["data"]["facets"]

    assert response.status_code == 200
    assert "0001" not in {item["value"] for item in facets.get("published_year", {}).get("values", [])}
    assert "unknown" not in {item["value"] for item in facets.get("category", {}).get("values", [])}
    for facet in facets.values():
        assert "quality" in facet


def test_corpus_dashboard_page_exists(tmp_path):
    """Corpus dashboard route renders the portfolio-facing audit page."""
    client = make_test_app(tmp_path).test_client()

    response = client.get("/corpus")

    assert response.status_code == 200
    assert b"Corpus Dashboard" in response.data


def test_corpus_audit_endpoint_reports_readiness_and_metadata(tmp_path):
    """Corpus audit summarizes data quality, distributions, and index cache state."""
    client = make_test_app(tmp_path).test_client()

    response = client.get("/api/corpus/audit")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["ok"] is True
    data = payload["data"]
    assert data["summary"]["total_documents"] > 0
    assert data["summary"]["dataset_source"]
    assert data["metadata_completeness"]["fields"]
    fields = {item["field"]: item for item in data["metadata_completeness"]["fields"]}
    assert fields["title"]["coverage"] == 1.0
    assert fields["content"]["coverage"] == 1.0
    assert data["distributions"]["source"]
    assert data["distributions"]["taxonomy_topic"]
    assert data["index"]["manifest"]["document_count"] == data["summary"]["total_documents"]
    assert "cache_files" in data["index"]
    assert data["readiness"]["checks"]
    assert data["external_sources"][0]["project"] == "news-web-crawler"
    assert data["flow"][0]["id"] == "crawl"
    assert payload["meta"]["execution_time"] >= 0


def test_faceted_search_filters_metadata(tmp_path):
    """Faceted search applies raw metadata filters with stable schema."""
    client = make_test_app(tmp_path).test_client()

    response = client.post(
        "/api/search/faceted",
        json={
            "query": "retrieval",
            "model": "bm25",
            "filters": {"category": ["models"], "content_type": ["news_article"]},
        },
    )
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["ok"] is True
    assert payload["data"]["results"]
    assert all(result["category"] == "models" for result in payload["data"]["results"])
    assert all(
        result["content_type"] == "news_article"
        for result in payload["data"]["results"]
    )


def test_facets_for_query_use_candidate_set_and_filters(tmp_path):
    """Facet counts can be requested for a query and active filters."""
    client = make_test_app(tmp_path).test_client()

    response = client.post(
        "/api/facets",
        json={
            "query": "retrieval",
            "model": "bm25",
            "filters": {"category": ["models"]},
        },
    )
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["ok"] is True
    assert payload["data"]["total_documents"] >= 1
    assert "category" in payload["data"]["facets"]
    category_values = payload["data"]["facets"]["category"]["values"]
    assert any(
        value["value"] == "models" and value["selected"] for value in category_values
    )


def test_lm_search_returns_language_model_explanation(tmp_path):
    """Language Model retrieval is available through the unified search API."""
    client = make_test_app(tmp_path).test_client()

    response = client.post(
        "/api/search", json={"query": "information retrieval", "model": "lm"}
    )
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["ok"] is True
    assert payload["data"]["model"] == "lm"
    assert payload["data"]["results"]
    first = payload["data"]["results"][0]
    assert "lm" in first["explanation"]["component_scores"]
    assert "lm" in first["explanation"]["ranking_features"]


def test_compare_endpoint_returns_unified_model_payloads(tmp_path):
    """Model comparison exposes query analysis, timings, overlap, and explanations."""
    client = make_test_app(tmp_path).test_client()

    response = client.post(
        "/api/search/compare",
        json={
            "query": "information retrieval",
            "models": ["bm25", "tfidf", "hybrid", "lm", "bim", "wand_bm25", "maxscore_bm25"],
            "top_k": 5,
        },
    )
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["ok"] is True
    assert payload["data"]["query_analysis"]["query_terms"]
    assert set(payload["data"]["models"]) == {
        "bm25",
        "tfidf",
        "hybrid",
        "lm",
        "bim",
        "wand_bm25",
        "maxscore_bm25",
    }
    assert "overlap" in payload["data"]["comparison"]
    assert "rank_changes" in payload["data"]["comparison"]
    for model_data in payload["data"]["models"].values():
        assert model_data["available"] is True
        assert model_data["execution_time"] >= 0
        assert "model_info" in model_data
        if model_data["results"]:
            assert "component_scores" in model_data["results"][0]["explanation"]


def test_compare_endpoint_applies_facet_filters_to_all_models(tmp_path):
    """Model comparison uses the same facet filters as regular search."""
    client = make_test_app(tmp_path).test_client()

    response = client.post(
        "/api/search/compare",
        json={
            "query": "retrieval",
            "models": ["bm25", "tfidf", "hybrid", "lm"],
            "filters": {"category": ["models"], "content_type": ["news_article"]},
            "top_k": 5,
        },
    )
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["ok"] is True
    for model_data in payload["data"]["models"].values():
        assert model_data["available"] is True
        assert model_data["execution_time"] >= 0
        for result in model_data["results"]:
            assert result["category"] == "models"
            assert result["content_type"] == "news_article"


def test_legacy_compare_alias_keeps_frontend_fields(tmp_path):
    """Legacy /api/compare keeps fields consumed by the existing compare page."""
    client = make_test_app(tmp_path).test_client()

    response = client.post(
        "/api/compare",
        json={"query": "information retrieval", "models": ["bm25", "lm"], "top_k": 3},
    )
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["ok"] is True
    assert "models" in payload
    assert "comparisons" in payload
    assert "timings" in payload
    assert set(payload["comparisons"]) == {"bm25", "lm"}


def test_algorithms_endpoint_exposes_lm_capabilities(tmp_path):
    """Algorithm discovery returns model capabilities used by the UI."""
    client = make_test_app(tmp_path).test_client()

    response = client.get("/api/algorithms")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["ok"] is True
    models = {item["id"]: item for item in payload["data"]["models"]}
    assert models["lm"]["available"] is True
    assert models["lm"]["supports_filters"] is True
    assert models["lm"]["supports_explanation"] is True
    assert models["lm"]["demo_status"] == "demo_ready"
    assert models["bert"]["available"] is False
    assert models["bert"]["demo_status"] == "optional_disabled"
    assert models["bim"]["demo_status"] == "demo_ready"
    assert models["wand_bm25"]["demo_status"] == "demo_ready"
    assert models["maxscore_bm25"]["demo_status"] == "demo_ready"
    assert models["clustering_topic"]["demo_status"] == "demo_ready"
    assert models["lda_bertopic"]["demo_status"] == "optional_disabled"


def test_cluster_endpoint_returns_topic_cards(tmp_path):
    """Cluster endpoint groups query results into lightweight topic cards."""
    client = make_test_app(tmp_path).test_client()

    response = client.post(
        "/api/cluster",
        json={
            "query": "information retrieval",
            "method": "kmeans",
            "n_clusters": 2,
            "sample_size": 10,
        },
    )
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["ok"] is True
    data = payload["data"]
    assert data["method"] == "kmeans"
    assert data["sample_size"] >= 2
    assert data["clusters"]
    assert data["topics"]
    first = data["clusters"][0]
    assert "keywords" in first
    assert "documents" in first
    assert payload["meta"]["execution_time"] >= 0


def test_topics_endpoint_validates_method(tmp_path):
    """Topic endpoint returns structured errors for unknown clustering methods."""
    client = make_test_app(tmp_path).test_client()

    response = client.post("/api/topics", json={"method": "bad-method"})
    payload = response.get_json()

    assert response.status_code == 400
    assert payload["ok"] is False
    assert payload["error"]["code"] == "INVALID_CLUSTER_REQUEST"


def test_evaluation_query_sets_endpoint_lists_demo_qrels(tmp_path):
    """Evaluation query set discovery exposes demo qrels metadata."""
    client = make_test_app(tmp_path).test_client()

    response = client.get("/api/evaluation/query_sets")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["ok"] is True
    query_sets = {item["id"]: item for item in payload["data"]["query_sets"]}
    assert "mini_ir" in query_sets
    assert payload["data"]["default_query_set"] == "mini_ir"
    assert query_sets["mini_ir"]["query_count"] >= 1


def test_evaluate_endpoint_computes_demo_metrics_and_breakdown(tmp_path):
    """Evaluation endpoint computes real metrics for multiple retrieval models."""
    client = make_test_app(tmp_path).test_client()

    response = client.post(
        "/api/evaluate",
        json={
            "query_set": "mini_ir",
            "models": ["bm25", "tfidf", "hybrid", "lm"],
            "top_k": 10,
            "k_values": [5, 10],
        },
    )
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["ok"] is True
    data = payload["data"]
    assert data["evaluation_type"] == "demo"
    assert "not full benchmark" in data["disclaimer"]
    assert data["qrels_coverage"]["coverage"] == 1.0
    assert set(data["results"]) == {"bm25", "tfidf", "hybrid", "lm"}
    assert set(data["timings"]) == {"bm25", "tfidf", "hybrid", "lm"}
    for model, metrics in data["results"].items():
        assert metrics["available"] is True
        assert 0.0 <= metrics["map"] <= 1.0
        assert metrics["precision_at_k"]
        assert metrics["recall_at_k"]
        assert metrics["ndcg_at_k"]
        assert data["timings"][model] >= 0
    first_query = data["per_query"]["M001"]
    assert "bm25" in first_query["models"]
    assert first_query["models"]["bm25"]["top_results"]
    assert payload["meta"]["execution_time"] >= 0


def test_evaluate_endpoint_uses_cache_on_repeated_request(tmp_path):
    """Repeated evaluation requests return a cache hit for the same payload."""
    client = make_test_app(tmp_path).test_client()
    request_payload = {
        "query_set": "mini_ir",
        "models": ["bm25"],
        "top_k": 5,
        "k_values": [5],
    }

    first = client.post("/api/evaluate", json=request_payload).get_json()
    second = client.post("/api/evaluate", json=request_payload).get_json()

    assert first["ok"] is True
    assert second["ok"] is True
    assert first["data"]["cache_key"] == second["data"]["cache_key"]
    assert second["data"]["cached"] is True
    assert second["meta"]["cached"] is True


def test_evaluation_job_endpoint_returns_completed_result(tmp_path):
    """Async evaluation jobs expose status and final result payloads."""
    client = make_test_app(tmp_path).test_client()

    response = client.post(
        "/api/evaluate/jobs",
        json={
            "query_set": "mini_ir",
            "models": ["bm25"],
            "top_k": 5,
            "k_values": [5],
            "force_refresh": True,
        },
    )
    payload = response.get_json()

    assert response.status_code in {200, 202}
    assert payload["ok"] is True
    job = payload["data"]
    if job["status"] != "completed":
        for _ in range(50):
            poll = client.get(f"/api/evaluate/jobs/{job['job_id']}").get_json()
            if poll["data"]["status"] == "completed":
                job = poll["data"]
                break
            time.sleep(0.02)
    assert job["status"] == "completed"
    assert job["result"]["results"]["bm25"]["available"] is True


def test_feedback_api_records_click_and_stats(tmp_path):
    """Feedback API stores click events and exposes aggregate stats."""
    client = make_test_app(tmp_path).test_client()

    response = client.post(
        "/api/feedback",
        json={
            "event_type": "click",
            "query": "information retrieval",
            "model": "bm25",
            "doc_id": 0,
            "rank": 1,
            "score": 1.23,
        },
        headers={"X-IR-Session": "test-session"},
    )
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["ok"] is True
    assert payload["data"]["event_type"] == "click"

    stats = client.get("/api/feedback/stats").get_json()
    assert stats["data"]["stats"]["total_clicks"] >= 1


def test_feedback_analytics_endpoint_summarizes_search_and_feedback(tmp_path):
    """Feedback analytics aggregates search logs, CTR, and zero-result queries."""
    client = make_test_app(tmp_path).test_client()

    client.post("/api/search", json={"query": "information retrieval", "model": "bm25"})
    client.post("/api/search", json={"query": "ai", "model": "bm25"})
    client.post(
        "/api/feedback",
        json={
            "event_type": "click",
            "query": "information retrieval",
            "model": "bm25",
            "doc_id": 0,
            "rank": 1,
            "score": 1.0,
        },
    )
    response = client.get("/api/feedback/analytics?days=365&limit=10")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["ok"] is True
    summary = payload["data"]["summary"]
    assert summary["total_searches"] >= 2
    assert summary["total_clicks"] >= 1
    assert summary["ctr"] > 0
    assert any(item["query"] == "ai" for item in payload["data"]["zero_result_queries"])
    assert payload["data"]["model_metrics"]
    assert "session_metrics" in payload["data"]
    assert "quality" in payload["data"]
    assert "position_bias" in payload["data"]


def test_feedback_features_endpoint_exports_ltr_rows(tmp_path):
    """LTR feature preview joins feedback rows with document metadata and diagnostics."""
    client = make_test_app(tmp_path).test_client()

    client.post(
        "/api/feedback",
        json={
            "event_type": "relevance",
            "query": "information retrieval",
            "model": "bm25",
            "doc_id": 0,
            "rank": 1,
            "score": 1.0,
            "relevance_grade": 3,
        },
    )
    response = client.get("/api/feedback/features?limit=10")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["ok"] is True
    assert payload["data"]["feature_set"] == "feedback_ltr_v1"
    assert payload["data"]["rows"]
    row = payload["data"]["rows"][0]
    assert row["query"] == "information retrieval"
    assert row["label"] == 1.0
    assert "field_boost" in row["features"]
    assert "bm25_score" in row["features"]
    assert row["metadata"]["title"]


def test_feedback_analytics_marks_duplicates_and_position_bias(tmp_path):
    """Raw feedback analytics count duplicates but expose quality warnings."""
    client = make_test_app(tmp_path).test_client()
    event = {
        "event_type": "click",
        "query": "information retrieval",
        "model": "bm25",
        "doc_id": 0,
        "rank": 1,
        "score": 1.0,
        "metadata": {"source": "playwright_demo"},
    }

    client.post("/api/feedback", json=event, headers={"X-IR-Session": "dup-session"})
    client.post("/api/feedback", json=event, headers={"X-IR-Session": "dup-session"})
    payload = client.get("/api/feedback/analytics?days=365&limit=10").get_json()

    quality = payload["data"]["quality"]
    assert quality["raw_metrics_count_all_events"] is True
    assert quality["duplicate_group_count"] >= 1
    assert quality["duplicate_event_count"] >= 1
    assert quality["demo_seed_event_count"] >= 2
    assert payload["data"]["position_bias"]["buckets"]["rank_1"]["clicks"] >= 2


def test_ltr_train_endpoint_trains_with_positive_and_negative_labels(tmp_path):
    """LTR sandbox trains a weak-supervision model when both classes exist."""
    client = make_test_app(tmp_path).test_client()

    for doc_id, grade in [(0, 3), (1, 0), (2, 3), (3, 0)]:
        client.post(
            "/api/feedback",
            json={
                "event_type": "relevance",
                "query": "information retrieval",
                "model": "bm25",
                "doc_id": doc_id,
                "rank": doc_id + 1,
                "score": 1.0,
                "relevance_grade": grade,
            },
        )
    response = client.post("/api/ltr/train", json={"limit": 50})
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["ok"] is True
    assert payload["data"]["trained"] is True
    assert payload["data"]["model"] == "sklearn_logistic_regression"
    assert payload["data"]["metrics"]["training_accuracy"] >= 0
    assert payload["data"]["coefficients"]
    assert payload["data"]["sample_predictions"]


def test_feedback_api_validates_relevance_grade(tmp_path):
    """Invalid relevance feedback returns a structured error."""
    client = make_test_app(tmp_path).test_client()

    response = client.post(
        "/api/feedback",
        json={"event_type": "relevance", "doc_id": 0, "relevance_grade": 5},
    )
    payload = response.get_json()

    assert response.status_code == 400
    assert payload["ok"] is False
    assert payload["error"]["code"] == "INVALID_FEEDBACK"


def test_ranking_diagnostics_endpoint_returns_term_breakdown(tmp_path):
    """Ranking diagnostics expose BM25, TF-IDF, and LM term contribution rows."""
    client = make_test_app(tmp_path).test_client()

    response = client.post(
        "/api/diagnostics/ranking",
        json={
            "query": "information retrieval",
            "doc_id": 0,
            "models": ["bm25", "tfidf", "lm"],
        },
    )
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["ok"] is True
    data = payload["data"]
    assert data["query_terms"]
    assert data["document"]["doc_id"] == 0
    assert set(data["models"]) == {"bm25", "tfidf", "lm"}
    assert data["query_coverage"]["coverage_ratio"] >= 0
    assert "title" in data["field_contributions"]["fields"]
    assert data["field_match_matrix"]
    assert data["models"]["bm25"]["terms"]
    assert data["models"]["tfidf"]["terms"]
    assert data["models"]["lm"]["terms"]


def test_search_log_service_writes_jsonl_event(tmp_path):
    """Search logs capture query, filters, latency, and top results."""
    log_path = tmp_path / "logs" / "search_logs.jsonl"
    service = SearchLogService(tmp_path, log_path)

    service.log_event(
        "/api/search",
        {"query": "retrieval", "model": "bm25", "filters": {"category": ["models"]}},
        {"results": [{"doc_id": 1, "article_id": "doc-1"}]},
        0.012,
    )

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert '"query": "retrieval"' in lines[0]
    assert '"doc-1"' in lines[0]


def test_query_analysis_removes_news_boilerplate_terms(tmp_path):
    """News boilerplate terms are exposed separately from significant terms."""
    client = make_test_app(tmp_path).test_client()

    response = client.post(
        "/api/search",
        json={"query": "中央社 information retrieval 報導", "model": "bm25"},
    )
    payload = response.get_json()

    assert response.status_code == 200
    analysis = payload["data"]["query_analysis"]
    assert "中央社" in analysis["removed_stopwords"]
    assert "報導" in analysis["removed_stopwords"]
    assert "information" in analysis["significant_terms"]
    assert "retrieval" in analysis["significant_terms"]


def test_no_result_search_returns_suggestions(tmp_path):
    """No-result searches return structured suggestions when variants exist."""
    client = make_test_app(tmp_path).test_client()

    response = client.post("/api/search", json={"query": "ai", "model": "bm25"})
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["ok"] is True
    assert payload["data"]["results"] == []
    assert payload["data"]["suggestions"]
    assert any(
        suggestion["type"] == "synonym" for suggestion in payload["data"]["suggestions"]
    )
