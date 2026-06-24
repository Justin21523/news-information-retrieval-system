"""Smoke tests for the Flask IR application layer."""

from src.ir_app import create_app
from src.ir_app.config import Settings


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
        fallback_dataset_path=settings.project_root / "datasets" / "mini" / "ir_documents.json",
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
    assert payload["stats"]["total_documents"] == payload["data"]["stats"]["total_documents"]
    assert "validation" in payload["data"]["stats"]
    assert "index" in payload["data"]["stats"]
    assert "dataset_limit" in payload["data"]["stats"]


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

    response = client.post("/api/search", json={"query": "information retrieval", "model": "bm25"})
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["ok"] is True
    assert payload["data"]["results"]
    first = payload["data"]["results"][0]
    assert {"doc_id", "title", "snippet", "highlighted_snippet", "score", "model"} <= set(first)
    assert payload["results"] == payload["data"]["results"]
    assert "ranking_features" in first["explanation"]


def test_supported_search_models_return_stable_schema(tmp_path):
    """Formal lexical, hybrid, LM, fuzzy, and CSoundex models share one result schema."""
    client = make_test_app(tmp_path).test_client()

    for model in ["tfidf", "boolean", "hybrid", "lm", "fuzzy", "csoundex"]:
        response = client.post(
            "/api/search",
            json={"query": "information retrieval", "model": model, "operator": "OR"},
        )
        payload = response.get_json()

        assert response.status_code == 200
        assert payload["ok"] is True
        assert payload["data"]["model_info"]["id"] == model
        assert "query_analysis" in payload["data"]
        assert payload["meta"]["execution_time"] >= 0
        assert "results" in payload["data"]
        if payload["data"]["results"]:
            result = payload["data"]["results"][0]
            assert "explanation" in result
            assert "component_scores" in result["explanation"]


def test_document_endpoint_returns_document(tmp_path):
    """Document endpoint supports numeric doc IDs."""
    client = make_test_app(tmp_path).test_client()

    response = client.get("/api/document/0")
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["ok"] is True
    assert payload["data"]["document"]["doc_id"] == 0
    assert payload["document"]["title"]


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
    assert payload["data"]["corpus_distribution"]["content_type"]


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
    assert all(result["content_type"] == "news_article" for result in payload["data"]["results"])


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
    assert any(value["value"] == "models" and value["selected"] for value in category_values)


def test_lm_search_returns_language_model_explanation(tmp_path):
    """Language Model retrieval is available through the unified search API."""
    client = make_test_app(tmp_path).test_client()

    response = client.post("/api/search", json={"query": "information retrieval", "model": "lm"})
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
            "models": ["bm25", "tfidf", "hybrid", "lm"],
            "top_k": 5,
        },
    )
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["ok"] is True
    assert payload["data"]["query_analysis"]["query_terms"]
    assert set(payload["data"]["models"]) == {"bm25", "tfidf", "hybrid", "lm"}
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
    assert models["bert"]["available"] is False
