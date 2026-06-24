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
    """Formal lexical, hybrid, fuzzy, and CSoundex models share one result schema."""
    client = make_test_app(tmp_path).test_client()

    for model in ["tfidf", "boolean", "hybrid", "fuzzy", "csoundex"]:
        response = client.post(
            "/api/search",
            json={"query": "information retrieval", "model": model, "operator": "OR"},
        )
        payload = response.get_json()

        assert response.status_code == 200
        assert payload["ok"] is True
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
