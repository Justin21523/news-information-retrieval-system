#!/usr/bin/env python3
"""Smoke-check the CNIRS portfolio demo locally or through a public URL."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ROOT_MARKER = "CNIRS"
CHINESE_DEMO_QUERY = "\u534a\u5c0e\u9ad4 \u4eba\u5de5\u667a\u6167"
ENGLISH_DEMO_QUERY = "information retrieval"


@dataclass
class SmokeResponse:
    """Small transport-neutral response container."""

    status: int
    text: str
    json_data: dict[str, Any] | None = None


class SmokeFailure(RuntimeError):
    """Raised when a smoke check fails."""


class LocalClient:
    """Flask test-client transport."""

    def __init__(self) -> None:
        os.environ.setdefault("IR_ENABLE_HEAVY_MODELS", "false")
        os.environ.setdefault("IR_TOKENIZER_ENGINE", "jieba")
        from src.ir_app.app_factory import create_app

        self.app = create_app()
        self.client = self.app.test_client()

    def get(self, path: str) -> SmokeResponse:
        response = self.client.get(path)
        return make_flask_response(response)

    def post(self, path: str, payload: dict[str, Any]) -> SmokeResponse:
        response = self.client.post(path, json=payload)
        return make_flask_response(response)


class RemoteClient:
    """HTTP transport for deployed demos."""

    def __init__(self, base_url: str, timeout: float = 90.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def get(self, path: str) -> SmokeResponse:
        return self._request("GET", path)

    def post(self, path: str, payload: dict[str, Any]) -> SmokeResponse:
        return self._request("POST", path, payload)

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> SmokeResponse:
        url = self.base_url + path
        body = None
        headers = {"Accept": "application/json,text/html"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(url, data=body, headers=headers, method=method)
        try:
            with urlopen(  # noqa: S310
                request,
                timeout=self.timeout,
            ) as response:
                text = response.read().decode("utf-8", errors="replace")
                return make_http_response(response.status, text)
        except HTTPError as exc:
            text = exc.read().decode("utf-8", errors="replace")
            return make_http_response(exc.code, text)
        except TimeoutError as exc:
            raise SmokeFailure(
                f"{method} {url} timed out after {self.timeout}s"
            ) from exc


def make_flask_response(response: Any) -> SmokeResponse:
    """Convert a Flask response into the neutral response container.

    Complexity:
        Time: O(n) for response size
        Space: O(n)
    """
    text = response.get_data(as_text=True)
    json_data = response.get_json(silent=True)
    return SmokeResponse(response.status_code, text, json_data)


def make_http_response(status: int, text: str) -> SmokeResponse:
    """Convert an HTTP response into the neutral response container.

    Complexity:
        Time: O(n) for JSON parsing
        Space: O(n)
    """
    json_data: dict[str, Any] | None = None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            json_data = parsed
    except json.JSONDecodeError:
        json_data = None
    return SmokeResponse(status, text, json_data)


def require(condition: bool, message: str) -> None:
    """Raise a smoke failure when a condition is false.

    Complexity:
        Time: O(1)
        Space: O(1)
    """
    if not condition:
        raise SmokeFailure(message)


def require_ok(
    response: SmokeResponse,
    label: str,
    max_status: int = 399,
) -> None:
    """Assert an HTTP response status is acceptable.

    Complexity:
        Time: O(1)
        Space: O(1)
    """
    require(
        response.status <= max_status,
        f"{label} returned HTTP {response.status}: {response.text[:240]}",
    )


def require_json_ok(response: SmokeResponse, label: str) -> dict[str, Any]:
    """Assert an API response has the project success envelope.

    Complexity:
        Time: O(1)
        Space: O(1)
    """
    require_ok(response, label)
    require(response.json_data is not None, f"{label} did not return JSON")
    payload = response.json_data
    assert payload is not None
    require(payload.get("ok") is True, f"{label} returned ok=false")
    return payload


def run_smoke(client: LocalClient | RemoteClient) -> None:
    """Run the complete demo smoke suite.

    Complexity:
        Time: O(checks * request latency)
        Space: O(response size)
    """
    started = time.perf_counter()
    check_pages(client)
    stats = check_stats(client)
    query, first_doc_id = check_search(client, stats)
    check_compare(client, query)
    check_facets(client)
    check_evaluation(client)
    check_diagnostics(client, query, first_doc_id)
    check_analysis_graph(client, query)
    check_heavy_model_fallback(client)
    elapsed = time.perf_counter() - started
    print(f"PASS CNIRS smoke checks completed in {elapsed:.2f}s")


def check_pages(client: LocalClient | RemoteClient) -> None:
    """Verify demo pages render the expected assets.

    Complexity:
        Time: O(pages)
        Space: O(1)
    """
    pages = {
        "/": ROOT_MARKER,
        "/guide": "demo-assistant.js",
        "/compare": "compare.js",
        "/corpus": "corpus.js",
        "/evaluation": "evaluation.js",
        "/diagnostics": "diagnostics.js",
        "/feedback": "feedback-analytics.js",
        "/analysis-graph": "analysis-graph.js",
    }
    for path, marker in pages.items():
        response = client.get(path)
        require_ok(response, f"GET {path}")
        require(
            marker in response.text,
            f"GET {path} missing marker {marker!r}",
        )


def check_stats(client: LocalClient | RemoteClient) -> dict[str, Any]:
    """Verify corpus and index stats.

    Complexity:
        Time: O(1)
        Space: O(1)
    """
    payload = require_json_ok(client.get("/api/stats"), "GET /api/stats")
    stats = payload["data"]["stats"]
    require(
        stats["total_documents"] > 0,
        "stats total_documents must be positive",
    )
    require(stats["total_terms"] > 0, "stats total_terms must be positive")
    require(
        stats["heavy_models_enabled"] is False,
        "heavy models must be disabled",
    )
    return stats


def check_search(
    client: LocalClient | RemoteClient,
    stats: dict[str, Any],
) -> tuple[str, int | str]:
    """Verify search returns results and explanations.

    Complexity:
        Time: O(search attempts)
        Space: O(result size)
    """
    queries = [CHINESE_DEMO_QUERY, ENGLISH_DEMO_QUERY]
    if stats["total_documents"] < 100:
        queries.reverse()
    for query in queries:
        response = client.post(
            "/api/search",
            {"query": query, "model": "hybrid", "top_k": 5},
        )
        payload = require_json_ok(response, "POST /api/search")
        results = payload["data"]["results"]
        if results:
            first = results[0]
            require(
                "explanation" in first,
                "search result missing explanation",
            )
            require(
                "component_scores" in first["explanation"],
                "search explanation missing component scores",
            )
            return query, first["doc_id"]
    raise SmokeFailure("POST /api/search returned no results for demo queries")


def check_compare(client: LocalClient | RemoteClient, query: str) -> None:
    """Verify model comparison returns multiple model payloads.

    Complexity:
        Time: O(models)
        Space: O(result size)
    """
    payload = require_json_ok(
        client.post(
            "/api/search/compare",
            {
                "query": query,
                "models": ["bm25", "tfidf", "hybrid", "lm"],
                "top_k": 5,
            },
        ),
        "POST /api/search/compare",
    )
    models = payload["data"]["models"]
    require(
        set(models) == {"bm25", "tfidf", "hybrid", "lm"},
        "compare model set mismatch",
    )
    require(
        "overlap" in payload["data"]["comparison"],
        "compare missing overlap",
    )


def check_facets(client: LocalClient | RemoteClient) -> None:
    """Verify facet discovery and browse mode.

    Complexity:
        Time: O(facets)
        Space: O(facet payload)
    """
    payload = require_json_ok(
        client.get("/api/all_facets"),
        "GET /api/all_facets",
    )
    facets = payload["data"]["facets"]
    require("taxonomy_topic" in facets, "taxonomy_topic facet missing")
    selected_field = None
    selected_value = None
    for field, facet in facets.items():
        values = facet.get("values") or []
        if values:
            selected_field = field
            selected_value = values[0]["value"]
            break
    require(selected_field is not None, "no facet values available")
    browse = require_json_ok(
        client.post(
            "/api/search/browse",
            {"filters": {selected_field: [selected_value]}, "top_k": 3},
        ),
        "POST /api/search/browse",
    )
    require(browse["data"]["browse_mode"] is True, "browse mode flag missing")


def check_evaluation(client: LocalClient | RemoteClient) -> None:
    """Verify demo qrels evaluation path.

    Complexity:
        Time: O(models * queries)
        Space: O(metrics)
    """
    payload = require_json_ok(
        client.post(
            "/api/evaluate",
            {
                "query_set": "mini_ir",
                "models": ["bm25", "tfidf", "hybrid", "lm"],
                "top_k": 5,
                "k_values": [5],
            },
        ),
        "POST /api/evaluate",
    )
    data = payload["data"]
    require(data["evaluation_type"] == "demo", "evaluation type mismatch")
    require(
        set(data["results"]) == {"bm25", "tfidf", "hybrid", "lm"},
        "missing eval models",
    )


def check_diagnostics(
    client: LocalClient | RemoteClient,
    query: str,
    doc_id: int | str,
) -> None:
    """Verify ranking diagnostics for a retrieved document.

    Complexity:
        Time: O(models * query terms)
        Space: O(term rows)
    """
    payload = require_json_ok(
        client.post(
            "/api/diagnostics/ranking",
            {
                "query": query,
                "doc_id": doc_id,
                "models": ["bm25", "tfidf", "lm"],
            },
        ),
        "POST /api/diagnostics/ranking",
    )
    data = payload["data"]
    require(data["query_terms"], "diagnostics missing query terms")
    require(
        set(data["models"]) == {"bm25", "tfidf", "lm"},
        "diagnostics model mismatch",
    )


def check_analysis_graph(
    client: LocalClient | RemoteClient,
    query: str,
) -> None:
    """Verify analysis graph nodes and edges.

    Complexity:
        Time: O(top_k * models)
        Space: O(nodes + edges)
    """
    graph_query = {
        "query": query,
        "models": "bm25,tfidf",
        "top_k": 3,
    }
    query_string = urlencode(graph_query)
    payload = require_json_ok(
        client.get(f"/api/analysis/graph?{query_string}"),
        "GET /api/analysis/graph",
    )
    data = payload["data"]
    require(data["nodes"], "analysis graph missing nodes")
    require(data["edges"], "analysis graph missing edges")


def check_heavy_model_fallback(client: LocalClient | RemoteClient) -> None:
    """Verify optional semantic search fails safely.

    Complexity:
        Time: O(1)
        Space: O(1)
    """
    response = client.post(
        "/api/search",
        {"query": ENGLISH_DEMO_QUERY, "model": "bert", "top_k": 3},
    )
    require(
        response.status == 503,
        f"BERT fallback expected 503, got {response.status}",
    )
    require(
        response.json_data is not None,
        "BERT fallback did not return JSON",
    )
    payload = response.json_data
    assert payload is not None
    error_code = payload.get("error", {}).get("code")
    require(
        error_code == "FEATURE_UNAVAILABLE",
        "BERT fallback error code mismatch",
    )


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse CLI arguments.

    Complexity:
        Time: O(1)
        Space: O(1)
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url",
        help="Public demo base URL. Omit to use the local Flask test client.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=90.0,
        help="Per-request timeout in seconds for remote URL checks.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Complexity:
        Time: O(smoke checks)
        Space: O(response size)
    """
    args = parse_args(argv or sys.argv[1:])
    client: LocalClient | RemoteClient
    if args.base_url:
        client = RemoteClient(args.base_url, args.timeout)
    else:
        client = LocalClient()
    try:
        run_smoke(client)
    except SmokeFailure as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
