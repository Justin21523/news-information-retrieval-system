"""Verify the Flask UI with Playwright screenshots and video capture."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "docs" / "assets" / "evaluation"
VIEWPORT = {"width": 1440, "height": 1100}


def wait_for_server(base_url: str, timeout: float = 180.0) -> None:
    """Wait until the Flask server answers HTTP requests.

    Complexity:
        Time: O(t)
        Space: O(1)
    """
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urlopen(base_url, timeout=2) as response:  # noqa: S310
                if response.status < 500:
                    return
        except Exception as exc:  # pragma: no cover - diagnostic loop
            last_error = exc
        time.sleep(1)
    raise RuntimeError(f"Server did not become ready: {last_error}")


def start_server(port: int) -> subprocess.Popen:
    """Start the Flask demo server for UI verification.

    Complexity:
        Time: O(1)
        Space: O(1)
    """
    env = os.environ.copy()
    env.setdefault("IR_PORT", str(port))
    env.setdefault("IR_HOST", "127.0.0.1")
    env.setdefault("IR_ENABLE_HEAVY_MODELS", "false")
    return subprocess.Popen(  # noqa: S603
        [sys.executable, "app.py"],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def run_verification(base_url: str, asset_dir: Path) -> None:
    """Run key demo flows and write visual artifacts.

    Complexity:
        Time: O(pages)
        Space: O(1)
    """
    asset_dir.mkdir(parents=True, exist_ok=True)
    video_dir = asset_dir / "video_raw"
    video_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(
            viewport=VIEWPORT,
            record_video_dir=str(video_dir),
            record_video_size=VIEWPORT,
        )
        page = context.new_page()
        page.set_default_timeout(45000)

        page.goto(f"{base_url}/guide", wait_until="networkidle")
        page.wait_for_selector(".guide-page", timeout=45000)
        page.wait_for_selector("#demo-assistant-app", timeout=45000)
        page.wait_for_selector("#demo-assistant-coach", timeout=45000)
        capture_viewport(page, asset_dir / "demo-guide.png")
        capture_viewport(page, asset_dir / "demo-assistant-guide.png")

        page.goto(f"{base_url}/?q=半導體%20人工智慧&model=hybrid&run=1&tour=1&step=search", wait_until="networkidle")
        page.wait_for_selector("#demo-assistant-app", timeout=45000)
        page.wait_for_selector("#demo-assistant-coach", timeout=45000)
        page.wait_for_selector(".search-box.demo-assistant-highlight", timeout=45000)
        page.wait_for_selector(".result-item, .results-list", timeout=45000)
        capture_viewport(page, asset_dir / "demo-assistant-search.png")

        page.goto(f"{base_url}/?q=台灣%20經濟&model=bm25&taxonomy_topic=business&run=1&tour=1&step=facets", wait_until="networkidle")
        page.wait_for_selector("#demo-assistant-app", timeout=45000)
        page.wait_for_selector("#demo-assistant-coach", timeout=45000)
        page.wait_for_selector("#filter-sidebar.demo-assistant-highlight", timeout=45000)
        page.wait_for_selector(".result-item, .results-list", timeout=45000)
        capture_viewport(page, asset_dir / "demo-assistant-facets.png")

        page.goto(f"{base_url}/", wait_until="networkidle")
        fill_search(page, "半導體")
        seed_feedback(page)
        capture_viewport(page, asset_dir / "search-results.png")
        capture_scroll_sequence(page, asset_dir, "search-results", steps=2)

        run_facet_browse(page)
        capture_viewport(page, asset_dir / "facet-browse.png")
        capture_scroll_sequence(page, asset_dir, "facet-browse", steps=2)

        open_document_detail(page)
        capture_viewport(page, asset_dir / "document-detail.png")
        capture_scroll_sequence(page, asset_dir, "document-detail", steps=2)

        page.goto(f"{base_url}/compare", wait_until="networkidle")
        fill_compare(page, "人工智慧")
        capture_viewport(page, asset_dir / "model-compare.png")
        capture_scroll_sequence(page, asset_dir, "model-compare", steps=2)

        page.goto(f"{base_url}/compare?q=美國%20中國&models=bm25,tfidf,hybrid,lm,bim,wand_bm25,maxscore_bm25&run=1&tour=1&step=compare", wait_until="networkidle")
        page.wait_for_selector("#demo-assistant-app", timeout=45000)
        page.wait_for_selector("#demo-assistant-coach", timeout=45000)
        page.wait_for_selector("#comparison-container .model-results", timeout=45000)
        capture_viewport(page, asset_dir / "demo-assistant-compare.png")

        page.goto(f"{base_url}/corpus", wait_until="networkidle")
        page.wait_for_selector("#corpus-content", state="visible", timeout=45000)
        capture_viewport(page, asset_dir / "corpus-dashboard.png")
        capture_scroll_sequence(page, asset_dir, "corpus-dashboard", steps=2)
        run_topic_explorer(page)
        capture_viewport(page, asset_dir / "topic-explorer.png")
        capture_scroll_sequence(page, asset_dir, "topic-explorer", steps=2)

        page.goto(f"{base_url}/corpus?tour=1&step=topics&topic_query=半導體%20人工智慧&run_topic=1", wait_until="networkidle")
        page.wait_for_selector("#demo-assistant-app", timeout=45000)
        page.wait_for_selector("#demo-assistant-coach", timeout=45000)
        page.wait_for_selector(".topic-card", timeout=60000)
        capture_viewport(page, asset_dir / "demo-assistant-corpus.png")

        page.goto(f"{base_url}/evaluation", wait_until="networkidle")
        run_evaluation(page)
        capture_viewport(page, asset_dir / "evaluation-dashboard.png")
        capture_scroll_sequence(page, asset_dir, "evaluation-dashboard", steps=3)

        page.goto(f"{base_url}/diagnostics", wait_until="networkidle")
        run_diagnostics(page)
        capture_viewport(page, asset_dir / "ranking-diagnostics.png")
        capture_scroll_sequence(page, asset_dir, "ranking-diagnostics", steps=2)

        page.goto(f"{base_url}/analysis-graph?query=information%20retrieval&models=bm25,tfidf,hybrid,lm&top_k=6", wait_until="networkidle")
        run_analysis_graph(page)
        capture_viewport(page, asset_dir / "analysis-graph.png")

        page.goto(f"{base_url}/analysis-graph?query=台灣%20經濟&models=bm25,tfidf,hybrid,lm&top_k=6&tour=1&step=analysis", wait_until="networkidle")
        run_analysis_graph(page)
        page.wait_for_selector("#demo-assistant-app", timeout=45000)
        page.wait_for_selector("#demo-assistant-coach", timeout=45000)
        capture_viewport(page, asset_dir / "demo-assistant-analysis-graph.png")

        page.goto(f"{base_url}/feedback", wait_until="networkidle")
        run_feedback_dashboard(page)
        capture_viewport(page, asset_dir / "feedback-analytics.png")
        capture_scroll_sequence(page, asset_dir, "feedback-analytics", steps=3)

        page.goto(f"{base_url}/guide?tour=1&step=wrap", wait_until="networkidle")
        page.wait_for_selector("#demo-assistant-app", timeout=45000)
        page.wait_for_selector("#demo-assistant-coach", timeout=45000)
        capture_viewport(page, asset_dir / "demo-assistant-wrap.png")

        page.close()
        context.close()
        browser.close()

    videos = sorted(video_dir.glob("*.webm"), key=lambda path: path.stat().st_mtime)
    if videos:
        target = asset_dir / "cnirs-demo.webm"
        if target.exists():
            target.unlink()
        videos[-1].rename(target)
    for leftover in video_dir.glob("*"):
        leftover.unlink(missing_ok=True)
    video_dir.rmdir()


def capture_viewport(page, path: Path) -> None:
    """Capture only the visible browser viewport.

    Complexity:
        Time: O(viewport pixels)
        Space: O(viewport pixels)
    """
    page.screenshot(path=path, full_page=False)


def capture_scroll_sequence(page, asset_dir: Path, stem: str, steps: int = 2) -> None:
    """Capture long pages as multiple viewport-sized scroll positions.

    Complexity:
        Time: O(steps * viewport pixels)
        Space: O(viewport pixels)
    """
    page.evaluate("window.scrollTo(0, 0)")
    page.wait_for_timeout(250)
    dimensions = page.evaluate(
        """() => ({
            height: Math.max(document.body.scrollHeight, document.documentElement.scrollHeight),
            viewport: window.innerHeight
        })"""
    )
    scrollable = max(0, int(dimensions["height"]) - int(dimensions["viewport"]))
    if scrollable <= int(dimensions["viewport"]) * 0.35:
        return

    for index in range(2, steps + 2):
        ratio = (index - 1) / steps
        y = int(scrollable * ratio)
        page.evaluate("(scrollY) => window.scrollTo(0, scrollY)", y)
        page.wait_for_timeout(350)
        capture_viewport(page, asset_dir / f"{stem}-scroll-{index:02d}.png")

    page.evaluate("window.scrollTo(0, 0)")
    page.wait_for_timeout(200)


def fill_search(page, query: str) -> None:
    """Run the main search flow.

    Complexity:
        Time: O(search)
        Space: O(1)
    """
    search_input = page.locator("#query-input, #search-input, input[type='search'], input[name='query']").first
    search_input.fill(query)
    button = page.locator("#search-btn, button:has-text('Search'), button:has-text('搜尋')").first
    button.click()
    page.wait_for_timeout(1000)
    page.wait_for_selector(".result-item, .result-card, .search-result, #results-list", timeout=45000)


def open_document_detail(page) -> None:
    """Open a document detail modal when available.

    Complexity:
        Time: O(1)
        Space: O(1)
    """
    selectors = [
        "button:has-text('Why this result?')",
        "button:has-text('Details')",
        "a:has-text('Details')",
        ".result-title",
    ]
    for selector in selectors:
        locator = page.locator(selector).first
        try:
            if locator.count():
                locator.click()
                page.wait_for_timeout(1000)
                return
        except PlaywrightTimeoutError:
            continue


def run_facet_browse(page) -> None:
    """Browse documents directly from a metadata facet without a query.

    Complexity:
        Time: O(request)
        Space: O(1)
    """
    page.locator("#query-input").fill("")
    page.wait_for_selector("#facet-groups .facet-label", timeout=45000)
    page.locator("#facet-groups input[type='checkbox']").first.check(force=True)
    page.wait_for_selector(".result-item", timeout=45000)
    page.wait_for_selector("#result-count", timeout=45000)


def seed_feedback(page) -> None:
    """Write demo feedback events for analytics screenshots.

    Complexity:
        Time: O(1)
        Space: O(1)
    """
    page.evaluate(
        """
        async () => {
            const base = {
                query: '半導體',
                model: 'hybrid',
                doc_id: 1,
                rank: 1,
                score: 1.0,
                metadata: { source: 'playwright_demo' }
            };
            await fetch('api/feedback', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-IR-Session': 'playwright-demo' },
                body: JSON.stringify({ ...base, event_type: 'click' })
            });
            await fetch('api/feedback', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-IR-Session': 'playwright-demo' },
                body: JSON.stringify({ ...base, event_type: 'relevance', relevance_grade: 3 })
            });
            await fetch('api/feedback', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json', 'X-IR-Session': 'playwright-demo' },
                body: JSON.stringify({
                    ...base,
                    event_type: 'relevance',
                    doc_id: 2,
                    rank: 2,
                    relevance_grade: 0
                })
            });
        }
        """
    )


def fill_compare(page, query: str) -> None:
    """Run the model comparison flow.

    Complexity:
        Time: O(search)
        Space: O(1)
    """
    input_box = page.locator("#query-input, #compare-query, #search-query, input[type='text']").first
    input_box.fill(query)
    button = page.locator("#compare-btn, button:has-text('Compare'), button:has-text('比較')").first
    button.click()
    page.wait_for_timeout(1500)
    page.wait_for_selector(".model-results, .comparison-grid, #comparison-container", timeout=45000)


def run_evaluation(page) -> None:
    """Run the evaluation dashboard flow.

    Complexity:
        Time: O(evaluation)
        Space: O(1)
    """
    query_input = page.locator("#eval-query")
    query_input.fill("")
    page.locator("#eval-topk").fill("10")
    page.locator("#run-eval-btn").click()
    page.wait_for_selector("#eval-results", state="visible", timeout=180000)
    page.wait_for_selector("#comparison-table table", timeout=180000)


def run_topic_explorer(page) -> None:
    """Run the corpus topic explorer flow.

    Complexity:
        Time: O(clustering)
        Space: O(1)
    """
    page.locator("#topic-query").fill("半導體 人工智慧")
    page.locator("#topic-clusters").fill("3")
    page.locator("#topic-sample-size").fill("30")
    page.locator("#run-topic-explorer").click()
    page.wait_for_selector(".topic-card", timeout=60000)


def run_diagnostics(page) -> None:
    """Run the ranking diagnostics dashboard flow.

    Complexity:
        Time: O(request)
        Space: O(1)
    """
    page.locator("#diag-query").fill("半導體")
    page.locator("#diag-doc-id").fill("1")
    page.locator("#run-diagnostics-btn").click()
    page.wait_for_selector(".diagnostics-card", timeout=45000)


def run_feedback_dashboard(page) -> None:
    """Load the feedback analytics dashboard.

    Complexity:
        Time: O(request)
        Space: O(1)
    """
    page.wait_for_selector(".feedback-dashboard", timeout=45000)
    page.wait_for_selector(".feedback-summary-card", timeout=45000)
    page.locator(
        "#ltr-training-result"
    ).scroll_into_view_if_needed()
    page.locator(
        "button:has-text('Train Demo Ranker'), button:has-text('訓練 Demo Ranker')"
    ).click()
    page.wait_for_selector("#ltr-training-result table", timeout=45000)


def run_analysis_graph(page) -> None:
    """Load the node-based analysis graph and preview a node.

    Complexity:
        Time: O(request)
        Space: O(1)
    """
    page.wait_for_selector(".graph-node", timeout=45000)
    node_count = page.locator(".graph-node").count()
    if node_count > 1:
        page.locator(".graph-node").nth(1).hover()
    page.wait_for_selector(".analysis-graph-panel", timeout=45000)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Complexity:
        Time: O(1)
        Space: O(1)
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:5001")
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument("--no-start-server", action="store_true")
    parser.add_argument("--asset-dir", type=Path, default=ASSET_DIR)
    return parser.parse_args()


def main() -> None:
    """Run Playwright verification.

    Complexity:
        Time: O(startup + pages)
        Space: O(1)
    """
    args = parse_args()
    server = None
    try:
        if not args.no_start_server:
            server = start_server(args.port)
        wait_for_server(args.base_url)
        run_verification(args.base_url.rstrip("/"), args.asset_dir)
        print(f"Artifacts written to {args.asset_dir}")
    except Exception:
        if server is not None and server.stdout is not None:
            server.terminate()
            try:
                output, _ = server.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                output, _ = server.communicate(timeout=5)
            print(output[-4000:])
            server = None
        raise
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()


if __name__ == "__main__":
    main()
