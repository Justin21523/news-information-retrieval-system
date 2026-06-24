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


def wait_for_server(base_url: str, timeout: float = 90.0) -> None:
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
            viewport={"width": 1440, "height": 1100},
            record_video_dir=str(video_dir),
            record_video_size={"width": 1440, "height": 1100},
        )
        page = context.new_page()
        page.set_default_timeout(45000)

        page.goto(f"{base_url}/", wait_until="networkidle")
        fill_search(page, "半導體")
        page.screenshot(path=asset_dir / "search-results.png", full_page=True)

        open_document_detail(page)
        page.screenshot(path=asset_dir / "document-detail.png", full_page=True)

        page.goto(f"{base_url}/compare", wait_until="networkidle")
        fill_compare(page, "人工智慧")
        page.screenshot(path=asset_dir / "model-compare.png", full_page=True)

        page.goto(f"{base_url}/evaluation", wait_until="networkidle")
        run_evaluation(page)
        page.screenshot(path=asset_dir / "evaluation-dashboard.png", full_page=True)

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
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()


if __name__ == "__main__":
    main()
