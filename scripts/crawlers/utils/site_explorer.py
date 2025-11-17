#!/usr/bin/env python
"""
Website Structure Explorer

This utility helps explore website structure using Playwright
to understand how to properly crawl dynamic websites.

Usage:
    python scripts/crawlers/utils/site_explorer.py --url https://www.cna.com.tw

Author: CNIRS Development Team
License: Educational Use Only
"""

import asyncio
import argparse
import logging
from pathlib import Path
from playwright.async_api import async_playwright
from playwright_stealth import stealth_async

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


async def explore_site(url: str, output_dir: str = 'data/debug', screenshot: bool = True):
    """
    Explore website structure using Playwright.

    Args:
        url: URL to explore
        output_dir: Directory to save debug output
        screenshot: Whether to take screenshots

    Returns:
        dict: Information about the page structure
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    async with async_playwright() as p:
        # Launch browser
        logger.info("Launching browser...")
        browser = await p.chromium.launch(
            headless=True,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--no-sandbox',
                '--disable-setuid-sandbox',
            ]
        )

        # Create context with fingerprint
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            locale='zh-TW',
            timezone_id='Asia/Taipei',
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        )

        # Create page
        page = await context.new_page()

        # Apply stealth
        await stealth_async(page)

        # Navigate to URL
        logger.info(f"Navigating to {url}...")
        try:
            response = await page.goto(url, wait_until='networkidle', timeout=30000)
            logger.info(f"Response status: {response.status}")
        except Exception as e:
            logger.error(f"Failed to navigate: {e}")
            await browser.close()
            return None

        # Take screenshot
        if screenshot:
            screenshot_path = output_path / f"screenshot_{url.replace('/', '_').replace(':', '_')}.png"
            await page.screenshot(path=str(screenshot_path), full_page=True)
            logger.info(f"Screenshot saved to {screenshot_path}")

        # Get page info
        info = {
            'url': url,
            'title': await page.title(),
            'final_url': page.url,
        }

        # Get HTML content
        html_content = await page.content()
        html_path = output_path / f"html_{url.replace('/', '_').replace(':', '_')}.html"
        html_path.write_text(html_content, encoding='utf-8')
        logger.info(f"HTML saved to {html_path}")

        # Extract links
        links = await page.eval_on_selector_all('a[href]', 'elements => elements.map(e => e.href)')
        logger.info(f"Found {len(links)} links")
        info['links'] = links[:50]  # First 50 links

        # Find article containers (common patterns)
        article_selectors = [
            'article',
            '.article',
            '.news-item',
            '.post',
            '[class*="article"]',
            '[class*="news"]',
            '[class*="item"]',
        ]

        for selector in article_selectors:
            count = await page.eval_on_selector_all(selector, 'elements => elements.length')
            if count > 0:
                logger.info(f"Found {count} elements matching '{selector}'")
                info[f'selector_{selector}'] = count

        # Check for list pages
        list_selectors = [
            'ul.news-list',
            '.list-group',
            '[class*="list"]',
        ]

        for selector in list_selectors:
            count = await page.eval_on_selector_all(selector, 'elements => elements.length')
            if count > 0:
                logger.info(f"Found {count} lists matching '{selector}'")

        # Close browser
        await browser.close()

        logger.info("=" * 70)
        logger.info("Exploration complete!")
        logger.info(f"Title: {info['title']}")
        logger.info(f"Final URL: {info['final_url']}")
        logger.info("=" * 70)

        return info


async def test_cna_structure():
    """
    Test CNA website structure.
    """
    logger.info("=" * 70)
    logger.info("Testing CNA Website Structure")
    logger.info("=" * 70)

    # Test homepage
    logger.info("\n[1] Testing homepage...")
    await explore_site('https://www.cna.com.tw', output_dir='data/debug/cna')

    # Test category page
    logger.info("\n[2] Testing category page (politics)...")
    await explore_site('https://www.cna.com.tw/list/aipl.aspx', output_dir='data/debug/cna')

    # Test search page
    logger.info("\n[3] Testing search page...")
    await explore_site('https://www.cna.com.tw/search/hysearchws.aspx', output_dir='data/debug/cna')


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Explore website structure with Playwright')
    parser.add_argument('--url', type=str, help='URL to explore')
    parser.add_argument('--test-cna', action='store_true', help='Test CNA website structure')
    parser.add_argument('--output-dir', type=str, default='data/debug', help='Output directory')
    parser.add_argument('--no-screenshot', action='store_true', help='Disable screenshots')

    args = parser.parse_args()

    if args.test_cna:
        asyncio.run(test_cna_structure())
    elif args.url:
        asyncio.run(explore_site(args.url, args.output_dir, not args.no_screenshot))
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
