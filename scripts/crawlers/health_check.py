#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Crawler Health Check System

Continuously monitors the health status of all news crawlers.
Generates reports and can send alerts for failing crawlers.

Usage:
    # Check all crawlers
    python health_check.py

    # Check specific crawlers
    python health_check.py --crawlers chinatimes,ettoday

    # Quick check (1 item each)
    python health_check.py --quick

    # Generate HTML report
    python health_check.py --html-report

Author: Information Retrieval System
Date: 2025-11-18
"""

import sys
import logging
import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


class CrawlerHealthChecker:
    """
    Health check system for news crawlers.

    Features:
        - Test all crawlers with minimal data
        - Generate JSON and HTML reports
        - Track historical health status
        - Alert on failures
    """

    def __init__(self, test_items: int = 1):
        """
        Initialize health checker.

        Args:
            test_items: Number of items to test per crawler (default: 1)
        """
        self.test_items = test_items
        self.output_dir = project_root / 'data' / 'health_check'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results = {
            'timestamp': datetime.now().isoformat(),
            'test_items': test_items,
            'crawlers': {},
            'summary': {
                'total': 0,
                'healthy': 0,
                'unhealthy': 0,
                'skipped': 0,
            }
        }

        # Crawler registry (same as in conftest.py)
        self.crawler_registry = {
            'cna': {'name': 'CNA中央社', 'playwright': False, 'priority': 'high'},
            'ltn': {'name': '自由時報', 'playwright': True, 'priority': 'high'},
            'pts': {'name': '公視', 'playwright': False, 'priority': 'medium'},
            'udn': {'name': '聯合報', 'playwright': True, 'priority': 'high'},
            'apple': {'name': '蘋果日報', 'playwright': True, 'priority': 'medium'},
            'tvbs': {'name': 'TVBS新聞', 'playwright': True, 'priority': 'medium', 'skip': True},
            'chinatimes': {'name': '中時新聞網', 'playwright': False, 'priority': 'high'},
            'ettoday': {'name': '東森新聞雲', 'playwright': True, 'priority': 'high'},
            'storm': {'name': '風傳媒', 'playwright': True, 'priority': 'medium'},
        }

    def check_crawler(self, crawler_name: str) -> Dict:
        """
        Check health of a single crawler.

        Args:
            crawler_name: Name of crawler to check

        Returns:
            dict: Health check result
        """
        config = self.crawler_registry.get(crawler_name)
        if not config:
            return {
                'status': 'error',
                'message': f'Unknown crawler: {crawler_name}'
            }

        if config.get('skip'):
            return {
                'status': 'skipped',
                'message': 'Crawler marked as skip',
                'crawler_name': config['name'],
            }

        logger.info(f"Checking {config['name']} ({crawler_name})...")

        start_time = time.time()

        try:
            # Run test using existing test script
            test_script = project_root / 'scripts' / 'crawlers' / 'test_single_crawler.py'

            import subprocess
            result = subprocess.run(
                [
                    sys.executable,
                    str(test_script),
                    crawler_name,
                    str(self.test_items)
                ],
                capture_output=True,
                text=True,
                timeout=180,  # 3 minutes timeout
                cwd=str(project_root)
            )

            elapsed_time = time.time() - start_time

            # Parse output
            success = result.returncode == 0
            output_lines = result.stdout.split('\n')

            # Extract statistics
            items_scraped = 0
            for line in output_lines:
                if 'TEST PASSED' in line and 'items scraped' in line:
                    try:
                        items_scraped = int(line.split()[2])
                    except:
                        pass

            return {
                'status': 'healthy' if success and items_scraped > 0 else 'unhealthy',
                'crawler_name': config['name'],
                'items_scraped': items_scraped,
                'elapsed_time': round(elapsed_time, 2),
                'playwright': config['playwright'],
                'priority': config['priority'],
                'message': 'Working normally' if success else 'Failed to scrape items',
                'details': {
                    'return_code': result.returncode,
                    'has_output': items_scraped > 0,
                }
            }

        except subprocess.TimeoutExpired:
            elapsed_time = time.time() - start_time
            return {
                'status': 'unhealthy',
                'crawler_name': config['name'],
                'items_scraped': 0,
                'elapsed_time': round(elapsed_time, 2),
                'playwright': config['playwright'],
                'priority': config['priority'],
                'message': 'Timeout (> 3 minutes)',
            }

        except Exception as e:
            elapsed_time = time.time() - start_time
            return {
                'status': 'error',
                'crawler_name': config['name'],
                'items_scraped': 0,
                'elapsed_time': round(elapsed_time, 2),
                'playwright': config['playwright'],
                'priority': config['priority'],
                'message': f'Error: {str(e)}',
            }

    def check_all(self, crawler_names: Optional[List[str]] = None) -> Dict:
        """
        Check health of all or specified crawlers.

        Args:
            crawler_names: List of crawler names (None = all)

        Returns:
            dict: Complete health check results
        """
        if crawler_names is None:
            crawler_names = list(self.crawler_registry.keys())

        self.results['summary']['total'] = len(crawler_names)

        for crawler_name in crawler_names:
            result = self.check_crawler(crawler_name)
            self.results['crawlers'][crawler_name] = result

            # Update summary
            status = result['status']
            if status == 'healthy':
                self.results['summary']['healthy'] += 1
            elif status == 'skipped':
                self.results['summary']['skipped'] += 1
            else:
                self.results['summary']['unhealthy'] += 1

        return self.results

    def save_json_report(self) -> Path:
        """
        Save health check results as JSON.

        Returns:
            Path: Path to JSON file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_file = self.output_dir / f'health_check_{timestamp}.json'

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        logger.info(f"JSON report saved to: {json_file}")
        return json_file

    def save_html_report(self) -> Path:
        """
        Save health check results as HTML.

        Returns:
            Path: Path to HTML file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        html_file = self.output_dir / f'health_check_{timestamp}.html'

        html_content = self._generate_html()

        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        logger.info(f"HTML report saved to: {html_file}")
        return html_file

    def _generate_html(self) -> str:
        """Generate HTML report."""
        summary = self.results['summary']
        health_percentage = (summary['healthy'] / summary['total'] * 100) if summary['total'] > 0 else 0

        # Generate crawler rows
        crawler_rows = []
        for crawler_name, result in self.results['crawlers'].items():
            status = result['status']
            status_color = {
                'healthy': '#28a745',
                'unhealthy': '#dc3545',
                'skipped': '#6c757d',
                'error': '#dc3545',
            }.get(status, '#ffc107')

            status_icon = {
                'healthy': '✓',
                'unhealthy': '✗',
                'skipped': '−',
                'error': '⚠',
            }.get(status, '?')

            row = f"""
            <tr>
                <td>{result.get('crawler_name', crawler_name)}</td>
                <td style="color: {status_color}; font-weight: bold;">{status_icon} {status.upper()}</td>
                <td>{result.get('items_scraped', 0)}</td>
                <td>{result.get('elapsed_time', 0)}s</td>
                <td>{'Yes' if result.get('playwright') else 'No'}</td>
                <td>{result.get('priority', 'N/A').upper()}</td>
                <td>{result.get('message', 'N/A')}</td>
            </tr>
            """
            crawler_rows.append(row)

        html = f"""
<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Crawler Health Check Report</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; margin-bottom: 10px; }}
        .timestamp {{ color: #666; font-size: 14px; margin-bottom: 20px; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 30px; }}
        .summary-card {{ padding: 20px; border-radius: 6px; color: white; }}
        .summary-card h3 {{ font-size: 16px; margin-bottom: 10px; opacity: 0.9; }}
        .summary-card .number {{ font-size: 36px; font-weight: bold; }}
        .healthy {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }}
        .unhealthy {{ background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }}
        .total {{ background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }}
        .skipped {{ background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th {{ background: #f8f9fa; padding: 12px; text-align: left; font-weight: 600; color: #333; border-bottom: 2px solid #dee2e6; }}
        td {{ padding: 12px; border-bottom: 1px solid #dee2e6; }}
        tr:hover {{ background: #f8f9fa; }}
        .health-bar {{ width: 100%; height: 30px; background: #e9ecef; border-radius: 15px; overflow: hidden; margin: 20px 0; }}
        .health-fill {{ height: 100%; background: linear-gradient(90deg, #28a745, #20c997); transition: width 0.3s; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔍 Crawler Health Check Report</h1>
        <div class="timestamp">Generated: {self.results['timestamp']}</div>

        <div class="health-bar">
            <div class="health-fill" style="width: {health_percentage}%;">{health_percentage:.1f}% Healthy</div>
        </div>

        <div class="summary">
            <div class="summary-card total">
                <h3>Total Crawlers</h3>
                <div class="number">{summary['total']}</div>
            </div>
            <div class="summary-card healthy">
                <h3>✓ Healthy</h3>
                <div class="number">{summary['healthy']}</div>
            </div>
            <div class="summary-card unhealthy">
                <h3>✗ Unhealthy</h3>
                <div class="number">{summary['unhealthy']}</div>
            </div>
            <div class="summary-card skipped">
                <h3>− Skipped</h3>
                <div class="number">{summary['skipped']}</div>
            </div>
        </div>

        <table>
            <thead>
                <tr>
                    <th>Crawler</th>
                    <th>Status</th>
                    <th>Items</th>
                    <th>Time</th>
                    <th>Playwright</th>
                    <th>Priority</th>
                    <th>Message</th>
                </tr>
            </thead>
            <tbody>
                {''.join(crawler_rows)}
            </tbody>
        </table>
    </div>
</body>
</html>
        """

        return html

    def print_summary(self):
        """Print health check summary to console."""
        summary = self.results['summary']

        print("\n" + "=" * 70)
        print("Crawler Health Check Summary")
        print("=" * 70)
        print(f"Timestamp: {self.results['timestamp']}")
        print(f"Test Items: {self.results['test_items']}")
        print(f"\nTotal Crawlers: {summary['total']}")
        print(f"✓ Healthy: {summary['healthy']}")
        print(f"✗ Unhealthy: {summary['unhealthy']}")
        print(f"− Skipped: {summary['skipped']}")

        health_percentage = (summary['healthy'] / summary['total'] * 100) if summary['total'] > 0 else 0
        print(f"\nOverall Health: {health_percentage:.1f}%")

        print("\nDetailed Results:")
        print("-" * 70)
        for crawler_name, result in self.results['crawlers'].items():
            status_icon = {'healthy': '✓', 'unhealthy': '✗', 'skipped': '−'}.get(result['status'], '?')
            print(f"{status_icon} {result.get('crawler_name', crawler_name):20s} | "
                  f"Items: {result.get('items_scraped', 0):2d} | "
                  f"Time: {result.get('elapsed_time', 0):5.1f}s | "
                  f"{result.get('message', 'N/A')}")
        print("=" * 70)


def main():
    """Main function for CLI."""
    parser = argparse.ArgumentParser(
        description='Health check system for news crawlers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        '--crawlers', '-c',
        type=str,
        help='Specific crawlers to check (comma-separated)'
    )

    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Quick check (1 item per crawler, default)'
    )

    parser.add_argument(
        '--items', '-n',
        type=int,
        default=1,
        help='Number of items to test per crawler (default: 1)'
    )

    parser.add_argument(
        '--html-report',
        action='store_true',
        help='Generate HTML report'
    )

    parser.add_argument(
        '--json-report',
        action='store_true',
        help='Generate JSON report'
    )

    args = parser.parse_args()

    # Determine test items
    test_items = 1 if args.quick else args.items

    # Parse crawler names
    crawler_names = None
    if args.crawlers:
        crawler_names = [c.strip() for c in args.crawlers.split(',')]

    # Run health check
    logger.info("Starting crawler health check...")
    logger.info(f"Test items per crawler: {test_items}")

    checker = CrawlerHealthChecker(test_items=test_items)
    checker.check_all(crawler_names=crawler_names)

    # Print summary
    checker.print_summary()

    # Generate reports
    if args.json_report or not args.html_report:
        checker.save_json_report()

    if args.html_report:
        html_file = checker.save_html_report()
        logger.info(f"\n📊 Open HTML report: {html_file}")


if __name__ == '__main__':
    main()
