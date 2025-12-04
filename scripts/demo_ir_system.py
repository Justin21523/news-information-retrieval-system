#!/usr/bin/env python
"""
IR System Interactive Demo

Interactive demonstration of all IR system capabilities including:
- Simple queries
- Boolean queries
- Field queries
- Different ranking models
- CKIP tokenization showcase

Usage:
    python scripts/demo_ir_system.py --index-dir data/index_50k

Author: Information Retrieval System
"""

import sys
import time
import logging
from pathlib import Path
from typing import List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ir.search import (
    UnifiedSearchEngine,
    QueryMode,
    RankingModel
)
from src.ir.text.ckip_tokenizer import get_tokenizer


class IRSystemDemo:
    """Interactive IR System Demonstration."""

    def __init__(self, index_dir: str):
        """
        Initialize demo.

        Args:
            index_dir: Path to index directory
        """
        self.logger = logging.getLogger(__name__)
        self.index_dir = index_dir

        print("="*80)
        print("IR 系統互動式展示 (Interactive IR System Demo)")
        print("="*80)
        print(f"\n載入索引: {index_dir}")

        self.engine = UnifiedSearchEngine(index_dir=index_dir)

        if not self.engine.is_indexed:
            print(f"\n❌ 索引不存在: {index_dir}")
            print("請先建立索引:")
            print(f"  python scripts/search_news.py --build --limit 50000")
            sys.exit(1)

        # Get stats
        stats = self.engine.get_stats()
        print(f"\n✓ 索引載入成功")
        print(f"  文檔數: {stats.get('total_documents', 0):,}")
        print(f"  詞彙數: {stats.get('unique_terms', 0):,}")
        print(f"  索引大小: {stats.get('index_size_mb', 0):.2f} MB")

        # Initialize tokenizer for demos
        self.tokenizer = get_tokenizer()

    def demo_ckip_tokenization(self):
        """Demonstrate CKIP tokenization."""
        print("\n" + "="*80)
        print("📝 DEMO 1: CKIP 中文分詞展示")
        print("="*80)

        test_sentences = [
            "台積電宣布擴大投資半導體製造",
            "人工智慧技術在醫療領域的應用",
            "COVID-19疫苗接種率持續上升",
            "行政院長出席記者會說明政策",
            "台灣股市今日大漲三百點"
        ]

        for i, sentence in enumerate(test_sentences, 1):
            print(f"\n[{i}] 原始句子: {sentence}")

            start = time.time()
            tokens = self.tokenizer.tokenize(sentence)
            elapsed = (time.time() - start) * 1000

            print(f"    分詞結果: {' | '.join(tokens)}")
            print(f"    詞彙數量: {len(tokens)} 個詞")
            print(f"    處理時間: {elapsed:.2f}ms")

        print("\n💡 觀察重點:")
        print("  • 正確識別專有名詞 (台積電、COVID-19)")
        print("  • 合理的詞界切分")
        print("  • 中英文混合處理")

        input("\n按 Enter 繼續...")

    def demo_simple_search(self):
        """Demonstrate simple search."""
        print("\n" + "="*80)
        print("🔍 DEMO 2: 簡單查詢 (Simple Query)")
        print("="*80)

        demo_queries = [
            ("台灣 經濟", "經濟相關新聞"),
            ("人工智慧", "AI 技術新聞"),
            ("疫情 防疫", "疫情防疫措施")
        ]

        for query, description in demo_queries:
            print(f"\n查詢: \"{query}\" ({description})")

            start = time.time()
            results = self.engine.search(
                query=query,
                mode=QueryMode.SIMPLE,
                ranking_model=RankingModel.BM25,
                top_k=5
            )
            elapsed = (time.time() - start) * 1000

            print(f"找到 {len(results)} 筆結果，用時 {elapsed:.2f}ms\n")

            for i, result in enumerate(results[:3], 1):
                print(f"  [{i}] {result.title[:60]}...")
                print(f"      分數: {result.score:.4f} | 來源: {result.source}")
                if result.content:
                    print(f"      內容: {result.content[:80]}...")

        input("\n按 Enter 繼續...")

    def demo_boolean_search(self):
        """Demonstrate Boolean search."""
        print("\n" + "="*80)
        print("🔢 DEMO 3: Boolean 查詢")
        print("="*80)

        demo_queries = [
            ("台灣 AND 經濟", "必須同時包含「台灣」和「經濟」"),
            ("人工智慧 OR 機器學習", "包含 AI 或 ML 的文章"),
            ("疫苗 AND NOT 副作用", "疫苗新聞但排除副作用")
        ]

        for query, explanation in demo_queries:
            print(f"\n查詢: \"{query}\"")
            print(f"說明: {explanation}")

            results = self.engine.search(
                query=query,
                mode=QueryMode.BOOLEAN,
                ranking_model=RankingModel.BM25,
                top_k=5
            )

            print(f"結果: {len(results)} 筆")

            for i, result in enumerate(results[:2], 1):
                print(f"  [{i}] {result.title[:65]}...")
                print(f"      來源: {result.source}")

        input("\n按 Enter 繼續...")

    def demo_field_search(self):
        """Demonstrate field-specific search."""
        print("\n" + "="*80)
        print("🎯 DEMO 4: 欄位查詢 (Field Query)")
        print("="*80)

        demo_queries = [
            ("title:台灣", "標題包含「台灣」"),
            ("category:政治", "政治類新聞"),
            ("source:ltn", "自由時報的文章"),
            ("title:AI AND category:科技", "標題有 AI 且分類為科技")
        ]

        for query, explanation in demo_queries:
            print(f"\n查詢: \"{query}\"")
            print(f"說明: {explanation}")

            results = self.engine.search(
                query=query,
                mode=QueryMode.FIELD,
                ranking_model=RankingModel.BM25,
                top_k=5
            )

            print(f"結果: {len(results)} 筆")

            for i, result in enumerate(results[:2], 1):
                print(f"  [{i}] {result.title[:65]}...")
                if hasattr(result, 'category') and result.category:
                    print(f"      分類: {result.category} | 來源: {result.source}")

        input("\n按 Enter 繼續...")

    def demo_ranking_models(self):
        """Demonstrate different ranking models."""
        print("\n" + "="*80)
        print("📊 DEMO 5: 排序模型比較")
        print("="*80)

        query = "台灣 經濟 發展"
        print(f"\n查詢: \"{query}\"\n")

        models = [
            (RankingModel.BM25, "BM25 (Best Match 25)"),
            (RankingModel.VSM, "VSM (Vector Space Model)"),
            (RankingModel.HYBRID, "Hybrid (BM25 + VSM)")
        ]

        for model, name in models:
            print(f"\n{name}:")
            print("-" * 60)

            start = time.time()
            results = self.engine.search(
                query=query,
                mode=QueryMode.SIMPLE,
                ranking_model=model,
                top_k=3
            )
            elapsed = (time.time() - start) * 1000

            print(f"查詢時間: {elapsed:.2f}ms\n")

            for i, result in enumerate(results, 1):
                print(f"  [{i}] 分數: {result.score:.4f}")
                print(f"      {result.title[:60]}...")

        print("\n💡 觀察重點:")
        print("  • BM25: 考慮詞頻和文檔長度，適合一般檢索")
        print("  • VSM: 基於向量相似度，適合找相似文章")
        print("  • Hybrid: 結合兩者優點，檢索品質較佳")

        input("\n按 Enter 繼續...")

    def demo_performance(self):
        """Demonstrate query performance."""
        print("\n" + "="*80)
        print("⚡ DEMO 6: 查詢效能展示")
        print("="*80)

        test_cases = [
            ("台灣", "單一高頻詞"),
            ("台灣 經濟", "兩個詞"),
            ("台灣 經濟 發展 政策 趨勢", "五個詞"),
            ("title:AI AND category:科技", "複雜欄位查詢")
        ]

        print("\n查詢效能測試 (每個查詢執行 3 次):\n")
        print(f"{'查詢':<40} {'平均時間':<15} {'結果數':<10}")
        print("-" * 70)

        for query, description in test_cases:
            times = []
            num_results = 0

            for _ in range(3):
                start = time.time()
                results = self.engine.search(query, top_k=10)
                elapsed = (time.time() - start) * 1000
                times.append(elapsed)
                num_results = len(results)

            avg_time = sum(times) / len(times)
            print(f"{description:<40} {avg_time:>10.2f}ms     {num_results:>5}")

        input("\n按 Enter 繼續...")

    def demo_interactive_search(self):
        """Interactive search session."""
        print("\n" + "="*80)
        print("💬 DEMO 7: 互動式查詢")
        print("="*80)
        print("\n現在您可以輸入自己的查詢來測試系統！")
        print("\n支援的查詢類型:")
        print("  • 簡單查詢: 台灣 經濟")
        print("  • Boolean: 台灣 AND 經濟")
        print("  • 欄位查詢: title:AI")
        print("\n輸入 'quit' 或 'q' 結束\n")

        while True:
            try:
                query = input("請輸入查詢 > ").strip()

                if query.lower() in ['quit', 'q', 'exit']:
                    break

                if not query:
                    continue

                # Auto-detect mode
                mode = QueryMode.AUTO
                model = RankingModel.BM25

                start = time.time()
                results = self.engine.search(
                    query=query,
                    mode=mode,
                    ranking_model=model,
                    top_k=5
                )
                elapsed = (time.time() - start) * 1000

                print(f"\n找到 {len(results)} 筆結果 (用時 {elapsed:.2f}ms)")

                if results:
                    print()
                    for i, result in enumerate(results, 1):
                        print(f"[{i}] {result.title}")
                        print(f"    分數: {result.score:.4f} | 來源: {result.source}")
                        if result.content:
                            print(f"    {result.content[:100]}...")
                        print()
                else:
                    print("沒有找到相關文章\n")

            except KeyboardInterrupt:
                print("\n\n查詢中斷")
                break
            except Exception as e:
                print(f"\n錯誤: {e}\n")

    def run_all_demos(self):
        """Run all demonstration modules."""
        try:
            self.demo_ckip_tokenization()
            self.demo_simple_search()
            self.demo_boolean_search()
            self.demo_field_search()
            self.demo_ranking_models()
            self.demo_performance()
            self.demo_interactive_search()

            print("\n" + "="*80)
            print("✓ Demo 展示完成！")
            print("="*80)
            print("\n感謝使用 IR 系統！")
            print("\n更多資訊請參考:")
            print("  • 使用指南: docs/guides/IR_SYSTEM_USER_GUIDE.md")
            print("  • 測試查詢: tests/test_queries.yaml")
            print("  • Web 介面: python app_simple.py")
            print("="*80 + "\n")

        except KeyboardInterrupt:
            print("\n\nDemo 已中斷")
        except Exception as e:
            self.logger.error(f"Demo 執行失敗: {e}", exc_info=True)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='IR System Interactive Demo'
    )
    parser.add_argument(
        '--index-dir',
        type=str,
        default='data/index_50k',
        help='Index directory (default: data/index_50k)'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Verbose output (DEBUG level)'
    )

    args = parser.parse_args()

    # Setup logging
    level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run demo
    demo = IRSystemDemo(index_dir=args.index_dir)
    demo.run_all_demos()


if __name__ == '__main__':
    main()
