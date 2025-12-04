#!/usr/bin/env python
"""
News Search CLI Tool

Command-line interface for searching news articles using the
Unified Search Engine. Supports multiple query modes and ranking models.

Usage Examples:
    # Simple keyword search with BM25
    python scripts/search_news.py --query "台灣 政治" --topk 10

    # Field-specific search
    python scripts/search_news.py --query "title:台灣 AND category:政治" --mode field

    # Use Vector Space Model
    python scripts/search_news.py --query "經濟 發展" --model vsm --topk 20

    # Hybrid ranking
    python scripts/search_news.py --query "人工智慧" --model hybrid

    # Build index from JSONL first
    python scripts/search_news.py --build --data-dir data/raw --limit 10000

    # Build from PostgreSQL
    python scripts/search_news.py --build --from-db --db-name ir_news --source ltn

    # Interactive search mode
    python scripts/search_news.py --interactive

Author: Information Retrieval System
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ir.search import (
    UnifiedSearchEngine,
    QueryMode,
    RankingModel,
    UnifiedSearchResult
)


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def print_results(results: list, show_content: bool = False, max_content_len: int = 150):
    """
    Print search results in a formatted manner.

    Args:
        results: List of UnifiedSearchResult objects
        show_content: Whether to show content snippets
        max_content_len: Maximum content snippet length
    """
    if not results:
        print("\n⚠ No results found.")
        return

    print(f"\n{'='*100}")
    print(f"Found {len(results)} results:")
    print(f"{'='*100}\n")

    for result in results:
        print(f"  [{result.rank}] {result.title}")
        print(f"      Score: {result.score:.4f} | Model: {result.ranking_model}")
        print(f"      Source: {result.source} | Category: {result.category or 'N/A'}")
        if result.published_at:
            print(f"      Date: {result.published_at}")
        if result.url:
            print(f"      URL: {result.url}")
        if show_content and result.content:
            content = result.content[:max_content_len]
            if len(result.content) > max_content_len:
                content += "..."
            print(f"      Content: {content}")
        if result.matched_fields:
            print(f"      Matched Fields: {', '.join(result.matched_fields)}")
        print()


def interactive_search(engine: UnifiedSearchEngine):
    """
    Interactive search mode.

    Args:
        engine: UnifiedSearchEngine instance
    """
    print("\n" + "="*100)
    print("Interactive Search Mode")
    print("="*100)
    print("\nCommands:")
    print("  Enter query to search")
    print("  :mode [auto|simple|field|boolean] - Change query mode")
    print("  :model [bm25|vsm|hybrid] - Change ranking model")
    print("  :topk N - Set number of results")
    print("  :content - Toggle content display")
    print("  :stats - Show index statistics")
    print("  :help - Show this help")
    print("  :quit - Exit")
    print("="*100)

    # Default settings
    mode = QueryMode.AUTO
    model = RankingModel.BM25
    topk = 10
    show_content = False

    while True:
        try:
            query = input("\nQuery> ").strip()

            if not query:
                continue

            # Handle commands
            if query.startswith(':'):
                cmd = query[1:].lower().split()

                if cmd[0] == 'quit' or cmd[0] == 'exit' or cmd[0] == 'q':
                    print("Goodbye!")
                    break

                elif cmd[0] == 'help' or cmd[0] == 'h':
                    print("\nAvailable commands:")
                    print("  :mode [auto|simple|field|boolean]")
                    print("  :model [bm25|vsm|hybrid]")
                    print("  :topk N")
                    print("  :content")
                    print("  :stats")
                    print("  :help")
                    print("  :quit")

                elif cmd[0] == 'mode':
                    if len(cmd) > 1:
                        try:
                            mode = QueryMode(cmd[1])
                            print(f"✓ Query mode set to: {mode.value}")
                        except ValueError:
                            print(f"⚠ Invalid mode. Use: auto, simple, field, boolean")
                    else:
                        print(f"Current mode: {mode.value}")

                elif cmd[0] == 'model':
                    if len(cmd) > 1:
                        try:
                            model = RankingModel(cmd[1])
                            print(f"✓ Ranking model set to: {model.value}")
                        except ValueError:
                            print(f"⚠ Invalid model. Use: bm25, vsm, hybrid")
                    else:
                        print(f"Current model: {model.value}")

                elif cmd[0] == 'topk':
                    if len(cmd) > 1:
                        try:
                            topk = int(cmd[1])
                            print(f"✓ Top-k set to: {topk}")
                        except ValueError:
                            print(f"⚠ Invalid number")
                    else:
                        print(f"Current top-k: {topk}")

                elif cmd[0] == 'content':
                    show_content = not show_content
                    print(f"✓ Content display: {'ON' if show_content else 'OFF'}")

                elif cmd[0] == 'stats':
                    stats = engine.get_stats()
                    print(f"\nIndex Statistics:")
                    for key, value in stats.items():
                        print(f"  {key}: {value}")

                else:
                    print(f"⚠ Unknown command: {cmd[0]}")

                continue

            # Execute search
            print(f"\nSearching: '{query}'")
            print(f"Mode: {mode.value} | Model: {model.value} | Top-K: {topk}")
            print("-"*100)

            results = engine.search(query, mode=mode, ranking_model=model, top_k=topk)
            print_results(results, show_content=show_content)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n⚠ Error: {e}")
            logging.error(f"Search error: {e}", exc_info=True)


def main():
    parser = argparse.ArgumentParser(
        description='Search news articles using Information Retrieval system',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple search
  python %(prog)s --query "台灣 政治" --topk 10

  # Field search
  python %(prog)s --query "title:台灣 AND category:政治" --mode field

  # Use VSM model
  python %(prog)s --query "經濟 發展" --model vsm

  # Hybrid ranking
  python %(prog)s --query "人工智慧" --model hybrid --topk 20

  # Build index first
  python %(prog)s --build --data-dir data/raw --limit 10000

  # Interactive mode
  python %(prog)s --interactive
        """
    )

    # Build index options
    build_group = parser.add_argument_group('Build Index')
    build_group.add_argument(
        '--build',
        action='store_true',
        help='Build search index before searching'
    )
    build_group.add_argument(
        '--data-dir',
        type=str,
        default='data/raw',
        help='Directory containing JSONL files (default: data/raw)'
    )
    build_group.add_argument(
        '--limit',
        type=int,
        help='Maximum documents to index (default: all)'
    )
    build_group.add_argument(
        '--index-dir',
        type=str,
        default='data/index',
        help='Directory for index storage (default: data/index)'
    )

    # PostgreSQL options
    pg_group = parser.add_argument_group('PostgreSQL')
    pg_group.add_argument(
        '--from-db',
        action='store_true',
        help='Build index from PostgreSQL database'
    )
    pg_group.add_argument(
        '--db-host',
        type=str,
        default='localhost',
        help='PostgreSQL host (default: localhost)'
    )
    pg_group.add_argument(
        '--db-port',
        type=int,
        default=5432,
        help='PostgreSQL port (default: 5432)'
    )
    pg_group.add_argument(
        '--db-name',
        type=str,
        default='ir_news',
        help='Database name (default: ir_news)'
    )
    pg_group.add_argument(
        '--db-user',
        type=str,
        default='postgres',
        help='Database user (default: postgres)'
    )
    pg_group.add_argument(
        '--db-password',
        type=str,
        default='postgres',
        help='Database password (default: postgres)'
    )
    pg_group.add_argument(
        '--source',
        type=str,
        help='Filter by news source (e.g., ltn, cna)'
    )
    pg_group.add_argument(
        '--category',
        type=str,
        help='Filter by category'
    )

    # Search options
    search_group = parser.add_argument_group('Search')
    search_group.add_argument(
        '--query',
        '-q',
        type=str,
        help='Search query string'
    )
    search_group.add_argument(
        '--mode',
        type=str,
        choices=['auto', 'simple', 'field', 'boolean'],
        default='auto',
        help='Query mode (default: auto)'
    )
    search_group.add_argument(
        '--model',
        type=str,
        choices=['bm25', 'vsm', 'hybrid'],
        default='bm25',
        help='Ranking model (default: bm25)'
    )
    search_group.add_argument(
        '--topk',
        '-k',
        type=int,
        default=10,
        help='Number of top results to return (default: 10)'
    )
    search_group.add_argument(
        '--show-content',
        action='store_true',
        help='Show content snippets in results'
    )

    # Interactive mode
    parser.add_argument(
        '--interactive',
        '-i',
        action='store_true',
        help='Interactive search mode'
    )

    # Other options
    parser.add_argument(
        '--ckip-model',
        type=str,
        default='bert-base',
        choices=['bert-base', 'albert-base', 'albert-tiny'],
        help='CKIP model variant (default: bert-base)'
    )
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Verbose output (DEBUG level)'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Print header
    print("=" * 100)
    print("News Search System - Information Retrieval")
    print("=" * 100)

    # Initialize search engine
    print(f"\nInitializing search engine (CKIP model: {args.ckip_model})...")
    engine = UnifiedSearchEngine(
        index_dir=args.index_dir,
        ckip_model=args.ckip_model
    )

    # Build index if requested
    if args.build:
        print(f"\nBuilding index...")
        print("-" * 100)

        if args.from_db:
            # Build from PostgreSQL
            try:
                from src.database.postgres_manager import PostgresManager

                print(f"Connecting to PostgreSQL: {args.db_host}:{args.db_port}/{args.db_name}")
                db_manager = PostgresManager(
                    host=args.db_host,
                    port=args.db_port,
                    database=args.db_name,
                    user=args.db_user,
                    password=args.db_password
                )

                stats = engine.build_index_from_postgres(
                    db_manager=db_manager,
                    source=args.source,
                    category=args.category,
                    limit=args.limit
                )

                db_manager.close()

            except ImportError:
                logger.error("PostgreSQL support not available. Install psycopg2-binary.")
                return 1
            except Exception as e:
                logger.error(f"Failed to build index from PostgreSQL: {e}")
                return 1

        else:
            # Build from JSONL
            data_dir = Path(args.data_dir)
            if not data_dir.exists():
                logger.error(f"Data directory not found: {data_dir}")
                return 1

            stats = engine.build_index_from_jsonl(
                data_dir=str(data_dir),
                limit=args.limit
            )

        print(f"\nIndex Build Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

    else:
        # Try to load existing index
        print("\n⚠ No --build flag specified.")
        print("Note: Index building is required before searching.")
        print("Use --build to build index from data sources.")

    # Interactive mode
    if args.interactive:
        if not engine.is_indexed:
            print("\n⚠ Index not built. Please use --build first or build index in interactive mode.")
            return 1

        interactive_search(engine)
        return 0

    # Single query mode
    if args.query:
        if not engine.is_indexed:
            print("\n⚠ Index not built. Please use --build first.")
            return 1

        # Parse mode and model
        mode = QueryMode(args.mode)
        model = RankingModel(args.model)

        print(f"\nSearching: '{args.query}'")
        print(f"Mode: {mode.value} | Model: {model.value} | Top-K: {args.topk}")
        print("-" * 100)

        try:
            results = engine.search(
                query=args.query,
                mode=mode,
                ranking_model=model,
                top_k=args.topk
            )

            print_results(results, show_content=args.show_content)

        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            return 1

    else:
        # No query provided
        if args.build:
            print("\nIndex built successfully!")
            print("Use --query to search or --interactive for interactive mode.")
        else:
            print("\nNo query or action specified.")
            print("Use --query for single search or --interactive for interactive mode.")
            print("Use --build to build index first.")
            print("\nFor help: python scripts/search_news.py --help")

    print("\n" + "=" * 100)
    return 0


if __name__ == '__main__':
    sys.exit(main())
