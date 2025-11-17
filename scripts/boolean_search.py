#!/usr/bin/env python3
"""
Boolean Search CLI Tool

Command-line interface for Boolean retrieval system with support for
AND, OR, NOT operators and phrase queries.

Usage:
    # Build index from documents
    python boolean_search.py --build --input docs.txt --index index.json

    # Search with Boolean query
    python boolean_search.py --query "information AND retrieval" --index index.json

    # Phrase query
    python boolean_search.py --query '"information retrieval"' --index index.json

    # Interactive mode
    python boolean_search.py --interactive --index index.json

Author: Information Retrieval System
License: Educational Use
"""

import sys
import argparse
import logging
import json
from pathlib import Path
from typing import List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ir.index.inverted_index import InvertedIndex
from src.ir.index.positional_index import PositionalIndex
from src.ir.retrieval.boolean import BooleanQueryEngine


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_documents(filepath: str) -> tuple[List[str], List[dict]]:
    """
    Load documents from file.

    Format:
        - Plain text: One document per line
        - JSON: Array of {text: ..., metadata: ...} objects

    Args:
        filepath: Input file path

    Returns:
        Tuple of (documents, metadata)
    """
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Try JSON first
    if path.suffix == '.json':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            documents = []
            metadata = []
            for item in data:
                if isinstance(item, dict):
                    documents.append(item.get('text', ''))
                    metadata.append({k: v for k, v in item.items() if k != 'text'})
                else:
                    documents.append(str(item))
                    metadata.append({})
            return documents, metadata

    # Plain text: one doc per line
    with open(path, 'r', encoding='utf-8') as f:
        documents = [line.strip() for line in f if line.strip()]

    metadata = [{'doc_id': i} for i in range(len(documents))]
    return documents, metadata


def build_index(input_file: str, index_file: str, positional: bool = True) -> None:
    """
    Build index from documents.

    Args:
        input_file: Input documents file
        index_file: Output index file
        positional: Whether to build positional index
    """
    print(f"Loading documents from {input_file}...")
    documents, metadata = load_documents(input_file)
    print(f"Loaded {len(documents)} documents")

    # Build inverted index
    print("\nBuilding inverted index...")
    inv_index = InvertedIndex()
    inv_index.build(documents, metadata)

    stats = inv_index.get_stats()
    print(f"  Documents: {stats['doc_count']}")
    print(f"  Vocabulary: {stats['vocabulary_size']}")
    print(f"  Total postings: {stats['total_postings']}")

    # Save inverted index
    inv_index_file = index_file.replace('.json', '_inverted.json')
    inv_index.save(inv_index_file)
    print(f"\nInverted index saved to {inv_index_file}")

    # Build positional index if requested
    if positional:
        print("\nBuilding positional index...")
        pos_index = PositionalIndex()
        pos_index.build(documents, metadata)

        pos_stats = pos_index.get_stats()
        print(f"  Total positions: {pos_stats['total_positions']}")

        # Save positional index
        pos_index_file = index_file.replace('.json', '_positional.json')
        pos_index.save(pos_index_file)
        print(f"Positional index saved to {pos_index_file}")

    print("\n✓ Index building complete!")


def load_indices(index_file: str) -> tuple[InvertedIndex, Optional[PositionalIndex]]:
    """
    Load indices from files.

    Args:
        index_file: Base index filename

    Returns:
        Tuple of (inverted_index, positional_index)
    """
    # Load inverted index
    inv_index_file = index_file.replace('.json', '_inverted.json')
    if not Path(inv_index_file).exists():
        raise FileNotFoundError(f"Inverted index not found: {inv_index_file}")

    inv_index = InvertedIndex()
    inv_index.load(inv_index_file)

    # Load positional index if exists
    pos_index_file = index_file.replace('.json', '_positional.json')
    pos_index = None
    if Path(pos_index_file).exists():
        pos_index = PositionalIndex()
        pos_index.load(pos_index_file)

    return inv_index, pos_index


def search(query_str: str, index_file: str, rank: bool = False,
          show_snippets: bool = False, max_results: int = 10) -> None:
    """
    Execute search query.

    Args:
        query_str: Query string
        index_file: Index file path
        rank: Whether to rank results
        show_snippets: Whether to show document snippets
        max_results: Maximum results to display
    """
    # Load indices
    inv_index, pos_index = load_indices(index_file)

    # Create engine
    engine = BooleanQueryEngine(inv_index, pos_index)

    # Execute query
    print(f"Query: {query_str}")
    print("-" * 60)

    result = engine.query(query_str, rank_results=rank)

    if result.num_results == 0:
        print("No results found.")
        return

    print(f"Found {result.num_results} documents\n")

    # Display results
    display_count = min(max_results, result.num_results)
    for i, doc_id in enumerate(result.doc_ids[:display_count], 1):
        print(f"{i}. Document {doc_id}")

        # Show metadata if available
        if doc_id in inv_index.doc_metadata:
            metadata = inv_index.doc_metadata[doc_id]
            for key, value in metadata.items():
                if key != 'text':
                    print(f"   {key}: {value}")

        # Show score if ranked
        if rank and result.scores:
            score = result.scores.get(doc_id, 0)
            print(f"   Score: {score:.2f}")

        print()

    if result.num_results > display_count:
        print(f"... and {result.num_results - display_count} more results")


def interactive_mode(index_file: str) -> None:
    """
    Interactive search mode.

    Args:
        index_file: Index file path
    """
    print("=" * 60)
    print("Boolean Search - Interactive Mode")
    print("=" * 60)
    print("\nCommands:")
    print("  - Enter query: search documents")
    print("  - 'help': Show query syntax")
    print("  - 'stats': Show index statistics")
    print("  - 'quit' or 'exit': Exit\n")

    # Load indices
    try:
        inv_index, pos_index = load_indices(index_file)
        engine = BooleanQueryEngine(inv_index, pos_index)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print(f"Loaded index: {inv_index.doc_count} documents, "
          f"{len(inv_index.vocabulary)} terms\n")

    while True:
        try:
            query = input("Query> ").strip()

            if not query:
                continue

            if query.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break

            elif query.lower() == 'help':
                show_help()
                continue

            elif query.lower() == 'stats':
                show_stats(inv_index, pos_index)
                continue

            # Execute query
            result = engine.query(query, rank_results=True)

            if result.num_results == 0:
                print("  No results found.\n")
            else:
                print(f"  Found {result.num_results} documents: {result.doc_ids[:10]}")
                if result.num_results > 10:
                    print(f"  ... and {result.num_results - 10} more")
                print()

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"  Error: {e}\n")


def show_help():
    """Display query syntax help."""
    print("\n" + "=" * 60)
    print("Query Syntax")
    print("=" * 60)
    print("""
Boolean Operators:
  AND    - Intersection (both terms must appear)
           Example: information AND retrieval

  OR     - Union (either term can appear)
           Example: boolean OR vector

  NOT    - Negation (exclude documents)
           Example: retrieval AND NOT extraction

Phrase Queries:
  "..."  - Exact phrase match
           Example: "information retrieval"

Grouping:
  (...)  - Group terms with parentheses
           Example: (boolean OR vector) AND model

Complex Examples:
  information AND (retrieval OR extraction)
  "vector space" AND model AND NOT boolean
  (term1 OR term2) AND NOT term3
""")
    print("=" * 60 + "\n")


def show_stats(inv_index: InvertedIndex, pos_index: Optional[PositionalIndex]):
    """Display index statistics."""
    print("\n" + "=" * 60)
    print("Index Statistics")
    print("=" * 60)

    inv_stats = inv_index.get_stats()
    print("\nInverted Index:")
    print(f"  Documents: {inv_stats['doc_count']}")
    print(f"  Vocabulary size: {inv_stats['vocabulary_size']}")
    print(f"  Total postings: {inv_stats['total_postings']}")
    print(f"  Avg posting length: {inv_stats['avg_posting_length']:.2f}")
    print(f"  Avg document length: {inv_stats['avg_doc_length']:.2f}")

    if pos_index:
        pos_stats = pos_index.get_stats()
        print("\nPositional Index:")
        print(f"  Total positions: {pos_stats['total_positions']}")
        print(f"  Avg positions per posting: {pos_stats['avg_positions_per_posting']:.2f}")

    print("=" * 60 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Boolean Search System - IR System CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build index from documents
  %(prog)s --build --input docs.txt --index my_index.json

  # Search with query
  %(prog)s --query "information AND retrieval" --index my_index.json

  # Phrase query
  %(prog)s --query '"vector space model"' --index my_index.json

  # Interactive mode
  %(prog)s --interactive --index my_index.json

  # Ranked results
  %(prog)s --query "information retrieval" --index my_index.json --rank

Query Syntax:
  - AND, OR, NOT operators
  - Phrase queries with "quotes"
  - Grouping with (parentheses)
        """
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--build', action='store_true',
                           help='Build index from documents')
    mode_group.add_argument('--query', type=str,
                           help='Execute search query')
    mode_group.add_argument('--interactive', action='store_true',
                           help='Interactive search mode')

    # Common options
    parser.add_argument('--index', type=str, required=True,
                       help='Index file path (JSON)')

    # Build options
    parser.add_argument('--input', type=str,
                       help='Input documents file (required for --build)')
    parser.add_argument('--no-positional', action='store_true',
                       help='Skip positional index (phrase queries disabled)')

    # Search options
    parser.add_argument('--rank', action='store_true',
                       help='Rank results by relevance')
    parser.add_argument('--max-results', type=int, default=10,
                       help='Maximum results to display (default: 10)')
    parser.add_argument('--show-snippets', action='store_true',
                       help='Show document snippets')

    # Utility
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    try:
        if args.build:
            # Build mode
            if not args.input:
                parser.error("--build requires --input")

            build_index(
                args.input,
                args.index,
                positional=not args.no_positional
            )

        elif args.query:
            # Query mode
            search(
                args.query,
                args.index,
                rank=args.rank,
                show_snippets=args.show_snippets,
                max_results=args.max_results
            )

        elif args.interactive:
            # Interactive mode
            interactive_mode(args.index)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
