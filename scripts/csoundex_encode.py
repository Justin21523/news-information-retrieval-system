#!/usr/bin/env python3
"""
CSoundex Encoding CLI Tool

Command-line interface for encoding Chinese text using CSoundex phonetic system.

Usage:
    # Encode text directly
    python csoundex_encode.py --text "三聚氰胺"

    # Encode from file
    python csoundex_encode.py --file input.txt --output encoded.txt

    # Encode from stdin
    echo "信息檢索" | python csoundex_encode.py --stdin

    # Batch encoding with similarity search
    python csoundex_encode.py --file names.txt --similar "張三" --threshold 0.6

Author: Information Retrieval System
License: Educational Use
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import List, Optional

# Add parent directory to path to import ir module
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ir.text.csoundex import CSoundex


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def encode_text(csoundex: CSoundex, text: str, include_tone: bool, show_original: bool) -> str:
    """
    Encode text and format output.

    Args:
        csoundex: CSoundex encoder instance
        text: Input text
        include_tone: Whether to include tone
        show_original: Whether to show original text

    Returns:
        Formatted output string
    """
    code = csoundex.encode(text, include_tone=include_tone)

    if show_original:
        return f"{text}\t{code}"
    else:
        return code


def encode_file(csoundex: CSoundex, input_path: str, output_path: Optional[str],
                include_tone: bool, show_original: bool) -> None:
    """
    Encode text from input file.

    Args:
        csoundex: CSoundex encoder instance
        input_path: Input file path
        output_path: Output file path (None for stdout)
        include_tone: Whether to include tone
        show_original: Whether to show original text
    """
    input_file = Path(input_path)

    if not input_file.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Read input
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Encode each line
    results = []
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue

        try:
            encoded = encode_text(csoundex, line, include_tone, show_original)
            results.append(encoded)
        except Exception as e:
            logging.error(f"Error encoding line {line_num}: {e}")

    # Output
    if output_path:
        output_file = Path(output_path)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(results))
        print(f"Encoded {len(results)} lines to {output_path}")
    else:
        for result in results:
            print(result)


def encode_stdin(csoundex: CSoundex, include_tone: bool, show_original: bool) -> None:
    """
    Encode text from stdin.

    Args:
        csoundex: CSoundex encoder instance
        include_tone: Whether to include tone
        show_original: Whether to show original text
    """
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        encoded = encode_text(csoundex, line, include_tone, show_original)
        print(encoded)


def find_similar_texts(csoundex: CSoundex, query: str, candidates_file: str,
                       threshold: float, topk: Optional[int]) -> None:
    """
    Find similar texts from a file of candidates.

    Args:
        csoundex: CSoundex encoder instance
        query: Query text
        candidates_file: File containing candidate texts (one per line)
        threshold: Minimum similarity threshold
        topk: Maximum number of results
    """
    candidates_path = Path(candidates_file)

    if not candidates_path.exists():
        print(f"Error: Candidates file not found: {candidates_file}", file=sys.stderr)
        sys.exit(1)

    # Read candidates
    with open(candidates_path, 'r', encoding='utf-8') as f:
        candidates = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    # Find similar
    results = csoundex.find_similar(query, candidates, threshold=threshold, topk=topk)

    # Display results
    print(f"Query: {query}")
    print(f"Query Code: {csoundex.encode(query)}")
    print(f"\nSimilar texts (threshold={threshold}):")
    print("-" * 60)

    if not results:
        print("No similar texts found.")
    else:
        for i, (text, score) in enumerate(results, 1):
            code = csoundex.encode(text)
            print(f"{i:2d}. {text:20s} | Code: {code:20s} | Score: {score:.3f}")


def batch_similarity_matrix(csoundex: CSoundex, texts_file: str, output_path: Optional[str]) -> None:
    """
    Compute pairwise similarity matrix for a list of texts.

    Args:
        csoundex: CSoundex encoder instance
        texts_file: File containing texts (one per line)
        output_path: Output file for similarity matrix (CSV format)
    """
    texts_path = Path(texts_file)

    if not texts_path.exists():
        print(f"Error: Texts file not found: {texts_file}", file=sys.stderr)
        sys.exit(1)

    # Read texts
    with open(texts_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    # Compute similarity matrix
    n = len(texts)
    print(f"Computing similarity matrix for {n} texts...")

    matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                sim = 1.0
            else:
                sim = csoundex.similarity(texts[i], texts[j], mode='fuzzy')
            row.append(f"{sim:.3f}")
        matrix.append(row)

    # Output
    output_lines = []

    # Header
    header = "," + ",".join([f'"{t}"' for t in texts])
    output_lines.append(header)

    # Data rows
    for i, row in enumerate(matrix):
        line = f'"{texts[i]}",' + ",".join(row)
        output_lines.append(line)

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))
        print(f"Similarity matrix saved to {output_path}")
    else:
        for line in output_lines:
            print(line)


def show_cache_info(csoundex: CSoundex) -> None:
    """Display cache statistics."""
    info = csoundex.get_cache_info()
    print("Cache Statistics:")
    print(f"  Hits:    {info['hits']}")
    print(f"  Misses:  {info['misses']}")
    print(f"  Size:    {info['size']}/{info['maxsize']}")
    if info['hits'] + info['misses'] > 0:
        hit_rate = info['hits'] / (info['hits'] + info['misses']) * 100
        print(f"  Hit Rate: {hit_rate:.1f}%")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='CSoundex - Chinese Soundex Phonetic Encoding Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Encode text directly
  %(prog)s --text "三聚氰胺"

  # Encode with tone information
  %(prog)s --text "張三" --tone

  # Encode from file
  %(prog)s --file input.txt --output encoded.txt

  # Encode from stdin
  echo "信息檢索" | %(prog)s --stdin

  # Show original text with encoding
  %(prog)s --text "資訊檢索" --show-original

  # Find similar names
  %(prog)s --similar "張偉" --file names.txt --threshold 0.6 --topk 5

  # Compute similarity matrix
  %(prog)s --file names.txt --matrix similarity.csv

  # Show cache statistics
  %(prog)s --text "測試" --cache-info

For more information, see docs/guides/CSOUNDEX_DESIGN.md
        """
    )

    # Input sources (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--text', type=str, help='Text to encode')
    input_group.add_argument('--file', type=str, help='Input file path')
    input_group.add_argument('--stdin', action='store_true', help='Read from stdin')

    # Encoding options
    parser.add_argument('--tone', action='store_true',
                        help='Include tone in encoding (default: no tone)')
    parser.add_argument('--show-original', action='store_true',
                        help='Show original text with encoding')

    # Output options
    parser.add_argument('--output', '-o', type=str,
                        help='Output file path (default: stdout)')

    # Similarity search
    parser.add_argument('--similar', type=str,
                        help='Find similar texts to this query (requires --file)')
    parser.add_argument('--threshold', type=float, default=0.6,
                        help='Similarity threshold (default: 0.6)')
    parser.add_argument('--topk', type=int,
                        help='Return top-k results only')

    # Similarity matrix
    parser.add_argument('--matrix', type=str,
                        help='Output similarity matrix to CSV file (requires --file)')

    # Configuration
    parser.add_argument('--config', type=str,
                        help='Path to csoundex.yaml config file')
    parser.add_argument('--lexicon', type=str,
                        help='Path to pinyin lexicon TSV file')

    # Utility
    parser.add_argument('--cache-info', action='store_true',
                        help='Show cache statistics')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Verbose output')

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    # Initialize CSoundex
    try:
        csoundex = CSoundex(config_path=args.config, lexicon_path=args.lexicon)
    except Exception as e:
        print(f"Error initializing CSoundex: {e}", file=sys.stderr)
        sys.exit(1)

    # Execute based on mode
    try:
        if args.similar:
            # Similarity search mode
            if not args.file:
                print("Error: --similar requires --file", file=sys.stderr)
                sys.exit(1)
            find_similar_texts(csoundex, args.similar, args.file, args.threshold, args.topk)

        elif args.matrix:
            # Similarity matrix mode
            if not args.file:
                print("Error: --matrix requires --file", file=sys.stderr)
                sys.exit(1)
            batch_similarity_matrix(csoundex, args.file, args.matrix)

        elif args.text:
            # Direct text encoding
            encoded = encode_text(csoundex, args.text, args.tone, args.show_original)
            print(encoded)

        elif args.file:
            # File encoding
            encode_file(csoundex, args.file, args.output, args.tone, args.show_original)

        elif args.stdin:
            # Stdin encoding
            encode_stdin(csoundex, args.tone, args.show_original)

        # Show cache info if requested
        if args.cache_info:
            print("\n" + "=" * 60)
            show_cache_info(csoundex)

    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
