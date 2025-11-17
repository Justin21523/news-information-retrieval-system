#!/usr/bin/env python3
"""Document Summarization CLI Tool

Command-line tool for document summarization using various techniques.

Usage:
    # Static summarization (Lead-k)
    python scripts/summarize_doc.py --input doc.txt --method lead-k --k 3

    # Key sentence extraction
    python scripts/summarize_doc.py --input doc.txt --method key-sentence --k 5

    # Query-focused summarization
    python scripts/summarize_doc.py --input doc.txt --method query-focused --query "machine learning" --k 3

    # KWIC generation
    python scripts/summarize_doc.py --input doc.txt --method kwic --query "neural networks" --window 50

    # Multi-document summarization
    python scripts/summarize_doc.py --input-dir docs/ --method multi-doc --k 10

Author: Information Retrieval System
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ir.summarize.static import StaticSummarizer
from src.ir.summarize.dynamic import KWICGenerator


def read_file(file_path: str) -> str:
    """Read text file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


def read_directory(dir_path: str) -> list:
    """Read all text files in directory."""
    documents = []
    dir_path = Path(dir_path)

    for file_path in dir_path.glob('*.txt'):
        with open(file_path, 'r', encoding='utf-8') as f:
            documents.append(f.read())

    return documents


def write_output(output: str, file_path: str = None):
    """Write output to file or stdout."""
    if file_path:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"Output written to {file_path}")
    else:
        print(output)


def main():
    parser = argparse.ArgumentParser(description='Document Summarization Tool')

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', type=str,
                            help='Input document file')
    input_group.add_argument('--input-dir', type=str,
                            help='Input directory (for multi-document)')

    # Method selection
    parser.add_argument('--method', type=str, required=True,
                       choices=['lead-k', 'key-sentence', 'query-focused',
                               'multi-doc', 'kwic'],
                       help='Summarization method')

    # Common parameters
    parser.add_argument('--k', type=int, default=3,
                       help='Number of sentences to extract (default: 3)')
    parser.add_argument('--query', type=str,
                       help='Query for query-focused or KWIC')
    parser.add_argument('--output', type=str,
                       help='Output file (default: stdout)')

    # Static summarization options
    parser.add_argument('--position-bias', action='store_true',
                       help='Use position bias for key-sentence (default: True)')
    parser.add_argument('--no-position-bias', action='store_false',
                       dest='position_bias',
                       help='Disable position bias')
    parser.set_defaults(position_bias=True)

    parser.add_argument('--min-sentence-length', type=int, default=5,
                       help='Minimum sentence length in tokens (default: 5)')
    parser.add_argument('--max-sentence-length', type=int, default=100,
                       help='Maximum sentence length in tokens (default: 100)')

    # KWIC options
    parser.add_argument('--window', type=int, default=50,
                       help='KWIC window size (default: 50)')
    parser.add_argument('--window-type', type=str, default='fixed',
                       choices=['fixed', 'sentence', 'adaptive'],
                       help='KWIC window type (default: fixed)')
    parser.add_argument('--max-matches', type=int,
                       help='Maximum KWIC matches to display')
    parser.add_argument('--highlight-style', type=str, default='markdown',
                       choices=['markdown', 'ansi', 'html'],
                       help='KWIC highlight style (default: markdown)')
    parser.add_argument('--case-sensitive', action='store_true',
                       help='Case-sensitive KWIC matching')

    # Multi-document options
    parser.add_argument('--diversity-threshold', type=float, default=0.5,
                       help='Diversity threshold for multi-doc (default: 0.5)')

    # Display options
    parser.add_argument('--verbose', action='store_true',
                       help='Show detailed output')
    parser.add_argument('--show-scores', action='store_true',
                       help='Show sentence scores')

    args = parser.parse_args()

    # Validate query requirement
    if args.method in ['query-focused', 'kwic'] and not args.query:
        parser.error(f"--query is required for {args.method}")

    # ========================================================================
    # Static Summarization
    # ========================================================================

    if args.method in ['lead-k', 'key-sentence', 'query-focused']:
        # Read input
        if args.input:
            text = read_file(args.input)
        else:
            parser.error("--input required for this method")

        # Initialize summarizer
        summarizer = StaticSummarizer(
            min_sentence_length=args.min_sentence_length,
            max_sentence_length=args.max_sentence_length
        )

        # Generate summary
        if args.method == 'lead-k':
            if args.verbose:
                print(f"Running Lead-k summarization (k={args.k})...")
            summary = summarizer.lead_k_summarization(text, k=args.k)

        elif args.method == 'key-sentence':
            if args.verbose:
                print(f"Running key sentence extraction (k={args.k}, position_bias={args.position_bias})...")
            summary = summarizer.key_sentence_extraction(
                text, k=args.k, use_position_bias=args.position_bias
            )

        elif args.method == 'query-focused':
            if args.verbose:
                print(f"Running query-focused summarization (query='{args.query}', k={args.k})...")
            summary = summarizer.query_focused_summarization(
                text, query=args.query, k=args.k
            )

        # Format output
        lines = []
        lines.append("=" * 60)
        lines.append(f"Method: {summary.method}")
        lines.append(f"Sentences: {summary.length}/{summary.original_length}")
        lines.append(f"Compression: {summary.compression_ratio:.2%}")
        lines.append("=" * 60)
        lines.append("")

        # Display sentences
        for i, sent in enumerate(summary.sentences):
            if args.show_scores:
                lines.append(f"[{i+1}] (pos={sent.position}, score={sent.score:.3f})")
            else:
                lines.append(f"[{i+1}]")
            lines.append(sent.text)
            lines.append("")

        lines.append("=" * 60)
        lines.append("Summary:")
        lines.append("-" * 60)
        lines.append(summary.text)
        lines.append("=" * 60)

        output = '\n'.join(lines)
        write_output(output, args.output)

    # ========================================================================
    # Multi-Document Summarization
    # ========================================================================

    elif args.method == 'multi-doc':
        # Read documents
        if args.input_dir:
            documents = read_directory(args.input_dir)
        else:
            parser.error("--input-dir required for multi-doc")

        if not documents:
            print("Error: No text files found in directory")
            return

        if args.verbose:
            print(f"Loaded {len(documents)} documents")

        # Initialize summarizer
        summarizer = StaticSummarizer(
            min_sentence_length=args.min_sentence_length,
            max_sentence_length=args.max_sentence_length
        )

        # Generate summary
        if args.verbose:
            print(f"Running multi-document summarization (k={args.k}, diversity={args.diversity_threshold})...")
        summary = summarizer.multi_document_summarization(
            documents, k=args.k, diversity_threshold=args.diversity_threshold
        )

        # Format output
        lines = []
        lines.append("=" * 60)
        lines.append(f"Method: {summary.method}")
        lines.append(f"Documents: {len(documents)}")
        lines.append(f"Sentences: {summary.length}/{summary.original_length}")
        lines.append(f"Compression: {summary.compression_ratio:.2%}")
        lines.append("=" * 60)
        lines.append("")

        # Display sentences
        for i, sent in enumerate(summary.sentences):
            if args.show_scores:
                lines.append(f"[{i+1}] (doc={sent.doc_id}, pos={sent.position}, score={sent.score:.3f})")
            else:
                lines.append(f"[{i+1}] (doc={sent.doc_id})")
            lines.append(sent.text)
            lines.append("")

        lines.append("=" * 60)
        lines.append("Summary:")
        lines.append("-" * 60)
        lines.append(summary.text)
        lines.append("=" * 60)

        output = '\n'.join(lines)
        write_output(output, args.output)

    # ========================================================================
    # KWIC (Dynamic Summarization)
    # ========================================================================

    elif args.method == 'kwic':
        # Read input
        if args.input:
            text = read_file(args.input)

            # Initialize generator
            generator = KWICGenerator(
                window_size=args.window,
                window_type=args.window_type,
                case_sensitive=args.case_sensitive,
                enable_cache=True
            )

            # Generate KWIC
            if args.verbose:
                print(f"Running KWIC (query='{args.query}', window={args.window}, type={args.window_type})...")
            result = generator.generate(text, args.query, max_matches=args.max_matches)

            # Format output
            output = generator.format_results(
                result,
                max_display=args.max_matches,
                highlight_style=args.highlight_style
            )
            write_output(output, args.output)

        elif args.input_dir:
            # Multi-document KWIC
            documents = read_directory(args.input_dir)

            if not documents:
                print("Error: No text files found in directory")
                return

            if args.verbose:
                print(f"Loaded {len(documents)} documents")

            # Initialize generator
            generator = KWICGenerator(
                window_size=args.window,
                window_type=args.window_type,
                case_sensitive=args.case_sensitive,
                enable_cache=True
            )

            # Generate KWIC
            if args.verbose:
                print(f"Running multi-document KWIC (query='{args.query}')...")

            max_per_doc = args.max_matches // len(documents) if args.max_matches else 3
            result = generator.generate_multi(documents, args.query, max_matches_per_doc=max_per_doc)

            # Format output
            output = generator.format_results(
                result,
                max_display=args.max_matches,
                highlight_style=args.highlight_style
            )
            write_output(output, args.output)


if __name__ == '__main__':
    main()
