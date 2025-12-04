#!/usr/bin/env python3
"""
Keyword Extraction CLI Tool

This script provides a command-line interface for extracting keywords from
Chinese and English text using multiple algorithms.

Supported Algorithms:
    - TextRank: Graph-based ranking with position weighting
    - YAKE: Statistical feature-based extraction
    - RAKE: Rapid automatic keyword extraction
    - KeyBERT: BERT embeddings with MMR

Usage:
    # Extract from text
    python scripts/extract_keywords.py --text "機器學習是人工智慧的分支" --method textrank

    # Extract from file
    python scripts/extract_keywords.py --input doc.txt --method yake --top-k 10

    # Batch processing
    python scripts/extract_keywords.py --input-dir docs/ --method rake --output results.json

    # Evaluate with ground truth
    python scripts/extract_keywords.py --input doc.txt --ground-truth gt.txt --method all

    # Compare all methods
    python scripts/extract_keywords.py --text "深度學習" --method all --compare

Author: Information Retrieval System
License: Educational Use
"""

import argparse
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ir.keyextract import (
    TextRankExtractor,
    YAKEExtractor,
    RAKEExtractor,
    KeyBERTExtractor
)
from src.ir.keyextract.evaluator import KeywordEvaluator


class KeywordExtractionCLI:
    """CLI for keyword extraction."""

    def __init__(self, args):
        """Initialize CLI with arguments."""
        self.args = args
        self.logger = logging.getLogger(__name__)

        # Initialize extractors based on method
        self.extractors = {}

        if args.method in ['textrank', 'all']:
            self.extractors['textrank'] = TextRankExtractor(
                window_size=args.textrank_window,
                use_position_weight=args.textrank_position,
                pos_filter=['N', 'V'] if args.textrank_pos else None,
                tokenizer_engine=args.tokenizer
            )

        if args.method in ['yake', 'all']:
            self.extractors['yake'] = YAKEExtractor(
                language='zh' if args.language == 'chinese' else 'en',
                max_ngram_size=args.yake_ngram,
                tokenizer_engine=args.tokenizer
            )

        if args.method in ['rake', 'all']:
            self.extractors['rake'] = RAKEExtractor(
                max_length=args.rake_maxlen,
                ranking_metric=args.rake_metric,
                tokenizer_engine=args.tokenizer
            )

        if args.method in ['keybert', 'all']:
            try:
                self.extractors['keybert'] = KeyBERTExtractor(
                    model_name=args.keybert_model,
                    use_mmr=args.keybert_mmr,
                    diversity=args.keybert_diversity,
                    tokenizer_engine=args.tokenizer,
                    device=args.device
                )
            except ImportError as e:
                self.logger.warning(f"KeyBERT not available: {e}")
                if args.method == 'keybert':
                    print("Error: KeyBERT requires: pip install keybert sentence-transformers")
                    sys.exit(1)

        # Initialize evaluator if needed
        self.evaluator = None
        if args.ground_truth or args.evaluate:
            self.evaluator = KeywordEvaluator(k_values=[1, 3, 5, 10, 15])

    def extract_from_text(self, text: str, method: str) -> List[tuple]:
        """Extract keywords from text using specified method."""
        if method not in self.extractors:
            raise ValueError(f"Unknown method: {method}")

        extractor = self.extractors[method]

        start_time = time.time()
        keywords = extractor.extract(text, top_k=self.args.top_k)
        elapsed = time.time() - start_time

        # Convert to list of tuples
        results = [(kw.word, kw.score, kw.frequency) for kw in keywords]

        self.logger.info(
            f"{method}: Extracted {len(results)} keywords in {elapsed:.3f}s"
        )

        return results

    def extract_all_methods(self, text: str) -> Dict[str, List[tuple]]:
        """Extract keywords using all available methods."""
        results = {}

        for method in self.extractors.keys():
            results[method] = self.extract_from_text(text, method)

        return results

    def load_text(self, file_path: str) -> str:
        """Load text from file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(path, 'r', encoding='utf-8') as f:
            return f.read()

    def load_ground_truth(self, file_path: str) -> List[str]:
        """Load ground truth keywords from file (one per line)."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {file_path}")

        keywords = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    keywords.append(line)

        return keywords

    def format_output(self, results: Dict[str, List[tuple]], text: str = None) -> str:
        """Format extraction results for display."""
        output = []
        output.append("=" * 70)
        output.append("Keyword Extraction Results")
        output.append("=" * 70)

        if text and len(text) < 200:
            output.append(f"\nText: {text[:200]}...")

        for method, keywords in results.items():
            output.append(f"\n{method.upper()}:")
            output.append("-" * 70)

            for i, (word, score, freq) in enumerate(keywords, 1):
                output.append(
                    f"{i:2d}. {word:25s}  score={score:.4f}  freq={freq}"
                )

        output.append("=" * 70)

        return "\n".join(output)

    def compare_methods(self, results: Dict[str, List[tuple]]) -> str:
        """Create comparison table of all methods."""
        output = []
        output.append("\n" + "=" * 70)
        output.append("Method Comparison (Top 5 Keywords)")
        output.append("=" * 70)

        # Get top 5 from each method
        max_k = min(5, self.args.top_k)

        # Create table header
        methods = list(results.keys())
        header = "Rank | " + " | ".join(f"{m:15s}" for m in methods)
        output.append(header)
        output.append("-" * len(header))

        # Create rows
        for i in range(max_k):
            row = f"{i+1:4d} | "
            for method in methods:
                if i < len(results[method]):
                    word = results[method][i][0][:15]
                    row += f"{word:15s} | "
                else:
                    row += f"{'':15s} | "
            output.append(row)

        output.append("=" * 70)

        return "\n".join(output)

    def evaluate_results(self, extracted: List[str], ground_truth: List[str],
                        text: str = None) -> str:
        """Evaluate extraction results against ground truth."""
        if not self.evaluator:
            return ""

        result = self.evaluator.evaluate(extracted, ground_truth, text)

        output = []
        output.append("\n" + "=" * 70)
        output.append("Evaluation Results")
        output.append("=" * 70)

        # Precision, Recall, F1
        for k in [1, 3, 5, 10]:
            if k in result.precision_at_k:
                p = result.precision_at_k[k]
                r = result.recall_at_k[k]
                f1 = result.f1_at_k[k]
                output.append(
                    f"@{k:2d}: P={p:.4f}  R={r:.4f}  F1={f1:.4f}"
                )

        # MAP, MRR
        output.append(f"\nMAP  = {result.map_score:.4f}")
        output.append(f"MRR  = {result.mrr:.4f}")

        # nDCG
        output.append("\nnDCG@K:")
        for k in [1, 3, 5, 10]:
            if k in result.ndcg_at_k:
                output.append(f"  @{k:2d} = {result.ndcg_at_k[k]:.4f}")

        output.append("=" * 70)

        return "\n".join(output)

    def save_results(self, results: Dict, output_path: str):
        """Save results to JSON file."""
        path = Path(output_path)

        # Convert to serializable format
        serializable = {}
        for method, keywords in results.items():
            serializable[method] = [
                {
                    'word': word,
                    'score': float(score),
                    'frequency': int(freq)
                }
                for word, score, freq in keywords
            ]

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)

        self.logger.info(f"Results saved to {output_path}")

    def run(self):
        """Run the CLI application."""
        try:
            # Get input text
            if self.args.text:
                text = self.args.text
            elif self.args.input:
                text = self.load_text(self.args.input)
            else:
                print("Error: Must provide --text or --input")
                return 1

            # Extract keywords
            if self.args.method == 'all':
                results = self.extract_all_methods(text)
            else:
                keywords = self.extract_from_text(text, self.args.method)
                results = {self.args.method: keywords}

            # Format and display results
            if not self.args.quiet:
                print(self.format_output(results, text))

                if self.args.compare and len(results) > 1:
                    print(self.compare_methods(results))

            # Evaluate if ground truth provided
            if self.args.ground_truth:
                ground_truth = self.load_ground_truth(self.args.ground_truth)

                for method, keywords in results.items():
                    extracted_words = [kw[0] for kw in keywords]
                    eval_output = self.evaluate_results(
                        extracted_words, ground_truth, text
                    )
                    if not self.args.quiet:
                        print(f"\n{method.upper()} Evaluation:")
                        print(eval_output)

            # Save results if output specified
            if self.args.output:
                self.save_results(results, self.args.output)

            return 0

        except Exception as e:
            self.logger.error(f"Error: {e}", exc_info=True)
            print(f"Error: {e}")
            return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Extract keywords from Chinese and English text',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract from text with TextRank
  python scripts/extract_keywords.py --text "機器學習是人工智慧的分支" --method textrank

  # Extract from file with YAKE
  python scripts/extract_keywords.py --input document.txt --method yake --top-k 15

  # Compare all methods
  python scripts/extract_keywords.py --input doc.txt --method all --compare

  # Evaluate with ground truth
  python scripts/extract_keywords.py --input doc.txt --ground-truth keywords.txt --method textrank

  # Save results to JSON
  python scripts/extract_keywords.py --input doc.txt --method all --output results.json
        """
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--text',
        type=str,
        help='Input text directly'
    )
    input_group.add_argument(
        '--input', '-i',
        type=str,
        help='Input file path'
    )

    # Method selection
    parser.add_argument(
        '--method', '-m',
        type=str,
        choices=['textrank', 'yake', 'rake', 'keybert', 'all'],
        default='textrank',
        help='Extraction method (default: textrank)'
    )

    # General options
    parser.add_argument(
        '--top-k', '-k',
        type=int,
        default=10,
        help='Number of keywords to extract (default: 10)'
    )

    parser.add_argument(
        '--language', '-l',
        type=str,
        choices=['chinese', 'english'],
        default='chinese',
        help='Text language (default: chinese)'
    )

    parser.add_argument(
        '--tokenizer',
        type=str,
        choices=['ckip', 'jieba', 'auto'],
        default='jieba',
        help='Chinese tokenizer (default: jieba)'
    )

    # TextRank options
    parser.add_argument(
        '--textrank-window',
        type=int,
        default=5,
        help='TextRank co-occurrence window size (default: 5)'
    )

    parser.add_argument(
        '--textrank-position',
        action='store_true',
        help='Enable TextRank position weighting'
    )

    parser.add_argument(
        '--textrank-pos',
        action='store_true',
        help='Enable TextRank POS filtering (nouns & verbs)'
    )

    # YAKE options
    parser.add_argument(
        '--yake-ngram',
        type=int,
        default=3,
        help='YAKE max n-gram size (default: 3)'
    )

    # RAKE options
    parser.add_argument(
        '--rake-maxlen',
        type=int,
        default=4,
        help='RAKE max phrase length (default: 4)'
    )

    parser.add_argument(
        '--rake-metric',
        type=str,
        choices=['degree_to_frequency', 'word_degree', 'word_frequency'],
        default='degree_to_frequency',
        help='RAKE ranking metric (default: degree_to_frequency)'
    )

    # KeyBERT options
    parser.add_argument(
        '--keybert-model',
        type=str,
        default='paraphrase-multilingual-MiniLM-L12-v2',
        help='KeyBERT model name (default: multilingual MiniLM)'
    )

    parser.add_argument(
        '--keybert-mmr',
        action='store_true',
        default=True,
        help='Enable KeyBERT MMR (default: enabled)'
    )

    parser.add_argument(
        '--keybert-diversity',
        type=float,
        default=0.5,
        help='KeyBERT diversity parameter (0.0-1.0, default: 0.5)'
    )

    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda'],
        default='cpu',
        help='Device for KeyBERT (default: cpu)'
    )

    # Evaluation options
    parser.add_argument(
        '--ground-truth', '-g',
        type=str,
        help='Ground truth keywords file (one per line)'
    )

    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Enable evaluation mode'
    )

    # Output options
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file path (JSON format)'
    )

    parser.add_argument(
        '--compare',
        action='store_true',
        help='Show comparison table (with --method all)'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress output (only save to file)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose logging'
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run CLI
    cli = KeywordExtractionCLI(args)
    sys.exit(cli.run())


if __name__ == '__main__':
    main()
