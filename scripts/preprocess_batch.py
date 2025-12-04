#!/usr/bin/env python3
"""
Batch Preprocessing for CNIRS

This script performs batch preprocessing on merged JSONL data:
- CKIP tokenization (word segmentation)
- Field standardization
- Progress tracking with checkpoints

Usage:
    python scripts/preprocess_batch.py \
        --input data/raw/merged_14days.jsonl \
        --output data/preprocessed/merged_14days_preprocessed.jsonl

Author: CNIRS Project
"""

import json
import argparse
import logging
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PreprocessStats:
    """Statistics for preprocessing."""
    total: int = 0
    processed: int = 0
    skipped: int = 0
    errors: int = 0
    start_time: float = 0.0


class BatchPreprocessor:
    """
    Batch preprocessor for news articles.

    Performs CKIP tokenization and field standardization.
    """

    def __init__(self, batch_size: int = 32, use_ckip: bool = True):
        """
        Initialize the preprocessor.

        Args:
            batch_size: Number of documents to process at once
            use_ckip: Whether to use CKIP tokenizer (if False, use jieba)
        """
        self.batch_size = batch_size
        self.use_ckip = use_ckip
        self.tokenizer = None
        self._init_tokenizer()

    def _init_tokenizer(self):
        """Initialize the tokenizer."""
        if self.use_ckip:
            try:
                from src.ir.text.ckip_tokenizer import CKIPTokenizer
                logger.info("Initializing CKIP tokenizer...")
                self.tokenizer = CKIPTokenizer(
                    model_name="bert-base",
                    use_gpu=False
                )
                logger.info("CKIP tokenizer initialized successfully")
            except Exception as e:
                logger.warning(f"CKIP init failed: {e}, falling back to jieba")
                self.use_ckip = False

        if not self.use_ckip:
            try:
                import jieba
                jieba.setLogLevel(logging.WARNING)
                self.tokenizer = jieba
                logger.info("Using jieba tokenizer")
            except ImportError:
                logger.error("Neither CKIP nor jieba available")
                raise RuntimeError("No tokenizer available")

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        if not text or not text.strip():
            return []

        try:
            if self.use_ckip:
                return self.tokenizer.tokenize(text)
            else:
                return list(self.tokenizer.cut(text))
        except Exception as e:
            logger.warning(f"Tokenization error: {e}")
            return []

    def tokenize_batch(self, texts: List[str]) -> List[List[str]]:
        """
        Tokenize a batch of texts.

        Args:
            texts: List of input texts

        Returns:
            List of token lists
        """
        if self.use_ckip and hasattr(self.tokenizer, 'tokenize_batch'):
            try:
                return self.tokenizer.tokenize_batch(texts)
            except Exception as e:
                logger.warning(f"Batch tokenization failed: {e}, falling back to sequential")

        # Sequential fallback
        return [self.tokenize(text) for text in texts]

    def preprocess_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess a single record.

        Args:
            record: Input record

        Returns:
            Preprocessed record
        """
        preprocessed = record.copy()

        # Tokenize title
        title = record.get('title', '')
        preprocessed['tokens_title'] = self.tokenize(title)

        # Tokenize content
        content = record.get('content', '')
        preprocessed['tokens_content'] = self.tokenize(content)

        # Ensure required fields exist
        if 'source' not in preprocessed:
            preprocessed['source'] = ''
        if 'category' not in preprocessed:
            preprocessed['category'] = ''
        if 'category_name' not in preprocessed:
            preprocessed['category_name'] = preprocessed.get('category', '')
        if 'published_date' not in preprocessed:
            preprocessed['published_date'] = ''
        if 'author' not in preprocessed:
            preprocessed['author'] = ''
        if 'tags' not in preprocessed:
            preprocessed['tags'] = []

        return preprocessed

    def process_file(
        self,
        input_file: Path,
        output_file: Path,
        checkpoint_file: Optional[Path] = None,
        limit: Optional[int] = None
    ) -> PreprocessStats:
        """
        Process an entire JSONL file.

        Args:
            input_file: Input JSONL file
            output_file: Output preprocessed JSONL file
            checkpoint_file: Optional checkpoint file for resume
            limit: Optional limit on number of records to process

        Returns:
            Processing statistics
        """
        stats = PreprocessStats(start_time=time.time())

        # Check for checkpoint
        start_line = 0
        if checkpoint_file and checkpoint_file.exists():
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                start_line = checkpoint.get('processed', 0)
                logger.info(f"Resuming from checkpoint: line {start_line}")

        # Open output file (append if resuming)
        mode = 'a' if start_line > 0 else 'w'

        # Count total lines
        with open(input_file, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
        stats.total = min(total_lines, limit) if limit else total_lines

        logger.info(f"Processing {stats.total} records...")

        batch_texts_title = []
        batch_texts_content = []
        batch_records = []

        with open(input_file, 'r', encoding='utf-8') as inf, \
             open(output_file, mode, encoding='utf-8') as outf:

            for line_num, line in enumerate(inf):
                # Skip already processed lines
                if line_num < start_line:
                    continue

                # Check limit
                if limit and stats.processed >= limit:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                    batch_records.append(record)
                    batch_texts_title.append(record.get('title', ''))
                    batch_texts_content.append(record.get('content', ''))

                    # Process batch
                    if len(batch_records) >= self.batch_size:
                        self._process_batch(
                            batch_records,
                            batch_texts_title,
                            batch_texts_content,
                            outf,
                            stats
                        )
                        batch_records = []
                        batch_texts_title = []
                        batch_texts_content = []

                        # Progress update
                        if stats.processed % 500 == 0:
                            elapsed = time.time() - stats.start_time
                            rate = stats.processed / elapsed if elapsed > 0 else 0
                            remaining = (stats.total - stats.processed) / rate if rate > 0 else 0
                            logger.info(
                                f"Progress: {stats.processed}/{stats.total} "
                                f"({100*stats.processed/stats.total:.1f}%) "
                                f"Rate: {rate:.1f}/s "
                                f"ETA: {remaining/60:.1f}min"
                            )

                            # Save checkpoint
                            if checkpoint_file:
                                with open(checkpoint_file, 'w') as cf:
                                    json.dump({'processed': stats.processed}, cf)

                except json.JSONDecodeError as e:
                    stats.errors += 1
                    logger.warning(f"JSON error at line {line_num}: {e}")
                except Exception as e:
                    stats.errors += 1
                    logger.warning(f"Error at line {line_num}: {e}")

            # Process remaining batch
            if batch_records:
                self._process_batch(
                    batch_records,
                    batch_texts_title,
                    batch_texts_content,
                    outf,
                    stats
                )

        # Remove checkpoint file on success
        if checkpoint_file and checkpoint_file.exists():
            checkpoint_file.unlink()

        return stats

    def _process_batch(
        self,
        records: List[Dict],
        titles: List[str],
        contents: List[str],
        outf,
        stats: PreprocessStats
    ):
        """Process a batch of records."""
        try:
            # Batch tokenize if CKIP supports it
            if self.use_ckip and hasattr(self.tokenizer, 'tokenize_batch'):
                tokens_titles = self.tokenizer.tokenize_batch(titles)
                tokens_contents = self.tokenizer.tokenize_batch(contents)
            else:
                tokens_titles = [self.tokenize(t) for t in titles]
                tokens_contents = [self.tokenize(c) for c in contents]

            for i, record in enumerate(records):
                preprocessed = record.copy()
                preprocessed['tokens_title'] = tokens_titles[i] if i < len(tokens_titles) else []
                preprocessed['tokens_content'] = tokens_contents[i] if i < len(tokens_contents) else []

                # Ensure required fields
                if 'category_name' not in preprocessed:
                    preprocessed['category_name'] = preprocessed.get('category', '')

                outf.write(json.dumps(preprocessed, ensure_ascii=False) + '\n')
                stats.processed += 1

        except Exception as e:
            logger.error(f"Batch processing error: {e}")
            # Fallback to sequential processing
            for record in records:
                try:
                    preprocessed = self.preprocess_record(record)
                    outf.write(json.dumps(preprocessed, ensure_ascii=False) + '\n')
                    stats.processed += 1
                except Exception as e2:
                    stats.errors += 1
                    logger.warning(f"Record error: {e2}")


def main():
    parser = argparse.ArgumentParser(description='Batch preprocess JSONL files')
    parser.add_argument(
        '--input',
        type=str,
        default='data/raw/merged_14days.jsonl',
        help='Input JSONL file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/preprocessed/merged_14days_preprocessed.jsonl',
        help='Output preprocessed JSONL file'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for processing'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of records to process'
    )
    parser.add_argument(
        '--no-ckip',
        action='store_true',
        help='Use jieba instead of CKIP'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from checkpoint if available'
    )

    args = parser.parse_args()

    input_file = Path(args.input)
    output_file = Path(args.output)
    checkpoint_file = output_file.with_suffix('.checkpoint') if args.resume else None

    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return 1

    # Create output directory
    output_file.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("CNIRS Batch Preprocessor")
    logger.info("=" * 60)
    logger.info(f"Input: {input_file}")
    logger.info(f"Output: {output_file}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Tokenizer: {'jieba' if args.no_ckip else 'CKIP'}")
    logger.info("")

    # Initialize preprocessor
    preprocessor = BatchPreprocessor(
        batch_size=args.batch_size,
        use_ckip=not args.no_ckip
    )

    # Process file
    stats = preprocessor.process_file(
        input_file,
        output_file,
        checkpoint_file,
        args.limit
    )

    # Print summary
    elapsed = time.time() - stats.start_time
    logger.info("")
    logger.info("=" * 60)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total records: {stats.total}")
    logger.info(f"Processed: {stats.processed}")
    logger.info(f"Errors: {stats.errors}")
    logger.info(f"Elapsed time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    logger.info(f"Rate: {stats.processed/elapsed:.1f} records/second")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Output size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")

    return 0


if __name__ == '__main__':
    exit(main())
