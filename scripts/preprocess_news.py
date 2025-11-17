#!/usr/bin/env python3
"""
News Preprocessing Pipeline for CNIRS Project

This script performs comprehensive NLP preprocessing on news articles:
1. Chinese tokenization (Jieba/CKIP)
2. Named Entity Recognition (NER)
3. Keyword extraction (TextRank)
4. Automatic summarization (Lead-k)
5. Output enriched JSONL with NLP fields

Usage:
    python scripts/preprocess_news.py --input data/processed/cna_mvp_cleaned.jsonl \
                                      --output data/preprocessed/cna_mvp_preprocessed.jsonl \
                                      --report data/stats/preprocessing_report.txt

Complexity:
    Time: O(N*n) where N=number of articles, n=average article length
    Space: O(N) for storing processed articles

Author: Information Retrieval System
License: Educational Use
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from collections import Counter, defaultdict
from datetime import datetime
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import IR modules
from src.ir.text.chinese_tokenizer import ChineseTokenizer
from src.ir.text.ner_extractor import NERExtractor
from src.ir.keyextract.textrank import TextRankExtractor
from src.ir.summarize.static import StaticSummarizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingStats:
    """Statistics for preprocessing operation."""
    total_articles: int = 0
    successful: int = 0
    failed: int = 0
    total_tokens: int = 0
    total_entities: int = 0
    total_keywords: int = 0
    avg_tokens_per_article: float = 0.0
    avg_entities_per_article: float = 0.0
    avg_keywords_per_article: float = 0.0
    processing_time: float = 0.0
    errors: List[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []


class NewsPreprocessor:
    """
    News preprocessing pipeline integrating multiple NLP components.

    Attributes:
        tokenizer: ChineseTokenizer for word segmentation
        ner_extractor: NERExtractor for entity recognition
        keyword_extractor: TextRankExtractor for keyword extraction
        summarizer: StaticSummarizer for text summarization

    Examples:
        >>> preprocessor = NewsPreprocessor(engine='jieba')
        >>> article = {"title": "新聞標題", "content": "新聞內容..."}
        >>> processed = preprocessor.process_article(article)
        >>> print(processed['keywords'])
        ['關鍵詞1', '關鍵詞2', '關鍵詞3']
    """

    def __init__(self,
                 tokenizer_engine: str = 'jieba',
                 num_keywords: int = 5,
                 summary_sentences: int = 3,
                 device: int = -1):
        """
        Initialize preprocessing pipeline.

        Args:
            tokenizer_engine: 'jieba' (fast) or 'ckip' (accurate)
            num_keywords: Number of keywords to extract
            summary_sentences: Number of sentences in summary (Lead-k)
            device: GPU device (-1 for CPU)
        """
        logger.info(f"Initializing NewsPreprocessor with engine={tokenizer_engine}")

        # Initialize tokenizer
        self.tokenizer = ChineseTokenizer(
            engine=tokenizer_engine,
            mode='default',
            device=device
        )

        # Initialize NER extractor (CKIP-based, may fallback to tokenizer-based)
        try:
            self.ner_extractor = NERExtractor(device=device)
            self.ner_available = True
            logger.info("NER extractor initialized successfully")
        except Exception as e:
            logger.warning(f"NER extractor initialization failed: {e}")
            logger.warning("NER features will be disabled")
            self.ner_extractor = None
            self.ner_available = False

        # Initialize keyword extractor
        self.keyword_extractor = TextRankExtractor(
            tokenizer_engine=tokenizer_engine,
            window_size=5,
            damping_factor=0.85,
            max_iterations=100,
            device=device
        )

        # Initialize summarizer
        self.summarizer = StaticSummarizer()

        self.num_keywords = num_keywords
        self.summary_sentences = summary_sentences

        logger.info("NewsPreprocessor initialized successfully")

    def process_article(self, article: Dict) -> Dict:
        """
        Process a single news article with NLP pipeline.

        Args:
            article: Dictionary containing 'title', 'content', and metadata

        Returns:
            Enriched article dictionary with NLP fields:
                - tokens_title: List of tokens from title
                - tokens_content: List of tokens from content
                - entities: List of named entities (if NER available)
                - keywords: List of extracted keywords
                - summary: Lead-k summary text

        Raises:
            KeyError: If required fields missing
            Exception: For other processing errors
        """
        try:
            # Validate required fields
            if 'title' not in article or 'content' not in article:
                raise KeyError("Article missing required fields: 'title' or 'content'")

            title = article['title']
            content = article['content']
            full_text = f"{title} {content}"

            # 1. Tokenization
            tokens_title = self.tokenizer.tokenize(title)
            tokens_content = self.tokenizer.tokenize(content)

            # 2. Named Entity Recognition (optional)
            entities = []
            if self.ner_available and self.ner_extractor:
                try:
                    entity_objs = self.ner_extractor.extract(full_text)
                    entities = [
                        {
                            'text': e.text,
                            'type': e.type,
                            'start_pos': e.start_pos,
                            'end_pos': e.end_pos
                        }
                        for e in entity_objs
                    ]
                except Exception as e:
                    logger.debug(f"NER extraction failed for article {article.get('article_id', 'unknown')}: {e}")
                    entities = []

            # 3. Keyword Extraction
            keywords = []
            try:
                keyword_objs = self.keyword_extractor.extract(
                    full_text,
                    top_k=self.num_keywords
                )
                keywords = [kw.word for kw in keyword_objs]
            except Exception as e:
                logger.debug(f"Keyword extraction failed: {e}")
                keywords = []

            # 4. Summarization (Lead-k)
            summary = ""
            try:
                summary_obj = self.summarizer.lead_k(
                    content,
                    k=self.summary_sentences
                )
                summary = ' '.join([s.text for s in summary_obj.sentences])
            except Exception as e:
                logger.debug(f"Summarization failed: {e}")
                # Fallback: use first N characters
                summary = content[:200] if len(content) > 200 else content

            # Create enriched article
            enriched = article.copy()
            enriched.update({
                'tokens_title': tokens_title,
                'tokens_content': tokens_content,
                'entities': entities,
                'keywords': keywords,
                'summary': summary,
                'processed_at': datetime.now().isoformat()
            })

            return enriched

        except Exception as e:
            logger.error(f"Failed to process article {article.get('article_id', 'unknown')}: {e}")
            raise

    def process_batch(self,
                     articles: List[Dict],
                     verbose: bool = True) -> tuple[List[Dict], ProcessingStats]:
        """
        Process multiple articles in batch.

        Args:
            articles: List of article dictionaries
            verbose: Show progress bar

        Returns:
            Tuple of (processed_articles, statistics)
        """
        stats = ProcessingStats()
        stats.total_articles = len(articles)
        processed_articles = []

        start_time = time.time()

        for i, article in enumerate(articles):
            try:
                # Process article
                enriched = self.process_article(article)
                processed_articles.append(enriched)

                # Update stats
                stats.successful += 1
                stats.total_tokens += len(enriched['tokens_content'])
                stats.total_entities += len(enriched['entities'])
                stats.total_keywords += len(enriched['keywords'])

                # Progress logging
                if verbose and (i + 1) % 10 == 0:
                    logger.info(f"Processed {i + 1}/{len(articles)} articles")

            except Exception as e:
                stats.failed += 1
                stats.errors.append(f"Article {article.get('article_id', i)}: {str(e)}")
                logger.error(f"Failed to process article {i}: {e}")

        # Calculate averages
        if stats.successful > 0:
            stats.avg_tokens_per_article = stats.total_tokens / stats.successful
            stats.avg_entities_per_article = stats.total_entities / stats.successful
            stats.avg_keywords_per_article = stats.total_keywords / stats.successful

        stats.processing_time = time.time() - start_time

        return processed_articles, stats


def load_jsonl(file_path: Path) -> List[Dict]:
    """Load articles from JSONL file."""
    articles = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                articles.append(json.loads(line))
    return articles


def save_jsonl(articles: List[Dict], file_path: Path):
    """Save articles to JSONL file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for article in articles:
            f.write(json.dumps(article, ensure_ascii=False) + '\n')


def generate_report(stats: ProcessingStats,
                   articles: List[Dict],
                   output_path: Path):
    """Generate preprocessing report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect additional statistics
    entity_types = Counter()
    keyword_freq = Counter()

    for article in articles:
        for entity in article.get('entities', []):
            entity_types[entity['type']] += 1
        for keyword in article.get('keywords', []):
            keyword_freq[keyword] += 1

    # Generate report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("NEWS PREPROCESSING REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("PROCESSING STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total articles:           {stats.total_articles}\n")
        f.write(f"Successfully processed:   {stats.successful}\n")
        f.write(f"Failed:                   {stats.failed}\n")
        f.write(f"Success rate:             {stats.successful/stats.total_articles*100:.2f}%\n")
        f.write(f"Processing time:          {stats.processing_time:.2f} seconds\n")
        f.write(f"Avg time per article:     {stats.processing_time/stats.total_articles:.3f} seconds\n\n")

        f.write("NLP FEATURES STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total tokens extracted:   {stats.total_tokens}\n")
        f.write(f"Avg tokens per article:   {stats.avg_tokens_per_article:.1f}\n")
        f.write(f"Total entities found:     {stats.total_entities}\n")
        f.write(f"Avg entities per article: {stats.avg_entities_per_article:.1f}\n")
        f.write(f"Total keywords extracted: {stats.total_keywords}\n")
        f.write(f"Avg keywords per article: {stats.avg_keywords_per_article:.1f}\n\n")

        f.write("ENTITY TYPE DISTRIBUTION\n")
        f.write("-" * 80 + "\n")
        for entity_type, count in entity_types.most_common(15):
            f.write(f"{entity_type:15s} {count:5d}\n")
        f.write("\n")

        f.write("TOP 20 KEYWORDS\n")
        f.write("-" * 80 + "\n")
        for keyword, count in keyword_freq.most_common(20):
            f.write(f"{keyword:30s} {count:3d}\n")
        f.write("\n")

        if stats.errors:
            f.write("ERRORS\n")
            f.write("-" * 80 + "\n")
            for error in stats.errors[:10]:  # Show first 10 errors
                f.write(f"- {error}\n")
            if len(stats.errors) > 10:
                f.write(f"... and {len(stats.errors) - 10} more errors\n")

    logger.info(f"Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess news articles with NLP pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/preprocess_news.py \\
      --input data/processed/cna_mvp_cleaned.jsonl \\
      --output data/preprocessed/cna_mvp_preprocessed.jsonl

  # With custom settings
  python scripts/preprocess_news.py \\
      --input data/processed/cna_mvp_cleaned.jsonl \\
      --output data/preprocessed/cna_mvp_preprocessed.jsonl \\
      --engine jieba \\
      --keywords 10 \\
      --summary-sentences 5 \\
      --report data/stats/preprocessing_report.txt
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input JSONL file (cleaned articles)'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output JSONL file (preprocessed articles)'
    )

    parser.add_argument(
        '--report',
        type=str,
        default=None,
        help='Output path for preprocessing report (optional)'
    )

    parser.add_argument(
        '--engine',
        type=str,
        choices=['jieba', 'ckip', 'auto'],
        default='jieba',
        help='Tokenization engine (default: jieba for speed)'
    )

    parser.add_argument(
        '--keywords',
        type=int,
        default=5,
        help='Number of keywords to extract (default: 5)'
    )

    parser.add_argument(
        '--summary-sentences',
        type=int,
        default=3,
        help='Number of sentences in summary (default: 3)'
    )

    parser.add_argument(
        '--device',
        type=int,
        default=-1,
        help='GPU device (-1 for CPU, 0+ for GPU)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate paths
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    # Load articles
    logger.info(f"Loading articles from {input_path}")
    articles = load_jsonl(input_path)
    logger.info(f"Loaded {len(articles)} articles")

    # Initialize preprocessor
    preprocessor = NewsPreprocessor(
        tokenizer_engine=args.engine,
        num_keywords=args.keywords,
        summary_sentences=args.summary_sentences,
        device=args.device
    )

    # Process articles
    logger.info("Starting preprocessing...")
    processed_articles, stats = preprocessor.process_batch(
        articles,
        verbose=True
    )

    # Save results
    logger.info(f"Saving preprocessed articles to {output_path}")
    save_jsonl(processed_articles, output_path)

    # Generate report
    if args.report:
        report_path = Path(args.report)
    else:
        report_path = output_path.parent.parent / 'stats' / 'preprocessing_report.txt'

    generate_report(stats, processed_articles, report_path)

    # Print summary
    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETED")
    print("=" * 80)
    print(f"Input:        {input_path}")
    print(f"Output:       {output_path}")
    print(f"Report:       {report_path}")
    print(f"Processed:    {stats.successful}/{stats.total_articles} articles")
    print(f"Success rate: {stats.successful/stats.total_articles*100:.2f}%")
    print(f"Time:         {stats.processing_time:.2f} seconds")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
