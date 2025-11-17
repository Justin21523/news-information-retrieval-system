#!/usr/bin/env python3
"""
Test Queries and QRELS Generator for CNIRS Project

This script creates test queries and relevance judgments (QRELS) for
evaluating the retrieval system.

Approach:
1. Analyze existing articles to identify key topics
2. Generate queries based on article titles/keywords
3. Create QRELS by matching queries to relevant articles

Usage:
    python scripts/create_test_queries.py \\
        --input data/preprocessed/cna_mvp_preprocessed.jsonl \\
        --output-queries data/evaluation/test_queries.txt \\
        --output-qrels data/evaluation/qrels.txt \\
        --num-queries 15

Author: Information Retrieval System
License: Educational Use
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QueryGenerator:
    """
    Generate test queries from article collection.

    Creates realistic queries based on article topics and keywords,
    along with relevance judgments (QRELS).
    """

    def __init__(self):
        """Initialize query generator."""
        self.articles = []
        self.queries = []
        self.qrels = []  # (query_id, 0, doc_id, relevance)

    def load_articles(self, file_path: Path) -> List[Dict]:
        """Load preprocessed articles."""
        logger.info(f"Loading articles from {file_path}")
        articles = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    articles.append(json.loads(line))

        self.articles = articles
        logger.info(f"Loaded {len(articles)} articles")
        return articles

    def analyze_topics(self) -> Dict:
        """Analyze article topics and keywords."""
        logger.info("Analyzing article topics...")

        # Collect all keywords
        all_keywords = []
        all_entities = []
        all_tags = []

        for article in self.articles:
            keywords = article.get('keywords', [])
            # Filter out punctuation keywords
            keywords = [kw for kw in keywords if len(kw) > 1 and not kw.isspace()]
            all_keywords.extend(keywords)

            entities = article.get('entities', [])
            all_entities.extend([e['text'] for e in entities if len(e['text']) > 1])

            tags = article.get('tags', [])
            all_tags.extend(tags)

        # Get top keywords/entities/tags
        keyword_freq = Counter(all_keywords)
        entity_freq = Counter(all_entities)
        tag_freq = Counter(all_tags)

        topics = {
            'top_keywords': keyword_freq.most_common(30),
            'top_entities': entity_freq.most_common(30),
            'top_tags': tag_freq.most_common(20)
        }

        logger.info(f"Found {len(keyword_freq)} unique keywords")
        logger.info(f"Found {len(entity_freq)} unique entities")
        logger.info(f"Found {len(tag_freq)} unique tags")

        return topics

    def generate_queries_manual(self, num_queries: int = 15) -> List[Dict]:
        """
        Generate queries manually based on dataset analysis.

        This creates high-quality queries for the specific dataset.
        """
        logger.info(f"Generating {num_queries} manual queries...")

        # Analyze topics first
        topics = self.analyze_topics()

        # Manual query templates based on the dataset
        # (These are customized for the CNA political news dataset)
        manual_queries = [
            # Simple keyword queries
            {"id": "Q001", "text": "颱風災害", "type": "simple"},
            {"id": "Q002", "text": "淹水救援", "type": "simple"},
            {"id": "Q003", "text": "人工智慧", "type": "simple"},
            {"id": "Q004", "text": "中國政策", "type": "simple"},
            {"id": "Q005", "text": "美國關係", "type": "simple"},

            # Entity queries
            {"id": "Q006", "text": "蘇澳", "type": "entity"},
            {"id": "Q007", "text": "宜蘭", "type": "entity"},
            {"id": "Q008", "text": "台灣", "type": "entity"},

            # Phrase queries
            {"id": "Q009", "text": "颱風鳳凰影響", "type": "phrase"},
            {"id": "Q010", "text": "氣候變遷對策", "type": "phrase"},

            # Multi-word queries
            {"id": "Q011", "text": "災害 救援 停班停課", "type": "multi"},
            {"id": "Q012", "text": "AI 技術 發展", "type": "multi"},
            {"id": "Q013", "text": "兩岸 關係 政策", "type": "multi"},

            # Topic queries
            {"id": "Q014", "text": "經濟政策", "type": "topic"},
            {"id": "Q015", "text": "國防安全", "type": "topic"},
        ]

        # Select queries
        self.queries = manual_queries[:num_queries]

        logger.info(f"Generated {len(self.queries)} queries")
        return self.queries

    def create_qrels_automatic(self, use_keywords: bool = True):
        """
        Create QRELS automatically by matching queries to articles.

        Uses keyword overlap and entity matching to determine relevance.
        """
        logger.info("Creating QRELS automatically...")

        for query in self.queries:
            query_id = query['id']
            query_text = query['text']
            query_terms = set(query_text.split())

            relevance_scores = []

            for idx, article in enumerate(self.articles):
                article_id = article['article_id']
                score = 0

                # Title matching (high weight)
                title = article.get('title', '')
                if any(term in title for term in query_terms):
                    score += 3

                # Content matching
                content = article.get('content', '')
                for term in query_terms:
                    if term in content:
                        score += content.count(term)

                # Keyword matching
                if use_keywords:
                    keywords = article.get('keywords', [])
                    for term in query_terms:
                        if term in keywords:
                            score += 2

                # Entity matching
                entities = article.get('entities', [])
                entity_texts = [e['text'] for e in entities]
                for term in query_terms:
                    if term in entity_texts:
                        score += 2

                # Tags matching
                tags = article.get('tags', [])
                for term in query_terms:
                    if term in tags:
                        score += 2

                relevance_scores.append((article_id, score))

            # Sort by score and assign relevance labels
            relevance_scores.sort(key=lambda x: x[1], reverse=True)

            for rank, (article_id, score) in enumerate(relevance_scores):
                # Assign relevance: 2 = highly relevant, 1 = relevant, 0 = not relevant
                if score >= 5:
                    relevance = 2
                elif score >= 2:
                    relevance = 1
                else:
                    relevance = 0

                # Only add if relevant or in top 20
                if relevance > 0 or rank < 20:
                    self.qrels.append({
                        'query_id': query_id,
                        'doc_id': article_id,
                        'relevance': relevance,
                        'score': score
                    })

        logger.info(f"Created {len(self.qrels)} relevance judgments")

    def save_queries(self, output_path: Path):
        """Save queries in standard format."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for query in self.queries:
                f.write(f"{query['id']}\t{query['text']}\n")

        logger.info(f"Saved {len(self.queries)} queries to {output_path}")

    def save_qrels(self, output_path: Path):
        """Save QRELS in TREC format."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for qrel in self.qrels:
                # TREC format: query_id 0 doc_id relevance
                f.write(f"{qrel['query_id']} 0 {qrel['doc_id']} {qrel['relevance']}\n")

        logger.info(f"Saved {len(self.qrels)} qrels to {output_path}")

    def save_qrels_detailed(self, output_path: Path):
        """Save detailed QRELS with scores (for analysis)."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("query_id\tdoc_id\trelevance\tscore\n")
            for qrel in self.qrels:
                f.write(f"{qrel['query_id']}\t{qrel['doc_id']}\t{qrel['relevance']}\t{qrel['score']}\n")

        logger.info(f"Saved detailed qrels to {output_path}")

    def generate_statistics(self) -> Dict:
        """Generate statistics about queries and qrels."""
        stats = {
            'num_queries': len(self.queries),
            'num_qrels': len(self.qrels),
            'avg_relevant_per_query': 0,
            'queries_by_type': Counter(),
            'relevance_distribution': Counter()
        }

        # Count by query type
        for query in self.queries:
            stats['queries_by_type'][query['type']] += 1

        # Count relevant docs per query
        relevant_counts = Counter(qrel['query_id'] for qrel in self.qrels if qrel['relevance'] > 0)
        if relevant_counts:
            stats['avg_relevant_per_query'] = sum(relevant_counts.values()) / len(self.queries)

        # Relevance distribution
        for qrel in self.qrels:
            stats['relevance_distribution'][qrel['relevance']] += 1

        return stats


def generate_report(stats: Dict, output_path: Path):
    """Generate report about test queries."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("TEST QUERIES AND QRELS STATISTICS\n")
        f.write("=" * 80 + "\n\n")

        f.write("QUERY STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total queries:              {stats['num_queries']}\n")
        f.write(f"Avg relevant docs/query:    {stats['avg_relevant_per_query']:.1f}\n\n")

        f.write("QUERIES BY TYPE\n")
        f.write("-" * 80 + "\n")
        for qtype, count in stats['queries_by_type'].items():
            f.write(f"{qtype:15s} {count:3d}\n")
        f.write("\n")

        f.write("QRELS STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total judgments:            {stats['num_qrels']}\n\n")

        f.write("RELEVANCE DISTRIBUTION\n")
        f.write("-" * 80 + "\n")
        for relevance, count in sorted(stats['relevance_distribution'].items()):
            label = {0: "Not relevant", 1: "Relevant", 2: "Highly relevant"}
            f.write(f"{label.get(relevance, 'Unknown'):20s} {count:5d}\n")

    logger.info(f"Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate test queries and QRELS for evaluation',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input JSONL file (preprocessed articles)'
    )

    parser.add_argument(
        '--output-queries',
        type=str,
        required=True,
        help='Output file for queries (.txt)'
    )

    parser.add_argument(
        '--output-qrels',
        type=str,
        required=True,
        help='Output file for qrels (.txt)'
    )

    parser.add_argument(
        '--num-queries',
        type=int,
        default=15,
        help='Number of queries to generate (default: 15)'
    )

    parser.add_argument(
        '--report',
        type=str,
        default=None,
        help='Output path for statistics report (optional)'
    )

    args = parser.parse_args()

    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    # Initialize generator
    generator = QueryGenerator()

    # Load articles
    generator.load_articles(input_path)

    # Generate queries
    generator.generate_queries_manual(num_queries=args.num_queries)

    # Create qrels
    generator.create_qrels_automatic(use_keywords=True)

    # Save outputs
    queries_path = Path(args.output_queries)
    qrels_path = Path(args.output_qrels)

    generator.save_queries(queries_path)
    generator.save_qrels(qrels_path)

    # Save detailed qrels for analysis
    qrels_detailed_path = qrels_path.with_suffix('.detailed.txt')
    generator.save_qrels_detailed(qrels_detailed_path)

    # Generate statistics
    stats = generator.generate_statistics()

    # Save report
    if args.report:
        report_path = Path(args.report)
    else:
        report_path = queries_path.parent.parent / 'stats' / 'test_queries_stats.txt'

    generate_report(stats, report_path)

    # Print summary
    print("\n" + "=" * 80)
    print("TEST QUERIES GENERATION COMPLETED")
    print("=" * 80)
    print(f"Input:           {input_path}")
    print(f"Queries:         {queries_path} ({stats['num_queries']} queries)")
    print(f"QRELS:           {qrels_path} ({stats['num_qrels']} judgments)")
    print(f"Report:          {report_path}")
    print(f"Avg relevant:    {stats['avg_relevant_per_query']:.1f} docs/query")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
