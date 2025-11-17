#!/usr/bin/env python
"""
Comprehensive Module Testing Script

Tests all IR and NLP modules with the cleaned CNA dataset.

Modules tested:
- IR M1: Boolean Retrieval
- IR M2: Inverted Index & Positional Index
- IR M3: TF-IDF & VSM
- IR M4: Evaluation Metrics
- IR M5: Rocchio Query Expansion
- IR M6: Document & Term Clustering
- IR M7: Summarization (Static & Dynamic)

- NLP Phase 1: Tokenization
- NLP Phase 2: NER
- NLP Phase 3: Keyword Extraction
- NLP Phase 4: Topic Modeling
- NLP Phase 5: Syntactic Parsing

Usage:
    python scripts/test/test_all_modules.py \
        --dataset data/processed/cna_mvp_cleaned.jsonl \
        --output data/test_results/module_test_results.txt

Author: CNIRS Development Team
License: Educational Use Only
"""

import json
import argparse
import sys
from pathlib import Path
from datetime import datetime
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_dataset(dataset_path: str, limit: int = None):
    """Load dataset from JSONL file."""
    documents = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            doc = json.loads(line.strip())
            documents.append(doc)
    return documents


def test_ir_modules(documents, report_lines):
    """Test all IR modules."""
    report_lines.append("\n" + "=" * 70)
    report_lines.append("IR MODULES TESTING")
    report_lines.append("=" * 70)

    # M1: Boolean Retrieval
    try:
        from src.ir.retrieval.boolean import BooleanRetrieval
        from src.ir.index.inverted_index import InvertedIndex

        report_lines.append("\n## M1: Boolean Retrieval")
        corpus = [doc['content'] for doc in documents[:50]]  # Use 50 docs
        index = InvertedIndex()
        index.build(corpus)

        retrieval = BooleanRetrieval(index)
        results = retrieval.search("中國 AND 美國")
        report_lines.append(f"   ✅ Boolean search '中國 AND 美國': {len(results)} results")

        results2 = retrieval.search("台灣 OR 經濟")
        report_lines.append(f"   ✅ Boolean search '台灣 OR 經濟': {len(results2)} results")

    except Exception as e:
        report_lines.append(f"   ❌ Error: {str(e)}")

    # M2: Inverted Index & Positional Index
    try:
        from src.ir.index.inverted_index import InvertedIndex
        from src.ir.index.positional_index import PositionalIndex

        report_lines.append("\n## M2: Inverted Index & Positional Index")

        # Inverted index
        corpus = [doc['content'] for doc in documents[:50]]
        inv_index = InvertedIndex()
        inv_index.build(corpus)
        report_lines.append(f"   ✅ Inverted index built: {len(inv_index.index)} terms")

        # Positional index
        pos_index = PositionalIndex()
        pos_index.build(corpus)
        phrase_results = pos_index.phrase_search("中國")
        report_lines.append(f"   ✅ Positional index built: phrase search '中國' found in {len(phrase_results)} docs")

    except Exception as e:
        report_lines.append(f"   ❌ Error: {str(e)}")

    # M3: TF-IDF & VSM
    try:
        from src.ir.retrieval.vsm import VSMRetrieval
        from src.ir.index.term_weighting import TFIDFWeighting

        report_lines.append("\n## M3: TF-IDF & Vector Space Model")

        corpus = [doc['content'] for doc in documents[:50]]
        weighting = TFIDFWeighting()
        doc_vectors = weighting.fit_transform(corpus)

        vsm = VSMRetrieval(corpus)
        vsm.build_index()
        results = vsm.search("人工智慧", top_k=5)
        report_lines.append(f"   ✅ VSM search '人工智慧': top {len(results)} results")

        if results:
            top_score = results[0][1]
            report_lines.append(f"   ✅ Top result score: {top_score:.4f}")

    except Exception as e:
        report_lines.append(f"   ❌ Error: {str(e)}")

    # M4: Evaluation Metrics
    try:
        from src.ir.eval.metrics import precision, recall, f_measure, average_precision, ndcg

        report_lines.append("\n## M4: Evaluation Metrics")

        # Test metrics
        retrieved = [1, 2, 3, 4, 5]
        relevant = [1, 3, 5, 7, 9]

        p = precision(retrieved, relevant)
        r = recall(retrieved, relevant)
        f1 = f_measure(p, r)
        ap = average_precision(retrieved, relevant)

        report_lines.append(f"   ✅ Precision: {p:.3f}")
        report_lines.append(f"   ✅ Recall: {r:.3f}")
        report_lines.append(f"   ✅ F1-Score: {f1:.3f}")
        report_lines.append(f"   ✅ Average Precision: {ap:.3f}")

        # NDCG
        relevance_scores = [3, 2, 3, 0, 1]
        ndcg_score = ndcg(relevance_scores, k=5)
        report_lines.append(f"   ✅ NDCG@5: {ndcg_score:.3f}")

    except Exception as e:
        report_lines.append(f"   ❌ Error: {str(e)}")

    # M5: Rocchio Query Expansion
    try:
        from src.ir.ranking.rocchio import RocchioExpander
        from src.ir.retrieval.vsm import VSMRetrieval

        report_lines.append("\n## M5: Rocchio Query Expansion")

        # Need VSM for query expansion
        corpus = [doc['content'] for doc in documents[:50]]
        vsm = VSMRetrieval()
        vsm.build_index(corpus)

        # Search to get top-k for pseudo-relevance feedback
        original_query = "人工智慧"
        search_results = vsm.search(original_query, topk=10)

        if search_results.doc_ids:
            # Use top-3 as relevant docs for expansion
            relevant_doc_ids = search_results.doc_ids[:3]
            relevant_vectors = [vsm.get_document_vector(doc_id) for doc_id in relevant_doc_ids]

            # Build query vector
            query_tokens = vsm.inverted_index.tokenizer(original_query)
            query_vector = {term: 1.0 for term in query_tokens}

            # Expand query using Rocchio
            expander = RocchioExpander()
            expanded_result = expander.expand_query(query_vector, relevant_vectors)

            report_lines.append(f"   ✅ Original query: '{original_query}'")
            report_lines.append(f"   ✅ Expanded with {len(expanded_result.expanded_terms)} new terms")
            if expanded_result.expanded_terms:
                report_lines.append(f"   ✅ New terms: {', '.join(expanded_result.expanded_terms[:5])}")
        else:
            report_lines.append(f"   ⚠️  No search results for query expansion")

    except Exception as e:
        report_lines.append(f"   ❌ Error: {str(e)}")

    # M6: Clustering
    try:
        from src.ir.cluster.doc_cluster import HierarchicalDocCluster
        from src.ir.cluster.term_cluster import TermCluster

        report_lines.append("\n## M6: Document & Term Clustering")

        # Document clustering
        corpus = [doc['content'] for doc in documents[:20]]  # Use 20 docs
        doc_clusterer = HierarchicalDocCluster()
        doc_clusters = doc_clusterer.cluster(corpus, n_clusters=3)
        report_lines.append(f"   ✅ Document clustering: {len(doc_clusters)} clusters")

        # Term clustering
        term_clusterer = TermCluster()
        term_clusterer.build_from_corpus(corpus)
        similar_terms = term_clusterer.find_similar_terms("中國", k=5)
        report_lines.append(f"   ✅ Term clustering: found {len(similar_terms)} similar terms to '中國'")

    except Exception as e:
        report_lines.append(f"   ❌ Error: {str(e)}")

    # M7: Summarization
    try:
        from src.ir.summarize.static import lead_k_summary, key_sentence_summary
        from src.ir.summarize.dynamic import KWIC

        report_lines.append("\n## M7: Summarization (Static & Dynamic)")

        # Static summarization
        text = documents[0]['content']
        lead_summary = lead_k_summary(text, k=2)
        report_lines.append(f"   ✅ Lead-k summary: {len(lead_summary)} sentences")

        key_summary = key_sentence_summary(text, k=2)
        report_lines.append(f"   ✅ Key sentence summary: {len(key_summary)} sentences")

        # Dynamic KWIC
        kwic = KWIC()
        kwic_results = kwic.extract(text, "中國", window_size=10)
        report_lines.append(f"   ✅ KWIC extraction: {len(kwic_results)} contexts for '中國'")

    except Exception as e:
        report_lines.append(f"   ❌ Error: {str(e)}")


def test_nlp_modules(documents, report_lines):
    """Test all NLP modules."""
    report_lines.append("\n" + "=" * 70)
    report_lines.append("NLP MODULES TESTING")
    report_lines.append("=" * 70)

    # Phase 1: Tokenization
    try:
        from src.ir.text.chinese_tokenizer import ChineseTokenizer

        report_lines.append("\n## Phase 1: Tokenization")

        tokenizer = ChineseTokenizer()
        text = documents[0]['content']
        tokens = tokenizer.tokenize(text)
        report_lines.append(f"   ✅ Tokenization: {len(tokens)} tokens")
        report_lines.append(f"   ✅ Sample tokens: {' '.join(tokens[:10])}")

    except Exception as e:
        report_lines.append(f"   ❌ Error: {str(e)}")

    # Phase 2: NER
    try:
        from src.ir.text.ner_extractor import NERExtractor

        report_lines.append("\n## Phase 2: Named Entity Recognition")

        ner = NERExtractor()
        text = documents[0]['content']
        entities = ner.extract(text)
        report_lines.append(f"   ✅ NER extraction: {len(entities)} entities")

        # Count by type
        entity_types = {}
        for ent in entities:
            ent_type = ent.get('type', 'UNKNOWN')
            entity_types[ent_type] = entity_types.get(ent_type, 0) + 1

        for ent_type, count in entity_types.items():
            report_lines.append(f"      - {ent_type}: {count}")

    except Exception as e:
        report_lines.append(f"   ❌ Error: {str(e)}")

    # Phase 3: Keyword Extraction
    try:
        from src.ir.keyextract.textrank import TextRankExtractor

        report_lines.append("\n## Phase 3: Keyword Extraction")

        extractor = TextRankExtractor()
        text = documents[0]['content']
        keywords = extractor.extract(text, top_k=10)
        report_lines.append(f"   ✅ TextRank extraction: {len(keywords)} keywords")
        report_lines.append(f"   ✅ Top keywords: {', '.join([kw['word'] for kw in keywords[:5]])}")

    except Exception as e:
        report_lines.append(f"   ❌ Error: {str(e)}")

    # Phase 4: Topic Modeling
    try:
        from src.ir.topic.lda_model import LDATopicModel

        report_lines.append("\n## Phase 4: Topic Modeling")

        corpus = [doc['content'] for doc in documents[:50]]
        lda = LDATopicModel(n_topics=5)
        lda.fit(corpus)
        topics = lda.get_topics(top_words=5)
        report_lines.append(f"   ✅ LDA topic modeling: {len(topics)} topics")

        for i, topic in enumerate(topics[:3]):
            words = ', '.join([w for w, _ in topic])
            report_lines.append(f"      Topic {i+1}: {words}")

    except Exception as e:
        report_lines.append(f"   ❌ Error: {str(e)}")

    # Phase 5: Syntactic Parsing
    try:
        from src.ir.syntax.parser import DependencyParser

        report_lines.append("\n## Phase 5: Syntactic Parsing")

        parser = DependencyParser()
        text = "中國經濟快速發展"
        parse_tree = parser.parse(text)
        report_lines.append(f"   ✅ Dependency parsing completed")
        report_lines.append(f"   ✅ Sample text: '{text}'")

    except Exception as e:
        report_lines.append(f"   ❌ Error: {str(e)}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Test all IR and NLP modules')
    parser.add_argument('--dataset', type=str, required=True, help='Path to cleaned JSONL dataset')
    parser.add_argument('--output', type=str, help='Path to save test results')
    parser.add_argument('--limit', type=int, default=100, help='Limit number of documents to load')

    args = parser.parse_args()

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    documents = load_dataset(args.dataset, limit=args.limit)
    logger.info(f"Loaded {len(documents)} documents")

    # Report lines
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("COMPREHENSIVE MODULE TESTING REPORT")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Dataset: {args.dataset}")
    report_lines.append(f"Documents: {len(documents)}")
    report_lines.append("=" * 70)

    # Test IR modules
    logger.info("Testing IR modules...")
    test_ir_modules(documents, report_lines)

    # Test NLP modules
    logger.info("Testing NLP modules...")
    test_nlp_modules(documents, report_lines)

    # Summary
    report_lines.append("\n" + "=" * 70)
    report_lines.append("TESTING COMPLETE")
    report_lines.append("=" * 70)

    # Print report
    for line in report_lines:
        print(line)

    # Save report
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        logger.info(f"\n📁 Report saved to: {args.output}")


if __name__ == '__main__':
    main()
