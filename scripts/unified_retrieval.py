#!/usr/bin/env python3
"""
Unified Retrieval API for CNIRS Project

This module provides a unified interface for all retrieval models:
- Boolean Retrieval (exact match)
- TF-IDF (Vector Space Model)
- BM25 (Probabilistic ranking)
- BERT (Semantic search)

Usage:
    from unified_retrieval import UnifiedRetrieval

    retriever = UnifiedRetrieval()
    retriever.load_indexes('data/indexes')

    results = retriever.search("颱風災害", model='tfidf', top_k=10)

Author: Information Retrieval System
License: Educational Use
"""

import pickle
import json
import logging
import math
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import IR modules
from src.ir.text.chinese_tokenizer import ChineseTokenizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """
    Search result for a single document.

    Attributes:
        doc_id: Document ID (article_id)
        score: Relevance score
        rank: Rank position (1-based)
        title: Document title
        snippet: Text snippet/summary
        metadata: Additional metadata
    """
    doc_id: str
    score: float
    rank: int
    title: str = ""
    snippet: str = ""
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class UnifiedRetrieval:
    """
    Unified retrieval interface integrating multiple search models.

    Supports:
    - Boolean retrieval (AND/OR operations)
    - TF-IDF ranking (Vector Space Model)
    - BM25 ranking (Probabilistic model)
    - BERT semantic search

    Attributes:
        tokenizer: Chinese tokenizer
        inverted_index: Inverted index
        tfidf_data: TF-IDF vectors and IDF values
        bm25_data: BM25 index data
        bert_embeddings: BERT document embeddings
        doc_id_map: Mapping between numeric and article IDs
    """

    def __init__(self, tokenizer_engine: str = 'jieba'):
        """
        Initialize unified retrieval system.

        Args:
            tokenizer_engine: 'jieba' or 'ckip'
        """
        logger.info("Initializing UnifiedRetrieval system")

        # Initialize tokenizer
        self.tokenizer = ChineseTokenizer(engine=tokenizer_engine)

        # Initialize data structures
        self.inverted_index = None
        self.positional_index = None
        self.tfidf_data = None
        self.bm25_data = None
        self.bert_embeddings = None
        self.bert_metadata = None
        self.doc_id_map = None
        self.reverse_doc_map = None
        self.doc_metadata = {}

        # BERT model (lazy loading)
        self.bert_model = None

        logger.info("UnifiedRetrieval initialized")

    def load_indexes(self, index_dir: Path):
        """
        Load all indexes from directory.

        Args:
            index_dir: Path to indexes directory
        """
        index_dir = Path(index_dir)
        logger.info(f"Loading indexes from {index_dir}")

        # Load inverted index
        inverted_path = index_dir / 'inverted_index.pkl'
        if inverted_path.exists():
            with open(inverted_path, 'rb') as f:
                data = pickle.load(f)
                self.inverted_index = data['index']
                self.doc_metadata = data.get('doc_metadata', {})
            logger.info(f"Loaded inverted index: {len(self.inverted_index)} terms")

        # Load positional index
        positional_path = index_dir / 'positional_index.pkl'
        if positional_path.exists():
            with open(positional_path, 'rb') as f:
                data = pickle.load(f)
                self.positional_index = data['index']
            logger.info(f"Loaded positional index: {len(self.positional_index)} terms")

        # Load TF-IDF data
        tfidf_path = index_dir / 'tfidf_vectors.pkl'
        if tfidf_path.exists():
            with open(tfidf_path, 'rb') as f:
                self.tfidf_data = pickle.load(f)
            logger.info(f"Loaded TF-IDF vectors: {len(self.tfidf_data['document_vectors'])} docs")

        # Load BM25 data
        bm25_path = index_dir / 'bm25_index.pkl'
        if bm25_path.exists():
            with open(bm25_path, 'rb') as f:
                self.bm25_data = pickle.load(f)
            logger.info(f"Loaded BM25 index: {len(self.bm25_data['idf'])} terms")

        # Load BERT embeddings
        bert_path = index_dir / 'bert_embeddings.npy'
        if bert_path.exists():
            self.bert_embeddings = np.load(bert_path)

            # Load BERT metadata
            bert_meta_path = index_dir / 'bert_embeddings.json'
            if bert_meta_path.exists():
                with open(bert_meta_path, 'r') as f:
                    self.bert_metadata = json.load(f)

            logger.info(f"Loaded BERT embeddings: {self.bert_embeddings.shape}")

        # Load document ID mappings
        doc_map_path = index_dir / 'doc_id_map.json'
        if doc_map_path.exists():
            with open(doc_map_path, 'r') as f:
                data = json.load(f)
                self.doc_id_map = data['article_to_doc']
                # Convert string keys back to int
                self.reverse_doc_map = {int(k): v for k, v in data['doc_to_article'].items()}
            logger.info(f"Loaded document ID mappings: {len(self.doc_id_map)} docs")

    def search_boolean(self, query: str, operator: str = 'AND') -> List[SearchResult]:
        """
        Boolean retrieval (exact match).

        Args:
            query: Query string
            operator: 'AND' or 'OR'

        Returns:
            List of search results (unranked)
        """
        if not self.inverted_index:
            logger.error("Inverted index not loaded")
            return []

        # Tokenize query
        query_terms = self.tokenizer.tokenize(query)

        if not query_terms:
            return []

        # Get document sets for each term
        doc_sets = []
        for term in query_terms:
            if term in self.inverted_index:
                postings = self.inverted_index[term]
                doc_ids = {posting[0] for posting in postings}
                doc_sets.append(doc_ids)

        if not doc_sets:
            return []

        # Combine with AND/OR
        if operator == 'AND':
            result_docs = set.intersection(*doc_sets)
        else:  # OR
            result_docs = set.union(*doc_sets)

        # Convert to SearchResult
        results = []
        for rank, doc_id in enumerate(sorted(result_docs), 1):
            article_id = self.reverse_doc_map.get(doc_id, str(doc_id))
            metadata = self.doc_metadata.get(doc_id, {})

            results.append(SearchResult(
                doc_id=article_id,
                score=1.0,  # Boolean has no score
                rank=rank,
                title=metadata.get('title', ''),
                metadata=metadata
            ))

        return results

    def search_tfidf(self, query: str, top_k: int = 20) -> List[SearchResult]:
        """
        TF-IDF ranking (Vector Space Model).

        Args:
            query: Query string
            top_k: Number of top results to return

        Returns:
            Ranked list of search results
        """
        if not self.tfidf_data:
            logger.error("TF-IDF data not loaded")
            return []

        # Tokenize query
        query_terms = self.tokenizer.tokenize(query)

        if not query_terms:
            return []

        # Build query vector (TF-IDF)
        query_tf = {}
        for term in query_terms:
            query_tf[term] = query_tf.get(term, 0) + 1

        # Apply TF-IDF to query
        query_vector = {}
        for term, tf in query_tf.items():
            if term in self.tfidf_data['idf']:
                # Log TF
                tf_weight = 1.0 + math.log10(tf)
                idf_weight = self.tfidf_data['idf'][term]
                query_vector[term] = tf_weight * idf_weight

        # Normalize query vector
        query_norm = math.sqrt(sum(w ** 2 for w in query_vector.values()))
        if query_norm > 0:
            query_vector = {t: w / query_norm for t, w in query_vector.items()}

        # Compute cosine similarity with all documents
        scores = {}
        for doc_id, doc_vector in self.tfidf_data['document_vectors'].items():
            # Cosine similarity (dot product of normalized vectors)
            score = sum(query_vector.get(term, 0) * weight
                       for term, weight in doc_vector.items())
            if score > 0:
                scores[doc_id] = score

        # Sort by score
        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Convert to SearchResult
        results = []
        for rank, (doc_id, score) in enumerate(ranked_docs, 1):
            article_id = self.reverse_doc_map.get(doc_id, str(doc_id))
            metadata = self.doc_metadata.get(doc_id, {})

            results.append(SearchResult(
                doc_id=article_id,
                score=score,
                rank=rank,
                title=metadata.get('title', ''),
                metadata=metadata
            ))

        return results

    def search_bm25(self, query: str, top_k: int = 20) -> List[SearchResult]:
        """
        BM25 ranking (Probabilistic model).

        Args:
            query: Query string
            top_k: Number of top results to return

        Returns:
            Ranked list of search results
        """
        if not self.bm25_data or not self.inverted_index:
            logger.error("BM25 data not loaded")
            return []

        # Tokenize query
        query_terms = self.tokenizer.tokenize(query)

        if not query_terms:
            return []

        # BM25 parameters
        k1 = self.bm25_data.get('k1', 1.5)
        b = self.bm25_data.get('b', 0.75)
        avgdl = self.bm25_data.get('avg_doc_length', 1.0)

        # Compute BM25 scores
        scores = {}

        for term in set(query_terms):
            if term not in self.inverted_index:
                continue

            idf = self.bm25_data['idf'].get(term, 0)
            postings = self.inverted_index[term]

            for doc_id, tf in postings:
                # Get document length
                doc_length = self.doc_metadata.get(doc_id, {}).get('length', avgdl)

                # BM25 formula
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * (doc_length / avgdl))
                bm25_score = idf * (numerator / denominator)

                scores[doc_id] = scores.get(doc_id, 0) + bm25_score

        # Sort by score
        ranked_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Convert to SearchResult
        results = []
        for rank, (doc_id, score) in enumerate(ranked_docs, 1):
            article_id = self.reverse_doc_map.get(doc_id, str(doc_id))
            metadata = self.doc_metadata.get(doc_id, {})

            results.append(SearchResult(
                doc_id=article_id,
                score=score,
                rank=rank,
                title=metadata.get('title', ''),
                metadata=metadata
            ))

        return results

    def search_bert(self, query: str, top_k: int = 20) -> List[SearchResult]:
        """
        BERT semantic search.

        Args:
            query: Query string
            top_k: Number of top results to return

        Returns:
            Ranked list of search results
        """
        if self.bert_embeddings is None:
            logger.error("BERT embeddings not loaded")
            return []

        # Lazy load BERT model
        if self.bert_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                model_name = self.bert_metadata.get('model_name',
                                                   'paraphrase-multilingual-MiniLM-L12-v2')
                self.bert_model = SentenceTransformer(model_name)
                logger.info(f"Loaded BERT model: {model_name}")
            except ImportError:
                logger.error("sentence-transformers not available")
                return []

        # Encode query
        query_embedding = self.bert_model.encode([query],
                                                 convert_to_numpy=True,
                                                 normalize_embeddings=True)[0]

        # Compute cosine similarity (dot product for normalized vectors)
        similarities = np.dot(self.bert_embeddings, query_embedding)

        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Convert to SearchResult
        results = []
        for rank, doc_id in enumerate(top_indices, 1):
            article_id = self.reverse_doc_map.get(doc_id, str(doc_id))
            metadata = self.doc_metadata.get(doc_id, {})
            score = float(similarities[doc_id])

            results.append(SearchResult(
                doc_id=article_id,
                score=score,
                rank=rank,
                title=metadata.get('title', ''),
                metadata=metadata
            ))

        return results

    def search(self, query: str, model: str = 'tfidf', top_k: int = 20, **kwargs) -> List[SearchResult]:
        """
        Unified search interface.

        Args:
            query: Query string
            model: Model to use ('boolean', 'tfidf', 'bm25', 'bert')
            top_k: Number of results (ignored for boolean)
            **kwargs: Additional model-specific parameters

        Returns:
            List of search results
        """
        model = model.lower()

        if model == 'boolean':
            operator = kwargs.get('operator', 'AND')
            return self.search_boolean(query, operator=operator)
        elif model == 'tfidf':
            return self.search_tfidf(query, top_k=top_k)
        elif model == 'bm25':
            return self.search_bm25(query, top_k=top_k)
        elif model == 'bert':
            return self.search_bert(query, top_k=top_k)
        else:
            logger.error(f"Unknown model: {model}")
            return []


def main():
    """Demo usage of UnifiedRetrieval."""
    import argparse

    parser = argparse.ArgumentParser(description='Unified Retrieval Demo')
    parser.add_argument('--index-dir', type=str, default='data/indexes',
                       help='Path to indexes directory')
    parser.add_argument('--query', type=str, required=True,
                       help='Query string')
    parser.add_argument('--model', type=str, default='tfidf',
                       choices=['boolean', 'tfidf', 'bm25', 'bert'],
                       help='Retrieval model')
    parser.add_argument('--top-k', type=int, default=10,
                       help='Number of results')

    args = parser.parse_args()

    # Initialize retrieval system
    retriever = UnifiedRetrieval()
    retriever.load_indexes(Path(args.index_dir))

    # Search
    logger.info(f"Searching for '{args.query}' using {args.model}")
    results = retriever.search(args.query, model=args.model, top_k=args.top_k)

    # Display results
    print(f"\n{'='*80}")
    print(f"Query: {args.query}")
    print(f"Model: {args.model.upper()}")
    print(f"Results: {len(results)}")
    print(f"{'='*80}\n")

    for result in results:
        print(f"Rank {result.rank}: {result.title}")
        print(f"  Doc ID: {result.doc_id}")
        print(f"  Score:  {result.score:.4f}")
        print()


if __name__ == '__main__':
    main()
