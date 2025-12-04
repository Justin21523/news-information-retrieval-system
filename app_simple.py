#!/usr/bin/env python3
"""
CNIRS - Chinese News Intelligent Retrieval System (Simple Version)
Flask Web Application

This simplified version uses the unified retrieval system from Phase 4.
It provides a clean web interface for search and model comparison.

Author: Information Retrieval System
License: Educational Use
"""

import sys
import logging
import time
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from scripts.unified_retrieval import UnifiedRetrieval, SearchResult
from src.ir.ranking.rocchio import RocchioExpander
from src.ir.eval.metrics import precision, recall, f_measure, average_precision, ndcg, reciprocal_rank, _metrics_instance
from src.ir.summarize.static import lead_k_summary, key_sentence_summary
from src.ir.text.chinese_tokenizer import ChineseTokenizer
from src.ir.cluster.term_cluster import TermClusterer
from src.ir.cluster.doc_cluster import DocumentClusterer
from src.ir.index.pat_tree import PatriciaTree, build_pat_tree_from_documents
from src.ir.query import QueryParser, QueryExecutor, parse_query
from src.ir.index.field_indexer import FieldIndexer
from src.ir.facet import FacetEngine, FacetFilter, FilterCondition, FilterOperator, RangeFilter
from collections import Counter
import json
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['INDEX_DIR'] = project_root / 'data' / 'indexes_10k'  # Full index (9808 docs)
app.config['MAX_RESULTS'] = 100
app.config['DEFAULT_TOP_K'] = 20

# Global retrieval system (lazy loading)
retriever = None

# Document content cache
_doc_content_cache = None

# PAT-tree cache (lazy loading)
_pat_tree_cache = None

# Field indexer cache (lazy loading) - for advanced metadata search
_field_indexer_cache = None
_documents_cache = None

# Facet engine cache (lazy loading)
_facet_engine_cache = None


def get_pat_tree():
    """
    Get or build PAT-tree from document collection.

    Returns:
        PatriciaTree: Populated PAT-tree instance

    Note: This builds the tree on first access and caches it.
          Tree contains all terms from the inverted index.
    """
    global _pat_tree_cache

    if _pat_tree_cache is None:
        logger.info("Building PAT-tree from document collection...")
        start_time = time.time()

        ret = get_retriever()
        tokenizer = ChineseTokenizer()

        # Load documents from preprocessed file
        preprocessed_file = project_root / 'data' / 'preprocessed' / 'merged_14days_preprocessed.jsonl'
        documents = []

        try:
            with open(preprocessed_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        doc = json.loads(line)
                        doc_id = doc.get('article_id', '')
                        content = doc.get('content', '')
                        if doc_id and content:
                            documents.append({
                                'doc_id': doc_id,
                                'content': content
                            })

            logger.info(f"Loaded {len(documents)} documents for PAT-tree construction")

            # Build PAT-tree
            _pat_tree_cache = build_pat_tree_from_documents(documents, tokenizer)

            build_time = time.time() - start_time
            stats = _pat_tree_cache.get_statistics()

            logger.info(f"PAT-tree built in {build_time:.2f}s")
            logger.info(f"  Total terms: {stats['total_terms']}")
            logger.info(f"  Unique terms: {stats['unique_terms']}")
            logger.info(f"  Tree nodes: {stats['total_nodes']}")
            logger.info(f"  Compression ratio: {stats['compression_ratio']:.2f}x")

        except Exception as e:
            logger.error(f"Failed to build PAT-tree: {e}", exc_info=True)
            # Create empty tree as fallback
            _pat_tree_cache = PatriciaTree()

    return _pat_tree_cache


def load_document_content(doc_id):
    """
    Load document content from preprocessed JSONL file.

    Args:
        doc_id: Article ID

    Returns:
        str: Document content or empty string if not found
    """
    global _doc_content_cache

    # Build cache on first access
    if _doc_content_cache is None:
        _doc_content_cache = {}
        preprocessed_file = project_root / 'data' / 'preprocessed' / 'merged_14days_preprocessed.jsonl'

        try:
            with open(preprocessed_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        doc = json.loads(line)
                        article_id = doc.get('article_id', '')
                        content = doc.get('content', '')
                        if article_id:
                            _doc_content_cache[article_id] = content
            logger.info(f"Loaded content for {len(_doc_content_cache)} documents")
        except Exception as e:
            logger.error(f"Failed to load document content: {e}")
            return ""

    return _doc_content_cache.get(doc_id, "")


def get_retriever():
    """
    Get or initialize the retrieval system.

    Returns:
        UnifiedRetrieval instance
    """
    global retriever

    if retriever is None:
        logger.info("Initializing retrieval system...")
        retriever = UnifiedRetrieval()
        retriever.load_indexes(app.config['INDEX_DIR'])
        logger.info("Retrieval system ready")

    return retriever


def get_field_indexer():
    """
    Get or load the field indexer for advanced metadata search.

    Returns:
        tuple: (FieldIndexer, List[Dict]) - indexer and documents list
    """
    global _field_indexer_cache, _documents_cache

    if _field_indexer_cache is None:
        logger.info("Loading field indexer for advanced search...")
        start_time = time.time()

        # Load field index
        field_index_path = project_root / 'data' / 'indexes' / 'field_index.pkl'

        try:
            with open(field_index_path, 'rb') as f:
                _field_indexer_cache = pickle.load(f)

            # Load documents for metadata
            preprocessed_file = project_root / 'data' / 'preprocessed' / 'merged_14days_preprocessed.jsonl'
            _documents_cache = []

            with open(preprocessed_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        doc = json.loads(line)
                        _documents_cache.append(doc)

            load_time = time.time() - start_time
            logger.info(f"Field indexer loaded in {load_time:.2f}s")
            logger.info(f"  Documents: {len(_documents_cache)}")
            logger.info(f"  Fields indexed: {_field_indexer_cache.doc_count}")

        except FileNotFoundError:
            logger.error(f"Field index not found at {field_index_path}")
            logger.error("Please run: python scripts/build_field_index.py")
            _field_indexer_cache = None
            _documents_cache = []
        except Exception as e:
            logger.error(f"Failed to load field indexer: {e}", exc_info=True)
            _field_indexer_cache = None
            _documents_cache = []

    return _field_indexer_cache, _documents_cache


def get_facet_engine():
    """
    Get or initialize facet engine.

    Returns:
        FacetEngine: Initialized facet engine with configured facets
    """
    global _facet_engine_cache

    if _facet_engine_cache is None:
        logger.info("Initializing facet engine...")
        engine = FacetEngine()

        # Configure news-specific facets
        engine.configure_facet("source", "新聞來源", "term", max_values=20)
        engine.configure_facet("category", "分類", "term")
        engine.configure_facet("category_name", "分類名稱", "term")
        engine.configure_facet("published_date", "發布月份", "date_range", date_format="%Y-%m")
        engine.configure_facet("author", "作者", "term", max_values=15)

        _facet_engine_cache = engine
        logger.info("Facet engine initialized")

    return _facet_engine_cache


@app.route('/')
def index():
    """Main search page."""
    return render_template('search.html')


@app.route('/compare')
def compare():
    """Model comparison page."""
    return render_template('compare.html')


@app.route('/about')
def about():
    """About page."""
    return render_template('about.html')


@app.route('/expand')
def expand():
    """Query expansion page."""
    return render_template('expand.html')


@app.route('/evaluation')
def evaluation():
    """Evaluation and analysis page."""
    return render_template('evaluation.html')


@app.route('/pat_tree')
def pat_tree():
    """PAT-tree visualization page."""
    return render_template('pat_tree.html')


@app.route('/api/search', methods=['POST'])
def api_search():
    """
    Search API endpoint with advanced filtering.

    Request JSON:
        {
            "query": "search query",
            "model": "tfidf|bm25|bert|boolean",
            "top_k": 20,
            "operator": "AND|OR",  (for boolean only)
            "filters": {  (optional)
                "categories": ["aipl", "afe"],  // filter by categories
                "date_from": "2025-11-10",      // filter by date range
                "date_to": "2025-11-13"
            }
        }

    Response JSON:
        {
            "success": true,
            "query": "...",
            "model": "...",
            "total_results": 10,
            "filtered_results": 8,  // after filtering
            "response_time": 0.023,
            "results": [...]
        }
    """
    try:
        data = request.get_json()

        if not data or 'query' not in data:
            return jsonify({'success': False, 'error': 'Missing query parameter'}), 400

        query = data.get('query', '').strip()
        model = data.get('model', 'tfidf').lower()
        top_k = min(data.get('top_k', app.config['DEFAULT_TOP_K']), app.config['MAX_RESULTS'])
        operator = data.get('operator', 'AND').upper()
        filters = data.get('filters', {})

        if not query:
            return jsonify({'success': False, 'error': 'Empty query'}), 400

        if model not in ['boolean', 'tfidf', 'bm25', 'bert']:
            return jsonify({'success': False, 'error': f'Invalid model: {model}'}), 400

        # Perform search
        ret = get_retriever()
        start_time = time.time()
        results = ret.search(query, model=model, top_k=top_k, operator=operator)

        # Apply filters
        total_before_filter = len(results)
        if filters:
            results = apply_filters(results, filters)

        response_time = time.time() - start_time

        # Convert results to JSON with standardized field names
        results_json = [
            {
                'doc_id': r.doc_id,
                'id': r.doc_id,  # Alias for faceted search compatibility
                'title': r.title,
                'score': r.score,
                'rank': r.rank,
                'snippet': r.snippet,
                # Standardized date fields at top level
                'published_date': r.metadata.get('published_date', ''),
                'pub_date': r.metadata.get('published_date', ''),
                'date': r.metadata.get('published_date', ''),
                # Other common fields at top level
                'category': r.metadata.get('category', ''),
                'category_name': r.metadata.get('category_name', ''),
                'source': r.metadata.get('source', ''),
                'author': r.metadata.get('author', ''),
                'metadata': r.metadata
            }
            for r in results
        ]

        return jsonify({
            'success': True,
            'query': query,
            'model': model,
            'total_results': total_before_filter,
            'filtered_results': len(results),
            'response_time': response_time,
            'filters_applied': filters,
            'results': results_json
        })

    except Exception as e:
        logger.error(f"Search error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


def apply_filters(results, filters):
    """
    Apply filters to search results.

    Args:
        results: List of SearchResult objects
        filters: Dictionary with filter criteria
            - categories: List of categories to include
            - date_from: Minimum date (YYYY-MM-DD)
            - date_to: Maximum date (YYYY-MM-DD)

    Returns:
        Filtered list of SearchResult objects
    """
    filtered = results

    # Filter by categories
    if 'categories' in filters and filters['categories']:
        categories_set = set(filters['categories'])
        filtered = [r for r in filtered
                   if r.metadata.get('category') in categories_set]

    # Filter by date range
    date_from = filters.get('date_from')
    date_to = filters.get('date_to')

    if date_from:
        filtered = [r for r in filtered
                   if r.metadata.get('published_date', '') >= date_from]

    if date_to:
        filtered = [r for r in filtered
                   if r.metadata.get('published_date', '') <= date_to]

    return filtered


@app.route('/api/compare', methods=['POST'])
def api_compare():
    """
    Compare multiple models API endpoint.

    Request JSON:
        {
            "query": "search query",
            "models": ["tfidf", "bm25", "bert"],
            "top_k": 10
        }

    Response JSON:
        {
            "success": true,
            "query": "...",
            "models": {...}
        }
    """
    try:
        data = request.get_json()

        if not data or 'query' not in data:
            return jsonify({'success': False, 'error': 'Missing query parameter'}), 400

        query = data.get('query', '').strip()
        models = data.get('models', ['tfidf', 'bm25'])
        top_k = min(data.get('top_k', 10), app.config['MAX_RESULTS'])

        if not query:
            return jsonify({'success': False, 'error': 'Empty query'}), 400

        ret = get_retriever()
        comparison_results = {}

        for model in models:
            if model not in ['tfidf', 'bm25', 'bert', 'boolean']:
                continue

            start_time = time.time()
            results = ret.search(query, model=model, top_k=top_k)
            response_time = time.time() - start_time

            comparison_results[model] = {
                'total_results': len(results),
                'response_time': response_time,
                'results': [
                    {
                        'doc_id': r.doc_id,
                        'id': r.doc_id,
                        'title': r.title,
                        'score': r.score,
                        'rank': r.rank,
                        'snippet': r.snippet,
                        'published_date': r.metadata.get('published_date', ''),
                        'pub_date': r.metadata.get('published_date', ''),
                        'date': r.metadata.get('published_date', ''),
                        'category': r.metadata.get('category', ''),
                        'category_name': r.metadata.get('category_name', ''),
                        'source': r.metadata.get('source', ''),
                        'author': r.metadata.get('author', ''),
                        'metadata': r.metadata
                    }
                    for r in results
                ]
            }

        return jsonify({
            'success': True,
            'query': query,
            'models': comparison_results
        })

    except Exception as e:
        logger.error(f"Comparison error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/document/<doc_id>', methods=['GET'])
def api_document(doc_id):
    """
    Get full document by ID with similar documents.

    Query Parameters:
        include_similar: boolean (default: true) - include similar documents
        top_k: int (default: 5) - number of similar documents to return
    """
    try:
        ret = get_retriever()

        if doc_id not in ret.doc_id_map:
            return jsonify({'success': False, 'error': 'Document not found'}), 404

        numeric_id = ret.doc_id_map[doc_id]
        metadata = ret.doc_metadata.get(numeric_id, {})

        # Get query parameters
        include_similar = request.args.get('include_similar', 'true').lower() == 'true'
        top_k = min(int(request.args.get('top_k', 5)), 10)

        # Get content - try metadata first, then load from file if empty
        content = metadata.get('content', '')
        if not content:
            content = load_document_content(doc_id)

        # Ensure metadata has content for modal display
        enriched_metadata = {
            **metadata,
            'content': content,
            'source': metadata.get('source', ''),
            'author': metadata.get('author', ''),
            'category_name': metadata.get('category_name', ''),
        }

        response = {
            'success': True,
            'doc_id': doc_id,
            'title': metadata.get('title', ''),
            'content': content,
            'url': metadata.get('url', ''),
            'date': metadata.get('published_date', ''),
            'category': metadata.get('category', ''),
            'metadata': enriched_metadata
        }

        # Find similar documents using TF-IDF cosine similarity
        if include_similar and ret.tfidf_data:
            similar_docs = find_similar_documents(ret, numeric_id, top_k)
            response['similar_documents'] = similar_docs

        return jsonify(response)

    except Exception as e:
        logger.error(f"Document retrieval error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


def find_similar_documents(retriever, doc_id: int, top_k: int = 5):
    """
    Find similar documents using cosine similarity.

    Args:
        retriever: UnifiedRetrieval instance
        doc_id: Numeric document ID
        top_k: Number of similar documents to return

    Returns:
        List of similar documents with scores
    """
    import math

    if doc_id not in retriever.tfidf_data['document_vectors']:
        return []

    # Get document vector
    doc_vector = retriever.tfidf_data['document_vectors'][doc_id]

    # Compute cosine similarity with all other documents
    similarities = {}
    for other_id, other_vector in retriever.tfidf_data['document_vectors'].items():
        if other_id == doc_id:
            continue

        # Cosine similarity (vectors already normalized)
        similarity = sum(doc_vector.get(term, 0) * weight
                        for term, weight in other_vector.items())

        if similarity > 0:
            similarities[other_id] = similarity

    # Sort by similarity and get top-k
    top_similar = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # Format results
    results = []
    for rank, (similar_id, score) in enumerate(top_similar, 1):
        article_id = retriever.reverse_doc_map.get(similar_id, str(similar_id))
        metadata = retriever.doc_metadata.get(similar_id, {})

        results.append({
            'doc_id': article_id,
            'title': metadata.get('title', ''),
            'similarity': round(score, 4),  # Frontend expects 'similarity' not 'score'
            'score': round(score, 4),  # Keep for backward compatibility
            'rank': rank,
            'published_date': metadata.get('published_date', ''),
            'date': metadata.get('published_date', ''),  # Alias
            'category': metadata.get('category', ''),
            'url': metadata.get('url', '')
        })

    return results


@app.route('/api/advanced_search', methods=['POST'])
def api_advanced_search():
    """
    Advanced metadata-based search API endpoint.

    Supports library-style queries with field-specific searches, boolean operators,
    and date ranges.

    Request JSON:
        {
            "query": "title:台灣 AND category:政治",  // Query string format
            // OR structured format:
            "conditions": [
                {"field": "title", "operator": "contains", "value": "台灣"},
                {"field": "category", "operator": "equals", "value": "aipl"}
            ],
            "logic": "AND",  // For structured queries: AND or OR
            "top_k": 20,
            "sort_by": "date",  // date, relevance (doc_id for now)
            "sort_order": "desc"  // asc or desc
        }

    Response JSON:
        {
            "success": true,
            "query": "...",
            "total_results": 15,
            "response_time": 0.023,
            "results": [
                {
                    "doc_id": 0,
                    "article_id": "202511...",
                    "title": "...",
                    "category": "...",
                    "date": "2025-11-10",
                    "matched_fields": ["title", "category"],
                    "snippet": "..."
                },
                ...
            ]
        }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400

        # Get field indexer and documents
        field_indexer, documents = get_field_indexer()

        if field_indexer is None:
            return jsonify({
                'success': False,
                'error': 'Field index not available. Please run: python scripts/build_field_index.py'
            }), 503

        start_time = time.time()

        # Create query executor
        executor = QueryExecutor(field_indexer, documents)

        # Determine query mode: query string or structured
        if 'query' in data and data['query']:
            # Query string mode
            query_str = data['query'].strip()

            try:
                query_node = parse_query(query_str)
            except SyntaxError as e:
                return jsonify({
                    'success': False,
                    'error': f'Query syntax error: {str(e)}'
                }), 400

            top_k = data.get('top_k', app.config['DEFAULT_TOP_K'])
            search_results = executor.execute(query_node, top_k=top_k)

            query_repr = query_str

        elif 'conditions' in data:
            # Structured query mode
            conditions = data['conditions']
            logic = data.get('logic', 'AND')
            top_k = data.get('top_k', app.config['DEFAULT_TOP_K'])

            if not conditions:
                return jsonify({
                    'success': False,
                    'error': 'No conditions provided'
                }), 400

            search_results = executor.execute_structured_query(
                conditions, logic=logic, top_k=top_k
            )

            query_repr = f"{len(conditions)} conditions with {logic}"

        else:
            return jsonify({
                'success': False,
                'error': 'Must provide either "query" (string) or "conditions" (list)'
            }), 400

        # Apply sorting
        sort_by = data.get('sort_by', 'relevance')
        sort_order = data.get('sort_order', 'desc')

        if sort_by == 'date' and documents:
            # Sort by publication date
            def get_date_key(result):
                doc = documents[result.doc_id] if result.doc_id < len(documents) else {}
                return doc.get('published_date', '')

            search_results.sort(
                key=get_date_key,
                reverse=(sort_order == 'desc')
            )

        response_time = time.time() - start_time

        # Format results with metadata
        results_json = []
        for result in search_results:
            doc_id = result.doc_id
            doc = documents[doc_id] if doc_id < len(documents) else {}

            results_json.append({
                'doc_id': doc_id,
                'id': doc_id,  # Alias
                'article_id': doc.get('article_id', str(doc_id)),
                'title': doc.get('title', ''),
                'score': getattr(result, 'score', 0),
                'rank': len(results_json) + 1,
                'category': doc.get('category', ''),
                'category_name': doc.get('category_name', ''),
                # Standardized date fields
                'published_date': doc.get('published_date', ''),
                'pub_date': doc.get('published_date', ''),
                'date': doc.get('published_date', ''),
                'author': doc.get('author', ''),
                'source': doc.get('source', ''),
                'url': doc.get('url', ''),
                'matched_fields': result.matched_fields,
                'snippet': doc.get('content', '')[:200] + '...' if doc.get('content') else ''
            })

        return jsonify({
            'success': True,
            'query': query_repr,
            'total_results': len(results_json),
            'response_time': response_time,
            'results': results_json
        })

    except Exception as e:
        logger.error(f"Advanced search error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/query_fields', methods=['GET'])
def api_query_fields():
    """
    Get available fields for advanced search queries.

    Returns:
        JSON with field definitions and example queries.
    """
    try:
        field_indexer, _ = get_field_indexer()

        if field_indexer is None:
            return jsonify({
                'success': False,
                'error': 'Field index not available'
            }), 503

        fields_info = {
            'title': {
                'type': 'text',
                'description': '文章標題 (tokenized)',
                'example': 'title:台灣'
            },
            'content': {
                'type': 'text',
                'description': '文章內容 (tokenized)',
                'example': 'content:人工智慧'
            },
            'category': {
                'type': 'exact',
                'description': '類別代碼 (exact match)',
                'example': 'category:aipl',
                'values': ['aipl', 'afe', 'aie', 'aloc', 'aopl', 'asoc', 'aspt', 'ait', 'acul', 'amov']
            },
            'category_name': {
                'type': 'text',
                'description': '類別名稱 (tokenized)',
                'example': 'category_name:政治'
            },
            'author': {
                'type': 'text',
                'description': '作者 (tokenized)',
                'example': 'author:記者'
            },
            'source': {
                'type': 'exact',
                'description': '來源媒體 (exact match)',
                'example': 'source:中央社'
            },
            'published_date': {
                'type': 'date',
                'description': '發布日期 (range queries)',
                'example': 'published_date:[2025-11-01 TO 2025-11-13]'
            },
            'tags': {
                'type': 'multi',
                'description': '標籤 (multi-value)',
                'example': 'tags:(AI OR 機器學習)'
            }
        }

        example_queries = [
            "title:台灣",
            "title:台灣 AND category:aipl",
            "(title:台灣 OR title:中國) AND NOT category:aspt",
            "published_date:[2025-11-01 TO 2025-11-13]",
            "category:aipl AND tags:(政治 OR 選舉)",
            "author:記者 AND content:人工智慧"
        ]

        return jsonify({
            'success': True,
            'fields': fields_info,
            'example_queries': example_queries,
            'operators': {
                'boolean': ['AND', 'OR', 'NOT'],
                'grouping': ['(', ')'],
                'field': ':',
                'range': ['[', 'TO', ']']
            }
        })

    except Exception as e:
        logger.error(f"Query fields error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def api_stats():
    """Get system statistics."""
    try:
        ret = get_retriever()

        stats = {
            'total_documents': len(ret.doc_id_map) if ret.doc_id_map else 0,
            'total_terms': len(ret.inverted_index) if ret.inverted_index else 0,
            'models': ['boolean', 'tfidf', 'bm25', 'bert'],
            'indexes_loaded': ret.inverted_index is not None
        }

        return jsonify({'success': True, 'stats': stats})

    except Exception as e:
        logger.error(f"Stats error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/filters', methods=['GET'])
def api_filters():
    """
    Get available filter options (categories and date range).

    Returns:
        JSON with available categories and min/max dates
    """
    try:
        ret = get_retriever()

        # Collect all categories and dates
        categories = set()
        dates = []

        for metadata in ret.doc_metadata.values():
            if 'category' in metadata:
                categories.add(metadata['category'])
            if 'published_date' in metadata:
                dates.append(metadata['published_date'])

        # Sort categories and find date range
        categories_list = sorted(list(categories))
        date_range = {
            'min': min(dates) if dates else None,
            'max': max(dates) if dates else None
        }

        return jsonify({
            'success': True,
            'categories': categories_list,
            'date_range': date_range
        })

    except Exception as e:
        logger.error(f"Filters error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/all_facets', methods=['GET'])
def api_all_facets():
    """
    Get all available facets from all documents (without search).

    This endpoint is used to preload facet options when the page loads,
    before any search is performed. Uses field_index directly for efficiency
    since doc_metadata may be empty.

    Response JSON:
        {
            "success": true,
            "total_documents": 9808,
            "facets": {
                "source": { ... },
                "category": { ... },
                "published_date": { ... },
                ...
            }
        }
    """
    try:
        ret = get_retriever()

        # Facet configuration with display names and types
        facet_config = {
            'source': {'display_name': '新聞來源', 'type': 'term'},
            'category': {'display_name': '分類', 'type': 'term'},
            'category_name': {'display_name': '分類名稱', 'type': 'term'},
            'published_date': {'display_name': '發布日期', 'type': 'date_range'},
            'author': {'display_name': '作者', 'type': 'term', 'max_values': 15}
        }

        # Build facets directly from field_index (more reliable than doc_metadata)
        facets_data = {}
        total_docs = len(ret.doc_map) if hasattr(ret, 'doc_map') else 0

        for field_name, config in facet_config.items():
            if hasattr(ret, 'field_index') and field_name in ret.field_index:
                field_data = ret.field_index[field_name]

                # Build value list with counts
                values = []
                for value, doc_ids in field_data.items():
                    count = len(doc_ids) if isinstance(doc_ids, (list, set)) else doc_ids

                    # For date fields, convert to readable Chinese format
                    if config['type'] == 'date_range' and isinstance(value, str):
                        try:
                            # Parse date and display with full date for clarity
                            from datetime import datetime
                            dt = datetime.strptime(value, '%Y-%m-%d')
                            label = dt.strftime('%Y年%m月%d日')
                        except:
                            label = value
                    else:
                        label = value

                    values.append({
                        'value': value,
                        'count': count,
                        'label': label
                    })

                # Sort by count descending
                values.sort(key=lambda x: (-x['count'], x['value']))

                # Apply max_values limit if configured
                max_values = config.get('max_values')
                if max_values:
                    values = values[:max_values]

                facets_data[field_name] = {
                    'field_name': field_name,
                    'display_name': config['display_name'],
                    'facet_type': config['type'],
                    'total_docs': sum(v['count'] for v in values),
                    'values': values
                }

        return jsonify({
            'success': True,
            'total_documents': total_docs,
            'facets': facets_data
        })

    except Exception as e:
        logger.error(f"All facets error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/facets', methods=['POST'])
def api_facets():
    """
    Get facet information from search results.

    Request JSON:
        {
            "query": str,
            "model": str (default: "bm25"),
            "top_k": int (default: 100),
            "facet_fields": list (optional, all if not specified)
        }

    Response JSON:
        {
            "success": true,
            "total_results": 156,
            "facets": {
                "source": {
                    "field_name": "source",
                    "display_name": "新聞來源",
                    "facet_type": "term",
                    "total_docs": 156,
                    "values": [
                        {"value": "CNA", "count": 45, "label": "CNA"},
                        ...
                    ]
                },
                ...
            }
        }
    """
    try:
        data = request.get_json()

        if not data or 'query' not in data:
            return jsonify({'success': False, 'error': 'Missing query'}), 400

        query = data.get('query', '').strip()
        model = data.get('model', 'bm25')
        top_k = min(data.get('top_k', 100), app.config['MAX_RESULTS'])
        facet_fields = data.get('facet_fields', None)

        if not query:
            return jsonify({'success': False, 'error': 'Empty query'}), 400

        # Perform search
        ret = get_retriever()
        results = ret.search(query, model=model, top_k=top_k)

        # Convert SearchResult to dict format for facet engine
        documents = []
        for r in results:
            doc_dict = {
                'id': r.doc_id,
                'title': r.title,
                'source': r.metadata.get('source', ''),
                'category': r.metadata.get('category', ''),
                'category_name': r.metadata.get('category_name', ''),
                'pub_date': r.metadata.get('published_date', ''),
                'author': r.metadata.get('author', '')
            }
            documents.append(doc_dict)

        # Build facets
        engine = get_facet_engine()
        facets = engine.build_facets(documents, field_name=facet_fields)

        # Format response
        facets_data = {}
        for field_name, facet_result in facets.items():
            facets_data[field_name] = {
                'field_name': facet_result.field_name,
                'display_name': facet_result.display_name,
                'facet_type': facet_result.facet_type,
                'total_docs': facet_result.total_docs,
                'values': [
                    {
                        'value': fv.value,
                        'count': fv.count,
                        'label': fv.label
                    }
                    for fv in facet_result.values
                ]
            }

        return jsonify({
            'success': True,
            'total_results': len(documents),
            'facets': facets_data
        })

    except Exception as e:
        logger.error(f"Facets error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/search/faceted', methods=['POST'])
def api_faceted_search():
    """
    Search with faceted filtering.

    Request JSON:
        {
            "query": str,
            "model": str,
            "top_k": int,
            "filters": {
                "source": ["CNA", "UDN"],  // Multi-select
                "category": "finance",      // Single select
                "pub_date": ["2024-11-01", "2024-11-30"]  // Range
            }
        }

    Response JSON:
        {
            "success": true,
            "query": "...",
            "total_results": 45,
            "filtered_results": 23,
            "results": [...],
            "active_filters": {
                "count": 3,
                "conditions": [...]
            }
        }
    """
    try:
        data = request.get_json()

        if not data or 'query' not in data:
            return jsonify({'success': False, 'error': 'Missing query'}), 400

        query = data.get('query', '').strip()
        model = data.get('model', 'bm25')
        top_k = min(data.get('top_k', 20), app.config['MAX_RESULTS'])
        filters = data.get('filters', {})

        if not query:
            return jsonify({'success': False, 'error': 'Empty query'}), 400

        # Perform search - get more results for filtering
        ret = get_retriever()
        results = ret.search(query, model=model, top_k=top_k * 5)

        # Convert to document format
        documents = []
        for r in results:
            doc_dict = {
                'id': r.doc_id,
                'title': r.title,
                'snippet': r.snippet,
                'score': r.score,
                'rank': r.rank,
                'source': r.metadata.get('source', ''),
                'category': r.metadata.get('category', ''),
                'category_name': r.metadata.get('category_name', ''),
                'pub_date': r.metadata.get('published_date', ''),
                'author': r.metadata.get('author', ''),
                'url': r.metadata.get('url', ''),
                'metadata': r.metadata
            }
            documents.append(doc_dict)

        total_before_filter = len(documents)

        # Apply filters if provided
        if filters:
            filter_mgr = FacetFilter()

            for field, value in filters.items():
                if isinstance(value, list):
                    if len(value) == 2 and field in ['pub_date', 'score']:
                        # Range filter
                        filter_mgr.add_condition(
                            RangeFilter(field, value[0], value[1])
                        )
                    else:
                        # Multi-select filter
                        filter_mgr.add_condition(
                            FilterCondition(field, FilterOperator.IN, value)
                        )
                else:
                    # Single value filter
                    filter_mgr.add_condition(
                        FilterCondition(field, FilterOperator.EQUALS, value)
                    )

            documents = filter_mgr.filter(documents)

        # Limit to top_k after filtering
        documents = documents[:top_k]

        return jsonify({
            'success': True,
            'query': query,
            'model': model,
            'total_results': total_before_filter,
            'filtered_results': len(documents),
            'results': documents,
            'active_filters': {
                'count': len(filters),
                'filters': filters
            }
        })

    except Exception as e:
        logger.error(f"Faceted search error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/export', methods=['POST'])
def api_export():
    """
    Export search results to JSON or CSV format.

    Request body:
        {
            "results": [...],  # Search results to export
            "format": "json" or "csv",
            "query": "...",  # Original query
            "metadata": {...}  # Additional metadata (model, filters, etc.)
        }

    Returns:
        File download response (JSON or CSV)
    """
    import csv
    import io
    from datetime import datetime

    try:
        data = request.get_json()
        results = data.get('results', [])
        export_format = data.get('format', 'json').lower()
        query = data.get('query', '')
        metadata = data.get('metadata', {})

        if not results:
            return jsonify({'success': False, 'error': 'No results to export'}), 400

        # Generate timestamp for filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if export_format == 'json':
            # Export as JSON
            export_data = {
                'export_info': {
                    'query': query,
                    'timestamp': timestamp,
                    'total_results': len(results),
                    **metadata
                },
                'results': results
            }

            filename = f"cnirs_results_{timestamp}.json"

            return jsonify(export_data), 200, {
                'Content-Disposition': f'attachment; filename={filename}',
                'Content-Type': 'application/json'
            }

        elif export_format == 'csv':
            # Export as CSV
            output = io.StringIO()

            # Define CSV fields
            fieldnames = ['rank', 'doc_id', 'title', 'score', 'category', 'date', 'content_preview']
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()

            for result in results:
                row = {
                    'rank': result.get('rank', ''),
                    'doc_id': result.get('doc_id', ''),
                    'title': result.get('title', ''),
                    'score': result.get('score', ''),
                    'category': result.get('metadata', {}).get('category', ''),
                    'date': result.get('metadata', {}).get('date', ''),
                    'content_preview': result.get('metadata', {}).get('content', '')[:200] if result.get('metadata', {}).get('content') else ''
                }
                writer.writerow(row)

            csv_content = output.getvalue()
            output.close()

            filename = f"cnirs_results_{timestamp}.csv"

            return csv_content, 200, {
                'Content-Disposition': f'attachment; filename={filename}',
                'Content-Type': 'text/csv; charset=utf-8'
            }

        else:
            return jsonify({'success': False, 'error': 'Invalid format. Use "json" or "csv"'}), 400

    except Exception as e:
        logger.error(f"Export error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/expand_query', methods=['POST'])
def api_expand_query():
    """
    Query expansion using Rocchio algorithm.

    Request JSON:
        {
            "query": "search query",
            "model": "tfidf|bm25",
            "top_k": 5,
            "use_top_results": true,
            "relevant_docs": ["doc_id1", "doc_id2"]  // optional
        }

    Response JSON:
        {
            "success": true,
            "original_query": "...",
            "expanded_query": "...",
            "expansion_terms": [
                {"term": "...", "weight": 0.85},
                ...
            ],
            "num_relevant": 5,
            "original_results": {...},
            "expanded_results": {...}
        }
    """
    try:
        data = request.get_json()

        if not data or 'query' not in data:
            return jsonify({'success': False, 'error': 'Missing query parameter'}), 400

        query = data.get('query', '').strip()
        model = data.get('model', 'tfidf').lower()
        top_k = min(data.get('top_k', 5), 10)  # Limit to 10 for PRF
        use_top_results = data.get('use_top_results', True)
        relevant_doc_ids = data.get('relevant_docs', [])

        if not query:
            return jsonify({'success': False, 'error': 'Empty query'}), 400

        if model not in ['tfidf', 'bm25']:
            return jsonify({'success': False, 'error': 'Query expansion only supports tfidf/bm25'}), 400

        ret = get_retriever()

        # Step 1: Get initial search results
        initial_results = ret.search(query, model=model, top_k=20)

        if not initial_results:
            return jsonify({
                'success': False,
                'error': 'No initial results found for query expansion'
            }), 400

        # Step 2: Get relevant documents for Rocchio
        if use_top_results and not relevant_doc_ids:
            # Pseudo-relevance feedback: use top-k results
            relevant_doc_ids = [r.doc_id for r in initial_results[:top_k]]

        if not relevant_doc_ids:
            return jsonify({
                'success': False,
                'error': 'No relevant documents for expansion'
            }), 400

        # Step 3: Get document vectors based on model
        relevant_vectors = []

        if model == 'tfidf' and ret.tfidf_data:
            # Use pre-computed TF-IDF vectors
            for doc_id in relevant_doc_ids:
                if doc_id not in ret.doc_id_map:
                    continue
                numeric_id = ret.doc_id_map[doc_id]
                doc_vec = ret.tfidf_data['document_vectors'].get(numeric_id, {})
                if doc_vec:
                    relevant_vectors.append(doc_vec)

        elif model == 'bm25' and ret.bm25_data and ret.inverted_index:
            # Build proper BM25-style vectors using correct formula
            # BM25 = IDF × (tf × (k1 + 1)) / (tf + k1 × (1 - b + b × dl/avgdl))
            k1 = ret.bm25_data.get('k1', 1.5)
            b = ret.bm25_data.get('b', 0.75)
            avgdl = ret.bm25_data.get('avg_doc_length', 100)
            idf = ret.bm25_data.get('idf', {})

            # Build a reverse index for faster lookup: doc_id -> {term: tf}
            # Only build once and cache if not exists
            if not hasattr(ret, '_doc_term_tf_cache'):
                ret._doc_term_tf_cache = {}
                for term, postings in ret.inverted_index.items():
                    for posting_doc_id, tf in postings:
                        if posting_doc_id not in ret._doc_term_tf_cache:
                            ret._doc_term_tf_cache[posting_doc_id] = {}
                        ret._doc_term_tf_cache[posting_doc_id][term] = tf

            for doc_id in relevant_doc_ids:
                if doc_id not in ret.doc_id_map:
                    continue
                numeric_id = ret.doc_id_map[doc_id]

                # Get document length from metadata
                doc_metadata = ret.doc_metadata.get(numeric_id, {})
                doc_length = doc_metadata.get('length', avgdl)

                # Build document vector using cached term frequencies
                doc_vec = {}
                term_freqs = ret._doc_term_tf_cache.get(numeric_id, {})

                for term, tf in term_freqs.items():
                    if term in idf:
                        # Proper BM25 term weight formula
                        tf_component = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_length / avgdl))
                        doc_vec[term] = idf[term] * tf_component

                if doc_vec:
                    relevant_vectors.append(doc_vec)

        if not relevant_vectors:
            return jsonify({
                'success': False,
                'error': 'Could not retrieve document vectors'
            }), 400

        # Step 4: Build query vector
        query_terms = ret.tokenizer.tokenize(query)
        query_vector = {}
        idf_source = ret.tfidf_data if model == 'tfidf' else ret.bm25_data

        for term in query_terms:
            if term in idf_source.get('idf', {}):
                query_vector[term] = idf_source['idf'][term]

        if not query_vector:
            return jsonify({
                'success': False,
                'error': 'Could not build query vector'
            }), 400

        # Step 5: Apply Rocchio expansion
        # Use very low threshold since TF-IDF vectors are normalized
        rocchio = RocchioExpander(
            alpha=1.0,
            beta=0.75,
            gamma=0.0,  # No negative feedback for PRF
            max_expansion_terms=10,
            min_term_weight=0.001  # Low threshold for normalized vectors
        )

        expanded = rocchio.expand_query(
            query_vector=query_vector,
            relevant_vectors=relevant_vectors,
            original_terms=set(query_terms)
        )

        # Step 6: Build expanded query string
        expanded_query_str = query + ' ' + ' '.join(expanded.expanded_terms[:5])

        # Step 7: Search with expanded query
        expanded_results = ret.search(expanded_query_str, model=model, top_k=20)

        # Step 8: Prepare response
        expansion_terms = [
            {'term': term, 'weight': round(expanded.term_weights.get(term, 0.0), 4)}
            for term in expanded.expanded_terms[:10]
        ]

        return jsonify({
            'success': True,
            'original_query': query,
            'expanded_query': expanded_query_str,
            'expansion_terms': expansion_terms,
            'num_relevant': expanded.num_relevant,
            'original_results': {
                'total': len(initial_results),
                'results': [
                    {
                        'doc_id': r.doc_id,
                        'title': r.title,
                        'score': r.score,
                        'rank': r.rank
                    }
                    for r in initial_results[:10]
                ]
            },
            'expanded_results': {
                'total': len(expanded_results),
                'results': [
                    {
                        'doc_id': r.doc_id,
                        'title': r.title,
                        'score': r.score,
                        'rank': r.rank
                    }
                    for r in expanded_results[:10]
                ]
            }
        })

    except Exception as e:
        logger.error(f"Query expansion error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/evaluate', methods=['POST'])
def api_evaluate():
    """
    Evaluate retrieval models with metrics.

    Request JSON:
        {
            "query": "search query",
            "models": ["tfidf", "bm25", "boolean", "bert"],
            "top_k": 20,
            "relevant_docs": [0, 5, 12]  // optional ground truth
        }

    Response JSON:
        {
            "success": true,
            "query": "...",
            "results": {
                "tfidf": {
                    "precision_at_k": [...],
                    "recall_at_k": [...],
                    "f1_at_k": [...],
                    "map": 0.45,
                    "ndcg_at_k": [...]
                },
                ...
            }
        }
    """
    try:
        start_time = time.time()
        ret = get_retriever()

        data = request.get_json()
        query = data.get('query', '').strip()
        models = data.get('models', ['tfidf', 'bm25'])
        top_k = min(int(data.get('top_k', 20)), 50)
        relevant_docs = set(data.get('relevant_docs', []))

        if not query:
            return jsonify({'success': False, 'error': 'Query is required'}), 400

        results = {}

        # If no ground truth provided, use pseudo-relevance feedback
        # Use BM25 top-5 as pseudo-relevant (BM25 is generally more reliable)
        if not relevant_docs:
            # Prefer BM25 for pseudo-relevance, fall back to first model
            prf_model = 'bm25' if 'bm25' in models else models[0]
            prf_results = ret.search(query, model=prf_model, top_k=5)
            for result in prf_results:
                if result.doc_id in ret.doc_id_map:
                    relevant_docs.add(ret.doc_id_map[result.doc_id])

        for model in models:
            try:
                # Run search
                search_results = ret.search(query, model=model, top_k=top_k)

                # Extract doc IDs (numeric IDs)
                retrieved_ids = []
                for result in search_results:
                    # Get numeric ID from doc_id_map
                    doc_id = result.doc_id
                    if doc_id in ret.doc_id_map:
                        retrieved_ids.append(ret.doc_id_map[doc_id])

                # Calculate metrics at different K values
                k_values = [5, 10, 15, 20] if top_k >= 20 else [5, 10]

                precision_at_k = []
                recall_at_k = []
                f1_at_k = []
                ndcg_at_k = []

                for k in k_values:
                    if k > len(retrieved_ids):
                        break

                    retrieved_k = retrieved_ids[:k]

                    # Calculate precision and recall
                    prec = precision(retrieved_k, relevant_docs)
                    rec = recall(retrieved_k, relevant_docs)
                    f1 = f_measure(prec, rec) if (prec + rec) > 0 else 0.0

                    precision_at_k.append({'k': k, 'value': round(prec, 4)})
                    recall_at_k.append({'k': k, 'value': round(rec, 4)})
                    f1_at_k.append({'k': k, 'value': round(f1, 4)})

                    # Calculate nDCG using document IDs as keys
                    # Build relevance scores dict with doc_id as key
                    relevance_scores = {}
                    for doc_id in retrieved_k:
                        # Binary relevance: 1 if relevant, 0 otherwise
                        relevance_scores[doc_id] = 1.0 if doc_id in relevant_docs else 0.0

                    # Use the Metrics class's ndcg_at_k which properly computes
                    # DCG for actual ranking vs IDCG for ideal ranking
                    ndcg_score = _metrics_instance.ndcg_at_k(retrieved_k, relevance_scores, k)
                    ndcg_at_k.append({'k': k, 'value': round(ndcg_score, 4)})

                # Calculate MAP
                map_score = average_precision(retrieved_ids, relevant_docs)

                # Calculate MRR (Mean Reciprocal Rank)
                mrr_score = reciprocal_rank(retrieved_ids, relevant_docs)

                # Calculate R-Precision (Precision at R, where R = number of relevant docs)
                r = len(relevant_docs)
                r_precision = precision(retrieved_ids[:r], relevant_docs) if r > 0 else 0.0

                # Calculate Precision at specific recall levels
                precision_at_recalls = []
                target_recalls = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

                # Precision-Recall points for curve
                pr_points = []
                for i in range(1, min(len(retrieved_ids), top_k) + 1):
                    retrieved_subset = retrieved_ids[:i]
                    prec = precision(retrieved_subset, relevant_docs)
                    rec = recall(retrieved_subset, relevant_docs)
                    pr_points.append({'precision': round(prec, 4), 'recall': round(rec, 4), 'rank': i})

                # 11-point interpolated precision
                for target_recall in target_recalls:
                    max_prec = 0.0
                    for point in pr_points:
                        if point['recall'] >= target_recall:
                            max_prec = max(max_prec, point['precision'])
                    precision_at_recalls.append({
                        'recall': target_recall,
                        'precision': round(max_prec, 4)
                    })

                # Calculate Precision at different recall thresholds (non-interpolated)
                precision_at_recall_raw = []
                for target_recall in [0.2, 0.5, 0.8]:
                    # Find first point where recall >= target
                    found_prec = 0.0
                    for point in pr_points:
                        if point['recall'] >= target_recall:
                            found_prec = point['precision']
                            break
                    precision_at_recall_raw.append({
                        'recall': target_recall,
                        'precision': round(found_prec, 4)
                    })

                # Calculate F-beta scores (beta=0.5 and beta=2)
                f_beta_scores = []
                for beta in [0.5, 2.0]:
                    for k in k_values:
                        if k > len(retrieved_ids):
                            break
                        retrieved_k = retrieved_ids[:k]
                        prec = precision(retrieved_k, relevant_docs)
                        rec = recall(retrieved_k, relevant_docs)
                        if (beta**2 * prec + rec) > 0:
                            f_beta = ((1 + beta**2) * prec * rec) / (beta**2 * prec + rec)
                        else:
                            f_beta = 0.0
                        f_beta_scores.append({
                            'k': k,
                            'beta': beta,
                            'value': round(f_beta, 4)
                        })

                # Calculate Bpref (Binary Preference)
                bpref = 0.0
                if len(relevant_docs) > 0:
                    non_relevant_before_relevant = 0
                    for i, doc_id in enumerate(retrieved_ids):
                        if doc_id in relevant_docs:
                            # Count non-relevant docs before this relevant doc
                            bpref += 1 - (non_relevant_before_relevant / len(relevant_docs))
                        else:
                            non_relevant_before_relevant += 1
                    bpref /= len(relevant_docs)

                results[model] = {
                    'precision_at_k': precision_at_k,
                    'recall_at_k': recall_at_k,
                    'f1_at_k': f1_at_k,
                    'ndcg_at_k': ndcg_at_k,
                    'map': round(map_score, 4),
                    'mrr': round(mrr_score, 4),
                    'r_precision': round(r_precision, 4),
                    'bpref': round(bpref, 4),
                    'pr_curve': pr_points,
                    'interpolated_precision': precision_at_recalls,
                    'precision_at_recall': precision_at_recall_raw,
                    'f_beta_scores': f_beta_scores,
                    'total_retrieved': len(retrieved_ids)
                }

            except Exception as e:
                logger.error(f"Error evaluating model {model}: {e}")
                results[model] = {'error': str(e)}

        response_time = time.time() - start_time

        return jsonify({
            'success': True,
            'query': query,
            'relevant_count': len(relevant_docs),
            'results': results,
            'response_time': round(response_time, 4)
        })

    except Exception as e:
        logger.error(f"Evaluation error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/summarize', methods=['POST'])
def api_summarize():
    """
    Generate summary for a document.

    Request JSON:
        {
            "doc_id": "20251107_001",
            "method": "lead_k" or "key_sentence",
            "k": 3  // number of sentences
        }

    Response JSON:
        {
            "success": true,
            "doc_id": "...",
            "method": "...",
            "summary": "...",
            "processing_time": 0.023
        }
    """
    try:
        start_time = time.time()
        ret = get_retriever()

        data = request.get_json()
        doc_id = data.get('doc_id', '').strip()
        method = data.get('method', 'key_sentence')
        k = min(int(data.get('k', 3)), 10)

        if not doc_id:
            return jsonify({'success': False, 'error': 'doc_id is required'}), 400

        # Get document content
        numeric_id = ret.doc_id_map.get(doc_id)
        if numeric_id is None:
            return jsonify({'success': False, 'error': 'Document not found'}), 404

        # Load content from preprocessed file
        content = load_document_content(doc_id)

        if not content:
            return jsonify({'success': False, 'error': 'Document has no content'}), 400

        # Generate summary
        if method == 'lead_k':
            summary = lead_k_summary(content, k=k)
        else:  # key_sentence
            summary = key_sentence_summary(content, k=k)

        processing_time = time.time() - start_time

        return jsonify({
            'success': True,
            'doc_id': doc_id,
            'method': method,
            'k': k,
            'summary': summary,
            'processing_time': round(processing_time, 4)
        })

    except Exception as e:
        logger.error(f"Summarization error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/algorithms', methods=['GET'])
def api_algorithms():
    """
    Get list of available algorithms for各種功能。

    Response JSON:
        {
            "summarization": ["lead_k", "key_sentence", "tfidf", "position_weighted"],
            "keyword_extraction": ["tfidf", "textrank", "term_frequency"],
            "clustering": ["kmeans", "hierarchical", "star"],
            "topic_detection": ["lda", "term_clustering"]
        }
    """
    return jsonify({
        'summarization': {
            'lead_k': 'Lead-K: Extract first k sentences',
            'key_sentence': 'Key Sentence: TF-IDF based sentence extraction',
            'tfidf': 'TF-IDF Scoring: Content-based ranking',
            'position_weighted': 'Position-Weighted: Combine content and position scores'
        },
        'keyword_extraction': {
            'tfidf': 'TF-IDF: Term frequency × Inverse document frequency',
            'term_frequency': 'Term Frequency: Simple frequency counting',
            'textrank': 'TextRank: Graph-based keyword extraction (simulated)',
            'pat_tree': 'PAT-Tree: Patricia tree-based substring frequency extraction'
        },
        'clustering': {
            'kmeans': 'K-Means: Partition-based clustering',
            'hierarchical': 'Hierarchical: Agglomerative clustering',
            'star': 'Star Clustering: String-based clustering for terms'
        },
        'topic_detection': {
            'term_clustering': 'Term Clustering: Group similar terms',
            'simple_lda': 'Simple Topic Detection: Frequency-based topics'
        }
    })


@app.route('/api/extract_keywords', methods=['POST'])
def api_extract_keywords():
    """
    Extract keywords from a document with multiple algorithm support.

    Request JSON:
        {
            "doc_id": "20251107_001",
            "top_k": 10,  // number of keywords
            "method": "tfidf"  // "tfidf", "term_frequency", "textrank", or "pat_tree"
        }

    Response JSON:
        {
            "success": true,
            "doc_id": "...",
            "method": "tfidf",
            "keywords": [{"word": "...", "score": 0.95}, ...],
            "processing_time": 0.023
        }
    """
    try:
        start_time = time.time()
        ret = get_retriever()

        data = request.get_json()
        doc_id = data.get('doc_id', '').strip()
        top_k = min(int(data.get('top_k', 10)), 50)
        method = data.get('method', 'tfidf')  # Default to TF-IDF

        if not doc_id:
            return jsonify({'success': False, 'error': 'doc_id is required'}), 400

        # Get document content
        numeric_id = ret.doc_id_map.get(doc_id)
        if numeric_id is None:
            return jsonify({'success': False, 'error': 'Document not found'}), 404

        # Load content from preprocessed file
        content = load_document_content(doc_id)

        if not content:
            return jsonify({'success': False, 'error': 'Document has no content'}), 400

        # Tokenize content
        tokenizer = ChineseTokenizer()
        tokens = tokenizer.tokenize(content)

        # Calculate TF for this document
        tf = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1

        keywords = []

        if method == 'tfidf':
            # TF-IDF method
            for term, freq in tf.items():
                if term in ret.inverted_index:
                    doc_freq = len(ret.inverted_index[term])
                    total_docs = len(ret.doc_id_map)
                    idf = 1.0 + (total_docs / (1.0 + doc_freq))
                    tfidf = freq * idf
                    keywords.append({'word': term, 'score': round(tfidf, 4)})

        elif method == 'term_frequency':
            # Simple term frequency method
            for term, freq in tf.items():
                keywords.append({'word': term, 'score': float(freq)})

        elif method == 'textrank':
            # TextRank-like method (simplified using TF and term co-occurrence)
            # Build simple co-occurrence graph
            for term, freq in tf.items():
                # Score based on frequency and term length (as proxy for importance)
                score = freq * (1 + len(term) / 10.0)
                keywords.append({'word': term, 'score': round(score, 4)})

        elif method == 'pat_tree':
            # PAT-Tree method: Use Patricia tree for substring extraction
            # Build document-specific PAT-tree from tokens
            from src.ir.index.pat_tree import PatriciaTree
            doc_pat_tree = PatriciaTree()

            # Insert each token into the PAT-tree
            for token in tokens:
                if len(token) >= 2:  # Only consider terms with 2+ chars
                    doc_pat_tree.insert(token, doc_id)

            # Use term_stats from PAT-tree to extract keywords
            if hasattr(doc_pat_tree, 'term_stats') and doc_pat_tree.term_stats:
                for term, stats in doc_pat_tree.term_stats.items():
                    if len(term) >= 2:
                        freq = stats.get('frequency', 1)
                        # Score based on frequency and term length (longer terms are more specific)
                        score = freq * (1 + len(term) * 0.5)
                        keywords.append({'word': term, 'score': round(score, 4)})
            else:
                # Fallback: use term frequency with length bonus
                for term, freq in tf.items():
                    if len(term) >= 2:
                        score = freq * (1 + len(term) * 0.5)
                        keywords.append({'word': term, 'score': round(score, 4)})

        else:
            return jsonify({'success': False, 'error': f'Unknown method: {method}'}), 400

        # Sort by score and get top k
        keywords.sort(key=lambda x: x['score'], reverse=True)
        keywords = keywords[:top_k]

        processing_time = time.time() - start_time

        return jsonify({
            'success': True,
            'doc_id': doc_id,
            'method': method,
            'keywords': keywords,
            'processing_time': round(processing_time, 4)
        })

    except Exception as e:
        logger.error(f"Keyword extraction error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/detect_topics', methods=['POST'])
def api_detect_topics():
    """
    Detect topics from documents using clustering algorithms.

    Request JSON:
        {
            "query": "台灣",  // optional query to filter documents
            "top_k": 20,      // number of documents to analyze
            "num_topics": 5,  // number of topics to extract
            "method": "term_clustering"  // "term_clustering" or "simple_lda"
        }

    Response JSON:
        {
            "success": true,
            "method": "term_clustering",
            "topics": [
                {
                    "topic_id": 1,
                    "keywords": ["台灣", "政治", ...],
                    "weight": 0.85,
                    "doc_count": 15
                },
                ...
            ],
            "processing_time": 0.5
        }
    """
    try:
        start_time = time.time()
        ret = get_retriever()

        data = request.get_json()
        query = data.get('query', '').strip()
        top_k = min(int(data.get('top_k', 20)), 100)
        num_topics = min(int(data.get('num_topics', 5)), 10)
        method = data.get('method', 'term_clustering')

        # Get documents to analyze
        if query:
            # Search for relevant documents
            results = ret.search(query, model='tfidf', top_k=top_k)
            doc_ids = [r.doc_id for r in results]
        else:
            # Use top documents from collection
            doc_ids = list(ret.doc_id_map.keys())[:top_k]

        # Extract content from documents
        documents = []
        tokenizer = ChineseTokenizer()

        for doc_id in doc_ids:
            content = load_document_content(doc_id)
            if content:
                tokens = tokenizer.tokenize(content)
                documents.append({
                    'doc_id': doc_id,
                    'tokens': tokens,
                    'content': content
                })

        if not documents:
            return jsonify({'success': False, 'error': 'No documents found'}), 404

        topics = []

        if method == 'term_clustering':
            # Use term clustering to find topic keywords
            term_clusterer = TermClusterer()

            # Collect all terms with frequencies
            all_terms = []
            for doc in documents:
                all_terms.extend(doc['tokens'])

            term_freq = Counter(all_terms)
            # Get top terms as vocabulary
            vocab = [term for term, _ in term_freq.most_common(100)]

            if len(vocab) >= num_topics:
                # Cluster terms
                result = term_clusterer.cluster_strings(
                    vocab,
                    num_clusters=num_topics,
                    method='star'
                )

                for i, cluster in enumerate(result.clusters[:num_topics]):
                    topics.append({
                        'topic_id': i + 1,
                        'keywords': cluster.members[:10],  # Top 10 terms
                        'weight': len(cluster.members) / len(vocab),
                        'doc_count': len([d for d in documents if any(t in d['tokens'] for t in cluster.members)])
                    })

        elif method == 'simple_lda':
            # Simple topic detection based on term co-occurrence
            # Build term-document matrix
            all_terms = set()
            for doc in documents:
                all_terms.update(doc['tokens'])

            # Get top terms by frequency
            term_freq = Counter()
            for doc in documents:
                term_freq.update(doc['tokens'])

            top_terms = [term for term, _ in term_freq.most_common(50)]

            # Create topics by grouping related terms
            terms_per_topic = len(top_terms) // num_topics
            for i in range(num_topics):
                start_idx = i * terms_per_topic
                end_idx = start_idx + terms_per_topic
                topic_terms = top_terms[start_idx:end_idx]

                topics.append({
                    'topic_id': i + 1,
                    'keywords': topic_terms,
                    'weight': 1.0 / num_topics,
                    'doc_count': len([d for d in documents if any(t in d['tokens'] for t in topic_terms)])
                })

        processing_time = time.time() - start_time

        return jsonify({
            'success': True,
            'method': method,
            'topics': topics,
            'num_documents': len(documents),
            'processing_time': round(processing_time, 4)
        })

    except Exception as e:
        logger.error(f"Topic detection error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/pat_tree', methods=['GET'])
def api_pat_tree():
    """
    Get PAT-tree (Patricia Trie) visualization structure.

    Query parameters:
        prefix: Filter terms by prefix (optional)
        max_nodes: Maximum number of nodes to return (default: 100, max: 500)

    Response JSON:
        {
            "success": true,
            "tree": {
                "label": "ROOT",
                "key": "",
                "terminal": false,
                "frequency": 0,
                "doc_count": 0,
                "children": [...]
            },
            "statistics": {
                "total_terms": 45621,
                "unique_terms": 10248,
                "total_nodes": 15320,
                "max_depth": 12,
                "compression_ratio": 2.98,
                "avg_term_frequency": 4.45
            },
            "prefix": "台",
            "processing_time": 0.015
        }
    """
    try:
        start_time = time.time()

        # Get parameters
        prefix = request.args.get('prefix', '').strip()
        max_nodes = min(int(request.args.get('max_nodes', 100)), 500)

        # Get PAT-tree
        pat_tree = get_pat_tree()

        # Generate visualization data
        viz_data = pat_tree.visualize_tree(max_nodes=max_nodes, prefix=prefix)

        processing_time = time.time() - start_time

        return jsonify({
            'success': True,
            'tree': viz_data.get('tree'),
            'statistics': viz_data.get('statistics'),
            'prefix': prefix,
            'processing_time': round(processing_time, 4)
        })

    except Exception as e:
        logger.error(f"PAT-tree visualization error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/pat_tree_keywords', methods=['POST'])
def api_pat_tree_keywords():
    """
    Extract keywords from PAT-tree with post-processing.

    Request JSON:
        {
            "top_k": 20,
            "min_freq": 2,
            "min_doc_freq": 1,
            "method": "tfidf"  // or "frequency", "doc_frequency", "combined"
        }

    Response JSON:
        {
            "success": true,
            "keywords": [
                {
                    "term": "台灣",
                    "frequency": 1250,
                    "doc_frequency": 450,
                    "doc_count": 450,
                    "score": 12.34,
                    "tf": 0.0274,
                    "idf": 1.89,
                    "rank": 1,
                    "percentile": 99.5
                },
                ...
            ],
            "method": "tfidf",
            "total_candidates": 8453,
            "processing_time": 0.025
        }
    """
    try:
        start_time = time.time()
        data = request.get_json() or {}

        # Get parameters with defaults
        top_k = min(int(data.get('top_k', 20)), 100)
        min_freq = int(data.get('min_freq', 2))
        min_doc_freq = int(data.get('min_doc_freq', 1))
        method = data.get('method', 'tfidf')

        # Validate method
        valid_methods = ['tfidf', 'frequency', 'doc_frequency', 'combined']
        if method not in valid_methods:
            return jsonify({
                'success': False,
                'error': f'Invalid method. Must be one of: {", ".join(valid_methods)}'
            }), 400

        # Get PAT-tree
        pat_tree = get_pat_tree()

        # Extract keywords with post-processing
        keywords = pat_tree.extract_keywords(
            top_k=top_k,
            min_freq=min_freq,
            min_doc_freq=min_doc_freq,
            method=method
        )

        processing_time = time.time() - start_time

        return jsonify({
            'success': True,
            'keywords': keywords,
            'method': method,
            'total_candidates': len([k for k, s in pat_tree.term_stats.items()
                                    if s['frequency'] >= min_freq and s['doc_frequency'] >= min_doc_freq]),
            'parameters': {
                'top_k': top_k,
                'min_freq': min_freq,
                'min_doc_freq': min_doc_freq,
                'method': method
            },
            'processing_time': round(processing_time, 4)
        })

    except Exception as e:
        logger.error(f"PAT-tree keyword extraction error: {e}", exc_info=True)
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='CNIRS Web Application')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port to bind to (default: 5000)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    parser.add_argument('--index-dir', type=str,
                       help='Path to indexes directory')

    args = parser.parse_args()

    if args.index_dir:
        app.config['INDEX_DIR'] = Path(args.index_dir)

    logger.info(f"Starting CNIRS Web Application on {args.host}:{args.port}")
    logger.info(f"Index directory: {app.config['INDEX_DIR']}")

    app.run(host=args.host, port=args.port, debug=args.debug)
