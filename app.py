"""
Flask Web Application for Information Retrieval System

This demo application showcases the IR system capabilities:
- Boolean retrieval (AND/OR/NOT queries)
- Vector Space Model search (TF-IDF + Cosine Similarity)
- Document summarization (Lead-K, Key Sentences, KWIC)
- Query expansion (Rocchio algorithm)
- Document clustering

Dataset: 121 CNA news articles (data/processed/cna_mvp_cleaned.jsonl)

Author: Information Retrieval System
License: Educational Use
"""

import os
import json
import logging
from typing import List, Dict, Any
from datetime import datetime

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# Import IR modules
from src.ir.index.inverted_index import InvertedIndex
from src.ir.index.positional_index import PositionalIndex
from src.ir.index.field_indexer import FieldIndexer
from src.ir.retrieval.boolean import BooleanQueryEngine
from src.ir.retrieval.vsm import VectorSpaceModel
from src.ir.retrieval.bm25 import BM25Ranker
from src.ir.retrieval.language_model_retrieval import LanguageModelRetrieval
from src.ir.retrieval.bim import BinaryIndependenceModel
from src.ir.retrieval.query_optimization import WANDRetrieval, MaxScoreRetrieval
from src.ir.index.term_weighting import TermWeighting
from src.ir.summarize.static import StaticSummarizer
from src.ir.summarize.dynamic import KWICGenerator
from src.ir.ranking.rocchio import RocchioExpander
from src.ir.ranking.hybrid import HybridRanker
from src.ir.cluster.doc_cluster import DocumentClusterer
from src.ir.text.chinese_tokenizer import ChineseTokenizer
from src.ir.langmodel.ngram import NGramModel
from src.ir.langmodel.collocation import CollocationExtractor

# Optional: BERT retrieval (requires transformers)
try:
    from src.ir.semantic.bert_retrieval import BERTRetrieval
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    logger.warning("BERT retrieval not available (transformers/torch not installed)")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables for IR system components
inverted_index = None
positional_index = None
field_indexer = None  # Field-based metadata indexer
boolean_engine = None
vsm = None
bm25_ranker = None  # BM25 ranking
lm_retrieval = None  # Language Model retrieval
bim_model = None  # Binary Independence Model
bert_retrieval = None  # BERT semantic retrieval (optional)
hybrid_ranker = None  # Hybrid ranking (combines multiple rankers)
wand_retrieval = None  # WAND query optimization
maxscore_retrieval = None  # MaxScore query optimization
ngram_model = None  # N-gram language model
collocation_extractor = None  # Collocation extraction
summarizer = None
kwic_generator = None
rocchio_expander = None
doc_clusterer = None
chinese_tokenizer = None  # CKIP Transformers tokenizer
documents = []
doc_id_to_article = {}
doc_linguistic_data = {}  # Store POS, NER data for each document


def load_dataset(filepath: str = 'data/processed/cna_mvp_cleaned.jsonl') -> List[Dict[str, Any]]:
    """
    Load the cleaned CNA dataset.

    Args:
        filepath: Path to JSONL file

    Returns:
        List of article dictionaries
    """
    articles = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                articles.append(json.loads(line))
    logger.info(f"Loaded {len(articles)} articles from {filepath}")
    return articles


def initialize_ir_system():
    """Initialize all IR system components with the dataset."""
    global inverted_index, positional_index, field_indexer, boolean_engine, vsm
    global bm25_ranker, lm_retrieval, bim_model, bert_retrieval, hybrid_ranker
    global wand_retrieval, maxscore_retrieval
    global ngram_model, collocation_extractor
    global summarizer, kwic_generator, rocchio_expander, doc_clusterer
    global documents, doc_id_to_article, chinese_tokenizer, doc_linguistic_data

    logger.info("Initializing IR system...")

    # Initialize CKIP Transformers tokenizer
    logger.info("Loading CKIP Transformers (this may take a moment)...")
    chinese_tokenizer = ChineseTokenizer(
        engine='ckip',
        mode='default',
        use_pos=True,  # Enable POS tagging
        device=-1  # Use CPU (-1) or GPU (0+)
    )
    logger.info("CKIP Transformers loaded successfully")

    # Load dataset
    articles = load_dataset()
    documents = [article['content'] for article in articles]
    doc_id_to_article = {i: article for i, article in enumerate(articles)}

    # Preprocess documents with CKIP for linguistic analysis
    logger.info("Performing linguistic analysis (tokenization, POS, NER)...")
    for doc_id, doc_text in enumerate(documents):
        # Tokenize with POS
        tokens_with_pos = chinese_tokenizer.tokenize_with_pos(doc_text)

        # Extract entities
        entities = chinese_tokenizer.extract_entities(doc_text)

        # Store linguistic data
        doc_linguistic_data[doc_id] = {
            'tokens': [token for token, pos in tokens_with_pos],
            'pos_tags': tokens_with_pos,
            'entities': entities
        }

        if (doc_id + 1) % 20 == 0:
            logger.info(f"Processed {doc_id + 1}/{len(documents)} documents")

    logger.info(f"Linguistic analysis complete for {len(documents)} documents")

    # Build inverted index with CKIP tokenizer
    logger.info("Building inverted index...")
    inverted_index = InvertedIndex(tokenizer=chinese_tokenizer.tokenize)
    inverted_index.build(documents)

    # Build positional index with CKIP tokenizer
    logger.info("Building positional index...")
    positional_index = PositionalIndex(tokenizer=chinese_tokenizer.tokenize)
    positional_index.build(documents)

    # Build field indexer for metadata search
    logger.info("Building field indexer for metadata search...")
    field_indexer = FieldIndexer(tokenizer=chinese_tokenizer.tokenize)
    field_indexer.build(list(doc_id_to_article.values()))

    # Initialize Boolean retrieval engine with field support
    logger.info("Initializing Boolean retrieval engine...")
    boolean_engine = BooleanQueryEngine(
        inverted_index=inverted_index,
        positional_index=positional_index,
        field_indexer=field_indexer
    )

    # Build Vector Space Model
    logger.info("Building Vector Space Model...")
    vsm = VectorSpaceModel()
    vsm.build_index(documents)

    # Initialize summarizers
    logger.info("Initializing summarizers...")
    summarizer = StaticSummarizer()
    kwic_generator = KWICGenerator()

    # Initialize query expander
    logger.info("Initializing Rocchio query expander...")
    rocchio_expander = RocchioExpander()

    # Initialize document clusterer
    logger.info("Initializing document clusterer...")
    doc_clusterer = DocumentClusterer()

    # Initialize BM25 ranker
    logger.info("Building BM25 ranker...")
    bm25_ranker = BM25Ranker(tokenizer=chinese_tokenizer.tokenize, k1=1.5, b=0.75)
    bm25_ranker.build_index(documents)

    # Initialize Language Model retrieval
    logger.info("Building Language Model retrieval (Dirichlet smoothing)...")
    lm_retrieval = LanguageModelRetrieval(
        tokenizer=chinese_tokenizer.tokenize,
        smoothing='dirichlet',
        mu_param=2000
    )
    lm_retrieval.build_index(documents)

    # Initialize Binary Independence Model
    logger.info("Building Binary Independence Model...")
    bim_model = BinaryIndependenceModel(
        tokenizer=chinese_tokenizer.tokenize,
        use_idf=True
    )
    bim_model.build_index(documents)

    # Initialize N-gram model
    logger.info("Training N-gram language model (bigram with JM smoothing)...")
    ngram_model = NGramModel(
        n=2,
        smoothing='jm',
        lambda_param=0.7,
        tokenizer=chinese_tokenizer.tokenize
    )
    ngram_model.train(documents)

    # Initialize Collocation extractor
    logger.info("Training collocation extractor...")
    collocation_extractor = CollocationExtractor(
        tokenizer=chinese_tokenizer.tokenize,
        min_freq=3,
        window_size=2
    )
    collocation_extractor.train(documents)

    # Initialize BERT retrieval (optional, requires transformers)
    if BERT_AVAILABLE:
        try:
            logger.info("Building BERT semantic retrieval (this may take a while)...")
            bert_retrieval = BERTRetrieval(
                model_name="bert-base-chinese",
                device="cpu",
                use_faiss=False
            )
            bert_retrieval.build_index(documents, batch_size=8)
            logger.info("BERT retrieval initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize BERT retrieval: {e}")
            bert_retrieval = None
    else:
        logger.info("BERT retrieval skipped (transformers not available)")
        bert_retrieval = None

    # Initialize Hybrid ranker (combines BM25, VSM, LM, and optionally BERT)
    logger.info("Initializing hybrid ranker...")
    rankers_dict = {
        'bm25': bm25_ranker,
        'vsm': vsm,
        'lm': lm_retrieval
    }
    if bert_retrieval is not None:
        rankers_dict['bert'] = bert_retrieval

    hybrid_ranker = HybridRanker(
        rankers=rankers_dict,
        fusion_method='rrf',  # Reciprocal Rank Fusion
        weights=None,  # Equal weights
        normalization='minmax'
    )

    # Initialize WAND query optimization
    logger.info("Initializing WAND query optimization...")
    wand_retrieval = WANDRetrieval(
        inverted_index=inverted_index.index,
        doc_lengths={doc_id: len(doc.split()) for doc_id, doc in enumerate(documents)},
        doc_count=len(documents),
        avg_doc_length=sum(len(doc.split()) for doc in documents) / len(documents) if documents else 0
    )

    # Initialize MaxScore query optimization
    logger.info("Initializing MaxScore query optimization...")
    maxscore_retrieval = MaxScoreRetrieval(
        inverted_index=inverted_index.index,
        doc_lengths={doc_id: len(doc.split()) for doc_id, doc in enumerate(documents)},
        doc_count=len(documents),
        avg_doc_length=sum(len(doc.split()) for doc in documents) / len(documents) if documents else 0
    )

    logger.info("IR system initialized successfully!")


@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html', doc_count=len(documents))


@app.route('/api/stats')
def get_stats():
    """Get dataset and index statistics."""
    if not inverted_index:
        return jsonify({'error': 'System not initialized'}), 500

    stats = {
        'documents': len(documents),
        'vocabulary_size': len(inverted_index.vocabulary),
        'avg_doc_length': inverted_index.avg_doc_length if hasattr(inverted_index, 'avg_doc_length') else 0,
        'total_terms': sum(len(inverted_index.get_postings(term)) for term in inverted_index.vocabulary)
    }

    return jsonify(stats)


@app.route('/api/search/boolean', methods=['POST'])
def boolean_search():
    """
    Perform Boolean search.

    Request JSON:
        {
            "query": "台灣 AND 經濟",
            "limit": 10
        }

    Response JSON:
        {
            "query": "台灣 AND 經濟",
            "results": [
                {
                    "doc_id": 0,
                    "title": "...",
                    "snippet": "...",
                    "url": "..."
                }
            ],
            "total": 5,
            "execution_time": 0.123
        }
    """
    if not boolean_engine:
        return jsonify({'error': 'System not initialized'}), 500

    data = request.get_json()
    query = data.get('query', '')
    limit = data.get('limit', 10)

    if not query:
        return jsonify({'error': 'Query is required'}), 400

    # Perform search
    start_time = datetime.now()
    try:
        doc_ids = boolean_engine.search(query)
        execution_time = (datetime.now() - start_time).total_seconds()

        # Prepare results
        results = []
        for doc_id in doc_ids[:limit]:
            article = doc_id_to_article[doc_id]
            results.append({
                'doc_id': doc_id,
                'title': article['title'],
                'snippet': article['content'][:200] + '...' if len(article['content']) > 200 else article['content'],
                'url': article['url'],
                'date': article['published_date'],
                'category': article.get('category_name', '')
            })

        return jsonify({
            'query': query,
            'results': results,
            'total': len(doc_ids),
            'execution_time': execution_time
        })
    except Exception as e:
        logger.error(f"Boolean search error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/search/vsm', methods=['POST'])
def vsm_search():
    """
    Perform Vector Space Model search.

    Request JSON:
        {
            "query": "人工智慧發展",
            "limit": 10
        }

    Response JSON:
        {
            "query": "人工智慧發展",
            "results": [
                {
                    "doc_id": 0,
                    "title": "...",
                    "snippet": "...",
                    "score": 0.85,
                    "url": "..."
                }
            ],
            "total": 10,
            "execution_time": 0.234
        }
    """
    if not vsm:
        return jsonify({'error': 'System not initialized'}), 500

    data = request.get_json()
    query = data.get('query', '')
    limit = data.get('limit', 10)

    if not query:
        return jsonify({'error': 'Query is required'}), 400

    # Perform search
    start_time = datetime.now()
    try:
        search_result = vsm.search(query, topk=limit)
        execution_time = (datetime.now() - start_time).total_seconds()

        # Prepare results
        results = []
        for doc_id, score in zip(search_result.doc_ids, search_result.scores):
            article = doc_id_to_article[doc_id]
            results.append({
                'doc_id': doc_id,
                'title': article['title'],
                'snippet': article['content'][:200] + '...' if len(article['content']) > 200 else article['content'],
                'score': round(score, 4),
                'url': article['url'],
                'date': article['published_date'],
                'category': article.get('category_name', '')
            })

        return jsonify({
            'query': query,
            'results': results,
            'total': len(search_result.doc_ids),
            'execution_time': execution_time
        })
    except Exception as e:
        logger.error(f"VSM search error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/document/<int:doc_id>')
def get_document(doc_id):
    """Get full document details."""
    if doc_id not in doc_id_to_article:
        return jsonify({'error': 'Document not found'}), 404

    article = doc_id_to_article[doc_id]
    return jsonify(article)


@app.route('/api/summarize/<int:doc_id>', methods=['POST'])
def summarize_document(doc_id):
    """
    Generate summary for a document.

    Request JSON:
        {
            "method": "lead_k",  // lead_k, key_sentence, kwic
            "k": 3,
            "keyword": "台灣"  // for KWIC only
        }
    """
    if doc_id not in doc_id_to_article:
        return jsonify({'error': 'Document not found'}), 404

    if not summarizer:
        return jsonify({'error': 'System not initialized'}), 500

    data = request.get_json()
    method = data.get('method', 'lead_k')
    k = data.get('k', 3)
    keyword = data.get('keyword', '')

    article = doc_id_to_article[doc_id]
    text = article['content']

    try:
        if method == 'lead_k':
            summary = summarizer.lead_k_summarization(text, k=k)
        elif method == 'key_sentence':
            summary = summarizer.key_sentence_extraction(text, k=k)
        elif method == 'kwic':
            if not keyword:
                return jsonify({'error': 'Keyword required for KWIC'}), 400
            kwic_result = kwic_generator.generate(text, keyword)
            return jsonify({
                'method': 'kwic',
                'keyword': keyword,
                'contexts': kwic_result.contexts[:k] if kwic_result.contexts else []
            })
        else:
            return jsonify({'error': 'Invalid method'}), 400

        return jsonify({
            'method': method,
            'k': k,
            'summary': summary.sentences if hasattr(summary, 'sentences') else []
        })
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/expand_query', methods=['POST'])
def expand_query():
    """
    Perform query expansion using Rocchio algorithm.

    Request JSON:
        {
            "query": "人工智慧",
            "relevant_docs": [0, 1, 2]  // Optional: doc IDs
        }
    """
    if not vsm or not rocchio_expander:
        return jsonify({'error': 'System not initialized'}), 500

    data = request.get_json()
    query = data.get('query', '')
    relevant_doc_ids = data.get('relevant_docs', [])

    if not query:
        return jsonify({'error': 'Query is required'}), 400

    try:
        # Get initial search results if no relevant docs provided
        if not relevant_doc_ids:
            search_result = vsm.search(query, topk=5)
            relevant_doc_ids = search_result.doc_ids[:3] if search_result.doc_ids else []

        if not relevant_doc_ids:
            return jsonify({'error': 'No relevant documents found'}), 400

        # Get document vectors
        relevant_vectors = [vsm.get_document_vector(doc_id) for doc_id in relevant_doc_ids]

        # Get query vector
        query_tokens = vsm.inverted_index.tokenizer(query)
        query_vector = {term: 1.0 for term in query_tokens}

        # Expand query
        expanded_result = rocchio_expander.expand_query(query_vector, relevant_vectors)

        # Get top expansion terms
        expansion_terms = []
        for term, weight in sorted(expanded_result.expanded_vector.items(),
                                   key=lambda x: x[1], reverse=True)[:10]:
            if term not in query_vector:
                expansion_terms.append({'term': term, 'weight': round(weight, 4)})

        return jsonify({
            'original_query': query,
            'expansion_terms': expansion_terms,
            'relevant_docs': relevant_doc_ids
        })
    except Exception as e:
        logger.error(f"Query expansion error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/cluster', methods=['POST'])
def cluster_documents():
    """
    Perform document clustering.

    Request JSON:
        {
            "n_clusters": 3,
            "method": "hierarchical",  // hierarchical or kmeans
            "doc_ids": [0, 1, 2, ...]  // Optional: specific docs to cluster
        }
    """
    if not doc_clusterer:
        return jsonify({'error': 'System not initialized'}), 500

    data = request.get_json()
    n_clusters = data.get('n_clusters', 3)
    method = data.get('method', 'hierarchical')
    doc_ids = data.get('doc_ids', list(range(len(documents))))

    # Limit to first 50 docs for performance
    if len(doc_ids) > 50:
        doc_ids = doc_ids[:50]

    try:
        # Get documents to cluster
        docs_to_cluster = [documents[i] for i in doc_ids]

        # Perform clustering
        clusters = doc_clusterer.cluster(
            docs_to_cluster,
            n_clusters=n_clusters,
            method=method
        )

        # Prepare results
        cluster_results = []
        for cluster in clusters:
            cluster_doc_ids = [doc_ids[i] for i in cluster.doc_ids]
            cluster_results.append({
                'cluster_id': cluster.cluster_id,
                'size': len(cluster.doc_ids),
                'doc_ids': cluster_doc_ids,
                'documents': [
                    {
                        'doc_id': doc_id,
                        'title': doc_id_to_article[doc_id]['title']
                    }
                    for doc_id in cluster_doc_ids[:5]  # Only show first 5
                ]
            })

        return jsonify({
            'method': method,
            'n_clusters': n_clusters,
            'clusters': cluster_results
        })
    except Exception as e:
        logger.error(f"Clustering error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/search/bm25', methods=['POST'])
def bm25_search():
    """
    Perform BM25 ranking search.

    Request JSON:
        {
            "query": "人工智慧發展",
            "limit": 10
        }
    """
    if not bm25_ranker:
        return jsonify({'error': 'System not initialized'}), 500

    data = request.get_json()
    query = data.get('query', '')
    limit = data.get('limit', 10)

    if not query:
        return jsonify({'error': 'Query is required'}), 400

    start_time = datetime.now()
    try:
        result = bm25_ranker.search(query, topk=limit)
        execution_time = (datetime.now() - start_time).total_seconds()

        # Prepare results
        results = []
        for doc_id, score in zip(result.doc_ids, result.scores):
            article = doc_id_to_article[doc_id]
            results.append({
                'doc_id': doc_id,
                'title': article['title'],
                'snippet': article['content'][:200] + '...' if len(article['content']) > 200 else article['content'],
                'score': round(score, 4),
                'url': article['url'],
                'date': article['published_date'],
                'category': article.get('category_name', '')
            })

        return jsonify({
            'query': query,
            'model': 'BM25',
            'results': results,
            'total': len(result.doc_ids),
            'parameters': result.parameters,
            'execution_time': execution_time
        })
    except Exception as e:
        logger.error(f"BM25 search error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/search/lm', methods=['POST'])
def language_model_search():
    """
    Perform Language Model retrieval search.

    Request JSON:
        {
            "query": "機器學習應用",
            "limit": 10
        }
    """
    if not lm_retrieval:
        return jsonify({'error': 'System not initialized'}), 500

    data = request.get_json()
    query = data.get('query', '')
    limit = data.get('limit', 10)

    if not query:
        return jsonify({'error': 'Query is required'}), 400

    start_time = datetime.now()
    try:
        result = lm_retrieval.search(query, topk=limit)
        execution_time = (datetime.now() - start_time).total_seconds()

        # Prepare results
        results = []
        for doc_id, score in zip(result.doc_ids, result.scores):
            article = doc_id_to_article[doc_id]
            results.append({
                'doc_id': doc_id,
                'title': article['title'],
                'snippet': article['content'][:200] + '...' if len(article['content']) > 200 else article['content'],
                'score': round(score, 4),
                'url': article['url'],
                'date': article['published_date'],
                'category': article.get('category_name', '')
            })

        return jsonify({
            'query': query,
            'model': 'Language Model',
            'results': results,
            'total': len(result.doc_ids),
            'parameters': result.parameters,
            'execution_time': execution_time
        })
    except Exception as e:
        logger.error(f"LM search error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/search/hybrid', methods=['POST'])
def hybrid_search():
    """
    Perform Hybrid ranking search (combines multiple rankers).

    Request JSON:
        {
            "query": "深度學習",
            "limit": 10,
            "fusion_method": "rrf"  // optional: linear, rrf, combsum, combmnz
        }
    """
    if not hybrid_ranker:
        return jsonify({'error': 'System not initialized'}), 500

    data = request.get_json()
    query = data.get('query', '')
    limit = data.get('limit', 10)
    fusion_method = data.get('fusion_method', 'rrf')

    if not query:
        return jsonify({'error': 'Query is required'}), 400

    # Update fusion method if specified
    if fusion_method != hybrid_ranker.fusion_method:
        hybrid_ranker.fusion_method = fusion_method

    start_time = datetime.now()
    try:
        result = hybrid_ranker.search(query, topk=limit, ranker_topk=50)
        execution_time = (datetime.now() - start_time).total_seconds()

        # Prepare results
        results = []
        for doc_id, score in zip(result.doc_ids, result.scores):
            article = doc_id_to_article[doc_id]
            results.append({
                'doc_id': doc_id,
                'title': article['title'],
                'snippet': article['content'][:200] + '...' if len(article['content']) > 200 else article['content'],
                'score': round(score, 4),
                'url': article['url'],
                'date': article['published_date'],
                'category': article.get('category_name', '')
            })

        return jsonify({
            'query': query,
            'model': 'Hybrid',
            'results': results,
            'total': len(result.doc_ids),
            'fusion_method': result.fusion_method,
            'weights': result.weights,
            'component_scores': {
                name: [round(s, 4) for s in scores]
                for name, scores in result.component_scores.items()
            },
            'execution_time': execution_time
        })
    except Exception as e:
        logger.error(f"Hybrid search error: {e}")
        return jsonify({'error': str(e)}), 500


# ========================================
# Query Optimization APIs
# ========================================

@app.route('/api/search/wand', methods=['POST'])
def wand_search():
    """
    Perform WAND (Weak AND) optimized search.

    WAND uses query optimization to skip documents that cannot make it to
    the top-k results, providing significant speedup for large collections.

    Request JSON:
        {
            "query": "人工智慧深度學習",
            "limit": 10
        }

    Response JSON:
        {
            "query": "人工智慧深度學習",
            "results": [...],
            "total": 10,
            "statistics": {
                "num_scored_docs": 15,
                "num_candidate_docs": 98,
                "speedup_ratio": 6.53
            },
            "execution_time": 0.023
        }
    """
    if not wand_retrieval:
        return jsonify({'error': 'System not initialized'}), 500

    data = request.get_json()
    query = data.get('query', '')
    limit = data.get('limit', 10)

    if not query:
        return jsonify({'error': 'Query is required'}), 400

    start_time = datetime.now()
    try:
        # Tokenize query
        query_terms = chinese_tokenizer.tokenize(query)

        # Perform WAND search
        opt_result = wand_retrieval.search(query_terms, topk=limit)
        execution_time = (datetime.now() - start_time).total_seconds()

        # Prepare results
        results = []
        for doc_id, score in zip(opt_result.doc_ids, opt_result.scores):
            if doc_id in doc_id_to_article:
                article = doc_id_to_article[doc_id]
                results.append({
                    'doc_id': doc_id,
                    'title': article['title'],
                    'snippet': article['content'][:200] + '...' if len(article['content']) > 200 else article['content'],
                    'score': round(score, 4),
                    'url': article['url'],
                    'date': article['published_date'],
                    'category': article.get('category_name', '')
                })

        return jsonify({
            'query': query,
            'algorithm': opt_result.algorithm,
            'results': results,
            'total': opt_result.num_results,
            'statistics': {
                'num_scored_docs': opt_result.num_scored_docs,
                'num_candidate_docs': opt_result.num_candidate_docs,
                'speedup_ratio': round(opt_result.speedup_ratio, 2)
            },
            'execution_time': execution_time
        })
    except Exception as e:
        logger.error(f"WAND search error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/search/maxscore', methods=['POST'])
def maxscore_search():
    """
    Perform MaxScore optimized search.

    MaxScore partitions terms into essential and non-essential sets,
    allowing efficient top-k retrieval by scoring only essential documents.

    Request JSON:
        {
            "query": "台灣經濟發展趨勢",
            "limit": 10
        }

    Response JSON:
        {
            "query": "台灣經濟發展趨勢",
            "results": [...],
            "total": 10,
            "statistics": {
                "num_scored_docs": 22,
                "num_candidate_docs": 105,
                "speedup_ratio": 4.77
            },
            "execution_time": 0.018
        }
    """
    if not maxscore_retrieval:
        return jsonify({'error': 'System not initialized'}), 500

    data = request.get_json()
    query = data.get('query', '')
    limit = data.get('limit', 10)

    if not query:
        return jsonify({'error': 'Query is required'}), 400

    start_time = datetime.now()
    try:
        # Tokenize query
        query_terms = chinese_tokenizer.tokenize(query)

        # Perform MaxScore search
        opt_result = maxscore_retrieval.search(query_terms, topk=limit)
        execution_time = (datetime.now() - start_time).total_seconds()

        # Prepare results
        results = []
        for doc_id, score in zip(opt_result.doc_ids, opt_result.scores):
            if doc_id in doc_id_to_article:
                article = doc_id_to_article[doc_id]
                results.append({
                    'doc_id': doc_id,
                    'title': article['title'],
                    'snippet': article['content'][:200] + '...' if len(article['content']) > 200 else article['content'],
                    'score': round(score, 4),
                    'url': article['url'],
                    'date': article['published_date'],
                    'category': article.get('category_name', '')
                })

        return jsonify({
            'query': query,
            'algorithm': opt_result.algorithm,
            'results': results,
            'total': opt_result.num_results,
            'statistics': {
                'num_scored_docs': opt_result.num_scored_docs,
                'num_candidate_docs': opt_result.num_candidate_docs,
                'speedup_ratio': round(opt_result.speedup_ratio, 2)
            },
            'execution_time': execution_time
        })
    except Exception as e:
        logger.error(f"MaxScore search error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze/collocation', methods=['POST'])
def analyze_collocation():
    """
    Extract collocations from the corpus.

    Request JSON:
        {
            "measure": "pmi",  // pmi, llr, chi_square, t_score, dice
            "topk": 20
        }
    """
    if not collocation_extractor:
        return jsonify({'error': 'System not initialized'}), 500

    data = request.get_json()
    measure = data.get('measure', 'pmi')
    topk = data.get('topk', 20)

    try:
        collocations = collocation_extractor.extract_collocations(measure=measure, topk=topk)

        results = []
        for col in collocations:
            results.append({
                'bigram': ' '.join(col.bigram),
                'freq': col.freq,
                'pmi': round(col.pmi, 4),
                'llr': round(col.llr, 4),
                'chi_square': round(col.chi_square, 4),
                't_score': round(col.t_score, 4),
                'dice': round(col.dice, 4)
            })

        return jsonify({
            'measure': measure,
            'topk': topk,
            'collocations': results
        })
    except Exception as e:
        logger.error(f"Collocation analysis error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze/ngram', methods=['POST'])
def analyze_ngram():
    """
    Analyze n-gram probabilities.

    Request JSON:
        {
            "text": "資訊檢索系統",
            "calculate": "perplexity"  // perplexity or probability
        }
    """
    if not ngram_model:
        return jsonify({'error': 'System not initialized'}), 500

    data = request.get_json()
    text = data.get('text', '')
    calculate = data.get('calculate', 'perplexity')

    if not text:
        return jsonify({'error': 'Text is required'}), 400

    try:
        if calculate == 'perplexity':
            perplexity_value = ngram_model.perplexity(text)
            return jsonify({
                'text': text,
                'perplexity': round(perplexity_value, 4),
                'n': ngram_model.n,
                'smoothing': ngram_model.smoothing
            })
        elif calculate == 'probability':
            prob = ngram_model.sentence_probability(text)
            return jsonify({
                'text': text,
                'probability': prob,
                'log_probability': round(math.log(prob) if prob > 0 else float('-inf'), 4),
                'n': ngram_model.n,
                'smoothing': ngram_model.smoothing
            })
        else:
            return jsonify({'error': 'Invalid calculation type'}), 400
    except Exception as e:
        logger.error(f"N-gram analysis error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/extract/keywords', methods=['POST'])
def extract_keywords():
    """
    Extract keywords from text using multiple algorithms.

    Request JSON:
        {
            "text": "文本内容",
            "method": "textrank",  // textrank, yake, keybert, rake, ensemble
            "topk": 10,
            "use_pos_filter": true,
            "use_ner_boost": false
        }
    """
    data = request.get_json()
    text = data.get('text', '')
    method = data.get('method', 'textrank')
    topk = data.get('topk', 10)
    use_pos_filter = data.get('use_pos_filter', False)
    use_ner_boost = data.get('use_ner_boost', False)

    if not text:
        return jsonify({'error': 'Text is required'}), 400

    try:
        from src.ir.keyextract.textrank import TextRankExtractor
        from src.ir.keyextract.yake_extractor import YAKEExtractor

        # Determine POS filter
        pos_filter = ['N', 'V'] if use_pos_filter else None

        if method == 'textrank':
            extractor = TextRankExtractor(
                window_size=5,
                use_position_weight=True,
                use_ner_boost=use_ner_boost,
                pos_filter=pos_filter,
                tokenizer_engine='ckip' if chinese_tokenizer else 'jieba',
                device=0 if chinese_tokenizer else -1
            )
            keywords = extractor.extract(text, top_k=topk)
            results = [
                {
                    'keyword': kw.word,
                    'score': round(kw.score, 4),
                    'frequency': kw.frequency,
                    'positions': kw.positions[:5]  # First 5 positions
                }
                for kw in keywords
            ]

        elif method == 'yake':
            extractor = YAKEExtractor(
                language='zh',
                max_ngram_size=3,
                deduplication_threshold=0.9,
                num_keywords=topk
            )
            keywords = extractor.extract(text)
            results = [
                {
                    'keyword': kw.keyword,
                    'score': round(kw.score, 4),
                    'ngram_length': len(kw.keyword.split())
                }
                for kw in keywords
            ]

        elif method == 'keybert':
            try:
                from src.ir.keyextract.keybert_extractor import KeyBERTExtractor
                extractor = KeyBERTExtractor(
                    model_name='paraphrase-multilingual-MiniLM-L12-v2'
                )
                keywords = extractor.extract(text, top_k=topk)
                results = [
                    {
                        'keyword': kw.keyword,
                        'score': round(kw.score, 4)
                    }
                    for kw in keywords
                ]
            except ImportError:
                return jsonify({'error': 'KeyBERT requires sentence-transformers'}), 500

        elif method == 'rake':
            from src.ir.keyextract.rake_extractor import RAKEExtractor
            extractor = RAKEExtractor(
                language='chinese',
                min_phrase_length=1,
                max_phrase_length=4
            )
            keywords = extractor.extract(text, top_k=topk)
            results = [
                {
                    'keyword': kw.keyword,
                    'score': round(kw.score, 4),
                    'frequency': kw.frequency,
                    'degree': kw.degree
                }
                for kw in keywords
            ]

        else:
            return jsonify({'error': f'Unknown method: {method}'}), 400

        return jsonify({
            'method': method,
            'topk': topk,
            'keywords': results,
            'execution_time': 0  # Placeholder
        })

    except Exception as e:
        logger.error(f"Keyword extraction error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/extract/topics', methods=['POST'])
def extract_topics():
    """
    Extract topics from documents using LDA or BERTopic.

    Request JSON:
        {
            "documents": ["doc1", "doc2", ...],
            "method": "lda",  // lda or bertopic
            "n_topics": 5,
            "model_params": {}  // Optional model-specific parameters
        }
    """
    data = request.get_json()
    docs = data.get('documents', [])
    method = data.get('method', 'lda')
    n_topics = data.get('n_topics', 5)
    model_params = data.get('model_params', {})

    if not docs or len(docs) < 3:
        return jsonify({'error': 'At least 3 documents are required'}), 400

    try:
        if method == 'lda':
            from src.ir.topic.lda_model import LDAModel

            lda = LDAModel(
                n_topics=n_topics,
                iterations=model_params.get('iterations', 50),
                passes=model_params.get('passes', 10),
                tokenizer_engine='jieba'
            )
            lda.fit(docs)

            # Get topics
            topics = lda.get_topics()
            topic_info = lda.get_topic_info()

            # Transform documents
            doc_topics = lda.transform(docs, minimum_probability=0.01)

            # Evaluation metrics
            perplexity = lda.calculate_perplexity()
            coherence = lda.calculate_coherence('c_v')

            results = {
                'method': 'lda',
                'n_topics': n_topics,
                'topics': [
                    {
                        'topic_id': tid,
                        'words': [{'word': w, 'prob': round(p, 4)} for w, p in words[:10]]
                    }
                    for tid, words in topics.items()
                ],
                'topic_proportions': topic_info.to_dict('records'),
                'document_topics': [
                    {
                        'doc_index': i,
                        'topics': [(tid, round(prob, 4)) for tid, prob in dist]
                    }
                    for i, dist in enumerate(doc_topics)
                ],
                'metrics': {
                    'perplexity': round(perplexity, 4),
                    'coherence': round(coherence, 4)
                }
            }

            return jsonify(results)

        elif method == 'bertopic':
            from src.ir.topic.bertopic_model import BERTopicModel

            bertopic = BERTopicModel(
                n_topics=n_topics if n_topics > 0 else None,
                language='multilingual',
                calculate_probabilities=model_params.get('calculate_probabilities', False)
            )
            bertopic.fit(docs)

            # Get topics
            topics = bertopic.get_topics()
            topic_info = bertopic.get_topic_info()

            # Transform documents
            doc_topics, doc_probs = bertopic.transform(docs)

            results = {
                'method': 'bertopic',
                'n_topics': len(topics),
                'topics': [
                    {
                        'topic_id': tid,
                        'words': [{'word': w, 'score': round(s, 4)} for w, s in words[:10]]
                    }
                    for tid, words in topics.items() if tid != -1
                ],
                'topic_info': topic_info.to_dict('records'),
                'document_topics': [
                    {
                        'doc_index': i,
                        'topic_id': int(topic),
                        'probability': round(float(prob), 4) if doc_probs is not None else None
                    }
                    for i, (topic, prob) in enumerate(zip(doc_topics, doc_probs or [None] * len(doc_topics)))
                ]
            }

            return jsonify(results)

        else:
            return jsonify({'error': f'Unknown method: {method}'}), 400

    except Exception as e:
        logger.error(f"Topic modeling error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/extract/patterns', methods=['POST'])
def extract_patterns():
    """
    Extract frequent patterns using PAT-tree.

    Request JSON:
        {
            "texts": ["text1", "text2", ...],
            "min_pattern_length": 2,
            "max_pattern_length": 5,
            "min_frequency": 2,
            "topk": 20,
            "use_mi_score": true
        }
    """
    data = request.get_json()
    texts = data.get('texts', [])
    min_len = data.get('min_pattern_length', 2)
    max_len = data.get('max_pattern_length', 5)
    min_freq = data.get('min_frequency', 2)
    topk = data.get('topk', 20)
    use_mi = data.get('use_mi_score', True)

    if not texts:
        return jsonify({'error': 'Texts are required'}), 400

    try:
        from src.ir.patterns.pat_tree import PATTree

        # Initialize PAT-tree
        tree = PATTree(
            min_pattern_length=min_len,
            max_pattern_length=max_len,
            min_frequency=min_freq
        )

        # Insert all texts
        if chinese_tokenizer:
            tokenizer = chinese_tokenizer
        else:
            from src.ir.text.chinese_tokenizer import ChineseTokenizer
            tokenizer = ChineseTokenizer(engine='jieba')

        for text in texts:
            tree.insert_text(text, tokenizer)

        # Extract patterns
        patterns = tree.extract_patterns(top_k=topk, use_mi_score=use_mi)

        # Get statistics
        stats = tree.get_statistics()

        results = {
            'patterns': [
                {
                    'pattern': p.text,
                    'tokens': list(p.tokens),
                    'frequency': p.frequency,
                    'mi_score': round(p.mi_score, 4),
                    'positions': p.positions[:10]  # First 10 positions
                }
                for p in patterns
            ],
            'statistics': stats,
            'parameters': {
                'min_pattern_length': min_len,
                'max_pattern_length': max_len,
                'min_frequency': min_freq,
                'use_mi_score': use_mi
            }
        }

        return jsonify(results)

    except Exception as e:
        logger.error(f"Pattern extraction error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze/ner', methods=['POST'])
def analyze_ner():
    """
    Named Entity Recognition.

    Request JSON:
        {
            "text": "台積電在台灣新竹成立於1987年",
            "entity_types": ["PERSON", "ORG", "GPE", "LOC"]  // Optional
        }
    """
    data = request.get_json()
    text = data.get('text', '')
    entity_types = data.get('entity_types', None)

    if not text:
        return jsonify({'error': 'Text is required'}), 400

    try:
        from src.ir.text.ner_extractor import NERExtractor

        ner = NERExtractor(
            entity_types=set(entity_types) if entity_types else None,
            device=0 if chinese_tokenizer else -1
        )

        entities = ner.extract(text)

        results = {
            'text': text,
            'entities': [
                {
                    'text': e.text,
                    'type': e.type,
                    'start': e.start,
                    'end': e.end,
                    'confidence': round(e.confidence, 4) if e.confidence else None
                }
                for e in entities
            ],
            'entity_count': len(entities),
            'entity_types': list(set(e.type for e in entities))
        }

        # Group by type
        by_type = {}
        for e in entities:
            if e.type not in by_type:
                by_type[e.type] = []
            by_type[e.type].append(e.text)

        results['entities_by_type'] = by_type

        return jsonify(results)

    except Exception as e:
        logger.error(f"NER error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/analyze/syntax', methods=['POST'])
def analyze_syntax():
    """
    Syntactic parsing and SVO extraction.

    Request JSON:
        {
            "text": "台積電在台灣生產晶片",
            "analysis_type": "svo"  // svo or dependencies
        }
    """
    data = request.get_json()
    text = data.get('text', '')
    analysis_type = data.get('analysis_type', 'svo')

    if not text:
        return jsonify({'error': 'Text is required'}), 400

    try:
        from src.ir.syntax.parser import SyntaxAnalyzer

        analyzer = SyntaxAnalyzer(device=0 if chinese_tokenizer else -1)

        if analysis_type == 'svo':
            # Extract SVO triples
            triples = analyzer.extract_svo(text)

            results = {
                'text': text,
                'analysis_type': 'svo',
                'triples': [
                    {
                        'subject': t.subject,
                        'verb': t.verb,
                        'object': t.object,
                        'confidence': round(t.confidence, 4) if t.confidence else None
                    }
                    for t in triples
                ],
                'triple_count': len(triples)
            }

        elif analysis_type == 'dependencies':
            # Parse dependencies
            deps = analyzer.parse(text)

            results = {
                'text': text,
                'analysis_type': 'dependencies',
                'dependencies': [
                    {
                        'head': d.head,
                        'relation': d.relation,
                        'dependent': d.dependent,
                        'head_pos': d.head_pos,
                        'dep_pos': d.dep_pos
                    }
                    for d in deps
                ],
                'dependency_count': len(deps)
            }

        else:
            return jsonify({'error': f'Unknown analysis type: {analysis_type}'}), 400

        return jsonify(results)

    except Exception as e:
        logger.error(f"Syntax analysis error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/document/<int:doc_id>/analysis', methods=['GET'])
def get_document_analysis(doc_id):
    """
    Get comprehensive linguistic analysis for a document.

    Returns keywords, entities, topics, and syntax analysis.
    """
    if doc_id not in doc_id_to_article:
        return jsonify({'error': 'Document not found'}), 404

    article = doc_id_to_article[doc_id]
    text = article.get('content', '')

    if not text:
        return jsonify({'error': 'Document has no content'}), 400

    try:
        from src.ir.keyextract.textrank import TextRankExtractor
        from src.ir.text.ner_extractor import NERExtractor

        results = {
            'doc_id': doc_id,
            'title': article.get('title', ''),
            'analysis': {}
        }

        # Keywords (TextRank)
        try:
            keyword_extractor = TextRankExtractor(
                window_size=5,
                use_position_weight=True,
                tokenizer_engine='jieba',
                device=-1
            )
            keywords = keyword_extractor.extract(text, top_k=10)
            results['analysis']['keywords'] = [
                {'word': kw.word, 'score': round(kw.score, 4)}
                for kw in keywords
            ]
        except Exception as e:
            logger.warning(f"Keyword extraction failed: {e}")
            results['analysis']['keywords'] = []

        # Named Entities
        try:
            ner = NERExtractor(device=-1)
            entities = ner.extract(text)
            results['analysis']['entities'] = [
                {'text': e.text, 'type': e.type}
                for e in entities
            ]
        except Exception as e:
            logger.warning(f"NER failed: {e}")
            results['analysis']['entities'] = []

        # Use cached linguistic data if available
        if doc_id in doc_linguistic_data:
            results['analysis']['linguistic'] = doc_linguistic_data[doc_id]

        return jsonify(results)

    except Exception as e:
        logger.error(f"Document analysis error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/recommend/similar', methods=['POST'])
def recommend_similar():
    """
    Recommend similar documents using content-based filtering.

    Request JSON:
        {
            "doc_id": 5,
            "topk": 10,
            "use_embeddings": false,
            "apply_diversity": true
        }
    """
    data = request.get_json()
    doc_id = data.get('doc_id')
    topk = data.get('topk', 10)
    use_embeddings = data.get('use_embeddings', False)
    apply_diversity = data.get('apply_diversity', True)

    if doc_id is None or doc_id < 0 or doc_id >= len(documents):
        return jsonify({'error': 'Invalid doc_id'}), 400

    try:
        from src.ir.recommendation import ContentBasedRecommender

        # Initialize content-based recommender
        content_rec = ContentBasedRecommender(
            documents=documents,
            similarity_metric='cosine',
            diversity_lambda=0.3
        )

        # Build vectors
        if use_embeddings and bert_retrieval:
            content_rec.build_bert_embeddings(bert_retrieval)
        elif vsm:
            content_rec.build_tfidf_vectors(vsm)
        else:
            return jsonify({'error': 'No feature vectors available'}), 500

        # Get recommendations
        recs = content_rec.recommend_similar(
            doc_id=doc_id,
            top_k=topk,
            exclude_self=True,
            use_embeddings=use_embeddings,
            apply_diversity=apply_diversity
        )

        # Build response
        results = []
        for rec in recs:
            article = doc_id_to_article.get(rec.doc_id, {})
            results.append({
                'doc_id': rec.doc_id,
                'title': article.get('title', ''),
                'score': round(rec.score, 4),
                'reason': rec.reason,
                'features': rec.features,
                'snippet': article.get('content', '')[:200] + '...',
                'category': article.get('category', ''),
                'date': article.get('published_date', '')
            })

        return jsonify({
            'query_doc_id': doc_id,
            'query_title': doc_id_to_article.get(doc_id, {}).get('title', ''),
            'topk': topk,
            'method': 'content-based',
            'use_embeddings': use_embeddings,
            'diversity': apply_diversity,
            'recommendations': results
        })

    except Exception as e:
        logger.error(f"Similar recommendation error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/recommend/personalized', methods=['POST'])
def recommend_personalized():
    """
    Recommend documents based on user reading history.

    Request JSON:
        {
            "reading_history": [0, 5, 12, 23],
            "topk": 10,
            "use_embeddings": false,
            "apply_diversity": true
        }
    """
    data = request.get_json()
    history = data.get('reading_history', [])
    topk = data.get('topk', 10)
    use_embeddings = data.get('use_embeddings', False)
    apply_diversity = data.get('apply_diversity', True)

    if not history or len(history) == 0:
        return jsonify({'error': 'Reading history is required'}), 400

    # Validate history
    if any(doc_id < 0 or doc_id >= len(documents) for doc_id in history):
        return jsonify({'error': 'Invalid doc_id in reading history'}), 400

    try:
        from src.ir.recommendation import ContentBasedRecommender

        # Initialize recommender
        content_rec = ContentBasedRecommender(
            documents=documents,
            similarity_metric='cosine',
            diversity_lambda=0.3
        )

        # Build vectors
        if use_embeddings and bert_retrieval:
            content_rec.build_bert_embeddings(bert_retrieval)
        elif vsm:
            content_rec.build_tfidf_vectors(vsm)
        else:
            return jsonify({'error': 'No feature vectors available'}), 500

        # Get recommendations
        recs = content_rec.recommend_personalized(
            reading_history=history,
            top_k=topk,
            use_embeddings=use_embeddings,
            apply_diversity=apply_diversity
        )

        # Build response
        results = []
        for rec in recs:
            article = doc_id_to_article.get(rec.doc_id, {})
            results.append({
                'doc_id': rec.doc_id,
                'title': article.get('title', ''),
                'score': round(rec.score, 4),
                'reason': rec.reason,
                'snippet': article.get('content', '')[:200] + '...',
                'category': article.get('category', ''),
                'date': article.get('published_date', '')
            })

        # Include reading history info
        history_titles = [
            doc_id_to_article.get(doc_id, {}).get('title', f'Doc {doc_id}')
            for doc_id in history
        ]

        return jsonify({
            'reading_history': history,
            'history_titles': history_titles,
            'topk': topk,
            'method': 'personalized-content',
            'use_embeddings': use_embeddings,
            'diversity': apply_diversity,
            'recommendations': results
        })

    except Exception as e:
        logger.error(f"Personalized recommendation error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/recommend/trending', methods=['GET'])
def recommend_trending():
    """
    Recommend trending/popular documents.

    Query params:
        - topk: Number of recommendations (default: 10)
        - time_window: Time window in days (default: 7)
        - category: Filter by category (optional)
    """
    topk = int(request.args.get('topk', 10))
    time_window = int(request.args.get('time_window', 7))
    category = request.args.get('category', None)

    try:
        from datetime import datetime, timedelta

        # Get recent documents
        cutoff_date = (datetime.now() - timedelta(days=time_window)).strftime('%Y-%m-%d')

        # Filter documents
        recent_docs = []
        for i, doc in enumerate(documents):
            doc_date = doc.get('published_date', '')
            doc_category = doc.get('category', '')

            # Date filter
            if doc_date >= cutoff_date:
                # Category filter (if specified)
                if category is None or doc_category == category:
                    recent_docs.append({
                        'doc_id': i,
                        'date': doc_date,
                        'category': doc_category,
                        'title': doc.get('title', ''),
                        'content': doc.get('content', '')
                    })

        # Sort by date (most recent first)
        recent_docs.sort(key=lambda x: x['date'], reverse=True)

        # Take top-k
        trending = recent_docs[:topk]

        results = [
            {
                'doc_id': doc['doc_id'],
                'title': doc['title'],
                'category': doc['category'],
                'date': doc['date'],
                'snippet': doc['content'][:200] + '...',
                'score': 1.0,  # Placeholder
                'reason': f"Recent article from {doc['date']}"
            }
            for doc in trending
        ]

        return jsonify({
            'topk': topk,
            'time_window_days': time_window,
            'category': category,
            'method': 'trending',
            'total_recent': len(recent_docs),
            'recommendations': results
        })

    except Exception as e:
        logger.error(f"Trending recommendation error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# ========================================
# Collaborative Filtering APIs
# ========================================

@app.route('/api/recommend/cf/user-based', methods=['POST'])
def recommend_cf_user_based():
    """
    User-based Collaborative Filtering recommendations.

    POST /api/recommend/cf/user-based
    {
        "user_id": 0,
        "top_k": 10,
        "n_neighbors": 20,
        "similarity_metric": "cosine"  // Optional: cosine, pearson
    }

    Returns:
        {
            "user_id": 0,
            "method": "user_based_cf",
            "recommendations": [
                {
                    "doc_id": 123,
                    "score": 0.85,
                    "title": "...",
                    "reason": "Users similar to you liked this"
                }
            ],
            "n_neighbors_found": 15,
            "computation_time": 0.023
        }
    """
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        top_k = data.get('top_k', 10)
        n_neighbors = data.get('n_neighbors', 20)
        similarity_metric = data.get('similarity_metric', 'cosine')

        if user_id is None:
            return jsonify({"error": "user_id is required"}), 400

        # Import CF recommender
        from src.ir.recommendation import CollaborativeFilteringRecommender

        start_time = time.time()

        # Initialize CF recommender with user-item interaction data
        # For now, we'll create a mock CF recommender
        # In production, this should load from persistent storage
        n_users = max(user_id + 1, 100)  # Ensure we have enough users
        n_items = len(documents)

        cf_rec = CollaborativeFilteringRecommender(
            n_users=n_users,
            n_items=n_items,
            implicit_feedback=True
        )

        # Load interaction data (mock data for demo)
        # In production: cf_rec.load_interactions(interaction_file_path)
        # For now, generate some random interactions for demonstration
        import numpy as np
        np.random.seed(42)
        for uid in range(min(n_users, 50)):
            n_interactions = np.random.randint(5, 20)
            item_ids = np.random.choice(n_items, size=n_interactions, replace=False)
            for item_id in item_ids:
                cf_rec.add_interaction(uid, item_id, weight=1.0)

        # Compute user similarity if not already done
        cf_rec.compute_user_similarity(similarity_metric=similarity_metric, top_k=n_neighbors)

        # Get recommendations
        recs = cf_rec.recommend_user_based(
            user_id=user_id,
            top_k=top_k,
            n_neighbors=n_neighbors
        )

        computation_time = time.time() - start_time

        # Format results with document metadata
        recommendations = []
        for rec in recs:
            doc_id = rec.item_id
            if 0 <= doc_id < len(documents):
                doc = documents[doc_id]
                recommendations.append({
                    "doc_id": doc_id,
                    "score": round(rec.score, 4),
                    "title": doc.get('title', f'Document {doc_id}'),
                    "category": doc.get('category', 'unknown'),
                    "reason": "Users similar to you liked this"
                })

        return jsonify({
            "user_id": user_id,
            "method": "user_based_cf",
            "recommendations": recommendations,
            "n_neighbors_found": len(recommendations),
            "parameters": {
                "top_k": top_k,
                "n_neighbors": n_neighbors,
                "similarity_metric": similarity_metric
            },
            "computation_time": round(computation_time, 3)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/recommend/cf/item-based', methods=['POST'])
def recommend_cf_item_based():
    """
    Item-based Collaborative Filtering recommendations.

    POST /api/recommend/cf/item-based
    {
        "user_id": 0,
        "top_k": 10,
        "n_neighbors": 50,
        "similarity_metric": "cosine"  // Optional: cosine, adjusted_cosine, jaccard
    }

    Returns:
        {
            "user_id": 0,
            "method": "item_based_cf",
            "recommendations": [
                {
                    "doc_id": 456,
                    "score": 0.92,
                    "title": "...",
                    "reason": "Similar to items you liked"
                }
            ],
            "computation_time": 0.018
        }
    """
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        top_k = data.get('top_k', 10)
        n_neighbors = data.get('n_neighbors', 50)
        similarity_metric = data.get('similarity_metric', 'cosine')

        if user_id is None:
            return jsonify({"error": "user_id is required"}), 400

        from src.ir.recommendation import CollaborativeFilteringRecommender

        start_time = time.time()

        # Initialize CF recommender
        n_users = max(user_id + 1, 100)
        n_items = len(documents)

        cf_rec = CollaborativeFilteringRecommender(
            n_users=n_users,
            n_items=n_items,
            implicit_feedback=True
        )

        # Load interaction data (mock for demo)
        import numpy as np
        np.random.seed(42)
        for uid in range(min(n_users, 50)):
            n_interactions = np.random.randint(5, 20)
            item_ids = np.random.choice(n_items, size=n_interactions, replace=False)
            for item_id in item_ids:
                cf_rec.add_interaction(uid, item_id, weight=1.0)

        # Compute item similarity
        cf_rec.compute_item_similarity(similarity_metric=similarity_metric, top_k=n_neighbors)

        # Get recommendations
        recs = cf_rec.recommend_item_based(
            user_id=user_id,
            top_k=top_k,
            n_neighbors=n_neighbors
        )

        computation_time = time.time() - start_time

        # Format results
        recommendations = []
        for rec in recs:
            doc_id = rec.item_id
            if 0 <= doc_id < len(documents):
                doc = documents[doc_id]
                recommendations.append({
                    "doc_id": doc_id,
                    "score": round(rec.score, 4),
                    "title": doc.get('title', f'Document {doc_id}'),
                    "category": doc.get('category', 'unknown'),
                    "reason": "Similar to items you liked"
                })

        return jsonify({
            "user_id": user_id,
            "method": "item_based_cf",
            "recommendations": recommendations,
            "parameters": {
                "top_k": top_k,
                "n_neighbors": n_neighbors,
                "similarity_metric": similarity_metric
            },
            "computation_time": round(computation_time, 3)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/recommend/cf/matrix-factorization', methods=['POST'])
def recommend_cf_mf():
    """
    Matrix Factorization based recommendations (SVD or ALS).

    POST /api/recommend/cf/matrix-factorization
    {
        "user_id": 0,
        "top_k": 10,
        "n_factors": 50,
        "method": "svd"  // or "als"
    }

    Returns:
        {
            "user_id": 0,
            "method": "matrix_factorization_svd",
            "recommendations": [
                {
                    "doc_id": 789,
                    "score": 0.88,
                    "title": "...",
                    "reason": "Predicted based on latent factors"
                }
            ],
            "n_factors": 50,
            "computation_time": 0.156
        }
    """
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        top_k = data.get('top_k', 10)
        n_factors = data.get('n_factors', 50)
        method = data.get('method', 'svd')

        if user_id is None:
            return jsonify({"error": "user_id is required"}), 400

        if method not in ['svd', 'als']:
            return jsonify({"error": "method must be 'svd' or 'als'"}), 400

        from src.ir.recommendation import CollaborativeFilteringRecommender

        start_time = time.time()

        # Initialize CF recommender
        n_users = max(user_id + 1, 100)
        n_items = len(documents)

        cf_rec = CollaborativeFilteringRecommender(
            n_users=n_users,
            n_items=n_items,
            implicit_feedback=True
        )

        # Load interaction data (mock for demo)
        import numpy as np
        np.random.seed(42)
        for uid in range(min(n_users, 50)):
            n_interactions = np.random.randint(5, 20)
            item_ids = np.random.choice(n_items, size=n_interactions, replace=False)
            for item_id in item_ids:
                cf_rec.add_interaction(uid, item_id, weight=1.0)

        # Train matrix factorization
        cf_rec.train_matrix_factorization(n_factors=n_factors, method=method)

        # Get recommendations
        recs = cf_rec.recommend_mf(user_id=user_id, top_k=top_k)

        computation_time = time.time() - start_time

        # Format results
        recommendations = []
        for rec in recs:
            doc_id = rec.item_id
            if 0 <= doc_id < len(documents):
                doc = documents[doc_id]
                recommendations.append({
                    "doc_id": doc_id,
                    "score": round(rec.score, 4),
                    "title": doc.get('title', f'Document {doc_id}'),
                    "category": doc.get('category', 'unknown'),
                    "reason": "Predicted based on latent factors"
                })

        return jsonify({
            "user_id": user_id,
            "method": f"matrix_factorization_{method}",
            "recommendations": recommendations,
            "parameters": {
                "top_k": top_k,
                "n_factors": n_factors,
                "method": method
            },
            "computation_time": round(computation_time, 3)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ========================================
# Hybrid Recommendation APIs
# ========================================

@app.route('/api/recommend/hybrid', methods=['POST'])
def recommend_hybrid():
    """
    Hybrid recommendation combining content-based and collaborative filtering.

    POST /api/recommend/hybrid
    {
        "user_id": 0,
        "doc_id": 5,  // Optional: current document context
        "top_k": 10,
        "fusion_method": "weighted",  // weighted, cascade, or switching
        "content_weight": 0.5,  // For weighted fusion
        "cf_weight": 0.4,
        "popularity_weight": 0.1,
        "use_embeddings": false  // Use BERT embeddings for content
    }

    Returns:
        {
            "user_id": 0,
            "method": "hybrid_weighted",
            "recommendations": [
                {
                    "doc_id": 123,
                    "score": 0.87,
                    "title": "...",
                    "content_score": 0.85,
                    "cf_score": 0.92,
                    "popularity_score": 0.78,
                    "reason": "Combined content similarity and collaborative filtering"
                }
            ],
            "fusion_method": "weighted",
            "computation_time": 0.045
        }
    """
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        doc_id = data.get('doc_id')  # Optional context
        top_k = data.get('top_k', 10)
        fusion_method = data.get('fusion_method', 'weighted')
        content_weight = data.get('content_weight', 0.5)
        cf_weight = data.get('cf_weight', 0.4)
        popularity_weight = data.get('popularity_weight', 0.1)
        use_embeddings = data.get('use_embeddings', False)

        if user_id is None:
            return jsonify({"error": "user_id is required"}), 400

        if fusion_method not in ['weighted', 'cascade', 'switching']:
            return jsonify({"error": "fusion_method must be 'weighted', 'cascade', or 'switching'"}), 400

        from src.ir.recommendation import (
            ContentBasedRecommender,
            CollaborativeFilteringRecommender,
            HybridRecommender
        )

        start_time = time.time()

        # Initialize content-based recommender
        content_rec = ContentBasedRecommender(documents, similarity_metric='cosine')

        # Build feature vectors
        if use_embeddings and bert_retrieval:
            content_rec.build_bert_embeddings(bert_retrieval)
        elif vsm:
            content_rec.build_tfidf_vectors(vsm)
        else:
            return jsonify({"error": "No feature vectors available (VSM or BERT)"}), 500

        # Initialize CF recommender
        n_users = max(user_id + 1, 100)
        n_items = len(documents)

        cf_rec = CollaborativeFilteringRecommender(
            n_users=n_users,
            n_items=n_items,
            implicit_feedback=True
        )

        # Load interaction data (mock for demo)
        import numpy as np
        np.random.seed(42)
        for uid in range(min(n_users, 50)):
            n_interactions = np.random.randint(5, 20)
            item_ids = np.random.choice(n_items, size=n_interactions, replace=False)
            for item_id in item_ids:
                cf_rec.add_interaction(uid, item_id, weight=1.0)

        # Train CF models
        cf_rec.compute_item_similarity(similarity_metric='cosine', top_k=50)
        cf_rec.train_matrix_factorization(n_factors=50, method='svd')

        # Initialize hybrid recommender
        hybrid_rec = HybridRecommender(
            content_recommender=content_rec,
            cf_recommender=cf_rec,
            fusion_method=fusion_method,
            content_weight=content_weight,
            cf_weight=cf_weight,
            popularity_weight=popularity_weight
        )

        # Get recommendations based on fusion method
        if fusion_method == 'weighted':
            recs = hybrid_rec.recommend_weighted(
                user_id=user_id,
                doc_id=doc_id,
                top_k=top_k
            )
        elif fusion_method == 'cascade':
            recs = hybrid_rec.recommend_cascade(
                user_id=user_id,
                doc_id=doc_id,
                top_k=top_k
            )
        else:  # switching
            recs = hybrid_rec.recommend_switching(
                user_id=user_id,
                doc_id=doc_id,
                top_k=top_k
            )

        computation_time = time.time() - start_time

        # Format results
        recommendations = []
        for rec in recs:
            doc_id_rec = rec.item_id
            if 0 <= doc_id_rec < len(documents):
                doc = documents[doc_id_rec]
                rec_item = {
                    "doc_id": doc_id_rec,
                    "score": round(rec.score, 4),
                    "title": doc.get('title', f'Document {doc_id_rec}'),
                    "category": doc.get('category', 'unknown'),
                    "reason": rec.reason
                }

                # Add component scores if available
                if hasattr(rec, 'content_score'):
                    rec_item['content_score'] = round(rec.content_score, 4)
                if hasattr(rec, 'cf_score'):
                    rec_item['cf_score'] = round(rec.cf_score, 4)
                if hasattr(rec, 'popularity_score'):
                    rec_item['popularity_score'] = round(rec.popularity_score, 4)

                recommendations.append(rec_item)

        return jsonify({
            "user_id": user_id,
            "method": f"hybrid_{fusion_method}",
            "recommendations": recommendations,
            "fusion_method": fusion_method,
            "parameters": {
                "top_k": top_k,
                "content_weight": content_weight,
                "cf_weight": cf_weight,
                "popularity_weight": popularity_weight,
                "use_embeddings": use_embeddings
            },
            "computation_time": round(computation_time, 3)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ========================================
# User Interaction Tracking APIs
# ========================================

# Global interaction storage (in production, use database)
user_interactions = []

@app.route('/api/interaction/record', methods=['POST'])
def record_interaction():
    """
    Record user interaction with a document.

    POST /api/interaction/record
    {
        "user_id": 0,
        "doc_id": 123,
        "interaction_type": "click",  // click, read, like, share
        "duration": 45.5,  // Optional: time spent in seconds
        "timestamp": "2025-01-14T10:30:00"  // Optional: ISO format
    }

    Returns:
        {
            "status": "success",
            "interaction_id": 42,
            "message": "Interaction recorded"
        }
    """
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        doc_id = data.get('doc_id')
        interaction_type = data.get('interaction_type', 'click')
        duration = data.get('duration', 0)
        timestamp = data.get('timestamp')

        if user_id is None or doc_id is None:
            return jsonify({"error": "user_id and doc_id are required"}), 400

        if doc_id < 0 or doc_id >= len(documents):
            return jsonify({"error": f"Invalid doc_id: {doc_id}"}), 400

        # Create interaction record
        from datetime import datetime
        interaction = {
            "interaction_id": len(user_interactions),
            "user_id": user_id,
            "doc_id": doc_id,
            "interaction_type": interaction_type,
            "duration": duration,
            "timestamp": timestamp or datetime.now().isoformat()
        }

        user_interactions.append(interaction)

        return jsonify({
            "status": "success",
            "interaction_id": interaction["interaction_id"],
            "message": "Interaction recorded"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/interaction/history', methods=['GET'])
def get_interaction_history():
    """
    Get user interaction history.

    GET /api/interaction/history?user_id=0&limit=50

    Returns:
        {
            "user_id": 0,
            "interactions": [
                {
                    "interaction_id": 42,
                    "doc_id": 123,
                    "interaction_type": "read",
                    "duration": 45.5,
                    "timestamp": "2025-01-14T10:30:00"
                }
            ],
            "total": 142
        }
    """
    try:
        user_id = request.args.get('user_id', type=int)
        limit = request.args.get('limit', 50, type=int)

        if user_id is None:
            return jsonify({"error": "user_id is required"}), 400

        # Filter interactions for this user
        user_history = [
            interaction for interaction in user_interactions
            if interaction['user_id'] == user_id
        ]

        # Sort by timestamp (most recent first)
        user_history.sort(key=lambda x: x['timestamp'], reverse=True)

        # Limit results
        limited_history = user_history[:limit]

        return jsonify({
            "user_id": user_id,
            "interactions": limited_history,
            "total": len(user_history),
            "returned": len(limited_history)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Initialize the IR system before starting the server
    initialize_ir_system()

    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5001)
