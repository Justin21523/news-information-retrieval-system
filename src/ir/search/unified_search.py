"""
Unified Search Engine

This module provides a unified interface for all retrieval methods,
integrating Boolean retrieval, Vector Space Model, BM25, and field-based search.

Key Features:
    - Multiple retrieval models (Boolean, VSM, BM25)
    - Field-specific queries (title:xxx, category:xxx)
    - CKIP tokenization for Chinese text
    - Incremental index support
    - PostgreSQL and JSONL data sources
    - Hybrid ranking with score combination

Supported Query Types:
    - Simple queries: "台灣 政治"
    - Field queries: "title:台灣 AND category:政治"
    - Boolean queries: "(title:台灣 OR title:中國) AND NOT category:sports"
    - Range queries: "date:[2025-11-01 TO 2025-11-13]"

Author: Information Retrieval System
Date: 2025-11-20
"""

import logging
from typing import List, Dict, Optional, Set, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
from enum import Enum

from ..text.ckip_tokenizer_optimized import get_optimized_tokenizer
from ..index.incremental_builder import IncrementalIndexBuilder
from ..index.field_indexer import FieldIndexer
from ..retrieval.vsm import VectorSpaceModel
from ..retrieval.bm25 import BM25Ranker
from ..retrieval.boolean import BooleanRetrieval
from ..query.query_parser import QueryParser, QueryNode, Operator
from ..query.query_executor import QueryExecutor, SearchResult
from ..index.doc_reader import DocumentReader, NewsDocument


class QueryMode(Enum):
    """Query processing modes."""
    AUTO = "auto"           # Auto-detect query type
    SIMPLE = "simple"       # Simple keyword query (VSM/BM25)
    BOOLEAN = "boolean"     # Boolean query (AND/OR/NOT)
    FIELD = "field"         # Field-specific query
    STRUCTURED = "struct"   # Structured JSON query


class RankingModel(Enum):
    """Ranking models for retrieval."""
    VSM = "vsm"           # Vector Space Model (TF-IDF + cosine)
    BM25 = "bm25"         # BM25 ranking
    BOOLEAN = "boolean"   # Boolean retrieval (no ranking)
    HYBRID = "hybrid"     # Hybrid: combine multiple models


@dataclass
class UnifiedSearchResult:
    """
    Unified search result with ranking and metadata.

    Attributes:
        doc_id: Document ID
        score: Relevance score
        rank: Rank position (1-indexed)
        title: Document title
        content: Document content (snippet)
        source: News source
        category: Article category
        published_at: Publication date
        url: Article URL
        matched_fields: Fields that matched the query
        ranking_model: Model used for ranking
    """
    doc_id: int
    score: float
    rank: int
    title: str = ""
    content: str = ""
    source: str = ""
    category: str = ""
    published_at: str = ""
    url: str = ""
    matched_fields: List[str] = None
    ranking_model: str = "unknown"

    def __post_init__(self):
        if self.matched_fields is None:
            self.matched_fields = []


class UnifiedSearchEngine:
    """
    Unified search engine integrating all retrieval methods.

    This class provides a single interface for multiple retrieval models,
    supporting various query types and ranking strategies.

    Complexity:
        - Indexing: O(T) where T is total tokens
        - Search: Varies by model (VSM: O(V + k*log k), BM25: O(V + k*log k))

    Attributes:
        index_builder: Incremental index builder with CKIP
        field_indexer: Field-based indexer for metadata
        vsm_model: Vector Space Model retrieval
        bm25_model: BM25 retrieval
        boolean_model: Boolean retrieval
        query_parser: Query string parser
        doc_reader: Document reader for corpus
    """

    def __init__(self,
                 index_dir: str = "data/index",
                 ckip_model: str = "bert-base"):
        """
        Initialize Unified Search Engine.

        Args:
            index_dir: Directory for index storage
            ckip_model: CKIP model variant

        Complexity:
            Time: O(1) + model loading time
            Space: O(1)
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing UnifiedSearchEngine...")

        # Initialize CKIP tokenizer (singleton) with 32 threads optimization
        self.ckip_tokenizer = get_optimized_tokenizer(
            model_name=ckip_model,
            use_gpu=False,
            num_threads=32  # Use all 32 threads for maximum performance
        )

        # Tokenizer wrapper for compatibility
        self._tokenizer = self._ckip_tokenizer

        # Initialize index components
        self.index_builder = IncrementalIndexBuilder(
            index_dir=index_dir,
            ckip_model=ckip_model
        )

        # Initialize field indexer with CKIP
        self.field_indexer = FieldIndexer(tokenizer=self._tokenizer)

        # Initialize retrieval models (will be built when index is ready)
        self.vsm_model: Optional[VectorSpaceModel] = None
        self.bm25_model: Optional[BM25Ranker] = None
        self.boolean_model: Optional[BooleanRetrieval] = None

        # Query processing
        self.query_parser = QueryParser()
        self.query_executor: Optional[QueryExecutor] = None

        # Document reader
        self.doc_reader = DocumentReader()

        # Metadata storage (doc_id -> metadata)
        self.doc_metadata: Dict[int, Dict[str, Any]] = {}

        # Index state
        self.is_indexed = False

        self.logger.info("UnifiedSearchEngine initialized")

    def _ckip_tokenizer(self, text: str) -> List[str]:
        """
        Tokenize text using CKIP Transformers.

        Args:
            text: Input text

        Returns:
            List of tokens

        Complexity:
            Time: O(n) where n is text length
        """
        return self.ckip_tokenizer.tokenize(
            text,
            filter_stopwords=True,
            min_length=2
        )

    def _process_document_batch(self, doc_buffer: List, field_docs: List) -> None:
        """
        Process a batch of documents using optimized batch tokenization.

        Args:
            doc_buffer: List of NewsDocument objects to process
            field_docs: List to append field index entries to

        Complexity:
            Time: O(B * T / batch_size) where B is buffer size, T is avg tokens
            Space: O(B * T) for batch processing buffer
        """
        # DEBUG: Log batch processing invocation
        self.logger.info(f"🔵 DEBUG: _process_document_batch called with {len(doc_buffer)} documents")

        # Batch process documents
        results = self.index_builder.add_documents_batch(
            doc_buffer,
            ckip_batch_size=512  # CKIP internal batch size
        )

        # DEBUG: Log batch processing completion
        self.logger.info(f"✅ DEBUG: _process_document_batch completed, got {len(results)} results")

        # Track successful documents and update metadata
        for i, (success, message) in enumerate(results):
            if success:
                indexed_doc = doc_buffer[i]
                doc_id = self.index_builder.docs_indexed - len([r for r in results if r[0]]) + i

                # Store metadata AND content for retrieval
                self.doc_metadata[doc_id] = {
                    'title': indexed_doc.title,
                    'content': indexed_doc.content,  # Store full content for snippet extraction
                    'source': indexed_doc.source,
                    'category': indexed_doc.category,
                    'published_at': indexed_doc.published_at,
                    'url': indexed_doc.url,
                    'author': indexed_doc.author
                }

                # Prepare for field indexing
                field_docs.append({
                    'doc_id': doc_id,
                    'title': indexed_doc.title,
                    'content': indexed_doc.content,
                    'source': indexed_doc.source,
                    'category': indexed_doc.category,
                    'published_date': indexed_doc.published_at,
                    'author': indexed_doc.author or '',
                    'url': indexed_doc.url or ''
                })

    def build_index_from_jsonl(self,
                               data_dir: str,
                               pattern: str = "*.jsonl",
                               limit: Optional[int] = None,
                               doc_batch_size: int = 100) -> Dict[str, Any]:
        """
        Build search index from JSONL files with batch processing optimization.

        Args:
            data_dir: Directory containing JSONL files
            pattern: File pattern to match
            limit: Maximum documents to index (None for all)
            doc_batch_size: Number of documents to process in each batch (default: 100)

        Returns:
            Build statistics dictionary

        Complexity:
            Time: O(N * T_avg / B) where N is documents, T_avg is avg tokens, B is batch size
            Space: O(V + P + B) where V is vocabulary, P is postings, B is batch buffer

        Performance:
            Batch processing reduces CKIP tokenization calls by ~100x and provides
            ~5-6x speedup for large document sets.

        Examples:
            >>> engine = UnifiedSearchEngine()
            >>> stats = engine.build_index_from_jsonl("data/raw", limit=10000)
            >>> print(f"Indexed {stats['total_docs']} documents")

            >>> # Use larger batches for maximum throughput
            >>> stats = engine.build_index_from_jsonl("data/raw", doc_batch_size=200)
        """
        self.logger.info(f"Building index from {data_dir} (batch_size={doc_batch_size})...")

        # Read and index documents with batch processing
        field_docs = []
        doc_buffer = []
        processed_count = 0

        for doc in self.doc_reader.read_directory(
            directory=data_dir,
            pattern=pattern,
            total_limit=limit
        ):
            doc_buffer.append(doc)

            # Process batch when buffer is full
            if len(doc_buffer) >= doc_batch_size:
                self.logger.info(f"🟡 DEBUG: Buffer full ({len(doc_buffer)}/{doc_batch_size}), calling _process_document_batch...")
                self._process_document_batch(doc_buffer, field_docs)
                processed_count += len(doc_buffer)

                # Log progress periodically
                if processed_count % 1000 == 0:
                    self.logger.info(f"Processed {processed_count} documents...")

                doc_buffer = []

        # Process remaining documents in buffer
        if doc_buffer:
            self.logger.info(f"🟠 DEBUG: Processing remaining {len(doc_buffer)} documents in buffer...")
            self._process_document_batch(doc_buffer, field_docs)
            processed_count += len(doc_buffer)

        # Build field indexes
        self.field_indexer.build(field_docs)

        # Build retrieval models
        self._build_retrieval_models()

        # Initialize query executor
        self.query_executor = QueryExecutor(
            field_indexer=self.field_indexer,
            documents=field_docs
        )

        self.is_indexed = True

        stats = self.index_builder.get_stats()
        stats['field_index_docs'] = self.field_indexer.doc_count
        stats['total_metadata'] = len(self.doc_metadata)

        # Add 'total_docs' for compatibility
        stats['total_docs'] = stats.get('docs_indexed', 0)

        self.logger.info(
            f"Index built: {stats['total_docs']} docs, "
            f"{stats.get('unique_terms', stats.get('vocab_size', 0))} terms"
        )

        return stats

    def build_index_from_postgres(self,
                                  db_manager: Any,
                                  source: Optional[str] = None,
                                  category: Optional[str] = None,
                                  limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Build search index from PostgreSQL database.

        Args:
            db_manager: PostgresManager instance
            source: Filter by news source (None for all)
            category: Filter by category (None for all)
            limit: Maximum documents to index (None for all)

        Returns:
            Build statistics dictionary

        Examples:
            >>> from src.database.postgres_manager import PostgresManager
            >>> db = PostgresManager()
            >>> engine = UnifiedSearchEngine()
            >>> stats = engine.build_index_from_postgres(db, source='ltn', limit=10000)
        """
        self.logger.info("Building index from PostgreSQL...")

        # Read and index documents from database
        field_docs = []
        for doc in self.doc_reader.read_from_postgres(
            db_manager=db_manager,
            source=source,
            category=category,
            limit=limit
        ):
            # Add to incremental index (uses NewsDocument directly)
            self.index_builder.add_document(doc)

            # Track doc_id (0-indexed, matching index_builder)
            doc_id = self.index_builder.docs_indexed - 1

            # Store metadata AND content for retrieval
            self.doc_metadata[doc_id] = {
                'db_doc_id': doc.doc_id,
                'title': doc.title,
                'content': doc.content,  # Store full content for snippet extraction
                'source': doc.source,
                'category': doc.category,
                'published_at': doc.published_at,
                'url': doc.url,
                'author': doc.author
            }

            # Prepare for field indexing
            field_docs.append({
                'doc_id': doc_id,
                'title': doc.title,
                'content': doc.content,
                'source': doc.source,
                'category': doc.category,
                'published_date': doc.published_at,
                'author': doc.author or '',
                'url': doc.url or ''
            })

        # Build field indexes
        self.field_indexer.build(field_docs)

        # Build retrieval models
        self._build_retrieval_models()

        # Initialize query executor
        self.query_executor = QueryExecutor(
            field_indexer=self.field_indexer,
            documents=field_docs
        )

        self.is_indexed = True

        stats = self.index_builder.get_stats()
        stats['field_index_docs'] = self.field_indexer.doc_count
        stats['total_metadata'] = len(self.doc_metadata)

        self.logger.info(
            f"Index built from PostgreSQL: {stats['total_docs']} docs, "
            f"{stats['unique_terms']} terms"
        )

        return stats

    def _build_retrieval_models(self):
        """Build VSM, BM25, and Boolean retrieval models from index."""
        inverted_index = self.index_builder.index

        # VSM model
        self.vsm_model = VectorSpaceModel(inverted_index=inverted_index)
        self.vsm_model.term_weighting.build_from_index(inverted_index)
        self.vsm_model._compute_document_vectors()

        # BM25 model - manually populate from InvertedIndex
        self.bm25_model = BM25Ranker(tokenizer=self._tokenizer)

        # Convert InvertedIndex structure to BM25 structure
        # InvertedIndex: {term: [(doc_id, tf), ...]}
        # BM25: {term: {doc_id: tf}}
        for term, postings in inverted_index.index.items():
            self.bm25_model.inverted_index[term] = {doc_id: tf for doc_id, tf in postings}

        # Copy document lengths and counts
        self.bm25_model.doc_lengths = inverted_index.doc_lengths.copy()
        self.bm25_model.doc_count = inverted_index.doc_count

        # Calculate average document length
        if self.bm25_model.doc_count > 0:
            total_length = sum(self.bm25_model.doc_lengths.values())
            self.bm25_model.avg_doc_length = total_length / self.bm25_model.doc_count

        # Compute IDF values
        self.bm25_model._compute_idf()

        self.logger.info(f"BM25 model built: {self.bm25_model.doc_count} docs, "
                        f"{len(self.bm25_model.inverted_index)} terms, "
                        f"avg_doc_length={self.bm25_model.avg_doc_length:.2f}")

        # Boolean model
        self.boolean_model = BooleanRetrieval(inverted_index=inverted_index)

        self.logger.info("Retrieval models built (VSM, BM25, Boolean)")

    def search(self,
              query: str,
              mode: QueryMode = QueryMode.AUTO,
              ranking_model: RankingModel = RankingModel.BM25,
              top_k: int = 20) -> List[UnifiedSearchResult]:
        """
        Search for documents matching the query.

        Args:
            query: Query string
            mode: Query processing mode (auto-detect, simple, boolean, field)
            ranking_model: Ranking model to use (vsm, bm25, boolean, hybrid)
            top_k: Number of top results to return

        Returns:
            List of UnifiedSearchResult objects ranked by relevance

        Complexity:
            Time: O(V + k*log k) for VSM/BM25 where V is vocabulary, k is top_k
            Space: O(k)

        Examples:
            >>> engine = UnifiedSearchEngine()
            >>> engine.build_index_from_jsonl("data/raw")
            >>> results = engine.search("台灣 政治", top_k=10)
            >>> for r in results:
            ...     print(f"{r.rank}. {r.title} (score: {r.score:.4f})")
        """
        if not self.is_indexed:
            raise RuntimeError("Index not built. Call build_index_* first.")

        self.logger.debug(f"Searching: '{query}' (mode={mode.value}, model={ranking_model.value})")

        # Auto-detect query mode
        if mode == QueryMode.AUTO:
            mode = self._detect_query_mode(query)

        # Execute query based on mode
        if mode == QueryMode.FIELD or mode == QueryMode.BOOLEAN:
            # Parse and execute structured query
            return self._execute_field_query(query, top_k)

        elif mode == QueryMode.SIMPLE:
            # Simple keyword query - use ranking model
            return self._execute_simple_query(query, ranking_model, top_k)

        else:
            raise ValueError(f"Unsupported query mode: {mode}")

    def _detect_query_mode(self, query: str) -> QueryMode:
        """
        Auto-detect query mode from query string.

        Args:
            query: Query string

        Returns:
            Detected QueryMode

        Detection Rules:
            - Contains "field:" -> FIELD
            - Contains "AND", "OR", "NOT" -> BOOLEAN
            - Otherwise -> SIMPLE
        """
        query_upper = query.upper()

        # Check for field queries (field:value)
        if ':' in query and any(field in query.lower() for field in
                               ['title:', 'category:', 'source:', 'author:', 'content:', 'date:']):
            return QueryMode.FIELD

        # Check for boolean operators
        if any(op in query_upper for op in ['AND', 'OR', 'NOT']):
            return QueryMode.BOOLEAN

        # Default to simple query
        return QueryMode.SIMPLE

    def _execute_field_query(self, query: str, top_k: int) -> List[UnifiedSearchResult]:
        """
        Execute field-based or boolean query.

        Args:
            query: Query string with fields or boolean operators
            top_k: Number of results

        Returns:
            List of search results
        """
        try:
            # Parse query
            query_node = self.query_parser.parse(query)

            # Execute query
            results = self.query_executor.execute(query_node, top_k=top_k)

            # Convert to unified results
            unified_results = []
            for rank, result in enumerate(results, 1):
                doc_id = result.doc_id
                metadata = self.doc_metadata.get(doc_id, {})

                unified_results.append(UnifiedSearchResult(
                    doc_id=doc_id,
                    score=result.score,
                    rank=rank,
                    title=metadata.get('title', ''),
                    source=metadata.get('source', ''),
                    category=metadata.get('category', ''),
                    published_at=metadata.get('published_at', ''),
                    url=metadata.get('url', ''),
                    matched_fields=result.matched_fields,
                    ranking_model='field_query'
                ))

            return unified_results

        except Exception as e:
            self.logger.error(f"Field query execution failed: {e}")
            return []

    def _execute_simple_query(self,
                             query: str,
                             ranking_model: RankingModel,
                             top_k: int) -> List[UnifiedSearchResult]:
        """
        Execute simple keyword query using specified ranking model.

        Args:
            query: Query string
            ranking_model: Ranking model (VSM, BM25, HYBRID)
            top_k: Number of results

        Returns:
            List of search results
        """
        if ranking_model == RankingModel.BM25:
            # Use BM25
            result = self.bm25_model.search(query, topk=top_k)
            model_name = "bm25"

        elif ranking_model == RankingModel.VSM:
            # Use VSM
            result = self.vsm_model.search(query, topk=top_k)
            model_name = "vsm"

        elif ranking_model == RankingModel.HYBRID:
            # Hybrid: combine BM25 and VSM scores
            return self._execute_hybrid_query(query, top_k)

        else:
            raise ValueError(f"Unsupported ranking model: {ranking_model}")

        # Convert to unified results
        unified_results = []
        for rank, doc_id in enumerate(result.doc_ids, 1):
            # Scores is a list parallel to doc_ids, not a dict
            score = result.scores[rank - 1]  # rank starts at 1, list index at 0
            metadata = self.doc_metadata.get(doc_id, {})

            # Get content snippet
            content = self._get_content_snippet(doc_id, query)

            unified_results.append(UnifiedSearchResult(
                doc_id=doc_id,
                score=score,
                rank=rank,
                title=metadata.get('title', ''),
                content=content,
                source=metadata.get('source', ''),
                category=metadata.get('category', ''),
                published_at=metadata.get('published_at', ''),
                url=metadata.get('url', ''),
                ranking_model=model_name
            ))

        return unified_results

    def _execute_hybrid_query(self, query: str, top_k: int) -> List[UnifiedSearchResult]:
        """
        Execute hybrid query combining BM25 and VSM scores.

        Args:
            query: Query string
            top_k: Number of results

        Returns:
            List of search results with combined scores
        """
        # Get results from both models
        bm25_result = self.bm25_model.search(query, topk=top_k * 2)
        vsm_result = self.vsm_model.search(query, topk=top_k * 2)

        # Combine scores (weighted average)
        bm25_weight = 0.6
        vsm_weight = 0.4

        combined_scores = {}

        # Normalize and combine BM25 scores
        bm25_max = max(bm25_result.scores.values()) if bm25_result.scores else 1.0
        for doc_id, score in bm25_result.scores.items():
            combined_scores[doc_id] = (score / bm25_max) * bm25_weight

        # Normalize and add VSM scores
        vsm_max = max(vsm_result.scores.values()) if vsm_result.scores else 1.0
        for doc_id, score in vsm_result.scores.items():
            normalized = (score / vsm_max) * vsm_weight
            combined_scores[doc_id] = combined_scores.get(doc_id, 0) + normalized

        # Sort by combined score and get top-k
        sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Convert to unified results
        unified_results = []
        for rank, (doc_id, score) in enumerate(sorted_docs, 1):
            metadata = self.doc_metadata.get(doc_id, {})
            content = self._get_content_snippet(doc_id, query)

            unified_results.append(UnifiedSearchResult(
                doc_id=doc_id,
                score=score,
                rank=rank,
                title=metadata.get('title', ''),
                content=content,
                source=metadata.get('source', ''),
                category=metadata.get('category', ''),
                published_at=metadata.get('published_at', ''),
                url=metadata.get('url', ''),
                ranking_model='hybrid'
            ))

        return unified_results

    def _get_content_snippet(self, doc_id: int, query: str, max_length: int = 200) -> str:
        """
        Get content snippet showing query context.

        Implements intelligent snippet extraction that shows content around
        query terms for better user experience.

        Args:
            doc_id: Document ID
            query: Query string
            max_length: Maximum snippet length

        Returns:
            Content snippet with query context

        Complexity:
            Time: O(n + m) where n is content length, m is query terms
            Space: O(k) where k is snippet length

        Examples:
            >>> snippet = self._get_content_snippet(0, "台灣 政治", max_length=150)
            >>> "台灣" in snippet or "政治" in snippet
            True
        """
        # Get document content from metadata
        if doc_id not in self.doc_metadata:
            return ""

        content = self.doc_metadata[doc_id].get('content', '')
        if not content:
            return ""

        # If content is short enough, return it all
        if len(content) <= max_length:
            return content

        # Tokenize query to get search terms
        try:
            query_terms = self._tokenizer(query)
            if not query_terms:
                # No valid query terms, return beginning
                return content[:max_length] + "..."
        except Exception:
            # Tokenization failed, return beginning
            return content[:max_length] + "..."

        # Find best snippet position (where query terms appear)
        best_position = self._find_best_snippet_position(
            content=content,
            query_terms=query_terms,
            max_length=max_length
        )

        if best_position is None:
            # No query terms found in content, return beginning
            return content[:max_length] + "..."

        # Extract snippet around best position
        # Calculate start and end to center the snippet around the query term
        snippet_start = max(0, best_position - max_length // 3)
        snippet_end = min(len(content), snippet_start + max_length)

        # Adjust if we're at the end
        if snippet_end == len(content) and snippet_end - snippet_start < max_length:
            snippet_start = max(0, snippet_end - max_length)

        snippet = content[snippet_start:snippet_end]

        # Add ellipsis if we're not at the boundaries
        if snippet_start > 0:
            snippet = "..." + snippet
        if snippet_end < len(content):
            snippet = snippet + "..."

        return snippet

    def _find_best_snippet_position(self,
                                    content: str,
                                    query_terms: List[str],
                                    max_length: int) -> Optional[int]:
        """
        Find the best position in content to extract a snippet.

        Uses a sliding window approach to find the position that contains
        the most query terms.

        Args:
            content: Document content
            query_terms: List of query terms
            max_length: Maximum snippet length

        Returns:
            Best position index, or None if no query terms found

        Complexity:
            Time: O(n * m) where n is content length, m is query terms
            Space: O(1)
        """
        if not query_terms:
            return None

        best_position = None
        best_score = -1

        # Try to find each query term in content
        for term in query_terms:
            # Case-insensitive search
            term_lower = term.lower()
            content_lower = content.lower()

            # Find all occurrences of this term
            position = 0
            while True:
                pos = content_lower.find(term_lower, position)
                if pos == -1:
                    break

                # Count how many query terms appear in a window around this position
                window_start = max(0, pos - max_length // 2)
                window_end = min(len(content), pos + max_length // 2)
                window = content_lower[window_start:window_end]

                # Score this position by counting query terms in window
                score = sum(1 for qt in query_terms if qt.lower() in window)

                if score > best_score:
                    best_score = score
                    best_position = pos

                position = pos + 1

        return best_position

    def get_stats(self) -> Dict[str, Any]:
        """Get search engine statistics."""
        if not self.is_indexed:
            return {'status': 'not_indexed'}

        stats = self.index_builder.get_stats()
        stats['field_index_docs'] = self.field_indexer.doc_count
        stats['total_metadata'] = len(self.doc_metadata)
        stats['indexed'] = self.is_indexed

        return stats


def demo():
    """Demonstration of UnifiedSearchEngine."""
    print("=" * 80)
    print("Unified Search Engine Demo")
    print("=" * 80)

    # Initialize engine
    print("\n1. Initializing search engine...")
    engine = UnifiedSearchEngine()

    # Check for data
    data_dir = Path("data/raw")
    if not data_dir.exists() or not list(data_dir.glob("*.jsonl")):
        print(f"\n⚠ No data found in {data_dir}")
        print("Please run crawlers first to collect data.")
        return

    # Build index
    print("\n2. Building index from JSONL files...")
    print("-" * 80)
    stats = engine.build_index_from_jsonl(str(data_dir), limit=100)
    print(f"\nIndex Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value}")

    # Test queries
    print("\n3. Test Searches:")
    print("-" * 80)

    test_queries = [
        ("台灣 政治", QueryMode.SIMPLE, RankingModel.BM25),
        ("台灣 經濟", QueryMode.SIMPLE, RankingModel.VSM),
        ("title:台灣 AND category:政治", QueryMode.FIELD, RankingModel.BM25),
    ]

    for query, mode, model in test_queries:
        print(f"\n   Query: '{query}'")
        print(f"   Mode: {mode.value}, Model: {model.value}")

        try:
            results = engine.search(query, mode=mode, ranking_model=model, top_k=5)
            print(f"   Results: {len(results)} documents")

            for result in results[:3]:
                print(f"      {result.rank}. {result.title[:50]}... "
                      f"(score: {result.score:.4f}, source: {result.source})")

        except Exception as e:
            print(f"   ⚠ Error: {e}")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    demo()
