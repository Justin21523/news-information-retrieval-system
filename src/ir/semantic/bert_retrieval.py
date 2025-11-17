"""
BERT-based Semantic Retrieval

This module implements dense retrieval using BERT embeddings for semantic search.
Unlike traditional sparse retrieval (BM25, TF-IDF), dense retrieval uses neural
embeddings to capture semantic similarity beyond exact term matching.

Dense Retrieval:
    - Documents and queries encoded as dense vectors (768-dim for BERT-base)
    - Similarity computed using cosine similarity or dot product
    - Captures semantic relationships (synonyms, paraphrases, concepts)
    - No vocabulary mismatch problem

Key Concepts:
    - Pre-trained Language Models: BERT, RoBERTa, ELECTRA
    - Sentence Embeddings: Dense vector representations
    - Bi-encoder Architecture: Separate encoders for query and document
    - Approximate Nearest Neighbor (ANN): Efficient similarity search

Formulas:
    - Cosine Similarity: sim(q, d) = (q · d) / (||q|| * ||d||)
    - Dot Product: sim(q, d) = q · d
    - Euclidean Distance: dist(q, d) = ||q - d||

Models:
    - Chinese BERT: bert-base-chinese, hfl/chinese-bert-wwm-ext
    - Multilingual: paraphrase-multilingual-MiniLM-L12-v2
    - Sentence-BERT: Optimized for sentence similarity

Key Features:
    - Multiple BERT model support
    - Efficient batch encoding
    - Cosine similarity ranking
    - Optional ANN indexing (FAISS)
    - Hybrid sparse-dense retrieval support

Reference:
    Devlin et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers"
    Karpukhin et al. (2020). "Dense Passage Retrieval for Open-Domain QA"
    Reimers & Gurevych (2019). "Sentence-BERT: Sentence Embeddings using Siamese BERT"

Author: Information Retrieval System
License: Educational Use
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass

# Optional imports (graceful degradation if not available)
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers or torch not available. BERT retrieval will not work.")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available. Using brute-force search instead.")


@dataclass
class SemanticResult:
    """
    Result of semantic search.

    Attributes:
        query: Original query string
        doc_ids: List of document IDs ranked by semantic similarity
        scores: Corresponding similarity scores
        num_results: Total number of results
        model_name: BERT model used
    """
    query: str
    doc_ids: List[int]
    scores: List[float]
    num_results: int
    model_name: str


class BERTRetrieval:
    """
    BERT-based semantic retrieval using dense embeddings.

    Encodes documents and queries into dense vectors and ranks by
    cosine similarity. Supports multiple Chinese and multilingual BERT models.

    Attributes:
        model_name: Name of BERT model to use
        tokenizer: BERT tokenizer
        model: BERT model
        doc_embeddings: Document embeddings (N x D matrix)
        doc_count: Number of documents
        embedding_dim: Dimension of embeddings
        device: Computation device (cpu or cuda)
        use_faiss: Whether to use FAISS for ANN search

    Complexity:
        - Encoding: O(N * M * D^2) where N=docs, M=avg length, D=model dim
        - Search: O(N * D) for brute-force, O(log N * D) with FAISS
        - Space: O(N * D) for document embeddings

    Recommended Models:
        - Chinese: "bert-base-chinese", "hfl/chinese-bert-wwm-ext"
        - Multilingual: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        - English: "sentence-transformers/all-MiniLM-L6-v2"
    """

    def __init__(self,
                 model_name: str = "bert-base-chinese",
                 device: str = "cpu",
                 use_faiss: bool = False,
                 max_length: int = 512):
        """
        Initialize BERT retrieval.

        Args:
            model_name: Hugging Face model name
            device: Computation device ('cpu' or 'cuda')
            use_faiss: Use FAISS for approximate nearest neighbor search
            max_length: Maximum sequence length

        Complexity:
            Time: O(1) + model loading time
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers and torch are required for BERT retrieval. "
                "Install with: pip install transformers torch"
            )

        self.logger = logging.getLogger(__name__)

        self.model_name = model_name
        self.device = device
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        self.max_length = max_length

        # Load model and tokenizer
        self.logger.info(f"Loading BERT model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Document embeddings
        self.doc_embeddings: Optional[np.ndarray] = None
        self.doc_count: int = 0
        self.embedding_dim: int = self.model.config.hidden_size

        # FAISS index (optional)
        self.faiss_index = None

        self.logger.info(
            f"BERTRetrieval initialized: {model_name}, "
            f"dim={self.embedding_dim}, device={device}, faiss={self.use_faiss}"
        )

    def encode(self, texts: List[str], batch_size: int = 32,
               show_progress: bool = False) -> np.ndarray:
        """
        Encode texts into BERT embeddings.

        Uses mean pooling over token embeddings for sentence representation.

        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            show_progress: Show progress bar (requires tqdm)

        Returns:
            Embeddings matrix (N x D)

        Complexity:
            Time: O(N * M * D^2) where N=texts, M=avg length, D=hidden dim
            Space: O(N * D)

        Examples:
            >>> bert = BERTRetrieval()
            >>> embeddings = bert.encode(["text 1", "text 2"])
            >>> embeddings.shape
            (2, 768)
        """
        embeddings = []

        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            # Tokenize
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )

            # Move to device
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            # Encode (no gradient computation)
            with torch.no_grad():
                outputs = self.model(**encoded)

            # Mean pooling over tokens (excluding padding)
            attention_mask = encoded['attention_mask']
            token_embeddings = outputs.last_hidden_state

            # Expand attention mask for broadcasting
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

            # Sum embeddings and divide by number of tokens
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            batch_embeddings = sum_embeddings / sum_mask

            # Convert to numpy
            batch_embeddings = batch_embeddings.cpu().numpy()
            embeddings.append(batch_embeddings)

        # Concatenate all batches
        all_embeddings = np.vstack(embeddings)

        return all_embeddings

    def build_index(self, documents: List[str], batch_size: int = 32) -> None:
        """
        Build document embeddings index.

        Args:
            documents: List of document texts
            batch_size: Batch size for encoding

        Complexity:
            Time: O(N * M * D^2) where N=docs, M=avg doc length
            Space: O(N * D) for embeddings
        """
        self.logger.info(f"Building BERT index for {len(documents)} documents...")

        # Encode all documents
        self.doc_embeddings = self.encode(documents, batch_size=batch_size)
        self.doc_count = len(documents)

        # Build FAISS index if enabled
        if self.use_faiss:
            self._build_faiss_index()

        self.logger.info(
            f"BERT index built: {self.doc_count} docs, "
            f"embedding_dim={self.embedding_dim}"
        )

    def _build_faiss_index(self) -> None:
        """
        Build FAISS index for approximate nearest neighbor search.

        Uses IndexFlatIP (Inner Product) for exact cosine similarity.
        For large collections, can use IndexIVFFlat or IndexHNSW.

        Complexity:
            Time: O(N * D) for flat index
        """
        if not FAISS_AVAILABLE:
            self.logger.warning("FAISS not available, skipping FAISS index")
            return

        # Normalize embeddings for cosine similarity
        normalized_embeddings = self.doc_embeddings / np.linalg.norm(
            self.doc_embeddings, axis=1, keepdims=True
        )

        # Create FAISS index (Inner Product = cosine similarity for normalized vectors)
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        self.faiss_index.add(normalized_embeddings.astype('float32'))

        self.logger.info(f"FAISS index built with {self.faiss_index.ntotal} vectors")

    def search(self, query: str, topk: int = 10) -> SemanticResult:
        """
        Search documents using semantic similarity.

        Args:
            query: Query string
            topk: Number of top results to return

        Returns:
            SemanticResult with ranked documents

        Complexity:
            Time: O(M * D^2 + N * D) where M=query length, N=docs, D=embedding dim
            Space: O(D) for query embedding

        Examples:
            >>> bert = BERTRetrieval()
            >>> bert.build_index(documents)
            >>> result = bert.search("information retrieval", topk=10)
            >>> result.doc_ids
            [5, 12, 3, 18, ...]
        """
        if self.doc_embeddings is None:
            raise ValueError("Index not built. Call build_index() first.")

        # Encode query
        query_embedding = self.encode([query])[0]  # Shape: (D,)

        # Use FAISS or brute-force search
        if self.use_faiss and self.faiss_index is not None:
            scores, doc_ids = self._search_faiss(query_embedding, topk)
        else:
            scores, doc_ids = self._search_brute_force(query_embedding, topk)

        return SemanticResult(
            query=query,
            doc_ids=doc_ids.tolist(),
            scores=scores.tolist(),
            num_results=len(doc_ids),
            model_name=self.model_name
        )

    def _search_brute_force(self, query_embedding: np.ndarray,
                           topk: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Brute-force search using cosine similarity.

        Args:
            query_embedding: Query embedding vector (D,)
            topk: Number of results

        Returns:
            (scores, doc_ids) arrays

        Complexity:
            Time: O(N * D) where N=docs
        """
        # Normalize for cosine similarity
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        doc_norms = self.doc_embeddings / np.linalg.norm(
            self.doc_embeddings, axis=1, keepdims=True
        )

        # Compute cosine similarities
        similarities = np.dot(doc_norms, query_norm)

        # Get top-k
        topk_indices = np.argsort(similarities)[::-1][:topk]
        topk_scores = similarities[topk_indices]

        return topk_scores, topk_indices

    def _search_faiss(self, query_embedding: np.ndarray,
                     topk: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search using FAISS index.

        Args:
            query_embedding: Query embedding vector (D,)
            topk: Number of results

        Returns:
            (scores, doc_ids) arrays

        Complexity:
            Time: O(log N * D) with approximate index, O(N * D) with flat index
        """
        # Normalize query
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        query_norm = query_norm.reshape(1, -1).astype('float32')

        # Search
        scores, doc_ids = self.faiss_index.search(query_norm, topk)

        return scores[0], doc_ids[0]

    def batch_search(self, queries: List[str], topk: int = 10) -> List[SemanticResult]:
        """
        Search multiple queries in batch.

        Args:
            queries: List of query strings
            topk: Number of results per query

        Returns:
            List of SemanticResult objects

        Complexity:
            Time: O(Q * (M * D^2 + N * D)) where Q=num queries
        """
        results = []
        for query in queries:
            result = self.search(query, topk=topk)
            results.append(result)

        return results

    def get_stats(self) -> Dict:
        """Get BERT retrieval statistics."""
        return {
            'model_name': self.model_name,
            'doc_count': self.doc_count,
            'embedding_dim': self.embedding_dim,
            'device': self.device,
            'use_faiss': self.use_faiss,
            'max_length': self.max_length
        }


def demo():
    """Demonstration of BERT retrieval."""
    if not TRANSFORMERS_AVAILABLE:
        print("ERROR: transformers and torch not available.")
        print("Install with: pip install transformers torch")
        return

    print("=" * 60)
    print("BERT Semantic Retrieval Demo")
    print("=" * 60)

    # Sample documents (Chinese)
    documents = [
        "資訊檢索是從大量資料中找到相關資訊的過程",
        "檢索模型包括布林模型和向量空間模型",
        "BM25是一種機率檢索函數",
        "搜尋引擎使用各種排序演算法包括BM25",
        "語言模型估計詞序列的機率",
        "BERT是一種預訓練的深度學習模型"
    ]

    print(f"\nDataset: {len(documents)} documents (Chinese)")

    # Initialize BERT retrieval
    # Note: Using a smaller multilingual model for demo
    # For production, use "hfl/chinese-bert-wwm-ext" or similar
    print("\nInitializing BERT model...")
    print("(For Chinese text, consider using 'hfl/chinese-bert-wwm-ext')")

    try:
        # Use a smaller model for demo (faster loading)
        bert = BERTRetrieval(
            model_name="bert-base-chinese",
            device="cpu",
            use_faiss=False
        )

        # Build index
        print("\nBuilding document embeddings...")
        bert.build_index(documents, batch_size=8)

        stats = bert.get_stats()
        print(f"Model: {stats['model_name']}")
        print(f"Documents: {stats['doc_count']}")
        print(f"Embedding dimension: {stats['embedding_dim']}")

        # Test queries
        queries = [
            "檢索模型",
            "搜尋引擎排序",
            "深度學習BERT"
        ]

        print("\n" + "-" * 60)
        print("Semantic Search Results")
        print("-" * 60)

        for query in queries:
            print(f"\nQuery: '{query}'")
            result = bert.search(query, topk=3)

            print(f"  Results: {result.num_results}")
            for i, (doc_id, score) in enumerate(zip(result.doc_ids, result.scores), 1):
                print(f"  {i}. Doc {doc_id}: {score:.4f}")
                print(f"     {documents[doc_id]}")

    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure you have transformers and torch installed:")
        print("  pip install transformers torch")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    demo()
