#!/usr/bin/env python3
"""
BERT Embeddings Builder for CNIRS Project

This script precomputes BERT embeddings for news articles to enable semantic search.
Uses Sentence-BERT (sentence-transformers) for efficient semantic similarity.

Features:
1. Compute document embeddings (title + content)
2. Save embeddings as NumPy arrays
3. Optional FAISS index for fast similarity search
4. Batch processing for efficiency

Usage:
    python scripts/build_bert_embeddings.py \\
        --input data/preprocessed/cna_mvp_preprocessed.jsonl \\
        --output data/indexes/bert_embeddings.npy \\
        --model paraphrase-multilingual-MiniLM-L12-v2

Complexity:
    Time: O(N*M) where N=documents, M=avg length (GPU accelerated)
    Space: O(N*D) where D=embedding dimension (typically 384 or 768)

Author: Information Retrieval System
License: Educational Use
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict
import time
from datetime import datetime
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Try importing sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available. Install with: pip install sentence-transformers")

# Try importing FAISS (optional)
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.info("FAISS not available (optional). Install with: pip install faiss-cpu")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BERTEmbeddingBuilder:
    """
    BERT embedding builder for semantic search.

    Uses Sentence-BERT to compute dense vector representations
    of news articles for semantic similarity search.

    Attributes:
        model_name: Sentence-BERT model name
        model: Loaded SentenceTransformer model
        device: Computation device ('cuda' or 'cpu')
        batch_size: Batch size for encoding
    """

    def __init__(self,
                 model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2',
                 device: str = None,
                 batch_size: int = 32):
        """
        Initialize BERT embedding builder.

        Args:
            model_name: Sentence-BERT model name
                Recommended models:
                - 'paraphrase-multilingual-MiniLM-L12-v2' (384 dim, 118M params, fast)
                - 'paraphrase-multilingual-mpnet-base-v2' (768 dim, better quality)
                - 'distiluse-base-multilingual-cased-v2' (512 dim, balanced)
            device: 'cuda', 'cpu', or None (auto-detect)
            batch_size: Batch size for encoding (larger = faster but more memory)
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            )

        logger.info(f"Initializing BERT embedding builder with model: {model_name}")

        # Load model
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.device = self.model.device
        self.batch_size = batch_size

        # Get embedding dimension
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        logger.info(f"Model loaded on device: {self.device}")
        logger.info(f"Embedding dimension: {self.embedding_dim}")
        logger.info(f"Batch size: {batch_size}")

    def prepare_texts(self, articles: List[Dict], use_summary: bool = False) -> List[str]:
        """
        Prepare texts for embedding.

        Args:
            articles: List of article dictionaries
            use_summary: Use summary instead of full content (faster, less accurate)

        Returns:
            List of text strings to encode
        """
        texts = []

        for article in articles:
            title = article.get('title', '')

            if use_summary:
                # Use summary if available
                content = article.get('summary', article.get('content', ''))
            else:
                # Use full content
                content = article.get('content', '')

            # Combine title and content
            # Title is weighted more by placing it first
            text = f"{title} {content}"
            texts.append(text)

        return texts

    def compute_embeddings(self,
                          articles: List[Dict],
                          use_summary: bool = False,
                          show_progress: bool = True) -> np.ndarray:
        """
        Compute BERT embeddings for articles.

        Args:
            articles: List of article dictionaries
            use_summary: Use summary instead of full content
            show_progress: Show progress bar

        Returns:
            NumPy array of embeddings (N x D)
        """
        logger.info(f"Computing embeddings for {len(articles)} articles...")
        start_time = time.time()

        # Prepare texts
        texts = self.prepare_texts(articles, use_summary=use_summary)

        # Compute embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalization for cosine similarity
        )

        elapsed_time = time.time() - start_time

        logger.info(f"Embeddings computed in {elapsed_time:.2f} seconds")
        logger.info(f"Throughput: {len(articles)/elapsed_time:.1f} articles/second")
        logger.info(f"Embedding shape: {embeddings.shape}")

        return embeddings

    def save_embeddings(self,
                       embeddings: np.ndarray,
                       output_path: Path,
                       article_ids: List[str] = None):
        """
        Save embeddings to disk.

        Args:
            embeddings: NumPy array of embeddings
            output_path: Output file path (.npy)
            article_ids: Optional list of article IDs (for mapping)
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save embeddings
        np.save(output_path, embeddings)
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"Saved embeddings: {output_path} ({file_size_mb:.2f} MB)")

        # Save metadata
        metadata_path = output_path.with_suffix('.json')
        metadata = {
            'model_name': self.model_name,
            'embedding_dim': self.embedding_dim,
            'num_documents': embeddings.shape[0],
            'created_at': datetime.now().isoformat(),
            'article_ids': article_ids if article_ids else []
        }

        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved metadata: {metadata_path}")

    def build_faiss_index(self,
                         embeddings: np.ndarray,
                         output_path: Path,
                         index_type: str = 'flat'):
        """
        Build FAISS index for fast similarity search (optional).

        Args:
            embeddings: NumPy array of embeddings
            output_path: Output file path (.faiss or .index)
            index_type: Index type
                - 'flat': Exact search (best accuracy, slower for large datasets)
                - 'ivf': Inverted file index (faster, approximate)
                - 'hnsw': Hierarchical NSW (best for < 1M docs)
        """
        if not FAISS_AVAILABLE:
            logger.warning("FAISS not available, skipping index building")
            return

        logger.info(f"Building FAISS index (type: {index_type})...")

        # Ensure embeddings are float32
        embeddings = embeddings.astype('float32')
        d = embeddings.shape[1]
        n = embeddings.shape[0]

        if index_type == 'flat':
            # Exact search with inner product (for normalized vectors = cosine)
            index = faiss.IndexFlatIP(d)

        elif index_type == 'ivf':
            # IVF index (approximate search)
            nlist = min(100, n // 10)  # Number of clusters
            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
            # Train index
            index.train(embeddings)

        elif index_type == 'hnsw':
            # HNSW index (good for < 1M docs)
            M = 32  # Number of connections
            index = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)

        else:
            raise ValueError(f"Unknown index type: {index_type}")

        # Add embeddings to index
        index.add(embeddings)

        # Save index
        output_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(output_path))

        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"Saved FAISS index: {output_path} ({file_size_mb:.2f} MB)")


def load_articles(file_path: Path) -> List[Dict]:
    """Load articles from JSONL file."""
    articles = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                articles.append(json.loads(line))
    return articles


def generate_report(embeddings: np.ndarray,
                   stats: Dict,
                   output_path: Path):
    """Generate embedding statistics report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("BERT EMBEDDINGS STATISTICS\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("MODEL INFORMATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Model name:           {stats['model_name']}\n")
        f.write(f"Embedding dimension:  {stats['embedding_dim']}\n")
        f.write(f"Device:               {stats['device']}\n")
        f.write(f"Batch size:           {stats['batch_size']}\n\n")

        f.write("COLLECTION STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total documents:      {stats['num_documents']}\n")
        f.write(f"Embedding shape:      {embeddings.shape}\n")
        f.write(f"Total size:           {embeddings.nbytes / (1024*1024):.2f} MB\n")
        f.write(f"Per document:         {embeddings.nbytes / stats['num_documents'] / 1024:.2f} KB\n\n")

        f.write("PERFORMANCE\n")
        f.write("-" * 80 + "\n")
        f.write(f"Encoding time:        {stats['encoding_time']:.2f} seconds\n")
        f.write(f"Throughput:           {stats['throughput']:.1f} docs/second\n")
        f.write(f"Time per document:    {stats['encoding_time']/stats['num_documents']:.4f} seconds\n\n")

        f.write("EMBEDDING STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Mean norm:            {np.linalg.norm(embeddings, axis=1).mean():.4f}\n")
        f.write(f"Std norm:             {np.linalg.norm(embeddings, axis=1).std():.4f}\n")
        f.write(f"Min value:            {embeddings.min():.4f}\n")
        f.write(f"Max value:            {embeddings.max():.4f}\n")

    logger.info(f"Report saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Build BERT embeddings for semantic search',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/build_bert_embeddings.py \\
      --input data/preprocessed/cna_mvp_preprocessed.jsonl \\
      --output data/indexes/bert_embeddings.npy

  # With FAISS index
  python scripts/build_bert_embeddings.py \\
      --input data/preprocessed/cna_mvp_preprocessed.jsonl \\
      --output data/indexes/bert_embeddings.npy \\
      --faiss data/indexes/bert_faiss.index \\
      --faiss-type hnsw

  # Use summary instead of full content (faster)
  python scripts/build_bert_embeddings.py \\
      --input data/preprocessed/cna_mvp_preprocessed.jsonl \\
      --output data/indexes/bert_embeddings.npy \\
      --use-summary

  # Custom model
  python scripts/build_bert_embeddings.py \\
      --input data/preprocessed/cna_mvp_preprocessed.jsonl \\
      --output data/indexes/bert_embeddings.npy \\
      --model paraphrase-multilingual-mpnet-base-v2
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input JSONL file (preprocessed articles)'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output file for embeddings (.npy)'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='paraphrase-multilingual-MiniLM-L12-v2',
        help='Sentence-BERT model name (default: paraphrase-multilingual-MiniLM-L12-v2)'
    )

    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu', 'auto'],
        default='auto',
        help='Computation device (default: auto)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for encoding (default: 32)'
    )

    parser.add_argument(
        '--use-summary',
        action='store_true',
        help='Use summary instead of full content (faster but less accurate)'
    )

    parser.add_argument(
        '--faiss',
        type=str,
        default=None,
        help='Build FAISS index and save to this path (optional)'
    )

    parser.add_argument(
        '--faiss-type',
        type=str,
        choices=['flat', 'ivf', 'hnsw'],
        default='flat',
        help='FAISS index type (default: flat)'
    )

    parser.add_argument(
        '--report',
        type=str,
        default=None,
        help='Output path for statistics report (optional)'
    )

    args = parser.parse_args()

    # Validate paths
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    # Auto-detect device
    device = args.device
    if device == 'auto':
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load articles
    logger.info(f"Loading articles from {input_path}")
    articles = load_articles(input_path)
    logger.info(f"Loaded {len(articles)} articles")

    # Initialize builder
    builder = BERTEmbeddingBuilder(
        model_name=args.model,
        device=device,
        batch_size=args.batch_size
    )

    # Compute embeddings
    start_time = time.time()
    embeddings = builder.compute_embeddings(
        articles,
        use_summary=args.use_summary,
        show_progress=True
    )
    encoding_time = time.time() - start_time

    # Extract article IDs
    article_ids = [article['article_id'] for article in articles]

    # Save embeddings
    builder.save_embeddings(embeddings, output_path, article_ids=article_ids)

    # Build FAISS index (optional)
    if args.faiss:
        faiss_path = Path(args.faiss)
        builder.build_faiss_index(embeddings, faiss_path, index_type=args.faiss_type)

    # Generate report
    if args.report:
        report_path = Path(args.report)
    else:
        report_path = output_path.parent.parent / 'stats' / 'bert_embeddings_stats.txt'

    stats = {
        'model_name': args.model,
        'embedding_dim': builder.embedding_dim,
        'device': str(builder.device),
        'batch_size': args.batch_size,
        'num_documents': len(articles),
        'encoding_time': encoding_time,
        'throughput': len(articles) / encoding_time
    }

    generate_report(embeddings, stats, report_path)

    # Print summary
    print("\n" + "=" * 80)
    print("BERT EMBEDDINGS BUILD COMPLETED")
    print("=" * 80)
    print(f"Input:           {input_path}")
    print(f"Output:          {output_path}")
    print(f"Report:          {report_path}")
    print(f"Model:           {args.model}")
    print(f"Documents:       {len(articles)}")
    print(f"Embedding dim:   {builder.embedding_dim}")
    print(f"Device:          {device}")
    print(f"Encoding time:   {encoding_time:.2f} seconds")
    print(f"Throughput:      {len(articles)/encoding_time:.1f} docs/second")
    if args.faiss:
        print(f"FAISS index:     {args.faiss}")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
