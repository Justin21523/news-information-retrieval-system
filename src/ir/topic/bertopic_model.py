"""
BERTopic Model Wrapper for Traditional Chinese

This module provides a high-level wrapper for BERTopic topic modeling,
optimized for Traditional Chinese documents.

BERTopic Pipeline:
    1. Document Embeddings (BERT/Sentence-Transformers)
    2. Dimensionality Reduction (UMAP)
    3. Clustering (HDBSCAN)
    4. Topic Representation (c-TF-IDF)

Key Features:
    - Multilingual BERT embeddings for Traditional Chinese
    - Automatic optimal topic number detection
    - Dynamic topic modeling support
    - Topic coherence evaluation
    - Interactive visualizations
    - Model persistence

Complexity:
    - fit(): O(n*d + n log n) where n=docs, d=embed_dim
    - transform(): O(m*d) where m=new docs
    - Embedding: O(n*L) where L=avg doc length

Reference:
    Grootendorst, M. (2022). "BERTopic: Neural topic modeling with a
        class-based TF-IDF procedure". arXiv:2203.05794.

Author: Information Retrieval System
License: Educational Use
"""

from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import logging
import pickle

import numpy as np
import pandas as pd

# BERTopic dependencies
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN

# Visualization
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


@dataclass
class TopicInfo:
    """
    Topic information container.

    Attributes:
        topic_id: Topic ID (-1 for outliers)
        topic_words: List of (word, score) tuples
        topic_size: Number of documents in topic
        representative_docs: Sample documents from topic
    """
    topic_id: int
    topic_words: List[Tuple[str, float]]
    topic_size: int
    representative_docs: Optional[List[str]] = None

    def __str__(self) -> str:
        words = ", ".join([f"{w}" for w, s in self.topic_words[:5]])
        return f"Topic {self.topic_id} ({self.topic_size} docs): {words}"


class BERTopicModel:
    """
    BERTopic wrapper for Traditional Chinese topic modeling.

    Provides high-level interface for training, inference, and analysis
    of topics using state-of-the-art neural topic modeling.

    Attributes:
        model: BERTopic model instance
        embedding_model: Sentence transformer for embeddings
        umap_model: UMAP for dimensionality reduction
        hdbscan_model: HDBSCAN for clustering
        logger: Logger instance

    Examples:
        >>> # Initialize and train
        >>> topic_model = BERTopicModel(language='multilingual')
        >>> topics, probs = topic_model.fit_transform(documents)
        
        >>> # Get topic information
        >>> topic_info = topic_model.get_topic_info()
        >>> print(topic_info.head())
        
        >>> # Visualize topics
        >>> fig = topic_model.visualize_topics()
        >>> fig.write_html('topics.html')
    """

    def __init__(self,
                 embedding_model: str = 'paraphrase-multilingual-MiniLM-L12-v2',
                 language: str = 'multilingual',
                 n_neighbors: int = 15,
                 n_components: int = 5,
                 min_cluster_size: int = 10,
                 min_topic_size: int = 10,
                 nr_topics: Optional[int] = None,
                 top_n_words: int = 10,
                 verbose: bool = False,
                 device: str = 'cpu'):
        """
        Initialize BERTopic model.

        Args:
            embedding_model: Sentence transformer model name
                - 'paraphrase-multilingual-MiniLM-L12-v2': Best for Chinese (default)
                - 'distiluse-base-multilingual-cased-v1': Faster alternative
                - 'xlm-roberta-base': More accurate but slower
            language: Language for stopwords ('multilingual', 'chinese', 'english')
            n_neighbors: UMAP n_neighbors (controls local vs global structure)
            n_components: UMAP target dimensions (typically 5-10)
            min_cluster_size: HDBSCAN minimum cluster size
            min_topic_size: Minimum documents per topic
            nr_topics: Target number of topics (None=automatic)
            top_n_words: Number of words per topic representation
            verbose: Enable verbose logging
            device: Device for embeddings ('cpu' or 'cuda')

        Complexity:
            Time: O(1) - lazy initialization
            Space: O(V) where V=vocab size
        """
        self.logger = logging.getLogger(__name__)
        self.verbose = verbose
        self.device = device
        self.top_n_words = top_n_words

        # Initialize embedding model
        self.logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model, device=device)
        self.logger.info(f"Embedding dimension: {self.embedding_model.get_sentence_embedding_dimension()}")

        # Initialize UMAP for dimensionality reduction
        self.umap_model = UMAP(
            n_neighbors=n_neighbors,
            n_components=n_components,
            min_dist=0.0,
            metric='cosine',
            random_state=42
        )

        # Initialize HDBSCAN for clustering
        self.hdbscan_model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )

        # Initialize BERTopic
        self.model = BERTopic(
            embedding_model=self.embedding_model,
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            language=language,
            top_n_words=top_n_words,
            nr_topics=nr_topics,
            min_topic_size=min_topic_size,
            verbose=verbose,
            calculate_probabilities=True
        )

        self.is_fitted = False
        self.documents = None
        self.topics = None
        self.probabilities = None

        self.logger.info(
            f"BERTopicModel initialized: embedding={embedding_model}, "
            f"n_components={n_components}, min_cluster_size={min_cluster_size}"
        )

    # ========================================================================
    # Training Methods
    # ========================================================================

    def fit(self, documents: List[str], embeddings: Optional[np.ndarray] = None) -> 'BERTopicModel':
        """
        Fit the topic model on documents.

        Args:
            documents: List of document texts
            embeddings: Pre-computed embeddings (optional, will compute if None)

        Returns:
            Self for method chaining

        Complexity:
            Time: O(n*d + n log n) where n=docs, d=embed_dim
            Space: O(n*d + k*V) where k=topics, V=vocab

        Examples:
            >>> model = BERTopicModel()
            >>> model.fit(documents)
            >>> print(f"Found {len(model.get_topic_info())} topics")
        """
        if not documents:
            raise ValueError("Documents list cannot be empty")

        self.logger.info(f"Fitting BERTopic on {len(documents)} documents")

        # Fit model
        topics, probs = self.model.fit_transform(documents, embeddings=embeddings)

        # Store results
        self.documents = documents
        self.topics = topics
        self.probabilities = probs
        self.is_fitted = True

        # Log statistics
        unique_topics = len(set(topics))
        outliers = sum(1 for t in topics if t == -1)
        self.logger.info(
            f"BERTopic fitted: {unique_topics} topics, "
            f"{outliers} outliers ({outliers/len(topics)*100:.1f}%)"
        )

        return self

    def fit_transform(self, documents: List[str],
                     embeddings: Optional[np.ndarray] = None) -> Tuple[List[int], np.ndarray]:
        """
        Fit the model and return topic assignments.

        Args:
            documents: List of document texts
            embeddings: Pre-computed embeddings (optional)

        Returns:
            Tuple of (topics, probabilities)
                - topics: List of topic IDs for each document
                - probabilities: Array of topic probabilities

        Complexity:
            Time: O(n*d + n log n)
            Space: O(n*d)

        Examples:
            >>> model = BERTopicModel()
            >>> topics, probs = model.fit_transform(documents)
            >>> print(f"Document 0 belongs to topic {topics[0]}")
        """
        self.fit(documents, embeddings=embeddings)
        return self.topics, self.probabilities

    def transform(self, documents: List[str]) -> Tuple[List[int], np.ndarray]:
        """
        Predict topics for new documents.

        Args:
            documents: List of new document texts

        Returns:
            Tuple of (topics, probabilities)

        Raises:
            ValueError: If model not fitted

        Complexity:
            Time: O(m*d) where m=new docs
            Space: O(m*d)

        Examples:
            >>> model.fit(train_docs)
            >>> new_topics, new_probs = model.transform(test_docs)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transform")

        topics, probs = self.model.transform(documents)
        return topics, probs

    # ========================================================================
    # Topic Information Methods
    # ========================================================================

    def get_topic_info(self, topic_id: Optional[int] = None) -> Union[pd.DataFrame, TopicInfo]:
        """
        Get information about topics.

        Args:
            topic_id: Specific topic ID (None=all topics)

        Returns:
            DataFrame with topic info or TopicInfo for specific topic

        Raises:
            ValueError: If model not fitted

        Examples:
            >>> # Get all topics
            >>> info = model.get_topic_info()
            >>> print(info.head())
            
            >>> # Get specific topic
            >>> topic_0 = model.get_topic_info(topic_id=0)
            >>> print(topic_0)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        if topic_id is None:
            # Return DataFrame for all topics
            return self.model.get_topic_info()
        else:
            # Return TopicInfo for specific topic
            topic_words = self.model.get_topic(topic_id)
            if not topic_words:
                raise ValueError(f"Topic {topic_id} not found")

            topic_size = self.model.get_topic_info().loc[
                self.model.get_topic_info()['Topic'] == topic_id,
                'Count'
            ].values[0]

            # Get representative documents
            rep_docs = self.get_representative_docs(topic_id, n_docs=3)

            return TopicInfo(
                topic_id=topic_id,
                topic_words=topic_words,
                topic_size=int(topic_size),
                representative_docs=rep_docs
            )

    def get_topics(self) -> Dict[int, List[Tuple[str, float]]]:
        """
        Get all topics with their word representations.

        Returns:
            Dictionary mapping topic_id -> [(word, score), ...]

        Examples:
            >>> topics = model.get_topics()
            >>> for topic_id, words in topics.items():
            ...     print(f"Topic {topic_id}: {words[:3]}")
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        return self.model.get_topics()

    def get_topic_words(self, topic_id: int, top_n: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        Get words for a specific topic.

        Args:
            topic_id: Topic ID
            top_n: Number of words (None=all)

        Returns:
            List of (word, score) tuples

        Examples:
            >>> words = model.get_topic_words(0, top_n=5)
            >>> print(words)
            [('機器學習', 0.42), ('深度學習', 0.38), ...]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        words = self.model.get_topic(topic_id)
        if top_n:
            words = words[:top_n]
        return words

    def get_representative_docs(self, topic_id: int, n_docs: int = 5) -> List[str]:
        """
        Get representative documents for a topic.

        Args:
            topic_id: Topic ID
            n_docs: Number of documents

        Returns:
            List of representative document texts

        Examples:
            >>> docs = model.get_representative_docs(0, n_docs=3)
            >>> for doc in docs:
            ...     print(doc[:100])
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        # Get document indices for this topic
        topic_docs_idx = [i for i, t in enumerate(self.topics) if t == topic_id]

        if not topic_docs_idx:
            return []

        # Get documents with highest probability for this topic
        topic_probs = [self.probabilities[i][topic_id + 1] for i in topic_docs_idx]
        top_idx = np.argsort(topic_probs)[-n_docs:][::-1]

        return [self.documents[topic_docs_idx[i]] for i in top_idx]

    # ========================================================================
    # Visualization Methods
    # ========================================================================

    def visualize_topics(self, width: int = 1200, height: int = 800) -> 'go.Figure':
        """
        Create interactive 2D visualization of topics.

        Args:
            width: Figure width in pixels
            height: Figure height in pixels

        Returns:
            Plotly figure

        Raises:
            ImportError: If plotly not available

        Examples:
            >>> fig = model.visualize_topics()
            >>> fig.write_html('topics.html')
            >>> fig.show()
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly required for visualization")

        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        return self.model.visualize_topics(width=width, height=height)

    def visualize_barchart(self, topics: Optional[List[int]] = None,
                          n_words: int = 10, width: int = 800, height: int = 600) -> 'go.Figure':
        """
        Create bar chart visualization of topic words.

        Args:
            topics: List of topic IDs (None=top 8)
            n_words: Number of words per topic
            width: Figure width
            height: Figure height

        Returns:
            Plotly figure

        Examples:
            >>> fig = model.visualize_barchart(topics=[0, 1, 2])
            >>> fig.write_html('barchart.html')
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly required for visualization")

        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        return self.model.visualize_barchart(
            topics=topics,
            top_n_topics=n_words,
            width=width,
            height=height
        )

    def visualize_hierarchy(self, width: int = 1000, height: int = 800) -> 'go.Figure':
        """
        Create hierarchical clustering visualization of topics.

        Returns:
            Plotly figure

        Examples:
            >>> fig = model.visualize_hierarchy()
            >>> fig.write_html('hierarchy.html')
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly required for visualization")

        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        return self.model.visualize_hierarchy(width=width, height=height)

    # ========================================================================
    # Model Persistence
    # ========================================================================

    def save(self, path: Union[str, Path], save_embeddings: bool = False):
        """
        Save the fitted model to disk.

        Args:
            path: Save path
            save_embeddings: Whether to save embeddings (increases size)

        Examples:
            >>> model.fit(documents)
            >>> model.save('models/topic_model.pkl')
        """
        if not self.is_fitted:
            self.logger.warning("Saving unfitted model")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save using BERTopic's method
        self.model.save(
            str(path),
            serialization='pickle',
            save_embedding_model=True if save_embeddings else False
        )

        # Save additional metadata
        metadata = {
            'documents': self.documents if save_embeddings else None,
            'topics': self.topics,
            'probabilities': self.probabilities,
            'is_fitted': self.is_fitted,
            'top_n_words': self.top_n_words
        }

        metadata_path = path.parent / f"{path.stem}_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)

        self.logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path], device: str = 'cpu') -> 'BERTopicModel':
        """
        Load a fitted model from disk.

        Args:
            path: Model path
            device: Device for embeddings

        Returns:
            Loaded BERTopicModel instance

        Examples:
            >>> model = BERTopicModel.load('models/topic_model.pkl')
            >>> new_topics, probs = model.transform(new_documents)
        """
        path = Path(path)

        # Load BERTopic model
        bertopic_model = BERTopic.load(str(path))

        # Create wrapper instance
        instance = cls.__new__(cls)
        instance.model = bertopic_model
        instance.logger = logging.getLogger(__name__)
        instance.device = device
        instance.verbose = False

        # Load metadata
        metadata_path = path.parent / f"{path.stem}_metadata.pkl"
        if metadata_path.exists():
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)

            instance.documents = metadata.get('documents')
            instance.topics = metadata.get('topics')
            instance.probabilities = metadata.get('probabilities')
            instance.is_fitted = metadata.get('is_fitted', True)
            instance.top_n_words = metadata.get('top_n_words', 10)
        else:
            instance.documents = None
            instance.topics = None
            instance.probabilities = None
            instance.is_fitted = True
            instance.top_n_words = 10

        instance.logger.info(f"Model loaded from {path}")
        return instance

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def get_document_topic(self, doc_idx: int) -> Tuple[int, float]:
        """
        Get topic assignment for a document.

        Args:
            doc_idx: Document index

        Returns:
            Tuple of (topic_id, probability)

        Examples:
            >>> topic_id, prob = model.get_document_topic(0)
            >>> print(f"Doc 0: Topic {topic_id} (prob={prob:.3f})")
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        topic = self.topics[doc_idx]
        prob = self.probabilities[doc_idx][topic + 1] if topic != -1 else 0.0

        return topic, prob

    def reduce_topics(self, nr_topics: int) -> 'BERTopicModel':
        """
        Reduce number of topics by merging similar ones.

        Args:
            nr_topics: Target number of topics

        Returns:
            Self for method chaining

        Examples:
            >>> model.fit(documents)
            >>> print(f"Original: {len(model.get_topics())} topics")
            >>> model.reduce_topics(10)
            >>> print(f"Reduced: {len(model.get_topics())} topics")
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        self.model.reduce_topics(self.documents, nr_topics=nr_topics)

        # Update topics and probabilities
        self.topics, self.probabilities = self.model.transform(self.documents)

        self.logger.info(f"Topics reduced to {nr_topics}")
        return self

    def find_topics(self, search_term: str, top_n: int = 5) -> List[Tuple[int, float]]:
        """
        Find topics related to a search term.

        Args:
            search_term: Term to search for
            top_n: Number of topics to return

        Returns:
            List of (topic_id, similarity_score) tuples

        Examples:
            >>> topics = model.find_topics("機器學習", top_n=3)
            >>> for topic_id, score in topics:
            ...     print(f"Topic {topic_id}: {score:.3f}")
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        similar_topics, similarities = self.model.find_topics(search_term, top_n=top_n)
        return list(zip(similar_topics, similarities))

    def __repr__(self) -> str:
        if self.is_fitted:
            n_topics = len(set(self.topics))
            n_docs = len(self.topics)
            return f"BERTopicModel(fitted=True, topics={n_topics}, documents={n_docs})"
        else:
            return "BERTopicModel(fitted=False)"


def demo():
    """Demonstration of BERTopicModel capabilities."""
    print("=" * 70)
    print("BERTopic Model Demo (Traditional Chinese)")
    print("=" * 70)

    # Sample Traditional Chinese documents (academic domain)
    documents = [
        "機器學習是人工智慧的重要分支，包括深度學習與強化學習",
        "資訊檢索系統需要使用倒排索引來提高查詢效率",
        "自然語言處理技術應用於文本分類與情感分析",
        "向量空間模型使用TF-IDF權重計算文檔相似度",
        "卷積神經網路在影像辨識領域取得突破性進展",
        "布林檢索模型支援AND OR NOT等邏輯運算子",
        "循環神經網路適合處理序列資料如文本與語音",
        "精確率召回率與F值是評估檢索系統的重要指標",
        "深度學習模型需要大量標註資料進行訓練",
        "倒排索引包含詞彙表與倒排列表兩個主要部分",
        "強化學習通過獎勵機制訓練智能代理",
        "餘弦相似度常用於計算文檔向量之間的相似性",
        "Transformer架構徹底改變了自然語言處理領域",
        "查詢擴展技術可以提高檢索系統的召回率",
        "生成對抗網路由生成器與判別器組成"
    ]

    # Initialize model
    print("\n[1] Initializing BERTopic Model...")
    print("-" * 70)
    model = BERTopicModel(
        language='multilingual',
        min_cluster_size=2,
        min_topic_size=2,
        top_n_words=5,
        verbose=True
    )
    print("Model initialized successfully")

    # Fit model
    print("\n[2] Fitting Model on Documents...")
    print("-" * 70)
    topics, probs = model.fit_transform(documents)
    print(f"Fitted on {len(documents)} documents")
    print(f"Found {len(set(topics))} topics")

    # Get topic info
    print("\n[3] Topic Information:")
    print("-" * 70)
    topic_info = model.get_topic_info()
    print(topic_info[['Topic', 'Count', 'Name']])

    # Show topic words
    print("\n[4] Topic Words:")
    print("-" * 70)
    topics_dict = model.get_topics()
    for topic_id in sorted(topics_dict.keys()):
        if topic_id == -1:
            continue
        words = model.get_topic_words(topic_id, top_n=5)
        words_str = ", ".join([w for w, s in words])
        print(f"Topic {topic_id}: {words_str}")

    # Get representative documents
    print("\n[5] Representative Documents:")
    print("-" * 70)
    for topic_id in range(min(3, len(topics_dict))):
        if topic_id == -1:
            continue
        rep_docs = model.get_representative_docs(topic_id, n_docs=2)
        print(f"\nTopic {topic_id}:")
        for i, doc in enumerate(rep_docs, 1):
            print(f"  {i}. {doc[:60]}...")

    # Document-topic assignments
    print("\n[6] Document-Topic Assignments:")
    print("-" * 70)
    for i in range(min(5, len(documents))):
        topic_id, prob = model.get_document_topic(i)
        print(f"Doc {i}: Topic {topic_id} (prob={prob:.3f})")
        print(f"       {documents[i][:60]}...")

    # Find topics
    print("\n[7] Finding Topics by Search Term:")
    print("-" * 70)
    search_results = model.find_topics("機器學習", top_n=3)
    print(f"Topics related to '機器學習':")
    for topic_id, score in search_results:
        print(f"  Topic {topic_id}: {score:.3f}")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    demo()
