"""
LDA (Latent Dirichlet Allocation) Model Wrapper for Traditional Chinese

This module provides a high-level wrapper for LDA topic modeling using Gensim,
optimized for Traditional Chinese documents.

LDA Overview:
    - Probabilistic generative model (Blei et al. 2003)
    - Documents as mixtures of topics
    - Topics as distributions over words
    - High interpretability
    - Efficient inference algorithms (Gibbs sampling, variational Bayes)

Key Features:
    - Traditional Chinese text preprocessing
    - Perplexity and coherence evaluation
    - Model persistence and reusability
    - Topic visualization support
    - Integration with ChineseTokenizer

Complexity:
    - Training: O(K*D*N*I) where K=topics, D=docs, N=words, I=iterations
    - Inference: O(K*N*I)
    - Space: O(K*V + D*K) where V=vocabulary

Reference:
    Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). "Latent Dirichlet
        Allocation". Journal of Machine Learning Research, 3, 993-1022.

Author: Information Retrieval System
License: Educational Use
"""

from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
import logging
import pickle

import numpy as np
import pandas as pd

# Gensim for LDA
from gensim import corpora, models
from gensim.models import CoherenceModel

# Chinese tokenizer
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
try:
    from src.ir.text.chinese_tokenizer import ChineseTokenizer
    from src.ir.text.stopwords import StopwordsFilter
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False


@dataclass
class LDATopicInfo:
    """
    LDA topic information container.

    Attributes:
        topic_id: Topic ID
        topic_words: List of (word, probability) tuples
        topic_proportion: Overall proportion in corpus
    """
    topic_id: int
    topic_words: List[Tuple[str, float]]
    topic_proportion: float

    def __str__(self) -> str:
        words = ", ".join([f"{w}({p:.3f})" for w, p in self.topic_words[:5]])
        return f"Topic {self.topic_id} ({self.topic_proportion:.1%}): {words}"


class LDAModel:
    """
    LDA topic modeling wrapper for Traditional Chinese.

    Provides high-level interface for training, inference, and analysis
    using Gensim's LDA implementation.

    Attributes:
        model: Gensim LDA model
        dictionary: Gensim dictionary (word -> ID mapping)
        corpus: Document-term matrix in bag-of-words format
        tokenizer: ChineseTokenizer instance
        stopwords: StopwordsFilter instance
        logger: Logger instance

    Examples:
        >>> # Initialize and train
        >>> lda = LDAModel(n_topics=10, tokenizer_engine='jieba')
        >>> lda.fit(documents)
        
        >>> # Get topics
        >>> topics = lda.get_topics()
        >>> for topic_id, words in topics.items():
        ...     print(f"Topic {topic_id}: {words[:5]}")
        
        >>> # Infer topic distribution for new document
        >>> topic_dist = lda.transform(["新文檔內容"])
    """

    def __init__(self,
                 n_topics: int = 10,
                 alpha: Union[str, float] = 'symmetric',
                 eta: Union[str, float] = 'auto',
                 iterations: int = 50,
                 passes: int = 10,
                 min_word_freq: int = 2,
                 max_word_freq: float = 0.5,
                 top_n_words: int = 10,
                 tokenizer_engine: str = 'jieba',
                 use_stopwords: bool = True,
                 random_state: int = 42):
        """
        Initialize LDA model.

        Args:
            n_topics: Number of topics
            alpha: Document-topic density ('symmetric', 'asymmetric', or float)
            eta: Topic-word density ('symmetric', 'auto', or float)
            iterations: Training iterations per document
            passes: Number of passes through corpus
            min_word_freq: Minimum word frequency (filter rare words)
            max_word_freq: Maximum word frequency proportion (filter common words)
            top_n_words: Number of words per topic representation
            tokenizer_engine: Chinese tokenizer ('jieba' or 'ckip')
            use_stopwords: Enable stopwords filtering
            random_state: Random seed for reproducibility

        Complexity:
            Time: O(1) - lazy initialization
            Space: O(V) where V=vocabulary size
        """
        self.logger = logging.getLogger(__name__)
        self.n_topics = n_topics
        self.alpha = alpha
        self.eta = eta
        self.iterations = iterations
        self.passes = passes
        self.min_word_freq = min_word_freq
        self.max_word_freq = max_word_freq
        self.top_n_words = top_n_words
        self.random_state = random_state

        # Initialize tokenizer
        if TOKENIZER_AVAILABLE:
            self.tokenizer = ChineseTokenizer(engine=tokenizer_engine)
            self.stopwords = StopwordsFilter() if use_stopwords else None
        else:
            self.logger.warning("ChineseTokenizer not available, using basic split")
            self.tokenizer = None
            self.stopwords = None

        # Model components (lazy initialization)
        self.model = None
        self.dictionary = None
        self.corpus = None
        self.documents = None
        self.processed_docs = None
        self.is_fitted = False

        self.logger.info(
            f"LDAModel initialized: n_topics={n_topics}, alpha={alpha}, "
            f"iterations={iterations}, passes={passes}"
        )

    # ========================================================================
    # Preprocessing
    # ========================================================================

    def _preprocess_documents(self, documents: List[str]) -> List[List[str]]:
        """
        Preprocess documents into tokenized word lists.

        Args:
            documents: List of document texts

        Returns:
            List of tokenized word lists

        Complexity:
            Time: O(D*N) where D=documents, N=avg length
            Space: O(D*N)
        """
        processed = []

        for doc in documents:
            # Tokenize
            if self.tokenizer:
                tokens = self.tokenizer.tokenize(doc)
            else:
                tokens = doc.split()

            # Remove stopwords
            if self.stopwords:
                tokens = [t for t in tokens if not self.stopwords.is_stopword(t)]

            # Filter short tokens
            tokens = [t for t in tokens if len(t) > 1]

            processed.append(tokens)

        self.logger.debug(f"Preprocessed {len(documents)} documents")
        return processed

    # ========================================================================
    # Training Methods
    # ========================================================================

    def fit(self, documents: List[str]) -> 'LDAModel':
        """
        Fit LDA model on documents.

        Args:
            documents: List of document texts

        Returns:
            Self for method chaining

        Complexity:
            Time: O(K*D*N*I) where K=topics, D=docs, N=words, I=iterations
            Space: O(K*V + D*K) where V=vocabulary

        Examples:
            >>> lda = LDAModel(n_topics=10)
            >>> lda.fit(documents)
            >>> print(f"Trained on {len(lda.dictionary)} unique words")
        """
        if not documents:
            raise ValueError("Documents list cannot be empty")

        self.logger.info(f"Fitting LDA on {len(documents)} documents")

        # Preprocess documents
        self.documents = documents
        self.processed_docs = self._preprocess_documents(documents)

        # Build dictionary
        self.dictionary = corpora.Dictionary(self.processed_docs)

        # Filter extremes
        self.dictionary.filter_extremes(
            no_below=self.min_word_freq,
            no_above=self.max_word_freq,
            keep_n=None
        )

        self.logger.info(f"Dictionary: {len(self.dictionary)} unique words")

        # Build corpus (bag-of-words)
        self.corpus = [self.dictionary.doc2bow(doc) for doc in self.processed_docs]

        # Train LDA model
        self.model = models.LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=self.n_topics,
            alpha=self.alpha,
            eta=self.eta,
            iterations=self.iterations,
            passes=self.passes,
            random_state=self.random_state,
            per_word_topics=True
        )

        self.is_fitted = True

        # Log perplexity
        perplexity = self.model.log_perplexity(self.corpus)
        self.logger.info(f"LDA trained: {self.n_topics} topics, perplexity={perplexity:.2f}")

        return self

    def transform(self, documents: List[str],
                 minimum_probability: float = 0.01) -> List[List[Tuple[int, float]]]:
        """
        Infer topic distributions for new documents.

        Args:
            documents: List of new document texts
            minimum_probability: Minimum topic probability to include

        Returns:
            List of topic distributions [(topic_id, probability), ...]

        Raises:
            ValueError: If model not fitted

        Complexity:
            Time: O(K*N*I) per document
            Space: O(K) per document

        Examples:
            >>> lda.fit(train_docs)
            >>> topic_dists = lda.transform(test_docs)
            >>> for dist in topic_dists:
            ...     print([(tid, prob) for tid, prob in dist])
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transform")

        # Preprocess new documents
        processed = self._preprocess_documents(documents)

        # Convert to bag-of-words
        bows = [self.dictionary.doc2bow(doc) for doc in processed]

        # Infer topics
        results = []
        for bow in bows:
            topic_dist = self.model.get_document_topics(
                bow,
                minimum_probability=minimum_probability
            )
            results.append(list(topic_dist))

        return results

    # ========================================================================
    # Topic Information Methods
    # ========================================================================

    def get_topics(self) -> Dict[int, List[Tuple[str, float]]]:
        """
        Get all topics with word probabilities.

        Returns:
            Dictionary mapping topic_id -> [(word, probability), ...]

        Examples:
            >>> topics = lda.get_topics()
            >>> for topic_id, words in topics.items():
            ...     print(f"Topic {topic_id}: {words[:3]}")
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        topics = {}
        for topic_id in range(self.n_topics):
            words = self.model.show_topic(topic_id, topn=self.top_n_words)
            topics[topic_id] = words

        return topics

    def get_topic_info(self, topic_id: Optional[int] = None) -> Union[pd.DataFrame, LDATopicInfo]:
        """
        Get topic information.

        Args:
            topic_id: Specific topic ID (None=all topics)

        Returns:
            DataFrame for all topics or LDATopicInfo for specific topic

        Examples:
            >>> # All topics
            >>> df = lda.get_topic_info()
            >>> print(df.head())
            
            >>> # Specific topic
            >>> info = lda.get_topic_info(topic_id=0)
            >>> print(info)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        if topic_id is None:
            # Return DataFrame for all topics
            data = []
            for tid in range(self.n_topics):
                words = self.model.show_topic(tid, topn=5)
                words_str = ", ".join([w for w, p in words])
                
                # Calculate topic proportion
                topic_counts = Counter()
                for doc_topics in self.model.get_document_topics(self.corpus):
                    for t, p in doc_topics:
                        topic_counts[t] += p
                
                total = sum(topic_counts.values())
                proportion = topic_counts[tid] / total if total > 0 else 0
                
                data.append({
                    'Topic': tid,
                    'Words': words_str,
                    'Proportion': proportion
                })
            
            return pd.DataFrame(data)
        else:
            # Return LDATopicInfo for specific topic
            if topic_id < 0 or topic_id >= self.n_topics:
                raise ValueError(f"Invalid topic_id: {topic_id}")
            
            words = self.model.show_topic(topic_id, topn=self.top_n_words)
            
            # Calculate proportion
            topic_counts = Counter()
            for doc_topics in self.model.get_document_topics(self.corpus):
                for t, p in doc_topics:
                    topic_counts[t] += p
            
            total = sum(topic_counts.values())
            proportion = topic_counts[topic_id] / total if total > 0 else 0
            
            return LDATopicInfo(
                topic_id=topic_id,
                topic_words=words,
                topic_proportion=proportion
            )

    def get_topic_words(self, topic_id: int, top_n: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        Get words for a specific topic.

        Args:
            topic_id: Topic ID
            top_n: Number of words (None=all)

        Returns:
            List of (word, probability) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        if topic_id < 0 or topic_id >= self.n_topics:
            raise ValueError(f"Invalid topic_id: {topic_id}")

        top_n = top_n or self.top_n_words
        return self.model.show_topic(topic_id, topn=top_n)

    # ========================================================================
    # Evaluation Methods
    # ========================================================================

    def calculate_perplexity(self, documents: Optional[List[str]] = None) -> float:
        """
        Calculate perplexity (lower is better).

        Args:
            documents: Test documents (None=use training corpus)

        Returns:
            Perplexity score

        Examples:
            >>> perplexity = lda.calculate_perplexity(test_docs)
            >>> print(f"Perplexity: {perplexity:.2f}")
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        if documents is None:
            return np.exp(-self.model.log_perplexity(self.corpus))
        else:
            processed = self._preprocess_documents(documents)
            corpus = [self.dictionary.doc2bow(doc) for doc in processed]
            return np.exp(-self.model.log_perplexity(corpus))

    def calculate_coherence(self, coherence_type: str = 'c_v') -> float:
        """
        Calculate topic coherence (higher is better).

        Args:
            coherence_type: Coherence measure ('c_v', 'u_mass', 'c_uci', 'c_npmi')
                - 'c_v': Best correlation with human judgment
                - 'u_mass': Based on document co-occurrence
                - 'c_uci': Pointwise mutual information
                - 'c_npmi': Normalized PMI

        Returns:
            Coherence score

        Examples:
            >>> coherence = lda.calculate_coherence(coherence_type='c_v')
            >>> print(f"Coherence (c_v): {coherence:.4f}")
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        coherence_model = CoherenceModel(
            model=self.model,
            texts=self.processed_docs,
            dictionary=self.dictionary,
            coherence=coherence_type
        )

        return coherence_model.get_coherence()

    # ========================================================================
    # Model Persistence
    # ========================================================================

    def save(self, path: Union[str, Path]):
        """
        Save the fitted model to disk.

        Args:
            path: Save directory path

        Examples:
            >>> lda.fit(documents)
            >>> lda.save('models/lda_model')
        """
        if not self.is_fitted:
            self.logger.warning("Saving unfitted model")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save Gensim model and dictionary
        self.model.save(str(path / 'lda_model'))
        self.dictionary.save(str(path / 'dictionary'))

        # Save additional metadata
        metadata = {
            'n_topics': self.n_topics,
            'alpha': self.alpha,
            'eta': self.eta,
            'top_n_words': self.top_n_words,
            'documents': self.documents,
            'processed_docs': self.processed_docs,
            'is_fitted': self.is_fitted
        }

        with open(path / 'metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)

        self.logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'LDAModel':
        """
        Load a fitted model from disk.

        Args:
            path: Model directory path

        Returns:
            Loaded LDAModel instance

        Examples:
            >>> lda = LDAModel.load('models/lda_model')
            >>> topics = lda.get_topics()
        """
        path = Path(path)

        # Load Gensim model and dictionary
        lda_model = models.LdaModel.load(str(path / 'lda_model'))
        dictionary = corpora.Dictionary.load(str(path / 'dictionary'))

        # Load metadata
        with open(path / 'metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)

        # Create instance
        instance = cls.__new__(cls)
        instance.logger = logging.getLogger(__name__)
        instance.model = lda_model
        instance.dictionary = dictionary
        instance.n_topics = metadata['n_topics']
        instance.alpha = metadata['alpha']
        instance.eta = metadata['eta']
        instance.top_n_words = metadata['top_n_words']
        instance.documents = metadata['documents']
        instance.processed_docs = metadata['processed_docs']
        instance.is_fitted = metadata['is_fitted']

        # Rebuild corpus
        if instance.processed_docs:
            instance.corpus = [dictionary.doc2bow(doc) for doc in instance.processed_docs]
        else:
            instance.corpus = None

        instance.logger.info(f"Model loaded from {path}")
        return instance

    def __repr__(self) -> str:
        if self.is_fitted:
            return f"LDAModel(fitted=True, topics={self.n_topics}, vocabulary={len(self.dictionary)})"
        else:
            return f"LDAModel(fitted=False, topics={self.n_topics})"


def demo():
    """Demonstration of LDAModel capabilities."""
    print("=" * 70)
    print("LDA Model Demo (Traditional Chinese)")
    print("=" * 70)

    # Sample documents (same as BERTopic demo)
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
        "倒排索引包含詞彙表與倒排列表兩個主要部分"
    ]

    # Initialize model
    print("\n[1] Initializing LDA Model...")
    print("-" * 70)
    lda = LDAModel(
        n_topics=3,
        iterations=50,
        passes=10,
        top_n_words=5,
        tokenizer_engine='jieba'
    )
    print("Model initialized")

    # Fit model
    print("\n[2] Fitting Model...")
    print("-" * 70)
    lda.fit(documents)
    print(f"Vocabulary: {len(lda.dictionary)} unique words")

    # Get topics
    print("\n[3] Topics:")
    print("-" * 70)
    topics = lda.get_topics()
    for topic_id, words in topics.items():
        words_str = ", ".join([f"{w}({p:.3f})" for w, p in words])
        print(f"Topic {topic_id}: {words_str}")

    # Topic info
    print("\n[4] Topic Information:")
    print("-" * 70)
    topic_info = lda.get_topic_info()
    print(topic_info)

    # Document topics
    print("\n[5] Document-Topic Distributions:")
    print("-" * 70)
    doc_topics = lda.transform(documents[:3])
    for i, dist in enumerate(doc_topics):
        print(f"Doc {i}: {dist}")

    # Evaluation
    print("\n[6] Evaluation Metrics:")
    print("-" * 70)
    perplexity = lda.calculate_perplexity()
    print(f"Perplexity: {perplexity:.2f}")
    
    coherence = lda.calculate_coherence('c_v')
    print(f"Coherence (c_v): {coherence:.4f}")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    demo()
