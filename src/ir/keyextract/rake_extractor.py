"""
RAKE Keyword Extraction Wrapper

This module provides a wrapper for RAKE (Rapid Automatic Keyword Extraction)
with support for Traditional Chinese text.

Key Features:
    - Domain-independent keyword extraction
    - Fast processing with simple statistics
    - Multi-word keyphrase extraction
    - Chinese text support with custom tokenization
    - Uses stopwords as phrase delimiters

Algorithm Overview:
    1. Split text into candidate phrases using stopwords as delimiters
    2. Calculate word scores:
       - degree(word) = sum of co-occurrences with other words
       - freq(word) = number of occurrences
       - score(word) = degree(word) / freq(word)
    3. Score phrases as sum of word scores
    4. Return top-k phrases

Complexity:
    - Time: O(n) where n = number of words
    - Space: O(w) where w = unique words
    - Very fast for single documents

References:
    Rose et al. (2010). "Automatic Keyword Extraction from Individual
        Documents". Text Mining: Applications and Theory.

Author: Information Retrieval System
License: Educational Use
"""

from typing import List, Optional
import logging
from pathlib import Path
from dataclasses import dataclass

# RAKE library
try:
    from rake_nltk import Rake
    RAKE_AVAILABLE = True
except ImportError:
    RAKE_AVAILABLE = False
    logging.warning("RAKE not available. Install with: pip install rake-nltk")

# Import our Chinese processing modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ir.text.chinese_tokenizer import ChineseTokenizer
from ir.text.stopwords import StopwordsFilter


@dataclass
class Keyword:
    """
    Keyword extraction result.

    Attributes:
        word: The keyword or keyphrase
        score: RAKE score (higher = better)
        positions: List of positions (not available in RAKE)
        frequency: Number of occurrences
    """
    word: str
    score: float
    positions: List[int]
    frequency: int

    def __repr__(self):
        return f"Keyword(word='{self.word}', score={self.score:.4f}, freq={self.frequency})"


class RAKEExtractor:
    """
    RAKE keyword extractor with Chinese language support.

    Wraps the RAKE library with preprocessing for Traditional Chinese
    text using our unified tokenization interface.

    Attributes:
        min_length: Minimum characters per keyphrase
        max_length: Maximum words per keyphrase
        ranking_metric: Metric for ranking ('degree_to_frequency', 'word_degree', 'word_frequency')
        include_repeated_phrases: Include phrases that appear multiple times

    Examples:
        >>> extractor = RAKEExtractor()
        >>> text = "機器學習是人工智慧的重要分支，深度學習是機器學習的子領域。"
        >>> keywords = extractor.extract(text, top_k=5)
        >>> for kw in keywords:
        ...     print(f"{kw.word}: {kw.score:.4f}")
        機器學習: 8.5000
        人工智慧: 4.0000
        深度學習: 4.0000
    """

    def __init__(self,
                 min_length: int = 1,
                 max_length: int = 4,
                 ranking_metric: str = 'degree_to_frequency',
                 include_repeated_phrases: bool = True,
                 tokenizer_engine: str = 'auto',
                 stopwords_file: Optional[str] = None):
        """
        Initialize RAKE extractor.

        Args:
            min_length: Minimum characters per keyphrase (1+ recommended)
            max_length: Maximum words per keyphrase (3-5 recommended)
            ranking_metric: Scoring metric
                - 'degree_to_frequency': deg(w) / freq(w) (default, balanced)
                - 'word_degree': deg(w) (favors long phrases)
                - 'word_frequency': freq(w) (favors frequent words)
            include_repeated_phrases: Include phrases appearing multiple times
            tokenizer_engine: 'ckip' | 'jieba' | 'auto' (for Chinese)
            stopwords_file: Custom stopwords file path

        Raises:
            ImportError: If RAKE is not installed
        """
        if not RAKE_AVAILABLE:
            raise ImportError(
                "RAKE is required. Install with: pip install rake-nltk"
            )

        self.logger = logging.getLogger(__name__)

        # RAKE parameters
        self.min_length = min_length
        self.max_length = max_length
        self.ranking_metric = ranking_metric
        self.include_repeated_phrases = include_repeated_phrases

        # Chinese preprocessing
        self.tokenizer = ChineseTokenizer(engine=tokenizer_engine)
        self.stopwords_filter = StopwordsFilter(stopwords_file=stopwords_file)

        # Get stopwords list for RAKE
        stopwords_list = list(self.stopwords_filter.stopwords)

        # Initialize RAKE
        # Note: RAKE-NLTK uses Metric enum
        from rake_nltk import Metric

        metric_map = {
            'degree_to_frequency': Metric.DEGREE_TO_FREQUENCY_RATIO,
            'word_degree': Metric.WORD_DEGREE,
            'word_frequency': Metric.WORD_FREQUENCY
        }

        self.rake = Rake(
            stopwords=stopwords_list,
            min_length=min_length,
            max_length=max_length,
            ranking_metric=metric_map.get(ranking_metric, Metric.DEGREE_TO_FREQUENCY_RATIO),
            include_repeated_phrases=include_repeated_phrases
        )

        self.logger.info(
            f"RAKEExtractor initialized: min_length={min_length}, "
            f"max_length={max_length}, metric={ranking_metric}, "
            f"tokenizer={tokenizer_engine}"
        )

    # ========================================================================
    # Core Extraction Methods
    # ========================================================================

    def extract(self,
                text: str,
                top_k: int = 10,
                preprocess: bool = True) -> List[Keyword]:
        """
        Extract keywords from text using RAKE.

        Args:
            text: Input text (Chinese or English)
            top_k: Number of keywords to extract
            preprocess: Apply Chinese preprocessing if enabled

        Returns:
            List of Keyword objects sorted by score (descending)

        Complexity:
            Time: O(n) where n = number of words
            Space: O(w) where w = unique words

        Examples:
            >>> extractor = RAKEExtractor()
            >>> text = "深度學習是機器學習的一個分支"
            >>> keywords = extractor.extract(text, top_k=3)
            >>> print([kw.word for kw in keywords])
            ['機器學習分支', '深度學習', ...]
        """
        if not text or not text.strip():
            self.logger.warning("Empty text provided")
            return []

        # Preprocess Chinese text
        if preprocess:
            # Tokenize with spaces for RAKE (expects space-separated input)
            tokens = self.tokenizer.tokenize(text)
            processed_text = ' '.join(tokens)
        else:
            processed_text = text

        # Extract keywords with RAKE
        self.rake.extract_keywords_from_text(processed_text)

        # Get ranked phrases with scores
        ranked_phrases = self.rake.get_ranked_phrases_with_scores()

        # Convert to our Keyword format
        keywords = []
        for score, phrase in ranked_phrases[:top_k]:
            # For Chinese, remove spaces that we added
            if preprocess:
                phrase = phrase.replace(' ', '')

            # Count occurrences in original text
            frequency = text.count(phrase)

            keywords.append(Keyword(
                word=phrase,
                score=score,
                positions=[],  # RAKE doesn't track positions
                frequency=frequency
            ))

        self.logger.info(
            f"Extracted {len(keywords)} keywords using RAKE "
            f"(from {len(ranked_phrases)} candidates)"
        )

        return keywords

    def extract_from_documents(self,
                               documents: List[str],
                               top_k: int = 10) -> List[List[Keyword]]:
        """
        Extract keywords from multiple documents.

        Args:
            documents: List of text documents
            top_k: Number of keywords per document

        Returns:
            List of keyword lists (one per document)

        Complexity:
            Time: O(N×n) where N = documents, n = avg doc length
            Space: O(N×k)

        Examples:
            >>> extractor = RAKEExtractor()
            >>> docs = [
            ...     "機器學習是人工智慧的分支",
            ...     "深度學習使用神經網路"
            ... ]
            >>> results = extractor.extract_from_documents(docs, top_k=3)
        """
        results = []
        for i, doc in enumerate(documents):
            keywords = self.extract(doc, top_k=top_k)
            results.append(keywords)

            self.logger.debug(
                f"Document {i+1}/{len(documents)}: "
                f"Extracted {len(keywords)} keywords"
            )

        return results

    # ========================================================================
    # Configuration
    # ========================================================================

    def get_config(self) -> dict:
        """Get current configuration."""
        return {
            'min_length': self.min_length,
            'max_length': self.max_length,
            'ranking_metric': self.ranking_metric,
            'include_repeated_phrases': self.include_repeated_phrases,
            'tokenizer_engine': self.tokenizer.engine,
            'stopwords_count': len(self.stopwords_filter)
        }


def demo():
    """Demonstration of RAKE keyword extraction."""
    print("=" * 70)
    print("RAKE Keyword Extraction Demo")
    print("=" * 70)

    # Sample Traditional Chinese text
    text = """
    機器學習是人工智慧的重要分支，它讓電腦能夠從資料中學習模式。
    深度學習是機器學習的子領域，使用神經網路來建立複雜的模型。
    自然語言處理是人工智慧的另一個重要應用，涉及文字分析和理解。
    資訊檢索系統使用機器學習技術來改善搜尋結果的品質。
    """

    # Initialize extractor
    print("\n[1] Initialize RAKE Extractor")
    print("-" * 70)
    extractor = RAKEExtractor(
        min_length=1,
        max_length=4,
        ranking_metric='degree_to_frequency',
        tokenizer_engine='jieba'  # Fast for demo
    )
    print(f"Config: {extractor.get_config()}")

    # Extract keywords
    print("\n[2] Extract Keywords (Top 10)")
    print("-" * 70)
    keywords = extractor.extract(text, top_k=10)

    for i, kw in enumerate(keywords, 1):
        print(f"{i:2d}. {kw.word:25s}  score={kw.score:.4f}  freq={kw.frequency}")

    # Compare ranking metrics
    print("\n[3] Ranking Metric Comparison")
    print("-" * 70)

    metrics = ['degree_to_frequency', 'word_degree', 'word_frequency']
    for metric in metrics:
        extractor_metric = RAKEExtractor(
            max_length=3,
            ranking_metric=metric,
            tokenizer_engine='jieba'
        )
        keywords_metric = extractor_metric.extract(text, top_k=3)

        print(f"\n{metric}:")
        for kw in keywords_metric:
            print(f"  {kw.word:20s}  score={kw.score:.4f}")

    # Different max_length settings
    print("\n[4] Max Length Comparison")
    print("-" * 70)

    for max_len in [2, 3, 5]:
        extractor_len = RAKEExtractor(
            max_length=max_len,
            tokenizer_engine='jieba'
        )
        keywords_len = extractor_len.extract(text, top_k=3)

        print(f"\nmax_length={max_len}:")
        for kw in keywords_len:
            print(f"  {kw.word:30s}  score={kw.score:.4f}")

    # Batch processing
    print("\n[5] Batch Processing (3 Documents)")
    print("-" * 70)
    documents = [
        "機器學習使用統計方法從資料中學習規律和模式",
        "深度學習使用多層神經網路處理複雜的資料結構",
        "自然語言處理系統分析和理解人類語言的語義"
    ]

    batch_results = extractor.extract_from_documents(documents, top_k=3)
    for i, keywords in enumerate(batch_results):
        print(f"\nDoc {i+1}: {documents[i][:25]}...")
        for kw in keywords:
            print(f"  - {kw.word:20s}  score={kw.score:.4f}")

    # English example
    print("\n[6] English Text Example")
    print("-" * 70)
    extractor_en = RAKEExtractor(max_length=3)
    text_en = """
    Machine learning is a branch of artificial intelligence that enables
    computers to learn from data. Deep learning uses neural networks to
    build complex models. Natural language processing analyzes text.
    """

    keywords_en = extractor_en.extract(text_en, top_k=5, preprocess=False)
    for i, kw in enumerate(keywords_en, 1):
        print(f"{i}. {kw.word:30s}  score={kw.score:.4f}")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    demo()
