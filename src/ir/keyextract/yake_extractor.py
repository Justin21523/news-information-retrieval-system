"""
YAKE Keyword Extraction Wrapper

This module provides a wrapper for YAKE (Yet Another Keyword Extractor)
with support for Traditional Chinese text.

Key Features:
    - Statistical keyword extraction (no training required)
    - Fast processing: ~2000 documents/2 seconds
    - Multi-word keyphrase support
    - Chinese text support with custom tokenization
    - Unsupervised approach (no corpus needed)

Algorithm Overview:
    YAKE uses statistical features to score candidates:
    1. Term Frequency (TF)
    2. Casing (uppercase preference)
    3. Position (earlier = better)
    4. Term Relatedness to Context
    5. Term Different Sentence (dispersion)

    Final Score = lower is better (inverted from TextRank)

Complexity:
    - Time: O(n) where n = document length
    - Space: O(k) where k = number of candidates
    - Performance: 2000 docs/2s (very fast)

References:
    Campos et al. (2018). "YAKE! Collection-independent Automatic
        Keyword Extractor". ECIR 2018.
    Campos et al. (2020). "YAKE! Keyword Extraction from Single Documents
        using Multiple Local Features". Information Sciences.

Author: Information Retrieval System
License: Educational Use
"""

from typing import List, Optional
import logging
from pathlib import Path
from dataclasses import dataclass

# YAKE library
try:
    import yake
    YAKE_AVAILABLE = True
except ImportError:
    YAKE_AVAILABLE = False
    logging.warning("YAKE not available. Install with: pip install yake")

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
        score: YAKE score (lower = better, 0.0 to 1.0)
        positions: List of positions (not available in YAKE)
        frequency: Number of occurrences (estimated)
    """
    word: str
    score: float
    positions: List[int]
    frequency: int

    def __repr__(self):
        return f"Keyword(word='{self.word}', score={self.score:.4f}, freq={self.frequency})"


class YAKEExtractor:
    """
    YAKE keyword extractor with Chinese language support.

    Wraps the YAKE library with preprocessing for Traditional Chinese
    text using our unified tokenization interface.

    Attributes:
        language: Language code ('zh' for Chinese, 'en' for English)
        max_ngram_size: Maximum n-gram size (1-3 recommended)
        deduplication_threshold: Similarity threshold for deduplication
        deduplication_algo: Algorithm ('seqm' or 'levs')
        window_size: Co-occurrence window size
        num_keywords: Default number of keywords to extract

    Examples:
        >>> extractor = YAKEExtractor(language='zh')
        >>> text = "機器學習是人工智慧的重要分支，深度學習是機器學習的子領域。"
        >>> keywords = extractor.extract(text, top_k=5)
        >>> for kw in keywords:
        ...     print(f"{kw.word}: {kw.score:.4f}")
        機器學習: 0.0234
        深度學習: 0.0287
        人工智慧: 0.0356
    """

    def __init__(self,
                 language: str = 'zh',
                 max_ngram_size: int = 3,
                 deduplication_threshold: float = 0.9,
                 deduplication_algo: str = 'seqm',
                 window_size: int = 1,
                 num_keywords: int = 20,
                 tokenizer_engine: str = 'auto',
                 stopwords_file: Optional[str] = None,
                 custom_stopwords: Optional[List[str]] = None):
        """
        Initialize YAKE extractor.

        Args:
            language: Language code ('zh', 'en', 'pt', etc.)
            max_ngram_size: Maximum n-gram size (1=single words, 3=up to 3-grams)
            deduplication_threshold: Similarity threshold for removing duplicates (0.0-1.0)
            deduplication_algo: 'seqm' (sequence matcher) or 'levs' (Levenshtein)
            window_size: Co-occurrence window size (1-3 recommended)
            num_keywords: Default number of keywords to extract
            tokenizer_engine: 'ckip' | 'jieba' | 'auto' (for Chinese preprocessing)
            stopwords_file: Custom stopwords file path
            custom_stopwords: Additional stopwords to add

        Raises:
            ImportError: If YAKE is not installed
        """
        if not YAKE_AVAILABLE:
            raise ImportError(
                "YAKE is required. Install with: pip install yake"
            )

        self.logger = logging.getLogger(__name__)

        # YAKE parameters
        self.language = language
        self.max_ngram_size = max_ngram_size
        self.deduplication_threshold = deduplication_threshold
        self.deduplication_algo = deduplication_algo
        self.window_size = window_size
        self.num_keywords = num_keywords

        # Chinese preprocessing (if language is Chinese)
        if language == 'zh':
            self.tokenizer = ChineseTokenizer(engine=tokenizer_engine)
            self.stopwords_filter = StopwordsFilter(stopwords_file=stopwords_file)
            if custom_stopwords:
                self.stopwords_filter.add_stopwords(custom_stopwords)
        else:
            self.tokenizer = None
            self.stopwords_filter = None

        # Initialize YAKE extractor
        self._init_yake_extractor()

        self.logger.info(
            f"YAKEExtractor initialized: language={language}, "
            f"max_ngram={max_ngram_size}, window={window_size}, "
            f"dedup_threshold={deduplication_threshold}"
        )

    def _init_yake_extractor(self):
        """Initialize YAKE KeywordExtractor with current parameters."""
        self.extractor = yake.KeywordExtractor(
            lan=self.language,
            n=self.max_ngram_size,
            dedupLim=self.deduplication_threshold,
            dedupFunc=self.deduplication_algo,
            windowsSize=self.window_size,
            top=self.num_keywords
        )

    # ========================================================================
    # Core Extraction Methods
    # ========================================================================

    def extract(self,
                text: str,
                top_k: Optional[int] = None,
                preprocess: bool = True) -> List[Keyword]:
        """
        Extract keywords from text using YAKE.

        Args:
            text: Input text (Chinese or English)
            top_k: Number of keywords to extract (None = use default)
            preprocess: Apply Chinese preprocessing if enabled

        Returns:
            List of Keyword objects sorted by score (ascending, lower = better)

        Complexity:
            Time: O(n) where n = text length
            Space: O(k) where k = number of keywords

        Examples:
            >>> extractor = YAKEExtractor(language='zh')
            >>> text = "深度學習是機器學習的一個分支"
            >>> keywords = extractor.extract(text, top_k=3)
            >>> print([kw.word for kw in keywords])
            ['深度學習', '機器學習', '分支']
        """
        if not text or not text.strip():
            self.logger.warning("Empty text provided")
            return []

        # Preprocess Chinese text if needed
        if preprocess and self.language == 'zh' and self.tokenizer:
            # Tokenize with spaces for YAKE (expects space-separated input)
            tokens = self.tokenizer.tokenize(text)
            processed_text = ' '.join(tokens)
        else:
            processed_text = text

        # Extract keywords with YAKE
        if top_k is not None:
            # Temporarily override num_keywords
            original_top = self.num_keywords
            self.num_keywords = top_k
            self._init_yake_extractor()
            yake_results = self.extractor.extract_keywords(processed_text)
            self.num_keywords = original_top
            self._init_yake_extractor()
        else:
            yake_results = self.extractor.extract_keywords(processed_text)

        # Convert YAKE results to our Keyword format
        keywords = []
        for keyword_text, score in yake_results:
            # For Chinese, remove spaces that we added
            if self.language == 'zh':
                keyword_text = keyword_text.replace(' ', '')

            # Estimate frequency (YAKE doesn't provide this directly)
            # Count occurrences in original text
            frequency = text.count(keyword_text)

            keywords.append(Keyword(
                word=keyword_text,
                score=score,
                positions=[],  # YAKE doesn't track positions
                frequency=frequency
            ))

        self.logger.info(
            f"Extracted {len(keywords)} keywords using YAKE"
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
            >>> extractor = YAKEExtractor(language='zh')
            >>> docs = [
            ...     "機器學習是人工智慧的分支",
            ...     "深度學習使用神經網路"
            ... ]
            >>> results = extractor.extract_from_documents(docs, top_k=3)
            >>> for i, keywords in enumerate(results):
            ...     print(f"Doc {i}: {[kw.word for kw in keywords]}")
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

    def set_parameters(self,
                      max_ngram_size: Optional[int] = None,
                      deduplication_threshold: Optional[float] = None,
                      window_size: Optional[int] = None,
                      num_keywords: Optional[int] = None):
        """
        Update YAKE parameters and reinitialize extractor.

        Args:
            max_ngram_size: New max n-gram size
            deduplication_threshold: New deduplication threshold
            window_size: New window size
            num_keywords: New default number of keywords
        """
        if max_ngram_size is not None:
            self.max_ngram_size = max_ngram_size
        if deduplication_threshold is not None:
            self.deduplication_threshold = deduplication_threshold
        if window_size is not None:
            self.window_size = window_size
        if num_keywords is not None:
            self.num_keywords = num_keywords

        # Reinitialize with new parameters
        self._init_yake_extractor()

        self.logger.info("YAKE parameters updated and extractor reinitialized")

    def get_config(self) -> dict:
        """Get current configuration."""
        return {
            'language': self.language,
            'max_ngram_size': self.max_ngram_size,
            'deduplication_threshold': self.deduplication_threshold,
            'deduplication_algo': self.deduplication_algo,
            'window_size': self.window_size,
            'num_keywords': self.num_keywords,
            'tokenizer_engine': self.tokenizer.engine if self.tokenizer else None,
            'stopwords_count': len(self.stopwords_filter) if self.stopwords_filter else 0
        }


def demo():
    """Demonstration of YAKE keyword extraction."""
    print("=" * 70)
    print("YAKE Keyword Extraction Demo")
    print("=" * 70)

    # Sample Traditional Chinese text
    text = """
    機器學習是人工智慧的重要分支，它讓電腦能夠從資料中學習模式。
    深度學習是機器學習的子領域，使用神經網路來建立複雜的模型。
    自然語言處理是人工智慧的另一個重要應用，涉及文字分析和理解。
    資訊檢索系統使用機器學習技術來改善搜尋結果的品質。
    """

    # Initialize extractor
    print("\n[1] Initialize YAKE Extractor (Chinese)")
    print("-" * 70)
    extractor = YAKEExtractor(
        language='zh',
        max_ngram_size=3,
        num_keywords=20,
        tokenizer_engine='jieba'  # Fast for demo
    )
    print(f"Config: {extractor.get_config()}")

    # Extract keywords (single words and phrases)
    print("\n[2] Extract Keywords (Top 10)")
    print("-" * 70)
    keywords = extractor.extract(text, top_k=10)

    for i, kw in enumerate(keywords, 1):
        print(f"{i:2d}. {kw.word:20s}  score={kw.score:.4f}  freq={kw.frequency}")

    # Extract only single words
    print("\n[3] Extract Single Words Only (Top 5)")
    print("-" * 70)
    extractor_1gram = YAKEExtractor(
        language='zh',
        max_ngram_size=1,
        tokenizer_engine='jieba'
    )
    keywords_1gram = extractor_1gram.extract(text, top_k=5)

    for i, kw in enumerate(keywords_1gram, 1):
        print(f"{i}. {kw.word:12s}  score={kw.score:.4f}")

    # Extract multi-word phrases
    print("\n[4] Extract Multi-word Phrases (2-3 grams)")
    print("-" * 70)
    multi_word = [kw for kw in keywords if len(kw.word) >= 4][:5]

    for i, kw in enumerate(multi_word, 1):
        print(f"{i}. {kw.word:20s}  score={kw.score:.4f}")

    # Batch processing
    print("\n[5] Batch Processing (3 Documents)")
    print("-" * 70)
    documents = [
        "機器學習使用統計方法從資料中學習",
        "深度學習使用多層神經網路",
        "自然語言處理分析文字語義"
    ]

    batch_results = extractor.extract_from_documents(documents, top_k=3)
    for i, keywords in enumerate(batch_results):
        print(f"\nDoc {i+1}: {documents[i][:20]}...")
        for kw in keywords:
            print(f"  - {kw.word:15s}  score={kw.score:.4f}")

    # English example
    print("\n[6] English Text Example")
    print("-" * 70)
    extractor_en = YAKEExtractor(language='en', max_ngram_size=2)
    text_en = """
    Machine learning is a branch of artificial intelligence.
    Deep learning uses neural networks to build complex models.
    Natural language processing analyzes and understands text.
    """

    keywords_en = extractor_en.extract(text_en, top_k=5, preprocess=False)
    for i, kw in enumerate(keywords_en, 1):
        print(f"{i}. {kw.word:25s}  score={kw.score:.4f}")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    demo()
