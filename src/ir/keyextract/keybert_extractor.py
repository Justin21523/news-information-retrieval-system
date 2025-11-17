"""
KeyBERT Keyword Extraction Wrapper

This module provides a wrapper for KeyBERT with support for Traditional Chinese
text using multilingual and Chinese-specific BERT models.

Key Features:
    - BERT embeddings for semantic keyword extraction
    - Maximal Marginal Relevance (MMR) for diversity
    - Support for Chinese BERT models (multilingual, Chinese-specific)
    - N-gram keyphrase extraction
    - Batch processing with GPU acceleration
    - Document-level and corpus-level extraction

Algorithm Overview:
    1. Encode document with BERT to get document embedding
    2. Extract candidate keywords/keyphrases (n-grams)
    3. Encode candidates with BERT
    4. Compute cosine similarity between document and candidates
    5. Apply MMR to balance relevance and diversity

MMR Formula:
    MMR = λ × Sim(candidate, document) - (1-λ) × max Sim(candidate, selected)
    where λ controls relevance vs. diversity tradeoff

Complexity:
    - Time: O(n×d + V×d) where n=text length, V=candidates, d=embedding dim
    - Space: O(V×d + D×d) for embeddings
    - GPU acceleration available

References:
    Grootendorst (2020). "KeyBERT: Minimal keyword extraction with BERT"
    Carbonell & Goldstein (1998). "The use of MMR, diversity-based reranking..."
    Devlin et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers"

Author: Information Retrieval System
License: Educational Use
"""

from typing import List, Optional, Tuple, Union
import logging
from pathlib import Path
from dataclasses import dataclass

# KeyBERT library
try:
    from keybert import KeyBERT
    KEYBERT_AVAILABLE = True
except ImportError:
    KEYBERT_AVAILABLE = False
    logging.warning("KeyBERT not available. Install with: pip install keybert")

# Sentence Transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning(
        "SentenceTransformers not available. "
        "Install with: pip install sentence-transformers"
    )

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
        score: Similarity score (0.0 to 1.0, higher = better)
        positions: List of positions (not tracked in KeyBERT)
        frequency: Number of occurrences (estimated)
    """
    word: str
    score: float
    positions: List[int]
    frequency: int

    def __repr__(self):
        return f"Keyword(word='{self.word}', score={self.score:.4f}, freq={self.frequency})"


class KeyBERTExtractor:
    """
    KeyBERT keyword extractor with Chinese language support.

    Uses BERT embeddings and MMR for extracting semantically relevant
    and diverse keywords from text.

    Attributes:
        model_name: BERT model name or path
        use_mmr: Enable Maximal Marginal Relevance for diversity
        diversity: MMR diversity parameter (0.0 to 1.0)
        top_n: Default number of keywords to extract
        keyphrase_ngram_range: N-gram range for keyphrases

    Recommended Chinese Models:
        - 'paraphrase-multilingual-MiniLM-L12-v2': Fast, multilingual (default)
        - 'distiluse-base-multilingual-cased-v1': Good quality, 50+ languages
        - 'bert-base-chinese': Chinese-specific BERT
        - 'hfl/chinese-bert-wwm-ext': Whole Word Masking Chinese BERT

    Examples:
        >>> extractor = KeyBERTExtractor()
        >>> text = "機器學習是人工智慧的重要分支，深度學習是機器學習的子領域。"
        >>> keywords = extractor.extract(text, top_k=5)
        >>> for kw in keywords:
        ...     print(f"{kw.word}: {kw.score:.4f}")
        機器學習: 0.6521
        深度學習: 0.5873
        人工智慧: 0.5234
    """

    def __init__(self,
                 model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2',
                 use_mmr: bool = True,
                 diversity: float = 0.5,
                 top_n: int = 10,
                 keyphrase_ngram_range: Tuple[int, int] = (1, 3),
                 tokenizer_engine: str = 'auto',
                 stopwords_file: Optional[str] = None,
                 device: str = 'cpu'):
        """
        Initialize KeyBERT extractor.

        Args:
            model_name: BERT model name from HuggingFace or local path
            use_mmr: Use Maximal Marginal Relevance for diversity
            diversity: MMR diversity parameter (0.0=relevance only, 1.0=diversity only)
            top_n: Default number of keywords to extract
            keyphrase_ngram_range: N-gram range (min, max) for keyphrases
            tokenizer_engine: 'ckip' | 'jieba' | 'auto' (for Chinese preprocessing)
            stopwords_file: Custom stopwords file path
            device: 'cpu' or 'cuda' for GPU acceleration

        Raises:
            ImportError: If KeyBERT or SentenceTransformers not installed
        """
        if not KEYBERT_AVAILABLE:
            raise ImportError(
                "KeyBERT is required. Install with: pip install keybert"
            )

        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "SentenceTransformers is required. "
                "Install with: pip install sentence-transformers"
            )

        self.logger = logging.getLogger(__name__)

        # KeyBERT parameters
        self.model_name = model_name
        self.use_mmr = use_mmr
        self.diversity = diversity
        self.top_n = top_n
        self.keyphrase_ngram_range = keyphrase_ngram_range
        self.device = device

        # Chinese preprocessing
        self.tokenizer = ChineseTokenizer(engine=tokenizer_engine)
        self.stopwords_filter = StopwordsFilter(stopwords_file=stopwords_file)

        # Initialize BERT model
        try:
            self.logger.info(f"Loading BERT model: {model_name}")
            self.sentence_model = SentenceTransformer(model_name, device=device)
            self.keybert_model = KeyBERT(model=self.sentence_model)
            self.logger.info(f"BERT model loaded successfully on {device}")
        except Exception as e:
            self.logger.error(f"Failed to load BERT model: {e}")
            raise

        self.logger.info(
            f"KeyBERTExtractor initialized: model={model_name}, "
            f"use_mmr={use_mmr}, diversity={diversity}, "
            f"ngram_range={keyphrase_ngram_range}, device={device}"
        )

    # ========================================================================
    # Core Extraction Methods
    # ========================================================================

    def extract(self,
                text: str,
                top_k: Optional[int] = None,
                use_mmr: Optional[bool] = None,
                diversity: Optional[float] = None,
                preprocess: bool = True) -> List[Keyword]:
        """
        Extract keywords from text using KeyBERT.

        Args:
            text: Input text (Chinese or English)
            top_k: Number of keywords to extract (None = use default)
            use_mmr: Use MMR (None = use default setting)
            diversity: MMR diversity (None = use default)
            preprocess: Apply Chinese preprocessing if enabled

        Returns:
            List of Keyword objects sorted by score (descending)

        Complexity:
            Time: O(n×d + V×d) where n=text, V=candidates, d=embedding dim
            Space: O(V×d + D×d)

        Examples:
            >>> extractor = KeyBERTExtractor()
            >>> text = "深度學習是機器學習的一個分支"
            >>> keywords = extractor.extract(text, top_k=3)
            >>> print([kw.word for kw in keywords])
            ['深度學習', '機器學習', '分支']
        """
        if not text or not text.strip():
            self.logger.warning("Empty text provided")
            return []

        # Use default parameters if not specified
        if top_k is None:
            top_k = self.top_n
        if use_mmr is None:
            use_mmr = self.use_mmr
        if diversity is None:
            diversity = self.diversity

        # Preprocess Chinese text if needed
        if preprocess:
            # For KeyBERT, we keep the text as-is but use stopwords
            # KeyBERT handles tokenization internally
            processed_text = text
            stop_words = list(self.stopwords_filter.stopwords)
        else:
            processed_text = text
            stop_words = None

        # Extract keywords with KeyBERT
        try:
            if use_mmr:
                # Use MMR for diversity
                keybert_results = self.keybert_model.extract_keywords(
                    processed_text,
                    keyphrase_ngram_range=self.keyphrase_ngram_range,
                    stop_words=stop_words,
                    top_n=top_k,
                    use_mmr=True,
                    diversity=diversity
                )
            else:
                # Use cosine similarity only (no diversity)
                keybert_results = self.keybert_model.extract_keywords(
                    processed_text,
                    keyphrase_ngram_range=self.keyphrase_ngram_range,
                    stop_words=stop_words,
                    top_n=top_k,
                    use_mmr=False
                )
        except Exception as e:
            self.logger.error(f"KeyBERT extraction failed: {e}")
            return []

        # Convert to our Keyword format
        keywords = []
        for keyword_text, score in keybert_results:
            # Count occurrences in original text
            frequency = text.count(keyword_text)

            keywords.append(Keyword(
                word=keyword_text,
                score=float(score),
                positions=[],  # KeyBERT doesn't track positions
                frequency=frequency
            ))

        self.logger.info(
            f"Extracted {len(keywords)} keywords using KeyBERT "
            f"(MMR={use_mmr}, diversity={diversity:.2f})"
        )

        return keywords

    def extract_from_documents(self,
                               documents: List[str],
                               top_k: int = 10,
                               use_mmr: bool = True) -> List[List[Keyword]]:
        """
        Extract keywords from multiple documents.

        Args:
            documents: List of text documents
            top_k: Number of keywords per document
            use_mmr: Use MMR for diversity

        Returns:
            List of keyword lists (one per document)

        Complexity:
            Time: O(N×(n×d + V×d)) where N = documents
            Space: O(N×V×d)

        Examples:
            >>> extractor = KeyBERTExtractor()
            >>> docs = [
            ...     "機器學習是人工智慧的分支",
            ...     "深度學習使用神經網路"
            ... ]
            >>> results = extractor.extract_from_documents(docs, top_k=3)
        """
        results = []
        for i, doc in enumerate(documents):
            keywords = self.extract(doc, top_k=top_k, use_mmr=use_mmr)
            results.append(keywords)

            self.logger.debug(
                f"Document {i+1}/{len(documents)}: "
                f"Extracted {len(keywords)} keywords"
            )

        return results

    def extract_with_embeddings(self,
                                text: str,
                                top_k: int = 10) -> Tuple[List[Keyword], any, any]:
        """
        Extract keywords and return embeddings for analysis.

        Args:
            text: Input text
            top_k: Number of keywords to extract

        Returns:
            (keywords, doc_embedding, candidate_embeddings)

        Use case:
            - Analyzing semantic similarity
            - Visualizing keyword space
            - Custom ranking algorithms
        """
        keywords = self.extract(text, top_k=top_k)

        # Get document embedding
        doc_embedding = self.sentence_model.encode([text])[0]

        # Get candidate embeddings
        candidate_texts = [kw.word for kw in keywords]
        candidate_embeddings = self.sentence_model.encode(candidate_texts)

        return keywords, doc_embedding, candidate_embeddings

    # ========================================================================
    # Configuration
    # ========================================================================

    def set_parameters(self,
                      use_mmr: Optional[bool] = None,
                      diversity: Optional[float] = None,
                      top_n: Optional[int] = None,
                      keyphrase_ngram_range: Optional[Tuple[int, int]] = None):
        """
        Update KeyBERT parameters.

        Args:
            use_mmr: Enable/disable MMR
            diversity: New diversity parameter
            top_n: New default top_n
            keyphrase_ngram_range: New n-gram range
        """
        if use_mmr is not None:
            self.use_mmr = use_mmr
        if diversity is not None:
            self.diversity = diversity
        if top_n is not None:
            self.top_n = top_n
        if keyphrase_ngram_range is not None:
            self.keyphrase_ngram_range = keyphrase_ngram_range

        self.logger.info("KeyBERT parameters updated")

    def get_config(self) -> dict:
        """Get current configuration."""
        return {
            'model_name': self.model_name,
            'use_mmr': self.use_mmr,
            'diversity': self.diversity,
            'top_n': self.top_n,
            'keyphrase_ngram_range': self.keyphrase_ngram_range,
            'tokenizer_engine': self.tokenizer.engine,
            'stopwords_count': len(self.stopwords_filter),
            'device': self.device
        }


def demo():
    """Demonstration of KeyBERT keyword extraction."""
    print("=" * 70)
    print("KeyBERT Keyword Extraction Demo")
    print("=" * 70)

    # Sample Traditional Chinese text
    text = """
    機器學習是人工智慧的重要分支，它讓電腦能夠從資料中學習模式。
    深度學習是機器學習的子領域，使用神經網路來建立複雜的模型。
    自然語言處理是人工智慧的另一個重要應用，涉及文字分析和理解。
    資訊檢索系統使用機器學習技術來改善搜尋結果的品質。
    """

    # Initialize extractor
    print("\n[1] Initialize KeyBERT Extractor")
    print("-" * 70)
    print("Loading multilingual BERT model (this may take a minute)...")

    try:
        extractor = KeyBERTExtractor(
            model_name='paraphrase-multilingual-MiniLM-L12-v2',
            use_mmr=True,
            diversity=0.5,
            tokenizer_engine='jieba',  # Fast for demo
            device='cpu'
        )
        print(f"Config: {extractor.get_config()}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please install: pip install keybert sentence-transformers")
        return

    # Extract keywords with MMR
    print("\n[2] Extract Keywords with MMR (Top 10)")
    print("-" * 70)
    keywords_mmr = extractor.extract(text, top_k=10, use_mmr=True, diversity=0.5)

    for i, kw in enumerate(keywords_mmr, 1):
        print(f"{i:2d}. {kw.word:20s}  score={kw.score:.4f}  freq={kw.frequency}")

    # Extract without MMR (pure cosine similarity)
    print("\n[3] Extract Keywords without MMR (Pure Similarity)")
    print("-" * 70)
    keywords_no_mmr = extractor.extract(text, top_k=5, use_mmr=False)

    for i, kw in enumerate(keywords_no_mmr, 1):
        print(f"{i}. {kw.word:20s}  score={kw.score:.4f}")

    # Compare diversity settings
    print("\n[4] MMR Diversity Comparison")
    print("-" * 70)

    for div in [0.2, 0.5, 0.8]:
        keywords_div = extractor.extract(text, top_k=3, use_mmr=True, diversity=div)
        print(f"\ndiversity={div}:")
        for kw in keywords_div:
            print(f"  {kw.word:20s}  score={kw.score:.4f}")

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
    text_en = """
    Machine learning is a branch of artificial intelligence that enables
    computers to learn from data without being explicitly programmed.
    Deep learning is a subset of machine learning that uses neural networks
    with multiple layers to process complex patterns in large datasets.
    """

    keywords_en = extractor.extract(text_en, top_k=5, preprocess=False)
    for i, kw in enumerate(keywords_en, 1):
        print(f"{i}. {kw.word:30s}  score={kw.score:.4f}")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    demo()
