"""
CKIP Transformers Tokenizer

Chinese word segmentation using CKIP Transformers.
Provides efficient CPU-based tokenization for IR systems.

Key Features:
    - CKIP Transformers for accurate Chinese segmentation
    - CPU-only mode for efficient processing
    - Singleton pattern for model reuse
    - Batch processing support
    - Stopword filtering

Author: Information Retrieval System
License: Educational Use
"""

import logging
from typing import List, Optional, Set
from pathlib import Path


class CKIPTokenizer:
    """
    CKIP Transformers tokenizer for Chinese text.

    Uses singleton pattern to ensure model is loaded only once.
    Configured for CPU-only processing for efficiency.

    Complexity:
        - Initialization: O(1) after first load (singleton)
        - Tokenize: O(n) where n is text length

    Attributes:
        ws: CKIP word segmenter
        stopwords: Set of stopwords to filter
        _instance: Singleton instance
    """

    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        """Singleton pattern: ensure only one instance exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self,
                 model_name: str = "bert-base",
                 use_gpu: bool = False,
                 stopwords_file: Optional[str] = None):
        """
        Initialize CKIP tokenizer.

        Only initializes once due to singleton pattern.

        Args:
            model_name: CKIP model name ('bert-base', 'albert-base', 'albert-tiny')
            use_gpu: Whether to use GPU (default False for CPU-only)
            stopwords_file: Path to stopwords file (optional)
        """
        # Only initialize once
        if self.__class__._initialized:
            return

        self.logger = logging.getLogger(__name__)

        try:
            from ckip_transformers.nlp import CkipWordSegmenter

            self.logger.info(f"Loading CKIP model: {model_name} (CPU mode)")

            # Initialize word segmenter with CPU
            self.ws = CkipWordSegmenter(
                model=model_name,
                device=-1 if not use_gpu else 0  # -1 for CPU, 0+ for GPU
            )

            # Load stopwords
            self.stopwords: Set[str] = set()
            if stopwords_file and Path(stopwords_file).exists():
                with open(stopwords_file, 'r', encoding='utf-8') as f:
                    self.stopwords = {line.strip() for line in f if line.strip()}
                self.logger.info(f"Loaded {len(self.stopwords)} stopwords")
            else:
                # Default Chinese stopwords
                self.stopwords = self._get_default_stopwords()
                self.logger.info("Using default stopwords")

            self.__class__._initialized = True
            self.logger.info("CKIP tokenizer initialized successfully")

        except ImportError:
            self.logger.error("ckip-transformers not installed. Run: pip install ckip-transformers")
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize CKIP tokenizer: {e}")
            raise

    def _get_default_stopwords(self) -> Set[str]:
        """
        Get default Chinese stopwords.

        Returns:
            Set of common Chinese stopwords
        """
        return {
            # Punctuation and symbols
            '，', '。', '、', '；', '：', '？', '！', '…', '—', '·',
            '「', '」', '『', '』', '（', '）', '《', '》', '〈', '〉',
            '"', '"', ''', ''', ',', '.', '?', '!', ';', ':',
            '(', ')', '[', ']', '{', '}', '<', '>',

            # Common function words
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人',
            '都', '一', '一個', '上', '也', '很', '到', '說', '要', '去',
            '你', '會', '著', '沒有', '看', '好', '自己', '這', '那',

            # Numbers and quantifiers
            '個', '些', '兩', '三', '四', '五', '六', '七', '八', '九', '十',

            # Time and location
            '年', '月', '日', '時', '分', '秒', '裡', '中', '內',

            # Other common words
            '及', '與', '或', '等', '等等', '之', '以', '為', '於'
        }

    def tokenize(self, text: str,
                 filter_stopwords: bool = True,
                 min_length: int = 2) -> List[str]:
        """
        Tokenize Chinese text using CKIP.

        Args:
            text: Input Chinese text
            filter_stopwords: Whether to filter stopwords
            min_length: Minimum token length to keep

        Returns:
            List of tokens

        Complexity:
            Time: O(n) where n is text length
            Space: O(k) where k is number of tokens

        Examples:
            >>> tokenizer = CKIPTokenizer()
            >>> tokens = tokenizer.tokenize("台灣是一個美麗的島嶼")
            >>> print(tokens)
            ['台灣', '美麗', '島嶼']
        """
        if not text or not text.strip():
            return []

        try:
            # CKIP segmentation (returns list of lists)
            result = self.ws([text])  # Batch processing
            tokens = result[0] if result else []

            # Post-processing
            processed = []
            for token in tokens:
                token = token.strip()

                # Skip empty or too short tokens
                if not token or len(token) < min_length:
                    continue

                # Skip stopwords
                if filter_stopwords and token in self.stopwords:
                    continue

                # Skip pure numbers (optional)
                if token.isdigit():
                    continue

                processed.append(token)

            return processed

        except Exception as e:
            self.logger.warning(f"Tokenization error: {e}, falling back to character split")
            # Fallback: simple character-based tokenization
            return [char for char in text if char.strip() and len(char) >= min_length]

    def tokenize_batch(self, texts: List[str],
                      filter_stopwords: bool = True,
                      min_length: int = 2,
                      batch_size: int = 256) -> List[List[str]]:
        """
        Tokenize multiple texts in batch (more efficient).

        Args:
            texts: List of input texts
            filter_stopwords: Whether to filter stopwords
            min_length: Minimum token length
            batch_size: Batch size for processing

        Returns:
            List of token lists

        Complexity:
            Time: O(n * m) where n is number of texts, m is avg text length
            Space: O(n * k) where k is avg tokens per text

        Examples:
            >>> tokenizer = CKIPTokenizer()
            >>> texts = ["台灣新聞", "今日頭條"]
            >>> result = tokenizer.tokenize_batch(texts)
            >>> print(result)
            [['台灣', '新聞'], ['今日', '頭條']]
        """
        if not texts:
            return []

        all_results = []

        try:
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]

                # CKIP batch processing
                batch_results = self.ws(batch)

                # Post-process each result
                for tokens in batch_results:
                    processed = []
                    for token in tokens:
                        token = token.strip()

                        if not token or len(token) < min_length:
                            continue

                        if filter_stopwords and token in self.stopwords:
                            continue

                        if token.isdigit():
                            continue

                        processed.append(token)

                    all_results.append(processed)

            return all_results

        except Exception as e:
            self.logger.error(f"Batch tokenization error: {e}")
            # Fallback to single tokenization
            return [self.tokenize(text, filter_stopwords, min_length) for text in texts]

    def get_stats(self) -> dict:
        """Get tokenizer statistics."""
        return {
            'model': 'CKIP Transformers',
            'device': 'CPU',
            'stopwords_count': len(self.stopwords),
            'initialized': self.__class__._initialized
        }


# Global instance for easy access
_tokenizer_instance = None


def get_tokenizer(model_name: str = "bert-base",
                 use_gpu: bool = False,
                 stopwords_file: Optional[str] = None) -> CKIPTokenizer:
    """
    Get singleton CKIP tokenizer instance.

    Args:
        model_name: CKIP model name
        use_gpu: Whether to use GPU
        stopwords_file: Path to stopwords file

    Returns:
        CKIPTokenizer instance

    Examples:
        >>> tokenizer = get_tokenizer()
        >>> tokens = tokenizer.tokenize("這是一個測試")
    """
    global _tokenizer_instance

    if _tokenizer_instance is None:
        _tokenizer_instance = CKIPTokenizer(
            model_name=model_name,
            use_gpu=use_gpu,
            stopwords_file=stopwords_file
        )

    return _tokenizer_instance


def demo():
    """Demonstration of CKIP tokenizer."""
    print("=" * 70)
    print("CKIP Tokenizer Demo")
    print("=" * 70)

    # Initialize tokenizer
    print("\n1. Initializing CKIP tokenizer (CPU mode)...")
    tokenizer = get_tokenizer()

    # Test samples
    samples = [
        "台灣是一個美麗的島嶼，擁有豐富的自然資源。",
        "人工智慧技術在近年來取得了突破性的進展。",
        "新聞報導指出，經濟成長率達到了預期目標。"
    ]

    print("\n2. Single Tokenization:")
    print("-" * 70)
    for i, text in enumerate(samples, 1):
        tokens = tokenizer.tokenize(text)
        print(f"\n   Text {i}: {text}")
        print(f"   Tokens: {tokens}")
        print(f"   Count: {len(tokens)} tokens")

    print("\n3. Batch Tokenization:")
    print("-" * 70)
    batch_results = tokenizer.tokenize_batch(samples)
    for i, (text, tokens) in enumerate(zip(samples, batch_results), 1):
        print(f"\n   Text {i}: {tokens}")

    print("\n4. Tokenizer Statistics:")
    print("-" * 70)
    stats = tokenizer.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    demo()
