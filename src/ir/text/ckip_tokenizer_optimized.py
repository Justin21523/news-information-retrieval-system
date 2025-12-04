"""
CKIP Transformers Tokenizer - Optimized Version

Enhanced version with multi-threading optimization for CPU processing.
Provides configurable thread settings for maximum performance.

Key Enhancements:
    - Configurable PyTorch thread count
    - Environment variable optimization
    - Batch processing with larger sizes
    - Performance monitoring

Author: Information Retrieval System
License: Educational Use
"""

import os
import logging
import torch
from typing import List, Optional, Set
from pathlib import Path


class CKIPTokenizerOptimized:
    """
    CKIP Transformers tokenizer with multi-threading optimization.

    Optimized for systems with high core count (e.g., 32 threads).
    Configures PyTorch and system threading for maximum CPU utilization.

    Complexity:
        - Initialization: O(1) after first load (singleton)
        - Tokenize: O(n) where n is text length
        - Batch tokenize: O(m*n) with better throughput

    Attributes:
        ws: CKIP word segmenter
        stopwords: Set of stopwords to filter
        num_threads: Number of threads for PyTorch
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
                 num_threads: Optional[int] = None,
                 stopwords_file: Optional[str] = None):
        """
        Initialize optimized CKIP tokenizer.

        Args:
            model_name: CKIP model name ('bert-base', 'albert-base', 'albert-tiny')
            use_gpu: Whether to use GPU (default False for CPU-only)
            num_threads: Number of threads to use (default: all available)
            stopwords_file: Path to stopwords file (optional)
        """
        # Only initialize once
        if self.__class__._initialized:
            return

        self.logger = logging.getLogger(__name__)

        # Optimize threading BEFORE importing CKIP
        self._optimize_threading(num_threads)

        try:
            from ckip_transformers.nlp import CkipWordSegmenter

            self.logger.info(f"Loading CKIP model: {model_name} (CPU mode, {self.num_threads} threads)")

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
            self.logger.info(f"CKIP tokenizer initialized successfully (threads: {self.num_threads})")

        except ImportError:
            self.logger.error("ckip-transformers not installed. Run: pip install ckip-transformers")
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize CKIP tokenizer: {e}")
            raise

    def _optimize_threading(self, num_threads: Optional[int] = None):
        """
        Optimize PyTorch and system threading for CPU processing.

        Sets environment variables and PyTorch thread settings for
        maximum CPU utilization.

        Args:
            num_threads: Number of threads (default: all available CPU threads)
        """
        import multiprocessing

        # Determine thread count
        if num_threads is None:
            # Use all available CPU threads
            num_threads = multiprocessing.cpu_count()

        self.num_threads = num_threads

        # Set environment variables (must be done BEFORE torch import)
        os.environ['OMP_NUM_THREADS'] = str(num_threads)
        os.environ['MKL_NUM_THREADS'] = str(num_threads)
        os.environ['NUMEXPR_NUM_THREADS'] = str(num_threads)

        # Set PyTorch threading
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(num_threads)

        self.logger.info(f"Threading optimized: {num_threads} threads")
        self.logger.info(f"  - PyTorch threads: {torch.get_num_threads()}")
        self.logger.info(f"  - PyTorch interop: {torch.get_num_interop_threads()}")
        self.logger.info(f"  - OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS')}")

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
            >>> tokenizer = CKIPTokenizerOptimized(num_threads=32)
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
                      batch_size: int = 512) -> List[List[str]]:
        """
        Tokenize multiple texts in batch (optimized for multi-threading).

        Increased default batch size from 256 to 512 for better throughput
        with higher thread counts.

        Args:
            texts: List of input texts
            filter_stopwords: Whether to filter stopwords
            min_length: Minimum token length
            batch_size: Batch size for processing (default 512, increased from 256)

        Returns:
            List of token lists

        Complexity:
            Time: O(n * m) where n is number of texts, m is avg text length
            Space: O(n * k) where k is avg tokens per text

        Examples:
            >>> tokenizer = CKIPTokenizerOptimized(num_threads=32)
            >>> texts = ["台灣新聞", "今日頭條"]
            >>> result = tokenizer.tokenize_batch(texts, batch_size=512)
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
        """Get tokenizer statistics including threading info."""
        return {
            'model': 'CKIP Transformers (Optimized)',
            'device': 'CPU',
            'num_threads': self.num_threads,
            'pytorch_threads': torch.get_num_threads(),
            'pytorch_interop_threads': torch.get_num_interop_threads(),
            'stopwords_count': len(self.stopwords),
            'initialized': self.__class__._initialized
        }


# Global instance for easy access
_optimized_tokenizer_instance = None


def get_optimized_tokenizer(model_name: str = "bert-base",
                            use_gpu: bool = False,
                            num_threads: Optional[int] = None,
                            stopwords_file: Optional[str] = None) -> CKIPTokenizerOptimized:
    """
    Get singleton optimized CKIP tokenizer instance.

    Args:
        model_name: CKIP model name
        use_gpu: Whether to use GPU
        num_threads: Number of threads (default: all available)
        stopwords_file: Path to stopwords file

    Returns:
        CKIPTokenizerOptimized instance

    Examples:
        >>> # Use all 32 threads on Ryzen 9 9950X
        >>> tokenizer = get_optimized_tokenizer(num_threads=32)
        >>> tokens = tokenizer.tokenize("這是一個測試")

        >>> # Auto-detect thread count
        >>> tokenizer = get_optimized_tokenizer()
        >>> tokens = tokenizer.tokenize("自動檢測執行緒數")
    """
    global _optimized_tokenizer_instance

    if _optimized_tokenizer_instance is None:
        _optimized_tokenizer_instance = CKIPTokenizerOptimized(
            model_name=model_name,
            use_gpu=use_gpu,
            num_threads=num_threads,
            stopwords_file=stopwords_file
        )

    return _optimized_tokenizer_instance


if __name__ == '__main__':
    import time

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 80)
    print("CKIP Tokenizer Optimization Demo")
    print("=" * 80)

    # Test with different thread counts
    for threads in [16, 32]:
        print(f"\n\n{'='*80}")
        print(f"Testing with {threads} threads")
        print('='*80)

        # Reset singleton for testing
        CKIPTokenizerOptimized._instance = None
        CKIPTokenizerOptimized._initialized = False

        tokenizer = get_optimized_tokenizer(num_threads=threads)

        # Test sentences
        test_texts = [
            "台灣是一個美麗的島嶼，擁有豐富的自然資源。",
            "人工智慧技術在近年來取得了突破性的進展。",
            "新聞報導指出，經濟成長率達到了預期目標。"
        ] * 10  # 30 sentences

        print(f"\nBatch processing {len(test_texts)} sentences...")
        start = time.time()
        results = tokenizer.tokenize_batch(test_texts, batch_size=512)
        elapsed = time.time() - start

        print(f"  Time: {elapsed:.4f}s")
        print(f"  Throughput: {len(test_texts)/elapsed:.2f} sentences/sec")
        print(f"  Total tokens: {sum(len(r) for r in results)}")

        stats = tokenizer.get_stats()
        print(f"\nTokenizer stats:")
        for k, v in stats.items():
            print(f"  {k}: {v}")
