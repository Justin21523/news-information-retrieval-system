"""
Chinese Tokenizer for Traditional Chinese Text

This module provides unified tokenization interface for Chinese text processing,
integrating multiple backends (CKIP Transformers, Jieba) optimized for
Traditional Chinese (繁體中文).

Key Features:
    - CKIP Transformers for high accuracy (F1 > 90% on Traditional Chinese)
    - Jieba for fast processing (fallback or speed-priority mode)
    - Custom dictionary support for domain-specific terms
    - Batch processing with optional GPU acceleration
    - POS tagging support (CKIP only)
    - Named Entity Recognition - NER (CKIP only, 18 entity types)
    - Caching for frequently tokenized texts

Complexity:
    - tokenize(): O(n) where n = text length
    - tokenize_batch(): O(N*n) with GPU acceleration
    - tokenize_cached(): O(1) for cache hits
    - extract_entities(): O(n) where n = text length

Reference:
    Chen et al. (2022). "Multifaceted Assessments of Traditional Chinese
    Word Segmentation Tool on Large Corpora". ROCLING 2022.

Author: Information Retrieval System
License: Educational Use
"""

from typing import List, Tuple, Optional, Literal, Callable
from functools import lru_cache
import logging
import re

# Try importing CKIP Transformers
try:
    from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger, CkipNerChunker
    CKIP_AVAILABLE = True
except ImportError:
    CKIP_AVAILABLE = False
    CkipWordSegmenter = None
    CkipPosTagger = None
    CkipNerChunker = None

# Jieba is always available (in requirements.txt)
import jieba
import jieba.posseg as pseg


class ChineseTokenizer:
    """
    Unified Chinese tokenizer supporting multiple backends.

    Provides consistent interface for word segmentation across different
    engines, with automatic fallback and caching mechanisms.

    Attributes:
        engine: Tokenization engine ('ckip' | 'jieba' | 'auto')
        mode: Segmentation mode ('default' | 'search' | 'precise')
        use_pos: Whether to perform POS tagging
        custom_dict_path: Path to custom dictionary file

    Examples:
        >>> tokenizer = ChineseTokenizer(engine='ckip')
        >>> tokens = tokenizer.tokenize("國立臺灣大學圖書資訊學系")
        >>> print(tokens)
        ['國立', '臺灣大學', '圖書資訊學系']

        >>> # With POS tagging
        >>> tokenizer = ChineseTokenizer(engine='ckip', use_pos=True)
        >>> tagged = tokenizer.tokenize_with_pos("臺灣大學位於臺北市")
        >>> print(tagged)
        [('臺灣大學', 'Nc'), ('位於', 'VCL'), ('臺北市', 'Nc')]
    """

    def __init__(self,
                 engine: Literal['ckip', 'jieba', 'auto'] = 'auto',
                 mode: Literal['default', 'search', 'precise'] = 'default',
                 use_pos: bool = False,
                 custom_dict_path: Optional[str] = None,
                 device: int = -1):
        """
        Initialize Chinese tokenizer.

        Args:
            engine: Tokenization backend
                - 'ckip': CKIP Transformers (high accuracy, slower, best for Traditional Chinese)
                - 'jieba': Jieba (fast, lower accuracy, good for real-time)
                - 'auto': Auto-select based on availability (prefers CKIP)
            mode: Segmentation mode
                - 'default': Standard segmentation
                - 'search': Search engine mode (more granular, better recall)
                - 'precise': Precise mode (avoid over-segmentation)
            use_pos: Enable POS tagging (CKIP only)
            custom_dict_path: Path to user dictionary (one word per line)
            device: Device for CKIP (-1 for CPU, 0+ for GPU)

        Raises:
            ImportError: If specified engine is not available

        Complexity:
            Time: O(1) for initialization
            Space: O(d) where d = dictionary size
        """
        self.logger = logging.getLogger(__name__)
        self.mode = mode
        self.use_pos = use_pos
        self.device = device

        # Select engine
        if engine == 'auto':
            self.engine = 'ckip' if CKIP_AVAILABLE else 'jieba'
            if not CKIP_AVAILABLE:
                self.logger.warning(
                    "CKIP Transformers not available. Falling back to Jieba. "
                    "Install with: pip install ckip-transformers"
                )
        else:
            self.engine = engine

        # Initialize backend
        self._ws = None  # Word segmenter
        self._pos = None  # POS tagger
        self._ner = None  # NER chunker

        if self.engine == 'ckip':
            self._init_ckip()
        else:
            self._init_jieba(custom_dict_path)

        self.logger.info(
            f"ChineseTokenizer initialized: engine={self.engine}, "
            f"mode={self.mode}, use_pos={self.use_pos}"
        )

    def _init_ckip(self):
        """Initialize CKIP Transformers backend."""
        if not CKIP_AVAILABLE:
            raise ImportError(
                "CKIP Transformers not available. "
                "Install with: pip install ckip-transformers"
            )

        # Initialize word segmenter
        self._ws = CkipWordSegmenter(
            model="bert-base",
            device=self.device
        )
        self.logger.info(f"CKIP Word Segmenter loaded (device={self.device})")

        # Initialize POS tagger if requested
        if self.use_pos:
            self._pos = CkipPosTagger(
                model="bert-base",
                device=self.device
            )
            self.logger.info("CKIP POS Tagger loaded")

    def _init_jieba(self, custom_dict_path: Optional[str] = None):
        """Initialize Jieba backend."""
        # Load custom dictionary
        if custom_dict_path:
            jieba.load_userdict(custom_dict_path)
            self.logger.info(f"Loaded custom dictionary: {custom_dict_path}")

        # Preload Jieba dictionary for faster first tokenization
        jieba.initialize()
        self.logger.info("Jieba initialized")

    # ========================================================================
    # Core Tokenization Methods
    # ========================================================================

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize Chinese text into words.

        Args:
            text: Input text (Traditional or Simplified Chinese)

        Returns:
            List of word tokens

        Complexity:
            Time: O(n) where n = text length
            Space: O(k) where k = number of tokens

        Examples:
            >>> tokenizer = ChineseTokenizer(engine='ckip')
            >>> tokenizer.tokenize("國立臺灣大學圖書資訊學系")
            ['國立', '臺灣大學', '圖書資訊學系']

            >>> tokenizer.tokenize("機器學習與深度學習")
            ['機器', '學習', '與', '深度', '學習']

            >>> tokenizer.tokenize("資訊檢索系統評估")
            ['資訊', '檢索', '系統', '評估']
        """
        if not text or not text.strip():
            return []

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text.strip())

        if self.engine == 'ckip':
            return self._tokenize_ckip(text)
        else:
            return self._tokenize_jieba(text)

    def _tokenize_ckip(self, text: str) -> List[str]:
        """
        CKIP tokenization.

        CKIP Transformers uses BERT-based model trained on Academia Sinica
        Balanced Corpus, achieving F1 > 90% on Traditional Chinese.
        """
        # CKIP expects list input, returns list of lists
        result = self._ws([text], batch_size=1, show_progress=False)
        return result[0] if result else []

    def _tokenize_jieba(self, text: str) -> List[str]:
        """
        Jieba tokenization with mode selection.

        Jieba uses HMM + prefix dictionary approach, achieving ~81% F1 score
        but with much faster speed.
        """
        if self.mode == 'search':
            # Search engine mode: more granular segmentation
            # Good for improving recall in IR systems
            return list(jieba.cut_for_search(text))
        elif self.mode == 'precise':
            # Precise mode: avoid over-segmentation
            return list(jieba.cut(text, cut_all=False, HMM=True))
        else:
            # Default mode
            return list(jieba.cut(text, cut_all=False))

    def tokenize_batch(self, texts: List[str],
                      batch_size: int = 32,
                      show_progress: bool = False) -> List[List[str]]:
        """
        Batch tokenization for efficiency.

        Significantly faster for large text collections, especially with
        CKIP on GPU.

        Args:
            texts: List of texts to tokenize
            batch_size: Batch size for processing (CKIP only)
            show_progress: Show progress bar (CKIP only)

        Returns:
            List of token lists, one per input text

        Complexity:
            Time: O(N*n) where N = number of texts, n = avg length
            Space: O(N*k) where k = avg tokens per text

        Examples:
            >>> tokenizer = ChineseTokenizer()
            >>> texts = ["機器學習", "深度學習", "自然語言處理"]
            >>> results = tokenizer.tokenize_batch(texts)
            >>> print(results)
            [['機器', '學習'], ['深度', '學習'], ['自然', '語言', '處理']]
        """
        if not texts:
            return []

        if self.engine == 'ckip':
            # CKIP supports native batch processing
            return self._ws(
                texts,
                batch_size=batch_size,
                show_progress=show_progress
            )
        else:
            # Jieba processes sequentially
            return [self.tokenize(text) for text in texts]

    # ========================================================================
    # POS Tagging
    # ========================================================================

    def tokenize_with_pos(self, text: str) -> List[Tuple[str, str]]:
        """
        Tokenize with part-of-speech tagging.

        Args:
            text: Input text

        Returns:
            List of (word, POS_tag) tuples

        POS Tags (CKIP Academia Sinica Tagset):
            Nouns:
            - Na: Common noun (普通名詞) e.g., 學生、書本
            - Nc: Place name (地名) e.g., 臺北、台灣
            - Nb: Proper noun (專有名詞) e.g., 張三、Google

            Verbs:
            - VA: Active verb (動作動詞) e.g., 跑、吃
            - VE: Existential verb (存在動詞) e.g., 有、在
            - VC: Copula (繫詞) e.g., 是

            Others:
            - A: Adjective (形容詞) e.g., 大、好
            - D: Adverb (副詞) e.g., 很、非常
            - P: Preposition (介詞) e.g., 在、從
            - Caa: Coordinating conjunction (對等連接詞) e.g., 和、與

        Raises:
            NotImplementedError: If engine is not 'ckip'

        Complexity:
            Time: O(n) where n = text length
            Space: O(k) where k = number of tokens

        Examples:
            >>> tokenizer = ChineseTokenizer(engine='ckip', use_pos=True)
            >>> tagged = tokenizer.tokenize_with_pos("臺灣大學位於臺北市")
            >>> print(tagged)
            [('臺灣大學', 'Nc'), ('位於', 'VCL'), ('臺北市', 'Nc')]

            >>> tagged = tokenizer.tokenize_with_pos("機器學習是人工智慧的重要領域")
            >>> print(tagged)
            [('機器', 'Na'), ('學習', 'VA'), ('是', 'SHI'), ('人工', 'Na'),
             ('智慧', 'Na'), ('的', 'DE'), ('重要', 'VH'), ('領域', 'Na')]
        """
        if self.engine == 'ckip':
            return self._pos_tag_ckip(text)
        elif self.engine == 'jieba':
            return self._pos_tag_jieba(text)
        else:
            raise NotImplementedError(
                f"POS tagging not available for engine: {self.engine}"
            )

    def _pos_tag_ckip(self, text: str) -> List[Tuple[str, str]]:
        """CKIP POS tagging."""
        if self._pos is None:
            # Lazy initialization
            self._pos = CkipPosTagger(
                model="bert-base",
                device=self.device
            )

        # Get word segmentation
        words = self._ws([text], show_progress=False)[0]

        # Get POS tags
        tags = self._pos([words], show_progress=False)[0]

        return list(zip(words, tags))

    def _pos_tag_jieba(self, text: str) -> List[Tuple[str, str]]:
        """Jieba POS tagging."""
        # Jieba uses simplified tagset
        return [(word, flag) for word, flag in pseg.cut(text)]

    # ========================================================================
    # Named Entity Recognition (NER)
    # ========================================================================

    def extract_entities(self, text: str) -> List[Tuple[str, str, int, int]]:
        """
        Extract named entities from text using CKIP NER.

        Identifies and classifies named entities into predefined categories.

        Args:
            text: Input text

        Returns:
            List of (entity, entity_type, start_pos, end_pos) tuples

        Entity Types (CKIP NER):
            - PERSON: Person names (人名) e.g., 張三、李明
            - GPE: Geopolitical entities (地緣政治實體) e.g., 台北、美國
            - ORG: Organizations (組織) e.g., 台灣大學、Google
            - LOC: Locations (地點) e.g., 太平洋、玉山
            - DATE: Dates (日期) e.g., 2025年、一月
            - TIME: Times (時間) e.g., 早上、三點
            - MONEY: Monetary values (金錢) e.g., 一百元、5美金
            - QUANTITY: Quantities (數量) e.g., 三個、一公斤
            - CARDINAL: Cardinal numbers (基數) e.g., 一、二、三
            - ORDINAL: Ordinal numbers (序數) e.g., 第一、第二
            - PERCENT: Percentages (百分比) e.g., 50%、百分之五十
            - EVENT: Events (事件) e.g., 世界大戰、奧運
            - FAC: Facilities (設施) e.g., 101大樓、機場
            - LAW: Laws (法律) e.g., 憲法、民法
            - LANGUAGE: Languages (語言) e.g., 中文、英文
            - NORP: Nationalities/religions (國籍/宗教) e.g., 台灣人、佛教
            - PRODUCT: Products (產品) e.g., iPhone、Windows
            - WORK_OF_ART: Works of art (藝術作品) e.g., 蒙娜麗莎

        Raises:
            NotImplementedError: If engine is not 'ckip'

        Complexity:
            Time: O(n) where n = text length
            Space: O(e) where e = number of entities

        Examples:
            >>> tokenizer = ChineseTokenizer(engine='ckip')
            >>> entities = tokenizer.extract_entities("張三在台灣大學讀書")
            >>> print(entities)
            [('張三', 'PERSON', 0, 2), ('台灣大學', 'ORG', 3, 7)]

            >>> entities = tokenizer.extract_entities("2025年一月在台北舉辦研討會")
            >>> print(entities)
            [('2025年', 'DATE', 0, 5), ('一月', 'DATE', 5, 7), ('台北', 'GPE', 8, 10)]
        """
        if self.engine != 'ckip':
            raise NotImplementedError(
                f"NER extraction only available with CKIP engine, not {self.engine}"
            )

        # Handle empty text
        if not text or not text.strip():
            return []

        # Lazy initialization of NER driver
        if self._ner is None:
            self._ner = CkipNerChunker(
                model="bert-base",
                device=self.device
            )
            self.logger.info(f"CKIP NER Chunker loaded (device={self.device})")

        # CKIP NER expects raw text, not tokenized words
        ner_results = self._ner([text], show_progress=False)[0]

        # Format results: (entity_text, entity_type, start_pos, end_pos)
        entities = []
        for entity_info in ner_results:
            entity_text = entity_info.word
            entity_type = entity_info.ner
            start_pos = entity_info.idx[0]
            end_pos = entity_info.idx[1]
            entities.append((entity_text, entity_type, start_pos, end_pos))

        return entities

    def extract_entities_batch(self, texts: List[str],
                              batch_size: int = 32,
                              show_progress: bool = False) -> List[List[Tuple[str, str, int, int]]]:
        """
        Batch named entity extraction for efficiency.

        Significantly faster for large text collections, especially with GPU.

        Args:
            texts: List of texts to process
            batch_size: Batch size for processing
            show_progress: Show progress bar

        Returns:
            List of entity lists, one per input text
            Each entity is (entity, entity_type, start_pos, end_pos)

        Complexity:
            Time: O(N*n) where N = number of texts, n = avg length
            Space: O(N*e) where e = avg entities per text

        Examples:
            >>> tokenizer = ChineseTokenizer(engine='ckip')
            >>> texts = ["張三在台大讀書", "李四在台北工作"]
            >>> results = tokenizer.extract_entities_batch(texts)
            >>> print(results)
            [[('張三', 'PERSON', 0, 2), ('台大', 'ORG', 3, 5)],
             [('李四', 'PERSON', 0, 2), ('台北', 'GPE', 3, 5)]]
        """
        if self.engine != 'ckip':
            raise NotImplementedError(
                f"NER extraction only available with CKIP engine, not {self.engine}"
            )

        if not texts:
            return []

        # Lazy initialization of NER driver
        if self._ner is None:
            self._ner = CkipNerChunker(
                model="bert-base",
                device=self.device
            )
            self.logger.info(f"CKIP NER Chunker loaded (device={self.device})")

        # CKIP NER expects raw text, not tokenized words
        all_ner_results = self._ner(texts, batch_size=batch_size, show_progress=show_progress)

        # Format results
        batch_entities = []
        for ner_results in all_ner_results:
            entities = []
            for entity_info in ner_results:
                entity_text = entity_info.word
                entity_type = entity_info.ner
                start_pos = entity_info.idx[0]
                end_pos = entity_info.idx[1]
                entities.append((entity_text, entity_type, start_pos, end_pos))
            batch_entities.append(entities)

        return batch_entities

    # ========================================================================
    # Caching
    # ========================================================================

    @lru_cache(maxsize=10000)
    def tokenize_cached(self, text: str) -> Tuple[str, ...]:
        """
        Cached version for frequently tokenized texts.

        Useful for repeated queries or document re-indexing.
        Returns tuple instead of list for hashability.

        Args:
            text: Input text

        Returns:
            Tuple of tokens (immutable)

        Complexity:
            Time: O(1) for cache hits, O(n) for cache misses
            Space: O(C*k) where C = cache size, k = avg tokens

        Examples:
            >>> tokenizer = ChineseTokenizer()
            >>> tokens1 = tokenizer.tokenize_cached("機器學習")
            >>> tokens2 = tokenizer.tokenize_cached("機器學習")  # Cache hit
            >>> assert tokens1 is tokens2  # Same object
        """
        return tuple(self.tokenize(text))

    def clear_cache(self):
        """Clear the tokenization cache."""
        self.tokenize_cached.cache_clear()
        self.logger.info("Tokenization cache cleared")

    def cache_info(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats (hits, misses, size, maxsize)
        """
        info = self.tokenize_cached.cache_info()
        return {
            'hits': info.hits,
            'misses': info.misses,
            'size': info.currsize,
            'maxsize': info.maxsize,
            'hit_rate': info.hits / (info.hits + info.misses) if (info.hits + info.misses) > 0 else 0.0
        }

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def add_custom_word(self, word: str, freq: Optional[int] = None, tag: Optional[str] = None):
        """
        Add a custom word to the dictionary (Jieba only).

        Useful for domain-specific terms that are not in the default dictionary.

        Args:
            word: Word to add
            freq: Word frequency (optional, higher = more likely to be segmented)
            tag: POS tag (optional)

        Examples:
            >>> tokenizer = ChineseTokenizer(engine='jieba')
            >>> tokenizer.add_custom_word("圖書資訊學", freq=1000, tag="Na")
            >>> tokens = tokenizer.tokenize("我在學圖書資訊學")
            >>> assert "圖書資訊學" in tokens
        """
        if self.engine != 'jieba':
            self.logger.warning(
                f"add_custom_word() only supported for Jieba, not {self.engine}"
            )
            return

        if freq and tag:
            jieba.add_word(word, freq, tag)
        elif freq:
            jieba.add_word(word, freq)
        else:
            jieba.add_word(word)

        self.logger.debug(f"Added custom word: {word}")

    def suggest_freq(self, segment: str, tune: bool = True) -> int:
        """
        Adjust word frequency to promote/demote segmentation (Jieba only).

        Args:
            segment: Word or phrase
            tune: If True, adjust frequency to promote this segmentation

        Returns:
            Suggested frequency

        Examples:
            >>> tokenizer = ChineseTokenizer(engine='jieba')
            >>> # Force "台中" to be segmented as one word
            >>> tokenizer.suggest_freq("台中", tune=True)
        """
        if self.engine != 'jieba':
            self.logger.warning("suggest_freq() only supported for Jieba")
            return 0

        return jieba.suggest_freq(segment, tune=tune)


def demo():
    """Demonstration of ChineseTokenizer capabilities."""
    print("=" * 70)
    print("Chinese Tokenizer Demo (Traditional Chinese)")
    print("=" * 70)

    # Sample texts (Traditional Chinese from various domains)
    texts = [
        "國立臺灣大學圖書資訊學系",
        "機器學習與深度學習的應用",
        "資訊檢索系統評估指標包括精確率、召回率與F分數",
        "這是一個中文斷詞測試案例",
        "自然語言處理是人工智慧的重要分支"
    ]

    # Example 1: CKIP tokenization
    if CKIP_AVAILABLE:
        print("\n[1] CKIP Tokenizer (High Accuracy, F1 > 90%):")
        print("-" * 70)
        tokenizer_ckip = ChineseTokenizer(engine='ckip')
        for text in texts:
            tokens = tokenizer_ckip.tokenize(text)
            print(f"Input:  {text}")
            print(f"Tokens: {' | '.join(tokens)}")
            print()
    else:
        print("\n[1] CKIP Tokenizer: NOT AVAILABLE")
        print("    Install with: pip install ckip-transformers")
        print()

    # Example 2: Jieba tokenization
    print("\n[2] Jieba Tokenizer (Fast, F1 ~81%):")
    print("-" * 70)
    tokenizer_jieba = ChineseTokenizer(engine='jieba')
    for text in texts:
        tokens = tokenizer_jieba.tokenize(text)
        print(f"Input:  {text}")
        print(f"Tokens: {' | '.join(tokens)}")
        print()

    # Example 3: Search mode comparison
    print("\n[3] Jieba Search Mode (More Granular):")
    print("-" * 70)
    tokenizer_search = ChineseTokenizer(engine='jieba', mode='search')
    test_text = "國立臺灣大學"
    tokens_default = tokenizer_jieba.tokenize(test_text)
    tokens_search = tokenizer_search.tokenize(test_text)
    print(f"Input:   {test_text}")
    print(f"Default: {' | '.join(tokens_default)}")
    print(f"Search:  {' | '.join(tokens_search)}")
    print()

    # Example 4: Batch processing
    print("\n[4] Batch Processing:")
    print("-" * 70)
    start_time = __import__('time').time()
    if CKIP_AVAILABLE:
        results = tokenizer_ckip.tokenize_batch(texts)
        elapsed = __import__('time').time() - start_time
        print(f"Processed {len(texts)} texts in {elapsed:.3f}s")
        for text, tokens in zip(texts, results):
            print(f"  {text[:30]:30s} → {len(tokens)} tokens")
    else:
        print("  CKIP not available for batch demo")
    print()

    # Example 5: POS tagging
    if CKIP_AVAILABLE:
        print("\n[5] POS Tagging (CKIP):")
        print("-" * 70)
        tokenizer_pos = ChineseTokenizer(engine='ckip', use_pos=True)
        test_text = "臺灣大學位於臺北市"
        tagged = tokenizer_pos.tokenize_with_pos(test_text)
        print(f"Input: {test_text}")
        print("Tagged:")
        for word, tag in tagged:
            print(f"  {word:10s} → {tag}")
        print()

    # Example 6: Caching
    print("\n[6] Caching Performance:")
    print("-" * 70)
    tokenizer = ChineseTokenizer(engine='jieba')
    test_text = "機器學習是人工智慧的重要領域"

    # First call (cache miss)
    start = __import__('time').time()
    tokens1 = tokenizer.tokenize_cached(test_text)
    time1 = __import__('time').time() - start

    # Second call (cache hit)
    start = __import__('time').time()
    tokens2 = tokenizer.tokenize_cached(test_text)
    time2 = __import__('time').time() - start

    print(f"First call:  {time1*1000:.3f}ms (cache miss)")
    print(f"Second call: {time2*1000:.3f}ms (cache hit)")
    print(f"Speedup: {time1/time2:.1f}x")
    print(f"\nCache info: {tokenizer.cache_info()}")
    print()

    print("=" * 70)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    demo()
