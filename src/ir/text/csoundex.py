"""
CSoundex - Chinese Soundex Phonetic Encoding System

This module implements a phonetic encoding system for Mandarin Chinese,
inspired by the classic Soundex algorithm. It groups Chinese characters
by their phonetic similarity using Hanyu Pinyin romanization.

Key Features:
    - Convert Chinese characters to phonetic codes
    - Support homophone matching (同音字匹配)
    - Handle variant characters (異體字處理)
    - Fuzzy similarity calculation
    - Mixed Chinese/English/punctuation support

Author: Information Retrieval System
License: Educational Use
"""

import re
import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import yaml

try:
    from pypinyin import pinyin, Style
    PYPINYIN_AVAILABLE = True
except ImportError:
    PYPINYIN_AVAILABLE = False
    logging.warning("pypinyin not available. CSoundex will use dictionary-only mode.")


class CSoundex:
    """
    Chinese Soundex encoder for phonetic similarity matching.

    This class converts Chinese characters to phonetic codes based on
    their Pinyin romanization, grouping similar sounds together.

    Encoding Format:
        Standard (4-char): [First_Letter][Initial_Code][Final_Code][Tone_Code]
        Example: 張 (zhang1) → Z811

    Complexity:
        - encode(): O(n) where n is character count (with LRU cache: O(1) amortized)
        - similarity(): O(min(len(s1), len(s2)))

    Attributes:
        config (dict): Configuration loaded from YAML
        initial_groups (dict): Initial consonant grouping rules
        final_groups (dict): Final vowel grouping rules
        lexicon (dict): Character to Pinyin mapping dictionary
    """

    def __init__(self, config_path: Optional[str] = None, lexicon_path: Optional[str] = None):
        """
        Initialize CSoundex encoder.

        Args:
            config_path: Path to csoundex.yaml config file. If None, uses default.
            lexicon_path: Path to pinyin dictionary TSV file. If None, uses default.

        Raises:
            FileNotFoundError: If config file not found
            ValueError: If config format is invalid
        """
        self.logger = logging.getLogger(__name__)

        # Load configuration
        if config_path is None:
            config_path = self._get_default_config_path()
        self.config = self._load_config(config_path)

        # Extract grouping rules
        self.initial_groups = self.config['initial_groups']
        self.final_groups = self.config['final_groups']

        # Build reverse mapping: pinyin initial/final -> group code
        self.initial_to_code = self._build_reverse_mapping(self.initial_groups)
        self.final_to_code = self._build_reverse_mapping(self.final_groups)

        # Load lexicon
        if lexicon_path is None:
            lexicon_path = self.config.get('lexicon_path', None)
        self.lexicon = self._load_lexicon(lexicon_path) if lexicon_path else {}

        # Configuration options
        self.use_pypinyin_fallback = self.config.get('use_pypinyin_fallback', True)
        self.pypinyin_style = getattr(Style, self.config.get('pypinyin_style', 'TONE3'))
        self.default_include_tone = self.config.get('default_include_tone', False)
        self.cache_enabled = self.config.get('enable_cache', True)

        self.logger.info(f"CSoundex initialized. Lexicon: {len(self.lexicon)} entries")

    def _get_default_config_path(self) -> str:
        """Get default config file path."""
        # Assume we're in src/ir/text/, config is in configs/
        base_dir = Path(__file__).parent.parent.parent.parent
        return str(base_dir / 'configs' / 'csoundex.yaml')

    def _load_config(self, config_path: str) -> dict:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to config file

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If file not found
            yaml.YAMLError: If YAML parsing fails
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        self.logger.debug(f"Loaded config from {config_path}")
        return config

    def _build_reverse_mapping(self, groups: Dict[int, List[str]]) -> Dict[str, int]:
        """
        Build reverse mapping from phoneme to group code.

        Args:
            groups: Dictionary of {code: [phonemes]}

        Returns:
            Dictionary of {phoneme: code}
        """
        mapping = {}
        for code, phonemes in groups.items():
            for phoneme in phonemes:
                mapping[phoneme] = code
        return mapping

    def _load_lexicon(self, lexicon_path: str) -> Dict[str, str]:
        """
        Load pinyin lexicon from TSV file.

        Format: character TAB pinyin (e.g., "張\tzhang1")

        Args:
            lexicon_path: Path to TSV file

        Returns:
            Dictionary mapping character to pinyin
        """
        lexicon = {}
        lexicon_file = Path(lexicon_path)

        if not lexicon_file.exists():
            self.logger.warning(f"Lexicon file not found: {lexicon_path}")
            return lexicon

        with open(lexicon_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split('\t')
                if len(parts) >= 2:
                    char, py = parts[0], parts[1]
                    lexicon[char] = py
                else:
                    self.logger.debug(f"Invalid line {line_num} in lexicon: {line}")

        self.logger.info(f"Loaded {len(lexicon)} entries from lexicon")
        return lexicon

    def get_pinyin(self, char: str) -> Optional[str]:
        """
        Get Pinyin romanization for a Chinese character.

        Priority:
            1. Built-in lexicon
            2. pypinyin library (if available and enabled)
            3. None (if not found)

        Args:
            char: Single Chinese character

        Returns:
            Pinyin string (e.g., "zhang1") or None
        """
        # Try lexicon first
        if char in self.lexicon:
            return self.lexicon[char]

        # Fallback to pypinyin
        if self.use_pypinyin_fallback and PYPINYIN_AVAILABLE:
            result = pinyin(char, style=self.pypinyin_style, errors='ignore')
            if result and result[0]:
                return result[0][0]

        return None

    def normalize_pinyin(self, py: str) -> Tuple[str, str, str]:
        """
        Normalize pinyin and extract components.

        Args:
            py: Pinyin string (e.g., "zhang1", "zhang", "ZHANG1")

        Returns:
            Tuple of (initial, final, tone)
            - initial: Initial consonant (e.g., "zh")
            - final: Final vowel/ending (e.g., "ang")
            - tone: Tone number as string (e.g., "1") or "0" for neutral

        Examples:
            >>> normalize_pinyin("zhang1")
            ("zh", "ang", "1")
            >>> normalize_pinyin("yi4")
            ("", "i", "4")
            >>> normalize_pinyin("a1")
            ("", "a", "1")
        """
        # Lowercase
        py = py.lower().strip()

        # Extract tone (last digit if present)
        tone_match = re.search(r'(\d)$', py)
        if tone_match:
            tone = tone_match.group(1)
            py = py[:-1]  # Remove tone digit
        else:
            tone = "0"  # Neutral tone / no tone

        # Split into initial and final
        initial, final = self._split_initial_final(py)

        return initial, final, tone

    def _split_initial_final(self, py: str) -> Tuple[str, str]:
        """
        Split pinyin into initial consonant and final.

        Args:
            py: Pinyin without tone (e.g., "zhang", "yi", "a")

        Returns:
            Tuple of (initial, final)
        """
        # Initials ordered by length (longest first to match correctly)
        initials = ['zh', 'ch', 'sh', 'b', 'p', 'm', 'f', 'd', 't', 'n', 'l',
                    'g', 'k', 'h', 'j', 'q', 'x', 'r', 'z', 'c', 's', 'y', 'w']

        for init in initials:
            if py.startswith(init):
                final = py[len(init):]
                # Handle special cases: yi -> i, wu -> u, yu -> v
                if init == 'y' and final in ['i', '']:
                    return '', 'i'
                elif init == 'w' and final in ['u', '']:
                    return '', 'u'
                elif init == 'y' and final.startswith('u'):
                    return '', 'v' + final[1:]  # yu -> v
                return init, final if final else py

        # No initial consonant (pure vowel)
        return '', py

    @lru_cache(maxsize=10000)
    def encode_character(self, char: str, include_tone: Optional[bool] = None) -> str:
        """
        Encode a single Chinese character to CSoundex code.

        Args:
            char: Single character (Chinese/English/punctuation)
            include_tone: Whether to include tone in encoding. If None, uses default.

        Returns:
            CSoundex code (e.g., "Z811" for 張)
            - Format: [First_Letter][Initial_Code][Final_Code][Tone_Code]
            - Non-Chinese returns uppercase first letter

        Complexity:
            Time: O(1) with LRU cache, O(k) without cache where k is pinyin lookup time
            Space: O(1)
        """
        if include_tone is None:
            include_tone = self.default_include_tone

        # Handle non-Chinese characters
        if not self._is_chinese(char):
            if char.isalpha():
                return char.upper()
            else:
                return ''  # Ignore punctuation/numbers

        # Get pinyin
        py = self.get_pinyin(char)
        if not py:
            self.logger.debug(f"No pinyin found for: {char}")
            return char  # Return original if can't encode

        # Normalize and extract components
        initial, final, tone = self.normalize_pinyin(py)

        # Get first letter (uppercase)
        first_letter = py[0].upper()

        # Get group codes
        initial_code = self.initial_to_code.get(initial, 0)
        final_code = self.final_to_code.get(final, 0)

        # Build code
        if include_tone:
            code = f"{first_letter}{initial_code}{final_code}{tone}"
        else:
            code = f"{first_letter}{initial_code}{final_code}"

        return code

    def encode(self, text: str, include_tone: Optional[bool] = None) -> str:
        """
        Encode a string of text to CSoundex codes.

        Args:
            text: Input text (mixed Chinese/English/punctuation)
            include_tone: Whether to include tone. If None, uses default.

        Returns:
            Space-separated CSoundex codes

        Examples:
            >>> csoundex.encode("張三")
            "Z811 S900"
            >>> csoundex.encode("三聚氰胺")
            "S900 J760 Q700 A300"
            >>> csoundex.encode("hello 世界")
            "H E L L O S840 J740"

        Complexity:
            Time: O(n) where n is character count
            Space: O(n) for output string
        """
        codes = []
        for char in text:
            code = self.encode_character(char, include_tone)
            if code:  # Skip empty codes (punctuation)
                codes.append(code)

        return ' '.join(codes)

    def encode_batch(self, texts: List[str], include_tone: Optional[bool] = None) -> List[str]:
        """
        Encode multiple texts in batch.

        Args:
            texts: List of text strings
            include_tone: Whether to include tone

        Returns:
            List of encoded strings

        Complexity:
            Time: O(n*m) where n is number of texts, m is average length
        """
        return [self.encode(text, include_tone) for text in texts]

    def similarity(self, text1: str, text2: str, mode: str = 'fuzzy') -> float:
        """
        Calculate phonetic similarity between two texts.

        Args:
            text1: First text
            text2: Second text
            mode: Similarity mode
                - 'exact': Binary (1.0 if codes match, 0.0 otherwise)
                - 'fuzzy': Character-level similarity (default)
                - 'weighted': Position-weighted similarity

        Returns:
            Similarity score in [0.0, 1.0]

        Examples:
            >>> csoundex.similarity("張三", "章三", mode='fuzzy')
            0.75  # 張≈章, 三=三
            >>> csoundex.similarity("張三", "李四", mode='fuzzy')
            0.0

        Complexity:
            Time: O(min(n, m)) where n, m are text lengths
            Space: O(n + m)
        """
        code1 = self.encode(text1, include_tone=False)
        code2 = self.encode(text2, include_tone=False)

        if mode == 'exact':
            return 1.0 if code1 == code2 else 0.0

        elif mode == 'fuzzy':
            # Character-level similarity
            codes1 = code1.split()
            codes2 = code2.split()

            if not codes1 or not codes2:
                return 0.0

            # Count matching positions
            matches = sum(1 for c1, c2 in zip(codes1, codes2) if c1 == c2)
            max_len = max(len(codes1), len(codes2))

            return matches / max_len if max_len > 0 else 0.0

        elif mode == 'weighted':
            # Position-weighted (earlier positions more important)
            codes1 = code1.split()
            codes2 = code2.split()

            if not codes1 or not codes2:
                return 0.0

            total_weight = 0.0
            matched_weight = 0.0

            for i, (c1, c2) in enumerate(zip(codes1, codes2)):
                weight = 1.0 / (i + 1)  # Decreasing weight
                total_weight += weight
                if c1 == c2:
                    matched_weight += weight

            # Add remaining positions to total weight
            longer = max(len(codes1), len(codes2))
            for i in range(len(codes1), longer):
                total_weight += 1.0 / (i + 1)

            return matched_weight / total_weight if total_weight > 0 else 0.0

        else:
            raise ValueError(f"Unknown similarity mode: {mode}")

    def find_similar(self, query: str, candidates: List[str],
                     threshold: float = 0.6, topk: Optional[int] = None) -> List[Tuple[str, float]]:
        """
        Find similar texts from a list of candidates.

        Args:
            query: Query text
            candidates: List of candidate texts
            threshold: Minimum similarity threshold (default: 0.6)
            topk: Return top-k results only (default: None, returns all above threshold)

        Returns:
            List of (text, similarity_score) tuples, sorted by score descending

        Examples:
            >>> candidates = ["張三", "章三", "李四", "王五"]
            >>> csoundex.find_similar("張三", candidates, threshold=0.5)
            [("張三", 1.0), ("章三", 0.75)]

        Complexity:
            Time: O(n*m) where n is candidates count, m is average text length
            Space: O(n)
        """
        results = []

        for candidate in candidates:
            sim = self.similarity(query, candidate, mode='fuzzy')
            if sim >= threshold:
                results.append((candidate, sim))

        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)

        # Return top-k if specified
        if topk is not None:
            results = results[:topk]

        return results

    def _is_chinese(self, char: str) -> bool:
        """
        Check if a character is Chinese (CJK Unified Ideographs).

        Args:
            char: Single character

        Returns:
            True if Chinese, False otherwise
        """
        if len(char) != 1:
            return False

        code = ord(char)
        return (
            0x4E00 <= code <= 0x9FFF or   # CJK Unified Ideographs
            0x3400 <= code <= 0x4DBF or   # CJK Extension A
            0x20000 <= code <= 0x2A6DF or # CJK Extension B
            0xF900 <= code <= 0xFAFF       # CJK Compatibility Ideographs
        )

    def clear_cache(self):
        """Clear LRU cache for encode_character."""
        self.encode_character.cache_clear()
        self.logger.debug("Cache cleared")

    def get_cache_info(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache hits, misses, size, maxsize
        """
        info = self.encode_character.cache_info()
        return {
            'hits': info.hits,
            'misses': info.misses,
            'size': info.currsize,
            'maxsize': info.maxsize
        }


def demo():
    """
    Demonstration of CSoundex functionality.

    Run this function to see CSoundex in action.
    """
    print("=" * 60)
    print("CSoundex - Chinese Soundex Phonetic Encoding Demo")
    print("=" * 60)

    # Initialize encoder
    csoundex = CSoundex()

    # Example 1: Basic encoding
    print("\n1. Basic Encoding:")
    examples = ["張三", "章三", "李四", "三聚氰胺", "信息檢索"]
    for text in examples:
        code = csoundex.encode(text)
        print(f"   {text:10s} → {code}")

    # Example 2: Homophone matching
    print("\n2. Homophone Detection:")
    pairs = [("張", "章"), ("李", "理"), ("王", "忘")]
    for c1, c2 in pairs:
        code1 = csoundex.encode(c1)
        code2 = csoundex.encode(c2)
        match = "✓" if code1 == code2 else "✗"
        print(f"   {c1} ({code1}) vs {c2} ({code2}): {match}")

    # Example 3: Similarity calculation
    print("\n3. Similarity Calculation:")
    comparisons = [("張三", "章三"), ("信息檢索", "資訊檢索"), ("你好", "再見")]
    for t1, t2 in comparisons:
        sim = csoundex.similarity(t1, t2, mode='fuzzy')
        print(f"   {t1} vs {t2}: {sim:.2f}")

    # Example 4: Finding similar texts
    print("\n4. Finding Similar Names:")
    query = "張偉"
    candidates = ["張偉", "章偉", "張維", "李偉", "王偉", "趙薇"]
    results = csoundex.find_similar(query, candidates, threshold=0.5, topk=3)
    print(f"   Query: {query}")
    for text, score in results:
        print(f"      {text}: {score:.2f}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run demo
    demo()
