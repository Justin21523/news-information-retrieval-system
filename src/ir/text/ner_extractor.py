"""
Named Entity Recognition (NER) Extractor for Traditional Chinese

This module provides a high-level interface for extracting and analyzing named
entities from Traditional Chinese text using CKIP Transformers.

Key Features:
    - 18 entity types (PERSON, GPE, ORG, LOC, DATE, TIME, etc.)
    - Entity filtering by type and confidence
    - Entity statistics and frequency analysis
    - Batch processing with GPU acceleration
    - LRU caching for frequently processed texts (10K entries)
    - Integration with ChineseTokenizer

Supported Entity Types (CKIP NER):
    PERSON      - Person names (人名)
    GPE         - Geopolitical entities (地緣政治實體)
    ORG         - Organizations (組織)
    LOC         - Locations (地點)
    DATE        - Dates (日期)
    TIME        - Times (時間)
    MONEY       - Monetary values (金錢)
    QUANTITY    - Quantities (數量)
    CARDINAL    - Cardinal numbers (基數)
    ORDINAL     - Ordinal numbers (序數)
    PERCENT     - Percentages (百分比)
    EVENT       - Events (事件)
    FAC         - Facilities (設施)
    LAW         - Laws (法律)
    LANGUAGE    - Languages (語言)
    NORP        - Nationalities/religions (國籍/宗教)
    PRODUCT     - Products (產品)
    WORK_OF_ART - Works of art (藝術作品)

Complexity:
    - extract(): O(n) where n = text length
    - extract_batch(): O(N*n) where N = number of texts
    - filter_by_type(): O(e) where e = number of entities
    - entity_statistics(): O(e)

Reference:
    CKIP Transformers: https://github.com/ckiplab/ckip-transformers
    Entity types based on OntoNotes 5.0 and Academia Sinica standards

Author: Information Retrieval System
License: Educational Use
"""

# Standard library imports (typing, lightweight data containers, counters, caching).
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
from collections import Counter, defaultdict
from functools import lru_cache
import logging

# Local dependency: `ChineseTokenizer` wraps the CKIP Transformer drivers and
# exposes a stable API for tokenization/POS/NER. Keeping NER calls inside the
# tokenizer centralizes optional heavy dependencies and lazy-loading behavior.
from .chinese_tokenizer import ChineseTokenizer


@dataclass
class Entity:
    """
    Named entity with metadata.

    Attributes:
        text: Entity text (e.g., "台灣大學")
        type: Entity type (e.g., "ORG")
        start_pos: Start character offset in the source text (0-based, inclusive)
        end_pos: End character offset in the source text (0-based, exclusive)
        source_text: Original text (optional)
    """
    text: str
    type: str
    start_pos: int
    end_pos: int
    source_text: Optional[str] = None

    def __str__(self) -> str:
        """Return a compact human-readable representation."""
        return f"{self.text} ({self.type})"

    def __repr__(self) -> str:
        """Return a debug-friendly representation."""
        return f"Entity(text='{self.text}', type='{self.type}', pos={self.start_pos}-{self.end_pos})"


class NERExtractor:
    """
    Named Entity Recognition extractor for Traditional Chinese.

    Provides high-level interface for extracting and analyzing entities from
    Chinese text using CKIP Transformers.

    Attributes:
        tokenizer: ChineseTokenizer instance
        entity_types: Set of entity types to extract (None = all types)
        logger: Logger instance

    Examples:
        >>> extractor = NERExtractor()
        >>> entities = extractor.extract("張三在台灣大學讀書")
        >>> print(entities)
        [Entity(text='張三', type='PERSON', pos=0-2),
         Entity(text='台灣大學', type='ORG', pos=3-7)]

        >>> # Filter by entity type
        >>> persons = extractor.filter_by_type(entities, ['PERSON'])
        >>> print(persons)
        [Entity(text='張三', type='PERSON', pos=0-2)]

        >>> # Get statistics
        >>> stats = extractor.entity_statistics(entities)
        >>> print(stats)
        {'total': 2, 'by_type': {'PERSON': 1, 'ORG': 1}, ...}
    """

    # All supported entity types
    ALL_ENTITY_TYPES = {
        'PERSON', 'GPE', 'ORG', 'LOC', 'DATE', 'TIME',
        'MONEY', 'QUANTITY', 'CARDINAL', 'ORDINAL', 'PERCENT',
        'EVENT', 'FAC', 'LAW', 'LANGUAGE', 'NORP', 'PRODUCT', 'WORK_OF_ART'
    }

    # Entity type descriptions (Chinese)
    ENTITY_TYPE_DESCRIPTIONS = {
        'PERSON': '人名',
        'GPE': '地緣政治實體',
        'ORG': '組織',
        'LOC': '地點',
        'DATE': '日期',
        'TIME': '時間',
        'MONEY': '金錢',
        'QUANTITY': '數量',
        'CARDINAL': '基數',
        'ORDINAL': '序數',
        'PERCENT': '百分比',
        'EVENT': '事件',
        'FAC': '設施',
        'LAW': '法律',
        'LANGUAGE': '語言',
        'NORP': '國籍/宗教',
        'PRODUCT': '產品',
        'WORK_OF_ART': '藝術作品'
    }

    def __init__(self,
                 entity_types: Optional[Set[str]] = None,
                 device: int = -1):
        """
        Initialize NER extractor.

        Args:
            entity_types: Set of entity types to extract (None = all types)
            device: Device for CKIP (-1 for CPU, 0+ for GPU)

        Raises:
            ImportError: If CKIP Transformers not available

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        # Logger is used to surface model initialization and runtime statistics.
        self.logger = logging.getLogger(__name__)

        # Keep a set of allowed entity labels so membership checks are O(1).
        self.entity_types = entity_types or self.ALL_ENTITY_TYPES

        # Device is forwarded to CKIP drivers (CPU=-1, GPU>=0).
        self.device = device

        # Initialize tokenizer with CKIP engine (NER depends on this engine).
        # `ChineseTokenizer` will lazily load the underlying CKIP NER model
        # only when `extract_entities*()` is called.
        self.tokenizer = ChineseTokenizer(engine='ckip', device=device)

        # Log the configuration to make debugging GPU/CPU routing easier.
        self.logger.info(
            f"NERExtractor initialized: device={device}, "
            f"entity_types={len(self.entity_types)}"
        )

    def extract(self, text: str, include_source: bool = True) -> List[Entity]:
        """
        Extract named entities from text.

        Args:
            text: Input text
            include_source: Whether to include source text in Entity objects

        Returns:
            List of Entity objects

        Complexity:
            Time: O(n) where n = text length
            Space: O(e) where e = number of entities

        Examples:
            >>> extractor = NERExtractor()
            >>> entities = extractor.extract("張三在台大讀書")
            >>> print(entities)
            [Entity(text='張三', type='PERSON', pos=0-2),
             Entity(text='台大', type='ORG', pos=3-5)]
        """
        # Treat empty/whitespace-only input as a no-op (common in pipelines).
        if not text or not text.strip():
            return []

        # Delegate extraction to `ChineseTokenizer` which wraps CKIP NER.
        # The returned offsets are character indices on the original text:
        #   (entity_text, entity_type, start_char, end_char)
        raw_entities = self.tokenizer.extract_entities(text)

        # Convert the low-level tuples into `Entity` objects and filter by type.
        # Filtering here avoids creating many objects that will be thrown away.
        entities = []
        for entity_text, entity_type, start_pos, end_pos in raw_entities:
            # Only keep types that the caller is interested in.
            if entity_type in self.entity_types:
                entity = Entity(
                    text=entity_text,  # Surface form produced by CKIP NER.
                    type=entity_type,  # Label from OntoNotes-like tag set.
                    start_pos=start_pos,  # Inclusive char offset in `text`.
                    end_pos=end_pos,  # Exclusive char offset in `text`.
                    # Keeping the full source text is convenient for debugging,
                    # but it increases memory (O(len(text)) per entity list).
                    source_text=text if include_source else None,
                )
                entities.append(entity)

        # Debug-level logs help spot unexpected entity counts per document.
        self.logger.debug(f"Extracted {len(entities)} entities from text")
        return entities

    def extract_batch(self,
                     texts: List[str],
                     batch_size: int = 32,
                     show_progress: bool = False,
                     include_source: bool = True) -> List[List[Entity]]:
        """
        Batch entity extraction for efficiency.

        Args:
            texts: List of texts to process
            batch_size: Batch size for processing
            show_progress: Show progress bar
            include_source: Whether to include source text in Entity objects

        Returns:
            List of entity lists, one per input text

        Complexity:
            Time: O(N*n) where N = number of texts, n = avg length
            Space: O(N*e) where e = avg entities per text

        Examples:
            >>> extractor = NERExtractor()
            >>> texts = ["張三在台大", "李四在台北"]
            >>> results = extractor.extract_batch(texts)
            >>> print(len(results))
            2
        """
        # Fast-path: empty input list yields empty output list.
        if not texts:
            return []

        # Batch extraction is typically faster than repeated single calls
        # because the underlying model can amortize overhead (especially on GPU).
        batch_raw_entities = self.tokenizer.extract_entities_batch(
            texts,
            batch_size=batch_size,
            show_progress=show_progress
        )

        # Convert each per-text result into a list of `Entity` objects.
        # We iterate with `zip()` to preserve the input order.
        batch_entities = []
        for text, raw_entities in zip(texts, batch_raw_entities):
            entities = []
            for entity_text, entity_type, start_pos, end_pos in raw_entities:
                # Apply the same type filter as `extract()`.
                if entity_type in self.entity_types:
                    entity = Entity(
                        text=entity_text,
                        type=entity_type,
                        start_pos=start_pos,
                        end_pos=end_pos,
                        source_text=text if include_source else None,
                    )
                    entities.append(entity)
            batch_entities.append(entities)

        # Info-level log summarizes throughput without spamming per-text details.
        self.logger.info(f"Batch extracted from {len(texts)} texts")
        return batch_entities

    @lru_cache(maxsize=10000)
    def extract_cached(self, text: str) -> Tuple[Tuple[str, str, int, int], ...]:
        """
        Cached entity extraction for frequently processed texts.

        Useful for repeated queries or document re-indexing.
        Returns tuple of tuples for hashability.

        Args:
            text: Input text

        Returns:
            Tuple of (entity_text, entity_type, start_pos, end_pos) tuples

        Complexity:
            Time: O(1) for cache hits, O(n) for cache misses
            Space: O(C*e) where C = cache size, e = avg entities

        Examples:
            >>> extractor = NERExtractor()
            >>> entities1 = extractor.extract_cached("張三在台大")
            >>> entities2 = extractor.extract_cached("張三在台大")  # Cache hit
            >>> assert entities1 is entities2  # Same object
        """
        # IMPORTANT: `lru_cache` requires the return value to be hashable.
        # We return a tuple-of-tuples instead of a list-of-tuples so the cached
        # object is immutable and can be safely shared across callers.
        #
        # Note: The cache key is (self, text). The cache is per-extractor
        # instance, so different `entity_types` configurations do not collide.
        raw_entities = self.tokenizer.extract_entities(text)

        # Filter by type and keep only primitive fields to minimize cache memory.
        filtered = [
            (entity_text, entity_type, start_pos, end_pos)
            for entity_text, entity_type, start_pos, end_pos in raw_entities
            if entity_type in self.entity_types
        ]

        # Tuple conversion makes the result hashable and enables cache hits.
        return tuple(filtered)

    def clear_cache(self):
        """Clear the NER extraction cache."""
        # Useful when processing long-running jobs where memory needs to be reset.
        self.extract_cached.cache_clear()
        self.logger.info("NER extraction cache cleared")

    def cache_info(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats (hits, misses, size, maxsize, hit_rate)

        Examples:
            >>> extractor = NERExtractor()
            >>> extractor.extract_cached("張三在台大")
            >>> extractor.extract_cached("張三在台大")  # Cache hit
            >>> info = extractor.cache_info()
            >>> print(info['hit_rate'])
            0.5
        """
        # `functools.lru_cache` exposes counters for monitoring cache behavior.
        info = self.extract_cached.cache_info()
        return {
            'hits': info.hits,
            'misses': info.misses,
            'size': info.currsize,
            'maxsize': info.maxsize,
            # Guard against division-by-zero for a never-used cache.
            'hit_rate': info.hits / (info.hits + info.misses) if (info.hits + info.misses) > 0 else 0.0
        }

    # ========================================================================
    # Entity Filtering
    # ========================================================================

    def filter_by_type(self,
                      entities: List[Entity],
                      types: List[str]) -> List[Entity]:
        """
        Filter entities by type.

        Args:
            entities: List of entities
            types: List of entity types to keep

        Returns:
            Filtered list of entities

        Complexity:
            Time: O(e) where e = number of entities
            Space: O(f) where f = number of filtered entities

        Examples:
            >>> extractor = NERExtractor()
            >>> entities = [
            ...     Entity('張三', 'PERSON', 0, 2),
            ...     Entity('台北', 'GPE', 3, 5)
            ... ]
            >>> persons = extractor.filter_by_type(entities, ['PERSON'])
            >>> print(len(persons))
            1
        """
        # Converting to a set makes membership tests O(1) rather than O(t).
        types_set = set(types)
        # List comprehension keeps the original entity ordering stable.
        return [e for e in entities if e.type in types_set]

    def filter_by_text(self,
                      entities: List[Entity],
                      patterns: List[str],
                      case_sensitive: bool = True) -> List[Entity]:
        """
        Filter entities by text pattern.

        Args:
            entities: List of entities
            patterns: List of text patterns to match (exact match)
            case_sensitive: Whether to use case-sensitive matching

        Returns:
            Filtered list of entities

        Complexity:
            Time: O(e*p) where e = entities, p = patterns
            Space: O(f) where f = filtered entities

        Examples:
            >>> extractor = NERExtractor()
            >>> entities = [
            ...     Entity('張三', 'PERSON', 0, 2),
            ...     Entity('李四', 'PERSON', 3, 5)
            ... ]
            >>> result = extractor.filter_by_text(entities, ['張三'])
            >>> print(len(result))
            1
        """
        # Exact matching is used here to keep the evaluator deterministic and fast.
        # Callers can pre-normalize patterns (e.g., regex) if they need fuzziness.
        if not case_sensitive:
            # Lowercasing both sides implements case-insensitive membership.
            patterns = [p.lower() for p in patterns]
            return [
                e for e in entities
                if e.text.lower() in patterns
            ]
        else:
            # Set membership is faster than scanning `patterns` for each entity.
            pattern_set = set(patterns)
            return [e for e in entities if e.text in pattern_set]

    # ========================================================================
    # Entity Statistics
    # ========================================================================

    def entity_statistics(self, entities: List[Entity]) -> Dict:
        """
        Compute statistics for extracted entities.

        Args:
            entities: List of entities

        Returns:
            Dictionary with statistics:
                - total: Total number of entities
                - by_type: Count by entity type
                - by_text: Count by entity text
                - unique_entities: Number of unique entity texts
                - type_distribution: Percentage distribution by type

        Complexity:
            Time: O(e) where e = number of entities
            Space: O(u) where u = unique entities

        Examples:
            >>> extractor = NERExtractor()
            >>> entities = [
            ...     Entity('張三', 'PERSON', 0, 2),
            ...     Entity('台大', 'ORG', 3, 5),
            ...     Entity('張三', 'PERSON', 6, 8)
            ... ]
            >>> stats = extractor.entity_statistics(entities)
            >>> print(stats['total'])
            3
            >>> print(stats['unique_entities'])
            2
        """
        # Empty input yields a fully-shaped, JSON-serializable result.
        if not entities:
            return {
                'total': 0,
                'by_type': {},
                'by_text': {},
                'unique_entities': 0,
                'type_distribution': {}
            }

        # Count entity labels (e.g., PERSON/ORG) for distribution analysis.
        type_counts = Counter(e.type for e in entities)

        # Count surface forms for frequency analysis (e.g., most common names).
        text_counts = Counter(e.text for e in entities)

        # Type distribution in percentage is convenient for dashboards/UI.
        total = len(entities)
        type_distribution = {
            etype: (count / total) * 100
            for etype, count in type_counts.items()
        }

        # Keep results simple (dict of primitives) for easy JSON export.
        return {
            'total': total,
            'by_type': dict(type_counts),
            'by_text': dict(text_counts),
            'unique_entities': len(text_counts),
            'type_distribution': type_distribution
        }

    def most_common_entities(self,
                           entities: List[Entity],
                           top_k: int = 10,
                           by_type: Optional[str] = None) -> List[Tuple[str, int]]:
        """
        Get most common entities.

        Args:
            entities: List of entities
            top_k: Number of top entities to return
            by_type: Filter by entity type (None = all types)

        Returns:
            List of (entity_text, count) tuples sorted by frequency

        Complexity:
            Time: O(e log k) where e = entities, k = top_k
            Space: O(u) where u = unique entities

        Examples:
            >>> extractor = NERExtractor()
            >>> entities = [
            ...     Entity('張三', 'PERSON', 0, 2),
            ...     Entity('張三', 'PERSON', 3, 5),
            ...     Entity('台大', 'ORG', 6, 8)
            ... ]
            >>> common = extractor.most_common_entities(entities, top_k=2)
            >>> print(common)
            [('張三', 2), ('台大', 1)]
        """
        # Optional filter to compute "most common" within a single entity label.
        if by_type:
            entities = self.filter_by_type(entities, [by_type])

        # Counter implements an efficient frequency table for hashable keys.
        text_counts = Counter(e.text for e in entities)
        # `most_common(k)` runs in O(u log k) where u is #unique keys.
        return text_counts.most_common(top_k)

    # ========================================================================
    # Utility Methods
    # ========================================================================

    def get_entity_types_in_text(self, text: str) -> Set[str]:
        """
        Get all entity types present in text.

        Args:
            text: Input text

        Returns:
            Set of entity types found

        Complexity:
            Time: O(n) where n = text length
            Space: O(t) where t = number of unique types

        Examples:
            >>> extractor = NERExtractor()
            >>> types = extractor.get_entity_types_in_text("張三在台大讀書")
            >>> print(types)
            {'PERSON', 'ORG'}
        """
        # Extract entities and project only their labels.
        entities = self.extract(text)
        return set(e.type for e in entities)

    def group_by_type(self, entities: List[Entity]) -> Dict[str, List[Entity]]:
        """
        Group entities by type.

        Args:
            entities: List of entities

        Returns:
            Dictionary mapping type -> list of entities

        Complexity:
            Time: O(e) where e = number of entities
            Space: O(e)

        Examples:
            >>> extractor = NERExtractor()
            >>> entities = [
            ...     Entity('張三', 'PERSON', 0, 2),
            ...     Entity('台大', 'ORG', 3, 5),
            ...     Entity('李四', 'PERSON', 6, 8)
            ... ]
            >>> grouped = extractor.group_by_type(entities)
            >>> print(len(grouped['PERSON']))
            2
        """
        # Use defaultdict to avoid repeated key existence checks.
        grouped = defaultdict(list)
        for entity in entities:
            # Preserve the original entity ordering within each type bucket.
            grouped[entity.type].append(entity)
        # Cast to a normal dict to keep the return type JSON-friendly.
        return dict(grouped)

    @staticmethod
    def get_entity_type_description(entity_type: str) -> str:
        """
        Get Chinese description for entity type.

        Args:
            entity_type: Entity type (e.g., 'PERSON')

        Returns:
            Chinese description (e.g., '人名')

        Examples:
            >>> desc = NERExtractor.get_entity_type_description('PERSON')
            >>> print(desc)
            人名
        """
        # Return the mapped Chinese description, falling back to the raw label.
        return NERExtractor.ENTITY_TYPE_DESCRIPTIONS.get(
            entity_type,
            entity_type
        )

    @staticmethod
    def print_entity_summary(entities: List[Entity], show_positions: bool = False):
        """
        Print a formatted summary of entities.

        Args:
            entities: List of entities
            show_positions: Whether to show positions

        Examples:
            >>> extractor = NERExtractor()
            >>> entities = [
            ...     Entity('張三', 'PERSON', 0, 2),
            ...     Entity('台大', 'ORG', 3, 5)
            ... ]
            >>> extractor.print_entity_summary(entities)
            Found 2 entities:
              PERSON (人名): 張三
              ORG (組織): 台大
        """
        # Printing is mainly for demos / notebooks; callers can build their own UI.
        if not entities:
            print("No entities found")
            return

        print(f"Found {len(entities)} entities:")
        for entity in entities:
            # Translate the tag into a Chinese label for readability.
            desc = NERExtractor.get_entity_type_description(entity.type)
            if show_positions:
                # Positions are [start, end) character spans, consistent with slicing.
                print(f"  {entity.type} ({desc}): {entity.text} [pos={entity.start_pos}-{entity.end_pos}]")
            else:
                print(f"  {entity.type} ({desc}): {entity.text}")


def demo():
    """Demonstration of NERExtractor capabilities."""
    # This demo intentionally uses small hard-coded examples so it can serve as
    # a quick sanity check in environments where CKIP is available.
    print("=" * 70)
    print("NER Extractor Demo (Traditional Chinese)")
    print("=" * 70)

    # Initialize extractor with default "all entity types" configuration.
    extractor = NERExtractor()

    # Sample texts cover people/organizations/dates/numbers to trigger many tags.
    texts = [
        "張三在國立臺灣大學圖書資訊學系讀書",
        "2025年一月台北將舉辦國際研討會",
        "Google和Meta是知名的科技公司",
        "這本書定價500元，打八折後是400元"
    ]

    # Example 1: Single text extraction (simplest API).
    print("\n[1] Single Text Extraction:")
    print("-" * 70)
    text = texts[0]
    entities = extractor.extract(text)
    print(f"Text: {text}")
    extractor.print_entity_summary(entities, show_positions=True)

    # Example 2: Batch extraction (preferred for large corpora).
    print("\n[2] Batch Extraction:")
    print("-" * 70)
    batch_entities = extractor.extract_batch(texts)
    for i, (text, entities) in enumerate(zip(texts, batch_entities), 1):
        print(f"\nText {i}: {text}")
        print(f"Entities: {len(entities)}")
        for entity in entities:
            desc = NERExtractor.get_entity_type_description(entity.type)
            print(f"  - {entity.text} ({entity.type}/{desc})")

    # Example 3: Filter by type across the whole batch.
    print("\n[3] Filter by Type:")
    print("-" * 70)
    all_entities = [e for entities in batch_entities for e in entities]
    persons = extractor.filter_by_type(all_entities, ['PERSON'])
    orgs = extractor.filter_by_type(all_entities, ['ORG'])
    print(f"PERSON entities: {[e.text for e in persons]}")
    print(f"ORG entities: {[e.text for e in orgs]}")

    # Example 4: Basic descriptive statistics.
    print("\n[4] Entity Statistics:")
    print("-" * 70)
    stats = extractor.entity_statistics(all_entities)
    print(f"Total entities: {stats['total']}")
    print(f"Unique entities: {stats['unique_entities']}")
    print("\nBy Type:")
    for etype, count in stats['by_type'].items():
        desc = NERExtractor.get_entity_type_description(etype)
        pct = stats['type_distribution'][etype]
        print(f"  {etype:12s} ({desc:10s}): {count:2d} ({pct:.1f}%)")

    # Example 5: Frequency analysis (top-k surface forms).
    print("\n[5] Most Common Entities:")
    print("-" * 70)
    common = extractor.most_common_entities(all_entities, top_k=5)
    for i, (text, count) in enumerate(common, 1):
        print(f"{i}. {text}: {count} occurrences")

    # Example 6: Group-by useful for UI display and downstream processing.
    print("\n[6] Group by Type:")
    print("-" * 70)
    grouped = extractor.group_by_type(all_entities)
    for etype, entities in grouped.items():
        desc = NERExtractor.get_entity_type_description(etype)
        texts = [e.text for e in entities]
        print(f"{etype} ({desc}): {', '.join(texts)}")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    demo()
