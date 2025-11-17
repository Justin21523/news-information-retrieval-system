"""
TextRank Keyword Extraction

This module implements the TextRank algorithm for keyword extraction with
enhancements for Traditional Chinese text based on recent research (2025).

Key Features:
    - Graph-based PageRank algorithm for keyword ranking
    - Co-occurrence window for building word graph
    - Support for Chinese text with proper tokenization
    - Position-based weighting (2025 improvement)
    - POS filtering for noun/verb extraction
    - Multi-word keyphrase extraction

Algorithm Overview:
    1. Tokenize text and filter stopwords
    2. Build word co-occurrence graph (window-based)
    3. Run PageRank to compute word scores
    4. Extract top-k keywords or keyphrases

Complexity:
    - Time: O(V² + I×V) where V=unique words, I=iterations
    - Space: O(V² + E) where E=edges in graph

References:
    Mihalcea & Tarau (2004). "TextRank: Bringing Order into Text"
    Chen et al. (2025). "An Improved Chinese Keyword Extraction Algorithm
        Based on Complex Networks" (位置權重 position weighting +6.3% precision)

Author: Information Retrieval System
License: Educational Use
"""

from typing import List, Set, Tuple, Optional, Dict
import logging
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass
import math

# NetworkX for graph algorithms
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logging.warning("NetworkX not available. Install with: pip install networkx")

# Import our Chinese processing modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ir.text.chinese_tokenizer import ChineseTokenizer
from ir.text.stopwords import StopwordsFilter
from ir.text.ner_extractor import NERExtractor, Entity


@dataclass
class Keyword:
    """
    Keyword extraction result.

    Attributes:
        word: The keyword or keyphrase
        score: TextRank score (0.0 to 1.0)
        positions: List of positions where keyword appears
        frequency: Number of occurrences
    """
    word: str
    score: float
    positions: List[int]
    frequency: int

    def __repr__(self):
        return f"Keyword(word='{self.word}', score={self.score:.4f}, freq={self.frequency})"


class TextRankExtractor:
    """
    TextRank keyword extractor with Chinese language support.

    Implements graph-based ranking using PageRank algorithm with
    improvements for Traditional Chinese text processing.

    Attributes:
        window_size: Co-occurrence window size (default: 5)
        damping_factor: PageRank damping factor (default: 0.85)
        max_iterations: Max PageRank iterations (default: 100)
        convergence_threshold: Convergence threshold (default: 1e-4)
        use_position_weight: Enable position-based weighting (2025)
        pos_filter: POS tags to keep (e.g., ['N', 'V'] for nouns/verbs)

    Examples:
        >>> extractor = TextRankExtractor(window_size=5)
        >>> text = "機器學習是人工智慧的重要分支，深度學習是機器學習的子領域。"
        >>> keywords = extractor.extract(text, top_k=5)
        >>> for kw in keywords:
        ...     print(f"{kw.word}: {kw.score:.4f}")
        機器學習: 0.2341
        深度學習: 0.1872
        人工智慧: 0.1654
    """

    def __init__(self,
                 window_size: int = 5,
                 damping_factor: float = 0.85,
                 max_iterations: int = 100,
                 convergence_threshold: float = 1e-4,
                 use_position_weight: bool = True,
                 use_ner_boost: bool = False,
                 ner_boost_weight: float = 0.3,
                 ner_entity_types: Optional[List[str]] = None,
                 pos_filter: Optional[List[str]] = None,
                 tokenizer_engine: str = 'auto',
                 stopwords_file: Optional[str] = None,
                 device: int = -1):
        """
        Initialize TextRank extractor.

        Args:
            window_size: Co-occurrence window size (2-10 recommended)
            damping_factor: PageRank damping (0.85 is standard)
            max_iterations: Max iterations for PageRank convergence
            convergence_threshold: Stop when score change < threshold
            use_position_weight: Enable position weighting (2025 improvement)
            use_ner_boost: Enable NER entity weight boosting
            ner_boost_weight: Boost multiplier for entity words (0.2-0.5 recommended)
            ner_entity_types: Entity types to boost (default: PERSON, ORG, GPE, LOC)
                             Use None for all entity types
            pos_filter: Keep only these POS tags (e.g., ['N', 'V'])
                       Use None to keep all words except stopwords
            tokenizer_engine: 'ckip' | 'jieba' | 'auto'
            stopwords_file: Custom stopwords file path
            device: GPU device (-1 for CPU, 0+ for GPU)

        Raises:
            ImportError: If NetworkX is not installed
        """
        if not NETWORKX_AVAILABLE:
            raise ImportError(
                "NetworkX is required for TextRank. "
                "Install with: pip install networkx"
            )

        self.logger = logging.getLogger(__name__)

        # Algorithm parameters
        self.window_size = window_size
        self.damping_factor = damping_factor
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.use_position_weight = use_position_weight
        self.use_ner_boost = use_ner_boost
        self.ner_boost_weight = ner_boost_weight
        self.pos_filter = set(pos_filter) if pos_filter else None

        # NER entity types to boost (default to important entity types)
        if ner_entity_types is None and use_ner_boost:
            # Default: boost person names, organizations, locations, and geopolitical entities
            self.ner_entity_types = {'PERSON', 'ORG', 'GPE', 'LOC'}
        else:
            self.ner_entity_types = set(ner_entity_types) if ner_entity_types else None

        # Chinese NLP components
        use_pos = bool(pos_filter)  # Enable POS tagging if filtering
        self.tokenizer = ChineseTokenizer(
            engine=tokenizer_engine,
            use_pos=use_pos,
            device=device
        )
        self.stopwords_filter = StopwordsFilter(stopwords_file=stopwords_file)

        # NER extractor (lazy initialization)
        self.ner_extractor = None
        if use_ner_boost:
            # Only initialize if CKIP engine is available
            if tokenizer_engine in ['ckip', 'auto']:
                try:
                    self.ner_extractor = NERExtractor(
                        entity_types=self.ner_entity_types,
                        device=device
                    )
                    self.logger.info(
                        f"NER entity boosting enabled for types: {self.ner_entity_types}"
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Failed to initialize NER extractor: {e}. "
                        f"NER boosting will be disabled."
                    )
                    self.use_ner_boost = False
            else:
                self.logger.warning(
                    f"NER boosting requires CKIP engine, but got {tokenizer_engine}. "
                    f"NER boosting will be disabled."
                )
                self.use_ner_boost = False

        self.logger.info(
            f"TextRankExtractor initialized: window={window_size}, "
            f"damping={damping_factor}, position_weight={use_position_weight}, "
            f"ner_boost={use_ner_boost}, pos_filter={pos_filter}, "
            f"tokenizer={tokenizer_engine}"
        )

    # ========================================================================
    # Core Extraction Methods
    # ========================================================================

    def extract(self,
                text: str,
                top_k: int = 10,
                return_scores: bool = True) -> List[Keyword]:
        """
        Extract keywords from text using TextRank.

        Args:
            text: Input text (Chinese or English)
            top_k: Number of keywords to extract
            return_scores: Include TextRank scores in results

        Returns:
            List of Keyword objects sorted by score (descending)

        Complexity:
            Time: O(n + V² + I×V) where:
                  n = text length (tokenization)
                  V = unique words (graph construction)
                  I = PageRank iterations
            Space: O(V² + E) for graph storage

        Examples:
            >>> extractor = TextRankExtractor()
            >>> text = "深度學習是機器學習的一個分支"
            >>> keywords = extractor.extract(text, top_k=3)
            >>> print([kw.word for kw in keywords])
            ['深度學習', '機器學習', '分支']
        """
        if not text or not text.strip():
            self.logger.warning("Empty text provided")
            return []

        # Step 1: Tokenize and filter
        tokens = self._preprocess_text(text)

        if len(tokens) < 2:
            self.logger.warning(f"Too few tokens after filtering: {len(tokens)}")
            return []

        # Step 2: Build co-occurrence graph
        graph = self._build_graph(tokens)

        if graph.number_of_nodes() == 0:
            self.logger.warning("Empty graph after construction")
            return []

        # Step 3: Run PageRank
        scores = self._run_pagerank(graph)

        # Step 4: Apply position weighting if enabled
        if self.use_position_weight:
            scores = self._apply_position_weighting(scores, tokens)

        # Step 5: Apply NER entity boosting if enabled
        if self.use_ner_boost and self.ner_extractor:
            scores = self._apply_ner_boost(scores, text)

        # Step 6: Collect word statistics
        keywords = self._build_keywords(scores, tokens)

        # Step 7: Sort and return top-k
        keywords.sort(key=lambda kw: kw.score, reverse=True)

        self.logger.info(
            f"Extracted {len(keywords[:top_k])} keywords from "
            f"{len(tokens)} tokens ({graph.number_of_nodes()} nodes, "
            f"{graph.number_of_edges()} edges)"
        )

        return keywords[:top_k]

    def extract_keyphrases(self,
                          text: str,
                          top_k: int = 10,
                          max_phrase_length: int = 4) -> List[Keyword]:
        """
        Extract multi-word keyphrases using TextRank scores.

        Algorithm:
            1. Run TextRank to get word scores
            2. Find sequences of high-scoring adjacent words
            3. Combine into keyphrases (up to max_phrase_length)
            4. Score phrases by average word score

        Args:
            text: Input text
            top_k: Number of keyphrases to extract
            max_phrase_length: Max words per phrase (2-4 recommended)

        Returns:
            List of Keyword objects for keyphrases

        Complexity:
            Time: O(n×m) where m = max_phrase_length
            Space: O(p) where p = number of phrases

        Examples:
            >>> extractor = TextRankExtractor()
            >>> text = "機器學習和深度學習是人工智慧的重要技術"
            >>> phrases = extractor.extract_keyphrases(text, top_k=3)
            >>> print([kw.word for kw in phrases])
            ['機器學習', '深度學習', '人工智慧']
        """
        if not text or not text.strip():
            return []

        # Get word-level scores
        word_keywords = self.extract(text, top_k=100, return_scores=True)
        word_scores = {kw.word: kw.score for kw in word_keywords}

        # Tokenize to get original sequence
        tokens = self._preprocess_text(text)

        # Extract phrases
        phrases = self._extract_phrase_candidates(
            tokens, word_scores, max_phrase_length
        )

        # Sort and return top-k
        phrases.sort(key=lambda kw: kw.score, reverse=True)

        self.logger.info(f"Extracted {len(phrases[:top_k])} keyphrases")

        return phrases[:top_k]

    # ========================================================================
    # Preprocessing
    # ========================================================================

    def _preprocess_text(self, text: str) -> List[Tuple[str, int]]:
        """
        Tokenize and filter text.

        Args:
            text: Input text

        Returns:
            List of (word, position) tuples

        Complexity:
            Time: O(n) where n = text length
            Space: O(t) where t = number of tokens
        """
        # Tokenize with POS if needed
        if self.pos_filter:
            tokens_with_pos = self.tokenizer.tokenize_with_pos(text)

            # Filter by POS tags (case-insensitive matching)
            # Jieba uses lowercase: 'n', 'v', 'a'
            # CKIP uses uppercase: 'Nc', 'VCL', 'A'
            filtered = []
            for i, (word, pos) in enumerate(tokens_with_pos):
                # Check if POS starts with any allowed prefix (case-insensitive)
                # E.g., 'Nc' matches filter=['N'], 'n' matches filter=['N']
                if any(pos.upper().startswith(tag.upper()) for tag in self.pos_filter):
                    if not self.stopwords_filter.is_stopword(word):
                        filtered.append((word, i))
        else:
            # Just tokenize and filter stopwords
            tokens = self.tokenizer.tokenize(text)
            filtered = [
                (word, i) for i, word in enumerate(tokens)
                if not self.stopwords_filter.is_stopword(word)
            ]

        self.logger.debug(
            f"Preprocessed text: {len(filtered)} tokens after filtering"
        )

        return filtered

    # ========================================================================
    # Graph Construction
    # ========================================================================

    def _build_graph(self, tokens: List[Tuple[str, int]]) -> nx.Graph:
        """
        Build co-occurrence graph from tokens.

        Algorithm:
            - For each token, connect it to tokens within window_size
            - Edge weight = co-occurrence count

        Args:
            tokens: List of (word, position) tuples

        Returns:
            Undirected weighted graph

        Complexity:
            Time: O(n×w) where n=tokens, w=window_size
            Space: O(V² + E) in worst case
        """
        graph = nx.Graph()

        # Add all nodes first
        words = [word for word, _ in tokens]
        graph.add_nodes_from(set(words))

        # Build edges within sliding window
        for i in range(len(tokens)):
            word_i, _ = tokens[i]

            # Look ahead within window
            for j in range(i + 1, min(i + self.window_size, len(tokens))):
                word_j, _ = tokens[j]

                if word_i != word_j:
                    if graph.has_edge(word_i, word_j):
                        # Increment edge weight
                        graph[word_i][word_j]['weight'] += 1
                    else:
                        # Create new edge
                        graph.add_edge(word_i, word_j, weight=1)

        self.logger.debug(
            f"Built graph: {graph.number_of_nodes()} nodes, "
            f"{graph.number_of_edges()} edges"
        )

        return graph

    # ========================================================================
    # PageRank
    # ========================================================================

    def _run_pagerank(self, graph: nx.Graph) -> Dict[str, float]:
        """
        Run PageRank algorithm on graph.

        Uses NetworkX's built-in PageRank implementation with
        edge weights for co-occurrence counts.

        Args:
            graph: Word co-occurrence graph

        Returns:
            Dictionary mapping word -> score

        Complexity:
            Time: O(I×V) where I=iterations, V=nodes
            Space: O(V)
        """
        try:
            scores = nx.pagerank(
                graph,
                alpha=self.damping_factor,
                max_iter=self.max_iterations,
                tol=self.convergence_threshold,
                weight='weight'
            )
        except nx.PowerIterationFailedConvergence:
            self.logger.warning(
                f"PageRank did not converge in {self.max_iterations} iterations"
            )
            # Return uniform scores as fallback
            scores = {node: 1.0 / graph.number_of_nodes()
                     for node in graph.nodes()}

        return scores

    # ========================================================================
    # Position Weighting (2025 Improvement)
    # ========================================================================

    def _apply_position_weighting(self,
                                  scores: Dict[str, float],
                                  tokens: List[Tuple[str, int]]) -> Dict[str, float]:
        """
        Apply position-based weighting to TextRank scores.

        2025 Research Finding:
            Words appearing earlier in text (title, first paragraph)
            are more likely to be keywords. Position weighting improves
            precision by +6.3% (Chen et al. 2025).

        Formula:
            weighted_score = base_score × (1 + α × position_weight)
            position_weight = 1 - (first_position / total_positions)

        Args:
            scores: Base TextRank scores
            tokens: List of (word, position) tuples

        Returns:
            Updated scores with position weighting

        Complexity:
            Time: O(n) where n = tokens
            Space: O(V) where V = unique words
        """
        # Collect first position for each word
        first_positions = {}
        for word, pos in tokens:
            if word in scores and word not in first_positions:
                first_positions[word] = pos

        total_positions = len(tokens)
        alpha = 0.3  # Position weight strength (0.2-0.5 recommended)

        # Apply position weighting
        weighted_scores = {}
        for word, score in scores.items():
            if word in first_positions:
                # Earlier position = higher weight
                pos_weight = 1 - (first_positions[word] / total_positions)
                weighted_scores[word] = score * (1 + alpha * pos_weight)
            else:
                weighted_scores[word] = score

        return weighted_scores

    # ========================================================================
    # NER Entity Boosting
    # ========================================================================

    def _apply_ner_boost(self,
                        scores: Dict[str, float],
                        text: str) -> Dict[str, float]:
        """
        Apply NER entity weight boosting to TextRank scores.

        Research Rationale:
            Named entities (people, organizations, locations) are often key
            concepts in a document. Boosting entity words improves keyword
            relevance for domain-specific and news texts.

        Formula:
            boosted_score = base_score × (1 + β × entity_boost)
            entity_boost = 1 if word is part of entity, 0 otherwise

        Args:
            scores: Base TextRank scores
            text: Original input text

        Returns:
            Updated scores with NER boosting

        Complexity:
            Time: O(n + e) where n = text length, e = entities
            Space: O(e) for entity set
        """
        if not self.ner_extractor:
            return scores

        # Extract entities from text
        try:
            entities = self.ner_extractor.extract(text)
        except Exception as e:
            self.logger.warning(f"NER extraction failed: {e}")
            return scores

        if not entities:
            self.logger.debug("No entities found for boosting")
            return scores

        # Collect entity words
        entity_words = set()
        for entity in entities:
            # Add the full entity text
            entity_words.add(entity.text)
            # Also add individual characters/words for Chinese
            # (in case tokenization splits entity differently)
            for char in entity.text:
                if len(char.strip()) > 0:
                    entity_words.add(char)

        self.logger.debug(
            f"Found {len(entities)} entities with {len(entity_words)} unique words"
        )

        # Apply boosting
        beta = self.ner_boost_weight  # Boost strength
        boosted_scores = {}

        for word, score in scores.items():
            if word in entity_words:
                # Boost entity words
                boosted_scores[word] = score * (1 + beta)
            else:
                boosted_scores[word] = score

        # Count how many words were boosted
        boosted_count = sum(1 for w in scores if w in entity_words)
        self.logger.debug(
            f"Applied NER boost to {boosted_count}/{len(scores)} words "
            f"(β={beta:.2f})"
        )

        return boosted_scores

    # ========================================================================
    # Keyword Construction
    # ========================================================================

    def _build_keywords(self,
                       scores: Dict[str, float],
                       tokens: List[Tuple[str, int]]) -> List[Keyword]:
        """
        Build Keyword objects from scores and tokens.

        Collects:
            - Word scores from TextRank
            - All positions where word appears
            - Frequency counts

        Args:
            scores: Word scores from PageRank
            tokens: List of (word, position) tuples

        Returns:
            List of Keyword objects
        """
        # Collect statistics
        word_positions = defaultdict(list)
        word_frequencies = Counter()

        for word, pos in tokens:
            if word in scores:
                word_positions[word].append(pos)
                word_frequencies[word] += 1

        # Build Keyword objects
        keywords = []
        for word, score in scores.items():
            keywords.append(Keyword(
                word=word,
                score=score,
                positions=word_positions[word],
                frequency=word_frequencies[word]
            ))

        return keywords

    # ========================================================================
    # Keyphrase Extraction
    # ========================================================================

    def _extract_phrase_candidates(self,
                                   tokens: List[Tuple[str, int]],
                                   word_scores: Dict[str, float],
                                   max_length: int) -> List[Keyword]:
        """
        Extract multi-word keyphrases from tokens.

        Algorithm:
            1. Find sequences of words all appearing in word_scores
            2. Combine into phrases up to max_length
            3. Score phrase by average word score

        Args:
            tokens: List of (word, position) tuples
            word_scores: Scores for individual words
            max_length: Maximum phrase length

        Returns:
            List of Keyword objects for phrases
        """
        phrases = []
        i = 0

        while i < len(tokens):
            word_i, pos_i = tokens[i]

            if word_i not in word_scores:
                i += 1
                continue

            # Try to extend phrase
            phrase_words = [word_i]
            phrase_positions = [pos_i]

            j = i + 1
            while j < len(tokens) and len(phrase_words) < max_length:
                word_j, pos_j = tokens[j]

                if word_j in word_scores:
                    phrase_words.append(word_j)
                    phrase_positions.append(pos_j)
                    j += 1
                else:
                    break

            # Create phrase if multi-word
            if len(phrase_words) >= 2:
                phrase_text = ''.join(phrase_words)  # Chinese no spaces
                phrase_score = sum(word_scores[w] for w in phrase_words) / len(phrase_words)

                phrases.append(Keyword(
                    word=phrase_text,
                    score=phrase_score,
                    positions=phrase_positions,
                    frequency=1  # Count phrases once
                ))

            i = j if j > i + 1 else i + 1

        return phrases

    # ========================================================================
    # Statistics
    # ========================================================================

    def get_config(self) -> dict:
        """Get current configuration."""
        return {
            'window_size': self.window_size,
            'damping_factor': self.damping_factor,
            'max_iterations': self.max_iterations,
            'convergence_threshold': self.convergence_threshold,
            'use_position_weight': self.use_position_weight,
            'use_ner_boost': self.use_ner_boost,
            'ner_boost_weight': self.ner_boost_weight,
            'ner_entity_types': list(self.ner_entity_types) if self.ner_entity_types else None,
            'pos_filter': list(self.pos_filter) if self.pos_filter else None,
            'tokenizer_engine': self.tokenizer.engine,
            'stopwords_count': len(self.stopwords_filter)
        }


def demo():
    """Demonstration of TextRank keyword extraction."""
    print("=" * 70)
    print("TextRank Keyword Extraction Demo")
    print("=" * 70)

    # Sample Traditional Chinese text
    text = """
    機器學習是人工智慧的重要分支，它讓電腦能夠從資料中學習模式。
    深度學習是機器學習的子領域，使用神經網路來建立複雜的模型。
    自然語言處理是人工智慧的另一個重要應用，涉及文字分析和理解。
    資訊檢索系統使用機器學習技術來改善搜尋結果的品質。
    """

    # Initialize extractor
    print("\n[1] Initialize TextRank Extractor")
    print("-" * 70)
    extractor = TextRankExtractor(
        window_size=5,
        use_position_weight=True,
        pos_filter=['N', 'V'],  # Nouns and verbs
        tokenizer_engine='jieba'  # Fast for demo
    )
    print(f"Config: {extractor.get_config()}")

    # Extract keywords
    print("\n[2] Extract Keywords (Top 10)")
    print("-" * 70)
    keywords = extractor.extract(text, top_k=10)

    for i, kw in enumerate(keywords, 1):
        print(f"{i:2d}. {kw.word:12s}  score={kw.score:.4f}  freq={kw.frequency}")

    # Extract keyphrases
    print("\n[3] Extract Keyphrases (Top 5)")
    print("-" * 70)
    keyphrases = extractor.extract_keyphrases(text, top_k=5, max_phrase_length=3)

    for i, kw in enumerate(keyphrases, 1):
        print(f"{i}. {kw.word:20s}  score={kw.score:.4f}")

    # Compare with/without position weighting
    print("\n[4] Position Weighting Comparison")
    print("-" * 70)

    extractor_no_pos = TextRankExtractor(
        window_size=5,
        use_position_weight=False,
        tokenizer_engine='jieba'
    )

    keywords_no_pos = extractor_no_pos.extract(text, top_k=5)

    print("With position weighting:")
    for kw in keywords[:5]:
        print(f"  {kw.word:12s}  {kw.score:.4f}")

    print("\nWithout position weighting:")
    for kw in keywords_no_pos:
        print(f"  {kw.word:12s}  {kw.score:.4f}")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    demo()
