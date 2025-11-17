"""
Dependency Parsing and SVO Extraction for Traditional Chinese

This module provides dependency parsing using SuPar and extracts
Subject-Verb-Object (SVO) triples for Chinese text analysis.

Key Features:
    - Dependency parsing with pre-trained models
    - SVO triple extraction
    - Sentence structure analysis
    - Integration with ChineseTokenizer

Complexity:
    - Parsing: O(n³) where n = sentence length
    - SVO extraction: O(n) where n = tree nodes

References:
    - Zhang et al. (2020). "SuPar: A Python Library for Structured Prediction"
    - Dozat & Manning (2017). "Deep Biaffine Attention for Neural Dependency Parsing"

Author: Information Retrieval System
License: Educational Use
"""

from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import logging

# SuPar for dependency parsing
try:
    from supar import Parser
    SUPAR_AVAILABLE = True
except ImportError:
    SUPAR_AVAILABLE = False
    logging.warning("SuPar not available. Install with: pip install supar")

# Chinese tokenizer
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from ir.text.chinese_tokenizer import ChineseTokenizer
    TOKENIZER_AVAILABLE = True
except ImportError:
    TOKENIZER_AVAILABLE = False


@dataclass
class DependencyEdge:
    """
    Dependency edge in parse tree.

    Attributes:
        head_index: Index of head word (0 = ROOT)
        dependent_index: Index of dependent word
        head_word: Head word text
        dependent_word: Dependent word text
        relation: Dependency relation label
    """
    head_index: int
    dependent_index: int
    head_word: str
    dependent_word: str
    relation: str

    def __repr__(self):
        return f"{self.dependent_word} --{self.relation}--> {self.head_word}"


@dataclass
class SVOTriple:
    """
    Subject-Verb-Object triple.

    Attributes:
        subject: Subject phrase
        verb: Verb phrase
        object: Object phrase (optional)
        subject_index: Token index of subject
        verb_index: Token index of verb
        object_index: Token index of object (None if no object)
        confidence: Extraction confidence score (0-1)
    """
    subject: str
    verb: str
    object: Optional[str] = None
    subject_index: Optional[int] = None
    verb_index: Optional[int] = None
    object_index: Optional[int] = None
    confidence: float = 1.0

    def __repr__(self):
        if self.object:
            return f"({self.subject}, {self.verb}, {self.object})"
        else:
            return f"({self.subject}, {self.verb})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'subject': self.subject,
            'verb': self.verb,
            'object': self.object,
            'subject_index': self.subject_index,
            'verb_index': self.verb_index,
            'object_index': self.object_index,
            'confidence': self.confidence
        }


class DependencyParser:
    """
    Dependency parser using SuPar.

    Wrapper for SuPar dependency parsing with support for
    Traditional Chinese text.

    Attributes:
        model_name: SuPar model name
        parser: SuPar Parser instance
        tokenizer: ChineseTokenizer instance
        logger: Logger instance

    Examples:
        >>> parser = DependencyParser()
        >>> edges = parser.parse("我喜歡吃蘋果")
        >>> for edge in edges:
        ...     print(edge)
        我 --nsubj--> 喜歡
        喜歡 --ROOT--> ROOT
        吃 --xcomp--> 喜歡
        蘋果 --dobj--> 吃
    """

    def __init__(self,
                 model_name: str = 'biaffine-dep-zh',
                 tokenizer_engine: str = 'jieba',
                 device: int = -1):
        """
        Initialize dependency parser.

        Args:
            model_name: SuPar model name
                - 'biaffine-dep-zh': Chinese dependency (default)
                - 'biaffine-dep-en': English dependency
            tokenizer_engine: 'jieba' | 'ckip' | 'auto'
            device: Device for parsing (-1=CPU, 0+=GPU)

        Raises:
            ImportError: If SuPar not available
        """
        if not SUPAR_AVAILABLE:
            raise ImportError(
                "SuPar is required. Install with: pip install supar"
            )

        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        self.device = device

        # Initialize tokenizer
        if TOKENIZER_AVAILABLE:
            self.tokenizer = ChineseTokenizer(engine=tokenizer_engine)
        else:
            self.logger.warning("ChineseTokenizer not available")
            self.tokenizer = None

        # Load SuPar parser (lazy loading)
        self.parser = None
        self._load_parser()

        self.logger.info(
            f"DependencyParser initialized: model={model_name}, "
            f"tokenizer={tokenizer_engine}, device={device}"
        )

    def _load_parser(self):
        """Load SuPar parser model with PyTorch 2.6+ compatibility."""
        try:
            import torch

            # PyTorch 2.6+ changed weights_only default to True, which breaks SuPar model loading
            # We need to temporarily patch torch.load to use weights_only=False
            # This is safe since SuPar models are from a trusted source
            original_load = torch.load

            def patched_load(*args, **kwargs):
                # Force weights_only=False for SuPar models
                kwargs['weights_only'] = False
                return original_load(*args, **kwargs)

            # Apply the patch
            torch.load = patched_load

            try:
                self.logger.info(f"Loading SuPar model: {self.model_name}...")
                self.parser = Parser.load(self.model_name)
            finally:
                # Restore original torch.load
                torch.load = original_load

            # Move to device
            if self.device >= 0:
                self.parser = self.parser.to(f'cuda:{self.device}')

            self.logger.info("Parser loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load parser: {e}")
            raise

    # ========================================================================
    # Core Parsing Methods
    # ========================================================================

    def parse(self,
              text: str,
              preprocess: bool = True) -> List[DependencyEdge]:
        """
        Parse text and return dependency edges.

        Args:
            text: Input text (Chinese sentence)
            preprocess: Apply tokenization preprocessing

        Returns:
            List of DependencyEdge objects

        Complexity:
            Time: O(n³) where n = sentence length
            Space: O(n²)

        Examples:
            >>> parser = DependencyParser()
            >>> edges = parser.parse("張三喜歡閱讀")
            >>> for edge in edges:
            ...     print(f"{edge.dependent_word} --{edge.relation}--> {edge.head_word}")
        """
        if not text or not text.strip():
            self.logger.warning("Empty text provided")
            return []

        # Preprocess: tokenize if needed
        if preprocess and self.tokenizer:
            tokens = self.tokenizer.tokenize(text)
        else:
            # Simple whitespace split or character split
            tokens = text.split() if ' ' in text else list(text)

        # Parse with SuPar
        try:
            # SuPar expects list of token lists
            dataset = self.parser.predict([[token for token in tokens]], verbose=False)

            # Extract dependency edges
            edges = []

            # Get first (and only) sentence
            sentence = dataset.sentences[0]

            for i, (word, head, relation) in enumerate(
                zip(sentence.values[1], sentence.values[6], sentence.values[7])
            ):
                # Skip ROOT
                if head == 0:
                    edges.append(DependencyEdge(
                        head_index=0,
                        dependent_index=i + 1,
                        head_word='ROOT',
                        dependent_word=word,
                        relation=relation
                    ))
                else:
                    edges.append(DependencyEdge(
                        head_index=head,
                        dependent_index=i + 1,
                        head_word=sentence.values[1][head - 1],
                        dependent_word=word,
                        relation=relation
                    ))

            self.logger.debug(f"Parsed {len(tokens)} tokens, {len(edges)} edges")
            return edges

        except Exception as e:
            self.logger.error(f"Parsing failed: {e}")
            return []

    def parse_batch(self,
                   texts: List[str],
                   preprocess: bool = True) -> List[List[DependencyEdge]]:
        """
        Parse multiple texts in batch.

        Args:
            texts: List of text strings
            preprocess: Apply tokenization preprocessing

        Returns:
            List of dependency edge lists

        Examples:
            >>> parser = DependencyParser()
            >>> texts = ["我喜歡你", "他學習中文"]
            >>> results = parser.parse_batch(texts)
        """
        results = []
        for text in texts:
            edges = self.parse(text, preprocess=preprocess)
            results.append(edges)

        self.logger.info(f"Batch parsed {len(texts)} texts")
        return results

    # ========================================================================
    # Analysis Methods
    # ========================================================================

    def get_dependency_tree(self,
                           text: str,
                           preprocess: bool = True) -> Dict[int, List[DependencyEdge]]:
        """
        Get dependency tree as adjacency list.

        Args:
            text: Input text
            preprocess: Apply tokenization

        Returns:
            Dictionary mapping head_index -> [edges]

        Examples:
            >>> parser = DependencyParser()
            >>> tree = parser.get_dependency_tree("我愛你")
            >>> # tree[0] = edges from ROOT
            >>> # tree[2] = edges where word 2 is head
        """
        edges = self.parse(text, preprocess=preprocess)

        tree = {}
        for edge in edges:
            if edge.head_index not in tree:
                tree[edge.head_index] = []
            tree[edge.head_index].append(edge)

        return tree

    def get_root_verb(self, edges: List[DependencyEdge]) -> Optional[str]:
        """
        Extract root verb from dependency edges.

        Args:
            edges: List of dependency edges

        Returns:
            Root verb word or None
        """
        for edge in edges:
            if edge.relation.lower() == 'root' and edge.head_index == 0:
                return edge.dependent_word
        return None


class SVOExtractor:
    """
    Subject-Verb-Object triple extractor.

    Extracts SVO triples from dependency parse trees for Chinese text.

    Attributes:
        parser: DependencyParser instance
        logger: Logger instance

    Examples:
        >>> extractor = SVOExtractor()
        >>> triples = extractor.extract("張三喜歡吃蘋果")
        >>> for triple in triples:
        ...     print(triple)
        (張三, 喜歡, 蘋果)
    """

    def __init__(self,
                 parser: Optional[DependencyParser] = None,
                 tokenizer_engine: str = 'jieba'):
        """
        Initialize SVO extractor.

        Args:
            parser: DependencyParser instance (creates new if None)
            tokenizer_engine: 'jieba' | 'ckip' | 'auto'
        """
        self.logger = logging.getLogger(__name__)

        if parser is None:
            self.parser = DependencyParser(tokenizer_engine=tokenizer_engine)
        else:
            self.parser = parser

        self.logger.info("SVOExtractor initialized")

    # ========================================================================
    # Core Extraction Methods
    # ========================================================================

    def extract(self,
                text: str,
                preprocess: bool = True,
                include_partial: bool = True) -> List[SVOTriple]:
        """
        Extract SVO triples from text.

        Args:
            text: Input text
            preprocess: Apply tokenization
            include_partial: Include SV triples (no object)

        Returns:
            List of SVOTriple objects

        Complexity:
            Time: O(n³) parsing + O(n) extraction = O(n³)
            Space: O(n)

        Examples:
            >>> extractor = SVOExtractor()
            >>> triples = extractor.extract("我喜歡吃蘋果")
            >>> print(triples[0])
            (我, 喜歡, 蘋果)
        """
        if not text or not text.strip():
            return []

        # Parse dependencies
        edges = self.parser.parse(text, preprocess=preprocess)
        if not edges:
            return []

        # Build dependency tree
        tree = {}
        for edge in edges:
            if edge.head_index not in tree:
                tree[edge.head_index] = []
            tree[edge.head_index].append(edge)

        # Extract triples
        triples = []

        # Find root verb
        root_verb = None
        root_index = None
        for edge in edges:
            if edge.relation.lower() == 'root' and edge.head_index == 0:
                root_verb = edge.dependent_word
                root_index = edge.dependent_index
                break

        if not root_verb:
            return triples

        # Find subject (nsubj, nsubjpass)
        subject = None
        subject_index = None
        if root_index in tree:
            for edge in tree[root_index]:
                if edge.relation in ['nsubj', 'nsubjpass', 'top']:
                    subject = edge.dependent_word
                    subject_index = edge.dependent_index
                    break

        # Find object (dobj, attr, ccomp)
        obj = None
        obj_index = None
        if root_index in tree:
            for edge in tree[root_index]:
                if edge.relation in ['dobj', 'attr', 'ccomp', 'iobj']:
                    obj = edge.dependent_word
                    obj_index = edge.dependent_index
                    break

        # Create triple
        if subject and root_verb:
            triple = SVOTriple(
                subject=subject,
                verb=root_verb,
                object=obj,
                subject_index=subject_index,
                verb_index=root_index,
                object_index=obj_index,
                confidence=1.0 if obj else 0.7  # Lower confidence for SV
            )

            if obj or include_partial:
                triples.append(triple)

        self.logger.debug(f"Extracted {len(triples)} SVO triples")
        return triples

    def extract_batch(self,
                     texts: List[str],
                     preprocess: bool = True,
                     include_partial: bool = True) -> List[List[SVOTriple]]:
        """
        Extract SVO triples from multiple texts.

        Args:
            texts: List of text strings
            preprocess: Apply tokenization
            include_partial: Include SV triples

        Returns:
            List of SVO triple lists
        """
        results = []
        for text in texts:
            triples = self.extract(text, preprocess=preprocess, include_partial=include_partial)
            results.append(triples)

        self.logger.info(f"Batch extracted from {len(texts)} texts")
        return results

    # ========================================================================
    # Analysis Methods
    # ========================================================================

    def extract_all_relations(self,
                             text: str,
                             preprocess: bool = True) -> List[Tuple[str, str, str]]:
        """
        Extract all dependency relations as (head, relation, dependent) tuples.

        Args:
            text: Input text
            preprocess: Apply tokenization

        Returns:
            List of (head, relation, dependent) tuples

        Examples:
            >>> extractor = SVOExtractor()
            >>> relations = extractor.extract_all_relations("我喜歡你")
            >>> for rel in relations:
            ...     print(f"{rel[0]} --{rel[1]}--> {rel[2]}")
        """
        edges = self.parser.parse(text, preprocess=preprocess)

        relations = []
        for edge in edges:
            relations.append((
                edge.head_word,
                edge.relation,
                edge.dependent_word
            ))

        return relations


class SyntaxAnalyzer:
    """
    High-level syntax analysis interface.

    Combines dependency parsing and SVO extraction with additional
    analysis utilities.

    Examples:
        >>> analyzer = SyntaxAnalyzer()
        >>> result = analyzer.analyze("張三在台北大學學習自然語言處理")
        >>> print(result['svo_triples'])
        >>> print(result['dependency_edges'])
    """

    def __init__(self, tokenizer_engine: str = 'jieba', device: int = -1):
        """
        Initialize syntax analyzer.

        Args:
            tokenizer_engine: 'jieba' | 'ckip' | 'auto'
            device: Device for parsing (-1=CPU, 0+=GPU)
        """
        self.logger = logging.getLogger(__name__)

        self.parser = DependencyParser(
            tokenizer_engine=tokenizer_engine,
            device=device
        )
        self.svo_extractor = SVOExtractor(parser=self.parser)

        self.logger.info("SyntaxAnalyzer initialized")

    def analyze(self,
               text: str,
               preprocess: bool = True,
               extract_svo: bool = True) -> Dict[str, Any]:
        """
        Perform comprehensive syntax analysis.

        Args:
            text: Input text
            preprocess: Apply tokenization
            extract_svo: Extract SVO triples

        Returns:
            Dictionary with analysis results:
                - 'text': Original text
                - 'tokens': Token list
                - 'dependency_edges': List of DependencyEdge
                - 'svo_triples': List of SVOTriple (if extract_svo=True)
                - 'root_verb': Root verb word

        Examples:
            >>> analyzer = SyntaxAnalyzer()
            >>> result = analyzer.analyze("我喜歡學習")
            >>> print(result['svo_triples'])
            >>> print(result['root_verb'])
        """
        # Parse dependencies
        edges = self.parser.parse(text, preprocess=preprocess)

        # Extract SVO
        triples = []
        if extract_svo:
            triples = self.svo_extractor.extract(
                text,
                preprocess=preprocess,
                include_partial=True
            )

        # Find root verb
        root_verb = self.parser.get_root_verb(edges)

        # Get tokens
        if preprocess and self.parser.tokenizer:
            tokens = self.parser.tokenizer.tokenize(text)
        else:
            tokens = text.split() if ' ' in text else list(text)

        return {
            'text': text,
            'tokens': tokens,
            'dependency_edges': edges,
            'svo_triples': triples,
            'root_verb': root_verb,
            'num_edges': len(edges),
            'num_triples': len(triples)
        }

    def analyze_batch(self,
                     texts: List[str],
                     preprocess: bool = True,
                     extract_svo: bool = True) -> List[Dict[str, Any]]:
        """
        Analyze multiple texts in batch.

        Args:
            texts: List of text strings
            preprocess: Apply tokenization
            extract_svo: Extract SVO triples

        Returns:
            List of analysis result dictionaries
        """
        results = []
        for text in texts:
            result = self.analyze(text, preprocess=preprocess, extract_svo=extract_svo)
            results.append(result)

        self.logger.info(f"Batch analyzed {len(texts)} texts")
        return results


def demo():
    """Demonstration of syntax parsing and SVO extraction."""
    print("=" * 70)
    print("Syntax Analysis Demo (Traditional Chinese)")
    print("=" * 70)

    # Sample sentences
    sentences = [
        "張三喜歡吃蘋果",
        "我在台北大學學習自然語言處理",
        "深度學習模型需要大量訓練資料",
        "她寫了一本關於機器學習的書"
    ]

    # Initialize analyzer
    print("\n[1] Initializing SyntaxAnalyzer...")
    print("-" * 70)
    analyzer = SyntaxAnalyzer(tokenizer_engine='jieba')
    print("Analyzer initialized")

    # Analyze sentences
    print("\n[2] Dependency Parsing & SVO Extraction")
    print("-" * 70)

    for i, sentence in enumerate(sentences, 1):
        print(f"\n句子 {i}: {sentence}")
        print("-" * 70)

        result = analyzer.analyze(sentence)

        # Show dependency edges
        print(f"依存邊 ({len(result['dependency_edges'])} 條):")
        for edge in result['dependency_edges'][:5]:  # Show first 5
            print(f"  {edge.dependent_word} --{edge.relation}--> {edge.head_word}")

        # Show SVO triples
        print(f"\nSVO 三元組 ({len(result['svo_triples'])} 組):")
        for triple in result['svo_triples']:
            print(f"  {triple}")

        # Show root verb
        print(f"\n根動詞: {result['root_verb']}")

    # Batch processing
    print("\n[3] Batch Processing")
    print("-" * 70)
    batch_results = analyzer.analyze_batch(sentences)

    for i, result in enumerate(batch_results, 1):
        print(f"句子 {i}: {result['num_triples']} SVO 三元組")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    demo()
