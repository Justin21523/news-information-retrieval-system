"""
Static Summarization Algorithms

This module implements extractive summarization techniques that generate
summaries from the original document text without generating new sentences.

Key Features:
    - Lead-k summarization (first k sentences)
    - Key sentence extraction (TF-IDF based)
    - Position-based scoring
    - Length-based filtering
    - Multi-document summarization support

Reference: "Introduction to Information Retrieval" (Manning et al.)
           Chapter 21: Information Retrieval and Information Extraction

Author: Information Retrieval System
License: Educational Use
"""

import logging
import re
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict, Counter
import math

import sys
from pathlib import Path

_parent_dir = Path(__file__).parent.parent.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))


@dataclass
class Sentence:
    """
    Sentence representation for summarization.

    Attributes:
        text: Original sentence text
        position: Position in document (0-indexed)
        doc_id: Document identifier (for multi-doc)
        tokens: List of tokens (words)
        score: Importance score
    """
    text: str
    position: int
    doc_id: Optional[int] = None
    tokens: Optional[List[str]] = None
    score: float = 0.0

    def __post_init__(self):
        if self.tokens is None:
            self.tokens = self._tokenize(self.text)

    def _tokenize(self, text: str) -> List[str]:
        """Simple word tokenization with Chinese support."""
        # Remove punctuation and lowercase for Latin text
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text.lower())

        # Split by whitespace first (for English words)
        tokens = []
        for segment in text.split():
            # If segment contains Chinese characters, split by character
            if any('\u4e00' <= c <= '\u9fff' for c in segment):
                # Split Chinese text into characters (simple approach)
                tokens.extend(c for c in segment if c.strip())
            else:
                if segment.strip():
                    tokens.append(segment)

        return tokens

    @property
    def length(self) -> int:
        return len(self.tokens)


@dataclass
class Summary:
    """
    Summary result container.

    Attributes:
        sentences: Selected sentences
        method: Summarization method used
        compression_ratio: Summary length / original length
        original_length: Original document length (sentences)
    """
    sentences: List[Sentence]
    method: str
    compression_ratio: float
    original_length: int

    @property
    def text(self) -> str:
        """Get summary as plain text."""
        return ' '.join(s.text for s in self.sentences)

    @property
    def length(self) -> int:
        """Number of sentences in summary."""
        return len(self.sentences)


class StaticSummarizer:
    """
    Static Extractive Summarization Engine.

    Implements various extractive summarization techniques:
    - Lead-k: Extract first k sentences
    - TF-IDF based: Extract high-scoring sentences
    - Position-weighted: Combine content and position

    Complexity:
        - Lead-k: O(n) where n = sentences
        - TF-IDF: O(n × m) where m = avg tokens per sentence
        - Key sentence extraction: O(n × m + n log n) for scoring and sorting
    """

    def __init__(self, min_sentence_length: int = 5, max_sentence_length: int = 100):
        """
        Initialize static summarizer.

        Args:
            min_sentence_length: Minimum tokens per sentence
            max_sentence_length: Maximum tokens per sentence
        """
        self.logger = logging.getLogger(__name__)
        self.min_sentence_length = min_sentence_length
        self.max_sentence_length = max_sentence_length
        self.logger.info("StaticSummarizer initialized")

    # ========================================================================
    # Document Preprocessing
    # ========================================================================

    def segment_sentences(self, text: str, doc_id: Optional[int] = None) -> List[Sentence]:
        """
        Segment document into sentences.

        Args:
            text: Document text
            doc_id: Document identifier

        Returns:
            List of Sentence objects

        Complexity:
            Time: O(n) where n = document length
            Space: O(s) where s = number of sentences

        Examples:
            >>> summarizer = StaticSummarizer()
            >>> text = "First sentence. Second sentence. Third sentence."
            >>> sentences = summarizer.segment_sentences(text)
            >>> len(sentences)
            3
        """
        # Split by sentence terminators (both English and Chinese)
        # Support English: . ! ?
        # Support Chinese: 。！？
        sentence_pattern = r'[.!?。！？]+'
        raw_sentences = re.split(sentence_pattern, text.strip())

        sentences = []
        position_counter = 0
        for raw_text in raw_sentences:
            raw_text = raw_text.strip()
            if not raw_text:
                continue

            sentence = Sentence(
                text=raw_text,
                position=position_counter,
                doc_id=doc_id
            )

            # Filter by length
            if (self.min_sentence_length <= sentence.length <= self.max_sentence_length):
                sentences.append(sentence)

            position_counter += 1

        self.logger.debug(f"Segmented {len(sentences)} valid sentences from document")
        return sentences

    def compute_term_frequencies(self, sentences: List[Sentence]) -> Dict[str, int]:
        """
        Compute term frequencies across all sentences.

        Args:
            sentences: List of sentences

        Returns:
            Dictionary {term: frequency}

        Complexity:
            Time: O(n × m) where n=sentences, m=avg tokens
            Space: O(v) where v=vocabulary size
        """
        term_freq = Counter()
        for sentence in sentences:
            term_freq.update(sentence.tokens)
        return dict(term_freq)

    def compute_idf(self, sentences: List[Sentence]) -> Dict[str, float]:
        """
        Compute IDF (Inverse Document Frequency) for terms.

        Each sentence is treated as a "document" for IDF calculation.

        Args:
            sentences: List of sentences

        Returns:
            Dictionary {term: idf_score}

        Complexity:
            Time: O(n × m) where n=sentences, m=avg tokens
            Space: O(v) where v=vocabulary size
        """
        num_sentences = len(sentences)
        doc_freq = defaultdict(int)

        # Count document frequency
        for sentence in sentences:
            unique_tokens = set(sentence.tokens)
            for token in unique_tokens:
                doc_freq[token] += 1

        # Calculate IDF
        idf = {}
        for term, df in doc_freq.items():
            idf[term] = math.log(num_sentences / df)

        return idf

    # ========================================================================
    # Lead-k Summarization
    # ========================================================================

    def lead_k_summarization(self, text: str, k: int = 3,
                            doc_id: Optional[int] = None) -> Summary:
        """
        Lead-k summarization: Extract first k sentences.

        Simple but effective baseline, especially for news articles
        where important information appears early.

        Args:
            text: Document text
            k: Number of sentences to extract
            doc_id: Document identifier

        Returns:
            Summary object

        Complexity:
            Time: O(n) where n = document length
            Space: O(k)

        Examples:
            >>> summarizer = StaticSummarizer()
            >>> text = "First. Second. Third. Fourth. Fifth."
            >>> summary = summarizer.lead_k_summarization(text, k=2)
            >>> summary.length
            2
            >>> "First" in summary.text
            True
        """
        self.logger.info(f"Lead-k summarization: k={k}")

        sentences = self.segment_sentences(text, doc_id)

        if not sentences:
            return Summary(
                sentences=[],
                method='lead-k',
                compression_ratio=0.0,
                original_length=0
            )

        # Take first k sentences
        selected = sentences[:min(k, len(sentences))]

        compression = len(selected) / len(sentences)

        return Summary(
            sentences=selected,
            method='lead-k',
            compression_ratio=compression,
            original_length=len(sentences)
        )

    # ========================================================================
    # TF-IDF Based Key Sentence Extraction
    # ========================================================================

    def score_sentence_tfidf(self, sentence: Sentence,
                            tf: Dict[str, int],
                            idf: Dict[str, float]) -> float:
        """
        Score sentence using TF-IDF.

        Sentence score = sum of TF-IDF scores for all tokens.

        Args:
            sentence: Sentence to score
            tf: Term frequencies
            idf: IDF scores

        Returns:
            Sentence score (higher = more important)

        Complexity:
            Time: O(m) where m = tokens in sentence
        """
        score = 0.0
        for token in sentence.tokens:
            term_tf = tf.get(token, 0)
            term_idf = idf.get(token, 0.0)
            score += term_tf * term_idf

        # Normalize by sentence length to avoid bias toward long sentences
        if sentence.length > 0:
            score /= sentence.length

        return score

    def key_sentence_extraction(self, text: str, k: int = 3,
                               doc_id: Optional[int] = None,
                               use_position_bias: bool = True) -> Summary:
        """
        Extract key sentences using TF-IDF scoring.

        Selects sentences with highest TF-IDF scores. Optionally applies
        position bias to favor sentences near the beginning.

        Args:
            text: Document text
            k: Number of sentences to extract
            doc_id: Document identifier
            use_position_bias: Apply position-based bonus

        Returns:
            Summary object

        Complexity:
            Time: O(n × m + n log n) where n=sentences, m=avg tokens
            Space: O(n + v) where v=vocabulary size

        Examples:
            >>> summarizer = StaticSummarizer()
            >>> text = "Machine learning is great. Deep learning is powerful. AI is the future."
            >>> summary = summarizer.key_sentence_extraction(text, k=2)
            >>> summary.length
            2
        """
        self.logger.info(f"Key sentence extraction: k={k}, position_bias={use_position_bias}")

        sentences = self.segment_sentences(text, doc_id)

        if not sentences:
            return Summary(
                sentences=[],
                method='key-sentence-tfidf',
                compression_ratio=0.0,
                original_length=0
            )

        # Compute TF and IDF
        tf = self.compute_term_frequencies(sentences)
        idf = self.compute_idf(sentences)

        # Score sentences
        for sentence in sentences:
            base_score = self.score_sentence_tfidf(sentence, tf, idf)

            # Apply position bias (earlier sentences get bonus)
            if use_position_bias:
                position_weight = 1.0 / (1.0 + sentence.position)
                sentence.score = base_score * (1.0 + 0.5 * position_weight)
            else:
                sentence.score = base_score

        # Sort by score (descending) and select top-k
        sentences_sorted = sorted(sentences, key=lambda s: s.score, reverse=True)
        selected = sentences_sorted[:min(k, len(sentences))]

        # Re-sort by position for coherent reading
        selected.sort(key=lambda s: s.position)

        compression = len(selected) / len(sentences)

        self.logger.info(f"Selected {len(selected)} sentences from {len(sentences)}")

        return Summary(
            sentences=selected,
            method='key-sentence-tfidf',
            compression_ratio=compression,
            original_length=len(sentences)
        )

    # ========================================================================
    # Query-Focused Summarization
    # ========================================================================

    def query_focused_summarization(self, text: str, query: str,
                                   k: int = 3,
                                   doc_id: Optional[int] = None) -> Summary:
        """
        Generate query-focused summary.

        Extracts sentences most relevant to the given query.
        Uses cosine similarity between query and sentences.

        Args:
            text: Document text
            query: Query string
            k: Number of sentences to extract
            doc_id: Document identifier

        Returns:
            Summary object

        Complexity:
            Time: O(n × m) where n=sentences, m=avg tokens
            Space: O(n + v)

        Examples:
            >>> summarizer = StaticSummarizer()
            >>> text = "Dogs are loyal. Cats are independent. Birds can fly."
            >>> summary = summarizer.query_focused_summarization(text, "dogs cats", k=2)
            >>> summary.length
            2
        """
        self.logger.info(f"Query-focused summarization: query='{query}', k={k}")

        sentences = self.segment_sentences(text, doc_id)

        if not sentences:
            return Summary(
                sentences=[],
                method='query-focused',
                compression_ratio=0.0,
                original_length=0
            )

        # Tokenize query
        query_tokens = set(re.sub(r'[^\w\s]', ' ', query.lower()).split())

        # Score sentences by query overlap
        for sentence in sentences:
            sentence_tokens = set(sentence.tokens)
            overlap = len(query_tokens & sentence_tokens)

            # Normalize by geometric mean of lengths
            if overlap > 0:
                norm = math.sqrt(len(query_tokens) * len(sentence_tokens))
                sentence.score = overlap / norm if norm > 0 else 0.0
            else:
                sentence.score = 0.0

        # Sort by score and select top-k
        sentences_sorted = sorted(sentences, key=lambda s: s.score, reverse=True)
        selected = sentences_sorted[:min(k, len(sentences))]

        # Re-sort by position
        selected.sort(key=lambda s: s.position)

        compression = len(selected) / len(sentences)

        return Summary(
            sentences=selected,
            method='query-focused',
            compression_ratio=compression,
            original_length=len(sentences)
        )

    # ========================================================================
    # Multi-Document Summarization
    # ========================================================================

    def multi_document_summarization(self, documents: List[str],
                                    k: int = 5,
                                    diversity_threshold: float = 0.5) -> Summary:
        """
        Multi-document summarization.

        Extracts representative sentences from multiple documents
        while avoiding redundancy.

        Args:
            documents: List of document texts
            k: Total number of sentences to extract
            diversity_threshold: Minimum diversity score (0-1)

        Returns:
            Summary object

        Complexity:
            Time: O(d × n × m + s²) where d=docs, n=sentences, s=selected
            Space: O(d × n)

        Examples:
            >>> summarizer = StaticSummarizer()
            >>> docs = ["First document.", "Second document.", "Third document."]
            >>> summary = summarizer.multi_document_summarization(docs, k=2)
            >>> summary.length
            2
        """
        self.logger.info(f"Multi-document summarization: {len(documents)} docs, k={k}")

        # Collect all sentences
        all_sentences = []
        for doc_id, doc_text in enumerate(documents):
            sentences = self.segment_sentences(doc_text, doc_id)
            all_sentences.extend(sentences)

        if not all_sentences:
            return Summary(
                sentences=[],
                method='multi-document',
                compression_ratio=0.0,
                original_length=0
            )

        # Compute TF-IDF scores
        tf = self.compute_term_frequencies(all_sentences)
        idf = self.compute_idf(all_sentences)

        for sentence in all_sentences:
            sentence.score = self.score_sentence_tfidf(sentence, tf, idf)

        # Greedy selection with diversity
        selected = []
        candidates = sorted(all_sentences, key=lambda s: s.score, reverse=True)

        for candidate in candidates:
            if len(selected) >= k:
                break

            # Check diversity
            is_diverse = True
            for existing in selected:
                similarity = self._sentence_similarity(candidate, existing)
                if similarity > diversity_threshold:
                    is_diverse = False
                    break

            if is_diverse:
                selected.append(candidate)

        # Sort by document order and position
        selected.sort(key=lambda s: (s.doc_id if s.doc_id is not None else 0, s.position))

        compression = len(selected) / len(all_sentences)

        self.logger.info(f"Selected {len(selected)} sentences from {len(all_sentences)} total")

        return Summary(
            sentences=selected,
            method='multi-document',
            compression_ratio=compression,
            original_length=len(all_sentences)
        )

    def _sentence_similarity(self, s1: Sentence, s2: Sentence) -> float:
        """
        Calculate Jaccard similarity between two sentences.

        Args:
            s1: First sentence
            s2: Second sentence

        Returns:
            Jaccard similarity (0-1)
        """
        tokens1 = set(s1.tokens)
        tokens2 = set(s2.tokens)

        if not tokens1 or not tokens2:
            return 0.0

        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)

        return intersection / union if union > 0 else 0.0


def demo():
    """Demonstration of static summarization."""
    print("=" * 60)
    print("Static Summarization Demo")
    print("=" * 60)

    # Sample document
    text = """
    Machine learning is a subset of artificial intelligence.
    It enables computers to learn from data without explicit programming.
    Deep learning is a type of machine learning based on neural networks.
    Neural networks consist of layers of interconnected nodes.
    Applications include image recognition and natural language processing.
    The field has grown rapidly in recent years.
    Large datasets and powerful hardware have accelerated progress.
    """

    summarizer = StaticSummarizer()

    # Example 1: Lead-k
    print("\n1. Lead-k Summarization (k=3):")
    summary = summarizer.lead_k_summarization(text, k=3)
    print(f"   Method: {summary.method}")
    print(f"   Sentences: {summary.length}/{summary.original_length}")
    print(f"   Compression: {summary.compression_ratio:.2%}")
    print(f"   Summary: {summary.text[:150]}...")

    # Example 2: Key sentence extraction
    print("\n2. Key Sentence Extraction (k=3, with position bias):")
    summary = summarizer.key_sentence_extraction(text, k=3, use_position_bias=True)
    print(f"   Method: {summary.method}")
    print(f"   Sentences: {summary.length}/{summary.original_length}")
    for i, sent in enumerate(summary.sentences):
        print(f"   [{i+1}] (pos={sent.position}, score={sent.score:.3f}): {sent.text[:60]}...")

    # Example 3: Query-focused
    print("\n3. Query-Focused Summarization (query='neural networks'):")
    summary = summarizer.query_focused_summarization(text, "neural networks", k=2)
    print(f"   Method: {summary.method}")
    print(f"   Sentences: {summary.length}/{summary.original_length}")
    for i, sent in enumerate(summary.sentences):
        print(f"   [{i+1}] (score={sent.score:.3f}): {sent.text[:60]}...")

    # Example 4: Multi-document
    print("\n4. Multi-Document Summarization:")
    docs = [
        "Python is a popular programming language. It is easy to learn.",
        "Java is widely used in enterprise applications. It is platform independent.",
        "JavaScript is essential for web development. It runs in browsers."
    ]
    summary = summarizer.multi_document_summarization(docs, k=3)
    print(f"   Documents: {len(docs)}")
    print(f"   Summary sentences: {summary.length}")
    for i, sent in enumerate(summary.sentences):
        print(f"   [{i+1}] Doc {sent.doc_id}: {sent.text}")

    print("\n" + "=" * 60)


# ============================================================================
# Convenience Functions (Backward Compatibility)
# ============================================================================

# Create a global instance for convenience functions
_summarizer_instance = StaticSummarizer()


def lead_k_summary(text: str, k: int = 3, return_indices: bool = False):
    """
    Convenience function for lead-k summarization.

    Args:
        text: Input text
        k: Number of leading sentences
        return_indices: Whether to return sentence indices

    Returns:
        Summary object with first k sentences
    """
    return _summarizer_instance.lead_k_summarization(text, k, return_indices)


def key_sentence_summary(text: str, k: int = 3, return_indices: bool = False):
    """
    Convenience function for key sentence extraction.

    Args:
        text: Input text
        k: Number of key sentences
        return_indices: Whether to return sentence indices

    Returns:
        Summary object with top k important sentences
    """
    return _summarizer_instance.key_sentence_extraction(text, k, return_indices)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    demo()
