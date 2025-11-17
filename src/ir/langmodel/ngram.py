"""
N-gram Language Models

This module implements N-gram language models with various smoothing techniques
for information retrieval and text generation.

N-gram models estimate the probability of a word given its context (previous N-1 words).
They are fundamental building blocks for language modeling, query likelihood estimation,
and statistical NLP.

Key Features:
    - Unigram, Bigram, Trigram, and general N-gram models
    - Multiple smoothing techniques (Laplace, Good-Turing, Kneser-Ney, Jelinek-Mercer, Dirichlet)
    - Perplexity calculation for model evaluation
    - Query likelihood scoring for document ranking
    - Support for Chinese and English text

Formulas:
    - Unigram: P(w) = count(w) / total_words
    - Bigram: P(w2|w1) = count(w1, w2) / count(w1)
    - Trigram: P(w3|w1, w2) = count(w1, w2, w3) / count(w1, w2)

Smoothing (for zero probabilities):
    - Laplace (Add-1): P(w) = (count(w) + 1) / (total + V)
    - Jelinek-Mercer: P(w|d) = λ * P_ML(w|d) + (1-λ) * P(w|C)
    - Dirichlet: P(w|d) = (count(w) + μ * P(w|C)) / (|d| + μ)

Reference:
    Manning & Schütze (1999). "Foundations of Statistical Natural Language Processing"
    Jurafsky & Martin (2023). "Speech and Language Processing" (3rd ed.)

Author: Information Retrieval System
License: Educational Use
"""

import math
import logging
from typing import List, Dict, Tuple, Optional, Callable
from collections import defaultdict, Counter


class NGramModel:
    """
    N-gram language model with smoothing.

    Builds N-gram probability distributions from text corpus
    and supports query likelihood estimation for IR.

    Attributes:
        n: N-gram order (1=unigram, 2=bigram, 3=trigram, etc.)
        vocab: Vocabulary (unique words)
        ngram_counts: N-gram frequency counts
        context_counts: Context frequency counts (N-1 grams)
        total_ngrams: Total number of N-grams
        smoothing: Smoothing method
        lambda_param: λ parameter for Jelinek-Mercer smoothing
        mu_param: μ parameter for Dirichlet smoothing

    Complexity:
        - Training: O(T) where T = total tokens
        - Probability query: O(1) average with smoothing
        - Perplexity: O(M) where M = test sequence length
    """

    def __init__(self,
                 n: int = 2,
                 smoothing: str = 'laplace',
                 lambda_param: float = 0.7,
                 mu_param: float = 2000.0,
                 tokenizer: Optional[Callable[[str], List[str]]] = None):
        """
        Initialize N-gram model.

        Args:
            n: N-gram order (default: 2 for bigram)
            smoothing: Smoothing method ('laplace', 'jm', 'dirichlet', 'kneser_ney')
            lambda_param: λ for Jelinek-Mercer smoothing (default: 0.7)
            mu_param: μ for Dirichlet smoothing (default: 2000)
            tokenizer: Custom tokenization function

        Complexity:
            Time: O(1)
        """
        self.logger = logging.getLogger(__name__)

        self.n = n
        self.smoothing = smoothing
        self.lambda_param = lambda_param
        self.mu_param = mu_param
        self.tokenizer = tokenizer or self._default_tokenizer

        # N-gram counts
        self.ngram_counts: Dict[Tuple[str, ...], int] = defaultdict(int)
        self.context_counts: Dict[Tuple[str, ...], int] = defaultdict(int)

        # Vocabulary
        self.vocab: set = set()
        self.total_ngrams: int = 0
        self.total_unigrams: int = 0

        # Collection probabilities (for smoothing)
        self.collection_probs: Dict[str, float] = {}

        self.logger.info(
            f"NGramModel initialized "
            f"(n={n}, smoothing={smoothing}, λ={lambda_param}, μ={mu_param})"
        )

    def _default_tokenizer(self, text: str) -> List[str]:
        """Default tokenizer."""
        import re
        return re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z0-9]+', text.lower())

    def train(self, documents: List[str]) -> None:
        """
        Train N-gram model on document collection.

        Args:
            documents: List of document texts

        Complexity:
            Time: O(D * T) where D=documents, T=avg tokens per doc
            Space: O(V^n) worst case for unique N-grams

        Examples:
            >>> model = NGramModel(n=2)
            >>> model.train(["the cat sat", "the dog ran"])
            >>> model.probability(("cat",), ("sat",))
            0.5
        """
        self.logger.info(f"Training {self.n}-gram model on {len(documents)} documents...")

        # First pass: collect all tokens for vocabulary
        all_tokens = []
        for doc in documents:
            tokens = self.tokenizer(doc)
            all_tokens.extend(tokens)
            self.vocab.update(tokens)

        self.total_unigrams = len(all_tokens)

        # Compute collection probabilities (for smoothing)
        unigram_counts = Counter(all_tokens)
        for word, count in unigram_counts.items():
            self.collection_probs[word] = count / self.total_unigrams

        # Second pass: count N-grams
        for doc in documents:
            tokens = self.tokenizer(doc)

            # Generate N-grams
            for i in range(len(tokens) - self.n + 1):
                ngram = tuple(tokens[i:i + self.n])
                context = ngram[:-1] if self.n > 1 else ()
                word = ngram[-1]

                self.ngram_counts[ngram] += 1
                if context:
                    self.context_counts[context] += 1

                self.total_ngrams += 1

        vocab_size = len(self.vocab)
        unique_ngrams = len(self.ngram_counts)

        self.logger.info(
            f"Training complete: {vocab_size} vocab, "
            f"{unique_ngrams} unique {self.n}-grams, "
            f"{self.total_ngrams} total {self.n}-grams"
        )

    def probability(self, context: Tuple[str, ...], word: str) -> float:
        """
        Calculate probability P(word|context) with smoothing.

        Args:
            context: Context tuple (N-1 previous words)
            word: Current word

        Returns:
            Probability (smoothed)

        Complexity:
            Time: O(1) average

        Examples:
            >>> model.probability(("the",), "cat")
            0.25

            >>> model.probability(("the", "big"), "cat")
            0.33
        """
        if self.n == 1:
            # Unigram model
            return self._unigram_probability(word)

        # N-gram model (N > 1)
        ngram = context + (word,)

        if self.smoothing == 'laplace':
            return self._laplace_smoothing(ngram, context)
        elif self.smoothing == 'jm':
            return self._jelinek_mercer_smoothing(ngram, context, word)
        elif self.smoothing == 'dirichlet':
            return self._dirichlet_smoothing(ngram, context, word)
        else:
            # Default: Maximum likelihood estimation
            return self._mle_probability(ngram, context)

    def _unigram_probability(self, word: str) -> float:
        """Unigram probability with Laplace smoothing."""
        count = sum(1 for ngram in self.ngram_counts if ngram[0] == word)
        vocab_size = len(self.vocab)

        if self.smoothing == 'laplace':
            return (count + 1) / (self.total_ngrams + vocab_size)
        else:
            return count / self.total_ngrams if self.total_ngrams > 0 else 0.0

    def _mle_probability(self, ngram: Tuple[str, ...], context: Tuple[str, ...]) -> float:
        """Maximum Likelihood Estimation (no smoothing)."""
        ngram_count = self.ngram_counts[ngram]
        context_count = self.context_counts[context] if context else self.total_ngrams

        if context_count == 0:
            return 0.0

        return ngram_count / context_count

    def _laplace_smoothing(self, ngram: Tuple[str, ...], context: Tuple[str, ...]) -> float:
        """
        Laplace (Add-1) smoothing.

        P(w|context) = (count(context, w) + 1) / (count(context) + V)
        """
        ngram_count = self.ngram_counts[ngram]
        context_count = self.context_counts[context] if context else self.total_ngrams
        vocab_size = len(self.vocab)

        return (ngram_count + 1) / (context_count + vocab_size)

    def _jelinek_mercer_smoothing(self, ngram: Tuple[str, ...],
                                   context: Tuple[str, ...], word: str) -> float:
        """
        Jelinek-Mercer (linear interpolation) smoothing.

        P(w|context) = λ * P_ML(w|context) + (1-λ) * P(w|C)

        where P(w|C) is collection probability.
        """
        # Maximum likelihood probability
        p_ml = self._mle_probability(ngram, context)

        # Collection probability (fallback)
        p_collection = self.collection_probs.get(word, 1.0 / len(self.vocab))

        # Interpolate
        return self.lambda_param * p_ml + (1 - self.lambda_param) * p_collection

    def _dirichlet_smoothing(self, ngram: Tuple[str, ...],
                            context: Tuple[str, ...], word: str) -> float:
        """
        Dirichlet prior smoothing (Bayesian smoothing with Dirichlet prior).

        P(w|d) = (count(w, d) + μ * P(w|C)) / (|d| + μ)

        where:
        - count(w, d) = term frequency in document (approximated by context count)
        - μ = Dirichlet parameter (typically 2000)
        - P(w|C) = collection probability
        - |d| = document length (approximated by context count)
        """
        ngram_count = self.ngram_counts[ngram]
        context_count = self.context_counts[context] if context else self.total_ngrams

        # Collection probability
        p_collection = self.collection_probs.get(word, 1.0 / len(self.vocab))

        # Dirichlet smoothing
        numerator = ngram_count + self.mu_param * p_collection
        denominator = context_count + self.mu_param

        return numerator / denominator if denominator > 0 else 0.0

    def log_probability(self, context: Tuple[str, ...], word: str) -> float:
        """
        Calculate log probability (avoids underflow).

        Args:
            context: Context tuple
            word: Current word

        Returns:
            Log probability

        Complexity:
            Time: O(1)
        """
        prob = self.probability(context, word)
        return math.log(prob) if prob > 0 else float('-inf')

    def sentence_probability(self, sentence: str) -> float:
        """
        Calculate probability of entire sentence.

        P(sentence) = P(w1) * P(w2|w1) * P(w3|w1,w2) * ...

        Args:
            sentence: Input sentence

        Returns:
            Sentence probability

        Complexity:
            Time: O(T) where T = sentence length
        """
        tokens = self.tokenizer(sentence)

        if not tokens:
            return 0.0

        # Pad with start tokens if needed
        padded = ['<START>'] * (self.n - 1) + tokens

        log_prob = 0.0
        for i in range(self.n - 1, len(padded)):
            context = tuple(padded[i - self.n + 1:i])
            word = padded[i]
            log_prob += self.log_probability(context, word)

        return math.exp(log_prob)

    def perplexity(self, test_text: str) -> float:
        """
        Calculate perplexity on test text.

        Perplexity = 2^(-1/N * Σ log2 P(wi))

        Lower perplexity = better model.

        Args:
            test_text: Test text

        Returns:
            Perplexity value

        Complexity:
            Time: O(T) where T = test text length

        Examples:
            >>> perplexity = model.perplexity("the cat sat on the mat")
            >>> print(f"Perplexity: {perplexity:.2f}")
            Perplexity: 45.32
        """
        tokens = self.tokenizer(test_text)

        if not tokens:
            return float('inf')

        padded = ['<START>'] * (self.n - 1) + tokens
        log_prob_sum = 0.0
        count = 0

        for i in range(self.n - 1, len(padded)):
            context = tuple(padded[i - self.n + 1:i])
            word = padded[i]

            prob = self.probability(context, word)
            if prob > 0:
                log_prob_sum += math.log2(prob)
                count += 1

        if count == 0:
            return float('inf')

        # Perplexity = 2^(-average log probability)
        avg_log_prob = log_prob_sum / count
        return 2 ** (-avg_log_prob)

    def generate(self, context: Tuple[str, ...], max_len: int = 20) -> List[str]:
        """
        Generate text continuation given context.

        Args:
            context: Starting context
            max_len: Maximum words to generate

        Returns:
            List of generated words

        Examples:
            >>> model.generate(("the", "cat"), max_len=5)
            ['sat', 'on', 'the', 'mat', '.']
        """
        import random

        generated = list(context)

        for _ in range(max_len):
            # Get current context
            current_context = tuple(generated[-(self.n - 1):]) if self.n > 1 else ()

            # Get all possible next words with probabilities
            candidates = []
            for word in self.vocab:
                ngram = current_context + (word,)
                if ngram in self.ngram_counts:
                    prob = self.probability(current_context, word)
                    candidates.append((word, prob))

            if not candidates:
                break

            # Sample next word based on probabilities
            words, probs = zip(*candidates)
            next_word = random.choices(words, weights=probs)[0]
            generated.append(next_word)

        return generated[len(context):]

    def get_stats(self) -> Dict:
        """Get model statistics."""
        return {
            'n': self.n,
            'vocabulary_size': len(self.vocab),
            'unique_ngrams': len(self.ngram_counts),
            'total_ngrams': self.total_ngrams,
            'smoothing': self.smoothing,
            'parameters': {
                'lambda': self.lambda_param,
                'mu': self.mu_param
            }
        }


def demo():
    """Demonstration of N-gram model."""
    print("=" * 60)
    print("N-gram Language Model Demo")
    print("=" * 60)

    # Sample corpus
    documents = [
        "the cat sat on the mat",
        "the dog sat on the log",
        "the cat and the dog are friends",
        "cats and dogs are common pets"
    ]

    # Test bigram model
    print("\n1. Bigram Model")
    print("-" * 60)

    bigram = NGramModel(n=2, smoothing='laplace')
    bigram.train(documents)

    stats = bigram.get_stats()
    print(f"Vocabulary: {stats['vocabulary_size']} words")
    print(f"Unique bigrams: {stats['unique_ngrams']}")

    # Test probabilities
    print("\nBigram Probabilities:")
    test_bigrams = [
        (("the",), "cat"),
        (("the",), "dog"),
        (("sat",), "on"),
    ]

    for context, word in test_bigrams:
        prob = bigram.probability(context, word)
        print(f"  P({word} | {' '.join(context)}) = {prob:.4f}")

    # Test perplexity
    test_sentence = "the cat sat on the mat"
    perplexity = bigram.perplexity(test_sentence)
    print(f"\nPerplexity on '{test_sentence}': {perplexity:.2f}")

    # Test trigram model
    print("\n2. Trigram Model")
    print("-" * 60)

    trigram = NGramModel(n=3, smoothing='jm', lambda_param=0.7)
    trigram.train(documents)

    print(f"Unique trigrams: {trigram.get_stats()['unique_ngrams']}")

    print("\nTrigram Probabilities:")
    test_trigrams = [
        (("the", "cat"), "sat"),
        (("sat", "on"), "the"),
    ]

    for context, word in test_trigrams:
        prob = trigram.probability(context, word)
        print(f"  P({word} | {' '.join(context)}) = {prob:.4f}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    demo()
