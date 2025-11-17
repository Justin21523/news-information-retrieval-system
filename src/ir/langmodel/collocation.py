"""
Collocation Extraction

This module implements statistical measures for identifying collocations
(word pairs that co-occur more often than chance).

Collocations are word combinations that appear together more frequently than
would be expected by chance. They are important for phrase detection, terminology
extraction, and improving retrieval effectiveness.

Statistical Measures:
    - PMI (Pointwise Mutual Information): Measures association strength
    - LLR (Log-Likelihood Ratio): Hypothesis testing for significance
    - Chi-Square (χ²): Tests independence of word occurrences
    - T-Test: Tests if co-occurrence frequency differs from expectation
    - Dice Coefficient: Measures similarity of word distributions

Formulas:
    PMI(x, y) = log2(P(x,y) / (P(x) * P(y)))
    LLR = 2 * Σ O * log(O/E)  where O=observed, E=expected
    χ² = Σ (O - E)² / E
    T-score = (P(x,y) - P(x)*P(y)) / sqrt(P(x,y) / N)

Reference:
    Manning & Schütze (1999). "Foundations of Statistical NLP", Chapter 5
    Dunning (1993). "Accurate Methods for the Statistics of Surprise and Coincidence"

Author: Information Retrieval System
License: Educational Use
"""

import math
import logging
from typing import List, Dict, Tuple, Set, Optional, Callable
from collections import defaultdict, Counter
from dataclasses import dataclass


@dataclass
class CollocationScore:
    """
    Collocation score result.

    Attributes:
        bigram: Word pair (w1, w2)
        freq: Frequency of bigram
        pmi: Pointwise Mutual Information score
        llr: Log-Likelihood Ratio score
        chi_square: Chi-square score
        t_score: T-test score
        dice: Dice coefficient
    """
    bigram: Tuple[str, str]
    freq: int
    pmi: float
    llr: float
    chi_square: float
    t_score: float
    dice: float


class CollocationExtractor:
    """
    Extract and score collocations from text corpus.

    Implements multiple statistical measures to identify
    statistically significant word combinations.

    Attributes:
        tokenizer: Tokenization function
        min_freq: Minimum frequency threshold for candidates
        window_size: Window size for co-occurrence (default: 2 for bigrams)

    Complexity:
        - Training: O(T) where T = total tokens
        - Scoring: O(B) where B = unique bigrams
        - Top-K: O(B log K)
    """

    def __init__(self,
                 tokenizer: Optional[Callable[[str], List[str]]] = None,
                 min_freq: int = 5,
                 window_size: int = 2):
        """
        Initialize CollocationExtractor.

        Args:
            tokenizer: Custom tokenization function
            min_freq: Minimum frequency for candidate collocations
            window_size: Co-occurrence window size

        Complexity:
            Time: O(1)
        """
        self.logger = logging.getLogger(__name__)

        self.tokenizer = tokenizer or self._default_tokenizer
        self.min_freq = min_freq
        self.window_size = window_size

        # Frequency counts
        self.bigram_freq: Counter = Counter()  # (w1, w2) -> count
        self.unigram_freq: Counter = Counter()  # w -> count
        self.total_bigrams: int = 0
        self.total_unigrams: int = 0

        # Contingency table components
        # For each bigram (w1, w2):
        #   n11 = freq(w1, w2)
        #   n12 = freq(w1, not w2)
        #   n21 = freq(not w1, w2)
        #   n22 = freq(not w1, not w2)
        self.contingency_tables: Dict[Tuple[str, str], Dict[str, int]] = {}

        self.logger.info(
            f"CollocationExtractor initialized "
            f"(min_freq={min_freq}, window_size={window_size})"
        )

    def _default_tokenizer(self, text: str) -> List[str]:
        """Default tokenizer."""
        import re
        return re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z0-9]+', text.lower())

    def train(self, documents: List[str]) -> None:
        """
        Extract bigrams and compute frequencies from corpus.

        Args:
            documents: List of document texts

        Complexity:
            Time: O(D * T) where D=documents, T=avg tokens
            Space: O(B) where B=unique bigrams
        """
        self.logger.info(f"Training collocation extractor on {len(documents)} documents...")

        # Collect all tokens
        all_tokens = []
        for doc in documents:
            tokens = self.tokenizer(doc)
            all_tokens.extend(tokens)

            # Count unigrams
            for token in tokens:
                self.unigram_freq[token] += 1
                self.total_unigrams += 1

            # Count bigrams
            for i in range(len(tokens) - 1):
                bigram = (tokens[i], tokens[i + 1])
                self.bigram_freq[bigram] += 1
                self.total_bigrams += 1

        # Build contingency tables for statistical tests
        self._build_contingency_tables()

        vocab_size = len(self.unigram_freq)
        bigram_count = len(self.bigram_freq)

        self.logger.info(
            f"Training complete: {vocab_size} unigrams, "
            f"{bigram_count} unique bigrams, "
            f"{self.total_bigrams} total bigrams"
        )

    def _build_contingency_tables(self) -> None:
        """
        Build 2x2 contingency tables for chi-square and LLR tests.

        For bigram (w1, w2):
            n11 = count(w1, w2)
            n12 = count(w1, ¬w2)
            n21 = count(¬w1, w2)
            n22 = count(¬w1, ¬w2)

        Complexity:
            Time: O(B) where B = unique bigrams
        """
        for (w1, w2), n11 in self.bigram_freq.items():
            c1 = self.unigram_freq[w1]  # count(w1)
            c2 = self.unigram_freq[w2]  # count(w2)

            # Contingency table
            n12 = c1 - n11  # w1 appears, w2 doesn't
            n21 = c2 - n11  # w2 appears, w1 doesn't
            n22 = self.total_bigrams - c1 - c2 + n11  # neither appears

            self.contingency_tables[(w1, w2)] = {
                'n11': n11,
                'n12': max(0, n12),
                'n21': max(0, n21),
                'n22': max(0, n22)
            }

    def pmi(self, w1: str, w2: str) -> float:
        """
        Calculate Pointwise Mutual Information.

        PMI(w1, w2) = log2(P(w1, w2) / (P(w1) * P(w2)))

        Higher PMI = stronger association.

        Args:
            w1: First word
            w2: Second word

        Returns:
            PMI score

        Complexity:
            Time: O(1)

        Examples:
            >>> pmi("New", "York")
            8.45  # Strong collocation

            >>> pmi("the", "cat")
            -2.1  # Weak/negative association
        """
        bigram = (w1, w2)

        if bigram not in self.bigram_freq:
            return float('-inf')

        # P(w1, w2)
        p_bigram = self.bigram_freq[bigram] / self.total_bigrams

        # P(w1) * P(w2)
        p_w1 = self.unigram_freq[w1] / self.total_unigrams
        p_w2 = self.unigram_freq[w2] / self.total_unigrams
        p_independent = p_w1 * p_w2

        if p_independent == 0:
            return float('-inf')

        return math.log2(p_bigram / p_independent)

    def llr(self, w1: str, w2: str) -> float:
        """
        Calculate Log-Likelihood Ratio.

        LLR = 2 * Σ O_ij * log(O_ij / E_ij)

        where O = observed frequency, E = expected frequency.

        Higher LLR = more significant collocation.
        Threshold: LLR > 10.83 is significant at p < 0.001

        Args:
            w1: First word
            w2: Second word

        Returns:
            LLR score

        Complexity:
            Time: O(1)
        """
        bigram = (w1, w2)

        if bigram not in self.contingency_tables:
            return 0.0

        table = self.contingency_tables[bigram]
        n11, n12, n21, n22 = table['n11'], table['n12'], table['n21'], table['n22']

        # Row and column totals
        r1 = n11 + n12  # count(w1)
        r2 = n21 + n22  # count(¬w1)
        c1 = n11 + n21  # count(w2)
        c2 = n12 + n22  # count(¬w2)
        n = r1 + r2     # total bigrams

        if n == 0:
            return 0.0

        # Expected frequencies
        e11 = r1 * c1 / n
        e12 = r1 * c2 / n
        e21 = r2 * c1 / n
        e22 = r2 * c2 / n

        # LLR calculation
        llr_score = 0.0
        for o, e in [(n11, e11), (n12, e12), (n21, e21), (n22, e22)]:
            if o > 0 and e > 0:
                llr_score += o * math.log(o / e)

        return 2 * llr_score

    def chi_square(self, w1: str, w2: str) -> float:
        """
        Calculate Chi-Square statistic.

        χ²(w1, w2) = Σ (O_ij - E_ij)² / E_ij

        Tests independence hypothesis.
        Higher χ² = words are NOT independent (collocation).
        Threshold: χ² > 10.83 is significant at p < 0.001

        Args:
            w1: First word
            w2: Second word

        Returns:
            Chi-square score

        Complexity:
            Time: O(1)
        """
        bigram = (w1, w2)

        if bigram not in self.contingency_tables:
            return 0.0

        table = self.contingency_tables[bigram]
        n11, n12, n21, n22 = table['n11'], table['n12'], table['n21'], table['n22']

        # Row and column totals
        r1 = n11 + n12
        r2 = n21 + n22
        c1 = n11 + n21
        c2 = n12 + n22
        n = r1 + r2

        if n == 0:
            return 0.0

        # Expected frequencies
        e11 = r1 * c1 / n
        e12 = r1 * c2 / n
        e21 = r2 * c1 / n
        e22 = r2 * c2 / n

        # Chi-square calculation
        chi2 = 0.0
        for o, e in [(n11, e11), (n12, e12), (n21, e21), (n22, e22)]:
            if e > 0:
                chi2 += (o - e) ** 2 / e

        return chi2

    def t_score(self, w1: str, w2: str) -> float:
        """
        Calculate T-score.

        T = (P(w1, w2) - P(w1)*P(w2)) / sqrt(P(w1, w2) / N)

        Measures how much co-occurrence exceeds independence.

        Args:
            w1: First word
            w2: Second word

        Returns:
            T-score

        Complexity:
            Time: O(1)
        """
        bigram = (w1, w2)

        if bigram not in self.bigram_freq:
            return 0.0

        # Probabilities
        p_bigram = self.bigram_freq[bigram] / self.total_bigrams
        p_w1 = self.unigram_freq[w1] / self.total_unigrams
        p_w2 = self.unigram_freq[w2] / self.total_unigrams

        # T-score formula
        numerator = p_bigram - (p_w1 * p_w2)
        denominator = math.sqrt(p_bigram / self.total_bigrams)

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def dice_coefficient(self, w1: str, w2: str) -> float:
        """
        Calculate Dice coefficient.

        Dice(w1, w2) = 2 * f(w1, w2) / (f(w1) + f(w2))

        Range: [0, 1], higher = stronger association.

        Args:
            w1: First word
            w2: Second word

        Returns:
            Dice coefficient

        Complexity:
            Time: O(1)
        """
        bigram = (w1, w2)

        if bigram not in self.bigram_freq:
            return 0.0

        f_bigram = self.bigram_freq[bigram]
        f_w1 = self.unigram_freq[w1]
        f_w2 = self.unigram_freq[w2]

        denominator = f_w1 + f_w2

        if denominator == 0:
            return 0.0

        return (2 * f_bigram) / denominator

    def extract_collocations(self,
                            measure: str = 'pmi',
                            topk: int = 100) -> List[CollocationScore]:
        """
        Extract top collocations using specified measure.

        Args:
            measure: Scoring measure ('pmi', 'llr', 'chi_square', 't_score', 'dice')
            topk: Number of top collocations to return

        Returns:
            List of CollocationScore objects, sorted by chosen measure

        Complexity:
            Time: O(B log K) where B=bigrams, K=topk
            Space: O(B)

        Examples:
            >>> collocations = extractor.extract_collocations('pmi', topk=20)
            >>> for coll in collocations[:5]:
            ...     print(f"{coll.bigram}: PMI={coll.pmi:.2f}")
            ('New', 'York'): PMI=8.45
            ('United', 'States'): PMI=7.92
            ...
        """
        # Filter by minimum frequency
        candidates = [
            bigram for bigram, freq in self.bigram_freq.items()
            if freq >= self.min_freq
        ]

        # Score all candidates
        scores = []
        for bigram in candidates:
            w1, w2 = bigram

            score = CollocationScore(
                bigram=bigram,
                freq=self.bigram_freq[bigram],
                pmi=self.pmi(w1, w2),
                llr=self.llr(w1, w2),
                chi_square=self.chi_square(w1, w2),
                t_score=self.t_score(w1, w2),
                dice=self.dice_coefficient(w1, w2)
            )
            scores.append(score)

        # Sort by chosen measure
        measure_map = {
            'pmi': lambda s: s.pmi,
            'llr': lambda s: s.llr,
            'chi_square': lambda s: s.chi_square,
            't_score': lambda s: s.t_score,
            'dice': lambda s: s.dice,
            'freq': lambda s: s.freq
        }

        if measure not in measure_map:
            self.logger.warning(f"Unknown measure '{measure}', using PMI")
            measure = 'pmi'

        scores.sort(key=measure_map[measure], reverse=True)

        return scores[:topk]

    def get_stats(self) -> Dict:
        """Get collocation extractor statistics."""
        return {
            'total_unigrams': self.total_unigrams,
            'total_bigrams': self.total_bigrams,
            'unique_unigrams': len(self.unigram_freq),
            'unique_bigrams': len(self.bigram_freq),
            'min_freq_threshold': self.min_freq,
            'candidates': sum(1 for freq in self.bigram_freq.values() if freq >= self.min_freq)
        }


def demo():
    """Demonstration of CollocationExtractor."""
    print("=" * 60)
    print("Collocation Extraction Demo")
    print("=" * 60)

    # Sample corpus
    documents = [
        "New York is a big city in the United States",
        "The United States and New York are famous",
        "Information retrieval systems process queries",
        "Machine learning and artificial intelligence are related",
        "Natural language processing uses machine learning",
        "The New York Times is a newspaper",
        "Information systems and retrieval methods"
    ]

    # Extract collocations
    extractor = CollocationExtractor(min_freq=2)
    extractor.train(documents)

    stats = extractor.get_stats()
    print(f"\nCorpus Statistics:")
    print(f"  Unique unigrams: {stats['unique_unigrams']}")
    print(f"  Unique bigrams: {stats['unique_bigrams']}")
    print(f"  Candidates (freq >= {stats['min_freq_threshold']}): {stats['candidates']}")

    # Extract collocations using different measures
    measures = ['pmi', 'llr', 'chi_square', 't_score']

    for measure in measures:
        print(f"\n{'-' * 60}")
        print(f"Top Collocations by {measure.upper()}")
        print(f"{'-' * 60}")

        collocations = extractor.extract_collocations(measure=measure, topk=5)

        for i, coll in enumerate(collocations, 1):
            print(f"{i}. {' '.join(coll.bigram)}")
            print(f"   Freq={coll.freq}, PMI={coll.pmi:.2f}, LLR={coll.llr:.2f}, "
                  f"χ²={coll.chi_square:.2f}, T={coll.t_score:.4f}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    demo()
