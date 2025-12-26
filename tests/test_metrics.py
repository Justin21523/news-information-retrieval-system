"""Tests for Evaluation Metrics"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ir.eval.metrics import Metrics, EvaluationResult


@pytest.fixture
def metrics():
    """Return a Metrics instance for evaluation tests."""
    return Metrics()


@pytest.fixture
def sample_retrieved():
    """Return a fixed retrieved list used across metric unit tests."""
    return [1, 2, 3, 4, 5]


@pytest.fixture
def sample_relevant():
    """Return a fixed relevant set used across metric unit tests."""
    return {1, 3, 5}


@pytest.fixture
def sample_relevance_scores():
    """Return a fixed graded relevance mapping for DCG/nDCG unit tests."""
    return {1: 3, 2: 0, 3: 2, 4: 0, 5: 3}


@pytest.mark.unit
class TestPrecisionRecall:
    """Unit tests for precision and recall calculations."""
    def test_precision(self, metrics, sample_retrieved, sample_relevant):
        """Compute precision as |retrieved ∩ relevant| / |retrieved|."""
        p = metrics.precision(sample_retrieved, sample_relevant)
        assert p == pytest.approx(0.6)  # 3/5

    def test_recall(self, metrics, sample_retrieved, sample_relevant):
        """Compute recall as |retrieved ∩ relevant| / |relevant|."""
        r = metrics.recall(sample_retrieved, sample_relevant)
        assert r == pytest.approx(1.0)  # 3/3

    def test_precision_empty_retrieved(self, metrics, sample_relevant):
        """Return precision 0.0 when the retrieved list is empty."""
        p = metrics.precision([], sample_relevant)
        assert p == 0.0

    def test_recall_empty_relevant(self, metrics, sample_retrieved):
        """Return recall 0.0 when the relevant set is empty."""
        r = metrics.recall(sample_retrieved, set())
        assert r == 0.0

    def test_precision_no_relevant_retrieved(self, metrics):
        """Return precision 0.0 when retrieved documents contain no relevant hits."""
        retrieved = [2, 4, 6]
        relevant = {1, 3, 5}
        p = metrics.precision(retrieved, relevant)
        assert p == 0.0

    def test_recall_all_relevant_retrieved(self, metrics):
        """Return recall 1.0 when all relevant documents are retrieved."""
        retrieved = [1, 2, 3, 4, 5]
        relevant = {1, 3, 5}
        r = metrics.recall(retrieved, relevant)
        assert r == 1.0


@pytest.mark.unit
class TestFMeasure:
    """Unit tests for F-measure variants (F1, F2, F0.5)."""
    def test_f1_score(self, metrics):
        """Compute F1 (beta=1) from precision and recall."""
        f1 = metrics.f_measure(0.6, 1.0)
        assert f1 == pytest.approx(0.75)

    def test_f2_score(self, metrics):
        """Compute F2 which favors recall more heavily than precision."""
        f2 = metrics.f_measure(0.6, 1.0, beta=2.0)
        assert f2 > 0.75  # F2 favors recall

    def test_f_half_score(self, metrics):
        """Compute F0.5 which favors precision more heavily than recall."""
        f_half = metrics.f_measure(0.6, 1.0, beta=0.5)
        assert f_half < 0.75  # F0.5 favors precision

    def test_f_measure_zero(self, metrics):
        """Return 0.0 when both precision and recall are 0.0."""
        f = metrics.f_measure(0.0, 0.0)
        assert f == 0.0


@pytest.mark.unit
class TestPrecisionAtK:
    """Unit tests for Precision@k behavior."""
    def test_precision_at_5(self, metrics):
        """Compute Precision@5 for a ranked list longer than k."""
        retrieved = [1, 2, 3, 4, 5, 6, 7]
        relevant = {1, 3, 5}
        p_at_5 = metrics.precision_at_k(retrieved, relevant, k=5)
        assert p_at_5 == pytest.approx(0.6)  # 3/5

    def test_precision_at_3(self, metrics):
        """Compute Precision@3 for the prefix of the ranked list."""
        retrieved = [1, 2, 3, 4, 5]
        relevant = {1, 3, 5}
        p_at_3 = metrics.precision_at_k(retrieved, relevant, k=3)
        assert p_at_3 == pytest.approx(2/3)  # 2/3

    def test_precision_at_k_exceeds_results(self, metrics):
        """Handle k larger than list size by using the available results."""
        retrieved = [1, 2, 3]
        relevant = {1, 3}
        p_at_10 = metrics.precision_at_k(retrieved, relevant, k=10)
        assert p_at_10 == pytest.approx(2/3)  # Only 3 results available

    def test_precision_at_k_zero(self, metrics):
        """Return Precision@0 as 0.0 by convention."""
        retrieved = [1, 2, 3]
        relevant = {1, 3}
        p_at_0 = metrics.precision_at_k(retrieved, relevant, k=0)
        assert p_at_0 == 0.0


@pytest.mark.unit
class TestRecallAtK:
    """Unit tests for Recall@k behavior."""
    def test_recall_at_5(self, metrics):
        """Compute Recall@5 when the relevant set is larger than k."""
        retrieved = [1, 2, 3, 4, 5, 6, 7]
        relevant = {1, 3, 5, 7, 9}
        r_at_5 = metrics.recall_at_k(retrieved, relevant, k=5)
        assert r_at_5 == pytest.approx(0.6)  # 3/5

    def test_recall_at_k_all_retrieved(self, metrics):
        """Return Recall@k = 1.0 when all relevant docs appear in top-k."""
        retrieved = [1, 2, 3, 4, 5]
        relevant = {1, 3, 5}
        r_at_5 = metrics.recall_at_k(retrieved, relevant, k=5)
        assert r_at_5 == 1.0


@pytest.mark.unit
class TestAveragePrecision:
    """Unit tests for Average Precision (AP)."""
    def test_ap_perfect_ranking(self, metrics):
        """Return AP=1.0 when all relevant documents rank before non-relevant ones."""
        # All relevant docs at the beginning
        retrieved = [1, 3, 5, 2, 4]
        relevant = {1, 3, 5}
        ap = metrics.average_precision(retrieved, relevant)
        # P@1=1.0, P@2=1.0, P@3=1.0
        # AP = (1.0 + 1.0 + 1.0) / 3 = 1.0
        assert ap == pytest.approx(1.0)

    def test_ap_interleaved(self, metrics):
        """Match a hand-calculated AP for interleaved relevant documents."""
        # Relevant docs interleaved
        retrieved = [1, 2, 3, 4, 5]
        relevant = {1, 3, 5}
        ap = metrics.average_precision(retrieved, relevant)
        # P@1=1.0, P@3=2/3, P@5=3/5
        # AP = (1.0 + 2/3 + 3/5) / 3 ≈ 0.756
        assert ap == pytest.approx(0.756, abs=0.001)

    def test_ap_worst_ranking(self, metrics):
        """Return a smaller AP when relevant documents are pushed to the tail."""
        # All relevant docs at the end
        retrieved = [2, 4, 6, 1, 3, 5]
        relevant = {1, 3, 5}
        ap = metrics.average_precision(retrieved, relevant)
        # P@4=1/4, P@5=2/5, P@6=3/6=0.5
        # AP = (0.25 + 0.4 + 0.5) / 3 ≈ 0.383
        assert ap == pytest.approx(0.383, abs=0.001)

    def test_ap_no_relevant(self, metrics):
        """Return AP=0.0 when the query has no relevant documents in qrels."""
        retrieved = [1, 2, 3]
        relevant = set()
        ap = metrics.average_precision(retrieved, relevant)
        assert ap == 0.0

    def test_ap_no_relevant_retrieved(self, metrics):
        """Return AP=0.0 when retrieved documents contain no relevant hits."""
        retrieved = [2, 4, 6]
        relevant = {1, 3, 5}
        ap = metrics.average_precision(retrieved, relevant)
        assert ap == 0.0


@pytest.mark.unit
class TestMeanAveragePrecision:
    """Unit tests for Mean Average Precision (MAP)."""
    def test_map_multiple_queries(self, metrics):
        """Compute MAP across multiple queries and keep it within [0, 1]."""
        results = {
            'q1': [1, 2, 3, 4],
            'q2': [5, 6, 7, 8],
            'q3': [9, 10, 11, 12]
        }
        qrels = {
            'q1': {1, 3},
            'q2': {6, 8},
            'q3': {11}
        }
        map_score = metrics.mean_average_precision(results, qrels)
        assert 0.0 < map_score < 1.0

    def test_map_empty(self, metrics):
        """Return MAP=0.0 when no queries are provided."""
        map_score = metrics.mean_average_precision({}, {})
        assert map_score == 0.0

    def test_map_single_query(self, metrics):
        """Return MAP equal to AP when there is exactly one query."""
        results = {'q1': [1, 2, 3]}
        qrels = {'q1': {1, 3}}
        map_score = metrics.mean_average_precision(results, qrels)
        ap = metrics.average_precision([1, 2, 3], {1, 3})
        assert map_score == pytest.approx(ap)


@pytest.mark.unit
class TestReciprocalRank:
    """Unit tests for Reciprocal Rank (RR)."""
    def test_rr_first_position(self, metrics):
        """Return RR=1.0 when the first retrieved document is relevant."""
        retrieved = [1, 2, 3, 4]
        relevant = {1}
        rr = metrics.reciprocal_rank(retrieved, relevant)
        assert rr == 1.0

    def test_rr_second_position(self, metrics):
        """Return RR=0.5 when the second retrieved document is the first relevant hit."""
        retrieved = [2, 1, 3, 4]
        relevant = {1}
        rr = metrics.reciprocal_rank(retrieved, relevant)
        assert rr == 0.5

    def test_rr_third_position(self, metrics):
        """Return RR=1/3 when the third retrieved document is the first relevant hit."""
        retrieved = [2, 4, 1, 3]
        relevant = {1, 3}
        rr = metrics.reciprocal_rank(retrieved, relevant)
        assert rr == pytest.approx(1/3)

    def test_rr_no_relevant(self, metrics):
        """Return RR=0.0 when no relevant documents are retrieved."""
        retrieved = [1, 2, 3]
        relevant = {4, 5}
        rr = metrics.reciprocal_rank(retrieved, relevant)
        assert rr == 0.0


@pytest.mark.unit
class TestMeanReciprocalRank:
    """Unit tests for Mean Reciprocal Rank (MRR)."""
    def test_mrr_multiple_queries(self, metrics):
        """Compute MRR as the mean of per-query reciprocal ranks."""
        results = {
            'q1': [1, 2, 3],  # RR = 1.0
            'q2': [2, 1, 3],  # RR = 0.5
            'q3': [2, 3, 1]   # RR = 1/3
        }
        qrels = {
            'q1': {1},
            'q2': {1},
            'q3': {1}
        }
        mrr = metrics.mean_reciprocal_rank(results, qrels)
        # MRR = (1.0 + 0.5 + 1/3) / 3 ≈ 0.611
        assert mrr == pytest.approx(0.611, abs=0.001)


@pytest.mark.unit
class TestDCG:
    """Unit tests for Discounted Cumulative Gain (DCG)."""
    def test_dcg_at_k(self, metrics):
        """Compute DCG@k for graded relevance and ensure it is positive."""
        retrieved = [1, 2, 3, 4, 5]
        relevance = {1: 3, 2: 2, 3: 3, 4: 0, 5: 1}
        dcg = metrics.dcg_at_k(retrieved, relevance, k=5)
        assert dcg > 0.0

    def test_dcg_at_k_3(self, metrics):
        """Match a hand-computed DCG@3 value using log2 discounting."""
        retrieved = [1, 2, 3]
        relevance = {1: 3, 2: 2, 3: 1}
        dcg = metrics.dcg_at_k(retrieved, relevance, k=3)
        # DCG = (2^3-1)/log2(2) + (2^2-1)/log2(3) + (2^1-1)/log2(4)
        # DCG = 7/1 + 3/1.585 + 1/2 = 7 + 1.893 + 0.5 = 9.393
        assert dcg == pytest.approx(9.393, abs=0.01)

    def test_dcg_zero_relevance(self, metrics):
        """Return DCG@k=0.0 when all graded relevance values are zero."""
        retrieved = [1, 2, 3]
        relevance = {1: 0, 2: 0, 3: 0}
        dcg = metrics.dcg_at_k(retrieved, relevance, k=3)
        assert dcg == 0.0


@pytest.mark.unit
class TestNDCG:
    """Unit tests for normalized DCG (nDCG)."""
    def test_ndcg_perfect_ranking(self, metrics):
        """Return nDCG close to 1.0 for an ideal ranking."""
        # Perfect ranking: sorted by relevance
        retrieved = [1, 3, 2, 5, 4]
        relevance = {1: 3, 2: 2, 3: 3, 4: 0, 5: 1}
        ndcg = metrics.ndcg_at_k(retrieved, relevance, k=5)
        assert ndcg == pytest.approx(1.0, abs=0.01)

    def test_ndcg_worst_ranking(self, metrics):
        """Return a substantially smaller nDCG for a poor (reversed) ranking."""
        # Worst ranking: reverse sorted
        retrieved = [4, 5, 2, 3, 1]
        relevance = {1: 3, 2: 2, 3: 3, 4: 0, 5: 1}
        ndcg = metrics.ndcg_at_k(retrieved, relevance, k=5)
        assert ndcg < 0.7  # Significantly worse than perfect

    def test_ndcg_at_k_3(self, metrics):
        """Return nDCG@3 = 1.0 when the top-3 ranking is ideal."""
        retrieved = [1, 2, 3]
        relevance = {1: 3, 2: 2, 3: 1}
        ndcg = metrics.ndcg_at_k(retrieved, relevance, k=3)
        # This is perfect ranking for top-3
        assert ndcg == pytest.approx(1.0)

    def test_ndcg_zero_ideal(self, metrics):
        """Return nDCG@k=0.0 when the ideal DCG is zero (no relevant signal)."""
        retrieved = [1, 2, 3]
        relevance = {1: 0, 2: 0, 3: 0}
        ndcg = metrics.ndcg_at_k(retrieved, relevance, k=3)
        assert ndcg == 0.0


@pytest.mark.unit
class TestEvaluateQuery:
    """Unit tests for per-query evaluation helper."""
    def test_evaluate_query_basic(self, metrics):
        """Return the expected metric keys and values for a basic query evaluation."""
        retrieved = [1, 2, 3, 4, 5]
        relevant = {1, 3, 5}
        results = metrics.evaluate_query(retrieved, relevant)

        assert 'precision' in results
        assert 'recall' in results
        assert 'f1' in results
        assert 'ap' in results
        assert 'rr' in results
        assert results['precision'] == pytest.approx(0.6)
        assert results['recall'] == pytest.approx(1.0)

    def test_evaluate_query_with_graded_relevance(self, metrics):
        """Include @k and graded metrics (e.g., nDCG) when relevance scores are provided."""
        retrieved = [1, 2, 3, 4, 5]
        relevant = {1, 3, 5}
        relevance = {1: 3, 2: 0, 3: 2, 4: 0, 5: 3}

        results = metrics.evaluate_query(
            retrieved, relevant, relevance, k_values=[3, 5]
        )

        assert 'ndcg@3' in results
        assert 'ndcg@5' in results
        assert 'p@3' in results
        assert 'r@5' in results


@pytest.mark.unit
class TestEvaluateRun:
    """Unit tests for multi-query run evaluation helper."""
    def test_evaluate_run_multiple_queries(self, metrics):
        """Aggregate evaluation metrics over multiple queries and return summary keys."""
        results = {
            'q1': [1, 2, 3, 4],
            'q2': [5, 6, 7, 8],
            'q3': [9, 10, 11, 12]
        }
        qrels = {
            'q1': {1, 3},
            'q2': {6, 8},
            'q3': {11}
        }

        eval_results = metrics.evaluate_run(results, qrels, k_values=[5, 10])

        assert 'map' in eval_results
        assert 'mrr' in eval_results
        assert 'precision' in eval_results
        assert 'recall' in eval_results
        assert 'p@5' in eval_results
        assert 'p@10' in eval_results

    def test_evaluate_run_with_graded_relevance(self, metrics):
        """Support graded relevance judgments when evaluating a run."""
        results = {
            'q1': [1, 2, 3],
            'q2': [4, 5, 6]
        }
        qrels = {
            'q1': {1, 3},
            'q2': {5}
        }
        relevance_scores = {
            'q1': {1: 3, 2: 0, 3: 2},
            'q2': {4: 0, 5: 3, 6: 1}
        }

        eval_results = metrics.evaluate_run(
            results, qrels, relevance_scores, k_values=[3]
        )

        assert 'ndcg@3' in eval_results
        assert eval_results['ndcg@3'] > 0.0


@pytest.mark.unit
class TestEdgeCases:
    """Unit tests for metric edge cases and degenerate inputs."""
    def test_empty_retrieved_list(self, metrics):
        """Return 0.0 for all metrics when no documents are retrieved."""
        p = metrics.precision([], {1, 2, 3})
        r = metrics.recall([], {1, 2, 3})
        ap = metrics.average_precision([], {1, 2, 3})

        assert p == 0.0
        assert r == 0.0
        assert ap == 0.0

    def test_empty_relevant_set(self, metrics):
        """Return 0.0 for all metrics when the relevant set is empty."""
        p = metrics.precision([1, 2, 3], set())
        r = metrics.recall([1, 2, 3], set())
        ap = metrics.average_precision([1, 2, 3], set())

        assert p == 0.0
        assert r == 0.0
        assert ap == 0.0

    def test_no_overlap(self, metrics):
        """Return 0.0 metrics when retrieved and relevant sets do not overlap."""
        retrieved = [1, 2, 3]
        relevant = {4, 5, 6}

        p = metrics.precision(retrieved, relevant)
        r = metrics.recall(retrieved, relevant)
        ap = metrics.average_precision(retrieved, relevant)
        rr = metrics.reciprocal_rank(retrieved, relevant)

        assert p == 0.0
        assert r == 0.0
        assert ap == 0.0
        assert rr == 0.0

    def test_complete_overlap(self, metrics):
        """Return precision=recall=1.0 when retrieved equals the relevant set."""
        retrieved = [1, 2, 3]
        relevant = {1, 2, 3}

        p = metrics.precision(retrieved, relevant)
        r = metrics.recall(retrieved, relevant)

        assert p == 1.0
        assert r == 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
