"""Tests for Evaluation Metrics"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ir.eval.metrics import Metrics, EvaluationResult


@pytest.fixture
def metrics():
    return Metrics()


@pytest.fixture
def sample_retrieved():
    return [1, 2, 3, 4, 5]


@pytest.fixture
def sample_relevant():
    return {1, 3, 5}


@pytest.fixture
def sample_relevance_scores():
    return {1: 3, 2: 0, 3: 2, 4: 0, 5: 3}


@pytest.mark.unit
class TestPrecisionRecall:
    def test_precision(self, metrics, sample_retrieved, sample_relevant):
        p = metrics.precision(sample_retrieved, sample_relevant)
        assert p == pytest.approx(0.6)  # 3/5

    def test_recall(self, metrics, sample_retrieved, sample_relevant):
        r = metrics.recall(sample_retrieved, sample_relevant)
        assert r == pytest.approx(1.0)  # 3/3

    def test_precision_empty_retrieved(self, metrics, sample_relevant):
        p = metrics.precision([], sample_relevant)
        assert p == 0.0

    def test_recall_empty_relevant(self, metrics, sample_retrieved):
        r = metrics.recall(sample_retrieved, set())
        assert r == 0.0

    def test_precision_no_relevant_retrieved(self, metrics):
        retrieved = [2, 4, 6]
        relevant = {1, 3, 5}
        p = metrics.precision(retrieved, relevant)
        assert p == 0.0

    def test_recall_all_relevant_retrieved(self, metrics):
        retrieved = [1, 2, 3, 4, 5]
        relevant = {1, 3, 5}
        r = metrics.recall(retrieved, relevant)
        assert r == 1.0


@pytest.mark.unit
class TestFMeasure:
    def test_f1_score(self, metrics):
        f1 = metrics.f_measure(0.6, 1.0)
        assert f1 == pytest.approx(0.75)

    def test_f2_score(self, metrics):
        f2 = metrics.f_measure(0.6, 1.0, beta=2.0)
        assert f2 > 0.75  # F2 favors recall

    def test_f_half_score(self, metrics):
        f_half = metrics.f_measure(0.6, 1.0, beta=0.5)
        assert f_half < 0.75  # F0.5 favors precision

    def test_f_measure_zero(self, metrics):
        f = metrics.f_measure(0.0, 0.0)
        assert f == 0.0


@pytest.mark.unit
class TestPrecisionAtK:
    def test_precision_at_5(self, metrics):
        retrieved = [1, 2, 3, 4, 5, 6, 7]
        relevant = {1, 3, 5}
        p_at_5 = metrics.precision_at_k(retrieved, relevant, k=5)
        assert p_at_5 == pytest.approx(0.6)  # 3/5

    def test_precision_at_3(self, metrics):
        retrieved = [1, 2, 3, 4, 5]
        relevant = {1, 3, 5}
        p_at_3 = metrics.precision_at_k(retrieved, relevant, k=3)
        assert p_at_3 == pytest.approx(2/3)  # 2/3

    def test_precision_at_k_exceeds_results(self, metrics):
        retrieved = [1, 2, 3]
        relevant = {1, 3}
        p_at_10 = metrics.precision_at_k(retrieved, relevant, k=10)
        assert p_at_10 == pytest.approx(2/3)  # Only 3 results available

    def test_precision_at_k_zero(self, metrics):
        retrieved = [1, 2, 3]
        relevant = {1, 3}
        p_at_0 = metrics.precision_at_k(retrieved, relevant, k=0)
        assert p_at_0 == 0.0


@pytest.mark.unit
class TestRecallAtK:
    def test_recall_at_5(self, metrics):
        retrieved = [1, 2, 3, 4, 5, 6, 7]
        relevant = {1, 3, 5, 7, 9}
        r_at_5 = metrics.recall_at_k(retrieved, relevant, k=5)
        assert r_at_5 == pytest.approx(0.6)  # 3/5

    def test_recall_at_k_all_retrieved(self, metrics):
        retrieved = [1, 2, 3, 4, 5]
        relevant = {1, 3, 5}
        r_at_5 = metrics.recall_at_k(retrieved, relevant, k=5)
        assert r_at_5 == 1.0


@pytest.mark.unit
class TestAveragePrecision:
    def test_ap_perfect_ranking(self, metrics):
        # All relevant docs at the beginning
        retrieved = [1, 3, 5, 2, 4]
        relevant = {1, 3, 5}
        ap = metrics.average_precision(retrieved, relevant)
        # P@1=1.0, P@2=1.0, P@3=1.0
        # AP = (1.0 + 1.0 + 1.0) / 3 = 1.0
        assert ap == pytest.approx(1.0)

    def test_ap_interleaved(self, metrics):
        # Relevant docs interleaved
        retrieved = [1, 2, 3, 4, 5]
        relevant = {1, 3, 5}
        ap = metrics.average_precision(retrieved, relevant)
        # P@1=1.0, P@3=2/3, P@5=3/5
        # AP = (1.0 + 2/3 + 3/5) / 3 ≈ 0.756
        assert ap == pytest.approx(0.756, abs=0.001)

    def test_ap_worst_ranking(self, metrics):
        # All relevant docs at the end
        retrieved = [2, 4, 6, 1, 3, 5]
        relevant = {1, 3, 5}
        ap = metrics.average_precision(retrieved, relevant)
        # P@4=1/4, P@5=2/5, P@6=3/6=0.5
        # AP = (0.25 + 0.4 + 0.5) / 3 ≈ 0.383
        assert ap == pytest.approx(0.383, abs=0.001)

    def test_ap_no_relevant(self, metrics):
        retrieved = [1, 2, 3]
        relevant = set()
        ap = metrics.average_precision(retrieved, relevant)
        assert ap == 0.0

    def test_ap_no_relevant_retrieved(self, metrics):
        retrieved = [2, 4, 6]
        relevant = {1, 3, 5}
        ap = metrics.average_precision(retrieved, relevant)
        assert ap == 0.0


@pytest.mark.unit
class TestMeanAveragePrecision:
    def test_map_multiple_queries(self, metrics):
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
        map_score = metrics.mean_average_precision({}, {})
        assert map_score == 0.0

    def test_map_single_query(self, metrics):
        results = {'q1': [1, 2, 3]}
        qrels = {'q1': {1, 3}}
        map_score = metrics.mean_average_precision(results, qrels)
        ap = metrics.average_precision([1, 2, 3], {1, 3})
        assert map_score == pytest.approx(ap)


@pytest.mark.unit
class TestReciprocalRank:
    def test_rr_first_position(self, metrics):
        retrieved = [1, 2, 3, 4]
        relevant = {1}
        rr = metrics.reciprocal_rank(retrieved, relevant)
        assert rr == 1.0

    def test_rr_second_position(self, metrics):
        retrieved = [2, 1, 3, 4]
        relevant = {1}
        rr = metrics.reciprocal_rank(retrieved, relevant)
        assert rr == 0.5

    def test_rr_third_position(self, metrics):
        retrieved = [2, 4, 1, 3]
        relevant = {1, 3}
        rr = metrics.reciprocal_rank(retrieved, relevant)
        assert rr == pytest.approx(1/3)

    def test_rr_no_relevant(self, metrics):
        retrieved = [1, 2, 3]
        relevant = {4, 5}
        rr = metrics.reciprocal_rank(retrieved, relevant)
        assert rr == 0.0


@pytest.mark.unit
class TestMeanReciprocalRank:
    def test_mrr_multiple_queries(self, metrics):
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
    def test_dcg_at_k(self, metrics):
        retrieved = [1, 2, 3, 4, 5]
        relevance = {1: 3, 2: 2, 3: 3, 4: 0, 5: 1}
        dcg = metrics.dcg_at_k(retrieved, relevance, k=5)
        assert dcg > 0.0

    def test_dcg_at_k_3(self, metrics):
        retrieved = [1, 2, 3]
        relevance = {1: 3, 2: 2, 3: 1}
        dcg = metrics.dcg_at_k(retrieved, relevance, k=3)
        # DCG = (2^3-1)/log2(2) + (2^2-1)/log2(3) + (2^1-1)/log2(4)
        # DCG = 7/1 + 3/1.585 + 1/2 = 7 + 1.893 + 0.5 = 9.393
        assert dcg == pytest.approx(9.393, abs=0.01)

    def test_dcg_zero_relevance(self, metrics):
        retrieved = [1, 2, 3]
        relevance = {1: 0, 2: 0, 3: 0}
        dcg = metrics.dcg_at_k(retrieved, relevance, k=3)
        assert dcg == 0.0


@pytest.mark.unit
class TestNDCG:
    def test_ndcg_perfect_ranking(self, metrics):
        # Perfect ranking: sorted by relevance
        retrieved = [1, 3, 2, 5, 4]
        relevance = {1: 3, 2: 2, 3: 3, 4: 0, 5: 1}
        ndcg = metrics.ndcg_at_k(retrieved, relevance, k=5)
        assert ndcg == pytest.approx(1.0, abs=0.01)

    def test_ndcg_worst_ranking(self, metrics):
        # Worst ranking: reverse sorted
        retrieved = [4, 5, 2, 3, 1]
        relevance = {1: 3, 2: 2, 3: 3, 4: 0, 5: 1}
        ndcg = metrics.ndcg_at_k(retrieved, relevance, k=5)
        assert ndcg < 0.7  # Significantly worse than perfect

    def test_ndcg_at_k_3(self, metrics):
        retrieved = [1, 2, 3]
        relevance = {1: 3, 2: 2, 3: 1}
        ndcg = metrics.ndcg_at_k(retrieved, relevance, k=3)
        # This is perfect ranking for top-3
        assert ndcg == pytest.approx(1.0)

    def test_ndcg_zero_ideal(self, metrics):
        retrieved = [1, 2, 3]
        relevance = {1: 0, 2: 0, 3: 0}
        ndcg = metrics.ndcg_at_k(retrieved, relevance, k=3)
        assert ndcg == 0.0


@pytest.mark.unit
class TestEvaluateQuery:
    def test_evaluate_query_basic(self, metrics):
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
    def test_evaluate_run_multiple_queries(self, metrics):
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
    def test_empty_retrieved_list(self, metrics):
        p = metrics.precision([], {1, 2, 3})
        r = metrics.recall([], {1, 2, 3})
        ap = metrics.average_precision([], {1, 2, 3})

        assert p == 0.0
        assert r == 0.0
        assert ap == 0.0

    def test_empty_relevant_set(self, metrics):
        p = metrics.precision([1, 2, 3], set())
        r = metrics.recall([1, 2, 3], set())
        ap = metrics.average_precision([1, 2, 3], set())

        assert p == 0.0
        assert r == 0.0
        assert ap == 0.0

    def test_no_overlap(self, metrics):
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
        retrieved = [1, 2, 3]
        relevant = {1, 2, 3}

        p = metrics.precision(retrieved, relevant)
        r = metrics.recall(retrieved, relevant)

        assert p == 1.0
        assert r == 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
