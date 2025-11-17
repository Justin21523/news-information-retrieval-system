"""Tests for Rocchio Query Expansion"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ir.ranking.rocchio import RocchioExpander, ExpandedQuery


@pytest.fixture
def expander():
    return RocchioExpander(alpha=1.0, beta=0.75, gamma=0.15)


@pytest.fixture
def query_vector():
    return {
        "information": 0.8,
        "retrieval": 0.6
    }


@pytest.fixture
def relevant_vectors():
    return [
        {"information": 0.5, "retrieval": 0.7, "system": 0.3, "database": 0.2},
        {"information": 0.6, "search": 0.4, "engine": 0.3},
        {"retrieval": 0.5, "document": 0.4, "index": 0.3}
    ]


@pytest.fixture
def nonrelevant_vectors():
    return [
        {"unrelated": 0.8, "noise": 0.5},
        {"spam": 0.7, "junk": 0.4}
    ]


@pytest.mark.unit
class TestBasicExpansion:
    def test_expand_query_with_relevant_docs(self, expander, query_vector, relevant_vectors):
        expanded = expander.expand_query(query_vector, relevant_vectors)

        assert isinstance(expanded, ExpandedQuery)
        assert len(expanded.original_terms) == 2
        assert "information" in expanded.original_terms
        assert "retrieval" in expanded.original_terms
        assert expanded.num_relevant == 3
        assert expanded.num_nonrelevant == 0

    def test_expanded_terms_not_in_original(self, expander, query_vector, relevant_vectors):
        expanded = expander.expand_query(query_vector, relevant_vectors)

        for term in expanded.expanded_terms:
            assert term not in expanded.original_terms

    def test_all_terms_include_original_and_expanded(self, expander, query_vector, relevant_vectors):
        expanded = expander.expand_query(query_vector, relevant_vectors)

        for term in expanded.original_terms:
            assert term in expanded.all_terms

        for term in expanded.expanded_terms:
            assert term in expanded.all_terms

    def test_term_weights_exist_for_all_terms(self, expander, query_vector, relevant_vectors):
        expanded = expander.expand_query(query_vector, relevant_vectors)

        for term in expanded.all_terms:
            assert term in expanded.term_weights
            assert expanded.term_weights[term] >= 0


@pytest.mark.unit
class TestWithNonRelevantDocs:
    def test_expand_with_nonrelevant_docs(self, expander, query_vector,
                                         relevant_vectors, nonrelevant_vectors):
        expanded = expander.expand_query(
            query_vector, relevant_vectors, nonrelevant_vectors
        )

        assert expanded.num_relevant == 3
        assert expanded.num_nonrelevant == 2

    def test_nonrelevant_docs_reduce_weights(self, expander, query_vector, relevant_vectors):
        # Expand without non-relevant docs
        expanded_without = expander.expand_query(query_vector, relevant_vectors)

        # Create non-relevant docs with same terms as relevant
        nonrelevant = [
            {"system": 0.9, "database": 0.8}
        ]

        # Expand with non-relevant docs
        expanded_with = expander.expand_query(
            query_vector, relevant_vectors, nonrelevant
        )

        # Weights for terms in non-relevant docs should be reduced
        if "system" in expanded_without.term_weights and "system" in expanded_with.term_weights:
            assert expanded_with.term_weights["system"] < expanded_without.term_weights["system"]


@pytest.mark.unit
class TestPseudoRelevanceFeedback:
    def test_pseudo_feedback_basic(self, expander, query_vector, relevant_vectors):
        expanded = expander.expand_with_pseudo_feedback(
            query_vector, relevant_vectors, num_relevant=3
        )

        assert expanded.num_relevant == 3
        assert expanded.num_nonrelevant == 0

    def test_pseudo_feedback_with_nonrelevant(self, expander, query_vector):
        top_docs = [
            {"term1": 0.9, "term2": 0.8},
            {"term1": 0.7, "term3": 0.6},
            {"term4": 0.5, "term5": 0.4},
            {"term6": 0.3, "term7": 0.2}
        ]

        expanded = expander.expand_with_pseudo_feedback(
            query_vector, top_docs, num_relevant=2, num_nonrelevant=2
        )

        assert expanded.num_relevant == 2
        assert expanded.num_nonrelevant == 2

    def test_pseudo_feedback_empty_docs(self, expander, query_vector):
        expanded = expander.expand_with_pseudo_feedback(
            query_vector, [], num_relevant=5
        )

        assert expanded.num_relevant == 0
        assert len(expanded.expanded_terms) == 0


@pytest.mark.unit
class TestParameters:
    def test_alpha_weight(self):
        # High alpha emphasizes original query
        expander_high_alpha = RocchioExpander(alpha=2.0, beta=0.5, gamma=0.1)

        query_vec = {"query": 1.0}
        rel_vecs = [{"other": 1.0}]

        expanded = expander_high_alpha.expand_query(query_vec, rel_vecs)

        # Original term should have higher weight
        assert expanded.term_weights["query"] > expanded.term_weights.get("other", 0)

    def test_beta_weight(self):
        # High beta emphasizes relevant docs
        expander_high_beta = RocchioExpander(alpha=1.0, beta=2.0, gamma=0.1)

        query_vec = {"query": 0.5}
        rel_vecs = [{"relevant": 1.0}]

        expanded = expander_high_beta.expand_query(query_vec, rel_vecs)

        # Relevant term should have significant weight
        assert "relevant" in expanded.term_weights

    def test_gamma_weight(self):
        # Gamma reduces weight of non-relevant terms
        expander = RocchioExpander(alpha=1.0, beta=0.75, gamma=0.5)

        query_vec = {"query": 1.0}
        rel_vecs = [{"relevant": 1.0, "shared": 0.5}]
        nonrel_vecs = [{"nonrelevant": 1.0, "shared": 0.5}]

        expanded = expander.expand_query(query_vec, rel_vecs, nonrel_vecs)

        # Shared term weight should be reduced by gamma
        assert "shared" in expanded.term_weights

    def test_set_parameters(self, expander):
        expander.set_parameters(alpha=1.5, beta=0.8, gamma=0.2)

        assert expander.alpha == 1.5
        assert expander.beta == 0.8
        assert expander.gamma == 0.2


@pytest.mark.unit
class TestExpansionControl:
    def test_max_expansion_terms(self):
        expander = RocchioExpander(max_expansion_terms=3)

        query_vec = {"query": 1.0}
        rel_vecs = [
            {"term1": 0.9, "term2": 0.8, "term3": 0.7, "term4": 0.6, "term5": 0.5}
        ]

        expanded = expander.expand_query(query_vec, rel_vecs)

        # Should have at most 3 expansion terms
        assert len(expanded.expanded_terms) <= 3

    def test_min_term_weight(self):
        expander = RocchioExpander(min_term_weight=0.5)

        query_vec = {"query": 1.0}
        rel_vecs = [
            {"high_weight": 0.9, "low_weight": 0.1}
        ]

        expanded = expander.expand_query(query_vec, rel_vecs)

        # Low weight terms should be filtered out
        for term in expanded.expanded_terms:
            assert expanded.term_weights[term] >= 0.5

    def test_zero_max_expansion_terms(self):
        expander = RocchioExpander(max_expansion_terms=0)

        query_vec = {"query": 1.0}
        rel_vecs = [{"term1": 0.9}]

        expanded = expander.expand_query(query_vec, rel_vecs)

        assert len(expanded.expanded_terms) == 0


@pytest.mark.unit
class TestReweighting:
    def test_reweight_query(self, expander, query_vector, relevant_vectors):
        expanded = expander.expand_query(query_vector, relevant_vectors)
        reweighted = expander.reweight_query(query_vector, expanded, normalize=False)

        assert isinstance(reweighted, dict)
        assert len(reweighted) == len(expanded.all_terms)

    def test_reweight_with_normalization(self, expander, query_vector, relevant_vectors):
        expanded = expander.expand_query(query_vector, relevant_vectors)
        reweighted = expander.reweight_query(query_vector, expanded, normalize=True)

        # Weights should sum to 1
        total = sum(reweighted.values())
        assert total == pytest.approx(1.0)

    def test_reweight_without_normalization(self, expander, query_vector, relevant_vectors):
        expanded = expander.expand_query(query_vector, relevant_vectors)
        reweighted = expander.reweight_query(query_vector, expanded, normalize=False)

        # Weights should match expanded query weights
        for term in reweighted:
            assert reweighted[term] == expanded.term_weights[term]


@pytest.mark.unit
class TestTopExpansionTerms:
    def test_get_top_expansion_terms(self, expander, query_vector):
        rel_vecs = [
            {"term1": 0.9, "term2": 0.8, "term3": 0.7, "term4": 0.6, "term5": 0.5}
        ]

        expanded = expander.expand_query(query_vector, rel_vecs)
        top_terms = expander.get_top_expansion_terms(expanded, k=3)

        assert len(top_terms) <= 3
        assert isinstance(top_terms, list)

        # Should be sorted by weight descending
        weights = [weight for _, weight in top_terms]
        assert weights == sorted(weights, reverse=True)

    def test_top_terms_format(self, expander, query_vector, relevant_vectors):
        expanded = expander.expand_query(query_vector, relevant_vectors)
        top_terms = expander.get_top_expansion_terms(expanded, k=5)

        for term, weight in top_terms:
            assert isinstance(term, str)
            assert isinstance(weight, float)
            assert weight >= 0

    def test_top_terms_all_from_expanded(self, expander, query_vector, relevant_vectors):
        expanded = expander.expand_query(query_vector, relevant_vectors)
        top_terms = expander.get_top_expansion_terms(expanded, k=10)

        for term, _ in top_terms:
            assert term in expanded.expanded_terms


@pytest.mark.unit
class TestEdgeCases:
    def test_no_relevant_documents(self, expander, query_vector):
        expanded = expander.expand_query(query_vector, [])

        assert expanded.num_relevant == 0
        assert len(expanded.expanded_terms) == 0
        assert expanded.all_terms == list(query_vector.keys())

    def test_empty_query_vector(self, expander, relevant_vectors):
        empty_query = {}
        expanded = expander.expand_query(empty_query, relevant_vectors)

        # Should have expansion terms but no original terms
        assert len(expanded.original_terms) == 0
        assert len(expanded.expanded_terms) > 0

    def test_no_overlap_terms(self, expander):
        query_vec = {"query_term": 1.0}
        rel_vecs = [{"completely_different": 1.0}]

        expanded = expander.expand_query(query_vec, rel_vecs)

        assert "query_term" in expanded.all_terms
        assert "completely_different" in expanded.all_terms or len(expanded.expanded_terms) == 0

    def test_negative_weights_filtered(self, expander):
        query_vec = {"term1": 1.0}
        rel_vecs = [{"term2": 0.5}]
        nonrel_vecs = [{"term2": 2.0}]  # High non-relevant weight

        expanded = expander.expand_query(query_vec, rel_vecs, nonrel_vecs)

        # All weights should be non-negative
        for weight in expanded.term_weights.values():
            assert weight >= 0

    def test_single_relevant_document(self, expander, query_vector):
        rel_vecs = [{"system": 0.5, "database": 0.3}]

        expanded = expander.expand_query(query_vector, rel_vecs)

        assert expanded.num_relevant == 1
        assert isinstance(expanded.expanded_terms, list)


@pytest.mark.unit
class TestRocchioFormula:
    def test_formula_with_all_components(self):
        """Test complete Rocchio formula: α*Q + β*R - γ*N"""
        expander = RocchioExpander(alpha=1.0, beta=1.0, gamma=1.0)

        query_vec = {"term": 1.0}
        rel_vecs = [{"term": 1.0}]  # +1.0 contribution
        nonrel_vecs = [{"term": 0.5}]  # -0.5 contribution

        expanded = expander.expand_query(query_vec, rel_vecs, nonrel_vecs)

        # Expected: 1.0*1.0 + 1.0*1.0 - 1.0*0.5 = 1.5
        expected_weight = 1.0 + 1.0 - 0.5
        assert expanded.term_weights["term"] == pytest.approx(expected_weight)

    def test_formula_averaging_relevant_docs(self):
        """Test that relevant docs are averaged: β * (1/|Dr|) * Σ Dr"""
        expander = RocchioExpander(alpha=0.0, beta=1.0, gamma=0.0)

        query_vec = {}
        rel_vecs = [
            {"term": 1.0},
            {"term": 3.0}
        ]

        expanded = expander.expand_query(query_vec, rel_vecs)

        # Expected: (1.0 + 3.0) / 2 = 2.0
        assert expanded.term_weights.get("term", 0) == pytest.approx(2.0)


@pytest.mark.unit
class TestIntegration:
    def test_full_workflow(self, expander):
        """Test complete workflow from query to expansion"""
        # Step 1: Original query
        query_vec = {"search": 0.8, "engine": 0.6}

        # Step 2: Get relevant and non-relevant docs
        rel_vecs = [
            {"search": 0.7, "engine": 0.5, "web": 0.4, "index": 0.3},
            {"search": 0.6, "query": 0.5, "result": 0.4}
        ]
        nonrel_vecs = [
            {"spam": 0.9, "advertising": 0.7}
        ]

        # Step 3: Expand
        expanded = expander.expand_query(query_vec, rel_vecs, nonrel_vecs)

        # Step 4: Verify results
        assert len(expanded.original_terms) == 2
        assert len(expanded.expanded_terms) > 0
        assert expanded.num_relevant == 2
        assert expanded.num_nonrelevant == 1

        # Step 5: Get top terms
        top_terms = expander.get_top_expansion_terms(expanded, k=3)
        assert len(top_terms) <= 3

        # Step 6: Reweight
        reweighted = expander.reweight_query(query_vec, expanded, normalize=True)
        assert sum(reweighted.values()) == pytest.approx(1.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
