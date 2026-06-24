"""Search, snippet, highlighting, and facet services for the Flask app."""

from __future__ import annotations

import html
import math
import re
import time
from collections import Counter
from typing import Any

from src.ir_app.config import Settings
from src.ir_app.schemas import SearchResult
from src.ir_app.services.document_service import DocumentService
from src.ir_app.services.facet_service import FacetService
from src.ir_app.services.index_service import IndexService


class FeatureUnavailableError(RuntimeError):
    """Raised when an optional retrieval feature is unavailable."""


class SearchService:
    """Stable lexical search service for the Flask demo.

    Complexity:
        Time: O(T) initialization, O(q * p) search where p is postings scanned
        Space: O(T)
    """

    def __init__(self, settings: Settings, document_service: DocumentService):
        self.settings = settings
        self.document_service = document_service
        self.index = IndexService(
            document_service.documents,
            settings.tokenizer_engine,
            settings.index_dir,
            {
                "dataset_source": document_service.dataset_source,
                "dataset_hash": document_service.dataset_hash,
                "dataset_mtime": document_service.dataset_mtime,
                "dataset_size": document_service.dataset_size,
            },
        )
        self.documents_by_id = {
            str(doc["doc_id"]): doc for doc in self.document_service.documents
        }
        self.facet_service = FacetService(self.document_service.documents)

    def stats(self) -> dict[str, Any]:
        """Return search/index stats.

        Complexity:
            Time: O(n)
            Space: O(1)
        """
        stats = self.document_service.stats()
        stats.update(
            {
                "total_terms": self.index.vocabulary_size(),
                "models": [
                    "boolean",
                    "tfidf",
                    "bm25",
                    "hybrid",
                    "lm",
                    "fuzzy",
                    "csoundex",
                ],
                "optional_models": ["bert"],
                "tokenizer_engine": self.settings.tokenizer_engine,
                "heavy_models_enabled": self.settings.enable_heavy_models,
                "index": self.index.stats(),
                "corpus_distribution": self.facet_service.corpus_distribution(),
            }
        )
        return stats

    def search(
        self,
        query: str,
        model: str = "bm25",
        top_k: int = 20,
        operator: str = "AND",
        filters: dict[str, list[str]] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Run a search and return normalized API result dictionaries.

        Complexity:
            Time: O(q * p + r log r)
            Space: O(r)
        """
        started = time.perf_counter()
        query = (query or "").strip()
        if not query:
            return [], {
                "execution_time": time.perf_counter() - started,
                "query_terms": [],
            }

        model = (model or "bm25").lower()
        top_k = max(1, min(int(top_k or 20), 100))
        query_terms = self.index.tokenize(query)
        quality_terms = self.index.text_quality.matching_terms(query_terms, query)
        retrieval_top_k = len(self.documents_by_id) if filters else top_k

        expanded_terms: list[str] = []
        component_scores: dict[str, dict[str, float]] = {}

        if model == "boolean":
            ranked = self._search_boolean(query, query_terms, operator)
        elif model in {"tfidf", "vsm"}:
            ranked = self._search_tfidf(query, retrieval_top_k)
            model = "tfidf"
        elif model == "bm25":
            ranked = self._search_bm25(query, retrieval_top_k)
        elif model == "hybrid":
            ranked, component_scores = self._search_hybrid(query, retrieval_top_k)
        elif model == "lm":
            ranked = self._search_lm(query, retrieval_top_k)
        elif model == "fuzzy":
            expanded_terms = self._fuzzy_expansion(query_terms)
            ranked = self._search_bm25(
                " ".join(expanded_terms or query_terms), retrieval_top_k
            )
        elif model == "csoundex":
            expanded_terms = self._csoundex_expansion(query_terms)
            ranked = self._search_bm25(
                " ".join(expanded_terms or query_terms), retrieval_top_k
            )
        elif model == "bert":
            raise FeatureUnavailableError(
                "BERT semantic search is disabled or unavailable in lightweight startup mode."
            )
        else:
            raise ValueError(f"Unknown retrieval model: {model}")

        base_scores = {str(doc_key): float(score) for doc_key, score in ranked}
        ranked, field_boosts = self._rerank_with_field_boost(ranked, quality_terms)
        ranked = self._apply_filters(ranked, filters)
        ranked = ranked[:top_k]

        results: list[dict[str, Any]] = []
        for rank, (doc_key, score) in enumerate(ranked, 1):
            doc = self.documents_by_id.get(str(doc_key))
            if not doc:
                continue
            doc_components = dict(component_scores.get(str(doc_key), {}))
            doc_components.setdefault(
                model, base_scores.get(str(doc_key), float(score))
            )
            doc_components["field_boost"] = float(
                field_boosts.get(str(doc_key), {}).get("boost", 0.0)
            )
            results.append(
                self._make_result(
                    doc,
                    query,
                    quality_terms or query_terms,
                    float(score),
                    model,
                    rank,
                    expanded_terms=expanded_terms,
                    component_scores=doc_components,
                    field_boost=field_boosts.get(str(doc_key), {}),
                ).to_dict()
            )

        suggestions = self.suggestions(query, query_terms) if not results else []
        meta = {
            "execution_time": time.perf_counter() - started,
            "query_terms": query_terms,
            "significant_terms": self.index.text_quality.significant_terms(query_terms),
            "expanded_terms": expanded_terms,
            "dataset_source": self.document_service.dataset_source,
            "suggestions": suggestions,
        }
        return results, meta

    def _search_boolean(
        self,
        query: str,
        query_terms: list[str],
        operator: str,
    ) -> list[tuple[str, float]]:
        """Search with the formal Boolean query engine.

        Complexity:
            Time: O(q * p)
            Space: O(r)
        """
        if not query_terms:
            return []
        boolean_query = query
        if re.search(r':|"|/|\bBEFORE\b|\bAFTER\b|\bBETWEEN\b', query, re.IGNORECASE):
            self.index.ensure_boolean_structural_indexes()
        if not re.search(r"\b(AND|OR|NOT)\b|:|\"|\(|\)", query, re.IGNORECASE):
            joiner = f" {(operator or 'AND').upper()} "
            boolean_query = joiner.join(query_terms)
        result = self.index.boolean_engine.query(boolean_query, rank_results=True)
        if result.scores:
            return [
                (str(doc_id), float(result.scores.get(doc_id, 1.0)))
                for doc_id in result.doc_ids
            ]
        return [(str(doc_id), 1.0) for doc_id in result.doc_ids]

    def _search_tfidf(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Search with cached TF-IDF cosine vectors.

        Complexity:
            Time: O(q * p + c log c)
            Space: O(r)
        """
        query_terms = self.index.tokenize(query)
        query_vector = self._query_vector(query_terms)
        if not query_vector:
            return []

        candidates: set[str] = set()
        for term in query_vector:
            candidates.update(self.index.inverted_index.get(term, {}).keys())

        scored: list[tuple[str, float]] = []
        for doc_key in candidates:
            doc_vector = self.index.tfidf_vectors.get(doc_key, {})
            score = sum(
                weight * doc_vector.get(term, 0.0)
                for term, weight in query_vector.items()
            )
            if score > 0:
                scored.append((doc_key, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def _search_bm25(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Search with the formal BM25 ranker.

        Complexity:
            Time: O(q * p)
            Space: O(r)
        """
        result = self.index.bm25.search(query, topk=top_k)
        return [
            (str(doc_id), float(score))
            for doc_id, score in zip(result.doc_ids, result.scores)
        ]

    def _search_hybrid(
        self, query: str, top_k: int
    ) -> tuple[list[tuple[str, float]], dict[str, dict[str, float]]]:
        """Search with reciprocal-rank fusion over BM25 and cached TF-IDF.

        Complexity:
            Time: O(q * p + r)
            Space: O(r * n)
        """
        ranker_topk = max(50, top_k * 5)
        bm25_ranked = self._search_bm25(query, ranker_topk)
        tfidf_ranked = self._search_tfidf(query, ranker_topk)
        bm25_scores = dict(bm25_ranked)
        tfidf_scores = dict(tfidf_ranked)
        fused: dict[str, float] = {}

        for weight, ranked_list in ((0.65, bm25_ranked), (0.35, tfidf_ranked)):
            for rank, (doc_key, _) in enumerate(ranked_list, 1):
                fused[doc_key] = fused.get(doc_key, 0.0) + weight * (1.0 / (60 + rank))

        ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:top_k]
        component_scores: dict[str, dict[str, float]] = {}
        for doc_key, _ in ranked:
            component_scores[doc_key] = {
                "bm25": float(bm25_scores.get(doc_key, 0.0)),
                "tfidf": float(tfidf_scores.get(doc_key, 0.0)),
            }
        return ranked, component_scores

    def _search_lm(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Search with query likelihood language-model retrieval.

        Complexity:
            Time: O(n * q)
            Space: O(n)
        """
        query_terms = self.index.tokenize(query)
        if not query_terms:
            return []

        candidates: set[int] = set()
        for term in query_terms:
            candidates.update(
                int(doc_id) for doc_id in self.index.inverted_index.get(term, {})
            )
        if not candidates:
            candidates = set(range(self.index.language_model.doc_count))

        scored = [
            (
                str(doc_id),
                float(self.index.language_model.query_likelihood(query_terms, doc_id)),
            )
            for doc_id in candidates
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def _query_vector(self, query_terms: list[str]) -> dict[str, float]:
        """Build a normalized TF-IDF query vector.

        Complexity:
            Time: O(q)
            Space: O(q)
        """
        freqs = Counter(query_terms)
        vector = {
            term: (1.0 + math.log10(freq)) * self.index.idf.get(term, 0.0)
            for term, freq in freqs.items()
            if freq > 0
        }
        norm = math.sqrt(sum(weight * weight for weight in vector.values()))
        if norm <= 0:
            return {}
        return {term: weight / norm for term, weight in vector.items()}

    def _fuzzy_expansion(self, query_terms: list[str]) -> list[str]:
        """Expand query terms with edit-distance fuzzy matches.

        Complexity:
            Time: O(q * V * m * n)
            Space: O(k)
        """
        vocabulary = set(self.index.inverted_index.keys())
        expanded: list[str] = []
        for term in query_terms:
            expanded.extend(self.index.fuzzy.expand(term, vocabulary, max_distance=1))
        return list(dict.fromkeys(expanded))

    def _csoundex_expansion(self, query_terms: list[str]) -> list[str]:
        """Expand query terms with CSoundex phonetic similarity.

        Complexity:
            Time: O(q * V)
            Space: O(k)
        """
        vocabulary = list(self.index.inverted_index.keys())
        expanded: list[str] = []
        for term in query_terms:
            matches = self.index.csoundex.find_similar(
                term, vocabulary, threshold=0.72, topk=8
            )
            expanded.extend(match for match, _ in matches)
        return list(dict.fromkeys(expanded))

    def _apply_filters(
        self,
        ranked: list[tuple[str, float]],
        filters: dict[str, list[str]] | None,
    ) -> list[tuple[str, float]]:
        """Apply sidebar facet filters.

        Complexity:
            Time: O(r * f)
            Space: O(r)
        """
        if not filters:
            return ranked

        allowed_doc_ids = self.facet_service.matching_doc_ids(filters)
        return [
            (doc_key, score)
            for doc_key, score in ranked
            if str(doc_key) in allowed_doc_ids
        ]

    def _rerank_with_field_boost(
        self,
        ranked: list[tuple[str, float]],
        query_terms: list[str],
    ) -> tuple[list[tuple[str, float]], dict[str, dict[str, Any]]]:
        """Apply field-aware boosts on top of model scores.

        Complexity:
            Time: O(r * f * q)
            Space: O(r)
        """
        boosts: dict[str, dict[str, Any]] = {}
        adjusted: list[tuple[str, float]] = []
        for doc_key, score in ranked:
            boost_info = self._field_boost(doc_key, query_terms)
            boosts[str(doc_key)] = boost_info
            adjusted.append((doc_key, float(score) + float(boost_info["boost"])))
        adjusted.sort(key=lambda item: item[1], reverse=True)
        return adjusted, boosts

    def _field_boost(self, doc_key: str, query_terms: list[str]) -> dict[str, Any]:
        """Compute title/tags/category/content boost for one document.

        Complexity:
            Time: O(f * q)
            Space: O(q)
        """
        weights = {
            "title": 0.45,
            "tags": 0.25,
            "category": 0.18,
            "content": 0.06,
        }
        field_freqs = self.index.field_term_freqs.get(str(doc_key), {})
        matches: dict[str, list[str]] = {}
        boost = 0.0
        for field, weight in weights.items():
            freqs = field_freqs.get(field, Counter())
            hits = [term for term in query_terms if freqs.get(term, 0) > 0]
            if not hits:
                continue
            unique_hits = list(dict.fromkeys(hits))
            matches[field] = unique_hits
            boost += weight * min(len(unique_hits), 3)
        return {
            **matches,
            "boost": round(min(boost, 1.5), 6),
        }

    def _document_matches_filters(
        self, doc: dict[str, Any], filters: dict[str, set[str]]
    ) -> bool:
        """Check if a document matches all filters.

        Complexity:
            Time: O(f + t)
            Space: O(t)
        """
        for field, allowed in filters.items():
            if field in {"pub_date", "published_date", "date"}:
                value = doc.get("published_date")
            else:
                value = doc.get(field)

            if field == "tags":
                values = {str(tag) for tag in (doc.get("tags") or [])}
                if not values.intersection(allowed):
                    return False
            elif str(value) not in allowed:
                return False
        return True

    def _make_result(
        self,
        doc: dict[str, Any],
        query: str,
        query_terms: list[str],
        score: float,
        model: str,
        rank: int,
        expanded_terms: list[str] | None = None,
        component_scores: dict[str, float] | None = None,
        field_boost: dict[str, Any] | None = None,
    ) -> SearchResult:
        """Build a normalized search result.

        Complexity:
            Time: O(n * q)
            Space: O(n)
        """
        snippet_info = self._snippet(doc, query_terms)
        snippet = snippet_info["text"]
        highlighted = self._highlight(snippet, query_terms)
        matched_terms = self._matched_terms(doc, query_terms)
        doc_id = int(doc.get("doc_id"))
        component_scores = component_scores or {model: score}
        field_boost = field_boost or {"boost": 0.0}
        ranking_features = {
            "index_cache_used": self.index.cache_used,
            "field_boost": field_boost,
            "snippet_source": snippet_info["source"],
        }
        if model in {"bm25", "hybrid", "fuzzy", "csoundex"}:
            bm25_explain = self.index.bm25.explain_score(query, doc_id)
            ranking_features["bm25"] = (
                bm25_explain if "error" not in bm25_explain else {}
            )
        if model == "lm":
            lm_explain = self.index.language_model.explain_score(query, doc_id)
            ranking_features["lm"] = lm_explain if "error" not in lm_explain else {}
        return SearchResult(
            doc_id=doc.get("doc_id"),
            article_id=doc.get("article_id"),
            title=doc.get("title") or "",
            snippet=snippet,
            highlighted_snippet=highlighted,
            score=score,
            model=model,
            url=doc.get("url"),
            published_date=doc.get("published_date"),
            category=doc.get("category"),
            category_name=doc.get("category_name"),
            source=doc.get("source"),
            source_label=doc.get("source_label"),
            content_type=doc.get("content_type"),
            taxonomy_topic=doc.get("taxonomy_topic"),
            taxonomy_label=doc.get("taxonomy_label"),
            taxonomy_path=doc.get("taxonomy_path"),
            author=doc.get("author"),
            tags=doc.get("tags") or [],
            rank=rank,
            explanation={
                "matched_terms": matched_terms,
                "expanded_terms": expanded_terms or [],
                "field_matches": self._field_matches(doc, query_terms),
                "component_scores": component_scores,
                "ranking_features": ranking_features,
                "query": query,
            },
        )

    def expand_query(self, query: str, top_k: int = 5) -> dict[str, Any]:
        """Expand a query with Rocchio pseudo relevance feedback.

        Complexity:
            Time: O(k * V)
            Space: O(V)
        """
        query_terms = self.index.tokenize(query)
        if not query_terms:
            return {
                "original_query": query,
                "expanded_query": query,
                "expanded_terms": [],
                "method": "rocchio_prf",
            }

        ranked = self._search_bm25(query, top_k)
        relevant_vectors = [
            self.index.tfidf_vectors.get(str(doc_id), {}) for doc_id, _ in ranked
        ]
        doc_scores = [score for _, score in ranked]
        query_vector = {term: self.index.idf.get(term, 1.0) for term in query_terms}
        expanded = self.index.rocchio.expand_query(
            query_vector=query_vector,
            relevant_vectors=relevant_vectors,
            original_terms=set(query_terms),
            doc_scores=doc_scores,
        )
        expanded_terms = expanded.expanded_terms
        all_terms = list(dict.fromkeys(query_terms + expanded_terms))
        return {
            "original_query": query,
            "expanded_query": " ".join(all_terms),
            "expanded_terms": expanded_terms,
            "term_weights": expanded.term_weights,
            "query_drift": expanded.query_drift,
            "drift_warning": expanded.drift_warning,
            "method": "rocchio_prf",
        }

    def _snippet(
        self, doc: dict[str, Any], query_terms: list[str], window: int = 180
    ) -> dict[str, str]:
        """Generate a query-centered snippet.

        Complexity:
            Time: O(n * q)
            Space: O(n)
        """
        content = doc.get("content") or ""
        title = doc.get("title") or ""
        title_matches = self._terms_in_text(title, query_terms)
        if title_matches:
            return {"text": title, "source": "title"}
        if not content:
            return {"text": title, "source": "title"}

        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[。！？.!?])", content)
            if sentence.strip()
        ]
        scored: list[tuple[int, int, str]] = []
        for index, sentence in enumerate(sentences):
            hits = self._terms_in_text(sentence, query_terms)
            if hits:
                scored.append((len(set(hits)), -index, sentence))
        if scored:
            sentence = sorted(scored, reverse=True)[0][2]
            return {
                "text": self._trim_snippet(sentence, window),
                "source": "content_sentence",
            }

        sentence = sentences[0] if sentences else content
        return {"text": self._trim_snippet(sentence, window), "source": "fallback_lead"}

    def _trim_snippet(self, text: str, window: int) -> str:
        """Trim a snippet to a stable display length.

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        text = (text or "").strip()
        return text[:window] + ("..." if len(text) > window else "")

    def _terms_in_text(self, text: str, terms: list[str]) -> list[str]:
        """Return normalized terms present in text.

        Complexity:
            Time: O(q * n)
            Space: O(q)
        """
        normalized = self.index.normalize_text(text)
        return [
            term
            for term in terms
            if term and self.index.normalize_text(term) in normalized
        ]

    def _highlight(self, text: str, query_terms: list[str]) -> str:
        """HTML-escape text and highlight matched query terms.

        Complexity:
            Time: O(n * q)
            Space: O(n)
        """
        escaped = html.escape(text or "")
        for term in sorted(set(query_terms), key=len, reverse=True):
            if not term:
                continue
            pattern = re.compile(re.escape(html.escape(term)), re.IGNORECASE)
            escaped = pattern.sub(
                lambda match: f"<mark>{match.group(0)}</mark>", escaped
            )
        return escaped

    def _matched_terms(self, doc: dict[str, Any], query_terms: list[str]) -> list[str]:
        """Return query terms that matched the document.

        Complexity:
            Time: O(q)
            Space: O(q)
        """
        doc_key = str(doc["doc_id"])
        freqs = self.index.doc_term_freqs.get(doc_key, {})
        direct = [term for term in query_terms if freqs.get(term, 0) > 0]
        if direct:
            return direct
        return [
            term
            for term in query_terms
            if self._terms_in_text(self.index._document_text(doc), [term])
        ]

    def _field_matches(
        self, doc: dict[str, Any], query_terms: list[str]
    ) -> dict[str, list[str]]:
        """Return matched terms by visible field.

        Complexity:
            Time: O(q * f)
            Space: O(q)
        """
        fields = {
            "title": doc.get("title") or "",
            "content": doc.get("content") or "",
            "category": doc.get("category") or "",
            "tags": " ".join(doc.get("tags") or []),
        }
        matches: dict[str, list[str]] = {}
        for field, value in fields.items():
            normalized = self.index.normalize_text(value)
            hits = [
                term
                for term in query_terms
                if term and self.index.normalize_text(term) in normalized
            ]
            if hits:
                matches[field] = hits
        return matches

    def suggestions(
        self, query: str, query_terms: list[str] | None = None
    ) -> list[dict[str, Any]]:
        """Return fallback suggestions for no-result searches.

        Complexity:
            Time: O(q * V)
            Space: O(k)
        """
        terms = query_terms or self.index.tokenize(query)
        suggestions: list[dict[str, Any]] = []
        synonyms = self.index.text_quality.synonym_terms(terms)
        if synonyms:
            suggestions.append(
                {
                    "type": "synonym",
                    "query": " ".join(list(dict.fromkeys(terms + synonyms))),
                    "terms": synonyms,
                }
            )
        fuzzy_terms = self._fuzzy_expansion(terms)
        if fuzzy_terms and fuzzy_terms != terms:
            suggestions.append(
                {
                    "type": "fuzzy",
                    "query": " ".join(fuzzy_terms),
                    "terms": fuzzy_terms,
                }
            )
        csoundex_terms = self._csoundex_expansion(terms)
        if csoundex_terms and csoundex_terms != terms:
            suggestions.append(
                {
                    "type": "csoundex",
                    "query": " ".join(csoundex_terms),
                    "terms": csoundex_terms,
                }
            )
        try:
            expansion = self.expand_query(query)
            expanded_terms = expansion.get("expanded_terms") or []
            if expanded_terms:
                suggestions.append(
                    {
                        "type": "rocchio",
                        "query": expansion.get("expanded_query", query),
                        "terms": expanded_terms,
                    }
                )
        except Exception:
            pass
        return suggestions[:5]

    def related_documents(
        self,
        doc: dict[str, Any],
        top_k: int = 5,
        model: str = "hybrid",
    ) -> list[dict[str, Any]]:
        """Return explainable lexical related documents.

        Complexity:
            Time: O(q * p + r log r)
            Space: O(r)
        """
        top_k = max(1, min(int(top_k or 5), 20))
        source_doc_key = str(doc.get("doc_id"))
        query = self._related_query(doc)
        if not query:
            return []

        model = (model or "hybrid").lower()
        search_top_k = min(len(self.documents_by_id), max(50, top_k * 12))
        if model == "tfidf":
            ranked = self._search_tfidf(query, search_top_k)
            component_scores: dict[str, dict[str, float]] = {}
            model = "tfidf"
        elif model == "bm25":
            ranked = self._search_bm25(query, search_top_k)
            component_scores = {}
        else:
            ranked, component_scores = self._search_hybrid(query, search_top_k)
            model = "hybrid"

        query_terms = self.index.tokenize(query)
        quality_terms = self.index.text_quality.matching_terms(query_terms, query)
        base_scores = {str(doc_key): float(score) for doc_key, score in ranked}
        ranked, field_boosts = self._rerank_with_field_boost(ranked, quality_terms)

        source_profile = self._relation_profile(doc)
        boosted: list[tuple[str, float, float]] = []
        for doc_key, score in ranked:
            if str(doc_key) == source_doc_key:
                continue
            candidate = self.documents_by_id.get(str(doc_key))
            if not candidate:
                continue
            relation_boost = self._relation_boost(source_profile, candidate)
            boosted.append(
                (str(doc_key), float(score) + relation_boost, relation_boost)
            )
        boosted.sort(key=lambda item: item[1], reverse=True)
        selected = boosted[:top_k]
        max_score = max((score for _, score, _ in selected), default=1.0) or 1.0

        results: list[dict[str, Any]] = []
        for rank, (doc_key, score, relation_boost) in enumerate(selected, 1):
            candidate = self.documents_by_id.get(doc_key)
            if not candidate:
                continue
            doc_components = dict(component_scores.get(doc_key, {}))
            doc_components.setdefault(model, base_scores.get(doc_key, score))
            doc_components["field_boost"] = float(
                field_boosts.get(doc_key, {}).get("boost", 0.0)
            )
            doc_components["relation_boost"] = float(relation_boost)
            result = self._make_result(
                candidate,
                query,
                quality_terms or query_terms,
                float(score),
                model,
                rank,
                component_scores=doc_components,
                field_boost=field_boosts.get(doc_key, {}),
            ).to_dict()
            result["similarity"] = min(1.0, max(0.0, float(score) / max_score))
            result["relation_reason"] = self._relation_reason(source_profile, candidate)
            result["explanation"]["relation_reason"] = result["relation_reason"]
            results.append(result)
        return results

    def _related_query(self, doc: dict[str, Any]) -> str:
        """Build a related-document query from title and salient terms.

        Complexity:
            Time: O(t log t)
            Space: O(k)
        """
        title = doc.get("title") or ""
        keywords = [
            item["word"]
            for item in self.extract_keywords(doc, "tfidf", 8)
            if item.get("word")
        ]
        tags = doc.get("tags") or []
        return " ".join(str(part) for part in [title, *keywords, *tags] if part)

    def _relation_profile(self, doc: dict[str, Any]) -> dict[str, Any]:
        """Return metadata used for related-document boosts.

        Complexity:
            Time: O(t)
            Space: O(t)
        """
        return {
            "source": doc.get("source"),
            "category": doc.get("category"),
            "taxonomy_topic": doc.get("taxonomy_topic"),
            "content_type": doc.get("content_type"),
            "tags": set(doc.get("tags") or []),
        }

    def _relation_boost(
        self, source_profile: dict[str, Any], candidate: dict[str, Any]
    ) -> float:
        """Compute metadata boost for related documents.

        Complexity:
            Time: O(t)
            Space: O(1)
        """
        boost = 0.0
        if source_profile.get("taxonomy_topic") and source_profile[
            "taxonomy_topic"
        ] == candidate.get("taxonomy_topic"):
            boost += 0.25
        if source_profile.get("category") and source_profile[
            "category"
        ] == candidate.get("category"):
            boost += 0.18
        if source_profile.get("source") and source_profile["source"] == candidate.get(
            "source"
        ):
            boost += 0.08
        if source_profile.get("content_type") and source_profile[
            "content_type"
        ] == candidate.get("content_type"):
            boost += 0.05
        tag_overlap = source_profile.get("tags", set()).intersection(
            set(candidate.get("tags") or [])
        )
        boost += min(len(tag_overlap) * 0.06, 0.18)
        return round(boost, 6)

    def _relation_reason(
        self, source_profile: dict[str, Any], candidate: dict[str, Any]
    ) -> dict[str, Any]:
        """Return related-document explanation fields.

        Complexity:
            Time: O(t)
            Space: O(t)
        """
        shared_tags = sorted(
            source_profile.get("tags", set()).intersection(
                set(candidate.get("tags") or [])
            )
        )
        return {
            "method": "hybrid_lexical",
            "same_taxonomy_topic": bool(
                source_profile.get("taxonomy_topic")
                and source_profile["taxonomy_topic"] == candidate.get("taxonomy_topic")
            ),
            "same_category": bool(
                source_profile.get("category")
                and source_profile["category"] == candidate.get("category")
            ),
            "same_source": bool(
                source_profile.get("source")
                and source_profile["source"] == candidate.get("source")
            ),
            "shared_tags": shared_tags,
        }

    def summarize(self, doc: dict[str, Any], method: str = "lead_k", k: int = 3) -> str:
        """Generate a lightweight extractive summary.

        Complexity:
            Time: O(n)
            Space: O(n)
        """
        content = doc.get("content") or ""
        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[。！？.!?])", content)
            if sentence.strip()
        ]
        if not sentences:
            return content[:300]
        k = max(1, min(int(k or 3), 10))
        if method == "key_sentence":
            term_counts = Counter(self.index.tokenize(content))
            scored = []
            for idx, sentence in enumerate(sentences):
                score = sum(
                    term_counts.get(term, 0) for term in self.index.tokenize(sentence)
                )
                scored.append((score, -idx, sentence))
            selected = [sentence for _, _, sentence in sorted(scored, reverse=True)[:k]]
            return "\n".join(selected)
        return "\n".join(sentences[:k])

    def extract_keywords(
        self, doc: dict[str, Any], method: str = "tfidf", top_k: int = 10
    ) -> list[dict[str, Any]]:
        """Extract lightweight keywords from one document.

        Complexity:
            Time: O(t log t)
            Space: O(t)
        """
        doc_key = str(doc["doc_id"])
        freqs = self.index.doc_term_freqs.get(doc_key, Counter())
        top_k = max(1, min(int(top_k or 10), 50))
        scored = []
        for term, freq in freqs.items():
            if len(term) <= 1:
                continue
            if method == "term_frequency":
                score = float(freq)
            else:
                score = float(freq) * self.index.idf.get(term, 0.0)
            scored.append((term, score, freq))
        scored.sort(key=lambda item: item[1], reverse=True)
        return [
            {"word": term, "score": score, "frequency": freq}
            for term, score, freq in scored[:top_k]
        ]

    def candidate_doc_ids(
        self,
        query: str,
        model: str = "bm25",
        operator: str = "AND",
        filters: dict[str, list[str]] | None = None,
    ) -> list[str]:
        """Return candidate document IDs for query-level facet counts.

        Complexity:
            Time: O(q * p + r)
            Space: O(r)
        """
        query = (query or "").strip()
        if not query:
            doc_ids = set(self.documents_by_id.keys())
        else:
            model = (model or "bm25").lower()
            query_terms = self.index.tokenize(query)
            if model == "boolean":
                ranked = self._search_boolean(query, query_terms, operator)
            elif model in {"tfidf", "vsm"}:
                ranked = self._search_tfidf(query, len(self.documents_by_id))
            elif model == "hybrid":
                ranked, _ = self._search_hybrid(query, len(self.documents_by_id))
            elif model == "lm":
                ranked = self._search_lm(query, len(self.documents_by_id))
            else:
                ranked = self._search_bm25(query, len(self.documents_by_id))
            doc_ids = {str(doc_id) for doc_id, _ in ranked}

        if filters:
            doc_ids = doc_ids.intersection(self.facet_service.matching_doc_ids(filters))
        return list(doc_ids)

    def facets(
        self,
        doc_ids: list[str] | set[str] | None = None,
        selected_filters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build indexed facet counts for all documents or a result set.

        Complexity:
            Time: O(f * v)
            Space: O(v)
        """
        return self.facet_service.build_facets(doc_ids, selected_filters)

    def corpus_distribution(self) -> dict[str, Any]:
        """Return corpus-level source/topic/content-type distributions.

        Complexity:
            Time: O(n)
            Space: O(v)
        """
        return self.facet_service.corpus_distribution()
