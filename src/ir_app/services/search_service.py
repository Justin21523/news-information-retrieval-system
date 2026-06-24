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
                "models": ["boolean", "tfidf", "bm25", "hybrid", "fuzzy", "csoundex"],
                "optional_models": ["bert"],
                "tokenizer_engine": self.settings.tokenizer_engine,
                "heavy_models_enabled": self.settings.enable_heavy_models,
                "index": self.index.stats(),
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
            return [], {"execution_time": time.perf_counter() - started, "query_terms": []}

        model = (model or "bm25").lower()
        top_k = max(1, min(int(top_k or 20), 100))
        query_terms = self.index.tokenize(query)

        expanded_terms: list[str] = []
        component_scores: dict[str, dict[str, float]] = {}

        if model == "boolean":
            ranked = self._search_boolean(query, query_terms, operator)
        elif model in {"tfidf", "vsm"}:
            ranked = self._search_tfidf(query, top_k)
            model = "tfidf"
        elif model == "bm25":
            ranked = self._search_bm25(query, top_k)
        elif model == "hybrid":
            ranked, component_scores = self._search_hybrid(query, top_k)
        elif model == "fuzzy":
            expanded_terms = self._fuzzy_expansion(query_terms)
            ranked = self._search_bm25(" ".join(expanded_terms or query_terms), top_k)
        elif model == "csoundex":
            expanded_terms = self._csoundex_expansion(query_terms)
            ranked = self._search_bm25(" ".join(expanded_terms or query_terms), top_k)
        elif model == "bert":
            raise FeatureUnavailableError(
                "BERT semantic search is disabled or unavailable in lightweight startup mode."
            )
        else:
            raise ValueError(f"Unknown retrieval model: {model}")

        ranked = self._apply_filters(ranked, filters)
        ranked = ranked[:top_k]

        results: list[dict[str, Any]] = []
        for rank, (doc_key, score) in enumerate(ranked, 1):
            doc = self.documents_by_id.get(str(doc_key))
            if not doc:
                continue
            results.append(
                self._make_result(
                    doc,
                    query,
                    query_terms,
                    float(score),
                    model,
                    rank,
                    expanded_terms=expanded_terms,
                    component_scores=component_scores.get(str(doc_key), {}),
                ).to_dict()
            )

        meta = {
            "execution_time": time.perf_counter() - started,
            "query_terms": query_terms,
            "expanded_terms": expanded_terms,
            "dataset_source": self.document_service.dataset_source,
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
        if not re.search(r"\b(AND|OR|NOT)\b|:|\"|\(|\)", query, re.IGNORECASE):
            joiner = f" {(operator or 'AND').upper()} "
            boolean_query = joiner.join(query_terms)
        result = self.index.boolean_engine.query(boolean_query, rank_results=True)
        if result.scores:
            return [(str(doc_id), float(result.scores.get(doc_id, 1.0))) for doc_id in result.doc_ids]
        return [(str(doc_id), 1.0) for doc_id in result.doc_ids]

    def _search_tfidf(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Search with the formal Vector Space Model.

        Complexity:
            Time: O(q + c log k)
            Space: O(r)
        """
        result = self.index.vsm.search(query, topk=top_k)
        return [(str(doc_id), float(result.scores.get(doc_id, 0.0))) for doc_id in result.doc_ids]

    def _search_bm25(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Search with the formal BM25 ranker.

        Complexity:
            Time: O(q * p)
            Space: O(r)
        """
        result = self.index.bm25.search(query, topk=top_k)
        return [(str(doc_id), float(score)) for doc_id, score in zip(result.doc_ids, result.scores)]

    def _search_hybrid(self, query: str, top_k: int) -> tuple[list[tuple[str, float]], dict[str, dict[str, float]]]:
        """Search with formal reciprocal-rank fusion over BM25 and TF-IDF.

        Complexity:
            Time: O(r * n)
            Space: O(r * n)
        """
        result = self.index.hybrid.search(query, topk=top_k, ranker_topk=max(50, top_k * 5))
        ranked = [(str(doc_id), float(score)) for doc_id, score in zip(result.doc_ids, result.scores)]
        component_scores: dict[str, dict[str, float]] = {}
        for ranker_name, scores in result.component_scores.items():
            for doc_id, score in zip(result.doc_ids, scores):
                component_scores.setdefault(str(doc_id), {})[ranker_name] = float(score)
        return ranked, component_scores

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
            matches = self.index.csoundex.find_similar(term, vocabulary, threshold=0.72, topk=8)
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

        normalized_filters = {
            key: {str(value) for value in values if value is not None}
            for key, values in filters.items()
            if values
        }
        if not normalized_filters:
            return ranked

        filtered: list[tuple[str, float]] = []
        for doc_key, score in ranked:
            doc = self.documents_by_id.get(str(doc_key))
            if doc and self._document_matches_filters(doc, normalized_filters):
                filtered.append((doc_key, score))
        return filtered

    def _document_matches_filters(self, doc: dict[str, Any], filters: dict[str, set[str]]) -> bool:
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
    ) -> SearchResult:
        """Build a normalized search result.

        Complexity:
            Time: O(n * q)
            Space: O(n)
        """
        snippet = self._snippet(doc, query_terms)
        highlighted = self._highlight(snippet, query_terms)
        matched_terms = self._matched_terms(doc, query_terms)
        doc_id = int(doc.get("doc_id"))
        bm25_explain = self.index.bm25.explain_score(query, doc_id)
        component_scores = component_scores or {model: score}
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
            author=doc.get("author"),
            tags=doc.get("tags") or [],
            rank=rank,
            explanation={
                "matched_terms": matched_terms,
                "expanded_terms": expanded_terms or [],
                "field_matches": self._field_matches(doc, query_terms),
                "component_scores": component_scores,
                "ranking_features": {
                    "bm25": bm25_explain if "error" not in bm25_explain else {},
                    "index_cache_used": self.index.cache_used,
                },
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
            self.index.vsm.get_document_vector(int(doc_id))
            for doc_id, _ in ranked
        ]
        doc_scores = [score for _, score in ranked]
        query_vector = {
            term: self.index.idf.get(term, 1.0)
            for term in query_terms
        }
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

    def _snippet(self, doc: dict[str, Any], query_terms: list[str], window: int = 180) -> str:
        """Generate a query-centered snippet.

        Complexity:
            Time: O(n * q)
            Space: O(n)
        """
        content = doc.get("content") or ""
        if not content:
            return doc.get("title") or ""

        normalized_content = self.index.normalize_text(content)
        positions = [
            normalized_content.find(self.index.normalize_text(term))
            for term in query_terms
            if term
        ]
        positions = [position for position in positions if position >= 0]
        if not positions:
            sentence = re.split(r"[。！？.!?]", content)[0].strip()
            return sentence[:window] + ("..." if len(sentence) > window else "")

        center = min(positions)
        start = max(0, center - window // 3)
        end = min(len(content), start + window)
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(content) else ""
        return f"{prefix}{content[start:end].strip()}{suffix}"

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
            escaped = pattern.sub(lambda match: f"<mark>{match.group(0)}</mark>", escaped)
        return escaped

    def _matched_terms(self, doc: dict[str, Any], query_terms: list[str]) -> list[str]:
        """Return query terms that matched the document.

        Complexity:
            Time: O(q)
            Space: O(q)
        """
        doc_key = str(doc["doc_id"])
        freqs = self.index.doc_term_freqs.get(doc_key, {})
        return [term for term in query_terms if freqs.get(term, 0) > 0]

    def _field_matches(self, doc: dict[str, Any], query_terms: list[str]) -> dict[str, list[str]]:
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
            hits = [term for term in query_terms if term and term in normalized]
            if hits:
                matches[field] = hits
        return matches

    def summarize(self, doc: dict[str, Any], method: str = "lead_k", k: int = 3) -> str:
        """Generate a lightweight extractive summary.

        Complexity:
            Time: O(n)
            Space: O(n)
        """
        content = doc.get("content") or ""
        sentences = [sentence.strip() for sentence in re.split(r"(?<=[。！？.!?])", content) if sentence.strip()]
        if not sentences:
            return content[:300]
        k = max(1, min(int(k or 3), 10))
        if method == "key_sentence":
            term_counts = Counter(self.index.tokenize(content))
            scored = []
            for idx, sentence in enumerate(sentences):
                score = sum(term_counts.get(term, 0) for term in self.index.tokenize(sentence))
                scored.append((score, -idx, sentence))
            selected = [sentence for _, _, sentence in sorted(scored, reverse=True)[:k]]
            return "\n".join(selected)
        return "\n".join(sentences[:k])

    def extract_keywords(self, doc: dict[str, Any], method: str = "tfidf", top_k: int = 10) -> list[dict[str, Any]]:
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

    def facets(self, doc_ids: list[str] | None = None) -> dict[str, Any]:
        """Build facet counts for all documents or a result set.

        Complexity:
            Time: O(n * f)
            Space: O(v)
        """
        docs = self.document_service.documents
        if doc_ids is not None:
            allowed = {str(doc_id) for doc_id in doc_ids}
            docs = [doc for doc in docs if str(doc.get("doc_id")) in allowed]

        fields = {
            "source": "來源 Source",
            "category": "分類 Category",
            "category_name": "分類名稱 Category Name",
            "pub_date": "日期 Date",
            "author": "作者 Author",
        }
        facets: dict[str, Any] = {}
        for field, display_name in fields.items():
            counts: Counter[str] = Counter()
            for doc in docs:
                if field == "pub_date":
                    value = doc.get("published_date")
                else:
                    value = doc.get(field)
                if value:
                    counts[str(value)] += 1
            if counts:
                facets[field] = {
                    "display_name": display_name,
                    "values": [
                        {"value": value, "label": value, "count": count}
                        for value, count in counts.most_common(50)
                    ],
                }
        return facets
