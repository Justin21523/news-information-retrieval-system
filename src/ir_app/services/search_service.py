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
        self.index = IndexService(document_service.documents, settings.tokenizer_engine)
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
                "models": ["boolean", "tfidf", "bm25"],
                "optional_models": ["bert"],
                "tokenizer_engine": self.settings.tokenizer_engine,
                "heavy_models_enabled": self.settings.enable_heavy_models,
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

        if model == "boolean":
            scores = self._boolean_scores(query_terms, operator)
        elif model in {"tfidf", "vsm"}:
            scores = self._tfidf_scores(query_terms)
            model = "tfidf"
        elif model == "bm25":
            scores = self._bm25_scores(query_terms)
        elif model == "bert":
            raise FeatureUnavailableError(
                "BERT semantic search is disabled or unavailable in lightweight startup mode."
            )
        else:
            raise ValueError(f"Unknown retrieval model: {model}")

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        ranked = self._apply_filters(ranked, filters)
        ranked = ranked[:top_k]

        results: list[dict[str, Any]] = []
        for rank, (doc_key, score) in enumerate(ranked, 1):
            doc = self.documents_by_id.get(str(doc_key))
            if not doc:
                continue
            results.append(
                self._make_result(doc, query, query_terms, float(score), model, rank).to_dict()
            )

        meta = {
            "execution_time": time.perf_counter() - started,
            "query_terms": query_terms,
            "dataset_source": self.document_service.dataset_source,
        }
        return results, meta

    def _boolean_scores(self, query_terms: list[str], operator: str) -> dict[str, float]:
        """Return Boolean matching scores.

        Complexity:
            Time: O(q * p)
            Space: O(r)
        """
        if not query_terms:
            return {}
        doc_sets = [
            set(self.index.inverted_index.get(term, {}).keys())
            for term in query_terms
            if term in self.index.inverted_index
        ]
        if not doc_sets:
            return {}
        if (operator or "AND").upper() == "OR":
            matched = set.union(*doc_sets)
        else:
            matched = set.intersection(*doc_sets)
        return {doc_key: 1.0 for doc_key in matched}

    def _tfidf_scores(self, query_terms: list[str]) -> dict[str, float]:
        """Return cosine similarity scores.

        Complexity:
            Time: O(q + c)
            Space: O(q + r)
        """
        if not query_terms:
            return {}
        query_tf = Counter(query_terms)
        query_vector = {
            term: (1.0 + math.log10(freq)) * self.index.idf.get(term, 0.0)
            for term, freq in query_tf.items()
        }
        norm = math.sqrt(sum(weight * weight for weight in query_vector.values()))
        if norm > 0:
            query_vector = {term: weight / norm for term, weight in query_vector.items()}

        candidate_docs = set()
        for term in query_terms:
            candidate_docs.update(self.index.inverted_index.get(term, {}).keys())

        scores: dict[str, float] = {}
        for doc_key in candidate_docs:
            doc_vector = self.index.tfidf_vectors.get(doc_key, {})
            score = sum(query_vector.get(term, 0.0) * doc_vector.get(term, 0.0) for term in query_vector)
            if score > 0:
                scores[doc_key] = score
        return scores

    def _bm25_scores(self, query_terms: list[str]) -> dict[str, float]:
        """Return BM25 scores.

        Complexity:
            Time: O(q * p)
            Space: O(r)
        """
        scores: dict[str, float] = {}
        k1 = 1.5
        b = 0.75
        avgdl = self.index.avg_doc_length or 1.0
        for term in set(query_terms):
            postings = self.index.inverted_index.get(term, {})
            idf = self.index.idf.get(term, 0.0)
            for doc_key, tf in postings.items():
                doc_len = self.index.doc_lengths.get(doc_key, avgdl)
                denominator = tf + k1 * (1 - b + b * (doc_len / avgdl))
                score = idf * ((tf * (k1 + 1)) / denominator)
                scores[doc_key] = scores.get(doc_key, 0.0) + score
        return scores

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
    ) -> SearchResult:
        """Build a normalized search result.

        Complexity:
            Time: O(n * q)
            Space: O(n)
        """
        snippet = self._snippet(doc, query_terms)
        highlighted = self._highlight(snippet, query_terms)
        matched_terms = self._matched_terms(doc, query_terms)
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
                "expanded_terms": [],
                "field_matches": self._field_matches(doc, query_terms),
                "component_scores": {model: score},
                "query": query,
            },
        )

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
