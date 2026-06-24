"""Small in-memory index for Flask startup fallback."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Any

from src.ir.text.chinese_tokenizer import ChineseTokenizer


class IndexService:
    """Build lightweight lexical indexes for the web demo.

    Complexity:
        Time: O(T) where T is total token count
        Space: O(T)
    """

    def __init__(self, documents: list[dict[str, Any]], tokenizer_engine: str = "jieba"):
        self.documents = documents
        self.tokenizer = ChineseTokenizer(engine=tokenizer_engine)
        self.doc_terms: dict[str, list[str]] = {}
        self.doc_term_freqs: dict[str, Counter[str]] = {}
        self.inverted_index: dict[str, dict[str, int]] = defaultdict(dict)
        self.idf: dict[str, float] = {}
        self.doc_lengths: dict[str, int] = {}
        self.avg_doc_length = 1.0
        self.tfidf_vectors: dict[str, dict[str, float]] = {}
        self._build()

    def normalize_text(self, text: str) -> str:
        """Normalize Chinese/news text before tokenization.

        Complexity:
            Time: O(n)
            Space: O(n)
        """
        text = unicodedata.normalize("NFKC", text or "")
        text = text.replace("臺", "台")
        text = text.lower()
        return re.sub(r"\s+", " ", text).strip()

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text and remove punctuation-only tokens.

        Complexity:
            Time: O(n)
            Space: O(k)
        """
        normalized = self.normalize_text(text)
        tokens = self.tokenizer.tokenize(normalized)
        cleaned: list[str] = []
        for token in tokens:
            token = token.strip().lower()
            if not token:
                continue
            if not re.search(r"[\w\u4e00-\u9fff]", token):
                continue
            cleaned.append(token)
        return cleaned

    def _document_text(self, doc: dict[str, Any]) -> str:
        """Build field-weighted text for lexical scoring.

        Complexity:
            Time: O(t)
            Space: O(t)
        """
        tags = " ".join(doc.get("tags") or [])
        title = doc.get("title") or ""
        content = doc.get("content") or ""
        category = doc.get("category_name") or doc.get("category") or ""
        return f"{title} {title} {title} {tags} {category} {content}"

    def _build(self) -> None:
        """Build inverted index, BM25 stats, and TF-IDF vectors.

        Complexity:
            Time: O(T)
            Space: O(T)
        """
        for doc in self.documents:
            doc_key = str(doc["doc_id"])
            terms = self.tokenize(self._document_text(doc))
            freqs = Counter(terms)
            self.doc_terms[doc_key] = terms
            self.doc_term_freqs[doc_key] = freqs
            self.doc_lengths[doc_key] = max(1, len(terms))
            for term, freq in freqs.items():
                self.inverted_index[term][doc_key] = freq

        total_docs = max(1, len(self.documents))
        total_length = sum(self.doc_lengths.values())
        self.avg_doc_length = total_length / total_docs if total_docs else 1.0

        for term, postings in self.inverted_index.items():
            df = len(postings)
            self.idf[term] = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))

        for doc_key, freqs in self.doc_term_freqs.items():
            vector = {
                term: (1.0 + math.log10(freq)) * self.idf.get(term, 0.0)
                for term, freq in freqs.items()
                if freq > 0
            }
            norm = math.sqrt(sum(weight * weight for weight in vector.values()))
            if norm > 0:
                vector = {term: weight / norm for term, weight in vector.items()}
            self.tfidf_vectors[doc_key] = vector

    def vocabulary_size(self) -> int:
        """Return vocabulary size.

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        return len(self.inverted_index)
