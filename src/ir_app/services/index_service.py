"""Index construction, persistence, and IR module adapters."""

from __future__ import annotations

import json
import math
import pickle
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.ir.index.field_indexer import FieldIndexer
from src.ir.index.inverted_index import InvertedIndex
from src.ir.index.positional_index import PositionalIndex
from src.ir.ranking.rocchio import RocchioExpander
from src.ir.retrieval.bm25 import BM25Ranker
from src.ir.retrieval.boolean import BooleanQueryEngine
from src.ir.retrieval.fuzzy import FuzzyMatcher
from src.ir.retrieval.language_model_retrieval import LanguageModelRetrieval
from src.ir.text.chinese_tokenizer import ChineseTokenizer
from src.ir.text.csoundex import CSoundex
from src.ir_app.services.text_quality import TextQualityService


INDEX_VERSION = "ir_app_v3"


class IndexService:
    """Build, cache, and expose retrieval indexes for the web app.

    Complexity:
        Time: O(T) on cache miss, O(V + P) on cache hit
        Space: O(T)
    """

    def __init__(
        self,
        documents: list[dict[str, Any]],
        tokenizer_engine: str = "jieba",
        index_dir: Path | None = None,
        dataset_metadata: dict[str, Any] | None = None,
    ):
        self.documents = documents
        self.tokenizer_engine = tokenizer_engine
        self.index_dir = Path(index_dir) if index_dir else None
        self.dataset_metadata = dataset_metadata or {}
        self.tokenizer = ChineseTokenizer(engine=tokenizer_engine)
        self.text_quality = TextQualityService(self.tokenize, self.normalize_text)

        self.doc_terms: dict[str, list[str]] = {}
        self.doc_term_freqs: dict[str, Counter[str]] = {}
        self.field_term_freqs: dict[str, dict[str, Counter[str]]] = {}
        self.inverted_index: dict[str, dict[str, int]] = defaultdict(dict)
        self.idf: dict[str, float] = {}
        self.doc_lengths: dict[str, int] = {}
        self.avg_doc_length = 1.0
        self.tfidf_vectors: dict[str, dict[str, float]] = {}
        self.cache_used = False
        self.manifest = self._build_manifest()

        if not self._load_cache():
            self._build_lexical_cache()
            self._save_cache()

        self._build_ir_adapters()

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

    def _build_manifest(self) -> dict[str, Any]:
        """Build the manifest used to validate index cache freshness.

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        return {
            "index_version": INDEX_VERSION,
            "dataset_source": self.dataset_metadata.get("dataset_source"),
            "dataset_hash": self.dataset_metadata.get("dataset_hash"),
            "dataset_mtime": self.dataset_metadata.get("dataset_mtime"),
            "dataset_size": self.dataset_metadata.get("dataset_size"),
            "tokenizer_engine": self.tokenizer_engine,
            "document_count": len(self.documents),
        }

    def _cache_dir(self) -> Path | None:
        """Return cache directory if persistence is configured.

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        if not self.index_dir:
            return None
        return self.index_dir / "ir_app_cache"

    def _manifest_path(self) -> Path | None:
        cache_dir = self._cache_dir()
        return cache_dir / "manifest.json" if cache_dir else None

    def _cache_path(self) -> Path | None:
        cache_dir = self._cache_dir()
        return cache_dir / "lexical_index.pkl" if cache_dir else None

    def _load_cache(self) -> bool:
        """Load lexical index cache when the manifest matches.

        Complexity:
            Time: O(T)
            Space: O(T)
        """
        manifest_path = self._manifest_path()
        cache_path = self._cache_path()
        if not manifest_path or not cache_path:
            return False
        if not manifest_path.exists() or not cache_path.exists():
            return False

        try:
            cached_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if cached_manifest != self.manifest:
                return False
            with cache_path.open("rb") as handle:
                payload = pickle.load(handle)
        except Exception:
            return False

        self.doc_terms = payload["doc_terms"]
        self.doc_term_freqs = {
            key: Counter(value) for key, value in payload["doc_term_freqs"].items()
        }
        self.inverted_index = defaultdict(dict, payload["inverted_index"])
        self.idf = payload["idf"]
        self.doc_lengths = payload["doc_lengths"]
        self.avg_doc_length = payload["avg_doc_length"]
        self.tfidf_vectors = payload["tfidf_vectors"]
        self.field_term_freqs = {
            doc_key: {
                field: Counter(freqs)
                for field, freqs in fields.items()
            }
            for doc_key, fields in payload.get("field_term_freqs", {}).items()
        }
        self.cache_used = True
        return True

    def _save_cache(self) -> None:
        """Persist the lightweight lexical index cache.

        Complexity:
            Time: O(T)
            Space: O(T)
        """
        manifest_path = self._manifest_path()
        cache_path = self._cache_path()
        if not manifest_path or not cache_path:
            return

        try:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(
                json.dumps(self.manifest, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            payload = {
                "doc_terms": self.doc_terms,
                "doc_term_freqs": {key: dict(value) for key, value in self.doc_term_freqs.items()},
                "inverted_index": dict(self.inverted_index),
                "idf": self.idf,
                "doc_lengths": self.doc_lengths,
                "avg_doc_length": self.avg_doc_length,
                "tfidf_vectors": self.tfidf_vectors,
                "field_term_freqs": {
                    doc_key: {
                        field: dict(freqs)
                        for field, freqs in fields.items()
                    }
                    for doc_key, fields in self.field_term_freqs.items()
                },
            }
            with cache_path.open("wb") as handle:
                pickle.dump(payload, handle)
        except Exception:
            # Cache is an optimization; failure should not block startup.
            return

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

    def _document_fields(self, doc: dict[str, Any]) -> dict[str, str]:
        """Return searchable document fields.

        Complexity:
            Time: O(t)
            Space: O(t)
        """
        tags = " ".join(doc.get("tags") or [])
        category = " ".join(
            str(value)
            for value in (
                doc.get("category_name"),
                doc.get("category"),
                doc.get("taxonomy_label"),
                doc.get("taxonomy_topic"),
            )
            if value
        )
        return {
            "title": doc.get("title") or "",
            "content": doc.get("content") or "",
            "tags": tags,
            "category": category,
        }

    def document_texts(self) -> list[str]:
        """Return field-weighted document texts aligned to doc_id.

        Complexity:
            Time: O(n)
            Space: O(n)
        """
        return [self._document_text(doc) for doc in self.documents]

    def _build_lexical_cache(self) -> None:
        """Build inverted index, BM25 stats, and TF-IDF vectors.

        Complexity:
            Time: O(T)
            Space: O(T)
        """
        for doc in self.documents:
            doc_key = str(doc["doc_id"])
            field_freqs: dict[str, Counter[str]] = {}
            for field, text in self._document_fields(doc).items():
                field_freqs[field] = Counter(self.tokenize(text))
            self.field_term_freqs[doc_key] = field_freqs

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

    def _build_ir_adapters(self) -> None:
        """Initialize formal IR modules from cached lexical stats.

        Complexity:
            Time: O(V + P + T_field)
            Space: O(V + P)
        """
        self.formal_inverted_index = InvertedIndex(tokenizer=self.tokenize)
        self.formal_inverted_index.index = {
            term: sorted((int(doc_id), tf) for doc_id, tf in postings.items())
            for term, postings in self.inverted_index.items()
        }
        self.formal_inverted_index.doc_count = len(self.documents)
        self.formal_inverted_index.doc_lengths = {
            int(doc_id): length for doc_id, length in self.doc_lengths.items()
        }
        self.formal_inverted_index.doc_metadata = {
            int(doc["doc_id"]): doc for doc in self.documents
        }

        self.positional_index: PositionalIndex | None = None
        self.field_indexer: FieldIndexer | None = None
        self._boolean_structural_ready = False

        self.boolean_engine = BooleanQueryEngine(
            inverted_index=self.formal_inverted_index,
            positional_index=self.positional_index,
            field_indexer=self.field_indexer,
        )

        self.bm25 = BM25Ranker(tokenizer=self.tokenize, k1=1.5, b=0.75)
        self.bm25.inverted_index = {
            term: {int(doc_id): tf for doc_id, tf in postings.items()}
            for term, postings in self.inverted_index.items()
        }
        self.bm25.doc_lengths = {int(doc_id): length for doc_id, length in self.doc_lengths.items()}
        self.bm25.doc_count = len(self.documents)
        self.bm25.avg_doc_length = self.avg_doc_length
        self.bm25._compute_idf()

        self.language_model = LanguageModelRetrieval(
            tokenizer=self.tokenize,
            smoothing="dirichlet",
            mu_param=2000.0,
        )
        self._hydrate_language_model()

        self.rocchio = RocchioExpander(max_expansion_terms=8, min_term_weight=0.01)
        self.fuzzy = FuzzyMatcher(max_distance=1, max_expansions=20)
        self.csoundex = CSoundex()

    def _hydrate_language_model(self) -> None:
        """Initialize LM retrieval from cached term-frequency statistics.

        Complexity:
            Time: O(T)
            Space: O(V + D)
        """
        collection_counts: Counter[str] = Counter()
        self.language_model.doc_count = len(self.documents)
        self.language_model.doc_models = {}
        self.language_model.doc_lengths = {}
        self.language_model.vocab = set()

        for doc_key, freqs in self.doc_term_freqs.items():
            doc_id = int(doc_key)
            self.language_model.doc_models[doc_id] = dict(freqs)
            self.language_model.doc_lengths[doc_id] = self.doc_lengths.get(doc_key, 0)
            collection_counts.update(freqs)
            self.language_model.vocab.update(freqs.keys())

        self.language_model.collection_size = sum(collection_counts.values())
        if self.language_model.collection_size <= 0:
            self.language_model.collection_model = {}
            return

        self.language_model.collection_model = {
            term: count / self.language_model.collection_size
            for term, count in collection_counts.items()
        }

    def ensure_boolean_structural_indexes(self) -> None:
        """Build positional and field indexes for advanced Boolean syntax.

        Complexity:
            Time: O(T)
            Space: O(T)
        """
        if self._boolean_structural_ready:
            return

        metadata = self.documents
        weighted_docs = self.document_texts()
        self.positional_index = PositionalIndex(tokenizer=self.tokenize)
        self.positional_index.build(weighted_docs, metadata)

        self.field_indexer = FieldIndexer(tokenizer=self.tokenize)
        self.field_indexer.build(metadata)

        self.boolean_engine.positional_index = self.positional_index
        self.boolean_engine.field_indexer = self.field_indexer
        self._boolean_structural_ready = True

    def vocabulary_size(self) -> int:
        """Return vocabulary size.

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        return len(self.inverted_index)

    def stats(self) -> dict[str, Any]:
        """Return index stats.

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        return {
            "cache_used": self.cache_used,
            "manifest": self.manifest,
            "vocabulary_size": self.vocabulary_size(),
            "avg_doc_length": self.avg_doc_length,
        }
