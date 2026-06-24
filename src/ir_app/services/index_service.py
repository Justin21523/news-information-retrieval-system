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
from src.ir.ranking.hybrid import HybridRanker
from src.ir.ranking.rocchio import RocchioExpander
from src.ir.retrieval.bm25 import BM25Ranker
from src.ir.retrieval.boolean import BooleanQueryEngine
from src.ir.retrieval.fuzzy import FuzzyMatcher
from src.ir.retrieval.vsm import VectorSpaceModel
from src.ir.text.chinese_tokenizer import ChineseTokenizer
from src.ir.text.csoundex import CSoundex


INDEX_VERSION = "ir_app_v2"


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

        self.doc_terms: dict[str, list[str]] = {}
        self.doc_term_freqs: dict[str, Counter[str]] = {}
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
        metadata = self.documents
        weighted_docs = self.document_texts()

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

        self.positional_index = PositionalIndex(tokenizer=self.tokenize)
        self.positional_index.build(weighted_docs, metadata)

        self.field_indexer = FieldIndexer(tokenizer=self.tokenize)
        self.field_indexer.build(metadata)

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

        self.vsm = VectorSpaceModel(inverted_index=self.formal_inverted_index)
        self.vsm.term_weighting.build_from_index(self.formal_inverted_index)
        self.vsm._compute_document_vectors()

        self.hybrid = HybridRanker(
            {"bm25": self.bm25, "tfidf": self.vsm},
            fusion_method="rrf",
            weights={"bm25": 0.65, "tfidf": 0.35},
            normalization="minmax",
        )
        self.rocchio = RocchioExpander(max_expansion_terms=8, min_term_weight=0.01)
        self.fuzzy = FuzzyMatcher(max_distance=1, max_expansions=20)
        self.csoundex = CSoundex()

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
