"""Lightweight clustering and topic exploration for the IR app."""

from __future__ import annotations

import time
from typing import Any

from src.ir.cluster.doc_cluster import DocumentClusterer
from src.ir_app.services.document_service import DocumentService
from src.ir_app.services.search_service import SearchService


class ClusterTopicService:
    """Cluster query results or a facet-filtered corpus sample.

    Complexity:
        Time: O(k * n * i * d) for K-means or O(n^3) for HAC
        Space: O(n * d)
    """

    def __init__(self, document_service: DocumentService, search_service: SearchService):
        self.document_service = document_service
        self.search_service = search_service
        self.clusterer = DocumentClusterer()

    def cluster(self, payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Build topic-like clusters from the current corpus.

        Complexity:
            Time: O(search + clustering)
            Space: O(n * d)
        """
        started = time.perf_counter()
        query = str(payload.get("query") or "").strip()
        method = str(payload.get("method") or "kmeans").lower()
        if method == "hac":
            method = "hierarchical"
        if method not in {"kmeans", "hierarchical"}:
            raise ValueError(f"Unknown clustering method: {method}")

        sample_limit = self._sample_limit(payload.get("sample_size"), method)
        n_clusters = max(2, min(int(payload.get("n_clusters") or 5), 12))
        filters = payload.get("filters") or None
        model = payload.get("model") or "bm25"

        doc_ids = self._sample_doc_ids(query, model, filters, sample_limit)
        vectors = {
            int(doc_id): self.search_service.index.tfidf_vectors.get(str(doc_id), {})
            for doc_id in doc_ids
        }
        vectors = {doc_id: vector for doc_id, vector in vectors.items() if vector}
        if len(vectors) < 2:
            data = self._empty_response(query, method, n_clusters, sample_limit)
            return data, {"execution_time": time.perf_counter() - started}

        n_clusters = min(n_clusters, len(vectors))
        if method == "hierarchical":
            result = self.clusterer.hierarchical_clustering(
                vectors,
                k=n_clusters,
                linkage=str(payload.get("linkage") or "complete"),
                similarity_metric="cosine",
            )
        else:
            result = self.clusterer.kmeans_clustering(
                vectors,
                k=n_clusters,
                max_iterations=min(int(payload.get("max_iterations") or 60), 100),
                random_seed=42,
            )

        quality_score = None
        if len(vectors) <= 120:
            quality_score = self.clusterer.evaluate_clusters(vectors, result)

        clusters = [
            self._cluster_payload(cluster, vectors)
            for cluster in sorted(result.clusters, key=lambda item: item.cluster_id)
        ]
        data = {
            "query": query,
            "method": method,
            "n_clusters": result.num_clusters,
            "sample_size": len(vectors),
            "sample_limit": sample_limit,
            "quality_score": quality_score,
            "clusters": clusters,
            "topics": self._topics_from_clusters(clusters),
            "parameters": {
                "model": model,
                "filters": filters or {},
                "linkage": payload.get("linkage") or "complete",
            },
        }
        return data, {"execution_time": time.perf_counter() - started}

    def _sample_limit(self, raw_limit: Any, method: str) -> int:
        """Return a safe sample limit for interactive clustering.

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        requested = int(raw_limit or (80 if method == "hierarchical" else 120))
        cap = 80 if method == "hierarchical" else 200
        return max(10, min(requested, cap))

    def _sample_doc_ids(
        self,
        query: str,
        model: str,
        filters: dict[str, list[str]] | None,
        sample_limit: int,
    ) -> list[str]:
        """Pick representative document IDs for clustering.

        Complexity:
            Time: O(search)
            Space: O(n)
        """
        if query:
            results, _ = self.search_service.search(query, model, sample_limit, "AND", filters)
            doc_ids = [str(item["doc_id"]) for item in results]
            if doc_ids:
                return doc_ids

        allowed = self.search_service.facet_service.matching_doc_ids(filters)
        return sorted(allowed, key=lambda value: int(value))[:sample_limit]

    def _cluster_payload(
        self, cluster: Any, vectors: dict[int, dict[str, float]]
    ) -> dict[str, Any]:
        """Convert one cluster to API data.

        Complexity:
            Time: O(c * d)
            Space: O(d)
        """
        centroid = cluster.centroid or self.clusterer._compute_centroid(
            [vectors[doc_id] for doc_id in cluster.doc_ids if doc_id in vectors]
        )
        centroid_terms = sorted(centroid.items(), key=lambda item: item[1], reverse=True)[:12]
        representatives = self._representatives(cluster.doc_ids, centroid, vectors)
        return {
            "cluster_id": cluster.cluster_id,
            "size": cluster.size,
            "keywords": [
                {"term": term, "weight": float(weight)}
                for term, weight in centroid_terms[:8]
            ],
            "centroid_terms": [
                {"term": term, "weight": float(weight)} for term, weight in centroid_terms
            ],
            "documents": representatives,
        }

    def _representatives(
        self,
        doc_ids: list[int],
        centroid: dict[str, float],
        vectors: dict[int, dict[str, float]],
        limit: int = 6,
    ) -> list[dict[str, Any]]:
        """Return representative documents nearest to a centroid.

        Complexity:
            Time: O(c * d)
            Space: O(c)
        """
        scored = []
        for doc_id in doc_ids:
            vector = vectors.get(doc_id, {})
            similarity = self.clusterer.cosine_similarity(vector, centroid)
            scored.append((similarity, doc_id))
        scored.sort(reverse=True)

        documents = []
        for similarity, doc_id in scored[:limit]:
            doc = self.document_service.get_document(doc_id)
            if not doc:
                continue
            documents.append(
                {
                    "doc_id": doc.get("doc_id"),
                    "title": doc.get("title"),
                    "source": doc.get("source"),
                    "source_label": doc.get("source_label"),
                    "published_date": doc.get("published_date"),
                    "taxonomy_label": doc.get("taxonomy_label"),
                    "similarity": similarity,
                }
            )
        return documents

    def _topics_from_clusters(self, clusters: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return topic-card aliases for clusters.

        Complexity:
            Time: O(k)
            Space: O(k)
        """
        topics = []
        for cluster in clusters:
            keywords = cluster.get("keywords", [])
            label = " / ".join(item["term"] for item in keywords[:3]) or f"Topic {cluster['cluster_id']}"
            topics.append(
                {
                    "topic_id": cluster["cluster_id"],
                    "label": label,
                    "size": cluster["size"],
                    "keywords": keywords,
                    "representative_documents": cluster.get("documents", []),
                }
            )
        return topics

    def _empty_response(
        self, query: str, method: str, n_clusters: int, sample_limit: int
    ) -> dict[str, Any]:
        """Return a stable empty clustering payload.

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        return {
            "query": query,
            "method": method,
            "n_clusters": n_clusters,
            "sample_size": 0,
            "sample_limit": sample_limit,
            "quality_score": None,
            "clusters": [],
            "topics": [],
            "parameters": {},
            "message": "Not enough vectorized documents for clustering.",
        }
