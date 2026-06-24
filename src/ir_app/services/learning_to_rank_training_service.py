"""Weak-supervision Learning-to-Rank training sandbox."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.ir_app.services.learning_to_rank_feature_service import (
    LearningToRankFeatureService,
)


FEATURE_NAMES = [
    "query_term_count",
    "title_match_count",
    "content_match_count",
    "tags_match_count",
    "category_match_count",
    "field_boost",
    "bm25_score",
    "tfidf_score",
    "lm_score",
]


class LearningToRankTrainingService:
    """Train a demo LTR model from weak feedback labels.

    Complexity:
        Time: O(n * f)
        Space: O(n * f)
    """

    def __init__(self, feature_service: LearningToRankFeatureService):
        self.feature_service = feature_service

    def train(self, limit: int = 500) -> dict[str, Any]:
        """Train a sklearn LogisticRegression sandbox model.

        Complexity:
            Time: O(n * f * i)
            Space: O(n * f)
        """
        sklearn = self._sklearn()
        if sklearn.get("available") is False:
            return self._not_trained("SKLEARN_UNAVAILABLE", sklearn["message"])

        limit = max(10, min(int(limit or 500), 5000))
        payload = self.feature_service.features(limit)
        rows = payload.get("rows", [])
        matrix = [self._vector(row) for row in rows]
        labels = [1 if float(row.get("label", 0.0)) >= 0.5 else 0 for row in rows]
        class_counts = {0: labels.count(0), 1: labels.count(1)}
        if len(rows) < 4:
            return self._not_trained(
                "INSUFFICIENT_ROWS",
                "At least 4 feature rows are required for the weak-supervision demo.",
                rows,
                class_counts,
            )
        if not class_counts[0] or not class_counts[1]:
            return self._not_trained(
                "SINGLE_CLASS",
                "Both positive and negative weak labels are required.",
                rows,
                class_counts,
            )

        np = sklearn["numpy"]
        LogisticRegression = sklearn["LogisticRegression"]
        StandardScaler = sklearn["StandardScaler"]
        metrics = sklearn["metrics"]

        x = np.asarray(matrix, dtype=float)
        y = np.asarray(labels, dtype=int)
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        model = LogisticRegression(class_weight="balanced", max_iter=1000)
        model.fit(x_scaled, y)
        probabilities = model.predict_proba(x_scaled)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        metric_payload = {
            "training_accuracy": round(float(metrics.accuracy_score(y, predictions)), 6),
            "average_precision": round(
                float(metrics.average_precision_score(y, probabilities)), 6
            ),
        }
        if len(set(labels)) == 2:
            metric_payload["roc_auc"] = round(
                float(metrics.roc_auc_score(y, probabilities)), 6
            )

        coefficients = [
            {
                "feature": name,
                "coefficient": round(float(value), 6),
                "direction": "positive" if value >= 0 else "negative",
            }
            for name, value in zip(FEATURE_NAMES, model.coef_[0])
        ]
        coefficients.sort(key=lambda item: abs(item["coefficient"]), reverse=True)
        return {
            "trained": True,
            "model": "sklearn_logistic_regression",
            "training_type": "weak_supervision_demo",
            "disclaimer": "Demo sandbox only. Labels are inferred from clicks and explicit feedback, not full editorial qrels.",
            "feature_set": payload.get("feature_set"),
            "row_count": len(rows),
            "class_balance": class_counts,
            "metrics": metric_payload,
            "feature_names": FEATURE_NAMES,
            "coefficients": coefficients,
            "top_positive_signals": [
                item for item in coefficients if item["coefficient"] > 0
            ][:5],
            "top_negative_signals": [
                item for item in coefficients if item["coefficient"] < 0
            ][:5],
            "sample_predictions": self._sample_predictions(rows, probabilities),
        }

    def export_report(self, output_path: Path, limit: int = 500) -> dict[str, Any]:
        """Write a training report as JSON.

        Complexity:
            Time: O(n * f * i)
            Space: O(n * f)
        """
        report = self.train(limit)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "output_path": str(output_path),
            "trained": report.get("trained", False),
            "row_count": report.get("row_count", 0),
        }

    def _vector(self, row: dict[str, Any]) -> list[float]:
        """Convert one feature row to a numeric vector.

        Complexity:
            Time: O(f)
            Space: O(f)
        """
        features = row.get("features") or {}
        return [float(features.get(name, 0.0) or 0.0) for name in FEATURE_NAMES]

    def _sample_predictions(
        self,
        rows: list[dict[str, Any]],
        probabilities: Any,
    ) -> list[dict[str, Any]]:
        """Return compact prediction samples.

        Complexity:
            Time: O(k)
            Space: O(k)
        """
        samples = []
        for row, probability in list(zip(rows, probabilities))[:10]:
            samples.append(
                {
                    "query": row.get("query"),
                    "doc_id": row.get("doc_id"),
                    "article_id": row.get("article_id"),
                    "label": row.get("label"),
                    "predicted_relevance": round(float(probability), 6),
                }
            )
        return samples

    def _not_trained(
        self,
        code: str,
        reason: str,
        rows: list[dict[str, Any]] | None = None,
        class_counts: dict[int, int] | None = None,
    ) -> dict[str, Any]:
        """Return a stable non-training payload.

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        return {
            "trained": False,
            "code": code,
            "reason": reason,
            "training_type": "weak_supervision_demo",
            "disclaimer": "Demo sandbox only. More diverse feedback is required before training is meaningful.",
            "row_count": len(rows or []),
            "class_balance": class_counts or {0: 0, 1: 0},
            "metrics": {},
            "coefficients": [],
            "sample_predictions": [],
        }

    def _sklearn(self) -> dict[str, Any]:
        """Import sklearn lazily.

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        try:
            import numpy
            from sklearn import metrics
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
        except ImportError as exc:
            return {"available": False, "message": str(exc)}
        return {
            "available": True,
            "numpy": numpy,
            "metrics": metrics,
            "LogisticRegression": LogisticRegression,
            "StandardScaler": StandardScaler,
        }
