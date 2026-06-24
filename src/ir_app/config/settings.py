"""Runtime settings for the Flask IR application."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    """Centralized runtime settings.

    Complexity:
        Time: O(1)
        Space: O(1)
    """

    project_root: Path
    dataset_path: Path
    fallback_dataset_path: Path
    index_dir: Path
    tokenizer_engine: str = "jieba"
    enable_heavy_models: bool = False
    max_documents: int | None = 25000
    host: str = "0.0.0.0"
    port: int = 5001
    debug: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        """Build settings from environment variables.

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        root = Path(__file__).resolve().parents[3]
        dataset_path = cls._default_dataset_path(root)
        if os.getenv("IR_DATASET_PATH"):
            dataset_path = Path(os.getenv("IR_DATASET_PATH", ""))
        fallback_dataset_path = Path(
            os.getenv("IR_FALLBACK_DATASET_PATH", root / "datasets" / "mini" / "ir_documents.json")
        )
        index_dir = Path(os.getenv("IR_INDEX_DIR", root / "data" / "indexes"))

        enable_heavy = os.getenv("IR_ENABLE_HEAVY_MODELS", "false").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

        tokenizer = os.getenv("IR_TOKENIZER_ENGINE", "jieba").strip().lower() or "jieba"
        if not enable_heavy and tokenizer in {"ckip", "auto"}:
            tokenizer = "jieba"

        max_documents = cls._parse_optional_int(os.getenv("IR_MAX_DOCUMENTS"), default=25000)

        return cls(
            project_root=root,
            dataset_path=dataset_path,
            fallback_dataset_path=fallback_dataset_path,
            index_dir=index_dir,
            tokenizer_engine=tokenizer,
            enable_heavy_models=enable_heavy,
            max_documents=max_documents,
            host=os.getenv("IR_HOST", "0.0.0.0"),
            port=int(os.getenv("IR_PORT", "5001")),
            debug=os.getenv("FLASK_DEBUG", "false").lower() in {"1", "true", "yes", "on"},
        )

    @staticmethod
    def _default_dataset_path(root: Path) -> Path:
        """Prefer the largest prepared corpus that exists locally.

        Complexity:
            Time: O(k)
            Space: O(1)
        """
        candidates = [
            root / "data" / "processed" / "unified_news_corpus_full.jsonl",
            root / "data" / "processed" / "unified_news_corpus.jsonl",
            root / "data" / "preprocessed" / "merged_14days_preprocessed.jsonl",
            root / "data" / "processed" / "cna_mvp_cleaned.jsonl",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[-1]

    @staticmethod
    def _parse_optional_int(value: str | None, default: int | None) -> int | None:
        """Parse an optional positive integer from environment text.

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        if value is None or value == "":
            return default
        parsed = int(value)
        return parsed if parsed > 0 else None
