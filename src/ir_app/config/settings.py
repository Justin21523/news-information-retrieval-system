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
        dataset_path = Path(
            os.getenv("IR_DATASET_PATH", root / "data" / "processed" / "cna_mvp_cleaned.jsonl")
        )
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

        return cls(
            project_root=root,
            dataset_path=dataset_path,
            fallback_dataset_path=fallback_dataset_path,
            index_dir=index_dir,
            tokenizer_engine=tokenizer,
            enable_heavy_models=enable_heavy,
            host=os.getenv("IR_HOST", "0.0.0.0"),
            port=int(os.getenv("IR_PORT", "5001")),
            debug=os.getenv("FLASK_DEBUG", "false").lower() in {"1", "true", "yes", "on"},
        )
