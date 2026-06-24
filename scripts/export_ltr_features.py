"""Export learning-to-rank feature rows from feedback logs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ir_app import create_app
from src.ir_app.config import Settings


def main() -> int:
    """Export LTR features as JSONL.

    Complexity:
        Time: O(n * q)
        Space: O(n)
    """
    parser = argparse.ArgumentParser(description="Export feedback LTR feature rows.")
    parser.add_argument(
        "--output",
        default="data/feedback/ltr_features.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument("--limit", type=int, default=1000, help="Maximum feature rows.")
    args = parser.parse_args()

    settings = Settings.from_env()
    app = create_app(settings)
    service = app.config["LTR_FEATURE_SERVICE"]
    result = service.export_jsonl(Path(args.output), args.limit)
    print(
        f"exported {result['row_count']} rows to {result['output_path']} "
        f"({result['feature_set']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
