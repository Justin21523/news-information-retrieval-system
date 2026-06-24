"""Train the weak-supervision LTR demo sandbox."""

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
    """Train and export an LTR demo report.

    Complexity:
        Time: O(n * f * i)
        Space: O(n * f)
    """
    parser = argparse.ArgumentParser(description="Train weak-supervision LTR demo.")
    parser.add_argument("--limit", type=int, default=500, help="Maximum feature rows.")
    parser.add_argument(
        "--output",
        default="data/feedback/ltr_training_report.json",
        help="Output JSON report path.",
    )
    args = parser.parse_args()

    app = create_app(Settings.from_env())
    service = app.config["LTR_TRAINING_SERVICE"]
    result = service.export_report(Path(args.output), args.limit)
    status = "trained" if result["trained"] else "not trained"
    print(f"{status}: {result['row_count']} rows -> {result['output_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
