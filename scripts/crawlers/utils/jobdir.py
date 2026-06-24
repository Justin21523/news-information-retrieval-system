from __future__ import annotations

from pathlib import Path
from typing import Optional


def has_pending_requests(jobdir: Optional[str]) -> bool:
    """
    Detect whether a Scrapy JOBDIR contains a non-empty disk request queue.

    This is used to safely resume long-running crawls without re-seeding
    start_requests() and accidentally duplicating large sequential queues.

    Args:
        jobdir: Scrapy JOBDIR path (string) or None.

    Returns:
        bool: True if the JOBDIR appears to contain pending requests.

    Complexity:
        Time: O(1) average (small constant number of filesystem checks)
        Space: O(1)
    """
    if not jobdir:
        return False

    jobdir_path = Path(jobdir)
    queue_dir = jobdir_path / "requests.queue"
    if not queue_dir.exists():
        return False

    primary_queue = queue_dir / "0"
    if primary_queue.exists() and primary_queue.stat().st_size > 0:
        return True

    # Fallback for other disk-queue layouts (start requests / priority buckets).
    for bucket in ("-2s", "-1s", "0s", "1s"):
        qfile = queue_dir / bucket / "q00000"
        if qfile.exists() and qfile.stat().st_size > 0:
            return True

    return False

