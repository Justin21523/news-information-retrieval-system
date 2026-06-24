"""Data contract and validation helpers for news article records."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


MIN_CONTENT_CHARS = 20
MIN_TITLE_CHARS = 2
REQUIRED_FIELDS = {"title", "content"}


@dataclass
class ValidationIssue:
    """One validation issue for a source record.

    Complexity:
        Time: O(1)
        Space: O(1)
    """

    row: int
    code: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        """Convert issue to JSON-serializable dict.

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        return {"row": self.row, "code": self.code, "message": self.message}


@dataclass
class DatasetValidationStats:
    """Aggregate validation and deduplication stats.

    Complexity:
        Time: O(1)
        Space: O(k) where k is issue count
    """

    total: int = 0
    valid: int = 0
    invalid: int = 0
    duplicates: int = 0
    missing_fields: dict[str, int] = field(default_factory=dict)
    issues: list[ValidationIssue] = field(default_factory=list)

    def add_missing(self, field_name: str) -> None:
        """Track a missing required field.

        Complexity:
            Time: O(1)
            Space: O(1)
        """
        self.missing_fields[field_name] = self.missing_fields.get(field_name, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        """Convert stats to JSON-serializable dict.

        Complexity:
            Time: O(k)
            Space: O(k)
        """
        return {
            "total": self.total,
            "valid": self.valid,
            "invalid": self.invalid,
            "duplicates": self.duplicates,
            "missing_fields": self.missing_fields,
            "issues": [issue.to_dict() for issue in self.issues[:20]],
        }


def compute_dedup_hash(title: str, url: str) -> str:
    """Compute a stable duplicate hash from title and URL.

    Complexity:
        Time: O(n)
        Space: O(n)
    """
    key = f"{(title or '').strip().lower()}||{(url or '').strip().lower()}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def validate_article(raw: dict[str, Any], row: int) -> list[ValidationIssue]:
    """Validate a raw article-like dictionary.

    Complexity:
        Time: O(f)
        Space: O(f)
    """
    issues: list[ValidationIssue] = []
    for field_name in REQUIRED_FIELDS:
        if not str(raw.get(field_name) or raw.get("text" if field_name == "content" else "")).strip():
            issues.append(
                ValidationIssue(row, "MISSING_FIELD", f"Missing required field: {field_name}")
            )

    title = str(raw.get("title") or "").strip()
    content = str(raw.get("content") or raw.get("text") or raw.get("body") or "").strip()
    if title and len(title) < MIN_TITLE_CHARS:
        issues.append(ValidationIssue(row, "TITLE_TOO_SHORT", "Title is too short"))
    if content and len(content) < MIN_CONTENT_CHARS:
        issues.append(ValidationIssue(row, "CONTENT_TOO_SHORT", "Content is too short"))

    return issues


def normalize_tags(tags: Any) -> list[str]:
    """Normalize tags into a list of strings.

    Complexity:
        Time: O(t)
        Space: O(t)
    """
    if tags is None:
        return []
    if isinstance(tags, str):
        return [tag.strip() for tag in tags.split(",") if tag.strip()]
    if isinstance(tags, list):
        return [str(tag).strip() for tag in tags if str(tag).strip()]
    return [str(tags).strip()] if str(tags).strip() else []
